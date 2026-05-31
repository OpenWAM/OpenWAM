from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from open_wam.configs import ActionSpace, ParallelRuntimeMode
from open_wam.models.action_decoders import ActionDecoderInferOutput
from open_wam.models.policy_variants import PolicyInferOutput, PolicyInferState
from open_wam.models.policy_variants.parallel_stream import ParallelStreamPolicyVariant
from open_wam.models.visual_tower import VisualStageOutputs

from .variant_pipeline import VariantPipeline


@dataclass
class LingbotExactSession:
    """Stateful exact parallel-stream session that mirrors the LingBot server lifecycle."""

    policy_state: PolicyInferState
    task_text: tuple[str | None, ...] | None = None
    text_context: torch.Tensor | None = None
    negative_text_context: torch.Tensor | None = None


@dataclass
class LingbotExactWarmupOutput:
    """Outputs produced by one exact cache-warmup step."""

    session: LingbotExactSession
    visual_outputs: VisualStageOutputs
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class LingbotExactChunkOutput:
    """Chunk-native exact LingBot parallel-stream inference outputs."""

    session: LingbotExactSession
    policy_output: PolicyInferOutput
    decoder_output: ActionDecoderInferOutput
    visual_outputs: VisualStageOutputs | None
    chunk_action_pred: torch.Tensor
    raw_chunk_action_pred: torch.Tensor | None
    predicted_latents: torch.Tensor
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class LingbotExactArtifactBundle:
    """Portable exact LingBot parallel-stream inputs for offline replay or cluster execution."""

    video_latents: torch.Tensor | None = None
    views: dict[str, torch.Tensor] | None = None
    action_history: torch.Tensor | None = None
    task_text: tuple[str | None, ...] | None = None
    text_context: torch.Tensor | None = None
    negative_text_context: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def save_lingbot_exact_artifact_bundle(path: str | Path, bundle: LingbotExactArtifactBundle) -> None:
    torch.save(
        {
            "video_latents": bundle.video_latents,
            "views": bundle.views,
            "action_history": bundle.action_history,
            "task_text": bundle.task_text,
            "text_context": bundle.text_context,
            "negative_text_context": bundle.negative_text_context,
            "metadata": bundle.metadata,
        },
        Path(path),
    )


def load_lingbot_exact_artifact_bundle(path: str | Path) -> LingbotExactArtifactBundle:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions do not expose `weights_only`; those loads should
        # still be treated as trusted local artifacts.
        payload = torch.load(path, map_location="cpu")
    return LingbotExactArtifactBundle(
        video_latents=payload.get("video_latents"),
        views=payload.get("views"),
        action_history=payload.get("action_history"),
        task_text=payload.get("task_text"),
        text_context=payload.get("text_context"),
        negative_text_context=payload.get("negative_text_context"),
        metadata=dict(payload.get("metadata", {})),
    )


class LingbotExactRunner:
    """Reset / cache-warmup / generate lifecycle for the exact method-1 runtime.

    The runner preserves LingBot's serving contract, but the transformer it
    drives is the shared local backbone owned by `VisualTower`.
    """

    def __init__(self, pipeline: VariantPipeline) -> None:
        self.pipeline = pipeline
        if not isinstance(self.pipeline.policy_variant, ParallelStreamPolicyVariant):
            raise TypeError("LingBot exact runner requires a parallel-stream policy variant.")
        if self.pipeline.policy_variant.config.runtime_mode not in {
            ParallelRuntimeMode.LINGBOT_EXACT,
            ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
            ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        }:
            raise ValueError(
                "LingBot exact runner requires `parallel_stream.runtime_mode` to be "
                "`lingbot_exact`, `lingbot_exact_action_conditioned`, or "
                "`current_frame_action_chunk`, or `fastwam_first_frame`."
            )

    @property
    def policy_variant(self) -> ParallelStreamPolicyVariant:
        return self.pipeline.policy_variant

    def reset(
        self,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        cache_name: str = "open_wam_exact",
    ) -> LingbotExactSession:
        self.pipeline.visual_tower.reset_runtime_state()
        policy_state = self.policy_variant.reset_reference_runtime(
            visual_tower=self.pipeline.visual_tower,
            cache_name=cache_name,
        )
        return LingbotExactSession(
            policy_state=policy_state,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
        )

    def warmup_cache(
        self,
        *,
        session: LingbotExactSession,
        action_history: torch.Tensor,
        views: Mapping[str, torch.Tensor] | None = None,
        video_latents: torch.Tensor | None = None,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        action_space: ActionSpace | str = ActionSpace.AUTO,
        frame_start_override: int | None = None,
        action_conditioning_mode: object = "vanilla_joint_rollout",
        proprio_state: torch.Tensor | None = None,
    ) -> LingbotExactWarmupOutput:
        # Warmup uses the same shared frontend/runtime owner as the normal
        # pipeline path, while preserving the exact slot-pool cache lifecycle
        # needed by staged method-1 rollout.
        visual_outputs = self._prepare_visual_outputs(
            session=session,
            views=views,
            video_latents=video_latents,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=True,
        )
        next_policy_state = self.policy_variant.warm_reference_cache(
            self.pipeline.visual_tower,
            visual_outputs,
            action_history=action_history,
            infer_state=session.policy_state,
            action_space=action_space,
            frame_start_override=frame_start_override,
            action_conditioning_mode=str(getattr(action_conditioning_mode, "value", action_conditioning_mode)),
            proprio_state=proprio_state,
        )
        return LingbotExactWarmupOutput(
            session=LingbotExactSession(
                policy_state=next_policy_state,
                task_text=self._resolve_task_text(session, task_text),
                text_context=visual_outputs.frontend.conditioning.text_context,
                negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
            ),
            visual_outputs=visual_outputs,
            debug=dict(next_policy_state.cache.get("debug_last_warmup", {})),
        )

    def infer_chunk(
        self,
        *,
        session: LingbotExactSession,
        views: Mapping[str, torch.Tensor] | None = None,
        video_latents: torch.Tensor | None = None,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        proprio_state: torch.Tensor | None = None,
        advance_frame_start: bool = False,
        skip_video_prediction: bool = False,
        action_conditioning_mode: object = "vanilla_joint_rollout",
    ) -> LingbotExactChunkOutput:
        visual_outputs = None
        if views is not None or video_latents is not None:
            visual_outputs = self._prepare_visual_outputs(
                session=session,
                views=views,
                video_latents=video_latents,
                task_text=task_text,
                text_context=text_context,
                negative_text_context=negative_text_context,
                preserve_stream_cache=True,
            )
        resolved_text_context = (
            visual_outputs.frontend.conditioning.text_context
            if visual_outputs is not None
            else (text_context if text_context is not None else session.text_context)
        )
        resolved_negative_text_context = (
            visual_outputs.frontend.conditioning.negative_text_context
            if visual_outputs is not None
            else (negative_text_context if negative_text_context is not None else session.negative_text_context)
        )
        policy_output = self.policy_variant.generate_reference_chunk(
            visual_tower=self.pipeline.visual_tower,
            visual_outputs=visual_outputs,
            infer_state=session.policy_state,
            text_context=resolved_text_context,
            negative_text_context=resolved_negative_text_context,
            proprio_state=proprio_state,
            advance_frame_start=advance_frame_start,
            skip_video_prediction=skip_video_prediction,
            action_conditioning_mode=str(getattr(action_conditioning_mode, "value", action_conditioning_mode)),
        )
        decoder_output = self.pipeline.resolve_infer_decoder_output(
            policy_output,
            previous_decoder_state=session.policy_state.decoder_state,
        )
        policy_output.next_state.decoder_state = decoder_output.next_state
        next_session = LingbotExactSession(
            policy_state=policy_output.next_state,
            task_text=self._resolve_task_text(session, task_text),
            text_context=resolved_text_context,
            negative_text_context=resolved_negative_text_context,
        )
        return LingbotExactChunkOutput(
            session=next_session,
            policy_output=policy_output,
            decoder_output=decoder_output,
            visual_outputs=visual_outputs,
            chunk_action_pred=policy_output.aux["chunk_action_pred"],
            raw_chunk_action_pred=policy_output.aux["raw_chunk_action_pred"],
            predicted_latents=policy_output.aux["predicted_latents"],
            debug=dict(policy_output.aux.get("debug", {})),
        )

    def _prepare_visual_outputs(
        self,
        *,
        session: LingbotExactSession,
        views: Mapping[str, torch.Tensor] | None,
        video_latents: torch.Tensor | None,
        task_text: tuple[str | None, ...] | None,
        text_context: torch.Tensor | None,
        negative_text_context: torch.Tensor | None,
        preserve_stream_cache: bool,
    ) -> VisualStageOutputs:
        resolved_task_text = self._resolve_task_text(session, task_text)
        resolved_text_context = text_context if text_context is not None else session.text_context
        resolved_negative_text_context = (
            negative_text_context if negative_text_context is not None else session.negative_text_context
        )
        if video_latents is not None:
            return self.pipeline.prepare_visual_outputs_from_latents(
                video_latents,
                task_text=resolved_task_text,
                text_context=resolved_text_context,
                negative_text_context=resolved_negative_text_context,
            )
        if views is None:
            raise ValueError("Exact LingBot warmup/infer requires either `views` or `video_latents`.")
        return self.pipeline.prepare_visual_outputs(
            views,
            task_text=resolved_task_text,
            text_context=resolved_text_context,
            negative_text_context=resolved_negative_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )

    def _resolve_task_text(
        self,
        session: LingbotExactSession,
        task_text: tuple[str | None, ...] | None,
    ) -> tuple[str | None, ...] | None:
        return task_text if task_text is not None else session.task_text
