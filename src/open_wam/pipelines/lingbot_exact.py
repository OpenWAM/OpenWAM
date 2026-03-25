from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from open_wam.models.action_decoders import ActionDecoderInferOutput
from open_wam.models.policy_variants import PolicyInferOutput, PolicyInferState
from open_wam.models.policy_variants.parallel_stream import ParallelStreamPolicyVariant
from open_wam.models.visual_tower import VisualStageOutputs

from .variant_pipeline import VariantPipeline


@dataclass
class LingbotExactSession:
    """Stateful exact method-1 session that mirrors LingBot server lifecycle."""

    policy_state: PolicyInferState
    task_text: tuple[str | None, ...] | None = None
    text_context: torch.Tensor | None = None


@dataclass
class LingbotExactWarmupOutput:
    """Outputs produced by one exact cache-warmup step."""

    session: LingbotExactSession
    visual_outputs: VisualStageOutputs


@dataclass
class LingbotExactChunkOutput:
    """Chunk-native exact method-1 inference outputs."""

    session: LingbotExactSession
    policy_output: PolicyInferOutput
    decoder_output: ActionDecoderInferOutput
    visual_outputs: VisualStageOutputs | None
    chunk_action_pred: torch.Tensor
    raw_chunk_action_pred: torch.Tensor | None
    predicted_latents: torch.Tensor


class LingbotExactMethod1Runner:
    """Reset / cache-warmup / generate lifecycle for exact LingBot method 1."""

    def __init__(self, pipeline: VariantPipeline) -> None:
        self.pipeline = pipeline
        if not isinstance(self.pipeline.policy_variant, ParallelStreamPolicyVariant):
            raise TypeError("LingBot exact runner requires a parallel-stream policy variant.")
        if self.pipeline.policy_variant.config.runtime_mode != "lingbot_exact":
            raise ValueError("LingBot exact runner requires `parallel_stream.runtime_mode = lingbot_exact`.")

    @property
    def policy_variant(self) -> ParallelStreamPolicyVariant:
        return self.pipeline.policy_variant

    def reset(
        self,
        *,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
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
        action_space: str = "auto",
    ) -> LingbotExactWarmupOutput:
        visual_outputs = self._prepare_visual_outputs(
            session=session,
            views=views,
            video_latents=video_latents,
            task_text=task_text,
            text_context=text_context,
            preserve_stream_cache=True,
        )
        next_policy_state = self.policy_variant.warm_reference_cache(
            self.pipeline.visual_tower,
            visual_outputs,
            action_history=action_history,
            infer_state=session.policy_state,
            action_space=action_space,
        )
        return LingbotExactWarmupOutput(
            session=LingbotExactSession(
                policy_state=next_policy_state,
                task_text=self._resolve_task_text(session, task_text),
                text_context=visual_outputs.frontend.conditioning.text_context,
            ),
            visual_outputs=visual_outputs,
        )

    def infer_chunk(
        self,
        *,
        session: LingbotExactSession,
        views: Mapping[str, torch.Tensor] | None = None,
        video_latents: torch.Tensor | None = None,
        task_text: tuple[str | None, ...] | None = None,
        text_context: torch.Tensor | None = None,
        advance_frame_start: bool = False,
    ) -> LingbotExactChunkOutput:
        visual_outputs = None
        if views is not None or video_latents is not None:
            visual_outputs = self._prepare_visual_outputs(
                session=session,
                views=views,
                video_latents=video_latents,
                task_text=task_text,
                text_context=text_context,
                preserve_stream_cache=True,
            )
        resolved_text_context = (
            visual_outputs.frontend.conditioning.text_context
            if visual_outputs is not None
            else (text_context if text_context is not None else session.text_context)
        )
        policy_output = self.policy_variant.generate_reference_chunk(
            visual_tower=self.pipeline.visual_tower,
            visual_outputs=visual_outputs,
            infer_state=session.policy_state,
            text_context=resolved_text_context,
            advance_frame_start=advance_frame_start,
        )
        decoder_output = self.pipeline.action_decoder.forward_infer(policy_output)
        next_session = LingbotExactSession(
            policy_state=policy_output.next_state,
            task_text=self._resolve_task_text(session, task_text),
            text_context=resolved_text_context,
        )
        return LingbotExactChunkOutput(
            session=next_session,
            policy_output=policy_output,
            decoder_output=decoder_output,
            visual_outputs=visual_outputs,
            chunk_action_pred=policy_output.aux["chunk_action_pred"],
            raw_chunk_action_pred=policy_output.aux["raw_chunk_action_pred"],
            predicted_latents=policy_output.aux["predicted_latents"],
        )

    def _prepare_visual_outputs(
        self,
        *,
        session: LingbotExactSession,
        views: Mapping[str, torch.Tensor] | None,
        video_latents: torch.Tensor | None,
        task_text: tuple[str | None, ...] | None,
        text_context: torch.Tensor | None,
        preserve_stream_cache: bool,
    ) -> VisualStageOutputs:
        resolved_task_text = self._resolve_task_text(session, task_text)
        resolved_text_context = text_context if text_context is not None else session.text_context
        if video_latents is not None:
            return self.pipeline.prepare_visual_outputs_from_latents(
                video_latents,
                task_text=resolved_task_text,
                text_context=resolved_text_context,
            )
        if views is None:
            raise ValueError("Exact method-1 warmup/infer requires either `views` or `video_latents`.")
        return self.pipeline.prepare_visual_outputs(
            views,
            task_text=resolved_task_text,
            text_context=resolved_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )

    def _resolve_task_text(
        self,
        session: LingbotExactSession,
        task_text: tuple[str | None, ...] | None,
    ) -> tuple[str | None, ...] | None:
        return task_text if task_text is not None else session.task_text
