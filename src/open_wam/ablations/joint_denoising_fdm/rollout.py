from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.configs import ActionSpace, CurrentBlockCoupling
from open_wam.models.common import RolloutCursor
from open_wam.models.policy_variants.contracts import PolicyInferState
from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
    resolve_parallel_current_block_coupling,
    run_parallel_action_conditioned_action_override_inference_rollout,
)
from open_wam.pipelines import LingbotExactRunner, LingbotExactSession

from .types import FdmAblationMode


@dataclass
class FdmChunkOutput:
    session: LingbotExactSession
    predicted_latents: torch.Tensor
    model_action_latents: torch.Tensor
    raw_action_sequence: torch.Tensor | None
    debug: dict[str, Any] = field(default_factory=dict)


class JointDenoisingFdmRollout:
    """Offline chunk rollout wrapper for maintained M1.2 joint denoising."""

    def __init__(self, runner: LingbotExactRunner) -> None:
        self.runner = runner
        coupling = resolve_parallel_current_block_coupling(runner.policy_variant.config)
        if coupling != CurrentBlockCoupling.JOINT:
            raise ValueError(
                "JointDenoisingFdmRollout requires maintained M1.2 joint denoising, "
                f"got current_block_coupling={coupling.value!r}."
            )

    @property
    def action_per_frame(self) -> int:
        return int(self.runner.policy_variant.config.action_per_frame)

    @property
    def frame_chunk_size(self) -> int:
        return int(self.runner.policy_variant.inference_config.frame_chunk_size)

    def reset_and_warmup(
        self,
        *,
        task_text: tuple[str | None, ...],
        video_context: torch.Tensor,
        action_context: torch.Tensor,
        text_context: torch.Tensor | None,
        negative_text_context: torch.Tensor | None,
        context_start_frame: int = 0,
        action_space: ActionSpace | str = ActionSpace.RAW,
        proprio_state: torch.Tensor | None = None,
    ) -> LingbotExactSession:
        session = self.runner.reset(
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
        )
        if context_start_frame:
            session.policy_state.cache["frame_start"] = int(context_start_frame)
            session.policy_state.cursor = RolloutCursor(
                current_start_frame=int(context_start_frame),
                block_index=session.policy_state.cursor.block_index,
                chunk_size=session.policy_state.cursor.chunk_size,
            )
        warmup = self.runner.warmup_cache(
            session=session,
            video_latents=video_context,
            action_history=action_context,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            action_space=action_space,
            proprio_state=proprio_state,
        )
        return warmup.session

    def infer_chunk(
        self,
        *,
        session: LingbotExactSession,
        mode: FdmAblationMode,
        raw_action_chunk: torch.Tensor | None,
        seed: int | None = None,
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
    ) -> FdmChunkOutput:
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        if mode == FdmAblationMode.VANILLA_JOINT_ROLLOUT:
            chunk = self.runner.infer_chunk(
                session=session,
                advance_frame_start=True,
                proprio_state=proprio_state,
            )
            return FdmChunkOutput(
                session=chunk.session,
                predicted_latents=chunk.predicted_latents,
                model_action_latents=chunk.chunk_action_pred,
                raw_action_sequence=chunk.raw_chunk_action_pred,
                debug=dict(chunk.debug),
            )

        if raw_action_chunk is None:
            raise ValueError(f"Mode {mode.value!r} requires a ground-truth raw action chunk.")
        forced_action_latents = self._raw_actions_to_model_latents(raw_action_chunk)
        if mode == FdmAblationMode.FORCED_ACTION_JOINT_FDM:
            return self._infer_action_override_chunk(
                session=session,
                forced_action_latents=forced_action_latents,
                commit_action_latents=forced_action_latents,
                action_conditioning_mode=mode.value,
                drop_text_conditioning=drop_text_conditioning,
                proprio_state=proprio_state,
            )
        if mode == FdmAblationMode.CLEAN_ACTION_FEEDBACK:
            return self._infer_action_override_chunk(
                session=session,
                forced_action_latents=None,
                commit_action_latents=forced_action_latents,
                action_conditioning_mode=mode.value,
                drop_text_conditioning=drop_text_conditioning,
                proprio_state=proprio_state,
            )
        raise ValueError(f"Unsupported FDM ablation mode: {mode!r}")

    def _raw_actions_to_model_latents(self, raw_action_chunk: torch.Tensor) -> torch.Tensor:
        reference_transformer = self.runner.pipeline.visual_tower.get_runtime_backbone(
            action_dim=self.runner.policy_variant.action_dim
        )
        parameter = next(reference_transformer.parameters())
        return self.runner.policy_variant.exact_action_adapter.to_model_action_latents(
            raw_action_chunk,
            action_per_frame=self.action_per_frame,
            action_space=ActionSpace.RAW,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _infer_action_override_chunk(
        self,
        *,
        session: LingbotExactSession,
        forced_action_latents: torch.Tensor | None,
        commit_action_latents: torch.Tensor,
        action_conditioning_mode: str,
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
    ) -> FdmChunkOutput:
        visual_tower = self.runner.pipeline.visual_tower
        policy_variant = self.runner.policy_variant
        reference_transformer = visual_tower.get_runtime_backbone(action_dim=policy_variant.action_dim)
        parameter = next(reference_transformer.parameters())
        text_emb = None if drop_text_conditioning else session.text_context
        negative_text_emb = None if drop_text_conditioning else session.negative_text_context
        artifacts = run_parallel_action_conditioned_action_override_inference_rollout(
            transformer=reference_transformer,
            backbone_config=policy_variant.backbone_config,
            policy_config=policy_variant.config,
            training_config=policy_variant.training_config,
            inference_config=policy_variant.inference_config,
            action_dim=policy_variant.action_dim,
            condition_latents=None,
            text_emb=text_emb,
            negative_text_emb=negative_text_emb,
            action_channel_mask=policy_variant._reference_action_channel_mask(
                device=parameter.device,
                dtype=parameter.dtype,
            ),
            infer_cache=session.policy_state.cache,
            advance_frame_start=True,
            forced_action_latents=forced_action_latents,
            commit_action_latents=commit_action_latents,
            action_conditioning_mode=action_conditioning_mode,
            proprio_state=proprio_state,
        )
        artifacts.debug["drop_text_conditioning"] = bool(drop_text_conditioning)
        next_cursor = RolloutCursor(
            current_start_frame=int(
                artifacts.next_cache.get("frame_start", session.policy_state.cursor.current_start_frame)
            ),
            block_index=int(artifacts.next_cache.get("step_index", session.policy_state.step_index)),
            chunk_size=policy_variant.inference_config.frame_chunk_size,
        )
        artifacts.next_cache["backbone_cache"] = visual_tower.advance_runtime_cache_state(
            visual_tower.resolve_runtime_cache_state(
                session.policy_state.cache.get("backbone_cache"),
                cursor=session.policy_state.cursor,
                stage="parallel_stream_lingbot_exact",
                payload={"cache_name": str(session.policy_state.cache.get("cache_name", "open_wam_exact"))},
            ),
            next_cursor=next_cursor,
            payload_updates={
                "cache_name": str(
                    artifacts.next_cache.get("cache_name", session.policy_state.cache.get("cache_name", "open_wam_exact"))
                )
            },
        )
        next_state = PolicyInferState(
            step_index=int(artifacts.next_cache["step_index"]),
            cursor=next_cursor,
            cache=artifacts.next_cache,
            decoder_state=session.policy_state.decoder_state,
        )
        raw_action_sequence = policy_variant.exact_action_adapter.to_raw_action_sequence(artifacts.action_pred)
        return FdmChunkOutput(
            session=LingbotExactSession(
                policy_state=next_state,
                task_text=session.task_text,
                text_context=session.text_context,
                negative_text_context=session.negative_text_context,
            ),
            predicted_latents=artifacts.predicted_latents,
            model_action_latents=artifacts.action_pred,
            raw_action_sequence=raw_action_sequence,
            debug=dict(artifacts.debug),
        )
