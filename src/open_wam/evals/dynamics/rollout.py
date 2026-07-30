from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from open_wam.configs import ActionSpace, CurrentBlockCoupling, ProprioContextMode
from open_wam.models.common import RolloutCursor
from open_wam.models.policy_variants import PolicyInferContext, PolicyInferState
from open_wam.models.policy_variants.mot.contracts import MoTRuntimeState
from open_wam.models.policy_variants.parallel_stream.runtime_semantics import (
    resolve_parallel_current_block_coupling,
)

from .types import FdmAblationMode


@dataclass
class FdmChunkOutput:
    session: Any
    predicted_latents: torch.Tensor
    model_action_latents: torch.Tensor
    raw_action_sequence: torch.Tensor | None
    debug: dict[str, Any] = field(default_factory=dict)


def should_drop_task_text_for_fdm_mode(mode: FdmAblationMode) -> bool:
    """Return whether a rollout mode should match text-free FDM training."""

    return FdmAblationMode(mode) in {
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        FdmAblationMode.VIDEO_CONDITIONED_ACTION,
    }


class JointDenoisingFdmRollout:
    """Offline chunk rollout wrapper for maintained M1.2 joint denoising."""

    def __init__(self, runner: Any) -> None:
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
        action_conditioning_mode: object = "vanilla_joint_rollout",
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
        hidden_proprio_history: torch.Tensor | None = None,
    ) -> LingbotExactSession:
        del hidden_proprio_history
        text_context, negative_text_context = self._resolve_warmup_text_context(
            video_context=video_context,
            text_context=text_context,
            negative_text_context=negative_text_context,
            drop_text_conditioning=drop_text_conditioning,
        )
        session = self.runner.reset(
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
        )
        if context_start_frame:
            session.policy_state.cache["frame_start"] = int(context_start_frame)
            session.policy_state.cursor = make_rollout_cursor(
                session.policy_state.cursor,
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
            action_conditioning_mode=action_conditioning_mode,
            proprio_state=proprio_state,
        )
        return warmup.session

    def infer_chunk(
        self,
        *,
        session: Any,
        mode: FdmAblationMode,
        raw_action_chunk: torch.Tensor | None,
        video_condition_latents: torch.Tensor | None = None,
        seed: int | None = None,
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
        allow_generated_action_commit: bool = False,
        extra_overrides: dict[str, Any] | None = None,
    ) -> FdmChunkOutput:
        del allow_generated_action_commit, extra_overrides
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        if mode == FdmAblationMode.VANILLA_JOINT_ROLLOUT:
            chunk = self.runner.infer_chunk(
                session=session,
                advance_frame_start=True,
                action_conditioning_mode=mode.value,
                proprio_state=proprio_state,
            )
            return FdmChunkOutput(
                session=chunk.session,
                predicted_latents=chunk.predicted_latents,
                model_action_latents=chunk.chunk_action_pred,
                raw_action_sequence=chunk.raw_chunk_action_pred,
                debug=dict(chunk.debug),
            )

        if mode == FdmAblationMode.VIDEO_CONDITIONED_ACTION:
            if video_condition_latents is None:
                raise ValueError("Mode 'video_conditioned_action' requires a ground-truth video latent chunk.")
            if raw_action_chunk is None:
                raise ValueError(
                    "Mode 'video_conditioned_action' requires a ground-truth raw action chunk "
                    "so IDM eval can return predicted actions while committing clean action history."
                )
            commit_action_latents = self._raw_actions_to_model_latents(raw_action_chunk)
            return self._infer_action_override_chunk(
                session=session,
                condition_latents=video_condition_latents,
                forced_action_latents=None,
                commit_action_latents=commit_action_latents,
                action_conditioning_mode=mode.value,
                drop_text_conditioning=drop_text_conditioning,
                proprio_state=proprio_state,
            )

        if raw_action_chunk is None:
            raise ValueError(f"Mode {mode.value!r} requires a ground-truth raw action chunk.")
        forced_action_latents = self._raw_actions_to_model_latents(raw_action_chunk)
        if mode == FdmAblationMode.FORCED_ACTION_JOINT_FDM:
            return self._infer_action_override_chunk(
                session=session,
                condition_latents=None,
                forced_action_latents=forced_action_latents,
                commit_action_latents=forced_action_latents,
                action_conditioning_mode=mode.value,
                drop_text_conditioning=drop_text_conditioning,
                proprio_state=proprio_state,
            )
        if mode == FdmAblationMode.CLEAN_ACTION_FEEDBACK:
            return self._infer_action_override_chunk(
                session=session,
                condition_latents=None,
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

    def _resolve_warmup_text_context(
        self,
        *,
        video_context: torch.Tensor,
        text_context: torch.Tensor | None,
        negative_text_context: torch.Tensor | None,
        drop_text_conditioning: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not drop_text_conditioning:
            return text_context, negative_text_context
        if text_context is not None:
            empty_text_context = torch.zeros_like(text_context)
        else:
            visual_config = self.runner.pipeline.visual_tower.config
            empty_text_context = torch.zeros(
                int(video_context.shape[0]),
                int(visual_config.max_text_tokens),
                int(visual_config.text_dim),
                device=video_context.device,
                dtype=video_context.dtype,
            )
        empty_negative_text_context = (
            torch.zeros_like(negative_text_context)
            if negative_text_context is not None
            else None
        )
        return empty_text_context, empty_negative_text_context

    def _infer_action_override_chunk(
        self,
        *,
        session: LingbotExactSession,
        condition_latents: torch.Tensor | None = None,
        forced_action_latents: torch.Tensor | None,
        commit_action_latents: torch.Tensor | None,
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
            condition_latents=(
                None
                if condition_latents is None
                else condition_latents.to(device=parameter.device, dtype=parameter.dtype)
            ),
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
        next_cursor = make_rollout_cursor(
            session.policy_state.cursor,
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
        next_state = SimpleNamespace(
            step_index=int(artifacts.next_cache["step_index"]),
            cursor=next_cursor,
            cache=artifacts.next_cache,
            decoder_state=session.policy_state.decoder_state,
        )
        raw_action_sequence = policy_variant.exact_action_adapter.to_raw_action_sequence(artifacts.action_pred)
        return FdmChunkOutput(
            session=SimpleNamespace(
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


class MotGeneralistDenoisingFdmRollout:
    """Offline chunk rollout wrapper for M5 generalist joint denoising."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        policy_variant = runner.pipeline.policy_variant
        if str(getattr(policy_variant.config, "name", "")) != "mot":
            raise TypeError("MotGeneralistDenoisingFdmRollout requires an M5/MoT policy variant.")
        if getattr(policy_variant.config, "mot_generalist_training_mode_probs", None) is None:
            raise ValueError("MotGeneralistDenoisingFdmRollout requires an M5 GJD config with mode probabilities.")
        coupling = CurrentBlockCoupling(getattr(policy_variant.config, "current_block_coupling", None))
        if coupling != CurrentBlockCoupling.JOINT:
            raise ValueError(
                "MotGeneralistDenoisingFdmRollout requires packed joint coupling so FDM/IDM "
                f"matches GJD training semantics, got current_block_coupling={coupling.value!r}."
            )

    @property
    def action_per_frame(self) -> int:
        return int(self.runner.pipeline.policy_variant.action_horizon) // self.frame_chunk_size

    @property
    def frame_chunk_size(self) -> int:
        return int(self.runner.pipeline.policy_variant.inference_config.frame_chunk_size)

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
        action_conditioning_mode: object = "vanilla_joint_rollout",
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
        hidden_proprio_history: torch.Tensor | None = None,
    ):
        del action_space
        if video_context.ndim != 5:
            raise ValueError(
                "M5 GJD warmup video_context must have shape [B, C, T, H, W], "
                f"got {tuple(video_context.shape)}."
            )
        if action_context.ndim != 3:
            raise ValueError(
                "M5 GJD warmup action_context must have shape [B, T, D], "
                f"got {tuple(action_context.shape)}."
            )
        text_context, negative_text_context = self._resolve_warmup_text_context(
            video_context=video_context,
            text_context=text_context,
            negative_text_context=negative_text_context,
            drop_text_conditioning=drop_text_conditioning,
        )
        self.runner.pipeline.visual_tower.reset_runtime_state()
        session = self.runner.reset(
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
        )
        context_frames = int(video_context.shape[2])
        action_tokens_per_frame = self.action_per_frame
        expected_action_tokens = context_frames * action_tokens_per_frame
        if int(action_context.shape[1]) != expected_action_tokens:
            raise ValueError(
                "M5 GJD warmup action history must align with video context frames, "
                f"got action_tokens={action_context.shape[1]}, context_frames={context_frames}, "
                f"action_per_frame={action_tokens_per_frame}."
            )
        policy_variant = self.runner.pipeline.policy_variant
        uses_hidden_proprio = (
            ProprioContextMode(getattr(policy_variant.config, "proprio_context_mode", ProprioContextMode.NONE))
            == ProprioContextMode.PER_CHUNK_ADDITIVE
        )
        if hidden_proprio_history is not None:
            if hidden_proprio_history.ndim != 3:
                raise ValueError(
                    "M5 GJD warmup hidden_proprio_history must have shape [B, T, state_dim], "
                    f"got {tuple(hidden_proprio_history.shape)}."
                )
            if int(hidden_proprio_history.shape[0]) != int(video_context.shape[0]):
                raise ValueError(
                    "M5 GJD warmup hidden proprio history batch size must match video context, "
                    f"got hidden={tuple(hidden_proprio_history.shape)}, video={tuple(video_context.shape)}."
                )
            if int(hidden_proprio_history.shape[1]) != context_frames:
                raise ValueError(
                    "M5 GJD warmup hidden proprio history must align with video context frames, "
                    f"got hidden_frames={hidden_proprio_history.shape[1]}, context_frames={context_frames}."
                )
        elif uses_hidden_proprio and context_frames > 0:
            raise ValueError(
                "M5 GJD offline rollout with proprio_context_mode=per_chunk_additive requires "
                "hidden_proprio_history aligned to the warmup video context."
            )
        current_start_frame = int(context_start_frame) + context_frames
        state = PolicyInferState(
            step_index=1,
            cursor=RolloutCursor(
                current_start_frame=current_start_frame,
                block_index=0,
                chunk_size=self.frame_chunk_size,
            ),
            variant_state=MoTRuntimeState(
                text_context=text_context,
                past_clean_latents=video_context.detach().clone(),
                past_clean_actions=action_context.detach().clone(),
                video_tokens_per_frame=None,
                next_condition_frame_start=current_start_frame,
                chunk_advance_frames=self.frame_chunk_size,
            ),
        )
        if proprio_state is not None:
            state.variant_state.proprio_state = proprio_state.detach().clone()
        if uses_hidden_proprio and proprio_state is not None:
            state.variant_state.hidden_proprio_state = proprio_state.detach().clone()
        if uses_hidden_proprio and hidden_proprio_history is not None:
            state.variant_state.past_hidden_proprio_states = hidden_proprio_history.detach().clone()
        session.policy_state = state
        return session

    def infer_chunk(
        self,
        *,
        session: Any,
        mode: FdmAblationMode,
        raw_action_chunk: torch.Tensor | None,
        video_condition_latents: torch.Tensor | None = None,
        seed: int | None = None,
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
        allow_generated_action_commit: bool = False,
        extra_overrides: dict[str, Any] | None = None,
    ) -> FdmChunkOutput:
        del drop_text_conditioning
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        extra: dict[str, Any] = {
            "action_conditioning_mode": mode.value,
            "mot_generalist_rollout_mode": mode.value,
        }
        if mode == FdmAblationMode.VIDEO_CONDITIONED_ACTION:
            if video_condition_latents is None:
                raise ValueError("Mode 'video_conditioned_action' requires a ground-truth video latent chunk.")
            if raw_action_chunk is None and not allow_generated_action_commit:
                raise ValueError("Mode 'video_conditioned_action' requires clean action history to commit.")
            extra["mot_video_condition_latents"] = video_condition_latents
            if raw_action_chunk is not None:
                extra["mot_commit_action_latents"] = raw_action_chunk
            video_latents = video_condition_latents
        elif mode == FdmAblationMode.FORCED_ACTION_JOINT_FDM:
            if raw_action_chunk is None:
                raise ValueError("Mode 'forced_action_joint_fdm' requires a ground-truth raw action chunk.")
            extra["mot_forced_action_latents"] = raw_action_chunk
            extra["mot_commit_action_latents"] = raw_action_chunk
            video_latents = self._history_video_template(session, video_condition_latents)
        elif mode == FdmAblationMode.CLEAN_ACTION_FEEDBACK:
            if raw_action_chunk is None:
                raise ValueError("Mode 'clean_action_feedback' requires a ground-truth raw action chunk.")
            extra["mot_commit_action_latents"] = raw_action_chunk
            video_latents = self._history_video_template(session, video_condition_latents)
        elif mode == FdmAblationMode.VANILLA_JOINT_ROLLOUT:
            video_latents = self._history_video_template(session, video_condition_latents)
        else:
            raise ValueError(f"Unsupported FDM ablation mode: {mode!r}")
        if extra_overrides:
            extra.update(extra_overrides)

        step = self.runner.infer_step(
            session=session,
            video_latents=video_latents,
            context=PolicyInferContext(
                state=proprio_state,
                extra=extra,
            ),
        )
        decoder_aux = step.infer_output.decoder_output.aux
        policy_aux = step.infer_output.policy_output.aux
        predicted_latents = decoder_aux.get("predicted_latents", policy_aux.get("predicted_latents"))
        if not isinstance(predicted_latents, torch.Tensor):
            raise RuntimeError("M5 GJD FDM rollout did not return predicted video latents.")
        action_pred = step.infer_output.decoder_output.action_pred
        return FdmChunkOutput(
            session=step.session,
            predicted_latents=predicted_latents,
            model_action_latents=action_pred,
            raw_action_sequence=action_pred,
            debug=dict(policy_aux),
        )

    def _history_video_template(self, session: Any, fallback: torch.Tensor | None) -> torch.Tensor:
        state = session.policy_state
        runtime_state = state.variant_state if isinstance(state.variant_state, MoTRuntimeState) else None
        past = None if runtime_state is None else runtime_state.past_clean_latents
        if isinstance(past, torch.Tensor) and past.shape[2] > 0:
            return past[:, :, -min(self.frame_chunk_size, int(past.shape[2])) :].contiguous()
        if fallback is not None:
            return fallback
        raise ValueError("M5 GJD rollout requires warm video history before infer_chunk.")

    def _resolve_warmup_text_context(
        self,
        *,
        video_context: torch.Tensor,
        text_context: torch.Tensor | None,
        negative_text_context: torch.Tensor | None,
        drop_text_conditioning: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not drop_text_conditioning:
            return text_context, negative_text_context
        if text_context is not None:
            empty_text_context = torch.zeros_like(text_context)
        else:
            visual_config = self.runner.pipeline.visual_tower.config
            empty_text_context = torch.zeros(
                int(video_context.shape[0]),
                int(visual_config.max_text_tokens),
                int(visual_config.text_dim),
                device=video_context.device,
                dtype=video_context.dtype,
            )
        empty_negative_text_context = (
            torch.zeros_like(negative_text_context)
            if negative_text_context is not None
            else None
        )
        return empty_text_context, empty_negative_text_context


def make_rollout_cursor(cursor, *, current_start_frame: int, block_index: int, chunk_size: int):
    cursor_type = type(cursor)
    try:
        return cursor_type(
            current_start_frame=int(current_start_frame),
            block_index=int(block_index),
            chunk_size=int(chunk_size),
        )
    except TypeError:
        return SimpleNamespace(
            current_start_frame=int(current_start_frame),
            block_index=int(block_index),
            chunk_size=int(chunk_size),
        )


def run_parallel_action_conditioned_action_override_inference_rollout(**kwargs):
    from open_wam.models.policy_variants.parallel_stream.packed_rollout import (
        run_parallel_packed_action_override_rollout,
    )

    return run_parallel_packed_action_override_rollout(**kwargs)
