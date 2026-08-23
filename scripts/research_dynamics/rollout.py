from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from open_wam.configs import (
    ActionSpace,
    CurrentBlockCoupling,
    DynamicsObjective,
    ExperimentConfig,
    PolicyVariantName,
    ProprioContextMode,
)
from open_wam.configs.policy_video_action import (
    VideoActionPolicyConfig,
    supports_dynamics_routing,
)
from open_wam.models.common import RolloutCursor
from open_wam.models.common.dynamics_objectives import (
    dynamics_objective_rollout_chunk_size,
)
from open_wam.models.policy_variants import (
    DynamicsRolloutRequest,
    PolicyInferContext,
    PolicyInferState,
)
from open_wam.models.policy_variants.dual_expert.contracts import DualExpertRuntimeState
from open_wam.runtime.checkpoints import (
    CheckpointCompatibilityPolicy,
    load_pipeline_checkpoint,
)

from .types import FdmAblationMode, dynamics_objective_for_ablation_mode

if TYPE_CHECKING:
    from open_wam.pipelines.lingbot_exact import LingbotExactSession
    from open_wam.pipelines.variant_pipeline import VariantPipeline


@dataclass
class DynamicsChunkOutput:
    session: Any
    predicted_latents: torch.Tensor
    model_action_latents: torch.Tensor
    raw_action_sequence: torch.Tensor | None
    debug: dict[str, Any] = field(default_factory=dict)


class DynamicsRolloutAdapter(Protocol):
    """Architecture-neutral interface used by offline dynamics evaluation."""

    @property
    def pipeline(self) -> VariantPipeline: ...

    @property
    def action_per_frame(self) -> int: ...

    @property
    def frame_chunk_size(self) -> int: ...

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
        mode: FdmAblationMode = FdmAblationMode.VANILLA_JOINT_ROLLOUT,
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
        hidden_proprio_history: torch.Tensor | None = None,
    ) -> Any: ...

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
    ) -> DynamicsChunkOutput: ...


def build_dynamics_rollout_adapter(
    *,
    config: ExperimentConfig,
    checkpoint_file: Path,
    runtime_device: torch.device,
    runtime_dtype: torch.dtype | None,
    checkpoint_compatibility: CheckpointCompatibilityPolicy = (
        CheckpointCompatibilityPolicy.ALLOW_CHECKPOINT_SUPERSET
    ),
) -> DynamicsRolloutAdapter:
    """Build the offline adapter for a supported policy architecture."""

    from open_wam.pipelines import build_variant_pipeline_from_config

    pipeline = build_variant_pipeline_from_config(config)
    _load_pipeline_checkpoint(
        pipeline,
        checkpoint_file,
        compatibility=checkpoint_compatibility,
    )
    if runtime_dtype is None:
        pipeline.to(runtime_device)
    else:
        pipeline.to(device=runtime_device, dtype=runtime_dtype)
    pipeline.eval()

    architecture = PolicyVariantName(config.policy_variant.name)
    if architecture == PolicyVariantName.DUAL_EXPERT:
        from open_wam.pipelines import VariantRolloutRunner

        return DualExpertDynamicsRollout(VariantRolloutRunner(pipeline))

    if architecture == PolicyVariantName.PARALLEL_STREAM:
        from open_wam.pipelines import LingbotExactRunner

        return ParallelStreamDynamicsRollout(LingbotExactRunner(pipeline))

    raise ValueError(
        "Dynamics evaluation has no runtime adapter for "
        f"policy architecture {architecture.value!r}."
    )


def resolve_action_per_frame(config: ExperimentConfig) -> int:
    """Resolve shared action/video frame geometry across policy architectures."""

    raw_action_per_frame = getattr(config.policy_variant, "action_per_frame", None)
    if raw_action_per_frame is not None:
        action_per_frame = int(raw_action_per_frame)
        if action_per_frame <= 0:
            raise ValueError(
                "policy_variant.action_per_frame must be positive, "
                f"got {raw_action_per_frame!r}."
            )
        return action_per_frame

    action_horizon = int(config.action_decoder.action_horizon)
    frame_chunk_size = int(config.inference.frame_chunk_size)
    if frame_chunk_size <= 0:
        raise ValueError(
            f"inference.frame_chunk_size must be positive, got {frame_chunk_size}."
        )
    if action_horizon <= 0 or action_horizon % frame_chunk_size:
        raise ValueError(
            "action_decoder.action_horizon must be positive and divisible by "
            "inference.frame_chunk_size; "
            f"got horizon={action_horizon}, frame_chunk_size={frame_chunk_size}."
        )
    return action_horizon // frame_chunk_size


def _validate_dynamics_rollout_policy(
    policy_config: VideoActionPolicyConfig,
) -> None:
    """Validate the model-neutral capabilities required by dynamics diagnostics."""

    program = policy_config.program
    if not supports_dynamics_routing(program):
        raise ValueError(
            "Dynamics rollout requires a GJD, forward-dynamics, or "
            f"inverse-dynamics program; got program={program.value!r}."
        )
    coupling = CurrentBlockCoupling(policy_config.current_block_coupling)
    if coupling != CurrentBlockCoupling.JOINT:
        raise ValueError(
            "Dynamics rollout requires packed joint coupling so FDM/IDM matches "
            f"training semantics, got current_block_coupling={coupling.value!r}."
        )


def _load_pipeline_checkpoint(
    pipeline: torch.nn.Module,
    checkpoint_file: Path,
    *,
    compatibility: CheckpointCompatibilityPolicy,
) -> None:
    report = load_pipeline_checkpoint(
        pipeline,
        checkpoint_file,
        map_location=torch.device("cpu"),
        compatibility=compatibility,
    )
    if report.missing_keys:
        print(f"dynamics_eval.checkpoint_missing_keys {len(report.missing_keys)}")
    if report.unexpected_keys:
        print(f"dynamics_eval.checkpoint_unexpected_keys {len(report.unexpected_keys)}")


def should_drop_task_text_for_fdm_mode(mode: FdmAblationMode) -> bool:
    """Return whether a rollout mode should match text-free FDM training."""

    return dynamics_objective_for_ablation_mode(mode) != DynamicsObjective.JOINT


def resolve_dynamics_rollout_frame_chunk_size(
    mode: FdmAblationMode,
    *,
    configured_frame_chunk_size: int,
) -> int:
    """Resolve diagnostic rollout geometry independently from train chunks."""

    return dynamics_objective_rollout_chunk_size(
        dynamics_objective_for_ablation_mode(mode),
        fallback_chunk_size=configured_frame_chunk_size,
    )


def build_diagnostic_dynamics_request(
    mode: FdmAblationMode | str,
    *,
    model_action_chunk: torch.Tensor | None,
    video_condition_latents: torch.Tensor | None,
    allow_generated_action_commit: bool = False,
) -> DynamicsRolloutRequest:
    """Compile one research intervention into the public dynamics contract.

    This is the sole mapping from diagnostic modes to model semantics. Policy
    adapters only convert raw actions to model space and execute the request.
    """

    resolved_mode = FdmAblationMode(mode)
    objective = dynamics_objective_for_ablation_mode(resolved_mode)
    if resolved_mode == FdmAblationMode.VIDEO_CONDITIONED_ACTION:
        if video_condition_latents is None:
            raise ValueError(
                "Mode 'video_conditioned_action' requires a ground-truth "
                "video latent chunk."
            )
        if model_action_chunk is None and not allow_generated_action_commit:
            raise ValueError(
                "Mode 'video_conditioned_action' requires clean action history "
                "to commit unless generated-action commit is explicitly enabled."
            )
        return DynamicsRolloutRequest(
            objective=objective,
            clean_video=video_condition_latents,
            history_action=model_action_chunk,
        )
    if resolved_mode == FdmAblationMode.FORCED_ACTION_JOINT_FDM:
        if model_action_chunk is None:
            raise ValueError(
                "Mode 'forced_action_joint_fdm' requires a ground-truth action chunk."
            )
        return DynamicsRolloutRequest(
            objective=objective,
            clean_action=model_action_chunk,
            history_action=model_action_chunk,
        )
    if resolved_mode == FdmAblationMode.CLEAN_ACTION_FEEDBACK:
        if model_action_chunk is None:
            raise ValueError(
                "Mode 'clean_action_feedback' requires a ground-truth action chunk."
            )
        return DynamicsRolloutRequest(
            objective=objective,
            history_action=model_action_chunk,
        )
    if resolved_mode == FdmAblationMode.VANILLA_JOINT_ROLLOUT:
        return DynamicsRolloutRequest(objective=objective)
    raise AssertionError(f"Unhandled research dynamics mode {resolved_mode!r}.")


def _resolve_warmup_text_context(
    *,
    runner: Any,
    video_context: torch.Tensor,
    text_context: torch.Tensor | None,
    negative_text_context: torch.Tensor | None,
    drop_text_conditioning: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not drop_text_conditioning:
        return text_context, negative_text_context
    if text_context is None:
        visual_config = runner.pipeline.visual_tower.config
        text_context = torch.zeros(
            int(video_context.shape[0]),
            int(visual_config.max_text_tokens),
            int(visual_config.text_dim),
            device=video_context.device,
            dtype=video_context.dtype,
        )
    else:
        text_context = torch.zeros_like(text_context)
    negative_text_context = (
        None
        if negative_text_context is None
        else torch.zeros_like(negative_text_context)
    )
    return text_context, negative_text_context


class ParallelStreamDynamicsRollout:
    """Offline dynamics adapter for the Parallel Stream policy runtime."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        _validate_dynamics_rollout_policy(runner.policy_variant.config)

    @property
    def pipeline(self) -> VariantPipeline:
        return self.runner.pipeline

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
        mode: FdmAblationMode = FdmAblationMode.VANILLA_JOINT_ROLLOUT,
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
        hidden_proprio_history: torch.Tensor | None = None,
    ) -> LingbotExactSession:
        text_context, negative_text_context = _resolve_warmup_text_context(
            runner=self.runner,
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
            dynamics=DynamicsRolloutRequest(
                objective=dynamics_objective_for_ablation_mode(mode),
            ),
            proprio_state=proprio_state,
            hidden_proprio_history=hidden_proprio_history,
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
    ) -> DynamicsChunkOutput:
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        model_action_chunk = (
            None
            if raw_action_chunk is None
            else self._raw_actions_to_model_sequence(raw_action_chunk)
        )
        request = build_diagnostic_dynamics_request(
            mode,
            model_action_chunk=model_action_chunk,
            video_condition_latents=video_condition_latents,
            allow_generated_action_commit=allow_generated_action_commit,
        )
        return self._infer_dynamics_chunk(
            session=session,
            request=request,
            drop_text_conditioning=drop_text_conditioning,
            proprio_state=proprio_state,
        )

    def _raw_actions_to_model_sequence(
        self,
        raw_action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        reference_transformer = self.runner.pipeline.visual_tower.get_runtime_backbone(
            action_dim=self.runner.policy_variant.action_dim
        )
        parameter = next(reference_transformer.parameters())
        return self.runner.policy_variant.exact_action_adapter.to_model_action_sequence(
            raw_action_chunk,
            action_space=ActionSpace.RAW,
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def _infer_dynamics_chunk(
        self,
        *,
        session: LingbotExactSession,
        request: DynamicsRolloutRequest,
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
    ) -> DynamicsChunkOutput:
        chunk = self.runner.infer_chunk(
            session=session,
            advance_frame_start=True,
            dynamics=request,
            proprio_state=proprio_state,
        )
        debug = dict(chunk.debug)
        debug["drop_text_conditioning"] = bool(drop_text_conditioning)
        return DynamicsChunkOutput(
            session=chunk.session,
            predicted_latents=chunk.predicted_latents,
            model_action_latents=chunk.chunk_action_pred,
            raw_action_sequence=chunk.raw_chunk_action_pred,
            debug=debug,
        )


class DualExpertDynamicsRollout:
    """Offline dynamics adapter for the Dual Expert policy runtime."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        policy_variant = runner.pipeline.policy_variant
        if (
            PolicyVariantName(policy_variant.config.name)
            != PolicyVariantName.DUAL_EXPERT
        ):
            raise TypeError(
                "Dual Expert dynamics rollout requires a dual-expert policy variant."
            )
        _validate_dynamics_rollout_policy(policy_variant.config)

    @property
    def pipeline(self) -> VariantPipeline:
        return self.runner.pipeline

    @property
    def action_per_frame(self) -> int:
        return (
            int(self.runner.pipeline.policy_variant.action_horizon)
            // self.frame_chunk_size
        )

    @property
    def frame_chunk_size(self) -> int:
        return int(
            self.runner.pipeline.policy_variant.inference_config.frame_chunk_size
        )

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
        mode: FdmAblationMode = FdmAblationMode.VANILLA_JOINT_ROLLOUT,
        drop_text_conditioning: bool = False,
        proprio_state: torch.Tensor | None = None,
        hidden_proprio_history: torch.Tensor | None = None,
    ):
        del action_space
        if video_context.ndim != 5:
            raise ValueError(
                "Dual-expert GJD warmup video_context must have shape [B, C, T, H, W], "
                f"got {tuple(video_context.shape)}."
            )
        if action_context.ndim != 3:
            raise ValueError(
                "Dual-expert GJD warmup action_context must have shape [B, T, D], "
                f"got {tuple(action_context.shape)}."
            )
        text_context, negative_text_context = _resolve_warmup_text_context(
            runner=self.runner,
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
                "Dual-expert GJD warmup action history must align with video context frames, "
                f"got action_tokens={action_context.shape[1]}, context_frames={context_frames}, "
                f"action_per_frame={action_tokens_per_frame}."
            )
        policy_variant = self.runner.pipeline.policy_variant
        uses_hidden_proprio = (
            ProprioContextMode(
                getattr(
                    policy_variant.config,
                    "proprio_context_mode",
                    ProprioContextMode.NONE,
                )
            )
            == ProprioContextMode.PER_CHUNK_ADDITIVE
        )
        if hidden_proprio_history is not None:
            if hidden_proprio_history.ndim != 3:
                raise ValueError(
                    "Dual-expert GJD warmup hidden_proprio_history must have shape "
                    "[B, T, state_dim], "
                    f"got {tuple(hidden_proprio_history.shape)}."
                )
            if int(hidden_proprio_history.shape[0]) != int(video_context.shape[0]):
                raise ValueError(
                    "Dual-expert GJD warmup hidden proprio history batch size must match "
                    "video context, "
                    f"got hidden={tuple(hidden_proprio_history.shape)}, video={tuple(video_context.shape)}."
                )
            if int(hidden_proprio_history.shape[1]) != context_frames:
                raise ValueError(
                    "Dual-expert GJD warmup hidden proprio history must align with "
                    "video context frames, "
                    f"got hidden_frames={hidden_proprio_history.shape[1]}, context_frames={context_frames}."
                )
        elif uses_hidden_proprio and context_frames > 0:
            raise ValueError(
                "Dual-expert GJD offline rollout with "
                "proprio_context_mode=per_chunk_additive requires "
                "hidden_proprio_history aligned to the warmup video context."
            )
        current_start_frame = int(context_start_frame) + context_frames
        rollout_frame_chunk_size = resolve_dynamics_rollout_frame_chunk_size(
            mode,
            configured_frame_chunk_size=self.frame_chunk_size,
        )
        state = PolicyInferState(
            step_index=1,
            cursor=RolloutCursor(
                current_start_frame=current_start_frame,
                block_index=0,
                chunk_size=rollout_frame_chunk_size,
            ),
            variant_state=DualExpertRuntimeState(
                text_context=text_context,
                past_clean_latents=video_context.detach().clone(),
                past_clean_actions=action_context.detach().clone(),
                video_tokens_per_frame=None,
                next_condition_frame_start=current_start_frame,
                chunk_advance_frames=rollout_frame_chunk_size,
            ),
        )
        if proprio_state is not None:
            state.variant_state.proprio_state = proprio_state.detach().clone()
        if uses_hidden_proprio and proprio_state is not None:
            state.variant_state.hidden_proprio_state = proprio_state.detach().clone()
        if uses_hidden_proprio and hidden_proprio_history is not None:
            state.variant_state.past_hidden_proprio_states = (
                hidden_proprio_history.detach().clone()
            )
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
    ) -> DynamicsChunkOutput:
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        rollout_frame_chunk_size = resolve_dynamics_rollout_frame_chunk_size(
            mode,
            configured_frame_chunk_size=self.frame_chunk_size,
        )
        request = build_diagnostic_dynamics_request(
            mode,
            model_action_chunk=raw_action_chunk,
            video_condition_latents=video_condition_latents,
            allow_generated_action_commit=allow_generated_action_commit,
        )
        video_latents = (
            request.clean_video
            if request.clean_video is not None
            else self._history_video_template(
                session,
                video_condition_latents,
                frame_count=rollout_frame_chunk_size,
            )
        )
        step = self.runner.infer_step(
            session=session,
            video_latents=video_latents,
            context=PolicyInferContext(
                state=proprio_state,
                dynamics=request,
            ),
        )
        decoder_aux = step.infer_output.decoder_output.aux
        policy_aux = step.infer_output.policy_output.aux
        predicted_latents = decoder_aux.get(
            "predicted_latents", policy_aux.get("predicted_latents")
        )
        if not isinstance(predicted_latents, torch.Tensor):
            raise TypeError(
                "Dual-expert GJD FDM rollout did not return predicted video latents."
            )
        action_pred = step.infer_output.decoder_output.action_pred
        debug = dict(policy_aux)
        debug["drop_text_conditioning"] = bool(drop_text_conditioning)
        return DynamicsChunkOutput(
            session=step.session,
            predicted_latents=predicted_latents,
            model_action_latents=action_pred,
            raw_action_sequence=action_pred,
            debug=debug,
        )

    def _history_video_template(
        self,
        session: Any,
        fallback: torch.Tensor | None,
        *,
        frame_count: int,
    ) -> torch.Tensor:
        state = session.policy_state
        runtime_state = (
            state.variant_state
            if isinstance(state.variant_state, DualExpertRuntimeState)
            else None
        )
        past = None if runtime_state is None else runtime_state.past_clean_latents
        if isinstance(past, torch.Tensor) and past.shape[2] > 0:
            return past[:, :, -min(int(frame_count), int(past.shape[2])) :].contiguous()
        if fallback is not None:
            return fallback
        raise ValueError(
            "Dual-expert GJD rollout requires warm video history before infer_chunk."
        )
