from __future__ import annotations

from open_wam.models.policy_variants import PolicyObservedHistory

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from open_wam.configs import (
    ActionSpace,
    CurrentBlockCoupling,
    DynamicsObjective,
    ExperimentConfig,
    ProprioContextMode,
)
from open_wam.configs.policy_video_action import (
    VideoActionPolicyConfig,
    supports_dynamics_routing,
)
from open_wam.models.common.dynamics_objectives import (
    dynamics_objective_rollout_chunk_size,
)
from open_wam.models.policy_variants import (
    DynamicsRolloutRequest,
    PolicyInferContext,
)
from open_wam.models.common.video_action_state import VideoActionRolloutState
from open_wam.runtime.checkpoints import (
    CheckpointCompatibilityPolicy,
    load_pipeline_checkpoint,
)

from .types import FdmAblationMode, dynamics_objective_for_ablation_mode

if TYPE_CHECKING:
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

    from open_wam.pipelines import VariantRolloutRunner

    return DynamicsRollout(VariantRolloutRunner(pipeline))


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


class DynamicsRollout:
    """Offline dynamics adapter over the shared policy session and action contracts."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        policy_variant = runner.pipeline.policy_variant
        _validate_dynamics_rollout_policy(policy_variant.config)

    @property
    def pipeline(self) -> VariantPipeline:
        return self.runner.pipeline

    @property
    def action_per_frame(self) -> int:
        return self.pipeline.policy_variant.rollout_contract.action_tokens_per_frame

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
        if video_context.ndim != 5:
            raise ValueError(
                "Dynamics warmup video_context must have shape [B, C, T, H, W], "
                f"got {tuple(video_context.shape)}."
            )
        if action_context.ndim != 3:
            raise ValueError(
                "Dynamics warmup action_context must have shape [B, T, D], "
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
                "Dynamics warmup action history must align with video context frames, "
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
                    "Dynamics warmup hidden_proprio_history must have shape "
                    "[B, T, state_dim], "
                    f"got {tuple(hidden_proprio_history.shape)}."
                )
            if int(hidden_proprio_history.shape[0]) != int(video_context.shape[0]):
                raise ValueError(
                    "Dynamics warmup hidden proprio history batch size must match "
                    "video context, "
                    f"got hidden={tuple(hidden_proprio_history.shape)}, video={tuple(video_context.shape)}."
                )
            if int(hidden_proprio_history.shape[1]) != context_frames:
                raise ValueError(
                    "Dynamics warmup hidden proprio history must align with "
                    "video context frames, "
                    f"got hidden_frames={hidden_proprio_history.shape[1]}, context_frames={context_frames}."
                )
        elif uses_hidden_proprio and context_frames > 0:
            raise ValueError(
                "Dynamics offline rollout with "
                "proprio_context_mode=per_chunk_additive requires "
                "hidden_proprio_history aligned to the warmup video context."
            )
        if ActionSpace(action_space) is ActionSpace.RAW:
            action_context = self.pipeline.action_adapter.to_model(action_context)
        visual = self.pipeline.prepare_visual_outputs_from_latents(
            video_context,
            text_context=text_context,
            negative_text_context=negative_text_context,
        )
        update = self.runner.reconcile_observed_history(
            session=session,
            history=PolicyObservedHistory(
                video_latents=visual.frontend.video_latents,
                observation_frame_count=context_frames * self.action_per_frame,
                action_history=action_context,
                proprio_history=hidden_proprio_history,
                start_frame=context_start_frame,
            ),
        )
        if not update.applied:
            raise RuntimeError("Policy refused the initial observed history.")
        return update.session

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
            model_action_chunk=(
                None
                if raw_action_chunk is None
                else self.pipeline.action_adapter.to_model(raw_action_chunk)
            ),
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
        policy_output = step.infer_output.policy_output
        policy_aux = policy_output.aux
        video = policy_output.generated_video
        predicted_latents = video_condition_latents if video is None else video.latents
        if not isinstance(predicted_latents, torch.Tensor):
            raise TypeError(
                "Dynamics FDM rollout did not return predicted video latents."
            )
        action_pred = step.infer_output.decoder_output.action_pred
        debug = dict(policy_aux)
        debug["drop_text_conditioning"] = bool(drop_text_conditioning)
        return DynamicsChunkOutput(
            session=step.session,
            predicted_latents=predicted_latents,
            model_action_latents=action_pred,
            raw_action_sequence=self.pipeline.action_adapter.to_source(action_pred),
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
            if isinstance(state.variant_state, VideoActionRolloutState)
            else None
        )
        past = None if runtime_state is None else runtime_state.past_clean_latents
        if isinstance(past, torch.Tensor) and past.shape[2] > 0:
            return past[:, :, -min(int(frame_count), int(past.shape[2])) :].contiguous()
        if fallback is not None:
            return fallback
        raise ValueError(
            "Dynamics rollout requires warm video history before infer_chunk."
        )
