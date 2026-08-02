"""Typed planner handoffs and LIBERO control-plan materialization.

This module owns planner request/result envelopes, model-action projection,
session handoff, result acceptance, and fallback plan construction. Model
execution, encoded-history preparation, cache mutation, and worker submission
remain in :mod:`open_wam.evals.libero_realtime_runtime`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from einops import rearrange

from open_wam.configs import ActionTargetRepresentation, ExperimentConfig, ParallelRuntimeMode
from open_wam.configs.enums import DeadlineMissPolicy
from open_wam.data.action_pose import PoseSequence
from open_wam.evals import realtime_history
from open_wam.integrations import LiberoControlConfig, compute_osc_pose_action
from open_wam.integrations import libero_rollout
from open_wam.integrations.realtime_contracts import (
    PlannedControlStep,
    PlannedFrameAction,
)
from open_wam.integrations.realtime_control import (
    make_planned_frame_actions,
    planned_frame_actions_to_control_steps,
)
from open_wam.integrations.realtime_plan_queue import (
    drop_partial_stale_control_chunk,
    future_control_steps,
    merge_future_control_steps,
)
from open_wam.models.common.rollout_startup import (
    require_strict_startup_generation_frame,
)
from open_wam.models.policy_variants import PolicyInferState, RolloutCursor
from open_wam.models.visual_tower import VisualRuntimeStateSnapshot
from open_wam.pipelines import VariantRolloutSession


__all__ = [
    "FramePlannerJobResult",
    "FramePlannerResultApplication",
    "SequenceReplanJobOptions",
    "SequenceReplanJobResult",
    "annotate_sequence_planner_acceptance",
    "apply_frame_planner_result",
    "apply_sequence_replan_result",
    "build_exact_startup_conditioning_history_record",
    "build_fallback_frame_actions",
    "exact_chunk_to_planned_steps",
    "materialize_sequence_control_action",
    "resolve_exact_startup_sessions",
    "resolve_next_exact_history_base_session",
    "sequence_chunk_to_planned_steps",
]


@dataclass(frozen=True)
class FramePlannerJobResult:
    """One frame-grouped planner result with explicit session ownership."""

    job_kind: Literal["history_replan", "open_loop_extension"]
    planned_frames: list[PlannedFrameAction]
    buffer_tail_session: VariantRolloutSession | None
    trace: dict[str, Any]
    submitted_through_frame: int | None
    session: VariantRolloutSession | None = None
    warmup_session: VariantRolloutSession | None = None


@dataclass(frozen=True)
class FramePlannerResultApplication:
    """Scheduler state after accepting or rejecting one frame planner job."""

    history_base_session: VariantRolloutSession
    current_chunk_session: VariantRolloutSession
    buffer_tail_session: VariantRolloutSession | None
    plan_by_action: dict[int, PlannedControlStep]
    pending_history: list[dict[str, Any]]


@dataclass(frozen=True)
class SequenceReplanJobOptions:
    """One observation-conditioned or open-loop sequence planner request."""

    prompt: str
    task_id: int
    episode_idx: int
    frontend_device: torch.device
    runtime_device: torch.device
    generation_action_start: int
    source: str
    reset_observation_conditioned_session: bool = True
    use_observation_update: bool = True
    runtime_cache_snapshot: VisualRuntimeStateSnapshot | None = None
    mot_condition_frame_start: int | None = None
    mot_action_cache_rewind_frame_start: int | None = None
    preserve_rng_state: bool = False


@dataclass(frozen=True)
class SequenceReplanJobResult:
    """Typed sequence plan, next runtime state, and stable trace metadata."""

    session: VariantRolloutSession
    runtime_cache_snapshot: VisualRuntimeStateSnapshot | None
    planned_steps: list[PlannedControlStep]
    next_generation_action_start: int
    trace: dict[str, Any]


def sequence_chunk_to_planned_steps(
    *,
    action_pred: np.ndarray,
    reference_obs: dict[str, np.ndarray],
    generation_action_start: int,
    execution_action_offset: int = 0,
    source: str,
    planner_step_index: int | None,
    ready_monotonic_s: float,
    action_target_representation: ActionTargetRepresentation | str,
    rotation_representation: str,
) -> list[PlannedControlStep]:
    """Project model-space sequence actions into typed LIBERO plan steps."""

    representation = ActionTargetRepresentation(action_target_representation)
    planned_steps: list[PlannedControlStep] = []
    execution_start = int(generation_action_start) - max(
        0,
        int(execution_action_offset),
    )
    if representation == ActionTargetRepresentation.RAW:
        for action_offset in range(action_pred.shape[0]):
            planned_steps.append(
                PlannedControlStep(
                    absolute_action_index=int(execution_start + action_offset),
                    generation_action_start=int(generation_action_start),
                    source=str(source),
                    planner_step_index=planner_step_index,
                    ready_monotonic_s=ready_monotonic_s,
                    raw_action=np.asarray(
                        action_pred[action_offset],
                        dtype=np.float32,
                    ).copy(),
                    desired_position=None,
                    desired_quaternion=None,
                    desired_gripper=None,
                )
            )
        return planned_steps

    if representation != ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
        raise ValueError(
            "Unsupported action target representation for sequence rollout: "
            f"{representation!r}."
        )

    desired_pose_targets = libero_rollout.reconstruct_libero_pose_targets(
        action_pred,
        reference_observation=reference_obs,
        rotation_representation=rotation_representation,
    )
    for action_offset in range(action_pred.shape[0]):
        desired_gripper = None
        if desired_pose_targets.gripper is not None:
            desired_gripper = (
                desired_pose_targets.gripper[action_offset]
                .detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
        planned_steps.append(
            PlannedControlStep(
                absolute_action_index=int(execution_start + action_offset),
                generation_action_start=int(generation_action_start),
                source=str(source),
                planner_step_index=planner_step_index,
                ready_monotonic_s=ready_monotonic_s,
                raw_action=None,
                desired_position=(
                    desired_pose_targets.position[action_offset]
                    .detach()
                    .to(dtype=torch.float32)
                    .cpu()
                    .numpy()
                ),
                desired_quaternion=(
                    desired_pose_targets.quaternion[action_offset]
                    .detach()
                    .to(dtype=torch.float32)
                    .cpu()
                    .numpy()
                ),
                desired_gripper=desired_gripper,
            )
        )
    return planned_steps


def materialize_sequence_control_action(
    planned_step: PlannedControlStep,
    *,
    current_obs: dict[str, np.ndarray],
    control_config: LiberoControlConfig,
    gripper_representation: str,
) -> np.ndarray:
    """Convert a typed raw or absolute-pose plan step to one LIBERO action."""

    if planned_step.raw_action is not None:
        return np.clip(
            np.asarray(planned_step.raw_action, dtype=np.float32),
            -1.0,
            1.0,
        )
    if (
        planned_step.desired_position is None
        or planned_step.desired_quaternion is None
    ):
        raise RuntimeError("Sequence rollout step is missing absolute pose targets.")
    desired_pose = PoseSequence(
        position=torch.from_numpy(
            np.asarray(planned_step.desired_position, dtype=np.float32)
        ),
        quaternion=torch.from_numpy(
            np.asarray(planned_step.desired_quaternion, dtype=np.float32)
        ),
        gripper=(
            None
            if planned_step.desired_gripper is None
            else torch.from_numpy(
                np.asarray(planned_step.desired_gripper, dtype=np.float32)
            )
        ),
    )
    return compute_osc_pose_action(
        current_pose=libero_rollout.pose_from_libero_observation(current_obs),
        desired_pose=desired_pose,
        control_config=control_config,
        gripper_representation=gripper_representation,
    ).astype(np.float32)


def exact_chunk_to_planned_steps(
    *,
    chunk,
    action_per_frame: int,
    frame_chunk_size: int,
    source: str,
    ready_monotonic_s: float,
) -> list[PlannedControlStep]:
    """Convert one strict frame-grouped chunk to executable control steps."""

    return planned_frame_actions_to_control_steps(
        _chunk_to_planned_frames(
            first_chunk=chunk,
            frame_chunk_size=frame_chunk_size,
            action_per_frame=action_per_frame,
            source=source,
            ready_monotonic_s=ready_monotonic_s,
        )
    )


def build_exact_startup_conditioning_history_record(
    *,
    chunk,
    initial_video_latents: torch.Tensor,
    initial_obs: dict[str, np.ndarray],
    action_per_frame: int,
    frame_chunk_size: int,
    conditioning_frame_index: int | None = None,
    raw_actions_override: np.ndarray | None = None,
    proprio_state: np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build the observed prefix record used by strict frame-grouped startup."""

    if chunk.raw_chunk_action_pred is None:
        raise RuntimeError("Exact runner did not produce raw 7D LIBERO actions.")
    raw_actions = rearrange(
        chunk.raw_chunk_action_pred[0],
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    resolved_conditioning_frame_index = (
        int(chunk.debug.get("generation_frame_start", 0))
        if conditioning_frame_index is None
        else int(conditioning_frame_index)
    )
    generation_frame_start = int(
        chunk.debug.get(
            "generation_frame_start",
            resolved_conditioning_frame_index,
        )
    )
    require_strict_startup_generation_frame(generation_frame_start)
    resolved_raw_actions = (
        raw_actions[0].detach().to(dtype=torch.float32).cpu().numpy()
        if raw_actions_override is None
        else np.asarray(raw_actions_override, dtype=np.float32)
    )
    raw_actions_valid = generation_frame_start <= resolved_conditioning_frame_index
    if not raw_actions_valid:
        resolved_raw_actions = np.zeros(
            (0, int(raw_actions.shape[-1])),
            dtype=np.float32,
        )
    record = {
        "absolute_frame_index": int(resolved_conditioning_frame_index),
        "obs": {
            key: np.array(value, copy=True)
            for key, value in initial_obs.items()
        },
        "obs_sequence": [],
        "raw_actions": resolved_raw_actions,
        "raw_actions_valid": bool(raw_actions_valid),
        "raw_action_dim": int(raw_actions.shape[-1]),
        "video_latents": initial_video_latents.detach(),
        "source": "startup_conditioning_frame",
    }
    if proprio_state is not None:
        record["proprio_state"] = realtime_history.proprio_state_to_numpy(
            proprio_state
        )
    return record


def apply_frame_planner_result(
    result: FramePlannerJobResult,
    *,
    config: ExperimentConfig,
    plan_by_action: dict[int, PlannedControlStep],
    next_action_to_execute: int,
    pending_history: list[dict[str, Any]],
    history_base_session: VariantRolloutSession,
    current_chunk_session: VariantRolloutSession,
    buffer_tail_session: VariantRolloutSession | None,
    replan_records: list[dict[str, Any]],
    extension_records: list[dict[str, Any]],
    min_future_actions_to_accept_stale_chunk: int = 0,
) -> FramePlannerResultApplication:
    """Apply one typed frame planner result to scheduler-owned state."""

    planned_steps = planned_frame_actions_to_control_steps(result.planned_frames)
    (
        mergeable_planned_steps,
        chunk_boundary_dropped_actions,
        partial_stale_chunk_accepted_actions,
    ) = drop_partial_stale_control_chunk(
        planned_steps,
        next_action_to_execute=next_action_to_execute,
        min_future_actions_to_accept_stale_chunk=(
            min_future_actions_to_accept_stale_chunk
        ),
    )
    future_planned_steps = future_control_steps(
        mergeable_planned_steps,
        next_action_to_execute=next_action_to_execute,
    )
    chunk_accepted = bool(future_planned_steps)
    result.trace["planned_action_indices"] = [
        int(step.absolute_action_index) for step in planned_steps
    ]
    result.trace["future_planned_actions"] = int(len(future_planned_steps))
    result.trace["stale_planned_actions"] = int(
        len(planned_steps) - len(future_planned_steps)
    )
    result.trace["chunk_boundary_dropped_actions"] = int(
        chunk_boundary_dropped_actions
    )
    result.trace["partial_stale_chunk_accepted_actions"] = int(
        partial_stale_chunk_accepted_actions
    )
    result.trace["accepted_chunk"] = bool(chunk_accepted)

    if result.job_kind == "history_replan":
        replan_records.append(result.trace)
        if chunk_accepted:
            if result.submitted_through_frame is None or result.session is None:
                raise RuntimeError(
                    "Accepted history replan is missing its submitted frame or "
                    "output session."
                )
            pending_history = [
                record
                for record in pending_history
                if int(record["absolute_frame_index"])
                > int(result.submitted_through_frame)
            ]
            current_chunk_session = result.session
            history_base_session = resolve_next_exact_history_base_session(
                config=config,
                result=result,
                history_base_session=history_base_session,
            )
            buffer_tail_session = result.buffer_tail_session
    elif result.job_kind == "open_loop_extension":
        extension_records.append(result.trace)
        buffer_tail_session = (
            result.buffer_tail_session if chunk_accepted else None
        )
    else:
        raise AssertionError(
            f"Unhandled frame planner job kind: {result.job_kind!r}."
        )

    return FramePlannerResultApplication(
        history_base_session=history_base_session,
        current_chunk_session=current_chunk_session,
        buffer_tail_session=buffer_tail_session,
        plan_by_action=merge_future_control_steps(
            plan_by_action,
            mergeable_planned_steps,
            next_action_to_execute=next_action_to_execute,
        ),
        pending_history=pending_history,
    )


def apply_sequence_replan_result(
    *,
    result: SequenceReplanJobResult,
    replan_records: list[dict[str, Any]],
    plan_by_action: dict[int, PlannedControlStep],
    next_action_to_execute: int,
) -> tuple[
    VariantRolloutSession,
    int,
    dict[int, PlannedControlStep],
]:
    """Commit one sequence replan result to the active executable plan."""

    replan_records.append(result.trace)
    return (
        result.session,
        int(result.next_generation_action_start),
        merge_future_control_steps(
            plan_by_action,
            result.planned_steps,
            next_action_to_execute=next_action_to_execute,
        ),
    )


def annotate_sequence_planner_acceptance(
    result: SequenceReplanJobResult,
    *,
    next_action_to_execute: int,
) -> list[PlannedControlStep]:
    """Record whether a completed sequence plan still has executable actions."""

    future_steps = future_control_steps(
        result.planned_steps,
        next_action_to_execute=next_action_to_execute,
    )
    result.trace["future_planned_actions"] = int(len(future_steps))
    result.trace["stale_planned_actions"] = int(
        len(result.planned_steps) - len(future_steps)
    )
    result.trace["accepted_chunk"] = bool(future_steps)
    return future_steps


def _chunk_to_planned_frames(
    *,
    first_chunk,
    frame_chunk_size: int,
    action_per_frame: int,
    source: str,
    ready_monotonic_s: float,
    generation_frame_start: int | None = None,
) -> list[PlannedFrameAction]:
    if first_chunk.raw_chunk_action_pred is None:
        raise RuntimeError("Exact runner did not produce raw 7D LIBERO actions.")
    raw_actions = rearrange(
        first_chunk.raw_chunk_action_pred[0],
        "(f a) c -> f a c",
        f=frame_chunk_size,
        a=action_per_frame,
    )
    resolved_generation_frame_start = (
        int(first_chunk.debug.get("generation_frame_start", 0))
        if generation_frame_start is None
        else int(generation_frame_start)
    )
    require_strict_startup_generation_frame(resolved_generation_frame_start)
    return make_planned_frame_actions(
        raw_actions.detach().to(dtype=torch.float32).cpu().numpy(),
        generation_frame_start=resolved_generation_frame_start,
        source=source,
        planner_step_index=int(first_chunk.session.policy_state.step_index),
        ready_monotonic_s=ready_monotonic_s,
    )


def _session_for_next_chunk(
    session,
    *,
    next_frame_start: int,
    frame_chunk_size: int,
):
    next_cache = dict(session.policy_state.cache)
    next_cache["frame_start"] = int(next_frame_start)
    next_state = PolicyInferState(
        step_index=int(session.policy_state.step_index),
        cursor=RolloutCursor(
            current_start_frame=int(next_frame_start),
            block_index=int(session.policy_state.cursor.block_index + 1),
            chunk_size=int(frame_chunk_size),
        ),
        cache=next_cache,
        decoder_state=session.policy_state.decoder_state,
    )
    return type(session)(
        policy_state=next_state,
        task_text=session.task_text,
        text_context=session.text_context,
        negative_text_context=session.negative_text_context,
    )


def resolve_exact_startup_sessions(
    *,
    config,
    startup_session,
    first_chunk,
    frame_chunk_size: int,
):
    generation_frame_start = int(first_chunk.debug.get("generation_frame_start", 0))
    require_strict_startup_generation_frame(generation_frame_start)
    current_chunk_session = first_chunk.session
    history_base_session = _resolve_exact_startup_history_base_session(
        config=config,
        startup_session=startup_session,
        current_chunk_session=current_chunk_session,
    )
    buffer_tail_session = _session_for_next_chunk(
        current_chunk_session,
        next_frame_start=generation_frame_start + int(frame_chunk_size),
        frame_chunk_size=int(frame_chunk_size),
    )
    return history_base_session, current_chunk_session, buffer_tail_session


def _resolve_exact_startup_history_base_session(
    *,
    config,
    startup_session,
    current_chunk_session,
):
    runtime_mode = getattr(config.policy_variant, "runtime_mode", None)
    if runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
        return startup_session
    return current_chunk_session


def resolve_next_exact_history_base_session(
    *,
    config: ExperimentConfig,
    result: FramePlannerJobResult,
    history_base_session: VariantRolloutSession,
) -> VariantRolloutSession:
    runtime_mode = getattr(config.policy_variant, "runtime_mode", None)
    if runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
        return (
            history_base_session
            if result.warmup_session is None
            else result.warmup_session
        )
    if result.session is None:
        raise RuntimeError("History replan result is missing its output session.")
    return result.session


def build_fallback_frame_actions(
    *,
    action_dim: int,
    action_per_frame: int,
    policy: str,
    last_action: np.ndarray,
    preserve_absolute_tail_from: int | None = 6,
) -> np.ndarray:
    deadline_policy = DeadlineMissPolicy(policy)
    if deadline_policy is DeadlineMissPolicy.ZERO:
        return np.zeros((action_per_frame, action_dim), dtype=np.float32)
    if deadline_policy is DeadlineMissPolicy.HOLD_STATE:
        action = np.zeros((action_dim,), dtype=np.float32)
        last = np.asarray(last_action, dtype=np.float32)
        if last.shape != (action_dim,):
            raise ValueError(
                "Fallback last-action shape mismatch, "
                f"expected {(action_dim,)}, got {tuple(last.shape)}."
            )
        if preserve_absolute_tail_from is not None and action_dim > int(preserve_absolute_tail_from):
            action[int(preserve_absolute_tail_from) :] = last[int(preserve_absolute_tail_from) :]
        return np.repeat(action[None, :], action_per_frame, axis=0)
    repeated = np.repeat(np.asarray(last_action, dtype=np.float32)[None, :], action_per_frame, axis=0)
    if repeated.shape != (action_per_frame, action_dim):
        raise ValueError(
            "Fallback last-action shape mismatch, "
            f"expected {(action_per_frame, action_dim)}, got {tuple(repeated.shape)}."
        )
    return repeated
