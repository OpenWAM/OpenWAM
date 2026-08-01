"""Realtime frame/action planning and control-loop reporting contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from open_wam.configs.enums import (
    FallbackHistoryPolicy,
    RealtimeEmptyPlanPolicy,
    RealtimePlannerJob,
    RealtimePlannerMode,
    RealtimeSchedulerProfile,
)


__all__ = [
    "PlannedControlStep",
    "PlannedFrameAction",
    "RealtimeSchedulerDefaults",
    "build_live_rollout_summary",
    "drop_control_steps_from",
    "drop_partial_stale_control_chunk",
    "frame_index_to_action_start",
    "future_control_depth",
    "future_control_steps",
    "make_planned_frame_actions",
    "merge_future_control_steps",
    "merge_future_frame_actions",
    "missing_control_action_indices",
    "planned_frame_actions_to_control_steps",
    "required_control_action_indices",
    "resolve_realtime_planner_mode",
    "resolve_realtime_scheduler_defaults",
    "select_realtime_planner_job",
    "should_submit_frame_grouped_planner",
    "should_submit_realtime_planner_job",
    "should_submit_sequence_planner",
    "summarize_scalars",
]


@dataclass(frozen=True)
class PlannedFrameAction:
    """One frame-aligned action block that can be scheduled in live control."""

    absolute_frame_index: int
    generation_frame_start: int
    frame_offset: int
    raw_actions: np.ndarray
    source: str = "history_replan"
    planner_step_index: int | None = None
    ready_monotonic_s: float | None = None


@dataclass(frozen=True)
class PlannedControlStep:
    """One executable control step with its prediction provenance.

    A step carries either a model-native ``raw_action`` or an absolute pose
    target that a benchmark adapter materializes against the live state.
    """

    absolute_action_index: int
    generation_action_start: int
    source: str
    planner_step_index: int | None = None
    ready_monotonic_s: float | None = None
    generation_frame_start: int | None = None
    raw_action: np.ndarray | None = None
    desired_position: np.ndarray | None = None
    desired_quaternion: np.ndarray | None = None
    desired_gripper: np.ndarray | None = None


@dataclass(frozen=True)
class RealtimeSchedulerDefaults:
    """Optional low-level overrides selected by one scheduler profile."""

    planner_mode: RealtimePlannerMode | None = None
    empty_plan_policy: RealtimeEmptyPlanPolicy | None = None
    fallback_history_policy: FallbackHistoryPolicy | None = None
    startup_open_loop_chunks: int | None = None
    replan_low_watermark_actions: int | None = None

    def to_override_mapping(self) -> dict[str, object]:
        """Return only profile-owned values using CLI/config field names."""

        values: dict[str, object | None] = {
            "planner_mode": self.planner_mode,
            "sequence_empty_plan_policy": self.empty_plan_policy,
            "fallback_history_policy": self.fallback_history_policy,
            "startup_open_loop_chunks": self.startup_open_loop_chunks,
            "replan_low_watermark_actions": self.replan_low_watermark_actions,
        }
        return {key: value for key, value in values.items() if value is not None}


_REALTIME_SCHEDULER_DEFAULTS: Mapping[
    RealtimeSchedulerProfile,
    RealtimeSchedulerDefaults,
] = {
    RealtimeSchedulerProfile.MANUAL: RealtimeSchedulerDefaults(),
    RealtimeSchedulerProfile.BLOCKING_CONTROL: RealtimeSchedulerDefaults(
        planner_mode=RealtimePlannerMode.HISTORY_ONLY,
        empty_plan_policy=RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN,
        fallback_history_policy=FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY,
        startup_open_loop_chunks=0,
        replan_low_watermark_actions=0,
    ),
    RealtimeSchedulerProfile.FREEZE_UNTIL_CLEAN_CHUNK: RealtimeSchedulerDefaults(
        planner_mode=RealtimePlannerMode.HISTORY_ONLY,
        empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
        fallback_history_policy=FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
        startup_open_loop_chunks=0,
        replan_low_watermark_actions=0,
    ),
    RealtimeSchedulerProfile.ASYNC_HISTORY_FIRST: RealtimeSchedulerDefaults(
        planner_mode=RealtimePlannerMode.ASYNC_HISTORY_FIRST,
        empty_plan_policy=RealtimeEmptyPlanPolicy.FALLBACK,
        fallback_history_policy=FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
        startup_open_loop_chunks=1,
    ),
}


def resolve_realtime_scheduler_defaults(
    profile: RealtimeSchedulerProfile | str,
) -> RealtimeSchedulerDefaults:
    """Resolve one named scheduler profile into typed optional overrides."""

    return _REALTIME_SCHEDULER_DEFAULTS[RealtimeSchedulerProfile(profile)]


def select_realtime_planner_job(
    *,
    planner_mode: RealtimePlannerMode | str,
    history_count: int,
    future_buffer_depth: int,
    has_buffer_tail_session: bool,
) -> RealtimePlannerJob | None:
    """Select observed-history replanning or open-loop buffer extension."""

    mode = RealtimePlannerMode(planner_mode)
    has_history = int(history_count) > 0
    if mode == RealtimePlannerMode.HISTORY_ONLY:
        return RealtimePlannerJob.HISTORY_REPLAN if has_history else None
    if mode == RealtimePlannerMode.ASYNC_BUFFER:
        if has_buffer_tail_session and int(future_buffer_depth) <= 3:
            return RealtimePlannerJob.BUFFER_EXTENSION
        if has_history:
            return RealtimePlannerJob.HISTORY_REPLAN
        if has_buffer_tail_session and int(future_buffer_depth) <= 6:
            return RealtimePlannerJob.BUFFER_EXTENSION
        return None
    if mode == RealtimePlannerMode.ASYNC_HISTORY_FIRST:
        if has_history:
            return RealtimePlannerJob.HISTORY_REPLAN
        if has_buffer_tail_session and int(future_buffer_depth) <= 6:
            return RealtimePlannerJob.BUFFER_EXTENSION
        return None
    if mode == RealtimePlannerMode.ASYNC_MIX:
        if int(history_count) >= 2 and int(future_buffer_depth) >= 2:
            return RealtimePlannerJob.HISTORY_REPLAN
        if has_buffer_tail_session and int(future_buffer_depth) <= 3:
            return RealtimePlannerJob.BUFFER_EXTENSION
        if has_history:
            return RealtimePlannerJob.HISTORY_REPLAN
        if has_buffer_tail_session and int(future_buffer_depth) <= 6:
            return RealtimePlannerJob.BUFFER_EXTENSION
        return None
    raise AssertionError(f"Unhandled realtime planner mode: {mode!r}")


def should_submit_realtime_planner_job(
    *,
    planner_mode: RealtimePlannerMode | str,
    history_count: int,
    future_buffer_depth: int,
    has_buffer_tail_session: bool,
) -> bool:
    """Return whether a planner scheduling slot has useful work."""

    return select_realtime_planner_job(
        planner_mode=planner_mode,
        history_count=history_count,
        future_buffer_depth=future_buffer_depth,
        has_buffer_tail_session=has_buffer_tail_session,
    ) is not None


def resolve_realtime_planner_mode(
    *,
    planner_mode: RealtimePlannerMode | str,
    empty_plan_policy: RealtimeEmptyPlanPolicy | str,
    has_history: bool,
) -> RealtimePlannerMode:
    """Force history-only planning while a blocking loop has observations."""

    mode = RealtimePlannerMode(planner_mode)
    policy = RealtimeEmptyPlanPolicy(empty_plan_policy)
    if policy == RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN and has_history:
        return RealtimePlannerMode.HISTORY_ONLY
    return mode


def should_submit_frame_grouped_planner(
    *,
    future_buffer_depth_actions: int,
    future_buffer_depth_frames: int,
    empty_plan_policy: RealtimeEmptyPlanPolicy | str,
    replan_low_watermark_actions: int = 0,
) -> bool:
    """Gate a frame-grouped planner using blocking or action low-watermark policy."""

    policy = RealtimeEmptyPlanPolicy(empty_plan_policy)
    if policy == RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN:
        return int(future_buffer_depth_frames) <= 0
    if int(replan_low_watermark_actions) <= 0:
        return True
    return int(future_buffer_depth_actions) <= int(replan_low_watermark_actions)


def should_submit_sequence_planner(
    *,
    planner_mode: RealtimePlannerMode | str,
    future_buffer_depth_actions: int,
    empty_plan_policy: RealtimeEmptyPlanPolicy | str,
    sequence_buffer_threshold: int,
    replan_low_watermark_actions: int = 0,
) -> bool:
    """Gate an action-sequence planner using its future-buffer threshold."""

    mode = RealtimePlannerMode(planner_mode)
    RealtimeEmptyPlanPolicy(empty_plan_policy)
    if mode == RealtimePlannerMode.HISTORY_ONLY:
        return int(future_buffer_depth_actions) <= 0
    threshold = (
        int(replan_low_watermark_actions)
        if int(replan_low_watermark_actions) > 0
        else int(sequence_buffer_threshold)
    )
    return int(future_buffer_depth_actions) <= threshold


def frame_index_to_action_start(frame_index: int, action_per_frame: int) -> int:
    """Map a generated frame index to its first zero-based control step.

    Frame zero is conditioning-only; generated frame one starts at action zero.
    """

    return max(0, int(frame_index) - 1) * int(action_per_frame)


def planned_frame_actions_to_control_steps(
    planned_frames: Sequence[PlannedFrameAction],
) -> list[PlannedControlStep]:
    """Expand frame-aligned action blocks into independently scheduled steps."""

    planned_steps: list[PlannedControlStep] = []
    for planned_frame in planned_frames:
        raw_actions = np.asarray(planned_frame.raw_actions, dtype=np.float32)
        action_per_frame = int(raw_actions.shape[0])
        generation_frame_start = int(planned_frame.generation_frame_start)
        generation_action_start = frame_index_to_action_start(
            generation_frame_start,
            action_per_frame,
        )
        for action_offset in range(action_per_frame):
            absolute_action_index = (
                frame_index_to_action_start(
                    int(planned_frame.absolute_frame_index),
                    action_per_frame,
                )
                + action_offset
            )
            planned_steps.append(
                PlannedControlStep(
                    absolute_action_index=int(absolute_action_index),
                    generation_action_start=int(generation_action_start),
                    generation_frame_start=int(generation_frame_start),
                    source=str(planned_frame.source),
                    planner_step_index=planned_frame.planner_step_index,
                    ready_monotonic_s=planned_frame.ready_monotonic_s,
                    raw_action=np.array(raw_actions[action_offset], copy=True),
                )
            )
    return planned_steps


def merge_future_control_steps(
    existing: Mapping[int, PlannedControlStep],
    incoming: Sequence[PlannedControlStep],
    *,
    next_action_to_execute: int,
) -> dict[int, PlannedControlStep]:
    """Drop stale steps and replace future indices with fresher predictions."""

    merged = {
        int(action_index): plan
        for action_index, plan in existing.items()
        if int(action_index) >= int(next_action_to_execute)
    }
    for plan in incoming:
        if int(plan.absolute_action_index) < int(next_action_to_execute):
            continue
        merged[int(plan.absolute_action_index)] = plan
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def future_control_depth(
    plan_by_action: Mapping[int, PlannedControlStep],
    *,
    next_action_to_execute: int,
) -> int:
    """Return the distance to the furthest planned action, including gaps."""

    if not plan_by_action:
        return 0
    return max(
        0,
        max(int(action_id) for action_id in plan_by_action)
        - int(next_action_to_execute)
        + 1,
    )


def required_control_action_indices(
    *,
    next_action_index: int,
    max_actions: int,
    action_per_frame: int,
) -> list[int]:
    """Return the next frame's executable indices within the rollout limit."""

    action_count = min(
        int(action_per_frame),
        max(0, int(max_actions) - int(next_action_index)),
    )
    return [int(next_action_index) + offset for offset in range(action_count)]


def missing_control_action_indices(
    plan_by_action: Mapping[int, PlannedControlStep],
    required_action_indices: Sequence[int],
) -> list[int]:
    """Report required action indices that do not yet have a plan."""

    return [
        int(action_index)
        for action_index in required_action_indices
        if int(action_index) not in plan_by_action
    ]


def future_control_steps(
    planned_steps: Sequence[PlannedControlStep],
    *,
    next_action_to_execute: int,
) -> list[PlannedControlStep]:
    """Select steps that have not passed the execution cursor."""

    return [
        step
        for step in planned_steps
        if int(step.absolute_action_index) >= int(next_action_to_execute)
    ]


def drop_control_steps_from(
    plan_by_action: Mapping[int, PlannedControlStep],
    *,
    replace_from_action: int,
) -> dict[int, PlannedControlStep]:
    """Keep only control steps before a replacement boundary."""

    return {
        int(action_index): plan
        for action_index, plan in plan_by_action.items()
        if int(action_index) < int(replace_from_action)
    }


def drop_partial_stale_control_chunk(
    planned_steps: Sequence[PlannedControlStep],
    *,
    next_action_to_execute: int,
    min_future_actions_to_accept_stale_chunk: int = 0,
) -> tuple[list[PlannedControlStep], int, int]:
    """Apply atomic acceptance policy when execution overtakes part of a chunk.

    Returns ``(mergeable, dropped_future_count, accepted_partial_count)``.
    Fully stale chunks are returned unchanged because the subsequent future-plan
    merge owns ordinary stale filtering. A partially stale chunk is either
    rejected as a unit or accepted as a sufficiently large future suffix.
    """

    steps = list(planned_steps)
    stale_steps = [
        step
        for step in steps
        if int(step.absolute_action_index) < int(next_action_to_execute)
    ]
    future_steps = [
        step
        for step in steps
        if int(step.absolute_action_index) >= int(next_action_to_execute)
    ]
    if stale_steps and future_steps:
        minimum = int(min_future_actions_to_accept_stale_chunk)
        if minimum > 0 and len(future_steps) >= minimum:
            return future_steps, 0, len(future_steps)
        return [], len(future_steps), 0
    return steps, 0, 0


def make_planned_frame_actions(
    frame_actions: np.ndarray,
    *,
    generation_frame_start: int,
    source: str = "history_replan",
    planner_step_index: int | None = None,
    ready_monotonic_s: float | None = None,
) -> list[PlannedFrameAction]:
    """Attach absolute rollout-frame ids to one generated action chunk."""

    actions = np.asarray(frame_actions, dtype=np.float32)
    if actions.ndim != 3:
        raise ValueError(
            "Expected `frame_actions` to have shape [num_frames, action_per_frame, action_dim], "
            f"got shape={tuple(actions.shape)}."
        )
    planned_frames: list[PlannedFrameAction] = []
    for frame_offset in range(actions.shape[0]):
        planned_frames.append(
            PlannedFrameAction(
                absolute_frame_index=int(generation_frame_start + frame_offset),
                generation_frame_start=int(generation_frame_start),
                frame_offset=int(frame_offset),
                raw_actions=np.array(actions[frame_offset], copy=True),
                source=str(source),
                planner_step_index=planner_step_index,
                ready_monotonic_s=ready_monotonic_s,
            )
        )
    return planned_frames


def merge_future_frame_actions(
    existing: Mapping[int, PlannedFrameAction],
    incoming: Sequence[PlannedFrameAction],
    *,
    next_frame_to_execute: int,
) -> dict[int, PlannedFrameAction]:
    """Drop stale plans and replace future frames with fresher predictions."""

    merged = {
        int(frame_index): plan
        for frame_index, plan in existing.items()
        if int(frame_index) >= int(next_frame_to_execute)
    }
    for plan in incoming:
        if int(plan.absolute_frame_index) < int(next_frame_to_execute):
            continue
        merged[int(plan.absolute_frame_index)] = plan
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def summarize_scalars(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return compact scalar distribution stats for JSON reporting."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_live_rollout_summary(
    *,
    action_records: Sequence[Mapping[str, Any]],
    replan_records: Sequence[Mapping[str, Any]],
    target_action_hz: float,
    live_wall_time_s: float,
    startup_prepare_s: float,
    startup_infer_s: float,
    deadline_tolerance_s: float = 0.002,
) -> dict[str, Any]:
    """Aggregate action-loop and replan-loop metrics for one rollout."""

    total_actions = len(action_records)
    planned_actions = sum(1 for record in action_records if not str(record.get("source", "")).startswith("fallback_"))
    startup_plan_actions = sum(1 for record in action_records if str(record.get("source")) == "startup_plan")
    history_replan_actions = sum(1 for record in action_records if str(record.get("source")) == "history_replan")
    observation_conditioned_actions = startup_plan_actions + history_replan_actions
    open_loop_extension_actions = sum(
        1 for record in action_records if str(record.get("source")) == "open_loop_extension"
    )
    fallback_actions = total_actions - planned_actions
    action_lateness = [float(record["lateness_s"]) for record in action_records]
    env_step_times = [float(record["env_step_s"]) for record in action_records]
    action_indices = [
        int(record["absolute_action_index"])
        for record in action_records
        if record.get("absolute_action_index") is not None
    ]
    frame_indices = [
        int(record["absolute_frame_index"])
        for record in action_records
        if record.get("absolute_frame_index") is not None
    ]
    generation_lag_actions = [
        int(record["generation_lag_actions"])
        for record in action_records
        if record.get("generation_lag_actions") is not None
    ]
    generation_lag_frames = [
        int(record["generation_lag_frames"])
        for record in action_records
        if record.get("generation_lag_frames") is not None
    ]
    replan_latencies = [float(record["total_latency_s"]) for record in replan_records]
    replan_prepare = [float(record["prepare_s"]) for record in replan_records]
    replan_warmup = [float(record["warmup_s"]) for record in replan_records]
    replan_infer = [float(record["infer_s"]) for record in replan_records]
    deadline_hits = sum(1 for value in action_lateness if value <= float(deadline_tolerance_s))
    unique_frames = set(frame_indices)
    unique_action_steps = set(action_indices)

    return {
        "target_action_hz": float(target_action_hz),
        "target_action_period_s": float(1.0 / target_action_hz),
        "live_wall_time_s": float(live_wall_time_s),
        "startup_prepare_s": float(startup_prepare_s),
        "startup_infer_s": float(startup_infer_s),
        "total_actions": int(total_actions),
        "total_frames": int(len(unique_frames)),
        "total_action_steps": int(len(unique_action_steps)) if unique_action_steps else int(total_actions),
        "planned_actions": int(planned_actions),
        "startup_plan_actions": int(startup_plan_actions),
        "history_replan_actions": int(history_replan_actions),
        "observation_conditioned_actions": int(observation_conditioned_actions),
        "open_loop_extension_actions": int(open_loop_extension_actions),
        "fallback_actions": int(fallback_actions),
        "achieved_action_hz": float(total_actions / live_wall_time_s) if live_wall_time_s > 0 else 0.0,
        "planned_action_hz": float(planned_actions / live_wall_time_s) if live_wall_time_s > 0 else 0.0,
        "startup_plan_action_hz": float(startup_plan_actions / live_wall_time_s) if live_wall_time_s > 0 else 0.0,
        "history_replan_action_hz": (
            float(history_replan_actions / live_wall_time_s) if live_wall_time_s > 0 else 0.0
        ),
        "observation_conditioned_action_hz": (
            float(observation_conditioned_actions / live_wall_time_s) if live_wall_time_s > 0 else 0.0
        ),
        "open_loop_extension_action_hz": (
            float(open_loop_extension_actions / live_wall_time_s) if live_wall_time_s > 0 else 0.0
        ),
        "fallback_action_hz": float(fallback_actions / live_wall_time_s) if live_wall_time_s > 0 else 0.0,
        "deadline_tolerance_s": float(deadline_tolerance_s),
        "deadline_hit_rate": float(deadline_hits / total_actions) if total_actions > 0 else 0.0,
        "action_lateness_s": summarize_scalars(action_lateness),
        "env_step_s": summarize_scalars(env_step_times),
        "generation_lag_actions": summarize_scalars(generation_lag_actions),
        "generation_lag_frames": summarize_scalars(generation_lag_frames),
        "replan_total_latency_s": summarize_scalars(replan_latencies),
        "replan_prepare_s": summarize_scalars(replan_prepare),
        "replan_warmup_s": summarize_scalars(replan_warmup),
        "replan_infer_s": summarize_scalars(replan_infer),
    }
