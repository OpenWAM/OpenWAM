"""NumPy-backed realtime plan materialization and rollout reporting."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from open_wam.integrations.realtime_contracts import (
    PlannedControlStep,
    PlannedFrameAction,
    RealtimeSchedulerDefaults,
)
from open_wam.integrations.realtime_plan_queue import (
    drop_control_steps_from,
    drop_partial_stale_control_chunk,
    future_control_depth,
    future_control_steps,
    merge_future_control_steps,
    merge_future_frame_actions,
    missing_control_action_indices,
    required_control_action_indices,
)
from open_wam.integrations.realtime_scheduling import (
    frame_index_to_action_start,
    resolve_realtime_planner_mode,
    resolve_realtime_scheduler_defaults,
    select_realtime_planner_job,
    should_submit_frame_grouped_planner,
    should_submit_realtime_planner_job,
    should_submit_sequence_planner,
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
