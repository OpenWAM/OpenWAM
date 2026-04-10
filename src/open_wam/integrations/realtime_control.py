from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


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
