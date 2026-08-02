"""Dependency-light policy for realtime planner scheduling."""

from __future__ import annotations

from typing import Mapping

from open_wam.configs.enums import (
    FallbackHistoryPolicy,
    RealtimeEmptyPlanPolicy,
    RealtimePlannerJob,
    RealtimePlannerMode,
    RealtimeSchedulerProfile,
)
from open_wam.integrations.realtime_contracts import RealtimeSchedulerDefaults


__all__ = [
    "frame_index_to_action_start",
    "resolve_realtime_planner_mode",
    "resolve_realtime_scheduler_defaults",
    "select_realtime_planner_job",
    "should_submit_frame_grouped_planner",
    "should_submit_realtime_planner_job",
    "should_submit_sequence_planner",
]


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
