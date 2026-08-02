"""Dependency-light transforms over realtime action-plan queues."""

from __future__ import annotations

from typing import Mapping, Sequence

from open_wam.integrations.realtime_contracts import (
    PlannedControlStep,
    PlannedFrameAction,
)


__all__ = [
    "drop_control_steps_from",
    "drop_partial_stale_control_chunk",
    "future_control_depth",
    "future_control_steps",
    "merge_future_control_steps",
    "merge_future_frame_actions",
    "missing_control_action_indices",
    "required_control_action_indices",
]


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
