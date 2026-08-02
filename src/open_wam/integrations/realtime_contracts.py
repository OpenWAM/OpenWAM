"""Dependency-light records shared by realtime control integrations."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from open_wam.configs.enums import (
    FallbackHistoryPolicy,
    RealtimeEmptyPlanPolicy,
    RealtimePlannerMode,
)

if TYPE_CHECKING:
    import numpy as np
else:
    np = SimpleNamespace(ndarray=Any)


__all__ = [
    "PlannedControlStep",
    "PlannedFrameAction",
    "RealtimeSchedulerDefaults",
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
