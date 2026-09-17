"""Typed planner/control records; mappings exist only at artifact boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .control import ControlCommand, ControlTransition
from .rollout_temporal import ResolvedRolloutTemporalContract

Observation = TypeVar("Observation")


@dataclass(frozen=True)
class PlannerReceipt:
    source: str
    use_observation_update: bool
    observation_action_start: int
    observation_action_end: int
    model_generation_frame_start: int
    generation_action_start: int
    planned_action_ids: tuple[int, ...]
    policy_action_shape: tuple[int, ...]
    prepare_s: float
    infer_s: float
    total_latency_s: float
    ready_monotonic_s: float
    accepted: bool | None = None
    base_revision: int | None = None
    accepted_revision: int | None = None
    acceptance_action_index: int | None = None
    stale_planned_actions: int = 0
    rejected_future_actions: int = 0

    def to_record(self) -> dict[str, object]:
        from dataclasses import asdict

        record = asdict(self)
        record["planned_action_ids"] = list(self.planned_action_ids)
        record["policy_action_shape"] = list(self.policy_action_shape)
        record["warmup_s"] = 0.0
        if self.accepted is None:
            for name in (
                "accepted",
                "base_revision",
                "accepted_revision",
                "acceptance_action_index",
                "stale_planned_actions",
                "rejected_future_actions",
            ):
                del record[name]
        return record


@dataclass(frozen=True)
class ExecutedControlReceipt(Generic[Observation]):
    index: int
    command: ControlCommand
    transition: ControlTransition[Observation]
    source: str
    temporal: ResolvedRolloutTemporalContract
    scheduled_start_s: float
    actual_start_s: float
    env_step_s: float
    wait_for_plan_s: float
    generation_action_start: int | None
    planner_step_index: int | None

    def to_record(self) -> dict[str, object]:
        frame = self.temporal.frame_for_control(self.index)
        origin = self.generation_action_start
        lag = None if origin is None else self.index - origin
        return {
            "action_index": self.index,
            "absolute_action_index": self.index,
            "model_action_index": self.index,
            "action_offset": self.temporal.control_offset(self.index),
            "model_frame_index": frame,
            "absolute_frame_index": frame,
            "source": self.source,
            "action": self.command.action.tolist(),
            "source_action": self.command.source_action.tolist(),
            "done": self.transition.done,
            "success": self.transition.success,
            "reward": self.transition.reward,
            "frame_history_decision": "included",
            "scheduled_start_s": self.scheduled_start_s,
            "actual_start_s": self.actual_start_s,
            "lateness_s": max(0.0, self.actual_start_s - self.scheduled_start_s),
            "env_step_s": self.env_step_s,
            "generation_action_start": origin,
            "planner_step_index": self.planner_step_index,
            "generation_lag_actions": lag,
            "generation_lag_frames": None
            if lag is None
            else lag // self.temporal.controls_per_frame,
            "wait_for_plan_s": self.wait_for_plan_s,
        }
