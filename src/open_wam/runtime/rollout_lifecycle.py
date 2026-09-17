"""Pure control/session publication rules, independent of models and scheduling.

Each operation returns a new state. Sessions and observations are opaque payloads;
only a driver executes controls and only a planner touches model state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Generic, TypeVar

from .control import (
    ControlCommand,
    ControlTransition,
    RolloutTermination,
    RolloutTerminationReason,
)
from .realtime_contracts import PlannedControlStep
from .realtime_plan_queue import (
    drop_control_steps_from,
    drop_partial_stale_control_chunk,
    future_control_steps,
    merge_future_control_steps,
)

Session = TypeVar("Session")
Observation = TypeVar("Observation")


@dataclass(frozen=True)
class RolloutLifecycle(Generic[Session, Observation]):
    """Authoritative accepted session, control plan and uncommitted observations.

    The session may contain a predicted tail. observed_control_end, not that
    tail's cursor, marks the history actually committed to the policy. Remaining
    observations/commands retain the exact executed interval, including a final
    partial model frame; termination never silently calls it reconciled.
    """

    session: Session
    observations: Mapping[int, Observation]
    commands: Mapping[int, ControlCommand]
    plan: Mapping[int, PlannedControlStep]
    next_control_index: int = 0
    observed_control_end: int = 0
    revision: int = 0
    termination: RolloutTermination[Observation] | None = None

    def __post_init__(self) -> None:
        for name in ("observations", "commands", "plan"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @classmethod
    def start(
        cls, session: Session, observation: Observation
    ) -> RolloutLifecycle[Session, Observation]:
        return cls(session, {0: observation}, {}, {})

    def publish(
        self,
        *,
        session: Session,
        steps: tuple[PlannedControlStep, ...],
        base_revision: int,
        observation_end: int | None,
    ) -> RolloutLifecycle[Session, Observation]:
        if self.termination is not None or base_revision != self.revision:
            return self
        mergeable, _ = drop_partial_stale_control_chunk(
            steps, next_action_to_execute=self.next_control_index
        )
        future = future_control_steps(
            mergeable, next_action_to_execute=self.next_control_index
        )
        if not future:
            return self
        plan = self.plan
        end = self.observed_control_end
        if observation_end is not None:
            if not end <= observation_end <= self.next_control_index:
                raise ValueError("A candidate cannot commit unobserved controls.")
            end = observation_end
            plan = drop_control_steps_from(
                plan, replace_from_action=future[0].absolute_action_index
            )
        return replace(
            self,
            session=session,
            plan=merge_future_control_steps(
                plan, future, next_action_to_execute=self.next_control_index
            ),
            observations={i: obs for i, obs in self.observations.items() if i >= end},
            commands={i: cmd for i, cmd in self.commands.items() if i >= end},
            observed_control_end=end,
            revision=self.revision + 1,
        )

    def commit_observed(
        self, *, session: Session, observation_end: int, base_revision: int
    ) -> RolloutLifecycle[Session, Observation]:
        """Commit observed history without publishing a new prediction.

        Used when a blocking driver stops at a plan budget. Unlike publish(),
        this operation cannot replace an executable plan or accept stale work.
        """
        if self.termination is not None or base_revision != self.revision:
            return self
        if self.plan or observation_end != self.next_control_index:
            raise ValueError("Observation-only commits require a fully executed plan.")
        if observation_end <= self.observed_control_end:
            raise ValueError("Observation-only commits must advance observed history.")
        return replace(
            self, session=session, observed_control_end=observation_end,
            observations={observation_end: self.observations[observation_end]},
            commands={}, revision=self.revision + 1,
        )

    def executed(
        self, command: ControlCommand, transition: ControlTransition[Observation]
    ) -> RolloutLifecycle[Session, Observation]:
        if self.termination is not None:
            raise ValueError("Cannot execute a control after rollout termination.")
        index = self.next_control_index
        state = replace(
            self,
            observations={**self.observations, index + 1: transition.observation},
            commands={**self.commands, index: command},
            plan={i: step for i, step in self.plan.items() if i != index},
            next_control_index=index + 1,
        )
        if transition.success or transition.done:
            return state.terminate(
                RolloutTerminationReason.SUCCESS
                if transition.success
                else RolloutTerminationReason.ENV_TERMINAL,
                transition=transition,
            )
        return state

    def terminate(
        self,
        reason: RolloutTerminationReason,
        *,
        transition: ControlTransition[Observation] | None = None,
        error: str | None = None,
    ) -> RolloutLifecycle[Session, Observation]:
        if self.termination is not None:
            return self
        return replace(
            self,
            termination=RolloutTermination(
                reason, self.next_control_index, transition=transition, error=error
            ),
        )
