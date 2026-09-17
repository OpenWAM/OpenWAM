"""Model-independent transactions accepted by the rollout control lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, TYPE_CHECKING

from .control import ControlCommand
from .realtime_contracts import PlannedControlStep
from .rollout_receipts import PlannerReceipt
from .rollout_temporal import ResolvedRolloutTemporalContract

if TYPE_CHECKING:
    import numpy as np

Observation = TypeVar("Observation")
Session = TypeVar("Session")


class ControlAdapter(Protocol[Observation]):
    """Materialize controls against live observations, independently of models."""

    def materialize(
        self, step: PlannedControlStep, observation: Observation
    ) -> ControlCommand: ...
    def fallback(
        self, last_action: np.ndarray | None, observation: Observation
    ) -> ControlCommand: ...


@dataclass(frozen=True)
class PlannerRequest(Generic[Session, Observation]):
    session: Session
    observations: tuple[Observation, ...]
    actions: tuple[np.ndarray, ...]
    start: int
    end: int
    reconcile: bool
    source: str
    base_revision: int


@dataclass(frozen=True)
class PlannerResult(Generic[Session]):
    session: Session
    steps: tuple[PlannedControlStep, ...]
    observation_end: int
    reconciled: bool
    receipt: PlannerReceipt
    base_revision: int


class RolloutPlanner(Protocol[Session, Observation]):
    """A model transaction over opaque sessions, never a simulator loop.

    observe commits a complete executed interval without predicting more controls.
    Stateful streaming frontends must prohibit speculative background execution.
    """

    temporal: ResolvedRolloutTemporalContract
    supports_speculative_continuation: bool
    supports_async: bool

    def plan(
        self, request: PlannerRequest[Session, Observation]
    ) -> PlannerResult[Session]: ...
    def observe(self, request: PlannerRequest[Session, Observation]) -> Session: ...
