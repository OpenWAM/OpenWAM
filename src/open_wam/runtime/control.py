"""Dependency-light controls and transitions shared by rollout schedulers and environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

if TYPE_CHECKING:
    import numpy as np

    _NumpyArray: TypeAlias = np.ndarray
else:
    _NumpyArray: TypeAlias = Any


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class ControlCommand:
    """Executable control and the same command in the dataset action schema.

    Backends apply clipping/conversion before publishing this pair. Recurrent
    history uses source_action, never an unexecuted model prediction.
    """

    action: _NumpyArray
    source_action: _NumpyArray


@dataclass(frozen=True)
class ControlTransition(Generic[ObservationT]):
    """One policy-visible simulator control transition."""

    observation: ObservationT
    reward: float | None = None
    done: bool = False
    success: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class RolloutTerminationReason(str, Enum):
    SUCCESS = "success"
    ENV_TERMINAL = "env_terminal"
    MAX_ACTIONS = "max_actions"
    MAX_PLANS = "max_plans"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RolloutTermination(Generic[ObservationT]):
    """Terminal outcome, preserving the actual final transition when available."""

    reason: RolloutTerminationReason
    control_count: int
    transition: ControlTransition[ObservationT] | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.reason is RolloutTerminationReason.SUCCESS
