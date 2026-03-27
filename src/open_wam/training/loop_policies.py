from __future__ import annotations

from dataclasses import dataclass

from .state import TrainState


@dataclass(frozen=True)
class EpochLoopPolicy:
    """Epoch-oriented loop policy."""

    max_epochs: int
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None

    @property
    def name(self) -> str:
        return "epochs"

    def should_continue(self, state: TrainState) -> bool:
        return state.epoch_index < self.max_epochs


@dataclass(frozen=True)
class StepLoopPolicy:
    """Step-oriented loop policy."""

    max_steps: int
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None

    @property
    def name(self) -> str:
        return "steps"

    def should_continue(self, state: TrainState) -> bool:
        return state.optimizer_step < self.max_steps
