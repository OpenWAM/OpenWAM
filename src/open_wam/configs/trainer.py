from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConfig:
    """Lightning trainer config stored separately from model configs."""

    max_epochs: int = 1
    limit_train_batches: int = 2
    limit_val_batches: int = 1
    log_every_n_steps: int = 1
    accelerator: str = "cpu"
    devices: int = 1
    precision: str = "32-true"
    enable_checkpointing: bool = False
    enable_model_summary: bool = False

