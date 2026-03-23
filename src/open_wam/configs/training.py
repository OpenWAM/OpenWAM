from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Training-layer config shared by all future action heads."""

    video_num_train_timesteps: int = 1000
    action_num_train_timesteps: int = 1000
    video_sigma_shift: float = 5.0
    action_sigma_shift: float = 1.0
    use_teacher_forcing: bool = False
    chunk_size: int = 2
    window_size: int = 8

