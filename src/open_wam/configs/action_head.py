from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionHeadConfig:
    """Head-layer config independent from train/infer orchestration."""

    name: str
    hidden_size: int
    action_dim: int
    action_horizon: int
    state_dim: int

