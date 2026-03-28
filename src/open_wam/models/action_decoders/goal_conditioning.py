from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from open_wam.configs import GoalConditioningAdapterFamily


class GoalConditioningAdapter(nn.Module, ABC):
    """Shared adapter for injecting goal/language context into sequence features."""

    @abstractmethod
    def forward(
        self,
        sequence_features: torch.Tensor,
        goal_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return sequence features conditioned on optional goal context."""


class PassthroughGoalConditioningAdapter(GoalConditioningAdapter):
    """Leave sequence features unchanged."""

    def forward(
        self,
        sequence_features: torch.Tensor,
        goal_features: torch.Tensor | None,
    ) -> torch.Tensor:
        del goal_features
        return sequence_features


class MeanPoolGoalConditioningAdapter(GoalConditioningAdapter):
    """Inject a pooled goal embedding additively across the sequence."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.goal_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        sequence_features: torch.Tensor,
        goal_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if goal_features is None:
            return sequence_features
        if goal_features.ndim == 3:
            pooled_goal = goal_features.mean(dim=1)
        elif goal_features.ndim == 2:
            pooled_goal = goal_features
        else:
            raise ValueError(
                "Goal conditioning expects `[B, L, D]` or `[B, D]`, "
                f"got {tuple(goal_features.shape)}"
            )
        return sequence_features + self.goal_proj(pooled_goal)[:, None, :]


def build_goal_conditioning_adapter(
    family: GoalConditioningAdapterFamily | str,
    *,
    hidden_size: int,
) -> GoalConditioningAdapter:
    resolved = GoalConditioningAdapterFamily(family)
    if resolved == GoalConditioningAdapterFamily.PASSTHROUGH:
        return PassthroughGoalConditioningAdapter()
    if resolved == GoalConditioningAdapterFamily.MEAN_POOL:
        return MeanPoolGoalConditioningAdapter(hidden_size=hidden_size)
    raise ValueError(f"Unsupported goal conditioning adapter family '{resolved}'.")
