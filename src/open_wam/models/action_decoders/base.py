from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.models.policy_variants.contracts import PolicyInferOutput, PolicyTrainBatch, PolicyTrainOutput


@dataclass
class ActionDecoderTrainOutput:
    """Common train-time action-decoder outputs."""

    action_pred: torch.Tensor
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecoderInferOutput:
    """Common inference-time action-decoder outputs."""

    action_pred: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)


def align_policy_features(policy_features: torch.Tensor, target_length: int) -> torch.Tensor:
    """Interpolate `[B, T, D]` features to the action horizon."""

    if policy_features.shape[1] == target_length:
        return policy_features
    return F.interpolate(
        policy_features.transpose(1, 2),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class ActionDecoder(nn.Module, ABC):
    """Action decoder interface shared across policy variants."""

    @abstractmethod
    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        """Decode actions and compute loss."""

    @abstractmethod
    def forward_infer(self, policy_output: PolicyInferOutput) -> ActionDecoderInferOutput:
        """Decode actions for one inference step."""


class LinearActionDecoder(ActionDecoder):
    """Reusable MLP decoder for horizon-aligned policy features."""

    def __init__(self, hidden_size: int, action_dim: int, action_horizon: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_dim),
        )

    def decode(self, policy_features: torch.Tensor) -> torch.Tensor:
        aligned = align_policy_features(policy_features, self.action_horizon)
        return self.proj(aligned)

    def forward_train(self, policy_output: PolicyTrainOutput, batch: PolicyTrainBatch) -> ActionDecoderTrainOutput:
        action_pred = self.decode(policy_output.policy_features)
        per_token_loss = F.mse_loss(action_pred, batch.actions, reduction="none")
        if batch.action_mask is not None:
            per_token_loss = per_token_loss * batch.action_mask.float()
            denom = batch.action_mask.float().sum().clamp_min(1.0)
        else:
            denom = torch.tensor(float(per_token_loss.numel()), device=per_token_loss.device)
        loss = per_token_loss.sum() / denom
        return ActionDecoderTrainOutput(
            action_pred=action_pred,
            loss=loss,
            metrics={"action_mse": loss.detach()},
            aux={"decoder": self.__class__.__name__},
        )

    def forward_infer(self, policy_output: PolicyInferOutput) -> ActionDecoderInferOutput:
        return ActionDecoderInferOutput(
            action_pred=self.decode(policy_output.policy_features),
            aux={"decoder": self.__class__.__name__},
        )
