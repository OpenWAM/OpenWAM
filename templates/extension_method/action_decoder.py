"""Minimal runnable application-owned action decoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from open_wam.sdk.config import ExtensionActionDecoderConfig
from open_wam.sdk.policy import (
    ActionDecoder,
    ActionDecoderInferOutput,
    ActionDecoderTrainOutput,
    PolicyInferOutput,
    PolicyTrainBatch,
    PolicyTrainOutput,
)


class TemplateActionDecoder(ActionDecoder):
    def __init__(self, config: ExtensionActionDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.action_horizon * config.action_dim,
        )

    def forward_train(
        self,
        policy_output: PolicyTrainOutput,
        batch: PolicyTrainBatch,
    ) -> ActionDecoderTrainOutput:
        action_pred = self._decode(policy_output.policy_features)
        if action_pred.shape != batch.actions.shape:
            raise ValueError(
                "Template decoder target shape does not match its configured "
                f"output: predicted {tuple(action_pred.shape)}, target "
                f"{tuple(batch.actions.shape)}."
            )
        squared_error = (action_pred - batch.actions).square()
        loss = self._masked_mean(squared_error, batch.action_mask)
        return ActionDecoderTrainOutput(
            action_pred=action_pred,
            loss=loss,
            metrics={"template_action_mse": loss.detach()},
        )

    def forward_infer(
        self,
        policy_output: PolicyInferOutput,
        previous_state: Any | None = None,
    ) -> ActionDecoderInferOutput:
        return ActionDecoderInferOutput(
            action_pred=self._decode(policy_output.policy_features),
            next_state=previous_state,
        )

    def _decode(self, policy_features: torch.Tensor) -> torch.Tensor:
        pooled = policy_features.mean(dim=1)
        return self.output_projection(pooled).reshape(
            pooled.shape[0],
            self.config.action_horizon,
            self.config.action_dim,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean()
        resolved_mask = mask.to(device=values.device, dtype=values.dtype)
        while resolved_mask.ndim < values.ndim:
            resolved_mask = resolved_mask.unsqueeze(-1)
        resolved_mask = resolved_mask.expand_as(values)
        return (values * resolved_mask).sum() / resolved_mask.sum().clamp_min(1.0)
