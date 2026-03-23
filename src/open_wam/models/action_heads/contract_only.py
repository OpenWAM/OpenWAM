from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.models.video_backbone import BackboneOutput

from .base import (
    ActionHead,
    ActionHeadInferContext,
    ActionHeadInferOutput,
    ActionHeadInferState,
    ActionHeadTrainingBatch,
    ActionHeadTrainOutput,
)


@dataclass(frozen=True)
class ContractOnlyActionHeadConfig:
    """Shape-correct phase-2 head used to validate interfaces before variants exist."""

    action_dim: int
    action_horizon: int
    state_dim: int
    hidden_size: int


class ContractOnlyActionHead(ActionHead):
    """Minimal head that validates the common train/infer contracts.

    This is intentionally simple. It does not encode any specific research
    hypothesis yet. Its purpose is to guarantee that future variants inherit a
    working interface rather than inventing their own orchestration paths.
    """

    def __init__(self, config: ContractOnlyActionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.state_proj = nn.Linear(config.state_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.action_dim)

    def prepare_train_inputs(
        self,
        backbone_output: BackboneOutput,
        batch: ActionHeadTrainingBatch,
    ) -> dict[str, Any]:
        if batch.actions.ndim != 3:
            raise ValueError(
                "Expected actions with shape [B, H_action, D_action], "
                f"got {tuple(batch.actions.shape)}"
            )
        if batch.actions.shape[1] != self.config.action_horizon:
            raise ValueError(
                f"Expected action horizon {self.config.action_horizon}, "
                f"got {batch.actions.shape[1]}"
            )
        if batch.actions.shape[2] != self.config.action_dim:
            raise ValueError(
                f"Expected action dim {self.config.action_dim}, "
                f"got {batch.actions.shape[2]}"
            )
        if batch.state is not None and batch.state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"Expected state dim {self.config.state_dim}, got {batch.state.shape[-1]}"
            )

        pooled_video = backbone_output.video_tokens.mean(dim=1)
        prepared = {
            "pooled_video": pooled_video,
            "actions": batch.actions,
            "action_mask": batch.action_mask,
            "state": batch.state,
        }
        return prepared

    def forward_train(
        self,
        backbone_output: BackboneOutput,
        prepared_inputs: dict[str, Any],
    ) -> ActionHeadTrainOutput:
        pooled_video = prepared_inputs["pooled_video"]
        target_actions = prepared_inputs["actions"]
        action_mask = prepared_inputs["action_mask"]
        state = prepared_inputs["state"]

        base_hidden = pooled_video[:, None, :].expand(-1, self.config.action_horizon, -1)
        if state is not None:
            # State is fused at the head boundary, not in the protected backbone.
            state_summary = state.mean(dim=1)
            state_hidden = self.state_proj(state_summary)[:, None, :]
            base_hidden = base_hidden + state_hidden

        action_pred = self.output_proj(base_hidden)
        per_token_loss = F.mse_loss(action_pred, target_actions, reduction="none")
        if action_mask is not None:
            per_token_loss = per_token_loss * action_mask.float()
            denom = action_mask.float().sum().clamp_min(1.0)
        else:
            denom = torch.tensor(float(per_token_loss.numel()), device=per_token_loss.device)
        loss = per_token_loss.sum() / denom

        return ActionHeadTrainOutput(
            action_pred=action_pred,
            loss=loss,
            metrics={"action_mse": loss.detach()},
            aux={"interface_head": "contract_only"},
        )

    def prepare_infer_state(
        self,
        backbone_output: BackboneOutput,
        context: ActionHeadInferContext,
        previous_state: ActionHeadInferState | None = None,
    ) -> ActionHeadInferState:
        if previous_state is None:
            return ActionHeadInferState(step_index=0, cache={})
        return previous_state

    def forward_infer_step(
        self,
        backbone_output: BackboneOutput,
        context: ActionHeadInferContext,
        infer_state: ActionHeadInferState,
    ) -> ActionHeadInferOutput:
        pooled_video = backbone_output.video_tokens.mean(dim=1)
        hidden = pooled_video[:, None, :].expand(-1, self.config.action_horizon, -1)
        if context.state is not None:
            state_summary = context.state.mean(dim=1)
            hidden = hidden + self.state_proj(state_summary)[:, None, :]
        action_pred = self.output_proj(hidden)
        next_state = ActionHeadInferState(step_index=infer_state.step_index + 1, cache=dict(infer_state.cache))
        return ActionHeadInferOutput(
            action_pred=action_pred,
            next_state=next_state,
            aux={"interface_head": "contract_only"},
        )

