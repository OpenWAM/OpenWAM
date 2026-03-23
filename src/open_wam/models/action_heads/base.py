from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from open_wam.models.video_backbone import BackboneOutput


@dataclass
class ActionHeadTrainingBatch:
    """Structured action-head training inputs independent from head family.

    Attributes:
        actions:
            Target action tensor of shape [B, H_action, D_action].
        action_mask:
            Optional action mask aligned to `actions`.
        state:
            Optional state tensor of shape [B, H_state, D_state].
        extra:
            Any dataset-specific fields that remain outside the protected backbone.
    """

    actions: torch.Tensor
    action_mask: torch.Tensor | None = None
    state: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionHeadTrainOutput:
    """Common train-time result returned by every action head."""

    action_pred: torch.Tensor
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionHeadInferState:
    """Per-head inference state carried across rollout steps."""

    step_index: int = 0
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionHeadInferContext:
    """Inputs required for one action-head inference step."""

    state: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionHeadInferOutput:
    """Common inference result returned by every action head."""

    action_pred: torch.Tensor
    next_state: ActionHeadInferState
    aux: dict[str, Any] = field(default_factory=dict)


class ActionHead(nn.Module, ABC):
    """Common interface implemented by all action-head variants.

    The backbone remains protected. Head variants can only interact with the
    shared video path through `BackboneOutput` and the structured train/infer
    contracts defined in this module.
    """

    @abstractmethod
    def prepare_train_inputs(
        self,
        backbone_output: BackboneOutput,
        batch: ActionHeadTrainingBatch,
    ) -> dict[str, Any]:
        """Prepare head-specific train-time inputs from shared backbone outputs."""

    @abstractmethod
    def forward_train(
        self,
        backbone_output: BackboneOutput,
        prepared_inputs: dict[str, Any],
    ) -> ActionHeadTrainOutput:
        """Run one train-time forward pass for this action head."""

    @abstractmethod
    def prepare_infer_state(
        self,
        backbone_output: BackboneOutput,
        context: ActionHeadInferContext,
        previous_state: ActionHeadInferState | None = None,
    ) -> ActionHeadInferState:
        """Prepare per-head inference state for the current rollout step."""

    @abstractmethod
    def forward_infer_step(
        self,
        backbone_output: BackboneOutput,
        context: ActionHeadInferContext,
        infer_state: ActionHeadInferState,
    ) -> ActionHeadInferOutput:
        """Run one inference step for this action head."""

