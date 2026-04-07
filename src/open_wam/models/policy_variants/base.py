from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from open_wam.models.visual_tower import VisualReadoutRequest, VisualStageOutputs, VisualTower

from .contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)


class PolicyVariant(nn.Module, ABC):
    """Attachment-aware policy variant interface."""

    def initialize_for_training(self, visual_tower: VisualTower) -> None:
        """Optional pre-wrap initialization hook for distributed training."""

        del visual_tower

    def requested_visual_readout(self) -> VisualReadoutRequest | None:
        """Return an optional visual-readout capture request for the shared core."""

        return None

    @abstractmethod
    def attach_site(self) -> str:
        """Return the declared attachment site."""

    @abstractmethod
    def required_visual_stages(self) -> tuple[str, ...]:
        """Return the visual stages the pipeline must prepare eagerly."""

    def requested_visual_readout(self) -> VisualReadoutRequest | None:
        """Optionally request one shared visual-core readout capture."""

        return None

    @abstractmethod
    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        """Prepare train-time variant inputs."""

    @abstractmethod
    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        """Run the train-time policy forward pass."""

    @abstractmethod
    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        """Prepare inference state for the current rollout step."""

    @abstractmethod
    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        """Run one inference step."""
