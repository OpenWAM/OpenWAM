from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from open_wam.data import (
    CanonicalVideoBatch,
    ConfiguredCanonicalVideoPreprocessor,
    RobotWinCanonicalVideoPreprocessor,
)
from open_wam.models.action_heads import (
    ActionHead,
    ActionHeadInferContext,
    ActionHeadInferOutput,
    ActionHeadInferState,
    ActionHeadTrainingBatch,
    ActionHeadTrainOutput,
)
from open_wam.models.video_backbone import (
    BackboneOutput,
    LingbotCompatibleVideoBackbone,
    LingbotCompatibleVideoBackboneConfig,
)


@dataclass
class UnifiedWAMTrainOutput:
    """Combined train-time outputs for the backbone + action-head pipeline."""

    backbone_output: BackboneOutput
    head_output: ActionHeadTrainOutput


@dataclass
class UnifiedWAMInferOutput:
    """Combined inference output for the backbone + action-head pipeline."""

    backbone_output: BackboneOutput
    head_output: ActionHeadInferOutput


class UnifiedWAMPipeline(nn.Module):
    """Shared train/infer orchestration outside backbone and action-head modules."""

    def __init__(
        self,
        action_head: ActionHead,
        backbone_config: LingbotCompatibleVideoBackboneConfig | None = None,
        preprocessor: ConfiguredCanonicalVideoPreprocessor | None = None,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor or RobotWinCanonicalVideoPreprocessor()
        self.backbone = LingbotCompatibleVideoBackbone(backbone_config)
        self.action_head = action_head

    def canonicalize(self, views: Mapping[str, torch.Tensor]) -> CanonicalVideoBatch:
        return self.preprocessor(views)

    def forward_backbone(self, views: Mapping[str, torch.Tensor]) -> BackboneOutput:
        canonical_batch = self.canonicalize(views)
        return self.backbone(canonical_batch.video)

    def forward_train(
        self,
        views: Mapping[str, torch.Tensor],
        batch: ActionHeadTrainingBatch,
    ) -> UnifiedWAMTrainOutput:
        backbone_output = self.forward_backbone(views)
        prepared_inputs = self.action_head.prepare_train_inputs(backbone_output, batch)
        head_output = self.action_head.forward_train(backbone_output, prepared_inputs)
        return UnifiedWAMTrainOutput(backbone_output=backbone_output, head_output=head_output)

    def forward_infer_step(
        self,
        views: Mapping[str, torch.Tensor],
        context: ActionHeadInferContext,
        infer_state: ActionHeadInferState | None = None,
    ) -> UnifiedWAMInferOutput:
        backbone_output = self.forward_backbone(views)
        resolved_state = self.action_head.prepare_infer_state(
            backbone_output=backbone_output,
            context=context,
            previous_state=infer_state,
        )
        head_output = self.action_head.forward_infer_step(
            backbone_output=backbone_output,
            context=context,
            infer_state=resolved_state,
        )
        return UnifiedWAMInferOutput(backbone_output=backbone_output, head_output=head_output)
