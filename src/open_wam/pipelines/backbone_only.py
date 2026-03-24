from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from open_wam.data import CanonicalVideoBatch, ConfiguredCanonicalVideoPreprocessor, RobotWinCanonicalVideoPreprocessor
from open_wam.models.video_backbone import (
    BackboneOutput,
    LingbotCompatibleVideoBackbone,
    LingbotCompatibleVideoBackboneConfig,
)


class BackboneOnlyPipeline(nn.Module):
    """Stage-1 pipeline from raw multi-view video to common backbone outputs."""

    def __init__(
        self,
        backbone_config: LingbotCompatibleVideoBackboneConfig | None = None,
        preprocessor: ConfiguredCanonicalVideoPreprocessor | None = None,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor or RobotWinCanonicalVideoPreprocessor()
        self.backbone = LingbotCompatibleVideoBackbone(backbone_config)

    def canonicalize(self, views: Mapping[str, torch.Tensor]) -> CanonicalVideoBatch:
        return self.preprocessor(views)

    def forward(self, views: Mapping[str, torch.Tensor]) -> BackboneOutput:
        canonical_batch = self.canonicalize(views)
        return self.backbone(canonical_batch.video)
