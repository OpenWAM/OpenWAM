"""Video-flow payloads handed across the policy/decoder boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT = "open_wam.video_flow.decoder.v1"


@dataclass(frozen=True)
class VideoFlowTrainArtifacts:
    flow_pred: torch.Tensor
    targets: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    future_loss_mask: torch.Tensor


@dataclass(frozen=True)
class VideoFlowInferArtifacts:
    predicted_latents: torch.Tensor


__all__ = [
    "VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT",
    "VideoFlowInferArtifacts",
    "VideoFlowTrainArtifacts",
]
