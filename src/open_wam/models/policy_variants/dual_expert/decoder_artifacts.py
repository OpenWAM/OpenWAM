"""Dual-expert payloads handed across the policy/decoder boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT = "open_wam.dual_expert.decoder.v1"


@dataclass(frozen=True)
class DualExpertActionTrainArtifacts:
    flow_pred: torch.Tensor
    targets: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    denoised_actions: torch.Tensor
    action_mask: torch.Tensor | None


@dataclass(frozen=True)
class DualExpertVideoTrainArtifacts:
    flow_pred: torch.Tensor
    targets: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    future_loss_mask: torch.Tensor


@dataclass(frozen=True)
class DualExpertTrainArtifacts:
    action: DualExpertActionTrainArtifacts
    video: DualExpertVideoTrainArtifacts | None
    condition_mode: str
    runtime_mode: str
    history_frames: int
    video_cache_seq_len: int | None = None


@dataclass(frozen=True)
class DualExpertInferArtifacts:
    action_pred: torch.Tensor
    predicted_latents: torch.Tensor | None
    condition_mode: str
    runtime_mode: str


__all__ = [
    "DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT",
    "DualExpertActionTrainArtifacts",
    "DualExpertInferArtifacts",
    "DualExpertTrainArtifacts",
    "DualExpertVideoTrainArtifacts",
]
