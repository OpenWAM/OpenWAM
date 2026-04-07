from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MoTVideoLayerCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class MoTVideoCache:
    layers: tuple[MoTVideoLayerCache, ...]
    video_seq_len: int


@dataclass
class MoTRuntimeState:
    """Typed MoT rollout state stored inside `PolicyInferState.variant_state`."""

    action_device: str | None = None
    text_context: torch.Tensor | None = None
    video_cache: MoTVideoCache | None = None
    video_tokens_per_frame: int | None = None


@dataclass(frozen=True)
class MoTActionTrainArtifacts:
    flow_pred: torch.Tensor
    targets: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    denoised_actions: torch.Tensor
    action_mask: torch.Tensor | None


@dataclass(frozen=True)
class MoTVideoTrainArtifacts:
    flow_pred: torch.Tensor
    targets: torch.Tensor
    timesteps: torch.Tensor
    scheduler: Any
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    future_loss_mask: torch.Tensor


@dataclass(frozen=True)
class MoTTrainArtifacts:
    action: MoTActionTrainArtifacts
    video: MoTVideoTrainArtifacts | None
    condition_mode: str
    runtime_mode: str
    video_prefix_frames: int
    video_cache_seq_len: int | None = None


@dataclass(frozen=True)
class MoTInferArtifacts:
    action_pred: torch.Tensor
    predicted_latents: torch.Tensor | None
    condition_mode: str
    runtime_mode: str
