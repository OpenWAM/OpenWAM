"""Typed payloads exchanged by policies and decoders, independent of implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_wam.configs.enums import DynamicsObjective
from open_wam.models.common.flow_schedule import FlowMatchScheduler


@dataclass
class ParallelTrainArtifacts:
    """Prepared parallel-stream inputs and flow schedulers."""

    input_dict: dict[str, Any]
    latent_scheduler: FlowMatchScheduler
    action_scheduler: FlowMatchScheduler
    dynamics_objective: DynamicsObjective | None = None


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
    program: str
    history_frames: int
    video_cache_seq_len: int | None = None


@dataclass(frozen=True)
class DualExpertInferArtifacts:
    action_pred: torch.Tensor
    predicted_latents: torch.Tensor | None
    condition_mode: str
    program: str


PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT = "open_wam.parallel_stream.decoder.v1"


@dataclass(frozen=True)
class ParallelDecoderTrainArtifacts:
    latent_pred: torch.Tensor
    runtime: ParallelTrainArtifacts
    loss_weights: dict[str, float]
    patch_size: tuple[int, int, int]


@dataclass(frozen=True)
class ParallelDecoderInferArtifacts:
    predicted_latents: torch.Tensor


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
