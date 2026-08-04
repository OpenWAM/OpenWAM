"""Parallel-stream payloads handed across the policy/decoder boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .training_artifact_contracts import ParallelTrainArtifacts

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
    raw_chunk_action_pred: torch.Tensor | None = None


__all__ = [
    "PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT",
    "ParallelDecoderInferArtifacts",
    "ParallelDecoderTrainArtifacts",
]
