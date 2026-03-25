from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.video_backbone.contracts import CacheState, ChunkMetadata, ConditioningState, TokenGridMetadata


@dataclass(frozen=True)
class DecodedFeatureLayout:
    """Layout metadata for decoded visual features."""

    kind: str
    num_frames: int
    tokens_per_frame: int
    hidden_size: int


@dataclass
class VisualFrontendOutput:
    """Outputs produced by the fixed visual frontend."""

    canonical_video: torch.Tensor
    video_latents: torch.Tensor
    video_tokens: torch.Tensor
    token_grid: TokenGridMetadata
    chunk: ChunkMetadata
    conditioning: ConditioningState


@dataclass
class VisualCoreInput:
    """Generic packed-sequence input accepted by the shared visual core."""

    tokens: torch.Tensor
    token_layout: Any | None = None
    position_context: torch.Tensor | None = None
    timestep_context: torch.Tensor | None = None
    grid_ids: torch.Tensor | None = None
    timestep_values: torch.Tensor | None = None
    stream_ids: torch.Tensor | None = None
    text_context: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    cache_state: CacheState | None = None
    conditioning: ConditioningState | None = None


@dataclass
class VisualCoreOutput:
    """Outputs returned by the shared visual core."""

    tokens: torch.Tensor
    token_layout: Any | None
    cache_state: CacheState
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualDecodeOutput:
    """Decoded visual features exposed to post-decoded policies."""

    decoded_features: torch.Tensor
    feature_layout: DecodedFeatureLayout
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualStageOutputs:
    """Stageful outputs computed for one forward pass."""

    frontend: VisualFrontendOutput
    core: VisualCoreOutput | None = None
    decode: VisualDecodeOutput | None = None
