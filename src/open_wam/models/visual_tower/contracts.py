from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.common import PreparedAttentionProfile
from open_wam.models.video_backbone.contracts import (
    CacheState,
    CacheUpdateMetadata,
    ChunkMetadata,
    ConditioningState,
    TokenGridMetadata,
)


@dataclass(frozen=True)
class VisualReadoutRequest:
    """Opt-in intermediate readout capture requested from the visual core."""

    capture_layer_indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VisualRuntimeStateSnapshot:
    """Copied frontend and named-backbone state for speculative execution."""

    frontend_state: object | None = None
    runtime_cache_name: str | None = None
    runtime_cache_existed: bool = False
    runtime_cache_state: CacheState | None = None


@dataclass
class VisualIntermediateReadout:
    """One captured intermediate visual-core layer output."""

    layer_index: int
    tokens: torch.Tensor
    token_layout: Any | None = None
    aux: dict[str, Any] = field(default_factory=dict)


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
    input_source: str
    token_grid: TokenGridMetadata
    chunk: ChunkMetadata
    conditioning: ConditioningState


@dataclass
class VisualSequenceMetadata:
    """Optional cache metadata for generic packed visual-core calls."""

    metadata: dict[str, Any] = field(default_factory=dict)


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
    attention_profile: PreparedAttentionProfile | None = None
    cache_state: CacheState | None = None
    cache_update_metadata: CacheUpdateMetadata | None = None
    conditioning: ConditioningState | None = None
    readout_request: VisualReadoutRequest | None = None
    sequence_metadata: VisualSequenceMetadata | None = None


@dataclass
class VisualCoreOutput:
    """Outputs returned by the shared visual core."""

    tokens: torch.Tensor
    token_layout: Any | None
    cache_state: CacheState
    intermediate_readouts: tuple[VisualIntermediateReadout, ...] = field(default_factory=tuple)
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
