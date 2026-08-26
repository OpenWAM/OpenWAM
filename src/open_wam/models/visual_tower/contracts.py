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

VisualComponent = torch.nn.Module | torch.nn.Parameter


@dataclass(frozen=True)
class VisualComponentTopology:
    """Semantic component ownership exposed to generic training controls."""

    shared_video_backbone: tuple[VisualComponent, ...] = ()
    shared_action_runtime: tuple[VisualComponent, ...] = ()
    shared_runtime_adapters: tuple[VisualComponent, ...] = ()


@dataclass(frozen=True)
class VisualRuntimeStateSnapshot:
    """Copied frontend and named-backbone state for speculative execution."""

    frontend_state: object | None = None
    runtime_cache_name: str | None = None
    runtime_cache_existed: bool = False
    runtime_cache_state: CacheState | None = None


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
    sequence_metadata: VisualSequenceMetadata | None = None


@dataclass
class VisualCoreOutput:
    """Outputs returned by the shared visual core."""

    tokens: torch.Tensor
    token_layout: Any | None
    cache_state: CacheState
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualStageOutputs:
    """Stageful outputs computed for one forward pass."""

    frontend: VisualFrontendOutput
    core: VisualCoreOutput | None = None
