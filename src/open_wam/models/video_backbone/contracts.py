from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TokenGridMetadata:
    """Metadata describing the backbone token geometry."""

    num_frames: int
    latent_height: int
    latent_width: int
    patch_size: tuple[int, int, int]
    patches_per_frame_h: int
    patches_per_frame_w: int
    tokens_per_frame: int
    sequence_length: int


@dataclass(frozen=True)
class ChunkMetadata:
    """Temporal metadata shared between the backbone and future action heads."""

    chunk_start_frame: int
    chunk_num_frames: int
    frame_stride: int
    chunk_type: str


@dataclass
class CacheState:
    """Backbone-owned cache contract exposed to downstream heads.

    Heads may read cache metadata, but the cache semantics remain backbone-owned.
    """

    supported: bool
    current_start_frame: int
    cached_frames: int
    chunk_size: int
    payload: dict[str, Any]


@dataclass
class ConditioningState:
    """Backbone conditioning contract for future text/observation context."""

    supported: bool
    text_context: torch.Tensor | None = None
    negative_text_context: torch.Tensor | None = None
    first_frame_context: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class BackboneOutput:
    """Common backbone outputs shared by all future action heads.

    Attributes:
        canonical_video:
            Input RGB video after multi-view canonicalization, shape [B, 3, T, H, W].
        video_latents:
            Stage-1 latent tensor that mirrors LingBot geometry, shape [B, C_lat, T, H_lat, W_lat].
        video_tokens:
            Flattened video token sequence consumed by future shared transformer blocks,
            shape [B, seq_len, hidden_size].
        token_grid:
            Geometry metadata for unpacking tokens back into latent space.
        chunk:
            Temporal chunk metadata that all head variants use for alignment.
        cache_state:
            Backbone-owned cache contract for future chunked inference.
        conditioning:
            Conditioning contract for future language or observation context.
    """

    canonical_video: torch.Tensor
    video_latents: torch.Tensor
    video_tokens: torch.Tensor
    token_grid: TokenGridMetadata
    chunk: ChunkMetadata
    cache_state: CacheState
    conditioning: ConditioningState
