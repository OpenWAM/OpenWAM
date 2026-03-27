from __future__ import annotations

from dataclasses import dataclass, field
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
class AttentionCacheEntry:
    """One cache slot owned by the backbone runtime.

    The tensors are optional because the first cache-aware rewrite slice only
    establishes the contract. Later stages will populate these with per-layer
    KV / cross-attention tensors.
    """

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheUpdateMetadata:
    """Runtime instructions for one cache-aware forward pass."""

    current_start_frame: int = 0
    update_kv_cache: bool = False
    update_cross_attention_cache: bool = False
    cfg_mode: str = "none"
    max_cached_frames: int | None = None
    sink_frames: int = 0
    local_attn_window: int | None = None
    cache_branch: str = "default"


@dataclass
class CacheBranchState:
    """One named runtime cache branch.

    This supports CFG-style runtimes that keep distinct conditioned and
    unconditioned cache pools while still sharing one top-level CacheState
    contract.
    """

    backend_name: str = "merged_prefix"
    backend_payload: Any | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    self_attention_kv: tuple["AttentionCacheEntry", ...] = field(default_factory=tuple)
    cross_attention_kv: tuple["AttentionCacheEntry", ...] = field(default_factory=tuple)


@dataclass
class CacheState:
    """Backbone-owned cache contract exposed to downstream heads.

    Heads may read cache metadata, but the cache semantics remain backbone-owned.
    """

    supported: bool
    current_start_frame: int
    cached_frames: int
    chunk_size: int
    capability: str = "none"
    backend_name: str = "merged_prefix"
    backend_payload: Any | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    self_attention_kv: tuple[AttentionCacheEntry, ...] = field(default_factory=tuple)
    cross_attention_kv: tuple[AttentionCacheEntry, ...] = field(default_factory=tuple)
    update_metadata: CacheUpdateMetadata = field(default_factory=CacheUpdateMetadata)
    branch_states: dict[str, CacheBranchState] = field(default_factory=dict)


def resolve_cache_branch_state(
    cache_state: CacheState | None,
    branch_name: str,
) -> CacheBranchState:
    """Return one named branch from a cache state.

    The top-level cache tensors remain the compatibility/default branch.
    """

    if cache_state is None:
        return CacheBranchState()
    if branch_name != "default" and branch_name in cache_state.branch_states:
        return cache_state.branch_states[branch_name]
    return CacheBranchState(
        backend_name=cache_state.backend_name,
        backend_payload=cache_state.backend_payload,
        payload=dict(cache_state.payload),
        self_attention_kv=cache_state.self_attention_kv,
        cross_attention_kv=cache_state.cross_attention_kv,
    )


def replace_cache_branch_state(
    cache_state: CacheState,
    *,
    branch_name: str,
    branch_state: CacheBranchState,
    mirror_to_top_level: bool = False,
) -> CacheState:
    """Write one named branch back into a cache state."""

    next_branch_states = dict(cache_state.branch_states)
    if branch_name != "default":
        next_branch_states[branch_name] = branch_state
    if branch_name == "default" or mirror_to_top_level:
        return CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=branch_state.backend_name,
            backend_payload=branch_state.backend_payload,
            payload=dict(branch_state.payload),
            self_attention_kv=branch_state.self_attention_kv,
            cross_attention_kv=branch_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
            branch_states=next_branch_states,
        )
    return CacheState(
        supported=cache_state.supported,
        current_start_frame=cache_state.current_start_frame,
        cached_frames=cache_state.cached_frames,
        chunk_size=cache_state.chunk_size,
        capability=cache_state.capability,
        backend_name=cache_state.backend_name,
        backend_payload=cache_state.backend_payload,
        payload=dict(cache_state.payload),
        self_attention_kv=cache_state.self_attention_kv,
        cross_attention_kv=cache_state.cross_attention_kv,
        update_metadata=cache_state.update_metadata,
        branch_states=next_branch_states,
    )


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
