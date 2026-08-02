"""Typed cache-backend payloads and backend-selection contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "CacheBackendSpec",
    "MergedPrefixCachePayload",
    "SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS",
    "SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION",
    "SlotPoolCachePayload",
    "SlotPoolLayerState",
    "cache_backend_uses_slot_pool",
    "resolve_cache_backend_spec",
]


SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS = (
    "allow_video_query_to_action_prefix_tail_tokens"
)
SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION = (
    "defer_eviction_until_after_write_attention"
)


@dataclass(frozen=True)
class CacheBackendSpec:
    """Declarative description of a reusable cache backend."""

    name: str
    family: str
    retention_style: str


@dataclass
class MergedPrefixCachePayload:
    """Generic cache payload used by the existing merged-prefix runtime."""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotPoolLayerState:
    """One per-layer slot pool matching LingBot-style self-attention cache layout."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    slot_ids: torch.Tensor | None = None
    stream_ids: torch.Tensor | None = None
    slot_mask: torch.Tensor | None = None
    prediction_mask: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotPoolCachePayload:
    """Backend payload for a LingBot-style slot-pooled cache."""

    layer_states: tuple[SlotPoolLayerState, ...]
    total_tokens: int | None = None
    num_heads: int | None = None
    head_dim: int | None = None
    batch_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_CACHE_BACKEND_SPECS: dict[str, CacheBackendSpec] = {
    "merged_prefix": CacheBackendSpec(
        name="merged_prefix",
        family="generic",
        retention_style="prefix_merge",
    ),
    "slot_pool_exact": CacheBackendSpec(
        name="slot_pool_exact",
        family="exact_runtime",
        retention_style="slot_pool",
    ),
}

_CACHE_BACKEND_ALIASES: dict[str, str] = {
    "merged_prefix": "merged_prefix",
    "slot_pool_exact": "slot_pool_exact",
    "lingbot_slot_pool": "slot_pool_exact",
}


def resolve_cache_backend_spec(name: str) -> CacheBackendSpec:
    try:
        canonical_name = _CACHE_BACKEND_ALIASES[name]
        return _CACHE_BACKEND_SPECS[canonical_name]
    except KeyError as exc:  # pragma: no cover - defensive config guard
        raise ValueError(
            f"Unsupported cache backend {name!r}. Expected one of {tuple(_CACHE_BACKEND_ALIASES)}."
        ) from exc


def cache_backend_uses_slot_pool(name: str | None) -> bool:
    if name is None:
        return False
    return resolve_cache_backend_spec(name).retention_style == "slot_pool"
