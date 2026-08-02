"""Historical cache-backend import surface.

Maintained code imports the role-owned modules directly. This facade preserves
existing direct imports, wildcard imports, and pickle lookup paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.common.attention_contracts import PreparedAttentionProfile
from open_wam.models.video_backbone.contracts import AttentionCacheEntry

from .cache_layout_policy import (
    merge_attention_cache_entries,
    packed_slot_pool_query_sequence_ids,
    prepend_cached_prefix_mask,
    prepare_sdpa_mask,
    resolve_slot_pool_prefix_visibility,
    retained_slot_pool_indices_for_current_write,
)
from .cache_backend_contracts import (
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    CacheBackendSpec,
    MergedPrefixCachePayload,
    SlotPoolCachePayload,
    SlotPoolLayerState,
    _CACHE_BACKEND_ALIASES,
    _CACHE_BACKEND_SPECS,
    cache_backend_uses_slot_pool,
    resolve_cache_backend_spec,
)
from .cache_backend_lifecycle import (
    allocate_slot_pool_slots,
    clear_cache_backend_payload,
    init_cache_backend_payload,
    materialize_cache_backend_entries,
    materialize_slot_pool_layer_entry,
    next_slot_pool_cache_id,
    restore_slot_pool_slots,
    update_slot_pool_layer_state,
)

(
    Any,
    AttentionCacheEntry,
    CacheBackendSpec,
    MergedPrefixCachePayload,
    PreparedAttentionProfile,
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    SlotPoolCachePayload,
    SlotPoolLayerState,
    _CACHE_BACKEND_ALIASES,
    _CACHE_BACKEND_SPECS,
    allocate_slot_pool_slots,
    cache_backend_uses_slot_pool,
    clear_cache_backend_payload,
    dataclass,
    field,
    init_cache_backend_payload,
    materialize_cache_backend_entries,
    materialize_slot_pool_layer_entry,
    math,
    merge_attention_cache_entries,
    next_slot_pool_cache_id,
    packed_slot_pool_query_sequence_ids,
    prepare_sdpa_mask,
    prepend_cached_prefix_mask,
    resolve_cache_backend_spec,
    resolve_slot_pool_prefix_visibility,
    restore_slot_pool_slots,
    retained_slot_pool_indices_for_current_write,
    torch,
    update_slot_pool_layer_state,
)
