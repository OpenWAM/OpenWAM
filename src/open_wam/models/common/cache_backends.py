from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from open_wam.models.video_backbone.contracts import AttentionCacheEntry


SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS = "allow_video_query_to_action_prefix_tail_tokens"
SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION = "defer_eviction_until_after_write_attention"


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


def init_cache_backend_payload(
    backend_name: str,
    *,
    num_layers: int = 0,
    total_tokens: int | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    metadata: dict[str, Any] | None = None,
) -> MergedPrefixCachePayload | SlotPoolCachePayload:
    backend_spec = resolve_cache_backend_spec(backend_name)
    if backend_spec.retention_style == "prefix_merge":
        return MergedPrefixCachePayload(metadata=dict(metadata or {}))

    layer_states: list[SlotPoolLayerState] = []
    for _ in range(max(0, int(num_layers))):
        if (
            total_tokens is not None
            and num_heads is not None
            and head_dim is not None
            and batch_size is not None
            and device is not None
            and dtype is not None
        ):
            key = torch.empty(batch_size, total_tokens, num_heads, head_dim, device=device, dtype=dtype)
            value = torch.empty(batch_size, total_tokens, num_heads, head_dim, device=device, dtype=dtype)
            slot_ids = torch.full((total_tokens,), -1, device=device, dtype=torch.long)
            stream_ids = torch.full((total_tokens,), -1, device=device, dtype=torch.long)
            slot_mask = torch.zeros((total_tokens,), dtype=torch.bool, device=device)
            prediction_mask = torch.zeros((total_tokens,), dtype=torch.bool, device=device)
        else:
            key = None
            value = None
            slot_ids = None
            stream_ids = None
            slot_mask = None
            prediction_mask = None
        layer_states.append(
            SlotPoolLayerState(
                key=key,
                value=value,
                slot_ids=slot_ids,
                stream_ids=stream_ids,
                slot_mask=slot_mask,
                prediction_mask=prediction_mask,
                metadata=dict(metadata or {}),
            )
        )

    return SlotPoolCachePayload(
        layer_states=tuple(layer_states),
        total_tokens=total_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        metadata=dict(metadata or {}),
    )


def clear_cache_backend_payload(
    payload: MergedPrefixCachePayload | SlotPoolCachePayload | None,
    *,
    clear_predictions_only: bool = False,
) -> MergedPrefixCachePayload | SlotPoolCachePayload | None:
    if payload is None:
        return None
    if isinstance(payload, MergedPrefixCachePayload):
        return MergedPrefixCachePayload(metadata=dict(payload.metadata))
    next_layers: list[SlotPoolLayerState] = []
    for layer_state in payload.layer_states:
        if clear_predictions_only:
            next_prediction_mask = layer_state.prediction_mask
            next_slot_mask = layer_state.slot_mask
            if next_slot_mask is not None and layer_state.prediction_mask is not None:
                next_slot_mask = next_slot_mask.clone()
                next_slot_mask[layer_state.prediction_mask] = False
            next_layer = SlotPoolLayerState(
                key=layer_state.key,
                value=layer_state.value,
                slot_ids=layer_state.slot_ids,
                stream_ids=layer_state.stream_ids,
                slot_mask=next_slot_mask,
                prediction_mask=next_prediction_mask,
                metadata=dict(layer_state.metadata),
            )
        else:
            next_layer = SlotPoolLayerState(metadata=dict(layer_state.metadata))
        next_layers.append(next_layer)
    return SlotPoolCachePayload(
        layer_states=tuple(next_layers),
        total_tokens=payload.total_tokens,
        num_heads=payload.num_heads,
        head_dim=payload.head_dim,
        batch_size=payload.batch_size,
        metadata=dict(payload.metadata),
    )


def allocate_slot_pool_slots(layer_state: SlotPoolLayerState, key_size: int) -> torch.Tensor:
    if layer_state.slot_mask is None or layer_state.slot_ids is None:
        raise ValueError("Slot-pool backend requires initialized `slot_mask` and `slot_ids` tensors.")
    mask = layer_state.slot_mask
    ids = layer_state.slot_ids
    free = (~mask).nonzero(as_tuple=False).squeeze(-1)

    if free.numel() < key_size:
        used = mask.nonzero(as_tuple=False).squeeze(-1)
        used_ids = ids[used]
        order = torch.argsort(used_ids, stable=True)
        need = key_size - free.numel()
        to_free = used[order[:need]]
        mask[to_free] = False
        ids[to_free] = -1
        if layer_state.prediction_mask is not None:
            layer_state.prediction_mask[to_free] = False
        if layer_state.stream_ids is not None:
            layer_state.stream_ids[to_free] = -1
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

    if free.numel() < key_size:  # pragma: no cover - defensive runtime guard
        raise RuntimeError("Slot-pool cache failed to allocate enough free slots.")
    return free[:key_size]


def next_slot_pool_cache_id(layer_state: SlotPoolLayerState) -> torch.Tensor:
    if layer_state.slot_ids is None or layer_state.slot_mask is None:
        raise ValueError("Slot-pool backend requires initialized `slot_ids` and `slot_mask` tensors.")
    if bool(layer_state.slot_mask.any().item()):
        return layer_state.slot_ids[layer_state.slot_mask].max() + 1
    return torch.tensor(0, device=layer_state.slot_ids.device, dtype=layer_state.slot_ids.dtype)


def update_slot_pool_layer_state(
    layer_state: SlotPoolLayerState,
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    is_pred: bool,
    stream_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Insert one layer's current KV tensors into the LingBot-style slot pool.

    Args:
        layer_state: mutable slot-pool state for one self-attention layer
        key: tensor shaped `[B, tokens, heads, dim]`
        value: tensor shaped `[B, tokens, heads, dim]`
        is_pred: whether these slots should be treated as predicted cache
        stream_ids: optional per-token stream ids, using 0 for video, 1 for
            action, and -1 for padded/non-semantic slots.

    Returns:
        The allocated slot indices, shaped `[tokens]`.
    """

    if layer_state.key is None or layer_state.value is None:
        raise ValueError("Slot-pool backend requires preallocated `key` and `value` tensors.")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"Expected slot-pool KV tensors with rank 4, got {tuple(key.shape)} / {tuple(value.shape)}")
    key_size = int(key.shape[1])
    if stream_ids is not None:
        if stream_ids.ndim == 2:
            if stream_ids.shape[0] != 1:
                raise ValueError(
                    "Slot-pool stream ids must be shared across batch or rank-1, "
                    f"got shape {tuple(stream_ids.shape)}."
                )
            stream_ids = stream_ids.squeeze(0)
        if stream_ids.ndim != 1 or int(stream_ids.shape[0]) != key_size:
            raise ValueError(
                "Slot-pool stream ids must have one value per KV token, "
                f"got shape {tuple(stream_ids.shape)} for key_size={key_size}."
            )
    slots = allocate_slot_pool_slots(layer_state, key_size)
    new_id = next_slot_pool_cache_id(layer_state)

    layer_state.key[:, slots] = key
    layer_state.value[:, slots] = value
    if layer_state.slot_mask is not None:
        layer_state.slot_mask[slots] = True
    if layer_state.slot_ids is not None:
        layer_state.slot_ids[slots] = new_id
    if layer_state.prediction_mask is not None:
        layer_state.prediction_mask[slots] = bool(is_pred)
    if layer_state.stream_ids is not None:
        if stream_ids is None:
            layer_state.stream_ids[slots] = -1
        else:
            layer_state.stream_ids[slots] = stream_ids.to(
                device=layer_state.stream_ids.device,
                dtype=layer_state.stream_ids.dtype,
            )
    return slots


def restore_slot_pool_slots(layer_state: SlotPoolLayerState, slots: torch.Tensor | None) -> None:
    if slots is None or slots.numel() == 0:
        return
    if layer_state.slot_mask is not None:
        layer_state.slot_mask[slots] = False
    if layer_state.stream_ids is not None:
        layer_state.stream_ids[slots] = -1


def materialize_slot_pool_layer_entry(layer_state: SlotPoolLayerState) -> AttentionCacheEntry:
    if (
        layer_state.key is None
        or layer_state.value is None
        or layer_state.slot_mask is None
        or not bool(layer_state.slot_mask.any().item())
    ):
        return AttentionCacheEntry(metadata=dict(layer_state.metadata))
    valid = layer_state.slot_mask.nonzero(as_tuple=False).squeeze(-1)
    if layer_state.slot_ids is not None and valid.numel() > 1:
        valid = valid[torch.argsort(layer_state.slot_ids[valid], stable=True)]
    key = layer_state.key[:, valid].transpose(1, 2).contiguous()
    value = layer_state.value[:, valid].transpose(1, 2).contiguous()
    metadata = dict(layer_state.metadata)
    metadata["cached_tokens"] = int(valid.numel())
    if layer_state.slot_ids is not None:
        metadata["slot_ids"] = layer_state.slot_ids[valid].clone()
    if layer_state.prediction_mask is not None:
        metadata["prediction_mask"] = layer_state.prediction_mask[valid].clone()
    if layer_state.stream_ids is not None:
        metadata["stream_ids"] = layer_state.stream_ids[valid].clone()
    return AttentionCacheEntry(key=key, value=value, metadata=metadata)


def materialize_cache_backend_entries(
    payload: MergedPrefixCachePayload | SlotPoolCachePayload | None,
) -> tuple[AttentionCacheEntry, ...]:
    if payload is None or isinstance(payload, MergedPrefixCachePayload):
        return tuple()
    return tuple(materialize_slot_pool_layer_entry(layer_state) for layer_state in payload.layer_states)
