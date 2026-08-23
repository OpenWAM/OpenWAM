"""Parameter-free diagnostics for parallel exact-cache state."""

from __future__ import annotations

import torch

from open_wam.models.common.cache_backend_contracts import (
    cache_backend_uses_slot_pool,
)


def summarize_slot_pool_cache_state(
    transformer: torch.nn.Module,
    cache_name: str,
) -> dict[str, int] | None:
    """Summarize the first slot-pool layer for rollout diagnostics."""

    cache_state = transformer.get_runtime_cache_state(cache_name)
    if cache_state is None or not cache_backend_uses_slot_pool(
        cache_state.backend_name
    ):
        return None
    backend_payload = cache_state.backend_payload
    if backend_payload is None or not getattr(
        backend_payload,
        "layer_states",
        None,
    ):
        return None
    layer_state = backend_payload.layer_states[0]
    if layer_state.slot_mask is None:
        return None
    cached_tokens = int(layer_state.slot_mask.sum().item())
    prediction_tokens = (
        int(layer_state.prediction_mask[layer_state.slot_mask].sum().item())
        if layer_state.prediction_mask is not None
        else 0
    )
    return {
        "cached_tokens": cached_tokens,
        "prediction_tokens": prediction_tokens,
        "total_slots": int(layer_state.slot_mask.numel()),
    }


__all__ = ["summarize_slot_pool_cache_state"]
