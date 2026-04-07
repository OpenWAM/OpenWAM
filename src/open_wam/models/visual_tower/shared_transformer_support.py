from __future__ import annotations

import torch

from .replica_core import (
    SharedTransformerAttention,
    SharedTransformerRotaryPositionalEmbedding,
    SharedTransformerTimeEmbedding,
    _apply_rotary_emb,
    _feed_forward_with_materialized_params,
    _layer_norm_with_materialized_params,
    _linear_with_materialized_params,
    _materialize_runtime_parameter,
    _rms_norm_with_materialized_weight,
    _select_chunk_slices,
)


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Public wrapper for the shared transformer rotary helper."""

    return _apply_rotary_emb(x, freqs)


def select_chunk_slices(tensor: torch.Tensor, count: int) -> tuple[torch.Tensor, ...]:
    """Public wrapper for chunk-slice selection used by shared transformer blocks."""

    return _select_chunk_slices(tensor, count)


def materialize_runtime_parameter(parameter: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Materialize an FSDP-safe parameter shard onto the runtime device."""

    return _materialize_runtime_parameter(parameter, device=device, dtype=dtype)


def linear_with_materialized_params(module, inputs: torch.Tensor) -> torch.Tensor:
    """Apply a linear module using fully materialized runtime parameters."""

    return _linear_with_materialized_params(module, inputs)


def rms_norm_with_materialized_weight(module, inputs: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm using materialized weights."""

    return _rms_norm_with_materialized_weight(module, inputs)


def layer_norm_with_materialized_params(module, inputs: torch.Tensor) -> torch.Tensor:
    """Apply LayerNorm using materialized runtime parameters."""

    return _layer_norm_with_materialized_params(module, inputs)


def feed_forward_with_materialized_params(module, inputs: torch.Tensor) -> torch.Tensor:
    """Apply a feed-forward block using materialized runtime parameters."""

    return _feed_forward_with_materialized_params(module, inputs)


__all__ = [
    "SharedTransformerAttention",
    "SharedTransformerRotaryPositionalEmbedding",
    "SharedTransformerTimeEmbedding",
    "apply_rotary_emb",
    "feed_forward_with_materialized_params",
    "layer_norm_with_materialized_params",
    "linear_with_materialized_params",
    "materialize_runtime_parameter",
    "rms_norm_with_materialized_weight",
    "select_chunk_slices",
]
