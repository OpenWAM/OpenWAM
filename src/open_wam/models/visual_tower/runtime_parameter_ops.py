"""FSDP-safe parameter materialization for explicit transformer execution."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from torch import nn


def materialize_runtime_parameter(
    parameter: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a dense tensor for helper paths that bypass FSDP pre-forward hooks."""

    if hasattr(parameter, "full_tensor"):
        return parameter.full_tensor().to(device=device, dtype=dtype)
    return parameter.to(device=device, dtype=dtype)


def linear_with_materialized_params(
    linear: nn.Linear,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply a linear layer after materializing any sharded parameters."""

    weight = materialize_runtime_parameter(
        linear.weight,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    bias = None
    if linear.bias is not None:
        bias = materialize_runtime_parameter(
            linear.bias,
            device=inputs.device,
            dtype=inputs.dtype,
        )
    return F.linear(inputs, weight, bias)


def rms_norm_with_materialized_weight(
    norm: nn.RMSNorm,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply RMS normalization with a materialized affine weight."""

    weight = None
    if norm.weight is not None:
        weight = materialize_runtime_parameter(
            norm.weight,
            device=inputs.device,
            dtype=inputs.dtype,
        )
    return F.rms_norm(
        inputs,
        list(norm.normalized_shape),
        weight=weight,
        eps=norm.eps,
    )


def layer_norm_with_materialized_params(
    norm: nn.LayerNorm,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply layer normalization with materialized affine parameters."""

    weight = None
    bias = None
    if getattr(norm, "weight", None) is not None:
        weight = materialize_runtime_parameter(
            norm.weight,
            device=inputs.device,
            dtype=inputs.dtype,
        )
    if getattr(norm, "bias", None) is not None:
        bias = materialize_runtime_parameter(
            norm.bias,
            device=inputs.device,
            dtype=inputs.dtype,
        )
    return F.layer_norm(
        inputs,
        list(norm.normalized_shape),
        weight=weight,
        bias=bias,
        eps=norm.eps,
    )


def feed_forward_with_materialized_params(
    ffn: FeedForward,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply the supported Diffusers feed-forward layout with dense parameters."""

    if len(ffn.net) != 3:
        raise ValueError(
            f"Unsupported FeedForward layout for materialized helper: {ffn.net!r}"
        )
    act = ffn.net[0]
    dropout = ffn.net[1]
    proj_out = ffn.net[2]
    if not hasattr(act, "proj"):
        raise ValueError(
            f"Unsupported FeedForward activation module for materialized helper: {act!r}"
        )
    hidden = linear_with_materialized_params(act.proj, inputs)
    hidden = F.gelu(hidden, approximate="tanh")
    hidden = dropout(hidden)
    return linear_with_materialized_params(proj_out, hidden)


__all__ = [
    "feed_forward_with_materialized_params",
    "layer_norm_with_materialized_params",
    "linear_with_materialized_params",
    "materialize_runtime_parameter",
    "rms_norm_with_materialized_weight",
]
