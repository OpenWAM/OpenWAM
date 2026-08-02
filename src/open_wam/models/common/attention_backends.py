"""Backend selection and execution for prepared attention profiles."""

from __future__ import annotations

from typing import Any

import torch

from open_wam.models.common.attention_contracts import PreparedAttentionProfile

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )
except (
    ImportError
):  # pragma: no cover - older torch builds may not expose FlexAttention
    BlockMask = Any  # type: ignore[misc,assignment]
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]


_COMPILED_FLEX_ATTENTION = None


_COMPILED_CREATE_BLOCK_MASK = None


def _resolve_compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if flex_attention is None:
        return None
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTENTION


def _resolve_compiled_create_block_mask():
    global _COMPILED_CREATE_BLOCK_MASK
    if create_block_mask is None:
        return None
    if _COMPILED_CREATE_BLOCK_MASK is None:
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)
    return _COMPILED_CREATE_BLOCK_MASK


def resolve_attention_profile_backend(
    profile: PreparedAttentionProfile | None,
    *,
    device: torch.device,
    prefer_flex: bool = False,
    is_cross_attention: bool = False,
) -> str:
    if profile is None:
        return "none"
    if prefer_flex and device.type == "cuda":
        if is_cross_attention and profile.cross_attention_block_mask is not None:
            return "lingbot_flex"
        if not is_cross_attention and profile.self_attention_block_mask is not None:
            return "lingbot_flex"
    return (
        "sdpa"
        if (
            (is_cross_attention and profile.cross_attention_mask is not None)
            or (not is_cross_attention and profile.self_attention_mask is not None)
        )
        else "none"
    )


def select_attention_profile_mask(
    profile: PreparedAttentionProfile | None,
    *,
    device: torch.device,
    prefer_flex: bool = False,
    is_cross_attention: bool = False,
) -> tuple[torch.Tensor | None, BlockMask | None]:
    backend = resolve_attention_profile_backend(
        profile,
        device=device,
        prefer_flex=prefer_flex,
        is_cross_attention=is_cross_attention,
    )
    if profile is None or backend == "none":
        return None, None
    if backend == "lingbot_flex":
        return (
            None,
            profile.cross_attention_block_mask
            if is_cross_attention
            else profile.self_attention_block_mask,
        )
    return (
        (
            profile.cross_attention_mask
            if is_cross_attention
            else profile.self_attention_mask
        ),
        None,
    )


def apply_attention_backend(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    block_mask: BlockMask | None = None,
    kernel_options: dict[str, Any] | None = None,
) -> torch.Tensor:
    if attention_mask is not None:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )
    if block_mask is not None:
        if flex_attention is None:
            raise RuntimeError("FlexAttention is not available in this torch build.")
        compiled_flex_attention = _resolve_compiled_flex_attention()
        if compiled_flex_attention is not None:
            return compiled_flex_attention(
                query, key, value, block_mask=block_mask, kernel_options=kernel_options
            )
        return flex_attention(
            query, key, value, block_mask=block_mask, kernel_options=kernel_options
        )
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)


__all__ = [
    "apply_attention_backend",
    "resolve_attention_profile_backend",
    "select_attention_profile_mask",
]
