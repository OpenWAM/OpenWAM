from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Fallback flash-attention symbol used only when the real package is absent."""

    del args, kwargs
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )
    return out.transpose(1, 2)
