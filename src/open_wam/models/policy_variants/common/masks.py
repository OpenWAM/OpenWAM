from __future__ import annotations

import torch


def full_attention_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Return a fully allowed attention mask with shape `[B, seq, seq]`."""

    return torch.ones(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
