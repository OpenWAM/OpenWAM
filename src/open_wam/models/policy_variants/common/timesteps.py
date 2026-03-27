from __future__ import annotations

import torch

from .positions import sinusoidal_embedding


def build_scalar_timestep_embedding(values: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Embed one scalar timestep per batch element."""

    return sinusoidal_embedding(values.float(), hidden_size)


def expand_token_timestep_context(base_embedding: torch.Tensor, length: int) -> torch.Tensor:
    """Expand `[B, D]` timestep embeddings across `length` tokens."""

    return base_embedding[:, None, :].expand(-1, length, -1)


def build_token_timestep_context(values: torch.Tensor, hidden_size: int) -> torch.Tensor:
    """Embed one scalar timestep per token from `[B, L]` to `[B, L, D]`."""

    return sinusoidal_embedding(values.float(), hidden_size)
