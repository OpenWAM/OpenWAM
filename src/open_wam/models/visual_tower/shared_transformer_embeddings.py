"""Timestep and positional embeddings for the shared transformer stack."""

from __future__ import annotations

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn


class SharedTransformerTimeEmbedding(nn.Module):
    """Wan-style timestep conditioner used by the shared transformer core."""

    def __init__(self, hidden_size: int, freq_dim: int) -> None:
        super().__init__()
        self.timesteps_proj = Timesteps(
            num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim, time_embed_dim=hidden_size
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(hidden_size, hidden_size * 6)

    def forward(
        self, timestep_values: torch.Tensor, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = timestep_values.shape
        flat = timestep_values.reshape(-1)
        projected = self.timesteps_proj(flat)
        projected = projected.to(self.time_embedder.linear_1.weight.dtype)
        temb = (
            self.time_embedder(projected)
            .to(dtype=dtype)
            .reshape(batch_size, seq_len, -1)
        )
        timestep_proj = self.time_proj(self.act_fn(temb)).reshape(
            batch_size, seq_len, 6, -1
        )
        return temb, timestep_proj


class SharedTransformerRotaryPositionalEmbedding(nn.Module):
    """Wan-style rotary embedding over frame, height, and width axes."""

    def __init__(self, attention_head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.theta = theta
        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3
        self.register_buffer(
            "f_freqs_base", self._make_freqs_base(self.f_dim), persistent=False
        )
        self.register_buffer(
            "h_freqs_base", self._make_freqs_base(self.h_dim), persistent=False
        )
        self.register_buffer(
            "w_freqs_base", self._make_freqs_base(self.w_dim), persistent=False
        )

    def _make_freqs_base(self, dim: int) -> torch.Tensor:
        half_dim = max(1, dim // 2)
        return 1.0 / (
            self.theta ** (torch.arange(0, dim, 2)[:half_dim].double() / max(dim, 1))
        )

    def forward(self, grid_ids: torch.Tensor) -> torch.Tensor:
        if grid_ids.ndim == 2:
            grid_ids = grid_ids.unsqueeze(0)
        f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(
            grid_ids.device
        )
        h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(
            grid_ids.device
        )
        w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(
            grid_ids.device
        )
        freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
        return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply complex rotary frequencies to per-head query or key features."""

    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    if freqs.ndim == 3:
        freqs = freqs[:, :, None, :]
    x_out = torch.view_as_real(x_complex * freqs).flatten(3)
    return x_out.to(x.dtype)


__all__ = [
    "SharedTransformerRotaryPositionalEmbedding",
    "SharedTransformerTimeEmbedding",
    "apply_rotary_emb",
]
