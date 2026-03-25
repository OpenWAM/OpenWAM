from __future__ import annotations

import torch
from torch import nn

from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.video_backbone.contracts import CacheState

from .contracts import VisualCoreInput, VisualCoreOutput


def _prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        if attention_mask.shape != (seq_len, seq_len):
            raise ValueError(
                "Expected 2D attention mask with shape [seq_len, seq_len], "
                f"got {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            float_mask = torch.zeros_like(attention_mask, dtype=dtype, device=device)
            float_mask = float_mask.masked_fill(~attention_mask.to(device=device), float("-inf"))
            return float_mask
        return attention_mask.to(device=device, dtype=dtype)
    if attention_mask.ndim == 3:
        if attention_mask.shape != (batch_size, seq_len, seq_len):
            raise ValueError(
                "Expected 3D attention mask with shape [B, seq_len, seq_len], "
                f"got {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype == torch.bool:
            float_mask = torch.zeros_like(attention_mask, dtype=dtype, device=device)
            float_mask = float_mask.masked_fill(~attention_mask.to(device=device), float("-inf"))
        else:
            float_mask = attention_mask.to(device=device, dtype=dtype)
        return float_mask[:, None, :, :].expand(batch_size, num_heads, seq_len, seq_len).reshape(
            batch_size * num_heads,
            seq_len,
            seq_len,
        )
    raise ValueError(
        "Expected attention mask with shape [seq_len, seq_len] or [B, seq_len, seq_len], "
        f"got {tuple(attention_mask.shape)}"
    )


class SimpleTransformerBlock(nn.Module):
    """Small transformer block that accepts optional batch-specific masks."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        prepared_mask = _prepare_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            num_heads=self.num_heads,
            seq_len=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        normed = self.norm1(hidden_states)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=prepared_mask, need_weights=False)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LingbotVisualCore(nn.Module):
    """Shared visual core over a generic packed token sequence."""

    def __init__(self, config: LingbotCompatibleVideoBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or LingbotCompatibleVideoBackboneConfig()
        self.blocks = nn.ModuleList(
            [
                SimpleTransformerBlock(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.config.hidden_size)

    def forward(self, core_input: VisualCoreInput) -> VisualCoreOutput:
        hidden_states = core_input.tokens
        if core_input.position_context is not None:
            hidden_states = hidden_states + core_input.position_context
        if core_input.timestep_context is not None:
            hidden_states = hidden_states + core_input.timestep_context
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=core_input.attention_mask)
        hidden_states = self.final_norm(hidden_states)
        cache_state = core_input.cache_state or CacheState(
            supported=False,
            current_start_frame=0,
            cached_frames=0,
            chunk_size=hidden_states.shape[1],
            payload={"stage": "visual_core"},
        )
        return VisualCoreOutput(
            tokens=hidden_states,
            token_layout=core_input.token_layout,
            cache_state=cache_state,
            aux={"used_attention_mask": core_input.attention_mask is not None},
        )
