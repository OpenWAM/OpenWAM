from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch import nn

from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.video_backbone.contracts import CacheState

from .contracts import VisualCoreInput, VisualCoreOutput


class LingbotReplicaTimeEmbedding(nn.Module):
    """Wan-style timestep conditioner used by the replica core."""

    def __init__(self, hidden_size: int, freq_dim: int) -> None:
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=freq_dim, time_embed_dim=hidden_size)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(hidden_size, hidden_size * 6)

    def forward(self, timestep_values: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = timestep_values.shape
        flat = timestep_values.reshape(-1)
        projected = self.timesteps_proj(flat)
        projected = projected.to(self.time_embedder.linear_1.weight.dtype)
        temb = self.time_embedder(projected).to(dtype=dtype).reshape(batch_size, seq_len, -1)
        timestep_proj = self.time_proj(self.act_fn(temb)).reshape(batch_size, seq_len, 6, -1)
        return temb, timestep_proj


class LingbotReplicaRotaryPosEmbed(nn.Module):
    """Wan-style rotary embedding over frame, height, and width axes."""

    def __init__(self, attention_head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.theta = theta
        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3
        self.register_buffer("f_freqs_base", self._make_freqs_base(self.f_dim), persistent=False)
        self.register_buffer("h_freqs_base", self._make_freqs_base(self.h_dim), persistent=False)
        self.register_buffer("w_freqs_base", self._make_freqs_base(self.w_dim), persistent=False)

    def _make_freqs_base(self, dim: int) -> torch.Tensor:
        half_dim = max(1, dim // 2)
        return 1.0 / (self.theta ** (torch.arange(0, dim, 2)[:half_dim].double() / max(dim, 1)))

    def forward(self, grid_ids: torch.Tensor) -> torch.Tensor:
        if grid_ids.ndim == 2:
            grid_ids = grid_ids.unsqueeze(0)
        f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
        h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
        w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
        freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
        return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    if freqs.ndim == 3:
        freqs = freqs[:, :, None, :]
    x_out = torch.view_as_real(x_complex * freqs).flatten(3)
    return x_out.to(x.dtype)


def _prepare_sdpa_mask(attention_mask: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        return attention_mask[None, None, :, :].to(device=device)
    if attention_mask.ndim == 3:
        return attention_mask[:, None, :, :].to(device=device)
    if attention_mask.ndim == 4:
        return attention_mask.to(device=device)
    raise ValueError(
        "Expected attention mask with shape [seq, seq], [B, seq, seq], or [B, H, seq, seq], "
        f"got {tuple(attention_mask.shape)}"
    )


class LingbotReplicaAttention(nn.Module):
    """Wan-style attention block with SDPA mask support."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float,
        dropout: float = 0.0,
        cross_attention_dim_head: int | None = None,
    ) -> None:
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=True), nn.Dropout(dropout)])
        self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        key = self.norm_k(self.to_k(k)).unflatten(2, (self.heads, -1))
        value = self.to_v(v).unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        sdpa_mask = _prepare_sdpa_mask(attention_mask, device=query.device)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=sdpa_mask)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class LingbotReplicaTransformerBlock(nn.Module):
    """Wan-style transformer block with self-attn, cross-attn, and FFN."""

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = LingbotReplicaAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
        )
        self.attn2 = LingbotReplicaAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = rearrange(
            temb_scale_shift_table,
            "b l n c -> b n l c",
        ).chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
        )
        hidden_states = (hidden_states.float() + attn_output.float() * gate_msa).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            rotary_emb=None,
            attention_mask=None,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class LingbotReplicaVisualCore(nn.Module):
    """Wan-style shared core that matches the LingBot block architecture more closely."""

    def __init__(self, config: LingbotCompatibleVideoBackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or LingbotCompatibleVideoBackboneConfig()
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"Expected hidden_size {self.config.hidden_size} to be divisible by num_heads {self.config.num_heads}."
            )
        self.inner_dim = self.config.hidden_size
        self.ffn_dim = self.config.ffn_dim or (self.config.hidden_size * self.config.mlp_ratio)
        self.rope = LingbotReplicaRotaryPosEmbed(self.config.hidden_size // self.config.num_heads)
        self.time_conditioner = LingbotReplicaTimeEmbedding(self.config.hidden_size, self.config.freq_dim)
        self.text_proj = PixArtAlphaTextProjection(self.config.text_dim, self.config.hidden_size, act_fn="gelu_tanh")
        self.blocks = nn.ModuleList(
            [
                LingbotReplicaTransformerBlock(
                    dim=self.config.hidden_size,
                    ffn_dim=self.ffn_dim,
                    num_heads=self.config.num_heads,
                    cross_attn_norm=self.config.cross_attn_norm,
                    eps=self.config.latent_norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(self.config.hidden_size, self.config.latent_norm_eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, self.config.hidden_size) / self.config.hidden_size**0.5)

    def _resolve_encoder_hidden_states(
        self,
        core_input: VisualCoreInput,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        text_context = core_input.text_context
        if text_context is None and core_input.conditioning is not None:
            text_context = core_input.conditioning.text_context
        if text_context is None:
            return torch.zeros(batch_size, 1, self.config.hidden_size, device=device, dtype=dtype)
        text_context = text_context.to(device=device)
        if text_context.ndim == 2:
            text_context = text_context[:, None, :]
        if text_context.shape[-1] == self.config.hidden_size:
            return text_context.to(dtype=dtype)
        return self.text_proj(text_context).to(dtype=dtype)

    def forward(self, core_input: VisualCoreInput) -> VisualCoreOutput:
        hidden_states = core_input.tokens
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        if core_input.position_context is not None and core_input.grid_ids is None:
            hidden_states = hidden_states + core_input.position_context

        timestep_values = core_input.timestep_values
        if timestep_values is None:
            if core_input.timestep_context is not None:
                hidden_states = hidden_states + core_input.timestep_context
                temb = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device, dtype=dtype)
                timestep_proj = torch.zeros(batch_size, seq_len, 6, self.config.hidden_size, device=device, dtype=dtype)
            else:
                timestep_values = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
                temb, timestep_proj = self.time_conditioner(timestep_values, dtype=dtype)
        else:
            temb, timestep_proj = self.time_conditioner(timestep_values.to(device=device), dtype=dtype)

        rotary_emb = self.rope(core_input.grid_ids.to(device=device))[:, :, None] if core_input.grid_ids is not None else None
        encoder_hidden_states = self._resolve_encoder_hidden_states(core_input, batch_size=batch_size, dtype=dtype, device=device)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                attention_mask=core_input.attention_mask,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)

        cache_state = core_input.cache_state or CacheState(
            supported=False,
            current_start_frame=0,
            cached_frames=0,
            chunk_size=seq_len,
            payload={"stage": "lingbot_replica_core", "implementation": "lingbot_replica"},
        )
        return VisualCoreOutput(
            tokens=hidden_states,
            token_layout=core_input.token_layout,
            cache_state=cache_state,
            aux={"implementation": "lingbot_replica", "used_rotary": core_input.grid_ids is not None},
        )
