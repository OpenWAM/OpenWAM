from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch import nn

from open_wam.models.common import build_register_attention_mask, build_register_position_context
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.video_backbone.contracts import AttentionCacheEntry, CacheState, CacheUpdateMetadata

from .contracts import RegisterSequenceComponents, VisualCoreInput, VisualCoreOutput
from .grid_ids import build_sequence_grid_ids, build_video_grid_ids


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


def _prepend_cached_prefix_mask(
    attention_mask: torch.Tensor | None,
    *,
    cached_prefix_visibility: torch.Tensor | None,
    prefix_len: int,
    cached_segment_lengths: tuple[int, ...] | None = None,
) -> torch.Tensor | None:
    if attention_mask is None or cached_prefix_visibility is None or prefix_len <= 0:
        return attention_mask
    visibility = cached_prefix_visibility
    visibility_width = int(visibility.shape[-1])
    if visibility_width != prefix_len:
        segment_lengths = tuple(int(length) for length in (cached_segment_lengths or ()))
        if not segment_lengths:
            segment_lengths = (visibility_width,)
        prefix_chunks: list[torch.Tensor] = []
        source_offset = 0
        remaining_prefix = prefix_len
        for segment_length in segment_lengths:
            if segment_length <= 0 or remaining_prefix <= 0:
                continue
            take = min(segment_length, remaining_prefix)
            if source_offset >= visibility_width:
                source_offset = 0
            source_end = min(source_offset + take, visibility_width)
            chunk = visibility[..., source_offset:source_end]
            if chunk.shape[-1] < take:
                # When the current visibility span is narrower than the total
                # cached prefix, repeat the source pattern across cached
                # segments. This keeps the mask width aligned with merged cache
                # entries produced by repeated warmup passes.
                repeat_factor = math.ceil(take / max(chunk.shape[-1], 1))
                repeats = [1] * chunk.ndim
                repeats[-1] = repeat_factor
                chunk = chunk.repeat(*repeats)[..., :take]
            prefix_chunks.append(chunk)
            source_offset = (source_offset + take) % max(visibility_width, 1)
            remaining_prefix -= take
        if remaining_prefix > 0:
            repeat_factor = math.ceil(remaining_prefix / max(visibility_width, 1))
            repeats = [1] * visibility.ndim
            repeats[-1] = repeat_factor
            tail = visibility.repeat(*repeats)[..., :remaining_prefix]
            prefix_chunks.append(tail)
        visibility = torch.cat(prefix_chunks, dim=-1)
    if attention_mask.ndim == 2:
        if visibility.ndim == 3:
            visibility = visibility[0]
        prefix = visibility
        return torch.cat([prefix.to(dtype=attention_mask.dtype), attention_mask], dim=-1)
    if attention_mask.ndim == 3:
        prefix = visibility
        return torch.cat([prefix.to(dtype=attention_mask.dtype), attention_mask], dim=-1)
    if attention_mask.ndim == 4:
        prefix = visibility[:, None, :, :].expand(
            -1,
            attention_mask.shape[1],
            -1,
            prefix_len,
        )
        return torch.cat([prefix.to(dtype=attention_mask.dtype), attention_mask], dim=-1)
    raise ValueError(
        "Expected attention mask with shape [seq, seq], [B, seq, seq], or [B, H, seq, seq], "
        f"got {tuple(attention_mask.shape)}"
    )


def _merge_attention_cache_entries(
    existing: AttentionCacheEntry | None,
    new_entry: AttentionCacheEntry | None,
    *,
    max_tokens: int | None,
) -> AttentionCacheEntry:
    if new_entry is None or new_entry.key is None or new_entry.value is None:
        return existing if existing is not None else AttentionCacheEntry()
    if existing is not None and existing.key is not None and existing.value is not None:
        key = torch.cat([existing.key, new_entry.key], dim=2)
        value = torch.cat([existing.value, new_entry.value], dim=2)
        metadata = dict(existing.metadata)
    else:
        key = new_entry.key
        value = new_entry.value
        metadata = {}
    existing_segments = metadata.get("segment_token_lengths")
    if existing_segments is None:
        existing_segments_tuple: tuple[int, ...] = tuple()
        if existing is not None and existing.key is not None:
            existing_segments_tuple = (int(existing.key.shape[2]),)
    else:
        existing_segments_tuple = tuple(int(length) for length in existing_segments)
    new_segments = new_entry.metadata.get("segment_token_lengths")
    if new_segments is None:
        new_segments_tuple = (int(new_entry.key.shape[2]),)
    else:
        new_segments_tuple = tuple(int(length) for length in new_segments)
    segment_token_lengths = existing_segments_tuple + new_segments_tuple
    if max_tokens is not None and key.shape[2] > max_tokens:
        trimmed_segments: list[int] = []
        remaining = max_tokens
        for segment_length in reversed(segment_token_lengths):
            if remaining <= 0:
                break
            take = min(segment_length, remaining)
            trimmed_segments.append(take)
            remaining -= take
        segment_token_lengths = tuple(reversed(trimmed_segments))
        key = key[:, :, -max_tokens:, :]
        value = value[:, :, -max_tokens:, :]
    metadata.update(new_entry.metadata)
    metadata["cached_tokens"] = int(key.shape[2])
    metadata["segment_token_lengths"] = segment_token_lengths
    return AttentionCacheEntry(key=key, value=value, metadata=metadata)


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
        cached_key_value: AttentionCacheEntry | None = None,
        cached_prefix_visibility: torch.Tensor | None = None,
        cache_current_token_count: int = 0,
        cache_current_token_span: tuple[int, int] | None = None,
        kv_cache_override: AttentionCacheEntry | None = None,
    ) -> tuple[torch.Tensor, AttentionCacheEntry | None]:
        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        if kv_cache_override is not None and kv_cache_override.key is not None and kv_cache_override.value is not None:
            key = kv_cache_override.key.to(device=q.device, dtype=q.dtype)
            value = kv_cache_override.value.to(device=q.device, dtype=q.dtype)
            current_cache_entry = kv_cache_override
        else:
            key = self.norm_k(self.to_k(k)).unflatten(2, (self.heads, -1))
            value = self.to_v(v).unflatten(2, (self.heads, -1))
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, rotary_emb)
                key = _apply_rotary_emb(key, rotary_emb)
            elif rotary_emb is None:
                query = query
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            current_cache_entry = (
                AttentionCacheEntry(
                    key=key[
                        :,
                        :,
                        (
                            cache_current_token_span[0]
                            if cache_current_token_span is not None
                            else 0
                        ) : (
                            cache_current_token_span[1]
                            if cache_current_token_span is not None
                            else cache_current_token_count
                        ),
                        :,
                    ].detach(),
                    value=value[
                        :,
                        :,
                        (
                            cache_current_token_span[0]
                            if cache_current_token_span is not None
                            else 0
                        ) : (
                            cache_current_token_span[1]
                            if cache_current_token_span is not None
                            else cache_current_token_count
                        ),
                        :,
                    ].detach(),
                    metadata={
                        "cached_tokens": int(
                            (
                                cache_current_token_span[1] - cache_current_token_span[0]
                            ) if cache_current_token_span is not None else cache_current_token_count
                        ),
                        "segment_token_lengths": (
                            int(
                                (
                                    cache_current_token_span[1] - cache_current_token_span[0]
                                ) if cache_current_token_span is not None else cache_current_token_count
                            ),
                        ),
                    },
                )
                if (
                    (cache_current_token_span[1] - cache_current_token_span[0]) > 0
                    if cache_current_token_span is not None
                    else cache_current_token_count > 0
                )
                else None
            )
        if kv_cache_override is None:
            query = query.transpose(1, 2)
        else:
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, rotary_emb)
            query = query.transpose(1, 2)
        if cached_key_value is not None and cached_key_value.key is not None and cached_key_value.value is not None:
            key = torch.cat([cached_key_value.key.to(device=q.device, dtype=key.dtype), key], dim=2)
            value = torch.cat([cached_key_value.value.to(device=q.device, dtype=value.dtype), value], dim=2)
            attention_mask = _prepend_cached_prefix_mask(
                attention_mask,
                cached_prefix_visibility=cached_prefix_visibility,
                prefix_len=int(cached_key_value.key.shape[2]),
                cached_segment_lengths=tuple(cached_key_value.metadata.get("segment_token_lengths", ())),
            )
        sdpa_mask = _prepare_sdpa_mask(attention_mask, device=query.device)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=sdpa_mask)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states, current_cache_entry


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
        self_attention_cache_entry: AttentionCacheEntry | None = None,
        cross_attention_cache_entry: AttentionCacheEntry | None = None,
        cached_prefix_visibility: torch.Tensor | None = None,
        cache_current_token_count: int = 0,
        cache_current_token_span: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, AttentionCacheEntry | None, AttentionCacheEntry | None]:
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
        attn_output, self_cache_entry = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            cached_key_value=self_attention_cache_entry,
            cached_prefix_visibility=cached_prefix_visibility,
            cache_current_token_count=cache_current_token_count,
            cache_current_token_span=cache_current_token_span,
        )
        hidden_states = (hidden_states.float() + attn_output.float() * gate_msa).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output, cross_cache_entry = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            rotary_emb=None,
            attention_mask=None,
            kv_cache_override=cross_attention_cache_entry,
            cache_current_token_count=encoder_hidden_states.shape[1] if cross_attention_cache_entry is None else 0,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states, self_cache_entry, cross_cache_entry


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
        self.action_time_conditioner = LingbotReplicaTimeEmbedding(self.config.hidden_size, self.config.freq_dim)
        self.text_proj = PixArtAlphaTextProjection(self.config.text_dim, self.config.hidden_size, act_fn="gelu_tanh")
        self.action_text_proj = PixArtAlphaTextProjection(self.config.text_dim, self.config.hidden_size, act_fn="gelu_tanh")
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

    def _resolve_stream_ids(
        self,
        stream_ids: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if stream_ids is None:
            return torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
        if stream_ids.ndim == 1:
            if stream_ids.shape[0] != seq_len:
                raise ValueError(f"Expected 1D stream_ids with length {seq_len}, got {tuple(stream_ids.shape)}")
            return stream_ids[None, :].expand(batch_size, -1).to(device=device, dtype=torch.long)
        if stream_ids.ndim == 2:
            if stream_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"Expected 2D stream_ids with shape {(batch_size, seq_len)}, got {tuple(stream_ids.shape)}"
                )
            return stream_ids.to(device=device, dtype=torch.long)
        raise ValueError(f"Expected stream_ids with ndim 1 or 2, got shape {tuple(stream_ids.shape)}")

    def _materialize_register_components(
        self,
        register_components: RegisterSequenceComponents,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = register_components.layout
        if register_components.semantics.sequence_family != "register_sequence":
            raise ValueError(
                "Replica core only supports the generic structured register-sequence family on the "
                f"structured register path, got {register_components.semantics.sequence_family!r}."
            )
        if register_components.semantics.attention_style != "blockwise_causal":
            raise ValueError(
                "Replica core currently supports only `blockwise_causal` register attention style, "
                f"got {register_components.semantics.attention_style!r}."
            )
        video_grid_ids = build_video_grid_ids(
            register_components.token_grid,
            device=device,
            frame_shift=float(register_components.current_start_frame),
        )
        packed_token_chunks = []
        packed_grid_chunks = []
        if register_components.clean_video_prefix_tokens is not None:
            packed_token_chunks.append(register_components.clean_video_prefix_tokens)
            packed_grid_chunks.append(video_grid_ids)
        packed_token_chunks.extend(
            [
                register_components.noisy_video_tokens,
                register_components.action_register_tokens,
                register_components.state_register_tokens,
            ]
        )
        packed_grid_chunks.extend(
            [
                video_grid_ids,
                build_sequence_grid_ids(
                    register_components.action_register_tokens.shape[1],
                    device=device,
                    offset=0.0,
                ),
                build_sequence_grid_ids(
                    register_components.state_register_tokens.shape[1],
                    device=device,
                    offset=float(register_components.action_register_tokens.shape[1]),
                ),
            ]
        )
        packed_tokens = torch.cat(packed_token_chunks, dim=1)
        packed_grid_ids = torch.cat(packed_grid_chunks, dim=1)
        position_context = build_register_position_context(
            layout=layout,
            token_grid=register_components.token_grid,
            hidden_size=self.config.hidden_size,
            device=device,
            current_start_frame=register_components.current_start_frame,
        )[None, :, :].expand(batch_size, -1, -1)
        clean_video_values = torch.zeros(
            batch_size,
            layout.clean_video_sequence_length,
            device=device,
            dtype=torch.float32,
        )
        noisy_video_values = register_components.video_timesteps.repeat_interleave(
            register_components.token_grid.tokens_per_frame,
            dim=1,
        )
        timestep_chunks = []
        if layout.has_clean_video_prefix:
            timestep_chunks.append(clean_video_values)
        timestep_chunks.append(noisy_video_values)
        if register_components.action_register_tokens.shape[1] > 0:
            timestep_chunks.append(register_components.action_timesteps)
        if register_components.state_register_tokens.shape[1] > 0:
            timestep_chunks.append(register_components.state_timesteps)
        timestep_values = torch.cat(timestep_chunks, dim=1)
        attention_mask = build_register_attention_mask(layout, batch_size=batch_size, device=device)
        stream_id_chunks = []
        if layout.has_clean_video_prefix:
            stream_id_chunks.append(
                torch.zeros(batch_size, layout.clean_video_sequence_length, device=device, dtype=torch.long)
            )
        stream_id_chunks.extend(
            [
                torch.zeros(batch_size, layout.noisy_video_sequence_length, device=device, dtype=torch.long),
                torch.ones(batch_size, register_components.action_register_tokens.shape[1], device=device, dtype=torch.long),
                torch.ones(batch_size, register_components.state_register_tokens.shape[1], device=device, dtype=torch.long),
            ]
        )
        stream_ids = torch.cat(stream_id_chunks, dim=1)
        return packed_tokens, position_context, packed_grid_ids, timestep_values, attention_mask, stream_ids

    def _select_stream_tensor(
        self,
        video_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        stream_ids: torch.Tensor,
    ) -> torch.Tensor:
        if video_tensor.ndim == 3:
            mask = stream_ids[..., None].bool()
        elif video_tensor.ndim == 4:
            mask = stream_ids[..., None, None].bool()
        else:
            raise ValueError(f"Unsupported stream-conditioned tensor rank {video_tensor.ndim}")
        return torch.where(mask, action_tensor, video_tensor)

    def _resolve_encoder_hidden_states(
        self,
        core_input: VisualCoreInput,
        stream_ids: torch.Tensor,
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
            video_hidden_states = text_context.to(dtype=dtype)
            action_hidden_states = video_hidden_states
        else:
            video_hidden_states = self.text_proj(text_context).to(dtype=dtype)
            action_hidden_states = self.action_text_proj(text_context).to(dtype=dtype)
        action_fraction = stream_ids.float().mean(dim=1, keepdim=True).unsqueeze(-1)
        return (1.0 - action_fraction) * video_hidden_states + action_fraction * action_hidden_states

    def forward(self, core_input: VisualCoreInput) -> VisualCoreOutput:
        token_layout = core_input.token_layout
        if core_input.register_components is not None:
            (
                hidden_states,
                position_context,
                grid_ids,
                timestep_values,
                attention_mask,
                stream_ids_tensor,
            ) = self._materialize_register_components(
                core_input.register_components,
                batch_size=core_input.register_components.noisy_video_tokens.shape[0],
                device=core_input.register_components.noisy_video_tokens.device,
            )
            token_layout = core_input.register_components.layout
        else:
            if core_input.tokens is None:
                raise ValueError("Replica core expected `tokens` unless `register_components` is provided.")
            hidden_states = core_input.tokens
            position_context = core_input.position_context
            grid_ids = core_input.grid_ids
            timestep_values = core_input.timestep_values
            attention_mask = core_input.attention_mask
            stream_ids_tensor = core_input.stream_ids
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        stream_ids = self._resolve_stream_ids(stream_ids_tensor, batch_size=batch_size, seq_len=seq_len, device=device)

        if position_context is not None and grid_ids is None:
            hidden_states = hidden_states + position_context

        if timestep_values is None:
            if core_input.timestep_context is not None:
                hidden_states = hidden_states + core_input.timestep_context
                video_temb = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device, dtype=dtype)
                video_timestep_proj = torch.zeros(
                    batch_size,
                    seq_len,
                    6,
                    self.config.hidden_size,
                    device=device,
                    dtype=dtype,
                )
                action_temb = video_temb
                action_timestep_proj = video_timestep_proj
            else:
                timestep_values = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
                video_temb, video_timestep_proj = self.time_conditioner(timestep_values, dtype=dtype)
                action_temb, action_timestep_proj = self.action_time_conditioner(timestep_values, dtype=dtype)
        else:
            timestep_values = timestep_values.to(device=device)
            video_temb, video_timestep_proj = self.time_conditioner(timestep_values, dtype=dtype)
            action_temb, action_timestep_proj = self.action_time_conditioner(timestep_values, dtype=dtype)

        temb = self._select_stream_tensor(video_temb, action_temb, stream_ids)
        timestep_proj = self._select_stream_tensor(video_timestep_proj, action_timestep_proj, stream_ids)

        rotary_emb = self.rope(grid_ids.to(device=device))[:, :, None] if grid_ids is not None else None
        encoder_hidden_states = self._resolve_encoder_hidden_states(
            core_input,
            stream_ids=stream_ids,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        cache_update_metadata = core_input.cache_update_metadata or CacheUpdateMetadata()
        cache_metadata = core_input.sequence_metadata.metadata if core_input.sequence_metadata is not None else {}
        cacheable_video_tokens = int(cache_metadata.get("cacheable_video_tokens", 0))
        cache_reference_start = int(cache_metadata.get("cache_reference_start", 0))
        cache_reference_end = int(cache_metadata.get("cache_reference_end", cache_reference_start))
        tokens_per_frame = int(cache_metadata.get("tokens_per_frame", 0))
        max_cached_tokens = None
        if cache_update_metadata.max_cached_frames is not None and tokens_per_frame > 0:
            max_cached_tokens = cache_update_metadata.max_cached_frames * tokens_per_frame
        cached_prefix_visibility = None
        if (
            attention_mask is not None
            and cache_reference_end > cache_reference_start
            and attention_mask.shape[-1] >= cache_reference_end
        ):
            cached_prefix_visibility = attention_mask[..., cache_reference_start:cache_reference_end]

        next_self_attention_kv: list[AttentionCacheEntry] = []
        next_cross_attention_kv: list[AttentionCacheEntry] = []
        incoming_self_attention_kv = core_input.cache_state.self_attention_kv if core_input.cache_state is not None else tuple()
        incoming_cross_attention_kv = core_input.cache_state.cross_attention_kv if core_input.cache_state is not None else tuple()

        for layer_index, block in enumerate(self.blocks):
            hidden_states, current_self_cache_entry, current_cross_cache_entry = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                attention_mask=attention_mask,
                self_attention_cache_entry=(
                    incoming_self_attention_kv[layer_index]
                    if layer_index < len(incoming_self_attention_kv)
                    else None
                ),
                cross_attention_cache_entry=(
                    incoming_cross_attention_kv[layer_index]
                    if layer_index < len(incoming_cross_attention_kv)
                    else None
                ),
                cached_prefix_visibility=cached_prefix_visibility,
                cache_current_token_count=(
                    cacheable_video_tokens if cache_update_metadata.update_kv_cache and cacheable_video_tokens > 0 else 0
                ),
                cache_current_token_span=(
                    (cache_reference_start, cache_reference_end)
                    if cache_update_metadata.update_kv_cache and cache_reference_end > cache_reference_start
                    else None
                ),
            )
            existing_self_entry = incoming_self_attention_kv[layer_index] if layer_index < len(incoming_self_attention_kv) else None
            if cache_update_metadata.update_kv_cache and current_self_cache_entry is not None:
                next_self_attention_kv.append(
                    _merge_attention_cache_entries(
                        existing_self_entry,
                        current_self_cache_entry,
                        max_tokens=max_cached_tokens,
                    )
                )
            else:
                next_self_attention_kv.append(existing_self_entry or AttentionCacheEntry())
            existing_cross_entry = incoming_cross_attention_kv[layer_index] if layer_index < len(incoming_cross_attention_kv) else None
            if existing_cross_entry is not None and existing_cross_entry.key is not None and existing_cross_entry.value is not None:
                next_cross_attention_kv.append(existing_cross_entry)
            elif cache_update_metadata.update_cross_attention_cache and current_cross_cache_entry is not None:
                next_cross_attention_kv.append(current_cross_cache_entry)
            else:
                next_cross_attention_kv.append(AttentionCacheEntry())

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)

        has_runtime_sequence = core_input.sequence_metadata is not None
        layer_cache_entries = (
            tuple(
                AttentionCacheEntry(
                    key=entry.key,
                    value=entry.value,
                    metadata={
                        **entry.metadata,
                        "layer_index": layer_index,
                        "sequence_length": int(entry.key.shape[2]) if entry.key is not None else seq_len,
                        "current_start_frame": cache_update_metadata.current_start_frame,
                        "implementation": "lingbot_replica",
                    },
                )
                for layer_index, entry in enumerate(next_self_attention_kv)
            )
            if has_runtime_sequence
            else tuple()
        )
        cross_layer_cache_entries = (
            tuple(
                AttentionCacheEntry(
                    key=entry.key,
                    value=entry.value,
                    metadata={
                        **entry.metadata,
                        "layer_index": layer_index,
                        "current_start_frame": cache_update_metadata.current_start_frame,
                        "implementation": "lingbot_replica",
                        "cache_kind": "cross_attention",
                    },
                )
                for layer_index, entry in enumerate(next_cross_attention_kv)
            )
            if has_runtime_sequence
            else tuple()
        )
        if core_input.cache_state is not None:
            cache_state = CacheState(
                supported=core_input.cache_state.supported or has_runtime_sequence,
                current_start_frame=cache_update_metadata.current_start_frame,
                cached_frames=core_input.cache_state.cached_frames,
                chunk_size=core_input.cache_state.chunk_size,
                capability=(
                    core_input.cache_state.capability
                    if core_input.cache_state.capability != "none"
                    else ("self_attn_plus_cross_attn" if has_runtime_sequence else "none")
                ),
                payload=dict(core_input.cache_state.payload),
                self_attention_kv=(
                    core_input.cache_state.self_attention_kv
                    if core_input.cache_state.self_attention_kv and not cache_update_metadata.update_kv_cache
                    else layer_cache_entries
                ),
                cross_attention_kv=(
                    core_input.cache_state.cross_attention_kv
                    if core_input.cache_state.cross_attention_kv
                    else cross_layer_cache_entries
                ),
                update_metadata=cache_update_metadata,
            )
        else:
            cache_state = CacheState(
                supported=has_runtime_sequence,
                current_start_frame=cache_update_metadata.current_start_frame,
                cached_frames=0,
                chunk_size=seq_len,
                capability="self_attn_plus_cross_attn" if has_runtime_sequence else "none",
                payload={"stage": "lingbot_replica_core", "implementation": "lingbot_replica"},
                self_attention_kv=layer_cache_entries,
                cross_attention_kv=cross_layer_cache_entries,
                update_metadata=cache_update_metadata,
            )
        return VisualCoreOutput(
            tokens=hidden_states,
            token_layout=token_layout,
            cache_state=cache_state,
            aux={
                "implementation": "lingbot_replica",
                "used_rotary": core_input.grid_ids is not None,
                "used_action_conditioner": bool((stream_ids != 0).any().item()),
                "has_sequence_metadata": core_input.sequence_metadata is not None,
                "cache_runtime_metadata": cache_update_metadata,
            },
        )
