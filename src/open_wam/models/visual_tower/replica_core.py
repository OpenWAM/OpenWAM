from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch import nn

from open_wam.models.common import (
    PreparedAttentionProfile,
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    apply_attention_backend,
    build_chunked_temporal_exact_attention_profile,
    cache_backend_uses_slot_pool,
    build_register_attention_mask,
    build_register_position_context,
    clear_cache_backend_payload,
    init_cache_backend_payload,
    materialize_cache_backend_entries,
    normalize_attention_profile_name,
    resolve_cache_backend_spec,
    select_attention_profile_mask,
    SlotPoolLayerState,
    unpatchify_video_tokens,
    update_slot_pool_layer_state,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig, resolve_stage_attention_mode
from open_wam.models.video_backbone.contracts import (
    AttentionCacheEntry,
    CacheBranchState,
    CacheState,
    CacheUpdateMetadata,
    replace_cache_branch_state,
    resolve_cache_branch_state,
)

from .contracts import (
    RegisterSequenceComponents,
    StructuredAttentionContext,
    StructuredBlockSemantics,
    StructuredFrequencyBundle,
    VisualCoreInput,
    VisualIntermediateReadout,
    VisualCoreOutput,
)
from .grid_ids import build_sequence_grid_ids, build_video_grid_ids
from .runtime_programs import RuntimeStepInput, RuntimeStepOutput
from .sequence_adapters import prepare_exact_dual_stream_train_sequence, prepare_runtime_sequence
from .stream_adapters import PreparedStreamInput, SharedRuntimeStreamAdapters
from .stream_heads import project_runtime_stream_outputs
from .structured_attention import (
    StructuredAttentionExecutionPlan,
    build_structured_attention_execution_plan,
    execute_structured_attention,
)


class SharedTransformerTimeEmbedding(nn.Module):
    """Wan-style timestep conditioner used by the shared transformer core."""

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


class SharedTransformerRotaryPositionalEmbedding(nn.Module):
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


def _select_chunk_slices(tensor: torch.Tensor, count: int) -> tuple[torch.Tensor, ...]:
    chunked = rearrange(tensor, "b l n c -> b n l c").contiguous()
    if int(chunked.shape[1]) != count:
        raise ValueError(f"Expected chunk axis length {count}, got {tuple(chunked.shape)}.")
    return tuple(chunked[:, index, :, :].clone() for index in range(count))


def _select_split_segments(tensor: torch.Tensor, lengths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    offset = 0
    segments: list[torch.Tensor] = []
    for length in lengths:
        segments.append(tensor.narrow(1, offset, length).clone())
        offset += length
    return tuple(segments)


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


def _resolve_slot_pool_prefix_visibility(
    attention_mask: torch.Tensor | None,
    *,
    prefix_len: int,
    prefix_visibility_mode: str,
    query_stream_ids: torch.Tensor | None = None,
    cached_prefix_stream_ids: torch.Tensor | None = None,
    query_sequence_ids: torch.Tensor | None = None,
    cached_prefix_sequence_ids: torch.Tensor | None = None,
    allow_video_query_to_action_prefix_tail_tokens: int = 0,
) -> torch.Tensor | None:
    if attention_mask is None or prefix_len <= 0:
        return attention_mask

    def _normalize_stream_ids(
        stream_ids: torch.Tensor | None,
        *,
        expected_len: int,
        label: str,
    ) -> torch.Tensor:
        if stream_ids is None:
            raise ValueError(
                f"Slot-pool prefix_visibility_mode={prefix_visibility_mode!r} requires `{label}`."
            )
        if stream_ids.ndim == 2:
            if stream_ids.shape[0] != 1:
                raise ValueError(
                    f"Slot-pool `{label}` must be rank-1 or batch-shared rank-2, "
                    f"got shape {tuple(stream_ids.shape)}."
                )
            stream_ids = stream_ids.squeeze(0)
        if stream_ids.ndim != 1 or int(stream_ids.shape[0]) != expected_len:
            raise ValueError(
                f"Slot-pool `{label}` must have length {expected_len}, "
                f"got shape {tuple(stream_ids.shape)}."
            )
        return stream_ids.to(device=attention_mask.device, dtype=torch.long)

    if prefix_visibility_mode == "full_history":
        cached_prefix_visibility_2d = torch.ones(
            attention_mask.shape[-2],
            prefix_len,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
    elif prefix_visibility_mode == "preserve_video_pretrain_history":
        q_stream = _normalize_stream_ids(
            query_stream_ids,
            expected_len=int(attention_mask.shape[-2]),
            label="query_stream_ids",
        )
        kv_stream = _normalize_stream_ids(
            cached_prefix_stream_ids,
            expected_len=prefix_len,
            label="cached_prefix_stream_ids",
        )
        valid_streams = (q_stream[:, None] >= 0) & (kv_stream[None, :] >= 0)
        cached_prefix_visibility_2d = (
            ((q_stream[:, None] == kv_stream[None, :]) | (q_stream[:, None] == 1))
            & valid_streams
        )
        tail_tokens = max(0, min(int(allow_video_query_to_action_prefix_tail_tokens), int(prefix_len)))
        if tail_tokens > 0:
            tail_positions = torch.arange(prefix_len, device=attention_mask.device) >= (prefix_len - tail_tokens)
            # Staged action-then-video commits the current clean action before
            # denoising current video. Training permits that same-current-chunk
            # action context while still hiding older action history from video.
            cached_prefix_visibility_2d = cached_prefix_visibility_2d | (
                (q_stream[:, None] == 0)
                & (kv_stream[None, :] == 1)
                & tail_positions[None, :]
                & valid_streams
            )
        cached_prefix_visibility_2d = cached_prefix_visibility_2d.to(dtype=attention_mask.dtype)
    elif prefix_visibility_mode == "video_history_only":
        q_stream = _normalize_stream_ids(
            query_stream_ids,
            expected_len=int(attention_mask.shape[-2]),
            label="query_stream_ids",
        )
        kv_stream = _normalize_stream_ids(
            cached_prefix_stream_ids,
            expected_len=prefix_len,
            label="cached_prefix_stream_ids",
        )
        valid_streams = (q_stream[:, None] >= 0) & (kv_stream[None, :] >= 0)
        cached_prefix_visibility_2d = (kv_stream[None, :] == 0) & valid_streams
        tail_tokens = max(0, min(int(allow_video_query_to_action_prefix_tail_tokens), int(prefix_len)))
        if tail_tokens > 0:
            tail_positions = torch.arange(prefix_len, device=attention_mask.device) >= (prefix_len - tail_tokens)
            cached_prefix_visibility_2d = cached_prefix_visibility_2d | (
                (q_stream[:, None] == 0)
                & (kv_stream[None, :] == 1)
                & tail_positions[None, :]
                & valid_streams
            )
        cached_prefix_visibility_2d = cached_prefix_visibility_2d.to(dtype=attention_mask.dtype)
    else:
        raise ValueError(f"Unsupported slot-pool prefix_visibility_mode {prefix_visibility_mode!r}.")
    if query_sequence_ids is not None or cached_prefix_sequence_ids is not None:
        q_seq = _normalize_stream_ids(
            query_sequence_ids,
            expected_len=int(attention_mask.shape[-2]),
            label="query_sequence_ids",
        )
        kv_seq = _normalize_stream_ids(
            cached_prefix_sequence_ids,
            expected_len=prefix_len,
            label="cached_prefix_sequence_ids",
        )
        same_sequence = (q_seq[:, None] == kv_seq[None, :]) & (q_seq[:, None] >= 0) & (kv_seq[None, :] >= 0)
        if cached_prefix_visibility_2d.dtype == torch.bool:
            cached_prefix_visibility_2d = cached_prefix_visibility_2d & same_sequence
        else:
            cached_prefix_visibility_2d = cached_prefix_visibility_2d * same_sequence.to(
                dtype=cached_prefix_visibility_2d.dtype
            )

    if attention_mask.ndim == 2:
        cached_prefix_visibility = cached_prefix_visibility_2d
    elif attention_mask.ndim == 3:
        cached_prefix_visibility = cached_prefix_visibility_2d[None].expand(
            attention_mask.shape[0],
            -1,
            -1,
        )
    elif attention_mask.ndim == 4:
        cached_prefix_visibility = cached_prefix_visibility_2d[None, None].expand(
            attention_mask.shape[0],
            attention_mask.shape[1],
            -1,
            -1,
        )
    else:  # pragma: no cover - defensive guard
        raise ValueError(
            "Unsupported attention mask rank while resolving slot-pool prefix visibility: "
            f"{tuple(attention_mask.shape)}"
        )
    return _prepend_cached_prefix_mask(
        attention_mask,
        cached_prefix_visibility=cached_prefix_visibility,
        prefix_len=prefix_len,
    )


def _packed_slot_pool_query_sequence_ids(
    *,
    attention_profile: PreparedAttentionProfile | None,
    query_stream_ids: torch.Tensor | None,
    query_len: int,
    cache_batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Return sequence ids for exact joint current tokens packed into batch 1."""

    if attention_profile is None:
        return None
    profile_batch_size = int(attention_profile.metadata.get("batch_size", 0))
    if profile_batch_size != int(cache_batch_size) or profile_batch_size <= 1:
        return None
    if query_stream_ids is None:
        return None
    stream_ids = query_stream_ids.to(device=device, dtype=torch.long)
    if stream_ids.ndim == 2:
        if int(stream_ids.shape[0]) != 1:
            raise ValueError(
                "Packed slot-pool query stream ids must be rank-1 or batch-shared rank-2, "
                f"got shape {tuple(stream_ids.shape)}."
            )
        stream_ids = stream_ids.squeeze(0)
    if stream_ids.ndim != 1 or int(stream_ids.shape[0]) != int(query_len):
        raise ValueError(
            "Packed slot-pool query stream ids must have one value per current KV token, "
            f"got shape {tuple(stream_ids.shape)} for query_len={int(query_len)}."
        )

    sequence_parts: list[torch.Tensor] = []
    offset = 0
    while offset < int(query_len):
        stream_value = int(stream_ids[offset].item())
        run_end = offset + 1
        while run_end < int(query_len) and int(stream_ids[run_end].item()) == stream_value:
            run_end += 1
        run_length = run_end - offset
        if stream_value < 0:
            sequence_parts.append(torch.full((run_length,), -1, device=device, dtype=torch.long))
        else:
            if run_length % (2 * profile_batch_size) != 0:
                raise ValueError(
                    "Packed exact slot-pool stream run must contain noisy+condition components "
                    "for every packed sequence row, "
                    f"got run_length={run_length}, packed_batch={profile_batch_size}."
                )
            tokens_per_component = run_length // (2 * profile_batch_size)
            component_ids = torch.arange(profile_batch_size, device=device, dtype=torch.long).repeat_interleave(
                tokens_per_component
            )
            sequence_parts.append(torch.cat([component_ids, component_ids], dim=0))
        offset = run_end
    return torch.cat(sequence_parts, dim=0)


def _retained_slot_pool_indices_for_current_write(
    layer_state: SlotPoolLayerState,
    *,
    valid: torch.Tensor,
    current_token_count: int,
    update_mode: int,
) -> torch.Tensor:
    """Return the prefix slots visible after a non-mutating slot allocation."""

    if int(update_mode) == 0 or int(current_token_count) <= 0 or int(valid.numel()) == 0:
        return valid
    if layer_state.slot_mask is None or layer_state.slot_ids is None:
        raise ValueError("Slot-pool backend requires initialized `slot_mask` and `slot_ids` tensors.")
    if bool(layer_state.metadata.get(SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION, False)):
        return valid
    free_count = int(layer_state.slot_mask.numel()) - int(valid.numel())
    evict_count = max(0, int(current_token_count) - free_count)
    if evict_count <= 0:
        return valid
    if evict_count >= int(valid.numel()):
        return valid.new_empty((0,), dtype=valid.dtype)
    slot_ids = layer_state.slot_ids[valid]
    order = torch.argsort(slot_ids, stable=True)
    return valid[order[evict_count:]]


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


class SharedTransformerAttention(nn.Module):
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
        structured_attention_context: StructuredAttentionContext | None = None,
        structured_attention_plan: StructuredAttentionExecutionPlan | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_profile: PreparedAttentionProfile | None = None,
        is_cross_attention: bool = False,
        cached_key_value: AttentionCacheEntry | None = None,
        cached_prefix_visibility: torch.Tensor | None = None,
        cache_current_token_count: int = 0,
        cache_current_token_span: tuple[int, int] | None = None,
        detach_cache_entry: bool = True,
        kv_cache_override: AttentionCacheEntry | None = None,
        cache_backend_name: str | None = None,
        cache_backend_state=None,
        cache_backend_update_mode: int = 0,
        cache_backend_stream_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AttentionCacheEntry | None]:
        q = q.contiguous().clone()
        k = k.contiguous().clone()
        v = v.contiguous().clone()
        query = _rms_norm_with_materialized_weight(
            self.norm_q,
            _linear_with_materialized_params(self.to_q, q),
        ).unflatten(2, (self.heads, -1))
        use_slot_pool_backend = cache_backend_uses_slot_pool(cache_backend_name) and cache_backend_state is not None
        current_cache_entry = None
        if kv_cache_override is not None and kv_cache_override.key is not None and kv_cache_override.value is not None:
            key = kv_cache_override.key.to(device=q.device, dtype=q.dtype)
            value = kv_cache_override.value.to(device=q.device, dtype=q.dtype)
            current_cache_entry = kv_cache_override
        else:
            key = _rms_norm_with_materialized_weight(
                self.norm_k,
                _linear_with_materialized_params(self.to_k, k),
            ).unflatten(2, (self.heads, -1))
            value = _linear_with_materialized_params(self.to_v, v).unflatten(2, (self.heads, -1))
            structured_rotary_emb = (
                structured_attention_plan.rotary_freqs if structured_attention_plan is not None else None
            )
            if structured_rotary_emb is not None:
                query = _apply_rotary_emb(query, structured_rotary_emb)
                key = _apply_rotary_emb(key, structured_rotary_emb)
            elif rotary_emb is not None:
                query = _apply_rotary_emb(query, rotary_emb)
                key = _apply_rotary_emb(key, rotary_emb)
            else:
                query = query
            if use_slot_pool_backend:
                current_cache_entry = None
            else:
                key_t = key.transpose(1, 2)
                value_t = value.transpose(1, 2)
                cache_start = cache_current_token_span[0] if cache_current_token_span is not None else 0
                cache_end = (
                    cache_current_token_span[1]
                    if cache_current_token_span is not None
                    else cache_current_token_count
                )
                cache_token_count = int(cache_end - cache_start)
                if cache_token_count > 0:
                    cache_key = key_t[:, :, cache_start:cache_end, :]
                    cache_value = value_t[:, :, cache_start:cache_end, :]
                    if detach_cache_entry:
                        cache_key = cache_key.detach()
                        cache_value = cache_value.detach()
                    current_cache_entry = AttentionCacheEntry(
                        key=cache_key,
                        value=cache_value,
                        metadata={
                            "cached_tokens": cache_token_count,
                            "segment_token_lengths": (cache_token_count,),
                        },
                    )
                key = key_t
                value = value_t
        if kv_cache_override is None:
            structured_hidden_states = execute_structured_attention(
                query,
                key.transpose(1, 2) if key.ndim == 4 and key.shape[1] == self.heads else key,
                value.transpose(1, 2) if value.ndim == 4 and value.shape[1] == self.heads else value,
                context=structured_attention_context,
                plan=structured_attention_plan,
                cached_key_value=cached_key_value,
            )
            if structured_hidden_states is not None and not use_slot_pool_backend:
                hidden_states = structured_hidden_states.flatten(2, 3)
                hidden_states = _linear_with_materialized_params(self.to_out[0], hidden_states)
                hidden_states = self.to_out[1](hidden_states)
                return hidden_states, current_cache_entry
            query = query.transpose(1, 2)
        else:
            structured_rotary_emb = (
                structured_attention_plan.rotary_freqs if structured_attention_plan is not None else None
            )
            if structured_rotary_emb is not None:
                query = _apply_rotary_emb(query, structured_rotary_emb)
            elif rotary_emb is not None:
                query = _apply_rotary_emb(query, rotary_emb)
            query = query.transpose(1, 2)
        slot_pool_update_key = None
        slot_pool_update_value = None
        slot_pool_update_stream_ids = cache_backend_stream_ids
        if use_slot_pool_backend and kv_cache_override is None:
            if cache_backend_state.slot_mask is None or cache_backend_state.key is None or cache_backend_state.value is None:
                raise ValueError("LingBot slot-pool backend requires initialized slot mask and KV tensors.")
            valid = cache_backend_state.slot_mask.nonzero(as_tuple=False).squeeze(-1)
            if cache_backend_state.slot_ids is not None and valid.numel() > 1:
                valid = valid[torch.argsort(cache_backend_state.slot_ids[valid], stable=True)]
            current_key = key.transpose(1, 2)
            current_value = value.transpose(1, 2)
            valid = _retained_slot_pool_indices_for_current_write(
                cache_backend_state,
                valid=valid,
                current_token_count=int(current_key.shape[2]),
                update_mode=int(cache_backend_update_mode),
            )
            prefix_key = cache_backend_state.key[:, valid].transpose(1, 2).to(device=q.device, dtype=query.dtype)
            prefix_value = cache_backend_state.value[:, valid].transpose(1, 2).to(device=q.device, dtype=query.dtype)
            prefix_stream_ids = (
                cache_backend_state.stream_ids[valid].to(device=q.device)
                if cache_backend_state.stream_ids is not None
                else None
            )
            query_sequence_ids = None
            cached_prefix_sequence_ids = None
            if valid.numel() > 0 and int(prefix_key.shape[0]) != int(current_key.shape[0]):
                if int(current_key.shape[0]) != 1:
                    raise ValueError(
                        "Slot-pool prefix/current batch mismatch is only supported for packed exact-runtime "
                        f"current tokens, got prefix_batch={int(prefix_key.shape[0])}, "
                        f"current_batch={int(current_key.shape[0])}."
                    )
                prefix_batch_size = int(prefix_key.shape[0])
                prefix_token_count = int(prefix_key.shape[2])
                query_sequence_ids = _packed_slot_pool_query_sequence_ids(
                    attention_profile=attention_profile,
                    query_stream_ids=cache_backend_stream_ids,
                    query_len=int(current_key.shape[2]),
                    cache_batch_size=prefix_batch_size,
                    device=q.device,
                )
                if query_sequence_ids is None:
                    raise ValueError(
                        "Slot-pool prefix/current batch mismatch requires packed exact-runtime attention metadata."
                    )
                cached_prefix_sequence_ids = torch.arange(
                    prefix_batch_size,
                    device=q.device,
                    dtype=torch.long,
                ).repeat_interleave(prefix_token_count)
                prefix_key = (
                    prefix_key.permute(1, 0, 2, 3)
                    .reshape(prefix_key.shape[1], prefix_batch_size * prefix_token_count, prefix_key.shape[3])
                    .unsqueeze(0)
                )
                prefix_value = (
                    prefix_value.permute(1, 0, 2, 3)
                    .reshape(prefix_value.shape[1], prefix_batch_size * prefix_token_count, prefix_value.shape[3])
                    .unsqueeze(0)
                )
                if prefix_stream_ids is not None:
                    prefix_stream_ids = prefix_stream_ids.repeat(prefix_batch_size)
            key = torch.cat([prefix_key, current_key], dim=2) if valid.numel() > 0 else current_key
            value = torch.cat([prefix_value, current_value], dim=2) if valid.numel() > 0 else current_value
            if prefix_stream_ids is not None:
                if cache_backend_stream_ids is None:
                    current_stream_ids = torch.full(
                        (int(current_key.shape[2]),),
                        -1,
                        device=prefix_stream_ids.device,
                        dtype=prefix_stream_ids.dtype,
                    )
                else:
                    current_stream_ids = cache_backend_stream_ids.to(
                        device=prefix_stream_ids.device,
                        dtype=prefix_stream_ids.dtype,
                    )
                    if current_stream_ids.ndim == 2:
                        if current_stream_ids.shape[0] != 1:
                            raise ValueError(
                                "Slot-pool current stream ids must be rank-1 or batch-shared rank-2, "
                                f"got shape {tuple(current_stream_ids.shape)}."
                        )
                        current_stream_ids = current_stream_ids.squeeze(0)
                    if current_stream_ids.ndim != 1 or int(current_stream_ids.shape[0]) != int(current_key.shape[2]):
                        raise ValueError(
                            "Slot-pool current stream ids must have one value per current KV token, "
                            f"got shape {tuple(current_stream_ids.shape)} for key_size={int(current_key.shape[2])}."
                        )
                valid_stream_ids = torch.cat([prefix_stream_ids, current_stream_ids], dim=0)
            else:
                valid_stream_ids = None
            slot_pool_update_key = current_key.transpose(1, 2).detach()
            slot_pool_update_value = current_value.transpose(1, 2).detach()
        else:
            valid_stream_ids = None
        if cached_key_value is not None and cached_key_value.key is not None and cached_key_value.value is not None:
            key = torch.cat([cached_key_value.key.to(device=q.device, dtype=key.dtype), key], dim=2)
            value = torch.cat([cached_key_value.value.to(device=q.device, dtype=value.dtype), value], dim=2)
            attention_mask = _prepend_cached_prefix_mask(
                attention_mask,
                cached_prefix_visibility=cached_prefix_visibility,
                prefix_len=int(cached_key_value.key.shape[2]),
                cached_segment_lengths=tuple(cached_key_value.metadata.get("segment_token_lengths", ())),
            )
        profile_attention_mask, profile_block_mask = select_attention_profile_mask(
            attention_profile,
            device=query.device,
            prefer_flex=(
                attention_mask is None
                and cached_key_value is None
                and cached_prefix_visibility is None
                and cache_current_token_count == 0
                and cache_current_token_span is None
                and kv_cache_override is None
            ),
            is_cross_attention=is_cross_attention,
        )
        resolved_attention_mask = attention_mask if attention_mask is not None else profile_attention_mask
        if use_slot_pool_backend and key.shape[2] > query.shape[2]:
            prefix_len = int(key.shape[2] - query.shape[2])
            prefix_visibility_mode = (
                str(cache_backend_state.metadata.get("prefix_visibility_mode", "full_history"))
                if cache_backend_state is not None
                else "full_history"
            )
            if resolved_attention_mask is None and prefix_visibility_mode != "full_history":
                query_len = int(query.shape[2])
                resolved_attention_mask = torch.ones(
                    query_len,
                    query_len,
                    device=query.device,
                    dtype=torch.bool,
                )
            if resolved_attention_mask is not None:
                resolved_attention_mask = _resolve_slot_pool_prefix_visibility(
                    resolved_attention_mask,
                    prefix_len=prefix_len,
                    prefix_visibility_mode=prefix_visibility_mode,
                    query_stream_ids=cache_backend_stream_ids,
                    cached_prefix_stream_ids=(
                        valid_stream_ids[:prefix_len]
                        if valid_stream_ids is not None
                        else None
                    ),
                    query_sequence_ids=query_sequence_ids,
                    cached_prefix_sequence_ids=cached_prefix_sequence_ids,
                    allow_video_query_to_action_prefix_tail_tokens=int(
                        cache_backend_state.metadata.get(
                            SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
                            0,
                        )
                    )
                    if cache_backend_state is not None
                    else 0,
                )
                profile_block_mask = None
        sdpa_mask = _prepare_sdpa_mask(resolved_attention_mask, device=query.device)
        hidden_states = apply_attention_backend(
            query=query,
            key=key,
            value=value,
            attention_mask=sdpa_mask,
            block_mask=profile_block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            }
            if profile_block_mask is not None
            else None,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = _linear_with_materialized_params(self.to_out[0], hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        if (
            use_slot_pool_backend
            and kv_cache_override is None
            and cache_backend_update_mode != 0
            and slot_pool_update_key is not None
            and slot_pool_update_value is not None
        ):
            if int(slot_pool_update_key.shape[0]) != int(cache_backend_state.key.shape[0]):
                raise ValueError(
                    "Cannot persist batch-packed current K/V into a slot-pool cache with a different batch size; "
                    f"got current_batch={int(slot_pool_update_key.shape[0])}, "
                    f"cache_batch={int(cache_backend_state.key.shape[0])}."
                )
            update_slot_pool_layer_state(
                cache_backend_state,
                key=slot_pool_update_key,
                value=slot_pool_update_value,
                is_pred=cache_backend_update_mode == 1,
                stream_ids=slot_pool_update_stream_ids,
            )
        return hidden_states, current_cache_entry


class SharedTransformerBlock(nn.Module):
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
        self.attn1 = SharedTransformerAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
        )
        self.attn2 = SharedTransformerAttention(
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

    def prepare_self_attention_inputs(
        self,
        hidden_states: torch.Tensor,
        *,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Build self-attention Q/K/V plus post-attention modulation state.

        This helper is used by method-5 MoT runtime paths that need to mix
        cached video K/V with action K/V without changing the existing block
        `forward()` contract used by other policy families.
        """

        temb_scale_shift_table = _materialize_runtime_parameter(
            self.scale_shift_table,
            device=temb.device,
            dtype=temb.dtype,
        )[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = _select_chunk_slices(
            temb_scale_shift_table,
            6,
        )
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        query = _rms_norm_with_materialized_weight(
            self.attn1.norm_q,
            _linear_with_materialized_params(self.attn1.to_q, norm_hidden_states),
        ).unflatten(2, (self.attn1.heads, -1))
        key = _rms_norm_with_materialized_weight(
            self.attn1.norm_k,
            _linear_with_materialized_params(self.attn1.to_k, norm_hidden_states),
        ).unflatten(2, (self.attn1.heads, -1))
        value = _linear_with_materialized_params(self.attn1.to_v, norm_hidden_states).unflatten(
            2,
            (self.attn1.heads, -1),
        )
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
        return {
            "query": query.transpose(1, 2).contiguous(),
            "key": key.transpose(1, 2).contiguous(),
            "value": value.transpose(1, 2).contiguous(),
            "gate_msa": gate_msa,
            "c_shift_msa": c_shift_msa,
            "c_scale_msa": c_scale_msa,
            "c_gate_msa": c_gate_msa,
            "hidden_states": hidden_states,
        }

    def apply_post_attention(
        self,
        hidden_states: torch.Tensor,
        *,
        mixed_attn_output: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        gate_msa: torch.Tensor,
        c_shift_msa: torch.Tensor,
        c_scale_msa: torch.Tensor,
        c_gate_msa: torch.Tensor,
        attention_profile: PreparedAttentionProfile | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        cross_attention_cache_entry: AttentionCacheEntry | None = None,
    ) -> tuple[torch.Tensor, AttentionCacheEntry | None]:
        """Apply residual, cross-attention, and FFN after external self-attn."""

        hidden_states = (hidden_states.float() + mixed_attn_output.float() * gate_msa).type_as(hidden_states)
        norm_hidden_states = (
            _layer_norm_with_materialized_params(self.norm2, hidden_states.float())
            if isinstance(self.norm2, nn.LayerNorm)
            else self.norm2(hidden_states.float())
        ).type_as(hidden_states)
        attn_output, cross_cache_entry = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            rotary_emb=None,
            attention_mask=cross_attention_mask,
            attention_profile=attention_profile,
            is_cross_attention=True,
            kv_cache_override=cross_attention_cache_entry,
            cache_current_token_count=encoder_hidden_states.shape[1] if cross_attention_cache_entry is None else 0,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = (
            _layer_norm_with_materialized_params(self.norm3, hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = _feed_forward_with_materialized_params(self.ffn, norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states, cross_cache_entry

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        structured_attention_context: StructuredAttentionContext | None = None,
        structured_block_semantics: StructuredBlockSemantics | None = None,
        structured_frequency_bundle: StructuredFrequencyBundle | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_profile: PreparedAttentionProfile | None = None,
        cross_attention_mask: torch.Tensor | None = None,
        self_attention_cache_entry: AttentionCacheEntry | None = None,
        cross_attention_cache_entry: AttentionCacheEntry | None = None,
        cached_prefix_visibility: torch.Tensor | None = None,
        cache_current_token_count: int = 0,
        cache_current_token_span: tuple[int, int] | None = None,
        detach_self_attention_cache: bool = True,
        self_attention_cache_backend_name: str | None = None,
        self_attention_cache_backend_state=None,
        self_attention_cache_update_mode: int = 0,
        self_attention_cache_stream_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AttentionCacheEntry | None, AttentionCacheEntry | None]:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = _select_chunk_slices(
            temb_scale_shift_table,
            6,
        )

        structured_attention_plan = build_structured_attention_execution_plan(
            structured_attention_context,
            batch_size=hidden_states.shape[0],
            device=hidden_states.device,
            cached_prefix_len=(
                int(self_attention_cache_entry.key.shape[2])
                if self_attention_cache_entry is not None and self_attention_cache_entry.key is not None
                else 0
            ),
            cached_segment_lengths=(
                tuple(self_attention_cache_entry.metadata.get("segment_token_lengths", ()))
                if self_attention_cache_entry is not None and self_attention_cache_entry.key is not None
                else ()
            ),
        )
        resolved_attention_mask = (
            structured_attention_plan.attention_mask
            if structured_attention_plan is not None and structured_attention_plan.attention_mask is not None
            else attention_mask
        )
        resolved_cached_prefix_visibility = (
            structured_attention_plan.cached_prefix_visibility
            if structured_attention_plan is not None and structured_attention_plan.cached_prefix_visibility is not None
            else cached_prefix_visibility
        )

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output, self_cache_entry = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb=rotary_emb,
            structured_attention_context=structured_attention_context,
            structured_attention_plan=structured_attention_plan,
            attention_mask=resolved_attention_mask,
            attention_profile=attention_profile,
            is_cross_attention=False,
            cached_key_value=self_attention_cache_entry,
            cached_prefix_visibility=resolved_cached_prefix_visibility,
            cache_current_token_count=cache_current_token_count,
            cache_current_token_span=cache_current_token_span,
            detach_cache_entry=detach_self_attention_cache,
            cache_backend_name=self_attention_cache_backend_name,
            cache_backend_state=self_attention_cache_backend_state,
            cache_backend_update_mode=self_attention_cache_update_mode,
            cache_backend_stream_ids=self_attention_cache_stream_ids,
        )
        hidden_states = (hidden_states.float() + attn_output.float() * gate_msa).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output, cross_cache_entry = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            rotary_emb=None,
            attention_mask=cross_attention_mask,
            attention_profile=attention_profile,
            is_cross_attention=True,
            kv_cache_override=cross_attention_cache_entry,
            cache_current_token_count=encoder_hidden_states.shape[1] if cross_attention_cache_entry is None else 0,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        del structured_attention_context, structured_attention_plan, structured_block_semantics, structured_frequency_bundle
        return hidden_states, self_cache_entry, cross_cache_entry


def _materialize_runtime_parameter(
    parameter: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a dense tensor for helper paths that bypass FSDP pre-forward hooks."""

    if hasattr(parameter, "full_tensor"):
        return parameter.full_tensor().to(device=device, dtype=dtype)
    return parameter.to(device=device, dtype=dtype)


def _linear_with_materialized_params(
    linear: nn.Linear,
    inputs: torch.Tensor,
) -> torch.Tensor:
    weight = _materialize_runtime_parameter(
        linear.weight,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    bias = None
    if linear.bias is not None:
        bias = _materialize_runtime_parameter(
            linear.bias,
            device=inputs.device,
            dtype=inputs.dtype,
        )
    return F.linear(inputs, weight, bias)


def _rms_norm_with_materialized_weight(
    norm: nn.RMSNorm,
    inputs: torch.Tensor,
) -> torch.Tensor:
    weight = None
    if norm.weight is not None:
        weight = _materialize_runtime_parameter(
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


def _layer_norm_with_materialized_params(
    norm: nn.LayerNorm,
    inputs: torch.Tensor,
) -> torch.Tensor:
    weight = None
    bias = None
    if getattr(norm, "weight", None) is not None:
        weight = _materialize_runtime_parameter(
            norm.weight,
            device=inputs.device,
            dtype=inputs.dtype,
        )
    if getattr(norm, "bias", None) is not None:
        bias = _materialize_runtime_parameter(
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


def _feed_forward_with_materialized_params(
    ffn: FeedForward,
    inputs: torch.Tensor,
) -> torch.Tensor:
    if len(ffn.net) != 3:
        raise ValueError(f"Unsupported FeedForward layout for materialized helper: {ffn.net!r}")
    act = ffn.net[0]
    dropout = ffn.net[1]
    proj_out = ffn.net[2]
    if not hasattr(act, "proj"):
        raise ValueError(f"Unsupported FeedForward activation module for materialized helper: {act!r}")
    hidden = _linear_with_materialized_params(act.proj, inputs)
    hidden = F.gelu(hidden, approximate="tanh")
    hidden = dropout(hidden)
    return _linear_with_materialized_params(proj_out, hidden)


class ProprioContextEncoder(nn.Module):
    """Deprecated adapter that projects proprio state into text-context space."""

    def __init__(self, state_dim: int, text_dim: int) -> None:
        super().__init__()
        state_dim = int(state_dim)
        text_dim = int(text_dim)
        if state_dim <= 0:
            raise ValueError(f"Expected positive proprio state_dim, got {state_dim}.")
        if text_dim <= 0:
            raise ValueError(f"Expected positive text_dim, got {text_dim}.")
        self.state_dim = state_dim
        self.text_dim = text_dim
        self.proj = nn.Linear(state_dim, text_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, proprio_state: torch.Tensor) -> torch.Tensor:
        if proprio_state.ndim != 2:
            raise ValueError(
                "Proprio context encoder expects anchor state with shape [B, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        if int(proprio_state.shape[-1]) != self.state_dim:
            raise ValueError(
                "Proprio state dim mismatch for context encoder, "
                f"got {proprio_state.shape[-1]} and expected {self.state_dim}."
            )
        return self.proj(proprio_state)


class ProprioHiddenContextEncoder(nn.Module):
    """Project proprio state into additive transformer hidden context."""

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        state_dim = int(state_dim)
        hidden_size = int(hidden_size)
        if state_dim <= 0:
            raise ValueError(f"Expected positive proprio state_dim, got {state_dim}.")
        if hidden_size <= 0:
            raise ValueError(f"Expected positive hidden_size, got {hidden_size}.")
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.proj = nn.Linear(state_dim, hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, proprio_state: torch.Tensor) -> torch.Tensor:
        if proprio_state.ndim != 2:
            raise ValueError(
                "Proprio hidden context encoder expects state with shape [B, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        if int(proprio_state.shape[-1]) != self.state_dim:
            raise ValueError(
                "Proprio hidden state dim mismatch, "
                f"got {proprio_state.shape[-1]} and expected {self.state_dim}."
            )
        return self.proj(proprio_state)


class GeneralistModeContextEncoder(nn.Module):
    """Learned text-space control token for GJD conditioning mode."""

    MODE_TO_INDEX = {
        "joint": 0,
        "action_conditioned_video": 1,
        "video_conditioned_action": 2,
    }

    def __init__(self, text_dim: int) -> None:
        super().__init__()
        text_dim = int(text_dim)
        if text_dim <= 0:
            raise ValueError(f"Expected positive text_dim, got {text_dim}.")
        self.text_dim = text_dim
        self.embedding = nn.Embedding(len(self.MODE_TO_INDEX), text_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    @classmethod
    def _index_for_mode(cls, mode: object) -> int:
        key = str(getattr(mode, "value", mode))
        try:
            return cls.MODE_TO_INDEX[key]
        except KeyError as exc:
            supported = ", ".join(sorted(cls.MODE_TO_INDEX))
            raise ValueError(f"Unsupported generalist mode {key!r}. Supported modes: {supported}.") from exc

    def _indices_for_modes(self, modes: object, *, batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(modes, torch.Tensor):
            indices = modes.to(device=device, dtype=torch.long).reshape(-1)
            if int(indices.numel()) > 0:
                min_index = int(indices.min().item())
                max_index = int(indices.max().item())
                if min_index < 0 or max_index >= len(self.MODE_TO_INDEX):
                    raise ValueError(
                        "Generalist mode tensor indices must be in "
                        f"[0, {len(self.MODE_TO_INDEX) - 1}], got min={min_index}, max={max_index}."
                    )
        elif isinstance(modes, str):
            index = self._index_for_mode(modes)
            indices = torch.full((batch_size,), index, device=device, dtype=torch.long)
        elif isinstance(modes, (list, tuple)):
            resolved = [self._index_for_mode(mode) for mode in modes]
            indices = torch.tensor(resolved, device=device, dtype=torch.long)
        else:
            index = self._index_for_mode(modes)
            indices = torch.full((batch_size,), index, device=device, dtype=torch.long)
        if int(indices.numel()) == 1 and batch_size != 1:
            indices = indices.expand(batch_size)
        if int(indices.numel()) != int(batch_size):
            raise ValueError(
                "Generalist mode token count must match text batch size, "
                f"got modes={int(indices.numel())} and batch={batch_size}."
            )
        return indices

    def forward(self, modes: object, *, batch_size: int) -> torch.Tensor:
        indices = self._indices_for_modes(
            modes,
            batch_size=int(batch_size),
            device=self.embedding.weight.device,
        )
        return self.embedding(indices)


class SharedVideoTransformerCore(nn.Module):
    """Shared Wan-style transformer core for all policy variants."""

    def __init__(
        self,
        config: SharedVideoTransformerConfig | None = None,
        *,
        action_dim: int | None = None,
        state_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SharedVideoTransformerConfig()
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"Expected hidden_size {self.config.hidden_size} to be divisible by num_heads {self.config.num_heads}."
            )
        self.action_dim = int(action_dim or 0)
        self.state_dim = int(state_dim or 0)
        self.inner_dim = self.config.hidden_size
        self.ffn_dim = self.config.ffn_dim or (self.config.hidden_size * self.config.mlp_ratio)
        self.patch_size = (
            self.config.patch_size_t,
            self.config.patch_size_h,
            self.config.patch_size_w,
        )
        self.rope = SharedTransformerRotaryPositionalEmbedding(self.config.hidden_size // self.config.num_heads)
        self.time_conditioner = SharedTransformerTimeEmbedding(self.config.hidden_size, self.config.freq_dim)
        self.action_time_conditioner = SharedTransformerTimeEmbedding(self.config.hidden_size, self.config.freq_dim)
        self.text_proj = PixArtAlphaTextProjection(self.config.text_dim, self.config.hidden_size, act_fn="gelu_tanh")
        self.action_text_proj = PixArtAlphaTextProjection(self.config.text_dim, self.config.hidden_size, act_fn="gelu_tanh")
        self.proprio_context_encoder: ProprioContextEncoder | None = None
        self.proprio_hidden_context_encoder: ProprioHiddenContextEncoder | None = None
        self.generalist_mode_context_encoder: GeneralistModeContextEncoder | None = None
        self.patch_embedding_mlp = nn.Linear(
            self.config.latent_channels * self.config.patch_size_t * self.config.patch_size_h * self.config.patch_size_w,
            self.config.hidden_size,
        )
        self.action_embedder = nn.Linear(max(self.action_dim, 1), self.config.hidden_size)
        self.runtime_stream_adapters = SharedRuntimeStreamAdapters(
            hidden_size=self.config.hidden_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )
        self.blocks = nn.ModuleList(
            [
                SharedTransformerBlock(
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
        self.proj_out = nn.Linear(
            self.config.hidden_size,
            self.config.latent_channels * self.config.patch_size_t * self.config.patch_size_h * self.config.patch_size_w,
        )
        self.action_proj_out = nn.Linear(self.config.hidden_size, max(self.action_dim, 1))
        self._exact_runtime_caches: dict[str, CacheState] = {}
        self._runtime_block_devices: tuple[torch.device, ...] = tuple()

    def configure_runtime_block_devices(
        self,
        devices: tuple[torch.device, ...],
        *,
        prep_device: torch.device | None = None,
        output_device: torch.device | None = None,
    ) -> None:
        if not devices:
            self._runtime_block_devices = tuple()
            return
        normalized = tuple(torch.device(device) for device in devices)
        self._runtime_block_devices = normalized
        input_device = torch.device(prep_device) if prep_device is not None else normalized[0]
        output_device = torch.device(output_device) if output_device is not None else normalized[-1]

        for module in (
            self.patch_embedding_mlp,
            self.action_embedder,
            self.time_conditioner,
            self.action_time_conditioner,
            self.text_proj,
            self.action_text_proj,
            self.proprio_context_encoder,
            self.proprio_hidden_context_encoder,
            self.generalist_mode_context_encoder,
            self.runtime_stream_adapters,
            self.rope,
        ):
            if module is None:
                continue
            module.to(device=input_device)

        for layer_index, block in enumerate(self.blocks):
            block.to(device=normalized[layer_index % len(normalized)])

        self.norm_out.to(device=output_device)
        self.proj_out.to(device=output_device)
        self.action_proj_out.to(device=output_device)
        self.scale_shift_table.data = self.scale_shift_table.data.to(device=output_device)

    def configure_proprio_context_encoder(self, *, enabled: bool, state_dim: int | None = None) -> None:
        if not enabled:
            self.proprio_context_encoder = None
            return
        resolved_state_dim = int(self.state_dim if state_dim is None else state_dim)
        if resolved_state_dim <= 0:
            raise ValueError("Proprio context mode requires a positive visual-tower state_dim.")
        if (
            self.proprio_context_encoder is not None
            and self.proprio_context_encoder.state_dim == resolved_state_dim
            and self.proprio_context_encoder.text_dim == self.config.text_dim
        ):
            return
        self.proprio_context_encoder = ProprioContextEncoder(
            state_dim=resolved_state_dim,
            text_dim=self.config.text_dim,
        )

    def configure_proprio_hidden_context_encoder(self, *, enabled: bool, state_dim: int | None = None) -> None:
        if not enabled:
            self.proprio_hidden_context_encoder = None
            return
        resolved_state_dim = int(self.state_dim if state_dim is None else state_dim)
        if resolved_state_dim <= 0:
            raise ValueError("Per-chunk proprio context mode requires a positive visual-tower state_dim.")
        if (
            self.proprio_hidden_context_encoder is not None
            and self.proprio_hidden_context_encoder.state_dim == resolved_state_dim
            and self.proprio_hidden_context_encoder.hidden_size == self.config.hidden_size
        ):
            return
        self.proprio_hidden_context_encoder = ProprioHiddenContextEncoder(
            state_dim=resolved_state_dim,
            hidden_size=self.config.hidden_size,
        )

    def configure_generalist_mode_context_encoder(self, *, enabled: bool) -> None:
        if not enabled:
            self.generalist_mode_context_encoder = None
            return
        if (
            self.generalist_mode_context_encoder is not None
            and self.generalist_mode_context_encoder.text_dim == self.config.text_dim
        ):
            return
        self.generalist_mode_context_encoder = GeneralistModeContextEncoder(text_dim=self.config.text_dim)

    def append_generalist_mode_context_token(
        self,
        text_emb: torch.Tensor,
        mode: object | None,
    ) -> torch.Tensor:
        if mode is None or self.generalist_mode_context_encoder is None:
            return text_emb
        if text_emb.ndim != 3:
            raise ValueError(
                "Generalist mode token appending expects text embeddings with shape [B, tokens, dim], "
                f"got {tuple(text_emb.shape)}."
            )
        if int(text_emb.shape[-1]) != int(self.config.text_dim):
            raise ValueError(
                "Text embedding dim mismatch for generalist mode appending, "
                f"got {text_emb.shape[-1]} and expected {self.config.text_dim}."
            )
        encoder = self.generalist_mode_context_encoder
        mode_tokens = encoder(mode, batch_size=int(text_emb.shape[0])).to(
            device=text_emb.device,
            dtype=text_emb.dtype,
        )
        return torch.cat([text_emb, mode_tokens[:, None, :]], dim=1)

    def append_proprio_context_tokens(
        self,
        text_emb: torch.Tensor,
        proprio_state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Deprecated text-space proprio token path; use hidden additive context for new runs."""

        if proprio_state is None or self.proprio_context_encoder is None:
            return text_emb
        if text_emb.ndim != 3:
            raise ValueError(
                "Proprio context appending expects text embeddings with shape [B, tokens, dim], "
                f"got {tuple(text_emb.shape)}."
            )
        if int(text_emb.shape[-1]) != int(self.config.text_dim):
            raise ValueError(
                "Text embedding dim mismatch for proprio appending, "
                f"got {text_emb.shape[-1]} and expected {self.config.text_dim}."
            )
        if proprio_state.ndim == 2:
            proprio_state = proprio_state[:, None, :]
        if proprio_state.ndim != 3:
            raise ValueError(
                "Proprio context appending expects state with shape [B, state_dim] or [B, chunks, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        if int(proprio_state.shape[0]) != int(text_emb.shape[0]):
            raise ValueError(
                "Proprio/text batch mismatch, "
                f"got proprio batch {proprio_state.shape[0]} and text batch {text_emb.shape[0]}."
            )
        encoder = self.proprio_context_encoder
        batch_size, chunk_count, state_dim = proprio_state.shape
        proprio_state = proprio_state.to(device=encoder.proj.weight.device, dtype=encoder.proj.weight.dtype)
        proprio_tokens = encoder(proprio_state.reshape(batch_size * chunk_count, state_dim))
        proprio_tokens = proprio_tokens.reshape(batch_size, chunk_count, -1).to(
            device=text_emb.device,
            dtype=text_emb.dtype,
        )
        return torch.cat([text_emb, proprio_tokens], dim=1)

    def encode_proprio_hidden_context(
        self,
        proprio_state: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        encoder = self.proprio_hidden_context_encoder
        if encoder is None:
            raise ValueError("Per-chunk proprio context mode requires a configured hidden-context encoder.")
        if proprio_state.ndim == 2:
            proprio_state = proprio_state[:, None, :]
        if proprio_state.ndim != 3:
            raise ValueError(
                "Per-chunk proprio hidden context expects state with shape [B, F, state_dim] or [B, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        batch_size, frame_count, state_dim = proprio_state.shape
        proprio_state = proprio_state.to(device=encoder.proj.weight.device, dtype=encoder.proj.weight.dtype)
        hidden_context = encoder(proprio_state.reshape(batch_size * frame_count, state_dim))
        return hidden_context.reshape(batch_size, frame_count, -1).to(device=device, dtype=dtype)

    @staticmethod
    def _move_optional_tensor(tensor: torch.Tensor | None, *, device: torch.device, dtype: torch.dtype | None = None):
        if tensor is None:
            return None
        kwargs = {"device": device}
        if dtype is not None and tensor.is_floating_point():
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)

    @classmethod
    def _cached_optional_tensor(
        cls,
        tensor: torch.Tensor | None,
        *,
        cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor],
        name: str,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        dtype_key = dtype if dtype is not None and tensor.is_floating_point() else None
        cache_key = (name, torch.device(device), dtype_key)
        cached = cache.get(cache_key)
        if cached is None:
            cached = cls._move_optional_tensor(tensor, device=torch.device(device), dtype=dtype)
            cache[cache_key] = cached
        return cached

    def _move_attention_profile(
        self,
        profile: PreparedAttentionProfile | None,
        *,
        device: torch.device,
    ) -> PreparedAttentionProfile | None:
        if profile is None:
            return None
        device = torch.device(device)
        has_block_masks = (
            profile.self_attention_block_mask is not None
            or profile.cross_attention_block_mask is not None
        )
        if has_block_masks:
            metadata = profile.metadata
            required_keys = (
                "latent_shape",
                "action_shape",
                "padded_length",
                "chunk_size",
                "window_size",
                "text_token_count",
            )
            if all(key in metadata for key in required_keys) and (
                "current_block_coupling" in metadata
                or "allow_joint_noisy_block_attention" in metadata
            ):
                current_block_coupling = metadata.get("current_block_coupling")
                if current_block_coupling is None:
                    current_block_coupling = (
                        "joint"
                        if bool(metadata["allow_joint_noisy_block_attention"])
                        else "video_then_action"
                    )
                return build_chunked_temporal_exact_attention_profile(
                    latent_shape=tuple(int(v) for v in metadata["latent_shape"]),
                    action_shape=tuple(int(v) for v in metadata["action_shape"]),
                    padded_length=int(metadata["padded_length"]),
                    chunk_size=int(metadata["chunk_size"]),
                    window_size=int(metadata["window_size"]),
                    patch_size=self.patch_size,
                    text_token_count=int(metadata["text_token_count"]),
                    base_text_token_count=(
                        None
                        if "base_text_token_count" not in metadata
                        else int(metadata["base_text_token_count"])
                    ),
                    proprio_context_token_count=int(metadata.get("proprio_context_token_count", 0)),
                    chunk_origin_frame=int(metadata.get("chunk_origin_frame", 0)),
                    prefix_condition_frames=int(metadata.get("prefix_condition_frames", 0)),
                    action_context_mask=(
                        torch.tensor(
                            metadata["action_context_valid_tokens"],
                            device=device,
                            dtype=torch.bool,
                        )[None, :]
                        if metadata.get("action_context_valid_tokens") is not None
                        else None
                    ),
                    device=device,
                    build_dense_masks=(
                        profile.self_attention_mask is not None
                        or profile.cross_attention_mask is not None
                    ),
                    build_flex_masks=True,
                    current_block_coupling=str(current_block_coupling),
                    preserve_video_pretrain_history=bool(
                        metadata.get("preserve_video_pretrain_history", False)
                    ),
                    history_stream_visibility=metadata.get("history_stream_visibility"),
                )
            if profile.self_attention_mask is None and profile.cross_attention_mask is None:
                return profile
            return replace(
                profile,
                self_attention_mask=self._move_optional_tensor(profile.self_attention_mask, device=device),
                cross_attention_mask=self._move_optional_tensor(profile.cross_attention_mask, device=device),
                self_attention_block_mask=None,
                cross_attention_block_mask=None,
            )
        return replace(
            profile,
            self_attention_mask=self._move_optional_tensor(profile.self_attention_mask, device=device),
            cross_attention_mask=self._move_optional_tensor(profile.cross_attention_mask, device=device),
        )

    def _cached_attention_profile(
        self,
        profile: PreparedAttentionProfile | None,
        *,
        cache: dict[torch.device, PreparedAttentionProfile | None],
        device: torch.device,
    ) -> PreparedAttentionProfile | None:
        device = torch.device(device)
        if device not in cache:
            cache[device] = self._move_attention_profile(profile, device=device)
        return cache[device]

    @staticmethod
    def _move_slot_pool_layer_state(
        layer_state: SlotPoolLayerState | None,
        *,
        device: torch.device,
    ) -> SlotPoolLayerState | None:
        if layer_state is None:
            return None
        for name in ("key", "value", "slot_ids", "stream_ids", "slot_mask", "prediction_mask"):
            tensor = getattr(layer_state, name)
            if tensor is not None and tensor.device != device:
                setattr(layer_state, name, tensor.to(device=device))
        return layer_state

    def _move_structured_attention_context(
        self,
        context: StructuredAttentionContext | None,
        *,
        device: torch.device,
    ) -> StructuredAttentionContext | None:
        if context is None:
            return None
        return replace(
            context,
            clean_prefix_grid_ids=self._move_optional_tensor(context.clean_prefix_grid_ids, device=device),
            video_grid_ids=self._move_optional_tensor(context.video_grid_ids, device=device),
            action_grid_ids=self._move_optional_tensor(context.action_grid_ids, device=device),
            state_grid_ids=self._move_optional_tensor(context.state_grid_ids, device=device),
            clean_prefix_freqs=self._move_optional_tensor(context.clean_prefix_freqs, device=device),
            video_freqs=self._move_optional_tensor(context.video_freqs, device=device),
            action_freqs=self._move_optional_tensor(context.action_freqs, device=device),
            state_freqs=self._move_optional_tensor(context.state_freqs, device=device),
        )

    def _move_structured_frequency_bundle(
        self,
        bundle: StructuredFrequencyBundle | None,
        *,
        device: torch.device,
    ) -> StructuredFrequencyBundle | None:
        if bundle is None:
            return None
        return replace(
            bundle,
            clean_prefix_grid_ids=self._move_optional_tensor(bundle.clean_prefix_grid_ids, device=device),
            video_grid_ids=self._move_optional_tensor(bundle.video_grid_ids, device=device),
            action_grid_ids=self._move_optional_tensor(bundle.action_grid_ids, device=device),
            state_grid_ids=self._move_optional_tensor(bundle.state_grid_ids, device=device),
            shared_grid_ids=self._move_optional_tensor(bundle.shared_grid_ids, device=device),
        )

    def prepare_runtime_stream_inputs(
        self,
        *,
        family: str,
        action_inputs: torch.Tensor | None,
        state_inputs: torch.Tensor | None,
        action_timesteps: torch.Tensor | None,
        state_timesteps: torch.Tensor | None,
        action_adapter_name: str = "mlp",
        state_adapter_name: str = "mlp",
        use_state_adapter: bool = True,
    ) -> dict[str, PreparedStreamInput]:
        adapter_device = self.runtime_stream_adapters.role_embedding.weight.device
        if action_inputs is not None and action_inputs.device != adapter_device:
            action_inputs = action_inputs.to(device=adapter_device)
        if state_inputs is not None and state_inputs.device != adapter_device:
            state_inputs = state_inputs.to(device=adapter_device)
        if action_timesteps is not None and action_timesteps.device != adapter_device:
            action_timesteps = action_timesteps.to(device=adapter_device)
        if state_timesteps is not None and state_timesteps.device != adapter_device:
            state_timesteps = state_timesteps.to(device=adapter_device)
        return self.runtime_stream_adapters.prepare_stream_inputs(
            family=family,
            action_inputs=action_inputs,
            state_inputs=state_inputs,
            action_timesteps=action_timesteps,
            state_timesteps=state_timesteps,
            action_adapter_name=action_adapter_name,
            state_adapter_name=state_adapter_name,
            use_state_adapter=use_state_adapter,
        )

    def project_runtime_stream_outputs(
        self,
        *,
        family: str,
        hidden_states: torch.Tensor,
        token_layout: object | None,
    ) -> dict[str, torch.Tensor]:
        return project_runtime_stream_outputs(
            family=family,
            hidden_states=hidden_states,
            token_layout=token_layout,
            video_projector=self.proj_out,
            action_projector=self.action_proj_out,
        )

    def project_video_tokens_to_latents(
        self,
        *,
        hidden_states: torch.Tensor,
        token_grid,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "Expected shared-core video tokens with shape [B, seq, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        video_patch_prediction = self.proj_out(hidden_states)
        return unpatchify_video_tokens(
            video_patch_prediction,
            token_grid=token_grid,
            latent_channels=self.config.latent_channels,
        )

    def _compose_structured_rotary_grid_ids(
        self,
        *,
        structured_block_semantics: StructuredBlockSemantics | None,
        structured_frequency_bundle: StructuredFrequencyBundle | None,
        fallback_grid_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if structured_block_semantics is None or structured_frequency_bundle is None:
            return (
                structured_frequency_bundle.shared_grid_ids
                if structured_frequency_bundle is not None and structured_frequency_bundle.shared_grid_ids is not None
                else fallback_grid_ids
            )

        frequency_chunks: list[torch.Tensor] = []
        if structured_block_semantics.clean_prefix_length > 0:
            clean_prefix_grid_ids = structured_frequency_bundle.clean_prefix_grid_ids
            if clean_prefix_grid_ids is None:
                raise ValueError(
                    "Structured block semantics requested an explicit clean-prefix span, but no "
                    "`clean_prefix_grid_ids` were provided."
                )
            if clean_prefix_grid_ids.shape[1] != structured_block_semantics.clean_prefix_length:
                raise ValueError(
                    "Structured clean-prefix frequency length mismatch: expected "
                    f"{structured_block_semantics.clean_prefix_length}, got {clean_prefix_grid_ids.shape[1]}."
                )
            frequency_chunks.append(clean_prefix_grid_ids)

        if structured_block_semantics.video_token_length > 0:
            video_grid_ids = structured_frequency_bundle.video_grid_ids
            if video_grid_ids is None:
                raise ValueError(
                    "Structured block semantics requested explicit video-token frequencies, but no "
                    "`video_grid_ids` were provided."
                )
            if video_grid_ids.shape[1] != structured_block_semantics.video_token_length:
                raise ValueError(
                    "Structured video frequency length mismatch: expected "
                    f"{structured_block_semantics.video_token_length}, got {video_grid_ids.shape[1]}."
                )
            frequency_chunks.append(video_grid_ids)

        if structured_block_semantics.action_register_length > 0:
            action_grid_ids = structured_frequency_bundle.action_grid_ids
            if action_grid_ids is None:
                raise ValueError(
                    "Structured block semantics requested explicit action-register frequencies, but no "
                    "`action_grid_ids` were provided."
                )
            if action_grid_ids.shape[1] != structured_block_semantics.action_register_length:
                raise ValueError(
                    "Structured action-register frequency length mismatch: expected "
                    f"{structured_block_semantics.action_register_length}, got {action_grid_ids.shape[1]}."
                )
            frequency_chunks.append(action_grid_ids)

        if structured_block_semantics.state_register_length > 0:
            state_grid_ids = structured_frequency_bundle.state_grid_ids
            if state_grid_ids is None:
                raise ValueError(
                    "Structured block semantics requested explicit state-register frequencies, but no "
                    "`state_grid_ids` were provided."
                )
            if state_grid_ids.shape[1] != structured_block_semantics.state_register_length:
                raise ValueError(
                    "Structured state-register frequency length mismatch: expected "
                    f"{structured_block_semantics.state_register_length}, got {state_grid_ids.shape[1]}."
                )
            frequency_chunks.append(state_grid_ids)

        if frequency_chunks:
            return torch.cat(frequency_chunks, dim=1)
        return (
            structured_frequency_bundle.shared_grid_ids
            if structured_frequency_bundle.shared_grid_ids is not None
            else fallback_grid_ids
        )

    def _resolve_structured_attention_context(
        self,
        core_input: VisualCoreInput,
        *,
        device: torch.device,
    ) -> StructuredAttentionContext | None:
        context = core_input.structured_attention_context
        if context is None and (
            core_input.structured_block_semantics is not None or core_input.structured_frequency_bundle is not None
        ):
            semantics = core_input.structured_block_semantics
            frequencies = core_input.structured_frequency_bundle
            if semantics is not None:
                context = StructuredAttentionContext(
                    mode=semantics.mode,
                    teacher_forcing_enabled=semantics.teacher_forcing_enabled,
                    clean_prefix_length=semantics.clean_prefix_length,
                    video_token_length=semantics.video_token_length,
                    action_register_length=semantics.action_register_length,
                    state_register_length=semantics.state_register_length,
                    current_start_frame=semantics.current_start_frame,
                    observed_prefix_frames=semantics.observed_prefix_frames,
                    num_frame_per_block=1,
                    num_action_per_block=0,
                    num_state_per_block=0,
                    num_video_blocks=0,
                    num_action_blocks=0,
                    num_state_blocks=0,
                    tokens_per_frame=0,
                    tokens_per_video_block=0,
                    frequency_mode=semantics.frequency_mode,
                    attention_kernel=semantics.metadata.get("attention_kernel", "mask_only"),
                    cache_kernel=semantics.metadata.get("cache_kernel", "prefix_mask_only"),
                    rollout_phase=semantics.metadata.get("rollout_phase", "teacher_forcing"),
                    action_state_index=int(semantics.metadata.get("action_state_index", 0)),
                    cached_video_tokens=int(semantics.metadata.get("cached_video_tokens", 0)),
                    cached_segment_lengths=tuple(semantics.metadata.get("cached_segment_lengths", ())),
                    clean_prefix_grid_ids=frequencies.clean_prefix_grid_ids if frequencies is not None else None,
                    video_grid_ids=frequencies.video_grid_ids if frequencies is not None else None,
                    action_grid_ids=frequencies.action_grid_ids if frequencies is not None else None,
                    state_grid_ids=frequencies.state_grid_ids if frequencies is not None else None,
                    metadata=dict(semantics.metadata),
                )
        if context is None or context.mode == "none":
            return context

        def _resolve_freq(grid_ids: torch.Tensor | None) -> torch.Tensor | None:
            if grid_ids is None:
                return None
            return self.rope(grid_ids.to(device=device))

        return StructuredAttentionContext(
            mode=context.mode,
            teacher_forcing_enabled=context.teacher_forcing_enabled,
            clean_prefix_length=context.clean_prefix_length,
            video_token_length=context.video_token_length,
            action_register_length=context.action_register_length,
            state_register_length=context.state_register_length,
            current_start_frame=context.current_start_frame,
            observed_prefix_frames=context.observed_prefix_frames,
            num_frame_per_block=context.num_frame_per_block,
            num_action_per_block=context.num_action_per_block,
            num_state_per_block=context.num_state_per_block,
            num_video_blocks=context.num_video_blocks,
            num_action_blocks=context.num_action_blocks,
            num_state_blocks=context.num_state_blocks,
            tokens_per_frame=context.tokens_per_frame,
            tokens_per_video_block=context.tokens_per_video_block,
            frequency_mode=context.frequency_mode,
            attention_kernel=context.attention_kernel,
            cache_kernel=context.cache_kernel,
            rollout_phase=context.rollout_phase,
            action_state_index=context.action_state_index,
            cached_video_tokens=context.cached_video_tokens,
            cached_segment_lengths=tuple(context.cached_segment_lengths),
            clean_prefix_grid_ids=context.clean_prefix_grid_ids,
            video_grid_ids=context.video_grid_ids,
            action_grid_ids=context.action_grid_ids,
            state_grid_ids=context.state_grid_ids,
            clean_prefix_freqs=(
                context.clean_prefix_freqs
                if context.clean_prefix_freqs is not None
                else _resolve_freq(context.clean_prefix_grid_ids)
            ),
            video_freqs=(
                context.video_freqs
                if context.video_freqs is not None
                else _resolve_freq(context.video_grid_ids)
            ),
            action_freqs=(
                context.action_freqs
                if context.action_freqs is not None
                else _resolve_freq(context.action_grid_ids)
            ),
            state_freqs=(
                context.state_freqs
                if context.state_freqs is not None
                else _resolve_freq(context.state_grid_ids)
            ),
            metadata=dict(context.metadata),
        )

    def _require_exact_action_dim(self) -> None:
        if self.action_dim <= 0:
            raise ValueError(
                "SharedVideoTransformerCore exact-runtime path requires a positive action_dim. "
                "Construct the shared VisualTower with the experiment action_dim."
            )

    def clear_cache(self, cache_name: str) -> None:
        self._exact_runtime_caches.pop(cache_name, None)

    def clear_pred_cache(self, cache_name: str) -> None:
        cache_state = self._exact_runtime_caches.get(cache_name)
        if cache_state is None:
            return
        cleared_backend = clear_cache_backend_payload(cache_state.backend_payload, clear_predictions_only=True)
        self._exact_runtime_caches[cache_name] = CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cleared_backend,
            payload=dict(cache_state.payload),
            self_attention_kv=materialize_cache_backend_entries(cleared_backend),
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
        )

    def clear_runtime_cache_state(self, cache_name: str) -> None:
        self.clear_cache(cache_name)

    def clear_runtime_prediction_cache(self, cache_name: str) -> None:
        self.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name: str,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        backend_name: str = "slot_pool_exact",
        prefix_visibility_mode: str = "full_history",
    ) -> None:
        total_tokens = int((attn_window // 2) * latent_token_per_chunk + (attn_window // 2) * action_token_per_chunk)
        backend_spec = resolve_cache_backend_spec(backend_name)
        backend_payload = init_cache_backend_payload(
            backend_spec.name,
            num_layers=len(self.blocks),
            total_tokens=total_tokens,
            num_heads=self.config.num_heads,
            head_dim=self.config.hidden_size // self.config.num_heads,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            metadata={
                "cache_name": cache_name,
                "attn_window": attn_window,
                "latent_token_per_chunk": latent_token_per_chunk,
                "action_token_per_chunk": action_token_per_chunk,
                "prefix_visibility_mode": prefix_visibility_mode,
            },
        )
        self._exact_runtime_caches[cache_name] = CacheState(
            supported=True,
            current_start_frame=0,
            cached_frames=0,
            chunk_size=attn_window,
            capability="self_attn_only",
            backend_name=backend_spec.name,
            backend_payload=backend_payload,
            payload={
                "cache_name": cache_name,
                "attn_window": attn_window,
                "latent_token_per_chunk": latent_token_per_chunk,
                "action_token_per_chunk": action_token_per_chunk,
                "max_tokens": total_tokens,
            },
            self_attention_kv=materialize_cache_backend_entries(backend_payload),
            cross_attention_kv=tuple(),
            update_metadata=CacheUpdateMetadata(),
        )

    def initialize_runtime_cache_backend(
        self,
        cache_name: str,
        *,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        backend_name: str = "slot_pool_exact",
        prefix_visibility_mode: str = "full_history",
    ) -> None:
        self.create_empty_cache(
            cache_name,
            attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            backend_name=backend_name,
            prefix_visibility_mode=prefix_visibility_mode,
        )

    def _resolve_exact_cache_state(self, cache_name: str) -> CacheState | None:
        return self._exact_runtime_caches.get(cache_name)

    def _exact_text_hidden_states(self, text_emb: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return self.text_proj(text_emb.clone()).to(dtype=dtype)

    def prepare_exact_single_stream_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        action_mode: bool,
    ) -> dict[str, torch.Tensor]:
        """Prepare exact-runtime embeddings without executing transformer blocks."""

        noisy_latents = input_dict["noisy_latents"]
        hidden_states = self._input_embed(noisy_latents, input_type="action" if action_mode else "latent")
        text_hidden_states = self._exact_text_hidden_states(input_dict["text_emb"], dtype=hidden_states.dtype)
        rotary_emb = self.rope(input_dict["grid_id"])[:, :, None]
        temb, timestep_proj = self._time_embed(
            input_dict["timesteps"],
            int(noisy_latents.shape[-2]),
            int(noisy_latents.shape[-1]),
            dtype=hidden_states.dtype,
            action_mode=action_mode,
        )
        return {
            "hidden_states": hidden_states,
            "text_hidden_states": text_hidden_states,
            "rotary_emb": rotary_emb,
            "temb": temb,
            "timestep_proj": timestep_proj,
        }

    def _input_embed(self, latents: torch.Tensor, input_type: str = "latent") -> torch.Tensor:
        if input_type == "latent":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            return self.patch_embedding_mlp(hidden_states.clone())
        if input_type == "action":
            self._require_exact_action_dim()
            hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
            return self.action_embedder(hidden_states.clone())
        if input_type == "text":
            return self.text_proj(latents.clone())
        raise ValueError(f"Unsupported input_type={input_type!r}")

    def _time_embed(
        self,
        timesteps: torch.Tensor,
        height: int,
        width: int,
        *,
        dtype: torch.dtype,
        action_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_scale_h, patch_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (height // patch_scale_h) * (width // patch_scale_w),
            dim=1,
        )
        conditioner = self.action_time_conditioner if action_mode else self.time_conditioner
        temb, timestep_proj = conditioner(latent_time_steps, dtype=dtype)
        return temb.contiguous().clone(), timestep_proj.contiguous().clone()

    def forward_train(self, input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        prepared = prepare_exact_dual_stream_train_sequence(
            input_dict,
            config=self.config,
            patch_size=self.patch_size,
            model_dtype=self.patch_embedding_mlp.weight.dtype,
            input_embed=lambda tensor, input_type: self._input_embed(tensor, input_type=input_type),
            exact_text_hidden_states=lambda text_emb: self._exact_text_hidden_states(text_emb, dtype=self.patch_embedding_mlp.weight.dtype),
            time_embed=lambda timesteps, height, width, dtype, action_mode: self._time_embed(
                timesteps,
                height,
                width,
                dtype=dtype,
                action_mode=action_mode,
            ),
            rope=self.rope,
        )
        batch_size = prepared.batch_size
        hidden_states = prepared.hidden_states
        text_hidden_states = prepared.text_hidden_states
        rotary_emb = prepared.rotary_emb
        temb = prepared.temb
        timestep_proj = prepared.timestep_proj
        split_list = prepared.split_list
        exact_attention_profile = prepared.attention_profile

        for block in self.blocks:
            hidden_states, _, _ = block(
                hidden_states,
                encoder_hidden_states=text_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
                attention_profile=exact_attention_profile,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = _select_split_segments(
            hidden_states,
            tuple(int(length) for length in split_list),
        )
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=batch_size,
        )
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(
            action_hidden_states,
            "1 (b l) c -> b l c",
            b=batch_size,
        )
        return latent_hidden_states, action_hidden_states

    def _forward_exact_single_stream(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        prepared = self.prepare_exact_single_stream_inputs(input_dict, action_mode=action_mode)
        hidden_states = prepared["hidden_states"]
        hidden_context = input_dict.get("hidden_context")
        if hidden_context is not None:
            if tuple(hidden_context.shape) != tuple(hidden_states.shape):
                raise ValueError(
                    "Exact single-stream hidden_context must match embedded hidden_states shape, "
                    f"got hidden_context={tuple(hidden_context.shape)}, hidden_states={tuple(hidden_states.shape)}."
                )
            hidden_states = hidden_states + hidden_context.to(device=hidden_states.device, dtype=hidden_states.dtype)
        text_hidden_states = prepared["text_hidden_states"]
        rotary_emb = prepared["rotary_emb"]
        temb = prepared["temb"]
        timestep_proj = prepared["timestep_proj"]
        cache_state = self._resolve_exact_cache_state(cache_name)
        cache_backend_name = cache_state.backend_name if cache_state is not None else None
        cache_backend_payload = cache_state.backend_payload if cache_state is not None else None
        detach_self_attention_cache = (
            bool(cache_state.payload.get("detach_self_attention_cache", True))
            if cache_state is not None
            else True
        )
        cache_current_token_count = 0
        if cache_state is not None and cache_state.update_metadata.update_kv_cache:
            # Exact single-stream cache writes are prefix-style: cache the visible
            # sequence being prefed unless the cache metadata narrows that span.
            tokens_per_frame = int(cache_state.payload.get("tokens_per_frame", 0))
            cached_frames = int(cache_state.cached_frames)
            if tokens_per_frame > 0 and cached_frames > 0:
                cache_current_token_count = tokens_per_frame * cached_frames
            else:
                cache_current_token_count = int(hidden_states.shape[1])
            cache_current_token_count = max(0, min(cache_current_token_count, int(hidden_states.shape[1])))
        next_self_attention_kv: list[AttentionCacheEntry] = []
        attention_mask = input_dict.get("attention_mask")
        cross_attention_mask = input_dict.get("cross_attention_mask")
        stream_id_value = 1 if action_mode else 0
        cache_backend_stream_ids = torch.full(
            (int(hidden_states.shape[1]),),
            stream_id_value,
            device=hidden_states.device,
            dtype=torch.long,
        )
        moved_tensor_cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor] = {}

        for layer_index, block in enumerate(self.blocks):
            block_device = (
                self._runtime_block_devices[layer_index % len(self._runtime_block_devices)]
                if self._runtime_block_devices
                else hidden_states.device
            )
            if hidden_states.device != block_device:
                hidden_states = hidden_states.to(device=block_device)
            block_text_hidden_states = self._cached_optional_tensor(
                text_hidden_states,
                cache=moved_tensor_cache,
                name="text_hidden_states",
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_timestep_proj = self._cached_optional_tensor(
                timestep_proj,
                cache=moved_tensor_cache,
                name="timestep_proj",
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_rotary_emb = self._cached_optional_tensor(
                rotary_emb,
                cache=moved_tensor_cache,
                name="rotary_emb",
                device=block_device,
            )
            block_attention_mask = self._cached_optional_tensor(
                attention_mask,
                cache=moved_tensor_cache,
                name="attention_mask",
                device=block_device,
            )
            block_cross_attention_mask = self._cached_optional_tensor(
                cross_attention_mask,
                cache=moved_tensor_cache,
                name="cross_attention_mask",
                device=block_device,
            )
            block_cache_backend_stream_ids = self._cached_optional_tensor(
                cache_backend_stream_ids,
                cache=moved_tensor_cache,
                name="cache_backend_stream_ids",
                device=block_device,
            )
            block_cache_backend_state = (
                cache_backend_payload.layer_states[layer_index]
                if cache_backend_uses_slot_pool(cache_backend_name)
                and cache_backend_payload is not None
                and layer_index < len(cache_backend_payload.layer_states)
                else None
            )
            block_cache_backend_state = self._move_slot_pool_layer_state(block_cache_backend_state, device=block_device)
            hidden_states, current_self_cache_entry, _ = block(
                hidden_states,
                encoder_hidden_states=block_text_hidden_states,
                temb=block_timestep_proj,
                rotary_emb=block_rotary_emb,
                attention_mask=block_attention_mask,
                cross_attention_mask=block_cross_attention_mask,
                self_attention_cache_backend_name=cache_backend_name,
                self_attention_cache_backend_state=block_cache_backend_state,
                cache_current_token_count=cache_current_token_count,
                detach_self_attention_cache=detach_self_attention_cache,
                self_attention_cache_update_mode=update_cache,
                self_attention_cache_stream_ids=block_cache_backend_stream_ids,
            )
            next_self_attention_kv.append(current_self_cache_entry or AttentionCacheEntry())

        output_device = self.scale_shift_table.device
        if hidden_states.device != output_device:
            hidden_states = hidden_states.to(device=output_device)
        temb = temb.to(device=output_device, dtype=hidden_states.dtype)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)

        if cache_state is not None:
            materialized_entries = (
                materialize_cache_backend_entries(cache_backend_payload)
                if cache_backend_uses_slot_pool(cache_backend_name)
                else tuple(next_self_attention_kv)
            )
            self._exact_runtime_caches[cache_name] = CacheState(
                supported=cache_state.supported,
                current_start_frame=cache_state.current_start_frame,
                cached_frames=cache_state.cached_frames,
                chunk_size=cache_state.chunk_size,
                capability=cache_state.capability,
                backend_name=cache_state.backend_name,
                backend_payload=cache_backend_payload,
                payload=dict(cache_state.payload),
                self_attention_kv=materialized_entries,
                cross_attention_kv=cache_state.cross_attention_kv,
                update_metadata=cache_state.update_metadata,
            )

        if action_mode:
            return self.action_proj_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return rearrange(hidden_states, "b l (n c) -> b (l n) c", n=math.prod(self.patch_size))

    def _forward_exact_dual_stream(
        self,
        prepared: PreparedExactTrainSequence,
        *,
        update_cache: int,
        cache_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = prepared.hidden_states
        text_hidden_states = prepared.text_hidden_states
        rotary_emb = prepared.rotary_emb
        temb = prepared.temb
        timestep_proj = prepared.timestep_proj
        split_list = prepared.split_list
        exact_attention_profile = prepared.attention_profile
        cache_backend_stream_ids = torch.cat(
            [
                torch.zeros(int(split_list[0]), device=hidden_states.device, dtype=torch.long),
                torch.zeros(int(split_list[1]), device=hidden_states.device, dtype=torch.long),
                torch.ones(int(split_list[2]), device=hidden_states.device, dtype=torch.long),
                torch.ones(int(split_list[3]), device=hidden_states.device, dtype=torch.long),
                torch.full((int(split_list[4]),), -1, device=hidden_states.device, dtype=torch.long),
            ],
            dim=0,
        )

        cache_state = self._resolve_exact_cache_state(cache_name)
        cache_backend_name = cache_state.backend_name if cache_state is not None else None
        cache_backend_payload = cache_state.backend_payload if cache_state is not None else None
        next_self_attention_kv: list[AttentionCacheEntry] = []
        moved_tensor_cache: dict[tuple[str, torch.device, torch.dtype | None], torch.Tensor] = {}
        attention_profile_cache: dict[torch.device, PreparedAttentionProfile | None] = {}

        for layer_index, block in enumerate(self.blocks):
            block_device = (
                self._runtime_block_devices[layer_index % len(self._runtime_block_devices)]
                if self._runtime_block_devices
                else hidden_states.device
            )
            if hidden_states.device != block_device:
                hidden_states = hidden_states.to(device=block_device)
            block_text_hidden_states = self._cached_optional_tensor(
                text_hidden_states,
                cache=moved_tensor_cache,
                name="text_hidden_states",
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_timestep_proj = self._cached_optional_tensor(
                timestep_proj,
                cache=moved_tensor_cache,
                name="timestep_proj",
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_rotary_emb = self._cached_optional_tensor(
                rotary_emb,
                cache=moved_tensor_cache,
                name="rotary_emb",
                device=block_device,
            )
            block_attention_profile = self._cached_attention_profile(
                exact_attention_profile,
                cache=attention_profile_cache,
                device=block_device,
            )
            block_cache_backend_stream_ids = self._cached_optional_tensor(
                cache_backend_stream_ids,
                cache=moved_tensor_cache,
                name="cache_backend_stream_ids",
                device=block_device,
            )
            block_cache_backend_state = (
                cache_backend_payload.layer_states[layer_index]
                if cache_backend_uses_slot_pool(cache_backend_name)
                and cache_backend_payload is not None
                and layer_index < len(cache_backend_payload.layer_states)
                else None
            )
            block_cache_backend_state = self._move_slot_pool_layer_state(block_cache_backend_state, device=block_device)
            hidden_states, current_self_cache_entry, _ = block(
                hidden_states,
                encoder_hidden_states=block_text_hidden_states,
                temb=block_timestep_proj,
                rotary_emb=block_rotary_emb,
                attention_profile=block_attention_profile,
                self_attention_cache_backend_name=cache_backend_name,
                self_attention_cache_backend_state=block_cache_backend_state,
                self_attention_cache_update_mode=update_cache,
                self_attention_cache_stream_ids=block_cache_backend_stream_ids,
            )
            next_self_attention_kv.append(current_self_cache_entry or AttentionCacheEntry())

        output_device = self.scale_shift_table.device
        if hidden_states.device != output_device:
            hidden_states = hidden_states.to(device=output_device)
        temb = temb.to(device=output_device, dtype=hidden_states.dtype)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
        hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = _select_split_segments(
            hidden_states,
            tuple(int(length) for length in split_list),
        )

        if cache_state is not None:
            materialized_entries = (
                materialize_cache_backend_entries(cache_backend_payload)
                if cache_backend_uses_slot_pool(cache_backend_name)
                else tuple(next_self_attention_kv)
            )
            self._exact_runtime_caches[cache_name] = CacheState(
                supported=cache_state.supported,
                current_start_frame=cache_state.current_start_frame,
                cached_frames=cache_state.cached_frames,
                chunk_size=cache_state.chunk_size,
                capability=cache_state.capability,
                backend_name=cache_state.backend_name,
                backend_payload=cache_backend_payload,
                payload=dict(cache_state.payload),
                self_attention_kv=materialized_entries,
                cross_attention_kv=cache_state.cross_attention_kv,
                update_metadata=cache_state.update_metadata,
            )

        video_prediction = self.proj_out(latent_hidden_states)
        video_prediction = rearrange(
            video_prediction,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=prepared.batch_size,
        )
        action_prediction = self.action_proj_out(action_hidden_states)
        action_prediction = rearrange(
            action_prediction,
            "1 (b l) c -> b l c",
            b=prepared.batch_size,
        )
        return video_prediction, action_prediction

    def execute_runtime_step(self, step_input: RuntimeStepInput) -> RuntimeStepOutput:
        prepared = prepare_runtime_sequence(
            step_input,
            hidden_size=self.config.hidden_size,
            exact_train_preparer=lambda payload: prepare_exact_dual_stream_train_sequence(
                payload,
                config=self.config,
                patch_size=self.patch_size,
                model_dtype=self.patch_embedding_mlp.weight.dtype,
                input_embed=lambda tensor, input_type: self._input_embed(tensor, input_type=input_type),
                exact_text_hidden_states=lambda text_emb: self._exact_text_hidden_states(
                    text_emb,
                    dtype=self.patch_embedding_mlp.weight.dtype,
                ),
                time_embed=lambda timesteps, height, width, dtype, action_mode: self._time_embed(
                    timesteps,
                    height,
                    width,
                    dtype=dtype,
                    action_mode=action_mode,
                ),
                rope=self.rope,
            ),
        )
        if prepared.mode == "core_input":
            if prepared.core_input is None:
                raise ValueError("Runtime sequence resolved to `core_input` without a core payload.")
            core_output = self.forward(prepared.core_input)
            core_output.aux.setdefault("runtime_program", step_input.program.name)
            core_output.aux.setdefault("sequence_family", step_input.program.sequence_family)
            projected_outputs = (
                self.project_runtime_stream_outputs(
                    family=step_input.program.output_head_family,
                    hidden_states=core_output.tokens,
                    token_layout=core_output.token_layout,
                )
                if step_input.program.output_head_family
                else {}
            )
            return RuntimeStepOutput(
                tokens=core_output.tokens,
                core_output=core_output,
                projected_outputs=projected_outputs,
                cache_state=core_output.cache_state,
                aux={
                    **core_output.aux,
                    "runtime_program": step_input.program.name,
                    "sequence_family": step_input.program.sequence_family,
                    "stream_output_head_family": step_input.program.output_head_family or "none",
                },
            )
        if prepared.mode == "exact_train":
            if prepared.exact_train is None:
                raise ValueError("Exact-train runtime step requires prepared exact-train state.")
            hidden_states = prepared.exact_train.hidden_states
            text_hidden_states = prepared.exact_train.text_hidden_states
            rotary_emb = prepared.exact_train.rotary_emb
            temb = prepared.exact_train.temb
            timestep_proj = prepared.exact_train.timestep_proj
            split_list = prepared.exact_train.split_list
            exact_attention_profile = prepared.exact_train.attention_profile

            for block in self.blocks:
                hidden_states, _, _ = block(
                    hidden_states,
                    encoder_hidden_states=text_hidden_states,
                    temb=timestep_proj,
                    rotary_emb=rotary_emb,
                    attention_profile=exact_attention_profile,
                )

            temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
            shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            hidden_states = (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
            latent_hidden_states, _, action_hidden_states, _, _ = _select_split_segments(
                hidden_states,
                tuple(int(length) for length in split_list),
            )
            video_prediction = self.proj_out(latent_hidden_states)
            video_prediction = rearrange(
                video_prediction,
                "1 (b l) (n c) -> b (l n) c",
                n=math.prod(self.patch_size),
                b=prepared.exact_train.batch_size,
            )
            action_prediction = self.action_proj_out(action_hidden_states)
            action_prediction = rearrange(
                action_prediction,
                "1 (b l) c -> b l c",
                b=prepared.exact_train.batch_size,
            )
            return RuntimeStepOutput(
                projected_outputs={
                    "video_prediction": video_prediction,
                    "action_prediction": action_prediction,
                },
                aux={
                    "runtime_program": step_input.program.name,
                    "sequence_family": step_input.program.sequence_family,
                },
            )
        if prepared.mode == "exact_inference":
            if prepared.exact_inference is None:
                raise ValueError("Exact-inference runtime step requires prepared exact-inference state.")
            video_prediction, action_prediction = self._forward_exact_dual_stream(
                prepared.exact_inference,
                update_cache=prepared.update_cache,
                cache_name=prepared.cache_name,
            )
            return RuntimeStepOutput(
                projected_outputs={
                    "video_prediction": video_prediction,
                    "action_prediction": action_prediction,
                },
                cache_state=self._resolve_exact_cache_state(prepared.cache_name),
                aux={
                    "runtime_program": step_input.program.name,
                    "sequence_family": step_input.program.sequence_family,
                },
            )
        if prepared.mode == "exact_single_stream":
            if prepared.payload is None:
                raise ValueError("Exact single-stream runtime step requires `payload`.")
            tokens = self._forward_exact_single_stream(
                prepared.payload,
                update_cache=prepared.update_cache,
                cache_name=prepared.cache_name,
                action_mode=prepared.action_mode,
            )
            return RuntimeStepOutput(
                tokens=tokens,
                projected_outputs={"stream_prediction": tokens},
                cache_state=self._resolve_exact_cache_state(prepared.cache_name),
                aux={
                    "runtime_program": step_input.program.name,
                    "sequence_family": step_input.program.sequence_family,
                    "action_mode": prepared.action_mode,
                },
            )
        raise ValueError(f"Unsupported prepared runtime sequence mode {prepared.mode!r}.")

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

    def forward(
        self,
        core_input: VisualCoreInput | dict[str, torch.Tensor],
        *,
        update_cache: int = 0,
        cache_name: str = "open_wam_exact",
        action_mode: bool = False,
        train_mode: bool = False,
    ) -> VisualCoreOutput | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(core_input, dict):
            if train_mode:
                return self.forward_train(core_input)
            return self._forward_exact_single_stream(
                core_input,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )
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
        prep_device = (
            self.time_conditioner.time_embedder.linear_1.weight.device
            if self._runtime_block_devices
            else hidden_states.device
        )
        if hidden_states.device != prep_device:
            hidden_states = hidden_states.to(device=prep_device)
        if position_context is not None and position_context.device != prep_device:
            position_context = position_context.to(device=prep_device, dtype=hidden_states.dtype)
        if grid_ids is not None and grid_ids.device != prep_device:
            grid_ids = grid_ids.to(device=prep_device)
        if timestep_values is not None and timestep_values.device != prep_device:
            timestep_values = timestep_values.to(device=prep_device)
        if attention_mask is not None and attention_mask.device != prep_device:
            attention_mask = attention_mask.to(device=prep_device)
        if stream_ids_tensor is not None and stream_ids_tensor.device != prep_device:
            stream_ids_tensor = stream_ids_tensor.to(device=prep_device)
        device = prep_device
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

        structured_block_semantics = core_input.structured_block_semantics
        structured_frequency_bundle = core_input.structured_frequency_bundle
        structured_attention_context = self._resolve_structured_attention_context(
            core_input,
            device=device,
        )
        rotary_grid_ids = self._compose_structured_rotary_grid_ids(
            structured_block_semantics=structured_block_semantics,
            structured_frequency_bundle=structured_frequency_bundle,
            fallback_grid_ids=grid_ids,
        )
        rotary_emb = self.rope(rotary_grid_ids.to(device=device))[:, :, None] if rotary_grid_ids is not None else None
        encoder_hidden_states = self._resolve_encoder_hidden_states(
            core_input,
            stream_ids=stream_ids,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        cache_update_metadata = core_input.cache_update_metadata or CacheUpdateMetadata()
        captured_readouts: list[VisualIntermediateReadout] = []
        requested_layers = (
            set(core_input.readout_request.capture_layer_indices)
            if core_input.readout_request is not None
            else set()
        )
        cache_branch = cache_update_metadata.cache_branch
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
        incoming_branch_state = resolve_cache_branch_state(core_input.cache_state, cache_branch)
        incoming_self_attention_kv = incoming_branch_state.self_attention_kv
        incoming_cross_attention_kv = incoming_branch_state.cross_attention_kv
        for layer_index, block in enumerate(self.blocks):
            block_device = (
                self._runtime_block_devices[layer_index % len(self._runtime_block_devices)]
                if self._runtime_block_devices
                else hidden_states.device
            )
            if hidden_states.device != block_device:
                hidden_states = hidden_states.to(device=block_device)
            block_timestep_proj = timestep_proj.to(device=block_device, dtype=hidden_states.dtype)
            block_encoder_hidden_states = encoder_hidden_states.to(device=block_device, dtype=hidden_states.dtype)
            block_rotary_emb = self._move_optional_tensor(rotary_emb, device=block_device)
            block_attention_mask = self._move_optional_tensor(attention_mask, device=block_device)
            block_cached_prefix_visibility = self._move_optional_tensor(
                cached_prefix_visibility,
                device=block_device,
                dtype=hidden_states.dtype,
            )
            block_structured_attention_context = self._move_structured_attention_context(
                structured_attention_context,
                device=block_device,
            )
            block_structured_frequency_bundle = self._move_structured_frequency_bundle(
                structured_frequency_bundle,
                device=block_device,
            )
            hidden_states, current_self_cache_entry, current_cross_cache_entry = block(
                hidden_states,
                encoder_hidden_states=block_encoder_hidden_states,
                temb=block_timestep_proj,
                rotary_emb=block_rotary_emb,
                structured_attention_context=block_structured_attention_context,
                structured_block_semantics=structured_block_semantics,
                structured_frequency_bundle=block_structured_frequency_bundle,
                attention_mask=block_attention_mask,
                attention_profile=core_input.attention_profile,
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
                cached_prefix_visibility=block_cached_prefix_visibility,
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
            if layer_index in requested_layers:
                captured_readouts.append(
                    VisualIntermediateReadout(
                        layer_index=layer_index,
                        tokens=hidden_states,
                        token_layout=token_layout,
                        aux={"implementation": "shared_transformer"},
                    )
                )

        output_device = self.scale_shift_table.device
        if hidden_states.device != output_device:
            hidden_states = hidden_states.to(device=output_device)
        temb = temb.to(device=output_device, dtype=hidden_states.dtype)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = _select_chunk_slices(temb_scale_shift_table, 2)
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
                        "implementation": "shared_transformer",
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
                        "implementation": "shared_transformer",
                        "cache_kind": "cross_attention",
                    },
                )
                for layer_index, entry in enumerate(next_cross_attention_kv)
            )
            if has_runtime_sequence
            else tuple()
        )
        if core_input.cache_state is not None:
            branch_state = CacheBranchState(
                backend_name=incoming_branch_state.backend_name,
                backend_payload=incoming_branch_state.backend_payload,
                payload=dict(incoming_branch_state.payload),
                self_attention_kv=(
                    incoming_branch_state.self_attention_kv
                    if incoming_branch_state.self_attention_kv and not cache_update_metadata.update_kv_cache
                    else layer_cache_entries
                ),
                cross_attention_kv=(
                    incoming_branch_state.cross_attention_kv
                    if incoming_branch_state.cross_attention_kv and not cache_update_metadata.update_cross_attention_cache
                    else cross_layer_cache_entries
                ),
            )
            cache_state = replace_cache_branch_state(
                CacheState(
                    supported=core_input.cache_state.supported or has_runtime_sequence,
                    current_start_frame=cache_update_metadata.current_start_frame,
                    cached_frames=core_input.cache_state.cached_frames,
                    chunk_size=core_input.cache_state.chunk_size,
                    capability=(
                        core_input.cache_state.capability
                        if core_input.cache_state.capability != "none"
                        else ("self_attn_plus_cross_attn" if has_runtime_sequence else "none")
                    ),
                    backend_name=core_input.cache_state.backend_name,
                    backend_payload=core_input.cache_state.backend_payload,
                    payload=dict(core_input.cache_state.payload),
                    self_attention_kv=core_input.cache_state.self_attention_kv,
                    cross_attention_kv=core_input.cache_state.cross_attention_kv,
                    update_metadata=cache_update_metadata,
                    branch_states=dict(core_input.cache_state.branch_states),
                ),
                branch_name=cache_branch,
                branch_state=branch_state,
                mirror_to_top_level=cache_branch in {"default", "conditioned"},
            )
        else:
            cache_state = CacheState(
                supported=has_runtime_sequence,
                current_start_frame=cache_update_metadata.current_start_frame,
                cached_frames=0,
                chunk_size=seq_len,
                capability="self_attn_plus_cross_attn" if has_runtime_sequence else "none",
                backend_name="merged_prefix",
                backend_payload=None,
                payload={"stage": "shared_transformer_core", "implementation": "shared_transformer"},
                self_attention_kv=layer_cache_entries,
                cross_attention_kv=cross_layer_cache_entries,
                update_metadata=cache_update_metadata,
                branch_states={},
            )
        return VisualCoreOutput(
            tokens=hidden_states,
            token_layout=token_layout,
            cache_state=cache_state,
            intermediate_readouts=tuple(captured_readouts),
            aux={
                "implementation": "shared_transformer",
                "used_rotary": rotary_grid_ids is not None,
                "used_action_conditioner": bool((stream_ids != 0).any().item()),
                "has_sequence_metadata": core_input.sequence_metadata is not None,
                "structured_block_mode": (
                    structured_block_semantics.mode if structured_block_semantics is not None else "none"
                ),
                "structured_attention_mode": (
                    structured_attention_context.mode if structured_attention_context is not None else "none"
                ),
                "structured_attention_kernel": (
                    structured_attention_context.attention_kernel
                    if structured_attention_context is not None
                    else "none"
                ),
                "structured_cache_kernel": (
                    structured_attention_context.cache_kernel
                    if structured_attention_context is not None
                    else "none"
                ),
                "structured_attention_internal_mask": bool(
                    structured_attention_context is not None
                    and structured_attention_context.mode == "register_explicit"
                ),
                "structured_attention_full_cache_prefix": bool(
                    structured_attention_context is not None
                    and structured_attention_context.mode == "register_explicit"
                    and incoming_self_attention_kv
                    and incoming_self_attention_kv[0].key is not None
                ),
                "structured_frequency_mode": (
                    structured_frequency_bundle.layout if structured_frequency_bundle is not None else "none"
                ),
                "structured_has_clean_prefix_frequencies": bool(
                    structured_frequency_bundle is not None
                    and structured_frequency_bundle.clean_prefix_grid_ids is not None
                ),
                "structured_has_action_frequencies": bool(
                    structured_frequency_bundle is not None
                    and structured_frequency_bundle.action_grid_ids is not None
                ),
                "structured_has_state_frequencies": bool(
                    structured_frequency_bundle is not None
                    and structured_frequency_bundle.state_grid_ids is not None
                ),
                "structured_register_frame_shift": (
                    structured_attention_context.metadata.get("register_frame_shift")
                    if structured_attention_context is not None
                    else None
                ),
                "structured_time_layout": (
                    structured_block_semantics.time_layout if structured_block_semantics is not None else "generic"
                ),
                "structured_position_layout": (
                    structured_block_semantics.position_layout
                    if structured_block_semantics is not None
                    else "generic"
                ),
                "structured_current_start_frame": (
                    structured_block_semantics.current_start_frame
                    if structured_block_semantics is not None
                    else None
                ),
                "structured_observed_prefix_frames": (
                    structured_block_semantics.observed_prefix_frames
                    if structured_block_semantics is not None
                    else None
                ),
                "structured_action_register_length": (
                    structured_block_semantics.action_register_length
                    if structured_block_semantics is not None
                    else None
                ),
                "structured_state_register_length": (
                    structured_block_semantics.state_register_length
                    if structured_block_semantics is not None
                    else None
                ),
                "structured_clean_prefix_length": (
                    structured_block_semantics.clean_prefix_length
                    if structured_block_semantics is not None
                    else None
                ),
                "cache_runtime_metadata": cache_update_metadata,
            },
        )


LingbotReplicaTimeEmbedding = SharedTransformerTimeEmbedding
LingbotReplicaRotaryPosEmbed = SharedTransformerRotaryPositionalEmbedding
LingbotReplicaAttention = SharedTransformerAttention
LingbotReplicaTransformerBlock = SharedTransformerBlock
LingbotReplicaVisualCore = SharedVideoTransformerCore
