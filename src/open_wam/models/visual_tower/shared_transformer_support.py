"""Reusable Wan-style transformer primitives shared by visual and action paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch import nn

from open_wam.models.common import (
    PreparedAttentionProfile,
    apply_attention_backend,
    select_attention_profile_mask,
)
from open_wam.models.common.cache_backend_contracts import (
    SLOT_POOL_ALLOW_VIDEO_TO_ACTION_PREFIX_TAIL_TOKENS,
    cache_backend_uses_slot_pool,
)
from open_wam.models.common.cache_backend_lifecycle import (
    update_slot_pool_layer_state,
)
from open_wam.models.common.cache_layout_policy import (
    packed_slot_pool_query_sequence_ids as _packed_slot_pool_query_sequence_ids,
)
from open_wam.models.common.cache_layout_policy import (
    prepare_sdpa_mask as _prepare_sdpa_mask,
)
from open_wam.models.common.cache_layout_policy import (
    prepend_cached_prefix_mask as _prepend_cached_prefix_mask,
)
from open_wam.models.common.cache_layout_policy import (
    resolve_slot_pool_prefix_visibility as _resolve_slot_pool_prefix_visibility,
)
from open_wam.models.common.cache_layout_policy import (
    retained_slot_pool_indices_for_current_write as _retained_slot_pool_indices_for_current_write,
)
from open_wam.models.video_backbone.contracts import AttentionCacheEntry

from .runtime_parameter_ops import (
    feed_forward_with_materialized_params,
    layer_norm_with_materialized_params,
    linear_with_materialized_params,
    materialize_runtime_parameter,
    rms_norm_with_materialized_weight,
)
from .shared_transformer_embeddings import (
    SharedTransformerRotaryPositionalEmbedding,
    SharedTransformerTimeEmbedding,
    apply_rotary_emb,
)
from .shared_transformer_layout import select_chunk_slices, select_split_segments

_COMPATIBILITY_EXPORTS = (F, TimestepEmbedding, Timesteps, rearrange)


# Private aliases preserve historical internal imports while this module owns no
# duplicate implementation.
_apply_rotary_emb = apply_rotary_emb
_select_chunk_slices = select_chunk_slices
_materialize_runtime_parameter = materialize_runtime_parameter
_linear_with_materialized_params = linear_with_materialized_params
_rms_norm_with_materialized_weight = rms_norm_with_materialized_weight
_layer_norm_with_materialized_params = layer_norm_with_materialized_params
_feed_forward_with_materialized_params = feed_forward_with_materialized_params


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
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, rotary_emb)
                key = _apply_rotary_emb(key, rotary_emb)
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
            query = query.transpose(1, 2)
        else:
            if rotary_emb is not None:
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

        This helper is used by dual-expert runtime paths that need to mix
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
        cache_text_context: bool = True,
    ) -> tuple[torch.Tensor, AttentionCacheEntry | None, AttentionCacheEntry | None]:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = _select_chunk_slices(
            temb_scale_shift_table,
            6,
        )

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output, self_cache_entry = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            attention_profile=attention_profile,
            is_cross_attention=False,
            cached_key_value=self_attention_cache_entry,
            cached_prefix_visibility=cached_prefix_visibility,
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
            cache_current_token_count=(
                encoder_hidden_states.shape[1]
                if cache_text_context and cross_attention_cache_entry is None else 0
            ),
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states, self_cache_entry, cross_cache_entry


__all__ = [
    "SharedTransformerAttention",
    "SharedTransformerBlock",
    "SharedTransformerRotaryPositionalEmbedding",
    "SharedTransformerTimeEmbedding",
    "apply_rotary_emb",
    "feed_forward_with_materialized_params",
    "layer_norm_with_materialized_params",
    "linear_with_materialized_params",
    "materialize_runtime_parameter",
    "rms_norm_with_materialized_weight",
    "select_chunk_slices",
    "select_split_segments",
]
