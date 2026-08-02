"""Learned MoT execution against prefilled video and action caches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from open_wam.models.common.sharded_execution import summon_full_parameters
from open_wam.models.visual_tower.runtime_parameter_ops import (
    linear_with_materialized_params,
)

from .contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTVideoCache,
    MoTVideoLayerCache,
)
from .modules import MoTActionExpert, MoTActionPreprocessOutput


def prefill_video_kv_cache(
    *,
    visual_tower,
    observed_prefix: torch.Tensor,
    text_context: torch.Tensor | None,
    frame_start: int = 0,
    attention_mask: torch.Tensor | None = None,
    cross_attention_mask: torch.Tensor | None = None,
    detach_cache: bool = True,
) -> MoTVideoCache:
    """Run the observed video prefix once and cache per-layer self-attention K/V.

    ``attention_mask``: optional square ``[S, S]`` mask over the post-patch
    video token sequence; used to make the prefill chunk-causal when
    teacher-forcing a full clean video during training. Default ``None``
    preserves the fully bidirectional prefill used at inference over
    observed history.

    ``detach_cache``: when True (inference default), K/V are detached so they
    can cross device/dtype boundaries without participating in autograd. Set
    to False during training if you want action-loss gradients to flow back
    through the shared video core (matches Method 1 shared-backbone behavior).
    """

    if observed_prefix.ndim != 5:
        raise ValueError(
            "MoT video prefill expects observed_prefix with shape [B, C, T, H, W], "
            f"got {tuple(observed_prefix.shape)}."
        )
    cache_state = visual_tower.prefill_exact_video_cache(
        observed_prefix=observed_prefix,
        text_context=text_context,
        frame_start=frame_start,
        cache_name="mot_video_prefill",
        attention_mask=attention_mask,
        cross_attention_mask=cross_attention_mask,
        detach_cache=detach_cache,
    )
    cache_layers: list[MoTVideoLayerCache] = []
    for layer_index, entry in enumerate(cache_state.self_attention_kv):
        if entry.key is None or entry.value is None:
            raise ValueError(
                "MoT video prefill expected materialized self-attention K/V entries, "
                f"but layer {layer_index} was empty."
            )
        cache_layers.append(
            MoTVideoLayerCache(
                key=entry.key.detach() if detach_cache else entry.key,
                value=entry.value.detach() if detach_cache else entry.value,
            )
        )
    if not cache_layers:
        raise ValueError("MoT video prefill did not materialize any layer cache entries.")
    return MoTVideoCache(layers=tuple(cache_layers), video_seq_len=int(cache_layers[0].key.shape[2]))


def forward_action_with_video_and_action_cache(
    *,
    action_expert: MoTActionExpert,
    action_pre: MoTActionPreprocessOutput,
    video_cache: MoTVideoCache,
    action_cache: MoTActionCache | None,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, MoTActionCache]:
    """Run the action expert against cached video AND cached past-action K/V.

    Method-1-aligned variant of `forward_action_with_video_cache`. Past
    action chunks contribute per-layer clean K/V via `action_cache`; the
    current (fresh, noisy) chunk's K/V is recomputed from `action_pre`
    each call. Returns the action hidden states plus the per-layer fresh
    K/V, so the caller can append to the running `MoTActionCache` at the
    last denoise step (matching Method 1's `update_cache=1` semantics).
    """

    if len(video_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached video K/V pair per action layer, "
            f"got video_cache_layers={len(video_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    if action_cache is not None and len(action_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached action K/V pair per action layer, "
            f"got action_cache_layers={len(action_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    fresh_action_seq_len = int(action_pre.tokens.shape[1])
    action_cache_seq_len = int(action_cache.action_seq_len) if action_cache is not None else 0
    expected_total = video_cache.video_seq_len + action_cache_seq_len + fresh_action_seq_len
    if attention_mask.shape != (expected_total, expected_total):
        raise ValueError(
            "MoT action runtime requires a square joint attention mask matching video+action_cache+fresh length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )

    hidden_states = action_pre.tokens
    fresh_q_start = video_cache.video_seq_len + action_cache_seq_len
    action_attention_mask = attention_mask[fresh_q_start:, :expected_total][None, None, :, :]
    action_rotary_emb = action_pre.freqs[:, :, None]
    fresh_kv_layers: list[MoTActionLayerCache] = []
    with summon_full_parameters(action_expert):
        for layer_index, (block, vid_layer) in enumerate(
            zip(action_expert.blocks, video_cache.layers, strict=True)
        ):
            attn_inputs = block.prepare_self_attention_inputs(
                hidden_states,
                temb=action_pre.t_mod,
                rotary_emb=action_rotary_emb,
            )
            key_parts: list[torch.Tensor] = [vid_layer.key]
            value_parts: list[torch.Tensor] = [vid_layer.value]
            if action_cache is not None:
                act_layer = action_cache.layers[layer_index]
                key_parts.append(act_layer.key)
                value_parts.append(act_layer.value)
            key_parts.append(attn_inputs["key"])
            value_parts.append(attn_inputs["value"])
            mixed = F.scaled_dot_product_attention(
                attn_inputs["query"],
                torch.cat(key_parts, dim=2),
                torch.cat(value_parts, dim=2),
                attn_mask=action_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            hidden_states, _ = block.apply_post_attention(
                attn_inputs["hidden_states"],
                mixed_attn_output=block.attn1.to_out[1](linear_with_materialized_params(block.attn1.to_out[0], mixed)),
                encoder_hidden_states=action_pre.context,
                gate_msa=attn_inputs["gate_msa"],
                c_shift_msa=attn_inputs["c_shift_msa"],
                c_scale_msa=attn_inputs["c_scale_msa"],
                c_gate_msa=attn_inputs["c_gate_msa"],
                cross_attention_mask=action_pre.cross_attention_mask,
            )
            fresh_kv_layers.append(
                MoTActionLayerCache(
                    key=attn_inputs["key"].detach(),
                    value=attn_inputs["value"].detach(),
                )
            )
    fresh_cache = MoTActionCache(
        layers=tuple(fresh_kv_layers),
        action_seq_len=int(fresh_kv_layers[0].key.shape[2]),
    )
    return hidden_states, fresh_cache


def forward_action_with_video_cache(
    *,
    action_expert: MoTActionExpert,
    action_pre: MoTActionPreprocessOutput,
    video_cache: MoTVideoCache,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the action expert against a cached video prefix.

    This mirrors the FastWAM `forward_action_with_video_cache` contract:
    action queries are recomputed every denoise step, while video K/V come from
    a prefilled observed-prefix cache. The helper deliberately leaves video
    cache construction to a later stage because that requires exposing
    additional blockwise helpers from the shared video core.
    """

    if len(video_cache.layers) != len(action_expert.blocks):
        raise ValueError(
            "MoT action runtime requires one cached video K/V pair per action layer, "
            f"got cache_layers={len(video_cache.layers)}, action_layers={len(action_expert.blocks)}."
        )
    action_seq_len = int(action_pre.tokens.shape[1])
    expected_total = video_cache.video_seq_len + action_seq_len
    if attention_mask.shape != (expected_total, expected_total):
        raise ValueError(
            "MoT action runtime requires a square joint attention mask matching cache+action length, "
            f"got attention_mask={tuple(attention_mask.shape)}, expected=({expected_total}, {expected_total})."
        )

    hidden_states = action_pre.tokens
    action_attention_mask = attention_mask[video_cache.video_seq_len:, :expected_total][None, None, :, :]
    action_rotary_emb = action_pre.freqs[:, :, None]
    with summon_full_parameters(action_expert):
        for block, layer_cache in zip(action_expert.blocks, video_cache.layers, strict=True):
            attn_inputs = block.prepare_self_attention_inputs(
                hidden_states,
                temb=action_pre.t_mod,
                rotary_emb=action_rotary_emb,
            )
            mixed = F.scaled_dot_product_attention(
                attn_inputs["query"],
                torch.cat([layer_cache.key, attn_inputs["key"]], dim=2),
                torch.cat([layer_cache.value, attn_inputs["value"]], dim=2),
                attn_mask=action_attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2).flatten(2, 3)
            hidden_states, _ = block.apply_post_attention(
                attn_inputs["hidden_states"],
                mixed_attn_output=block.attn1.to_out[1](linear_with_materialized_params(block.attn1.to_out[0], mixed)),
                encoder_hidden_states=action_pre.context,
                gate_msa=attn_inputs["gate_msa"],
                c_shift_msa=attn_inputs["c_shift_msa"],
                c_scale_msa=attn_inputs["c_scale_msa"],
                c_gate_msa=attn_inputs["c_gate_msa"],
                cross_attention_mask=action_pre.cross_attention_mask,
            )
    return hidden_states
