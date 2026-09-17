"""Owned video/action layer pairs with shared attention and native FSDP hooks.

Each pair remains one parameter owner and one FSDP unit during training and
inference. Call-local cache reuse changes attention work, not ownership,
visibility or residual order. Inference can skip invariant feature rows.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from open_wam.models.common.attention_backends import apply_attention_backend
from open_wam.models.common.denoising_cache import (
    DenoisingCache,
    InvariantTokenCache,
    select_block_mask,
)
from open_wam.models.visual_tower.shared_transformer_embeddings import apply_rotary_emb
from open_wam.models.visual_tower.shared_transformer_layout import select_chunk_slices


def _native_attention(
    attn: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rotary_emb: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    block_mask: Any | None = None,
) -> torch.Tensor:
    query = attn.norm_q(attn.to_q(q.contiguous())).unflatten(2, (attn.heads, -1))
    key = attn.norm_k(attn.to_k(k.contiguous())).unflatten(2, (attn.heads, -1))
    value = attn.to_v(v.contiguous()).unflatten(2, (attn.heads, -1))
    if rotary_emb is not None:
        query = apply_rotary_emb(query, rotary_emb)
        key = apply_rotary_emb(key, rotary_emb)
    if attention_mask is not None:
        if attention_mask.ndim == 3:
            attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim != 4:
            raise ValueError(
                "DualExpert packed block cross-attention mask must have shape [B, Q, K] or [B, H, Q, K], "
                f"got {tuple(attention_mask.shape)}."
            )
    if block_mask is not None:
        hidden_states = apply_attention_backend(
            query=query.transpose(1, 2).contiguous(),
            key=key.transpose(1, 2).contiguous(),
            value=value.transpose(1, 2).contiguous(),
            block_mask=block_mask,
        )
    else:
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
    return attn.to_out[1](attn.to_out[0](hidden_states))


def _prepare_self_attention_inputs_native(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    temb_scale_shift_table = (
        block.scale_shift_table.to(device=temb.device, dtype=temb.dtype)[None]
        + temb.float()
    )
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        select_chunk_slices(
            temb_scale_shift_table,
            6,
        )
    )
    norm_hidden_states = (
        block.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa
    ).type_as(hidden_states)
    query = block.attn1.norm_q(block.attn1.to_q(norm_hidden_states)).unflatten(
        2, (block.attn1.heads, -1)
    )
    key = block.attn1.norm_k(block.attn1.to_k(norm_hidden_states)).unflatten(
        2, (block.attn1.heads, -1)
    )
    value = block.attn1.to_v(norm_hidden_states).unflatten(2, (block.attn1.heads, -1))
    if rotary_emb is not None:
        query = apply_rotary_emb(query, rotary_emb)
        key = apply_rotary_emb(key, rotary_emb)
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


def _apply_post_attention_native(
    block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    mixed_attn_output: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    gate_msa: torch.Tensor,
    c_shift_msa: torch.Tensor,
    c_scale_msa: torch.Tensor,
    c_gate_msa: torch.Tensor,
    cross_attention_mask: torch.Tensor | None = None,
    cross_attention_block_mask: Any | None = None,
) -> torch.Tensor:
    hidden_states = (
        hidden_states.float() + mixed_attn_output.float() * gate_msa
    ).type_as(hidden_states)
    norm_hidden_states = block.norm2(hidden_states.float()).type_as(hidden_states)
    hidden_states = hidden_states + _native_attention(
        block.attn2,
        norm_hidden_states,
        encoder_hidden_states,
        encoder_hidden_states,
        attention_mask=cross_attention_mask,
        block_mask=cross_attention_block_mask,
    )
    norm_hidden_states = (
        block.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa
    ).type_as(hidden_states)
    ff_output = block.ffn(norm_hidden_states)
    return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(
        hidden_states
    )


class DualExpertPackedBlock(nn.Module):
    """One layer pair (video_block + action_block) with joint self-attention.

    The two underlying blocks are registered as children so PyTorch + FSDP
    see them under this module's parameter tree. Forward runs joint attention
    over packed ``[V_noisy, V_clean, A_noisy, A_clean]`` keys/values and then
    routes the per-stream outputs through native cross-attention + FFN calls.
    """

    def __init__(self, video_block: nn.Module, action_block: nn.Module) -> None:
        super().__init__()
        self.video_block = video_block
        self.action_block = action_block

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor | None,
        *,
        video_timestep_proj: torch.Tensor,
        video_rotary_emb: torch.Tensor | None,
        action_temb: torch.Tensor | None,
        action_rotary_emb: torch.Tensor | None,
        video_attention_mask: torch.Tensor | None,
        action_attention_mask: torch.Tensor | None,
        video_text_hidden_states: torch.Tensor,
        action_text_hidden_states: torch.Tensor | None,
        video_cross_attention_mask: torch.Tensor | None = None,
        action_cross_attention_mask: torch.Tensor | None = None,
        video_cross_attention_block_mask: Any | None = None,
        action_cross_attention_block_mask: Any | None = None,
        block_mask: Any | None = None,
        flex_kernel_options: dict[str, Any] | None = None,
        token_cache: tuple[InvariantTokenCache, InvariantTokenCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        streams = (
            (
                self.video_block,
                video_hidden_states,
                video_timestep_proj,
                video_rotary_emb,
                video_attention_mask,
                video_text_hidden_states,
                video_cross_attention_mask,
                video_cross_attention_block_mask,
            ),
            (
                self.action_block,
                action_hidden_states,
                action_temb,
                action_rotary_emb,
                action_attention_mask,
                action_text_hidden_states,
                action_cross_attention_mask,
                action_cross_attention_block_mask,
            ),
        )
        if token_cache is not None and torch.is_grad_enabled():
            raise ValueError(
                "Invariant feature reuse is only supported without autograd."
            )
        prepared = []
        for index, (
            block,
            hidden,
            temb,
            rope,
            mask,
            text,
            cross_mask,
            cross_block,
        ) in enumerate(streams):
            if hidden is None:
                continue
            cache = None if token_cache is None else token_cache[index]
            if cache is not None:
                mask = cache.select(mask, dim=-2)
                hidden = cache.select(hidden)
                temb = cache.select(temb)
                rope = cache.select(rope)
                cross_mask = cache.select(cross_mask, dim=-2)
                cross_block = select_block_mask(
                    cross_block,
                    cache.active_indices if cache.ready else cache.computed_indices,
                )
            inputs = _prepare_self_attention_inputs_native(
                block, hidden, temb=temb, rotary_emb=rope
            )
            if cache is not None:
                inputs["key"], inputs["value"] = cache.key_value(
                    inputs["key"], inputs["value"]
                )
            prepared.append(
                (index, block, inputs, mask, text, cross_mask, cross_block, cache)
            )

        joint_key = torch.cat([item[2]["key"] for item in prepared], dim=2)
        joint_value = torch.cat([item[2]["value"] for item in prepared], dim=2)
        if block_mask is not None:
            joint_query = torch.cat([item[2]["query"] for item in prepared], dim=2)
            mixed = apply_attention_backend(
                query=joint_query,
                key=joint_key,
                value=joint_value,
                block_mask=block_mask,
                kernel_options=flex_kernel_options,
            )
            mixed_streams = torch.split(
                mixed, [item[2]["query"].shape[2] for item in prepared], dim=2
            )
        else:
            mixed_streams = [
                F.scaled_dot_product_attention(
                    item[2]["query"],
                    joint_key,
                    joint_value,
                    attn_mask=item[3],
                    dropout_p=0.0,
                    is_causal=False,
                )
                for item in prepared
            ]

        results = [None, None]
        for item, mixed in zip(prepared, mixed_streams, strict=True):
            index, block, inputs, _, text, cross_mask, cross_block, cache = item
            mixed = mixed.transpose(1, 2).flatten(2, 3)
            attention_output = block.attn1.to_out[1](block.attn1.to_out[0](mixed))
            hidden = _apply_post_attention_native(
                block,
                inputs["hidden_states"],
                mixed_attn_output=attention_output,
                encoder_hidden_states=text,
                gate_msa=inputs["gate_msa"],
                c_shift_msa=inputs["c_shift_msa"],
                c_scale_msa=inputs["c_scale_msa"],
                c_gate_msa=inputs["c_gate_msa"],
                cross_attention_mask=cross_mask,
                cross_attention_block_mask=cross_block,
            )
            results[index] = hidden if cache is None else cache.output(hidden)
        return results[0], results[1]


class DualExpertPackedBlockStack(nn.Module):
    """Sequence of ``DualExpertPackedBlock`` running joint attention layer-by-layer."""

    def __init__(
        self,
        video_blocks: nn.ModuleList | list[nn.Module],
        action_blocks: nn.ModuleList | list[nn.Module],
    ) -> None:
        super().__init__()
        if len(video_blocks) != len(action_blocks):
            raise ValueError(
                "DualExpertPackedBlockStack requires equal video/action block counts, "
                f"got video={len(video_blocks)}, action={len(action_blocks)}."
            )
        self.packed_blocks = nn.ModuleList(
            [DualExpertPackedBlock(v, a) for v, a in zip(video_blocks, action_blocks)]
        )

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor | None,
        *,
        use_activation_checkpointing: bool = False,
        denoising_cache: DenoisingCache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if denoising_cache is not None and len(denoising_cache.layers) != len(
            self.packed_blocks
        ):
            raise ValueError("Denoising cache layer count must match its executor.")
        checkpoint_active = use_activation_checkpointing and torch.is_grad_enabled()
        for index, packed_block in enumerate(self.packed_blocks):
            block_kwargs = kwargs
            if denoising_cache is not None:
                token_cache = denoising_cache.layers[index]
                block_kwargs = {**kwargs, "token_cache": token_cache}
                for name in ("video_attention_mask", "action_attention_mask"):
                    if kwargs.get(name) is not None:
                        block_kwargs[name] = kwargs[name].index_select(
                            -1, denoising_cache.key_indices
                        )
                if kwargs.get("block_mask") is not None:
                    profile = (
                        denoising_cache.query_profile
                        if token_cache[0].ready
                        else denoising_cache.prefill_profile
                    )
                    block_kwargs["block_mask"] = profile.self_attention_block_mask
            if checkpoint_active:
                video_hidden_states, action_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        packed_block,
                        video_hidden_states,
                        action_hidden_states,
                        use_reentrant=False,
                        **block_kwargs,
                    )
                )
            else:
                video_hidden_states, action_hidden_states = packed_block(
                    video_hidden_states,
                    action_hidden_states,
                    **block_kwargs,
                )
        return video_hidden_states, action_hidden_states
