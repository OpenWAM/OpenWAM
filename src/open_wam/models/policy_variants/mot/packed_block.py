"""FSDP-friendly packed video+action block for M5 six-coupling experiments.

The legacy ``_packed_block_step`` closure inside
``forward_mot_packed_coupling_denoise`` calls ``_linear_with_materialized_params``
and similar helpers that bypass FSDP's standard pre/post-forward hooks. Under
FSDP2 with per-block ``fully_shard``, those helpers manually call
``param.full_tensor()``; the materialized view's storage is freed when the
surrounding ``with _unshard_runtime_params(...)`` exits, so backward fails with
``setStorage: ... out of bounds for storage of size 0``.

``MoTPackedBlock`` wraps one ``(video_block, action_block)`` pair so that the
joint attention runs inside a single ``nn.Module.forward``. When this module is
passed to ``fully_shard``, the standard FSDP hook lifecycle takes over: params
all_gather at module entry, register backward hooks for re-gather during
backward, then reshard after the optimizer step. No manual ``full_tensor``
calls are needed.

The packed block inlines the self-attention preparation and post-attention
residual/cross-attention/FFN path with native ``nn.Module`` calls. That keeps
all q/k/v, output projection, layer norm, cross-attention, and FFN parameters
on the standard autograd/FSDP hook path instead of using manual materialized
parameter views.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_wam.models.common.attention_profiles import apply_attention_backend
from open_wam.models.visual_tower.shared_transformer_support import apply_rotary_emb, select_chunk_slices


def _native_attention(
    attn: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rotary_emb: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
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
                "MoT packed block cross-attention mask must have shape [B, Q, K] or [B, H, Q, K], "
                f"got {tuple(attention_mask.shape)}."
            )
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
    temb_scale_shift_table = block.scale_shift_table.to(device=temb.device, dtype=temb.dtype)[None] + temb.float()
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = select_chunk_slices(
        temb_scale_shift_table,
        6,
    )
    norm_hidden_states = (block.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
    query = block.attn1.norm_q(block.attn1.to_q(norm_hidden_states)).unflatten(2, (block.attn1.heads, -1))
    key = block.attn1.norm_k(block.attn1.to_k(norm_hidden_states)).unflatten(2, (block.attn1.heads, -1))
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
) -> torch.Tensor:
    hidden_states = (hidden_states.float() + mixed_attn_output.float() * gate_msa).type_as(hidden_states)
    norm_hidden_states = block.norm2(hidden_states.float()).type_as(hidden_states)
    hidden_states = hidden_states + _native_attention(
        block.attn2,
        norm_hidden_states,
        encoder_hidden_states,
        encoder_hidden_states,
        attention_mask=cross_attention_mask,
    )
    norm_hidden_states = (block.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = block.ffn(norm_hidden_states)
    return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


class MoTPackedBlock(nn.Module):
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
        action_hidden_states: torch.Tensor,
        *,
        video_timestep_proj: torch.Tensor,
        video_rotary_emb: torch.Tensor | None,
        action_temb: torch.Tensor,
        action_rotary_emb: torch.Tensor | None,
        video_attention_mask: torch.Tensor | None,
        action_attention_mask: torch.Tensor | None,
        video_text_hidden_states: torch.Tensor,
        action_text_hidden_states: torch.Tensor,
        video_cross_attention_mask: torch.Tensor | None = None,
        action_cross_attention_mask: torch.Tensor | None = None,
        block_mask: Any | None = None,
        flex_kernel_options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_attn_inputs = _prepare_self_attention_inputs_native(
            self.video_block,
            video_hidden_states,
            temb=video_timestep_proj,
            rotary_emb=video_rotary_emb,
        )
        action_attn_inputs = _prepare_self_attention_inputs_native(
            self.action_block,
            action_hidden_states,
            temb=action_temb,
            rotary_emb=action_rotary_emb,
        )

        joint_key = torch.cat(
            [video_attn_inputs["key"], action_attn_inputs["key"]],
            dim=2,
        )
        joint_value = torch.cat(
            [video_attn_inputs["value"], action_attn_inputs["value"]],
            dim=2,
        )

        if block_mask is not None:
            joint_query = torch.cat(
                [video_attn_inputs["query"], action_attn_inputs["query"]],
                dim=2,
            )
            mixed = apply_attention_backend(
                query=joint_query,
                key=joint_key,
                value=joint_value,
                block_mask=block_mask,
                kernel_options=flex_kernel_options,
            )
            video_seq_len = int(video_attn_inputs["query"].shape[2])
            action_seq_len = int(action_attn_inputs["query"].shape[2])
            mixed_video, mixed_action = torch.split(
                mixed, [video_seq_len, action_seq_len], dim=2
            )
            mixed_video = mixed_video.transpose(1, 2).flatten(2, 3)
            mixed_action = mixed_action.transpose(1, 2).flatten(2, 3)
        else:
            mixed_video = (
                F.scaled_dot_product_attention(
                    video_attn_inputs["query"],
                    joint_key,
                    joint_value,
                    attn_mask=video_attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
                .transpose(1, 2)
                .flatten(2, 3)
            )
            mixed_action = (
                F.scaled_dot_product_attention(
                    action_attn_inputs["query"],
                    joint_key,
                    joint_value,
                    attn_mask=action_attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
                .transpose(1, 2)
                .flatten(2, 3)
            )

        video_self_out = self.video_block.attn1.to_out[1](self.video_block.attn1.to_out[0](mixed_video))
        action_self_out = self.action_block.attn1.to_out[1](self.action_block.attn1.to_out[0](mixed_action))

        new_video = _apply_post_attention_native(
            self.video_block,
            video_attn_inputs["hidden_states"],
            mixed_attn_output=video_self_out,
            encoder_hidden_states=video_text_hidden_states,
            gate_msa=video_attn_inputs["gate_msa"],
            c_shift_msa=video_attn_inputs["c_shift_msa"],
            c_scale_msa=video_attn_inputs["c_scale_msa"],
            c_gate_msa=video_attn_inputs["c_gate_msa"],
            cross_attention_mask=video_cross_attention_mask,
        )
        new_action = _apply_post_attention_native(
            self.action_block,
            action_attn_inputs["hidden_states"],
            mixed_attn_output=action_self_out,
            encoder_hidden_states=action_text_hidden_states,
            gate_msa=action_attn_inputs["gate_msa"],
            c_shift_msa=action_attn_inputs["c_shift_msa"],
            c_scale_msa=action_attn_inputs["c_scale_msa"],
            c_gate_msa=action_attn_inputs["c_gate_msa"],
            cross_attention_mask=action_cross_attention_mask,
        )
        return new_video, new_action


class MoTPackedBlockStack(nn.Module):
    """Sequence of ``MoTPackedBlock`` running joint attention layer-by-layer."""

    def __init__(
        self,
        video_blocks: nn.ModuleList | list[nn.Module],
        action_blocks: nn.ModuleList | list[nn.Module],
    ) -> None:
        super().__init__()
        if len(video_blocks) != len(action_blocks):
            raise ValueError(
                "MoTPackedBlockStack requires equal video/action block counts, "
                f"got video={len(video_blocks)}, action={len(action_blocks)}."
            )
        self.packed_blocks = nn.ModuleList(
            [MoTPackedBlock(v, a) for v, a in zip(video_blocks, action_blocks)]
        )

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for packed_block in self.packed_blocks:
            video_hidden_states, action_hidden_states = packed_block(
                video_hidden_states,
                action_hidden_states,
                **kwargs,
            )
        return video_hidden_states, action_hidden_states
