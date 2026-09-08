"""Prepare samples independently, execute their transformer tokens together."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.utils.checkpoint

from open_wam.models.common.attention_contracts import PreparedAttentionProfile
from open_wam.models.common.sequence_batch_attention import (
    build_sequence_batch_cross_attention,
    build_sequence_batch_self_attention,
)
from .dual_stream_execution import finish_packed_video, prepare_packed_video_inputs
from .modules import DualExpertActionPreprocessOutput
from .packed_block import DualExpertPackedBlock


@dataclass(frozen=True)
class DualExpertDenoiseRequest:
    """Parameter-free boundary between sample construction and shared heavy layers."""

    noisy_video_latents: torch.Tensor
    clean_video_latents: torch.Tensor
    noisy_video_timesteps: torch.Tensor
    clean_video_timesteps: torch.Tensor | None
    packed_action_pre: DualExpertActionPreprocessOutput
    attention_profile: PreparedAttentionProfile
    text_context: torch.Tensor | None
    frame_start: int = 0
    video_cross_attention_mask: torch.Tensor | None = None
    video_hidden_context: torch.Tensor | None = None

    def as_kwargs(self) -> dict[str, Any]:
        # dataclasses.asdict would copy tensors and break non-leaf autograd tensors.
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _join(values: Sequence[torch.Tensor], slots: Sequence[int]) -> torch.Tensor:
    return torch.cat(
        [
            torch.cat(
                [value, value.new_zeros((1, slot - value.shape[1], *value.shape[2:]))],
                dim=1,
            )
            for value, slot in zip(values, slots, strict=True)
        ],
        dim=1,
    )


def forward_dual_expert_sequence_batch(
    *,
    visual_tower,
    action_expert,
    requests: Sequence[DualExpertDenoiseRequest],
    padded: bool,
    use_activation_checkpointing: bool = False,
    packed_block_stack=None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """One heavy layer call per layer, never one full forward per sample.

    Packed mode removes collator padding. Padded/bucket mode deliberately retains
    max-length video/action slots per sample, so its compute remains measurable
    separately. Both modes preserve sequence-isolated self/cross attention.
    """
    if not requests:
        raise ValueError("Cannot execute an empty sequence batch.")
    if any(request.noisy_video_latents.shape[0] != 1 for request in requests):
        raise ValueError("Sequence execution expects individual B=1 samples.")
    video = [
        prepare_packed_video_inputs(
            visual_tower=visual_tower,
            noisy_video_latents=request.noisy_video_latents,
            clean_video_latents=request.clean_video_latents,
            noisy_video_timesteps=request.noisy_video_timesteps,
            clean_video_timesteps=request.clean_video_timesteps,
            text_context=request.text_context,
            frame_start=request.frame_start,
            video_hidden_context=request.video_hidden_context,
        )
        for request in requests
    ]
    action = [request.packed_action_pre for request in requests]
    video_lengths = [int(item["hidden_states"].shape[1]) for item in video]
    action_lengths = [int(item.tokens.shape[1]) for item in action]
    video_slots = [max(video_lengths)] * len(video) if padded else video_lengths
    action_slots = [max(action_lengths)] * len(action) if padded else action_lengths
    video_hidden = _join([item["hidden_states"] for item in video], video_slots)
    action_hidden = _join([item.tokens for item in action], action_slots)
    device = video_hidden.device
    dense, block_mask, _ = build_sequence_batch_self_attention(
        [request.attention_profile for request in requests],
        (video_lengths, action_lengths),
        (video_slots, action_slots),
        device=device,
    )
    video_count = sum(video_slots)
    video_cross, video_cross_blocks = build_sequence_batch_cross_attention(
        video_lengths,
        video_slots,
        [item["text_hidden_states"].shape[1] for item in video],
        [request.video_cross_attention_mask for request in requests],
        device=device,
    )
    action_cross, action_cross_blocks = build_sequence_batch_cross_attention(
        action_lengths,
        action_slots,
        [item.context.shape[1] for item in action],
        [item.cross_attention_mask for item in action],
        device=device,
    )
    kwargs = {
        "video_timestep_proj": _join(
            [item["timestep_proj"] for item in video], video_slots
        ),
        "video_rotary_emb": _join([item["rotary_emb"] for item in video], video_slots),
        "action_temb": _join([item.t_mod for item in action], action_slots),
        "action_rotary_emb": _join(
            [item.freqs[:, :, None] for item in action], action_slots
        ),
        "video_attention_mask": None
        if dense is None
        else dense[:video_count][None, None],
        "action_attention_mask": None
        if dense is None
        else dense[video_count:][None, None],
        "video_text_hidden_states": torch.cat(
            [item["text_hidden_states"] for item in video], dim=1
        ),
        "action_text_hidden_states": torch.cat(
            [item.context for item in action], dim=1
        ),
        "video_cross_attention_mask": None
        if video_cross is None
        else video_cross[None],
        "action_cross_attention_mask": None
        if action_cross is None
        else action_cross[None],
        "video_cross_attention_block_mask": video_cross_blocks,
        "action_cross_attention_block_mask": action_cross_blocks,
        "block_mask": block_mask,
        "flex_kernel_options": None
        if block_mask is None
        else {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        },
    }
    blocks = (
        packed_block_stack.packed_blocks
        if packed_block_stack is not None
        else [
            DualExpertPackedBlock(v, a)
            for v, a in zip(visual_tower.core.blocks, action_expert.blocks, strict=True)
        ]
    )
    checkpoint_active = use_activation_checkpointing and torch.is_grad_enabled()
    for block in blocks:
        if checkpoint_active:
            video_hidden, action_hidden = torch.utils.checkpoint.checkpoint(
                block,
                video_hidden,
                action_hidden,
                use_reentrant=False,
                **kwargs,
            )
        else:
            video_hidden, action_hidden = block(video_hidden, action_hidden, **kwargs)

    results = []
    video_offset = action_offset = 0
    for request, prepared, nv, na, sv, sa in zip(
        requests,
        video,
        video_lengths,
        action_lengths,
        video_slots,
        action_slots,
        strict=True,
    ):
        hidden = video_hidden[:, video_offset : video_offset + nv]
        flow = finish_packed_video(
            visual_tower,
            hidden,
            prepared["temb"],
            tuple(request.noisy_video_latents.shape),
        )
        results.append((flow, action_hidden[:, action_offset : action_offset + na]))
        video_offset += sv
        action_offset += sa
    return tuple(results)
