"""Exact packed video/action attention layouts for DualExpert execution."""

from __future__ import annotations

import torch

from open_wam.configs import CurrentBlockCoupling, HistoryStreamVisibility
from open_wam.models.common.attention_contracts import PreparedAttentionProfile
from open_wam.models.common.coupling_profiles import (
    build_exact_packed_video_action_coupling_profile,
)


def build_dual_expert_packed_coupling_attention_profile(
    *,
    num_video_frames: int,
    video_tokens_per_frame: int,
    num_action_frames: int,
    action_tokens_per_frame: int,
    chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
    current_block_coupling: CurrentBlockCoupling
    | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
    build_dense_masks: bool | None = None,
    build_flex_masks: bool | None = None,
    chunk_origin_frame: int = 0,
    action_context_mask: torch.Tensor | None = None,
    history_stream_visibility: HistoryStreamVisibility | str = (
        HistoryStreamVisibility.VIDEO_ONLY
    ),
    prefix_condition_frames: int = 0,
    singleton_chunk_frame: int | None = None,
    conditional_history_policy: str | None = None,
) -> PreparedAttentionProfile:
    """Build the parallel-stream exact attention profile for dual-expert packed coupling.

    Query/key layout is ``[V_noisy, V_clean, A_noisy, A_clean]``. The mask
    semantics are intentionally sourced from parallel-stream's chunked temporal exact
    profile, so dual-expert's two-expert topology uses the same six coupling contracts
    and clean-history visibility contract.
    """
    if num_video_frames <= 0 or video_tokens_per_frame <= 0:
        raise ValueError(
            "dual-expert packed coupling mask requires positive video geometry, "
            f"got num_video_frames={num_video_frames}, video_tokens_per_frame={video_tokens_per_frame}."
        )
    if num_action_frames <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "dual-expert packed coupling mask requires positive action geometry, "
            f"got num_action_frames={num_action_frames}, action_tokens_per_frame={action_tokens_per_frame}."
        )
    if chunk_size_frames <= 0:
        raise ValueError(
            f"dual-expert packed coupling mask requires positive chunk_size_frames, got {chunk_size_frames}."
        )

    return build_exact_packed_video_action_coupling_profile(
        num_video_frames=num_video_frames,
        video_tokens_per_frame=video_tokens_per_frame,
        num_action_frames=num_action_frames,
        action_tokens_per_frame=action_tokens_per_frame,
        chunk_size_frames=chunk_size_frames,
        device=device,
        build_dense_masks=build_dense_masks,
        build_flex_masks=build_flex_masks,
        attention_window_size=attention_window_size,
        current_block_coupling=current_block_coupling,
        chunk_origin_frame=int(chunk_origin_frame),
        action_context_mask=action_context_mask,
        history_stream_visibility=history_stream_visibility,
        prefix_condition_frames=int(prefix_condition_frames),
        singleton_chunk_frame=singleton_chunk_frame,
        conditional_history_policy=conditional_history_policy,
    )


def build_dual_expert_packed_coupling_attention_mask(
    **kwargs,
) -> torch.Tensor:
    """Return the dense parallel-stream exact mask for dual-expert packed coupling."""

    kwargs.setdefault("build_dense_masks", True)
    kwargs.setdefault("build_flex_masks", False)
    profile = build_dual_expert_packed_coupling_attention_profile(**kwargs)
    if profile.self_attention_mask is None:
        raise RuntimeError(
            "dual-expert packed coupling dense profile did not produce a self-attention mask."
        )
    return profile.self_attention_mask


__all__ = [
    "build_dual_expert_packed_coupling_attention_mask",
    "build_dual_expert_packed_coupling_attention_profile",
]
