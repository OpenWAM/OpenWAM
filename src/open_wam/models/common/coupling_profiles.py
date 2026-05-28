from __future__ import annotations

import torch

from open_wam.configs import CurrentBlockCoupling
from open_wam.models.common.attention_profiles import (
    PreparedAttentionProfile,
    build_chunked_temporal_exact_attention_profile,
)


def build_exact_packed_video_action_coupling_profile(
    *,
    num_video_frames: int,
    video_tokens_per_frame: int,
    num_action_frames: int,
    action_tokens_per_frame: int,
    chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
    current_block_coupling: CurrentBlockCoupling | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
    build_dense_masks: bool | None = None,
    build_flex_masks: bool | None = None,
    preserve_video_pretrain_history: bool = True,
    history_stream_visibility: str | None = None,
    chunk_origin_frame: int = 0,
    action_context_mask: torch.Tensor | None = None,
    prefix_condition_frames: int = 0,
) -> PreparedAttentionProfile:
    """Build the exact-runtime packed `[V_noisy,V_clean,A_noisy,A_clean]` profile.

    This is a four-stream adapter over the chunked temporal exact attention
    profile. It intentionally fixes exact-runtime metadata: no padding,
    unit patch geometry, and one text token.
    """

    if num_video_frames <= 0 or video_tokens_per_frame <= 0:
        raise ValueError(
            "Packed video/action coupling profile requires positive video geometry, "
            f"got num_video_frames={num_video_frames}, video_tokens_per_frame={video_tokens_per_frame}."
        )
    if num_action_frames <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "Packed video/action coupling profile requires positive action geometry, "
            f"got num_action_frames={num_action_frames}, action_tokens_per_frame={action_tokens_per_frame}."
        )
    if chunk_size_frames <= 0:
        raise ValueError(
            f"Packed video/action coupling profile requires positive chunk_size_frames, got {chunk_size_frames}."
        )

    resolved_build_dense = device.type != "cuda" if build_dense_masks is None else bool(build_dense_masks)
    resolved_build_flex = device.type == "cuda" if build_flex_masks is None else bool(build_flex_masks)
    return build_chunked_temporal_exact_attention_profile(
        latent_shape=(
            1,
            1,
            int(num_video_frames),
            1,
            int(video_tokens_per_frame),
        ),
        action_shape=(
            1,
            1,
            int(num_action_frames),
            1,
            int(action_tokens_per_frame),
        ),
        padded_length=0,
        chunk_size=int(chunk_size_frames),
        window_size=(
            int(attention_window_size)
            if attention_window_size is not None
            else max(int(num_video_frames), int(num_action_frames)) * 2
        ),
        patch_size=(1, 1, 1),
        text_token_count=1,
        device=device,
        build_dense_masks=resolved_build_dense,
        build_flex_masks=resolved_build_flex,
        current_block_coupling=CurrentBlockCoupling(current_block_coupling).value,
        chunk_origin_frame=int(chunk_origin_frame),
        action_context_mask=action_context_mask,
        preserve_video_pretrain_history=preserve_video_pretrain_history,
        history_stream_visibility=history_stream_visibility,
        prefix_condition_frames=int(prefix_condition_frames),
    )


def build_packed_video_action_coupling_profile(**kwargs) -> PreparedAttentionProfile:
    """Compatibility alias for the exact-runtime packed coupling profile."""

    return build_exact_packed_video_action_coupling_profile(**kwargs)
