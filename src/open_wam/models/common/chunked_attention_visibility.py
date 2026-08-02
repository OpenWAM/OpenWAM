"""Canonical token-pair visibility for chunked exact attention."""

from __future__ import annotations

import torch

from open_wam.models.common.attention_contracts import (
    ACTION_NOISY_TO_VIDEO_COUPLING,
    ACTION_THEN_VIDEO_COUPLING,
    CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    DECOUPLED_SAME_STEP_COUPLING,
    HISTORY_STREAM_VISIBILITY_FULL,
    HISTORY_STREAM_VISIBILITY_VIDEO_ONLY,
    HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY,
    JOINT_COUPLING,
    VIDEO_NOISY_TO_ACTION_COUPLING,
)
from open_wam.models.common.packed_token_layout import PackedTokenStream


def _effective_frame_ids_for_singleton_cutoff(
    frame_ids: torch.Tensor,
    stream_ids: torch.Tensor,
    *,
    prefix_condition_frames: int,
    singleton_chunk_frame: int | None,
) -> torch.Tensor:
    """Return sample-frame ids used by the CF t0 singleton history cutoff."""

    if singleton_chunk_frame is None:
        return frame_ids
    effective_frame_ids = frame_ids
    if int(prefix_condition_frames) > 0:
        video_tokens = stream_ids == int(PackedTokenStream.VIDEO)
        prefix_video_tokens = video_tokens & (frame_ids < int(prefix_condition_frames))
        shifted_video_frame_ids = (frame_ids - int(prefix_condition_frames)).clamp_min(
            0
        )
        effective_frame_ids = torch.where(
            video_tokens, shifted_video_frame_ids, effective_frame_ids
        )
        effective_frame_ids = torch.where(
            prefix_video_tokens,
            torch.full_like(effective_frame_ids, int(singleton_chunk_frame) - 1),
            effective_frame_ids,
        )
    return effective_frame_ids


def _previous_boundary_frame_ids(
    frame_ids: torch.Tensor,
    *,
    chunk_origin_frame: int,
    chunk_size: int,
) -> torch.Tensor:
    """Return the immediately previous chunk-boundary frame for each query frame."""

    chunk_ids = torch.div(
        frame_ids - int(chunk_origin_frame),
        max(1, int(chunk_size)),
        rounding_mode="floor",
    )
    return int(chunk_origin_frame) + chunk_ids * max(1, int(chunk_size)) - 1


def _build_chunked_self_attention_visibility(
    *,
    q_seq: torch.Tensor,
    kv_seq: torch.Tensor,
    q_block_id: torch.Tensor,
    kv_block_id: torch.Tensor,
    q_chunk: torch.Tensor,
    kv_chunk: torch.Tensor,
    q_noise: torch.Tensor,
    kv_noise: torch.Tensor,
    q_stream: torch.Tensor,
    kv_stream: torch.Tensor,
    q_effective_frame: torch.Tensor,
    kv_effective_frame: torch.Tensor,
    q_valid: torch.Tensor,
    kv_valid: torch.Tensor,
    window_size: int,
    chunk_size: int,
    chunk_origin_frame: int,
    prefix_condition_frames: int,
    singleton_chunk_frame: int | None,
    current_block_coupling: str,
    history_stream_visibility: str,
    conditional_history_policy: str,
) -> torch.Tensor:
    """Evaluate the exact visibility law for broadcastable query/KV tensors.

    Dense masks pass column and row tensors. FlexAttention passes scalar index
    lookups. Keeping this predicate representation-neutral prevents the two
    backends from acquiring different method semantics.
    """

    same_seq = (q_seq == kv_seq) & (q_seq >= 0) & (kv_seq >= 0) & q_valid & kv_valid
    if singleton_chunk_frame is None:
        singleton_history_ok = torch.ones_like(q_seq, dtype=torch.bool)
    else:
        singleton_history_ok = (q_effective_frame < int(singleton_chunk_frame)) | (
            kv_effective_frame >= int(singleton_chunk_frame)
        )

    if history_stream_visibility == HISTORY_STREAM_VISIBILITY_FULL:
        history_stream_ok = torch.ones_like(q_seq, dtype=torch.bool)
    elif (
        history_stream_visibility == HISTORY_STREAM_VISIBILITY_VIDEO_QUERIES_VIDEO_ONLY
    ):
        history_stream_ok = (q_stream == kv_stream) | (q_stream == 1)
    elif history_stream_visibility == HISTORY_STREAM_VISIBILITY_VIDEO_ONLY:
        history_stream_ok = kv_stream == 0
    else:  # pragma: no cover - normalized by the profile builder
        raise ValueError(
            f"Unsupported history stream visibility {history_stream_visibility!r}."
        )

    if (
        conditional_history_policy
        == CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY
    ):
        boundary_frame = _previous_boundary_frame_ids(
            q_effective_frame,
            chunk_origin_frame=chunk_origin_frame,
            chunk_size=chunk_size,
        )
        history_stream_ok = (kv_stream == int(PackedTokenStream.VIDEO)) & (
            kv_effective_frame == boundary_frame
        )

    if (
        prefix_condition_frames > 0
        or current_block_coupling == DECOUPLED_SAME_STEP_COUPLING
    ):
        clean_to_clean = (
            (q_noise == 1)
            & (kv_noise == 1)
            & (
                ((kv_chunk < q_chunk) & history_stream_ok)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
            )
        )
    else:
        clean_to_clean = (
            (q_noise == 1)
            & (kv_noise == 1)
            & (
                ((kv_chunk < q_chunk) & history_stream_ok)
                | ((kv_chunk == q_chunk) & (kv_block_id <= q_block_id))
            )
        )

    joint_like_couplings = {
        JOINT_COUPLING,
        DECOUPLED_SAME_STEP_COUPLING,
        VIDEO_NOISY_TO_ACTION_COUPLING,
        ACTION_NOISY_TO_VIDEO_COUPLING,
    }
    prefix_action_then_video = (
        prefix_condition_frames > 0
        and current_block_coupling == ACTION_THEN_VIDEO_COUPLING
    )
    if current_block_coupling in joint_like_couplings or prefix_action_then_video:
        noise_to_clean = (
            (q_noise == 0) & (kv_noise == 1) & (kv_chunk < q_chunk) & history_stream_ok
        )
        if prefix_action_then_video:
            noise_to_clean = noise_to_clean | (
                (q_noise == 0)
                & (q_stream == 0)
                & (kv_noise == 1)
                & (kv_stream == 1)
                & (kv_chunk == q_chunk)
            )
    else:
        in_history = kv_chunk < q_chunk
        in_current_chunk_earlier = (kv_chunk == q_chunk) & (kv_block_id < q_block_id)
        noise_to_clean = (
            (q_noise == 0)
            & (kv_noise == 1)
            & ((in_history & history_stream_ok) | in_current_chunk_earlier)
        )

    if current_block_coupling == JOINT_COUPLING:
        noise_to_noise = (q_noise == 0) & (kv_noise == 0) & (kv_chunk == q_chunk)
    elif current_block_coupling == VIDEO_NOISY_TO_ACTION_COUPLING:
        noise_to_noise = (
            (q_noise == 0)
            & (kv_noise == 0)
            & (kv_chunk == q_chunk)
            & ((q_stream == kv_stream) | ((q_stream == 1) & (kv_stream == 0)))
        )
    elif current_block_coupling == ACTION_NOISY_TO_VIDEO_COUPLING:
        noise_to_noise = (
            (q_noise == 0)
            & (kv_noise == 0)
            & (kv_chunk == q_chunk)
            & ((q_stream == kv_stream) | ((q_stream == 0) & (kv_stream == 1)))
        )
    elif prefix_condition_frames > 0:
        noise_to_noise = (
            (q_noise == 0)
            & (kv_noise == 0)
            & (kv_chunk == q_chunk)
            & (q_stream == kv_stream)
        )
    else:
        noise_to_noise = (q_noise == 0) & (kv_noise == 0) & (kv_block_id == q_block_id)

    within_window = (q_block_id - kv_block_id).abs() <= int(window_size)
    return (
        same_seq
        & within_window
        & singleton_history_ok
        & (clean_to_clean | noise_to_clean | noise_to_noise)
    )


def _build_chunked_cross_attention_visibility(
    *,
    q_seq: torch.Tensor,
    text_seq: torch.Tensor,
    q_chunk: torch.Tensor,
    text_position: torch.Tensor,
    q_valid: torch.Tensor,
    base_text_token_count: int,
    proprio_context_token_count: int,
) -> torch.Tensor:
    """Evaluate sample and per-chunk text/proprio visibility."""

    same_text_sample = (q_seq == text_seq) & (q_seq >= 0) & (text_seq >= 0) & q_valid
    if proprio_context_token_count <= 0:
        return same_text_sample
    base_text_visible = text_position < base_text_token_count
    proprio_index = text_position - base_text_token_count
    proprio_visible = (
        (proprio_index >= 0)
        & (proprio_index < proprio_context_token_count)
        & (proprio_index == q_chunk)
    )
    return same_text_sample & (base_text_visible | proprio_visible)


__all__: list[str] = []
