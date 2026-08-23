"""Exact packed video/action attention layouts for DualExpert execution."""

from __future__ import annotations

import torch

from open_wam.configs import CurrentBlockCoupling, HistoryStreamVisibility
from open_wam.models.common.attention_contracts import PreparedAttentionProfile
from open_wam.models.common.coupling_profiles import (
    build_exact_packed_video_action_coupling_profile,
)


def build_packed_action_attention_mask(
    *,
    num_video_frames: int,
    video_tokens_per_frame: int,
    num_action_frames: int,
    action_tokens_per_frame: int,
    action_chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
    current_block_coupling: CurrentBlockCoupling
    | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
) -> torch.Tensor:
    """Packed action-expert attention mask (parallel-stream-style).

    .. warning::
       This is **not** the mask used by dual-expert packed-coupling inference.
       That path calls :func:`build_dual_expert_packed_coupling_attention_profile`
       below, whose key layout is the four-segment
       ``[V_noisy | V_clean | A_noisy | A_clean]`` and whose action queries *do*
       attend ``V_noisy`` for the current chunk. The two builders live in this
       module under similar names and describe incompatible layouts; read the
       call site before trusting either docstring.

    Key/Value layout: ``[V_clean (T_v*ppF_v) | A_noisy (T_a*ppF_a) | A_clean (T_a*ppF_a)]``.
    Query layout: ``[A_noisy (T_a*ppF_a) | A_clean (T_a*ppF_a)]``. The video
    side contributes only its clean copy -- per parallel-stream, action queries
    never attend ``V_noisy`` (the frame-id parity prevents it regardless, so
    dropping the row saves memory and compute).

    Block ids follow parallel-stream: video chunk B -> block 2B (even); action
    chunk B -> block 2B+1 (odd). Rules:

    * ``clean_to_clean``: ``q_noise=1 & kv_noise=1 & kv_block <= q_block``
    * ``noise_to_clean``: ``q_noise=0 & kv_noise=1 & kv_block < q_block``
    * ``noise_to_noisy``: ``q_noise=0 & kv_noise=0 & kv_block == q_block``

    ``decoupled_same_step`` keeps the same layout but removes same-chunk
    cross-stream clean-video visibility from action queries.

    Returns a ``[2T_a*ppF_a, T_v*ppF_v + 2T_a*ppF_a]`` boolean mask.
    """
    coupling = CurrentBlockCoupling(current_block_coupling)

    if num_video_frames <= 0 or video_tokens_per_frame <= 0:
        raise ValueError(
            "Packed action mask requires positive video geometry, "
            f"got num_video_frames={num_video_frames}, video_tokens_per_frame={video_tokens_per_frame}."
        )
    if num_action_frames <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "Packed action mask requires positive action geometry, "
            f"got num_action_frames={num_action_frames}, action_tokens_per_frame={action_tokens_per_frame}."
        )
    if action_chunk_size_frames <= 0:
        raise ValueError(
            f"Packed action mask requires positive action_chunk_size_frames, got {action_chunk_size_frames}."
        )
    video_seq_len = int(num_video_frames) * int(video_tokens_per_frame)
    action_seq_len = int(num_action_frames) * int(action_tokens_per_frame)

    # Video K block ids (all clean, even blocks).
    video_token_ids = torch.arange(video_seq_len, device=device)
    video_frame_ids = torch.div(
        video_token_ids, int(video_tokens_per_frame), rounding_mode="floor"
    )
    video_block_ids = (
        torch.div(video_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
        * 2
    )
    video_chunk_ids = torch.div(
        video_frame_ids, int(action_chunk_size_frames), rounding_mode="floor"
    )
    # Action token ids per single copy.
    action_token_ids = torch.arange(action_seq_len, device=device)
    action_frame_ids = torch.div(
        action_token_ids, int(action_tokens_per_frame), rounding_mode="floor"
    )
    action_block_ids_single = (
        torch.div(
            action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor"
        )
        * 2
        + 1
    )
    action_chunk_ids_single = torch.div(
        action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor"
    )
    # Key side: V_clean + A_noisy + A_clean. Query side: A_noisy + A_clean.
    kv_block_ids = torch.cat(
        [video_block_ids, action_block_ids_single, action_block_ids_single], dim=0
    )
    kv_chunk_ids = torch.cat(
        [video_chunk_ids, action_chunk_ids_single, action_chunk_ids_single], dim=0
    )
    kv_stream_ids = torch.cat(
        [
            torch.zeros(video_seq_len, device=device, dtype=torch.long),
            torch.ones(action_seq_len, device=device, dtype=torch.long),
            torch.ones(action_seq_len, device=device, dtype=torch.long),
        ],
        dim=0,
    )
    kv_is_clean = torch.cat(
        [
            torch.ones(video_seq_len, device=device, dtype=torch.bool),
            torch.zeros(action_seq_len, device=device, dtype=torch.bool),
            torch.ones(action_seq_len, device=device, dtype=torch.bool),
        ],
        dim=0,
    )
    q_block_ids = torch.cat([action_block_ids_single, action_block_ids_single], dim=0)
    q_chunk_ids = torch.cat([action_chunk_ids_single, action_chunk_ids_single], dim=0)
    q_stream_ids = torch.ones(2 * action_seq_len, device=device, dtype=torch.long)
    q_is_clean = torch.cat(
        [
            torch.zeros(action_seq_len, device=device, dtype=torch.bool),
            torch.ones(action_seq_len, device=device, dtype=torch.bool),
        ],
        dim=0,
    )

    q_b = q_block_ids[:, None]
    kv_b = kv_block_ids[None, :]
    q_chunk = q_chunk_ids[:, None]
    kv_chunk = kv_chunk_ids[None, :]
    q_stream = q_stream_ids[:, None]
    kv_stream = kv_stream_ids[None, :]
    q_c = q_is_clean[:, None]
    kv_c = kv_is_clean[None, :]
    if coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
        clean_to_clean = (
            q_c
            & kv_c
            & ((kv_chunk < q_chunk) | ((kv_chunk == q_chunk) & (kv_stream == q_stream)))
        )
        noise_to_clean = (
            (~q_c)
            & kv_c
            & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_b < q_b))
            )
        )
    else:
        clean_to_clean = q_c & kv_c & (kv_b <= q_b)
        noise_to_clean = (~q_c) & kv_c & (kv_b < q_b)
    noise_to_noisy = (~q_c) & (~kv_c) & (kv_b == q_b)
    mask = clean_to_clean | noise_to_clean | noise_to_noisy
    if attention_window_size is not None:
        within_window = (q_b - kv_b).abs() <= int(attention_window_size)
        mask = mask & within_window
    return mask


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
        HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
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
    "build_packed_action_attention_mask",
]
