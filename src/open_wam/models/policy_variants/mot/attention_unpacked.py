"""Dense attention layouts for unpacked MoT training and inference."""

from __future__ import annotations

import torch

from open_wam.configs import CurrentBlockCoupling, MoTConditionMode


def build_chunk_causal_video_mask(
    *,
    video_seq_len: int,
    video_tokens_per_frame: int,
    action_chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
    chunk_origin_frame: int = 0,
) -> torch.Tensor:
    """Chunk-causal self-attention mask for a video-only forward.

    Tokens within the same chunk attend each other bidirectionally; past
    chunks are visible; future chunks are hidden. Used by MoT non-joint
    training so that:

    * the teacher-forced clean-video prefill (``prefill_video_kv_cache``)
      produces K/V that respect chunk causality, and
    * the standalone noisy-video flow forward (``_build_video_train_rollout``)
      does not leak future chunks into the video loss.

    Without this mask both forwards default to fully bidirectional self-
    attention, which leaks future frames through the video core's own
    activations and breaks alignment with Method 1's chunked_temporal_exact
    ``kv_frame <= q_frame`` rule.
    """

    if video_seq_len <= 0:
        raise ValueError(f"Expected positive video_seq_len, got {video_seq_len}.")
    if video_tokens_per_frame <= 0 or action_chunk_size_frames <= 0:
        raise ValueError(
            "Chunk-causal video mask requires positive geometry, "
            f"got video_tokens_per_frame={video_tokens_per_frame}, "
            f"action_chunk_size_frames={action_chunk_size_frames}."
        )
    token_ids = torch.arange(video_seq_len, device=device)
    frame_ids = torch.div(token_ids, int(video_tokens_per_frame), rounding_mode="floor")
    chunk_ids = torch.div(
        frame_ids - int(chunk_origin_frame),
        int(action_chunk_size_frames),
        rounding_mode="floor",
    )
    q_chunk = chunk_ids[:, None]
    kv_chunk = chunk_ids[None, :]
    mask = kv_chunk <= q_chunk
    if attention_window_size is not None:
        within_window = (q_chunk - kv_chunk).abs() <= int(attention_window_size)
        mask = mask & within_window
    return mask


def build_mot_attention_mask(
    *,
    video_seq_len: int,
    action_seq_len: int,
    device: torch.device,
    condition_mode: MoTConditionMode | str,
    video_tokens_per_frame: int | None = None,
    video_can_attend_action: bool = False,
    action_tokens_per_frame: int | None = None,
    action_chunk_size_frames: int | None = None,
    clean_video_frames: int | None = None,
    clean_action_frames: int | None = None,
    attention_window_size: int | None = None,
    action_frame_shift: int = 0,
    video_frame_shift: int = 0,
    current_block_coupling: CurrentBlockCoupling | str | None = None,
) -> torch.Tensor:
    """Build a shared MoT mask for the FastWAM conditioning variants.

    ``action_frame_shift`` / ``video_frame_shift`` offset sequential
    token-based frame ids into their actual rotary frame positions when
    constructing block ids. Training leaves both at 0 (packed video and
    action share frame ids 0..T-1). At inference the action cache's first
    entry may sit at a non-zero rotary position (e.g. chunk 0 action lives
    at rotary ``[chunk_frames, 2*chunk_frames)`` because video obs occupies
    ``[0, chunk_frames)`` first), and without the shift the mask
    underestimates action block ids by one block per missing obs chunk,
    preventing action from attending current-chunk clean video through
    ``noise_to_clean: kv_block < q_block``.
    """

    if video_seq_len <= 0 or action_seq_len <= 0:
        raise ValueError(
            "MoT attention mask requires positive video and action lengths, "
            f"got video_seq_len={video_seq_len}, action_seq_len={action_seq_len}."
        )
    resolved_mode = MoTConditionMode(condition_mode)
    resolved_coupling = (
        None
        if current_block_coupling is None
        else CurrentBlockCoupling(current_block_coupling)
    )
    if resolved_coupling in {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }:
        resolved_video_can_attend_action = True
    elif resolved_coupling in {
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
    }:
        resolved_video_can_attend_action = False
    else:
        resolved_video_can_attend_action = bool(video_can_attend_action)

    total_seq_len = video_seq_len + action_seq_len
    mask = torch.zeros(total_seq_len, total_seq_len, device=device, dtype=torch.bool)
    if clean_video_frames is not None and clean_action_frames is not None:
        if video_tokens_per_frame is None:
            raise ValueError(
                "MoT chunked history masking requires `video_tokens_per_frame`."
            )
        if action_tokens_per_frame is None or action_chunk_size_frames is None:
            raise ValueError(
                "MoT chunked history masking requires `action_tokens_per_frame` and `action_chunk_size_frames`, "
                f"got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        if clean_video_frames < 0 or clean_action_frames < 0:
            raise ValueError(
                "MoT chunked history masking requires non-negative clean spans, "
                f"got clean_video_frames={clean_video_frames}, clean_action_frames={clean_action_frames}."
            )
        video_token_ids = torch.arange(video_seq_len, device=device)
        video_frame_ids = torch.div(
            video_token_ids, int(video_tokens_per_frame), rounding_mode="floor"
        )
        action_token_ids = torch.arange(action_seq_len, device=device)
        action_frame_ids = torch.div(
            action_token_ids, int(action_tokens_per_frame), rounding_mode="floor"
        )
        # Shift sequential token-based frame ids into actual rotary frame
        # positions before computing block ids. Clean/noisy membership still
        # uses unshifted token-position counts (``clean_action_frames`` is a
        # count of past frames in the sequence, not a rotary threshold).
        video_block_source = video_frame_ids + int(video_frame_shift)
        action_block_source = action_frame_ids + int(action_frame_shift)
        video_chunk_ids = torch.div(
            video_block_source, int(action_chunk_size_frames), rounding_mode="floor"
        )
        action_chunk_ids = torch.div(
            action_block_source, int(action_chunk_size_frames), rounding_mode="floor"
        )
        video_block_ids = (
            torch.div(
                video_block_source, int(action_chunk_size_frames), rounding_mode="floor"
            )
            * 2
        )
        action_block_ids = (
            torch.div(
                action_block_source,
                int(action_chunk_size_frames),
                rounding_mode="floor",
            )
            * 2
            + 1
        )
        full_block_ids = torch.cat([video_block_ids, action_block_ids], dim=0)
        full_chunk_ids = torch.cat([video_chunk_ids, action_chunk_ids], dim=0)
        full_stream_ids = torch.cat(
            [
                torch.zeros(video_seq_len, device=device, dtype=torch.long),
                torch.ones(action_seq_len, device=device, dtype=torch.long),
            ],
            dim=0,
        )
        full_is_clean = torch.cat(
            [
                video_frame_ids < int(clean_video_frames),
                action_frame_ids < int(clean_action_frames),
            ],
            dim=0,
        )
        q_is_clean = full_is_clean[:, None]
        kv_is_clean = full_is_clean[None, :]
        q_block = full_block_ids[:, None]
        kv_block = full_block_ids[None, :]
        q_chunk = full_chunk_ids[:, None]
        kv_chunk = full_chunk_ids[None, :]
        q_stream = full_stream_ids[:, None]
        kv_stream = full_stream_ids[None, :]
        if resolved_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
            clean_to_clean = (
                q_is_clean
                & kv_is_clean
                & (
                    (kv_chunk < q_chunk)
                    | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
                )
            )
            noise_to_clean = (
                (~q_is_clean)
                & kv_is_clean
                & (
                    (kv_chunk < q_chunk)
                    | (
                        (kv_chunk == q_chunk)
                        & (kv_stream == q_stream)
                        & (kv_block < q_block)
                    )
                )
            )
        else:
            clean_to_clean = q_is_clean & kv_is_clean & (kv_block <= q_block)
            noise_to_clean = (~q_is_clean) & kv_is_clean & (kv_block < q_block)
        if resolved_coupling == CurrentBlockCoupling.JOINT:
            noise_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_chunk == q_chunk)
        elif resolved_coupling == CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION:
            noise_to_noisy = (
                (~q_is_clean)
                & (~kv_is_clean)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 1) & (kv_stream == 0)))
            )
        elif resolved_coupling == CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO:
            noise_to_noisy = (
                (~q_is_clean)
                & (~kv_is_clean)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 0) & (kv_stream == 1)))
            )
        else:
            noise_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_block == q_block)
        mask = clean_to_clean | noise_to_clean | noise_to_noisy
        if attention_window_size is not None:
            within_window = (q_block - kv_block).abs() <= int(attention_window_size)
            mask = mask & within_window
        if not resolved_video_can_attend_action:
            mask[:video_seq_len, video_seq_len:] = False
        return mask
    if clean_video_frames is not None:
        if video_tokens_per_frame is None:
            raise ValueError(
                "MoT joint chunk-causal masking requires `video_tokens_per_frame`."
            )
        if action_tokens_per_frame is None or action_chunk_size_frames is None:
            raise ValueError(
                "MoT joint chunk-causal masking requires `action_tokens_per_frame` and `action_chunk_size_frames`, "
                f"got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        if clean_video_frames < 0:
            raise ValueError(
                f"Expected non-negative `clean_video_frames`, got {clean_video_frames}."
            )
        video_token_ids = torch.arange(video_seq_len, device=device)
        video_frame_ids = torch.div(
            video_token_ids, int(video_tokens_per_frame), rounding_mode="floor"
        )
        action_token_ids = torch.arange(action_seq_len, device=device)
        action_frame_ids = torch.div(
            action_token_ids, int(action_tokens_per_frame), rounding_mode="floor"
        )
        action_frame_ids = action_frame_ids + int(clean_video_frames)

        full_frame_ids = torch.cat([video_frame_ids, action_frame_ids], dim=0)
        full_chunk_ids = torch.div(
            full_frame_ids, int(action_chunk_size_frames), rounding_mode="floor"
        )
        full_stream_ids = torch.cat(
            [
                torch.zeros(video_seq_len, device=device, dtype=torch.long),
                torch.ones(action_seq_len, device=device, dtype=torch.long),
            ],
            dim=0,
        )
        full_is_clean = torch.cat(
            [
                video_frame_ids < int(clean_video_frames),
                torch.zeros(action_seq_len, device=device, dtype=torch.bool),
            ],
            dim=0,
        )
        q_is_clean = full_is_clean[:, None]
        kv_is_clean = full_is_clean[None, :]
        q_chunk = full_chunk_ids[:, None]
        kv_chunk = full_chunk_ids[None, :]
        q_stream = full_stream_ids[:, None]
        kv_stream = full_stream_ids[None, :]
        q_frame = full_frame_ids[:, None]
        kv_frame = full_frame_ids[None, :]
        if resolved_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
            clean_to_clean = (
                q_is_clean
                & kv_is_clean
                & (
                    (kv_chunk < q_chunk)
                    | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
                )
            )
            noisy_to_clean = (
                (~q_is_clean)
                & kv_is_clean
                & (
                    (kv_chunk < q_chunk)
                    | (
                        (kv_chunk == q_chunk)
                        & (kv_stream == q_stream)
                        & (kv_frame < q_frame)
                    )
                )
            )
        else:
            clean_to_clean = q_is_clean & kv_is_clean & (kv_frame <= q_frame)
            noisy_to_clean = (~q_is_clean) & kv_is_clean & (kv_chunk < q_chunk)
        if resolved_coupling == CurrentBlockCoupling.JOINT:
            noisy_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_chunk == q_chunk)
        elif resolved_coupling == CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION:
            noisy_to_noisy = (
                (~q_is_clean)
                & (~kv_is_clean)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 1) & (kv_stream == 0)))
            )
        elif resolved_coupling == CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO:
            noisy_to_noisy = (
                (~q_is_clean)
                & (~kv_is_clean)
                & (kv_chunk == q_chunk)
                & ((q_stream == kv_stream) | ((q_stream == 0) & (kv_stream == 1)))
            )
        else:
            noisy_to_noisy = (~q_is_clean) & (~kv_is_clean) & (kv_chunk == q_chunk)
        mask = clean_to_clean | noisy_to_clean | noisy_to_noisy
        if attention_window_size is not None:
            within_window = (q_chunk - kv_chunk).abs() <= int(attention_window_size)
            mask = mask & within_window
        if not resolved_video_can_attend_action:
            mask[:video_seq_len, video_seq_len:] = False
        return mask
    mask[:video_seq_len, :video_seq_len] = True
    if action_tokens_per_frame is not None or action_chunk_size_frames is not None:
        if action_tokens_per_frame is None or action_chunk_size_frames is None:
            raise ValueError(
                "MoT chunk-causal action masking requires both `action_tokens_per_frame` "
                f"and `action_chunk_size_frames`, got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        if action_tokens_per_frame <= 0 or action_chunk_size_frames <= 0:
            raise ValueError(
                "MoT chunk-causal action masking requires positive action geometry, "
                f"got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        action_token_ids = torch.arange(action_seq_len, device=device)
        action_frame_ids = torch.div(
            action_token_ids, int(action_tokens_per_frame), rounding_mode="floor"
        )
        action_chunk_ids = torch.div(
            action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor"
        )
        mask[video_seq_len:, video_seq_len:] = (
            action_chunk_ids[:, None] >= action_chunk_ids[None, :]
        )
    else:
        mask[video_seq_len:, video_seq_len:] = True
    if resolved_mode == MoTConditionMode.FIRST_FRAME:
        if video_tokens_per_frame is None:
            raise ValueError(
                "MoT first-frame conditioning requires `video_tokens_per_frame`."
            )
        visible_video = min(video_tokens_per_frame, video_seq_len)
    elif resolved_mode in {
        MoTConditionMode.FULL_VIDEO,
        MoTConditionMode.TEACHER_FORCING_COND_VIDEO,
    }:
        visible_video = video_seq_len
    else:  # pragma: no cover - enum guard
        raise ValueError(f"Unsupported MoT condition mode {resolved_mode!r}.")
    mask[video_seq_len:, :visible_video] = True
    if resolved_video_can_attend_action:
        mask[:video_seq_len, video_seq_len:] = True
    return mask


__all__ = [
    "build_chunk_causal_video_mask",
    "build_mot_attention_mask",
]
