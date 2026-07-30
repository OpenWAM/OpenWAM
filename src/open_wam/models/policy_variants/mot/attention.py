"""Parameter-free attention layout builders for MoT policies."""

from __future__ import annotations

import torch

from open_wam.configs import CurrentBlockCoupling, MoTConditionMode
from open_wam.models.common.attention_profiles import PreparedAttentionProfile
from open_wam.models.common.coupling_profiles import (
    build_exact_packed_video_action_coupling_profile,
)


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


def build_packed_action_attention_mask(
    *,
    num_video_frames: int,
    video_tokens_per_frame: int,
    num_action_frames: int,
    action_tokens_per_frame: int,
    action_chunk_size_frames: int,
    device: torch.device,
    attention_window_size: int | None = None,
    current_block_coupling: CurrentBlockCoupling | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
) -> torch.Tensor:
    """Packed action-expert attention mask (Method-1-style).

    Key/Value layout: ``[V_clean (T_v*ppF_v) | A_noisy (T_a*ppF_a) | A_clean (T_a*ppF_a)]``.
    Query layout: ``[A_noisy (T_a*ppF_a) | A_clean (T_a*ppF_a)]``. The video
    side contributes only its clean copy -- per Method 1, action queries
    never attend ``V_noisy`` (the frame-id parity prevents it regardless, so
    dropping the row saves memory and compute).

    Block ids follow Method 1: video chunk B -> block 2B (even); action
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
    video_frame_ids = torch.div(video_token_ids, int(video_tokens_per_frame), rounding_mode="floor")
    video_block_ids = (
        torch.div(video_frame_ids, int(action_chunk_size_frames), rounding_mode="floor") * 2
    )
    video_chunk_ids = torch.div(video_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
    # Action token ids per single copy.
    action_token_ids = torch.arange(action_seq_len, device=device)
    action_frame_ids = torch.div(action_token_ids, int(action_tokens_per_frame), rounding_mode="floor")
    action_block_ids_single = (
        torch.div(action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor") * 2 + 1
    )
    action_chunk_ids_single = torch.div(action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
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
        clean_to_clean = q_c & kv_c & (
            (kv_chunk < q_chunk)
            | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
        )
        noise_to_clean = (~q_c) & kv_c & (
            (kv_chunk < q_chunk)
            | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_b < q_b))
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


def build_mot_packed_coupling_attention_profile(
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
    chunk_origin_frame: int = 0,
    action_context_mask: torch.Tensor | None = None,
    history_stream_visibility: str | None = None,
    prefix_condition_frames: int = 0,
    singleton_chunk_frame: int | None = None,
    conditional_history_policy: str | None = None,
) -> PreparedAttentionProfile:
    """Build the Method-1 exact attention profile for M5 packed coupling.

    Query/key layout is ``[V_noisy, V_clean, A_noisy, A_clean]``. The mask
    semantics are intentionally sourced from Method 1's chunked temporal exact
    profile, so M5's two-expert topology uses the same six coupling contracts
    and preserve-video-pretrain-history rule.
    """
    if num_video_frames <= 0 or video_tokens_per_frame <= 0:
        raise ValueError(
            "M5 packed coupling mask requires positive video geometry, "
            f"got num_video_frames={num_video_frames}, video_tokens_per_frame={video_tokens_per_frame}."
        )
    if num_action_frames <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "M5 packed coupling mask requires positive action geometry, "
            f"got num_action_frames={num_action_frames}, action_tokens_per_frame={action_tokens_per_frame}."
        )
    if chunk_size_frames <= 0:
        raise ValueError(f"M5 packed coupling mask requires positive chunk_size_frames, got {chunk_size_frames}.")

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
        preserve_video_pretrain_history=True,
        history_stream_visibility=history_stream_visibility,
        prefix_condition_frames=int(prefix_condition_frames),
        singleton_chunk_frame=singleton_chunk_frame,
        conditional_history_policy=conditional_history_policy,
    )


def build_mot_packed_coupling_attention_mask(
    **kwargs,
) -> torch.Tensor:
    """Return the dense Method-1 exact mask for M5 packed coupling."""

    kwargs.setdefault("build_dense_masks", True)
    kwargs.setdefault("build_flex_masks", False)
    profile = build_mot_packed_coupling_attention_profile(**kwargs)
    if profile.self_attention_mask is None:
        raise RuntimeError("M5 packed coupling dense profile did not produce a self-attention mask.")
    return profile.self_attention_mask


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
    resolved_coupling = None if current_block_coupling is None else CurrentBlockCoupling(current_block_coupling)
    if resolved_coupling in {CurrentBlockCoupling.JOINT, CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO}:
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
            raise ValueError("MoT chunked history masking requires `video_tokens_per_frame`.")
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
        video_frame_ids = torch.div(video_token_ids, int(video_tokens_per_frame), rounding_mode="floor")
        action_token_ids = torch.arange(action_seq_len, device=device)
        action_frame_ids = torch.div(action_token_ids, int(action_tokens_per_frame), rounding_mode="floor")
        # Shift sequential token-based frame ids into actual rotary frame
        # positions before computing block ids. Clean/noisy membership still
        # uses unshifted token-position counts (``clean_action_frames`` is a
        # count of past frames in the sequence, not a rotary threshold).
        video_block_source = video_frame_ids + int(video_frame_shift)
        action_block_source = action_frame_ids + int(action_frame_shift)
        video_chunk_ids = torch.div(video_block_source, int(action_chunk_size_frames), rounding_mode="floor")
        action_chunk_ids = torch.div(action_block_source, int(action_chunk_size_frames), rounding_mode="floor")
        video_block_ids = torch.div(video_block_source, int(action_chunk_size_frames), rounding_mode="floor") * 2
        action_block_ids = torch.div(action_block_source, int(action_chunk_size_frames), rounding_mode="floor") * 2 + 1
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
            clean_to_clean = q_is_clean & kv_is_clean & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
            )
            noise_to_clean = (~q_is_clean) & kv_is_clean & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_block < q_block))
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
            raise ValueError("MoT joint chunk-causal masking requires `video_tokens_per_frame`.")
        if action_tokens_per_frame is None or action_chunk_size_frames is None:
            raise ValueError(
                "MoT joint chunk-causal masking requires `action_tokens_per_frame` and `action_chunk_size_frames`, "
                f"got action_tokens_per_frame={action_tokens_per_frame}, "
                f"action_chunk_size_frames={action_chunk_size_frames}."
            )
        if clean_video_frames < 0:
            raise ValueError(f"Expected non-negative `clean_video_frames`, got {clean_video_frames}.")
        video_token_ids = torch.arange(video_seq_len, device=device)
        video_frame_ids = torch.div(video_token_ids, int(video_tokens_per_frame), rounding_mode="floor")
        action_token_ids = torch.arange(action_seq_len, device=device)
        action_frame_ids = torch.div(action_token_ids, int(action_tokens_per_frame), rounding_mode="floor")
        action_frame_ids = action_frame_ids + int(clean_video_frames)

        full_frame_ids = torch.cat([video_frame_ids, action_frame_ids], dim=0)
        full_chunk_ids = torch.div(full_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
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
            clean_to_clean = q_is_clean & kv_is_clean & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
            )
            noisy_to_clean = (~q_is_clean) & kv_is_clean & (
                (kv_chunk < q_chunk)
                | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_frame < q_frame))
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
        action_frame_ids = torch.div(action_token_ids, int(action_tokens_per_frame), rounding_mode="floor")
        action_chunk_ids = torch.div(action_frame_ids, int(action_chunk_size_frames), rounding_mode="floor")
        mask[video_seq_len:, video_seq_len:] = action_chunk_ids[:, None] >= action_chunk_ids[None, :]
    else:
        mask[video_seq_len:, video_seq_len:] = True
    if resolved_mode == MoTConditionMode.FIRST_FRAME:
        if video_tokens_per_frame is None:
            raise ValueError("MoT first-frame conditioning requires `video_tokens_per_frame`.")
        visible_video = min(video_tokens_per_frame, video_seq_len)
    elif resolved_mode in {MoTConditionMode.FULL_VIDEO, MoTConditionMode.TEACHER_FORCING_COND_VIDEO}:
        visible_video = video_seq_len
    else:  # pragma: no cover - enum guard
        raise ValueError(f"Unsupported MoT condition mode {resolved_mode!r}.")
    mask[video_seq_len:, :visible_video] = True
    if resolved_video_can_attend_action:
        mask[:video_seq_len, video_seq_len:] = True
    return mask


def build_mot_inference_action_attention_mask(
    *,
    video_seq_len: int,
    past_action_seq_len: int,
    current_action_seq_len: int,
    video_tokens_per_frame: int,
    action_tokens_per_frame: int,
    chunk_size_frames: int,
    window_size_frames: int,
    device: torch.device,
    video_can_attend_action: bool = False,
    video_frame_start: int = 0,
    past_action_frame_start: int = 0,
    current_action_frame_start: int | None = None,
    chunk_origin_frame: int = 0,
    current_block_coupling: CurrentBlockCoupling | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
) -> torch.Tensor:
    """Inference-only MoT action attention mask (Method-1 byte-aligned).

    Mirrors `build_chunked_temporal_exact_attention_profile` for the
    inference layout `[video_cache; past_action_cache; current_action]`:

      * chunk ids are computed relative to ``chunk_origin_frame``
      * block ids: video at chunk*2, action at chunk*2 + 1 (so the same chunk
        gets adjacent ids and `(q - kv).abs() <= window_size` collapses to the
        same window for both streams)
      * causal between clean: ``kv <= q`` (allows same-frame)
      * noisy queries past clean: ``kv < q``
      * noisy queries same-frame noisy: ``kv == q``
      * within_window on block-id delta, with ``window_size_frames`` passed
        through unchanged from ``training_config.window_size`` (matching
        Method 1's `input_dict["window_size"]`).

    All cached tokens are clean (already-denoised K/V written at past chunks
    or the current chunk's clean obs/pred); only the fresh current-action
    tokens are noisy. There is no "noisy video" stream at inference because
    the action expert never denoises video.

    ``decoupled_same_step`` is a safety mask for ablations where action should
    not read current generated clean-video K/V. The rollout also defers that
    video K/V write, so this branch mainly guards accidental same-step cache
    exposure.
    """
    coupling = CurrentBlockCoupling(current_block_coupling)

    if video_seq_len < 0 or past_action_seq_len < 0 or current_action_seq_len <= 0:
        raise ValueError(
            "MoT inference action mask requires non-negative cache lengths and a positive "
            f"current chunk, got video_seq_len={video_seq_len}, "
            f"past_action_seq_len={past_action_seq_len}, "
            f"current_action_seq_len={current_action_seq_len}."
        )
    if video_tokens_per_frame <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "MoT inference action mask requires positive tokens-per-frame, "
            f"got video_tokens_per_frame={video_tokens_per_frame}, "
            f"action_tokens_per_frame={action_tokens_per_frame}."
        )
    if chunk_size_frames <= 0 or window_size_frames <= 0:
        raise ValueError(
            "MoT inference action mask requires positive chunk/window sizes, "
            f"got chunk_size_frames={chunk_size_frames}, "
            f"window_size_frames={window_size_frames}."
        )
    # Slot-pool capacity (`(attn_window // 2) * (latent_token_per_chunk +
    # action_token_per_chunk)`) is sized in tokens, not frames. Method 1
    # writes both streams so eviction stays chunk-aligned; Method 5 only
    # writes video, so eviction can leave a partial leading frame in the
    # video cache. Don't reject that — floor-division below assigns the
    # partial frame to the oldest block_id, which is correct for the
    # relative within_window check the mask actually uses.
    if current_action_seq_len % action_tokens_per_frame != 0:
        raise ValueError(
            "MoT inference action mask expects current_action_seq_len divisible by action_tokens_per_frame, "
            f"got current_action_seq_len={current_action_seq_len}, action_tokens_per_frame={action_tokens_per_frame}."
        )

    chunk_origin_frame = int(chunk_origin_frame)
    video_token_ids = torch.arange(video_seq_len, device=device)
    video_frame_ids = torch.div(video_token_ids, int(video_tokens_per_frame), rounding_mode="floor") + int(
        video_frame_start
    )
    video_chunk_ids = torch.div(
        video_frame_ids - chunk_origin_frame,
        int(chunk_size_frames),
        rounding_mode="floor",
    )
    video_block_ids = video_chunk_ids * 2

    past_action_token_ids = torch.arange(past_action_seq_len, device=device)
    past_action_frame_ids = torch.div(
        past_action_token_ids, int(action_tokens_per_frame), rounding_mode="floor"
    ) + int(past_action_frame_start)
    past_action_chunk_ids = torch.div(
        past_action_frame_ids - chunk_origin_frame,
        int(chunk_size_frames),
        rounding_mode="floor",
    )
    past_action_block_ids = past_action_chunk_ids * 2 + 1

    past_action_frames_count = past_action_seq_len // int(action_tokens_per_frame)
    if current_action_frame_start is None:
        current_action_frame_start = int(past_action_frame_start) + int(past_action_frames_count)
    current_action_token_ids = torch.arange(current_action_seq_len, device=device)
    current_action_frame_ids = torch.div(
        current_action_token_ids, int(action_tokens_per_frame), rounding_mode="floor"
    ) + int(current_action_frame_start)
    current_action_chunk_ids = torch.div(
        current_action_frame_ids - chunk_origin_frame,
        int(chunk_size_frames),
        rounding_mode="floor",
    )
    current_action_block_ids = current_action_chunk_ids * 2 + 1

    block_ids = torch.cat(
        [video_block_ids, past_action_block_ids, current_action_block_ids], dim=0
    ).to(dtype=torch.long)
    chunk_ids = torch.cat(
        [video_chunk_ids, past_action_chunk_ids, current_action_chunk_ids], dim=0
    ).to(dtype=torch.long)
    stream_ids = torch.cat(
        [
            torch.zeros(video_seq_len, dtype=torch.long, device=device),
            torch.ones(past_action_seq_len, dtype=torch.long, device=device),
            torch.ones(current_action_seq_len, dtype=torch.long, device=device),
        ],
        dim=0,
    )
    is_clean = torch.cat(
        [
            torch.ones(video_seq_len, dtype=torch.bool, device=device),
            torch.ones(past_action_seq_len, dtype=torch.bool, device=device),
            torch.zeros(current_action_seq_len, dtype=torch.bool, device=device),
        ],
        dim=0,
    )

    q_frame = block_ids[:, None]
    kv_frame = block_ids[None, :]
    q_chunk = chunk_ids[:, None]
    kv_chunk = chunk_ids[None, :]
    q_stream = stream_ids[:, None]
    kv_stream = stream_ids[None, :]
    q_clean = is_clean[:, None]
    kv_clean = is_clean[None, :]

    if coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP:
        clean_to_clean = q_clean & kv_clean & (
            (kv_chunk < q_chunk)
            | ((kv_chunk == q_chunk) & (kv_stream == q_stream))
        )
        noise_to_clean = (~q_clean) & kv_clean & (
            (kv_chunk < q_chunk)
            | ((kv_chunk == q_chunk) & (kv_stream == q_stream) & (kv_frame < q_frame))
        )
    else:
        clean_to_clean = q_clean & kv_clean & (kv_frame <= q_frame)
        noise_to_clean = (~q_clean) & kv_clean & (kv_frame < q_frame)
    noise_to_noise = (~q_clean) & (~kv_clean) & (kv_frame == q_frame)

    within_window = (q_frame - kv_frame).abs() <= int(window_size_frames)
    mask = within_window & (clean_to_clean | noise_to_clean | noise_to_noise)

    if not video_can_attend_action:
        mask[:video_seq_len, video_seq_len:] = False
    return mask
