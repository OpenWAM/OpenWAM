"""Attention layouts for cached DualExpert action inference."""

from __future__ import annotations

import torch

from open_wam.configs import CurrentBlockCoupling


def build_dual_expert_inference_action_attention_mask(
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
    current_block_coupling: CurrentBlockCoupling
    | str = CurrentBlockCoupling.VIDEO_THEN_ACTION,
) -> torch.Tensor:
    """Inference-only DualExpert action attention mask (parallel-stream byte-aligned).

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
        parallel-stream's `input_dict["window_size"]`).

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
            "DualExpert inference action mask requires non-negative cache lengths and a positive "
            f"current chunk, got video_seq_len={video_seq_len}, "
            f"past_action_seq_len={past_action_seq_len}, "
            f"current_action_seq_len={current_action_seq_len}."
        )
    if video_tokens_per_frame <= 0 or action_tokens_per_frame <= 0:
        raise ValueError(
            "DualExpert inference action mask requires positive tokens-per-frame, "
            f"got video_tokens_per_frame={video_tokens_per_frame}, "
            f"action_tokens_per_frame={action_tokens_per_frame}."
        )
    if chunk_size_frames <= 0 or window_size_frames <= 0:
        raise ValueError(
            "DualExpert inference action mask requires positive chunk/window sizes, "
            f"got chunk_size_frames={chunk_size_frames}, "
            f"window_size_frames={window_size_frames}."
        )
    # Slot-pool capacity (`(attn_window // 2) * (latent_token_per_chunk +
    # action_token_per_chunk)`) is sized in tokens, not frames. parallel-stream
    # writes both streams so eviction stays chunk-aligned; dual-expert only
    # writes video, so eviction can leave a partial leading frame in the
    # video cache. Don't reject that — floor-division below assigns the
    # partial frame to the oldest block_id, which is correct for the
    # relative within_window check the mask actually uses.
    if current_action_seq_len % action_tokens_per_frame != 0:
        raise ValueError(
            "DualExpert inference action mask expects current_action_seq_len divisible by action_tokens_per_frame, "
            f"got current_action_seq_len={current_action_seq_len}, action_tokens_per_frame={action_tokens_per_frame}."
        )

    chunk_origin_frame = int(chunk_origin_frame)
    video_token_ids = torch.arange(video_seq_len, device=device)
    video_frame_ids = torch.div(
        video_token_ids, int(video_tokens_per_frame), rounding_mode="floor"
    ) + int(video_frame_start)
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
        current_action_frame_start = int(past_action_frame_start) + int(
            past_action_frames_count
        )
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
        clean_to_clean = (
            q_clean
            & kv_clean
            & ((kv_chunk < q_chunk) | ((kv_chunk == q_chunk) & (kv_stream == q_stream)))
        )
        noise_to_clean = (
            (~q_clean)
            & kv_clean
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
        clean_to_clean = q_clean & kv_clean & (kv_frame <= q_frame)
        noise_to_clean = (~q_clean) & kv_clean & (kv_frame < q_frame)
    noise_to_noise = (~q_clean) & (~kv_clean) & (kv_frame == q_frame)

    within_window = (q_frame - kv_frame).abs() <= int(window_size_frames)
    mask = within_window & (clean_to_clean | noise_to_clean | noise_to_noise)

    if not video_can_attend_action:
        mask[:video_seq_len, video_seq_len:] = False
    return mask


__all__ = ["build_dual_expert_inference_action_attention_mask"]
