"""Shared geometry for bounded recurrent temporal windows."""

from __future__ import annotations

from dataclasses import dataclass


def resolve_interleaved_history_frames(
    *,
    window_size: int,
    frame_chunk_size: int,
    block_id_stride: int = 2,
) -> int:
    """Return same-stream history represented by an interleaved block window.

    Dual-stream layouts reserve two block ids per temporal chunk. VTA uses them
    for video/action ordering; conditioned-video execution preserves the same
    temporal stride for parity. A W30 profile therefore exposes 15 prior
    same-stream chunks. Retaining at least one chunk preserves the established
    short-window rollout behavior.
    """

    chunk = max(1, int(frame_chunk_size))
    window = max(1, int(window_size))
    stride = max(1, int(block_id_stride))
    return max(chunk, (window // stride) * chunk)


def resolve_interleaved_cache_frames(
    *,
    window_size: int,
    frame_chunk_size: int,
    block_id_stride: int = 2,
) -> int:
    """Return bounded history plus one current generated chunk."""

    chunk = max(1, int(frame_chunk_size))
    return resolve_interleaved_history_frames(
        window_size=window_size,
        frame_chunk_size=chunk,
        block_id_stride=block_id_stride,
    ) + chunk


@dataclass(frozen=True, slots=True)
class OneFrameConditionedHistoryWindow:
    """A bounded contiguous history with its first frame used as condition."""

    dropped_frames: int
    retained_frames: int
    chunk_origin_frame: int


def resolve_one_frame_conditioned_history_window(
    *,
    history_frames: int,
    window_size: int,
    frame_chunk_size: int,
    chunk_origin_frame: int,
) -> OneFrameConditionedHistoryWindow:
    """Bound a causal-video history while preserving target chunk identities.

    The retained tensor is contiguous. Its oldest frame becomes the external
    clean condition and all following frames remain target history.  Adjusting
    ``chunk_origin_frame`` preserves the original target chunk partition even
    when trimming lands inside a chunk. An executed partial current chunk is
    retained in addition to the complete visible history because inference
    removes and reconstructs those partial frames before denoising.
    """

    history = int(history_frames)
    chunk = int(frame_chunk_size)
    if history <= 0:
        raise ValueError(f"Expected non-empty temporal history, got {history}.")
    if chunk <= 0:
        raise ValueError(f"Expected frame_chunk_size > 0, got {chunk}.")
    complete_history_frames = resolve_interleaved_history_frames(
        window_size=window_size,
        frame_chunk_size=chunk,
    )
    target_frames = history - 1
    partial_chunk_frames = (target_frames - int(chunk_origin_frame)) % chunk
    max_history_frames = 1 + complete_history_frames + partial_chunk_frames
    retained = min(history, max_history_frames)
    dropped = history - retained
    return OneFrameConditionedHistoryWindow(
        dropped_frames=dropped,
        retained_frames=retained,
        chunk_origin_frame=(int(chunk_origin_frame) - dropped) % chunk,
    )


__all__ = [
    "OneFrameConditionedHistoryWindow",
    "resolve_interleaved_cache_frames",
    "resolve_interleaved_history_frames",
    "resolve_one_frame_conditioned_history_window",
]
