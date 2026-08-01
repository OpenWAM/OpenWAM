"""Timeline normalization and tensor resampling for mixed-video decode."""

from __future__ import annotations

import math

import torch

from open_wam.contracts import (
    normalized_video_frame_count as _timeline_normalized_video_frame_count,
    resolve_video_source_fps,
)


def resolve_mixed_video_observation_fps(
    observation_fps: float | None,
    *,
    missing_observation_fps: float = 30.0,
) -> float:
    """Resolve a source FPS, using the mixed-video default when metadata is missing."""

    return resolve_video_source_fps(
        observation_fps,
        missing_observation_fps=missing_observation_fps,
    ).value


def normalized_video_frame_count(
    length_frames: int,
    *,
    source_fps: float | None,
    target_fps: float | None,
    missing_source_fps: float = 30.0,
) -> int:
    """Return the number of frames after resampling a clip onto `target_fps`."""

    length = int(length_frames)
    if length <= 0:
        return 0
    if target_fps is None:
        return length
    source = resolve_mixed_video_observation_fps(
        source_fps,
        missing_observation_fps=missing_source_fps,
    )
    target = float(target_fps)
    if target <= 0:
        raise ValueError("`target_fps` must be positive or None.")
    return _timeline_normalized_video_frame_count(
        length,
        source_fps=source,
        target_fps=target,
    )


def resample_video_frames_to_fps(
    frames: torch.Tensor,
    *,
    source_fps: float | None,
    target_fps: float | None,
    missing_source_fps: float = 30.0,
    target_start_index: int = 0,
    target_frame_count: int | None = None,
    native_start_index: int = 0,
    native_total_frames: int | None = None,
) -> torch.Tensor:
    """Linearly interpolate video frames from source FPS to a target FPS grid."""

    source = resolve_mixed_video_observation_fps(
        source_fps,
        missing_observation_fps=missing_source_fps,
    )
    resolved_native_total = (
        int(frames.shape[0])
        if native_total_frames is None
        else int(native_total_frames)
    )
    resolved_target_count = (
        normalized_video_frame_count(
            resolved_native_total,
            source_fps=source,
            target_fps=target_fps,
            missing_source_fps=missing_source_fps,
        )
        if target_frame_count is None
        else int(target_frame_count)
    )
    return _resample_video_frames_at_target_indices(
        frames,
        source_fps=source,
        target_fps=target_fps,
        target_start_index=int(target_start_index),
        target_frame_count=resolved_target_count,
        native_start_index=int(native_start_index),
        native_total_frames=resolved_native_total,
    )


def _native_span_for_target_chunk(
    *,
    chunk_start: int,
    chunk_end: int,
    native_length_frames: int,
    source_fps: float,
    target_fps: float | None,
) -> tuple[int, int]:
    if target_fps is None:
        return int(chunk_start), int(chunk_end)
    if chunk_end <= chunk_start:
        raise ValueError(f"Invalid target frame chunk [{chunk_start}, {chunk_end}).")
    first_position = float(chunk_start) * float(source_fps) / float(target_fps)
    last_position = float(chunk_end - 1) * float(source_fps) / float(target_fps)
    native_start = max(
        0,
        min(
            int(native_length_frames) - 1,
            int(math.floor(first_position)),
        ),
    )
    native_end = max(
        native_start + 1,
        min(
            int(native_length_frames),
            int(math.ceil(last_position)) + 1,
        ),
    )
    return native_start, native_end


def _resample_video_frames_at_target_indices(
    frames: torch.Tensor,
    *,
    source_fps: float,
    target_fps: float | None,
    target_start_index: int,
    target_frame_count: int,
    native_start_index: int,
    native_total_frames: int,
) -> torch.Tensor:
    if frames.ndim < 1:
        raise ValueError(
            "Expected video frames with leading time dimension, "
            f"got shape {tuple(frames.shape)}."
        )
    if target_frame_count <= 0:
        return frames[:0]
    if target_fps is None:
        start = int(target_start_index) - int(native_start_index)
        end = start + int(target_frame_count)
        return frames[start:end]
    if float(source_fps) <= 0 or float(target_fps) <= 0:
        raise ValueError(
            "FPS values must be positive, "
            f"got source={source_fps}, target={target_fps}."
        )
    if frames.shape[0] == 0:
        raise ValueError("Cannot resample an empty video frame tensor.")
    device = frames.device
    positions = (
        torch.arange(
            int(target_frame_count),
            dtype=torch.float32,
            device=device,
        )
        + float(target_start_index)
    ) * (float(source_fps) / float(target_fps))
    positions = positions.clamp(
        min=0.0,
        max=max(0.0, float(native_total_frames - 1)),
    )
    local_positions = positions - float(native_start_index)
    low = (
        torch.floor(local_positions)
        .to(dtype=torch.long)
        .clamp(min=0, max=frames.shape[0] - 1)
    )
    high = (low + 1).clamp(max=frames.shape[0] - 1)
    alpha = (local_positions - low.to(dtype=torch.float32)).clamp(min=0.0, max=1.0)
    while alpha.ndim < frames.ndim:
        alpha = alpha.unsqueeze(-1)
    source_dtype = frames.dtype
    interpolated = (
        frames[low].to(dtype=torch.float32) * (1.0 - alpha)
        + frames[high].to(dtype=torch.float32) * alpha
    )
    if source_dtype == torch.uint8:
        return interpolated.round().clamp(0, 255).to(dtype=source_dtype)
    return interpolated.to(dtype=source_dtype)


__all__ = [
    "normalized_video_frame_count",
    "resample_video_frames_to_fps",
    "resolve_mixed_video_observation_fps",
]
