"""Codec backends for mixed-video full-stream and chunked decode."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from open_wam.configs import MixedVideoDataConfig, MixedVideoFrameFitMode
from open_wam.data.mixed_video_catalog import MixedVideoStreamRecord
from open_wam.data.mixed_video_decode_frames import (
    MixedVideoResolvedDecodeSize,
    _batch_resize_frames,
    resolve_mixed_video_decode_size,
    transform_frame,
)
from open_wam.data.mixed_video_decode_timeline import resample_video_frames_to_fps


# WHY: decord's C++ batch decode is 3-5x faster than imageio's Python
# frame-by-frame iteration. We keep imageio as fallback for codec edge cases.
try:
    import decord

    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


_TIMESTAMP_BOUNDARY_EPSILON_SECONDS = 1e-4


def decode_video_frames(
    path: Path,
    *,
    target_height: int,
    target_width: int,
    center_crop: bool,
    allow_upscale: bool,
    fit_mode: MixedVideoFrameFitMode | str | None = None,
    source_fps: float | None = None,
    target_fps: float | None = None,
    missing_source_fps: float = 30.0,
    from_timestamp: float | None = None,
    to_timestamp: float | None = None,
    data_config: MixedVideoDataConfig | None = None,
) -> torch.Tensor:
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps", 0.0) or 0.0)
        frames = []
        resolved_height = int(target_height)
        resolved_width = int(target_width)
        for frame_index, frame in enumerate(reader):
            if fps > 0.0:
                timestamp = frame_index / fps
                # WHY epsilon: packed-bundle manifests can store boundary
                # timestamps slightly above the true frame time.
                if (
                    from_timestamp is not None
                    and timestamp < from_timestamp - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
                ):
                    continue
                if to_timestamp is not None and timestamp >= to_timestamp:
                    break
            if data_config is not None and not frames:
                frame_array = np.asarray(frame)
                resolved = resolve_mixed_video_decode_size(
                    data_config,
                    source_height=int(frame_array.shape[0]),
                    source_width=int(frame_array.shape[1]),
                )
                resolved_height = resolved.height
                resolved_width = resolved.width
            transformed = transform_frame(
                frame,
                target_height=resolved_height,
                target_width=resolved_width,
                center_crop=center_crop,
                allow_upscale=allow_upscale,
                fit_mode=fit_mode,
            )
            frames.append(
                torch.as_tensor(
                    np.array(transformed, copy=True),
                    dtype=torch.uint8,
                )
            )
        if not frames:
            raise ValueError(f"Video file has no decodable frames: {path}")
        decoded = torch.stack(frames, dim=0)
        effective_source_fps = (
            source_fps if source_fps is not None and float(source_fps) > 0.0 else fps
        )
        if effective_source_fps <= 0.0:
            effective_source_fps = None
        return resample_video_frames_to_fps(
            decoded,
            source_fps=effective_source_fps,
            target_fps=target_fps,
            missing_source_fps=missing_source_fps,
        )
    finally:
        reader.close()


def _iter_chunks_decord(
    path: Path,
    *,
    data_config: MixedVideoDataConfig,
    stream: MixedVideoStreamRecord,
    chunk_specs: list[tuple[int, int, int, int]],
    resolved_size: MixedVideoResolvedDecodeSize,
    source_fps: float,
    target_fps: float | None,
) -> Iterator[torch.Tensor]:
    """Decode video chunks using one decord batch call per chunk."""

    # CPU decoding avoids competing with the VAE for accelerator memory.
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    container_fps = float(vr.get_avg_fps())
    total_native_frames = len(vr)

    resolved_height = int(resolved_size.height)
    resolved_width = int(resolved_size.width)

    frame_offset = 0
    if stream.from_timestamp is not None and container_fps > 0:
        frame_offset = 0
        for index in range(total_native_frames):
            timestamp = float(index) / container_fps
            if timestamp >= stream.from_timestamp - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS:
                frame_offset = index
                break

    end_frame_limit = total_native_frames
    if stream.to_timestamp is not None and container_fps > 0:
        for index in range(frame_offset, total_native_frames):
            timestamp = float(index) / container_fps
            if timestamp >= stream.to_timestamp:
                end_frame_limit = index
                break

    for chunk_start, chunk_end, native_start, native_end in chunk_specs:
        absolute_start = frame_offset + native_start
        absolute_end = min(frame_offset + native_end, end_frame_limit)
        if absolute_end <= absolute_start:
            raise ValueError(
                f"Decoded stream shorter than manifest for "
                f"source={stream.source_id}, episode={stream.episode_index}, "
                f"chunk=[{chunk_start},{chunk_end})."
            )
        indices = list(range(absolute_start, absolute_end))
        raw_frames = vr.get_batch(indices).asnumpy()

        if stream.height is None or stream.width is None:
            resolved = resolve_mixed_video_decode_size(
                data_config,
                source_height=int(raw_frames.shape[1]),
                source_width=int(raw_frames.shape[2]),
            )
            resolved_height = int(resolved.height)
            resolved_width = int(resolved.width)

        resized = _batch_resize_frames(
            raw_frames,
            target_height=resolved_height,
            target_width=resolved_width,
            allow_upscale=data_config.decode_allow_upscale,
            fit_mode=data_config.decode_fit_mode,
            center_crop=data_config.decode_center_crop,
        )
        native_frames = torch.as_tensor(resized, dtype=torch.uint8)
        yield resample_video_frames_to_fps(
            native_frames,
            source_fps=source_fps,
            target_fps=target_fps,
            missing_source_fps=data_config.missing_observation_fps,
            target_start_index=chunk_start,
            target_frame_count=chunk_end - chunk_start,
            native_start_index=native_start,
            native_total_frames=int(stream.length_frames),
        )


def _iter_chunks_imageio(
    path: Path,
    *,
    data_config: MixedVideoDataConfig,
    stream: MixedVideoStreamRecord,
    chunk_specs: list[tuple[int, int, int, int]],
    resolved_size: MixedVideoResolvedDecodeSize,
    source_fps: float,
    target_fps: float | None,
) -> Iterator[torch.Tensor]:
    """Decode chunks with imageio for containers unsupported by decord."""

    frames_by_native_index: dict[int, torch.Tensor] = {}
    selected_index = 0
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps", 0.0) or 0.0)
        resolved_height = int(resolved_size.height)
        resolved_width = int(resolved_size.width)
        reader_iter = iter(enumerate(reader))
        reader_exhausted = False
        for chunk_index, (
            chunk_start,
            chunk_end,
            native_start,
            native_end,
        ) in enumerate(chunk_specs):
            while selected_index < native_end and not reader_exhausted:
                try:
                    frame_index, frame = next(reader_iter)
                except StopIteration:
                    reader_exhausted = True
                    break
                if fps > 0.0:
                    timestamp = frame_index / fps
                    # LeRobot v3 packed-bundle float64 timestamps can round
                    # above the true WAN-aligned boundary.
                    if (
                        stream.from_timestamp is not None
                        and timestamp
                        < stream.from_timestamp - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
                    ):
                        continue
                    if (
                        stream.to_timestamp is not None
                        and timestamp >= stream.to_timestamp
                    ):
                        reader_exhausted = True
                        break
                if selected_index >= native_start:
                    frame_array = np.asarray(frame)
                    if stream.height is None or stream.width is None:
                        resolved = resolve_mixed_video_decode_size(
                            data_config,
                            source_height=int(frame_array.shape[0]),
                            source_width=int(frame_array.shape[1]),
                        )
                        resolved_height = int(resolved.height)
                        resolved_width = int(resolved.width)
                    transformed = transform_frame(
                        frame_array,
                        target_height=resolved_height,
                        target_width=resolved_width,
                        center_crop=data_config.decode_center_crop,
                        allow_upscale=data_config.decode_allow_upscale,
                        fit_mode=data_config.decode_fit_mode,
                    )
                    frames_by_native_index[selected_index] = torch.as_tensor(
                        np.array(transformed, copy=True),
                        dtype=torch.uint8,
                    )
                selected_index += 1
            missing = [
                index
                for index in range(native_start, native_end)
                if index not in frames_by_native_index
            ]
            if missing:
                raise ValueError(
                    "Decoded stream is shorter than manifest metadata for "
                    f"source={stream.source_id}, "
                    f"episode={stream.episode_index}, "
                    f"stream={stream.stream_key}; missing native frames "
                    f"{missing[:5]} for normalized "
                    f"chunk=[{chunk_start}, {chunk_end})."
                )
            native_frames = torch.stack(
                [
                    frames_by_native_index[index]
                    for index in range(native_start, native_end)
                ],
                dim=0,
            )
            yield resample_video_frames_to_fps(
                native_frames,
                source_fps=source_fps,
                target_fps=target_fps,
                missing_source_fps=data_config.missing_observation_fps,
                target_start_index=chunk_start,
                target_frame_count=chunk_end - chunk_start,
                native_start_index=native_start,
                native_total_frames=int(stream.length_frames),
            )
            if chunk_index + 1 < len(chunk_specs):
                next_native_start = chunk_specs[chunk_index + 1][2]
                for cached_index in tuple(frames_by_native_index):
                    if cached_index < next_native_start:
                        del frames_by_native_index[cached_index]
    finally:
        reader.close()


__all__ = ["decode_video_frames"]
