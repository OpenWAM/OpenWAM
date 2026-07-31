from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from open_wam.configs import (
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoFrameFitMode,
    MixedVideoResizeBinConfig,
)
from open_wam.contracts import (
    normalized_video_frame_count as _timeline_normalized_video_frame_count,
    resolve_video_source_fps,
)

from .mixed_video_catalog import MixedVideoStreamRecord


# WHY: decord's C++ batch decode is 3-5x faster than imageio's Python
# frame-by-frame iteration. We keep imageio as fallback for codec edge cases.
try:
    import decord

    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


_TIMESTAMP_BOUNDARY_EPSILON_SECONDS = 1e-4


@dataclass(frozen=True)
class MixedVideoResolvedDecodeSize:
    """Resolved resize target for one mixed-video stream."""

    height: int
    width: int
    bin_name: str
    source_height: int | None
    source_width: int | None


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


def resolve_mixed_video_decode_size(
    data_config: MixedVideoDataConfig,
    *,
    source_height: int | None,
    source_width: int | None,
) -> MixedVideoResolvedDecodeSize:
    """Resolve the VAE input size for one mixed-video stream."""

    if data_config.decode_size_mode == MixedVideoDecodeSizeMode.FIXED:
        return MixedVideoResolvedDecodeSize(
            height=int(data_config.decode_height),
            width=int(data_config.decode_width),
            bin_name="fixed",
            source_height=source_height,
            source_width=source_width,
        )
    if source_height is None or source_width is None:
        return MixedVideoResolvedDecodeSize(
            height=int(data_config.decode_height),
            width=int(data_config.decode_width),
            bin_name="fixed_missing_source_size",
            source_height=source_height,
            source_width=source_width,
        )
    bin_config = _select_mixed_video_resize_bin(
        data_config.decode_resize_bins,
        source_height=int(source_height),
        source_width=int(source_width),
    )
    return MixedVideoResolvedDecodeSize(
        height=int(bin_config.target_height),
        width=int(bin_config.target_width),
        bin_name=str(bin_config.name),
        source_height=int(source_height),
        source_width=int(source_width),
    )


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
                    and timestamp
                    < from_timestamp - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
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
            source_fps
            if source_fps is not None and float(source_fps) > 0.0
            else fps
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


def decode_mixed_video_stream_frames(
    data_config: MixedVideoDataConfig,
    stream: MixedVideoStreamRecord,
) -> torch.Tensor:
    """Decode one complete stream onto its configured target timeline."""

    path = _resolve_stream_path(stream, cache_dir=data_config.cache_dir)
    resolved_size = resolve_mixed_video_decode_size(
        data_config,
        source_height=stream.height,
        source_width=stream.width,
    )
    return decode_video_frames(
        path,
        target_height=resolved_size.height,
        target_width=resolved_size.width,
        center_crop=data_config.decode_center_crop,
        allow_upscale=data_config.decode_allow_upscale,
        fit_mode=data_config.decode_fit_mode,
        source_fps=stream.observation_fps,
        target_fps=data_config.target_observation_fps,
        missing_source_fps=data_config.missing_observation_fps,
        from_timestamp=stream.from_timestamp,
        to_timestamp=stream.to_timestamp,
        data_config=(
            data_config
            if stream.height is None or stream.width is None
            else None
        ),
    )


def decode_mixed_video_stream_frame_chunk(
    data_config: MixedVideoDataConfig,
    stream: MixedVideoStreamRecord,
    *,
    start_frame: int,
    end_frame: int,
) -> torch.Tensor:
    return next(
        iter_mixed_video_stream_frame_chunks(
            data_config,
            stream,
            raw_chunk_ranges=((int(start_frame), int(end_frame)),),
        )
    )


def iter_mixed_video_stream_frame_chunks(
    data_config: MixedVideoDataConfig,
    stream: MixedVideoStreamRecord,
    *,
    raw_chunk_ranges: tuple[tuple[int, int], ...],
) -> Iterator[torch.Tensor]:
    """Decode chunks through the shared normalized mixed-video timeline."""

    if not raw_chunk_ranges:
        return
    for start_frame, end_frame in raw_chunk_ranges:
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(
                f"Invalid frame chunk [{start_frame}, {end_frame})."
            )
    for previous, current in zip(
        raw_chunk_ranges,
        raw_chunk_ranges[1:],
        strict=False,
    ):
        if previous[1] != current[0]:
            raise ValueError(
                "Frame chunks must be contiguous for streaming decode: "
                f"{raw_chunk_ranges!r}."
            )
    path = _resolve_stream_path(stream, cache_dir=data_config.cache_dir)
    resolved_size = resolve_mixed_video_decode_size(
        data_config,
        source_height=stream.height,
        source_width=stream.width,
    )
    source_fps = float(stream.clip.source_fps)
    target_fps = data_config.target_observation_fps
    target_length = int(stream.clip.normalized_length_frames)
    chunk_specs = [
        (
            int(chunk_start),
            int(chunk_end),
            *_native_span_for_target_chunk(
                chunk_start=int(chunk_start),
                chunk_end=int(chunk_end),
                native_length_frames=int(stream.length_frames),
                source_fps=source_fps,
                target_fps=target_fps,
            ),
        )
        for chunk_start, chunk_end in raw_chunk_ranges
    ]
    for chunk_start, chunk_end, _, _ in chunk_specs:
        if chunk_end > target_length:
            raise ValueError(
                f"Frame chunk [{chunk_start}, {chunk_end}) exceeds normalized "
                f"stream length {target_length} for source={stream.source_id}, "
                f"episode={stream.episode_index}, stream={stream.stream_key}."
            )

    # WHY try decord first: batch C++ decode avoids N Python round-trips per
    # frame; imageio fallback handles rare codec incompatibilities.
    if _HAS_DECORD:
        emitted_decord_chunk = False
        try:
            for chunk in _iter_chunks_decord(
                path,
                data_config=data_config,
                stream=stream,
                chunk_specs=chunk_specs,
                resolved_size=resolved_size,
                source_fps=source_fps,
                target_fps=target_fps,
            ):
                emitted_decord_chunk = True
                yield chunk
            return
        except Exception:
            if emitted_decord_chunk:
                raise
            # Decord may fail on unusual containers such as WebM.
            pass

    yield from _iter_chunks_imageio(
        path,
        data_config=data_config,
        stream=stream,
        chunk_specs=chunk_specs,
        resolved_size=resolved_size,
        source_fps=source_fps,
        target_fps=target_fps,
    )


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
            if (
                timestamp
                >= stream.from_timestamp
                - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
            ):
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
                        < stream.from_timestamp
                        - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
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
        raise ValueError(
            f"Invalid target frame chunk [{chunk_start}, {chunk_end})."
        )
    first_position = (
        float(chunk_start) * float(source_fps) / float(target_fps)
    )
    last_position = (
        float(chunk_end - 1) * float(source_fps) / float(target_fps)
    )
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
    alpha = (
        local_positions - low.to(dtype=torch.float32)
    ).clamp(min=0.0, max=1.0)
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


def _batch_resize_frames(
    frames: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    allow_upscale: bool,
    fit_mode: MixedVideoFrameFitMode | str | None = None,
    center_crop: bool = False,
) -> np.ndarray:
    """Resize an NHWC RGB frame batch with one interpolation call."""

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected [N,H,W,3] uint8 frames, got {frames.shape}."
        )
    n, height, width, _ = frames.shape
    if n == 0:
        return frames
    resolved_fit_mode = _resolve_frame_fit_mode(
        fit_mode,
        center_crop=center_crop,
    )

    if resolved_fit_mode == MixedVideoFrameFitMode.CENTER_CROP:
        target_aspect = target_width / target_height
        current_aspect = width / height
        if abs(current_aspect - target_aspect) > 1e-6:
            if current_aspect > target_aspect:
                crop_width = max(1, int(round(height * target_aspect)))
                left = max(0, (width - crop_width) // 2)
                frames = frames[
                    :,
                    :,
                    left : left + crop_width,
                    :,
                ]
            else:
                crop_height = max(1, int(round(width / target_aspect)))
                top = max(0, (height - crop_height) // 2)
                frames = frames[
                    :,
                    top : top + crop_height,
                    :,
                    :,
                ]
        n, height, width, _ = frames.shape
        if not allow_upscale and (
            height < target_height or width < target_width
        ):
            return frames
        if height == target_height and width == target_width:
            return frames
        tensor = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)
            .float()
        )
        tensor = F.interpolate(
            tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (
            tensor.clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .numpy()
        )

    if resolved_fit_mode == MixedVideoFrameFitMode.LETTERBOX_PAD:
        if (
            height <= 0
            or width <= 0
            or target_height <= 0
            or target_width <= 0
        ):
            raise ValueError(
                "Letterbox requires positive dims, "
                f"got input=({height},{width}) "
                f"target=({target_height},{target_width})."
            )
        scale = min(
            float(target_width) / float(width),
            float(target_height) / float(height),
        )
        if not allow_upscale:
            scale = min(scale, 1.0)
        resized_height = max(
            1,
            min(
                target_height,
                int(round(float(height) * scale)),
            ),
        )
        resized_width = max(
            1,
            min(
                target_width,
                int(round(float(width) * scale)),
            ),
        )
        if resized_height == height and resized_width == width:
            resized = frames
        else:
            tensor = (
                torch.from_numpy(frames)
                .permute(0, 3, 1, 2)
                .float()
            )
            tensor = F.interpolate(
                tensor,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = (
                tensor.clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .numpy()
            )
        canvas = np.zeros(
            (n, target_height, target_width, 3),
            dtype=np.uint8,
        )
        top_pad = max(0, (target_height - resized_height) // 2)
        left_pad = max(0, (target_width - resized_width) // 2)
        canvas[
            :,
            top_pad : top_pad + resized_height,
            left_pad : left_pad + resized_width,
        ] = resized[..., :3]
        return canvas

    raise ValueError(f"Unsupported fit mode: {resolved_fit_mode}")


def transform_frame(
    frame: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    center_crop: bool,
    allow_upscale: bool,
    fit_mode: MixedVideoFrameFitMode | str | None = None,
) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB frame [H,W,3+], got {array.shape}.")
    array = np.ascontiguousarray(array[..., :3])
    resolved_fit_mode = _resolve_frame_fit_mode(
        fit_mode,
        center_crop=center_crop,
    )
    if resolved_fit_mode == MixedVideoFrameFitMode.CENTER_CROP:
        array = _center_crop_to_aspect(
            array,
            target_height=target_height,
            target_width=target_width,
        )
        return _resize_frame(
            array,
            target_height=target_height,
            target_width=target_width,
            allow_upscale=allow_upscale,
        )
    if resolved_fit_mode == MixedVideoFrameFitMode.LETTERBOX_PAD:
        return _letterbox_pad_to_target(
            array,
            target_height=target_height,
            target_width=target_width,
            allow_upscale=allow_upscale,
        )
    raise ValueError(
        f"Unsupported mixed-video frame fit mode: {resolved_fit_mode}"
    )


def _resolve_frame_fit_mode(
    fit_mode: MixedVideoFrameFitMode | str | None,
    *,
    center_crop: bool,
) -> MixedVideoFrameFitMode:
    if fit_mode is not None:
        return (
            fit_mode
            if isinstance(fit_mode, MixedVideoFrameFitMode)
            else MixedVideoFrameFitMode(str(fit_mode))
        )
    return (
        MixedVideoFrameFitMode.CENTER_CROP
        if center_crop
        else MixedVideoFrameFitMode.LETTERBOX_PAD
    )


def _resize_frame(
    array: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    allow_upscale: bool,
) -> np.ndarray:
    input_height, input_width = int(array.shape[0]), int(array.shape[1])
    if not allow_upscale and (
        input_height < target_height or input_width < target_width
    ):
        return array
    if input_height == target_height and input_width == target_width:
        return array
    image = Image.fromarray(array)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.resize((target_width, target_height), resampling)
    return np.asarray(resized, dtype=np.uint8)


def _letterbox_pad_to_target(
    array: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    allow_upscale: bool,
) -> np.ndarray:
    input_height, input_width = int(array.shape[0]), int(array.shape[1])
    if (
        input_height <= 0
        or input_width <= 0
        or target_height <= 0
        or target_width <= 0
    ):
        raise ValueError(
            "Mixed-video letterbox resize expects positive dimensions, "
            f"got input=({input_height}, {input_width}) "
            f"target=({target_height}, {target_width})."
        )
    scale = min(
        float(target_width) / float(input_width),
        float(target_height) / float(input_height),
    )
    if not allow_upscale:
        scale = min(scale, 1.0)
    resized_height = max(
        1,
        min(
            int(target_height),
            int(round(float(input_height) * scale)),
        ),
    )
    resized_width = max(
        1,
        min(
            int(target_width),
            int(round(float(input_width) * scale)),
        ),
    )
    if resized_height == input_height and resized_width == input_width:
        resized = array
    else:
        image = Image.fromarray(array)
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        resized = np.asarray(
            image.resize((resized_width, resized_height), resampling),
            dtype=np.uint8,
        )
    canvas = np.zeros(
        (int(target_height), int(target_width), 3),
        dtype=np.uint8,
    )
    top = max(0, (int(target_height) - int(resized_height)) // 2)
    left = max(0, (int(target_width) - int(resized_width)) // 2)
    canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized[..., :3]
    return canvas


def _resolve_stream_path(
    stream: MixedVideoStreamRecord,
    *,
    cache_dir: str | None,
) -> Path:
    if stream.local_path is not None:
        if not stream.local_path.exists():
            raise FileNotFoundError(
                f"Missing mixed-video file for source={stream.source_id}, "
                f"episode={stream.episode_index}, "
                f"stream={stream.stream_key}: {stream.local_path}"
            )
        return stream.local_path
    if stream.repo_id is None or stream.shard_relative_path is None:
        raise FileNotFoundError(
            "Mixed-video stream has neither local_path nor HF repo/shard "
            f"path: source={stream.source_id}, "
            f"episode={stream.episode_index}, stream={stream.stream_key}."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for remote mixed-video manifests."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=stream.repo_id,
            filename=stream.shard_relative_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )


def _center_crop_to_aspect(
    array: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    height, width = int(array.shape[0]), int(array.shape[1])
    target_aspect = target_width / target_height
    current_aspect = width / height
    if abs(current_aspect - target_aspect) < 1e-6:
        return array
    if current_aspect > target_aspect:
        crop_width = max(1, int(round(height * target_aspect)))
        left = max(0, (width - crop_width) // 2)
        return array[:, left : left + crop_width]
    crop_height = max(1, int(round(width / target_aspect)))
    top = max(0, (height - crop_height) // 2)
    return array[top : top + crop_height, :]


def _select_mixed_video_resize_bin(
    bins: Sequence[MixedVideoResizeBinConfig],
    *,
    source_height: int,
    source_width: int,
) -> MixedVideoResizeBinConfig:
    if source_height <= 0 or source_width <= 0:
        raise ValueError(
            "Mixed-video source dimensions must be positive, "
            f"got height={source_height}, width={source_width}."
        )
    if not bins:
        raise ValueError("At least one mixed-video resize bin is required.")
    source_ratio = float(source_width) / float(source_height)
    ranked = sorted(
        bins,
        key=lambda bin_config: (
            abs(math.log(source_ratio / bin_config.aspect_ratio)),
            (
                float("inf")
                if bin_config.max_pixels is None
                else float(bin_config.max_pixels)
            ),
        ),
    )
    best_distance = abs(math.log(source_ratio / ranked[0].aspect_ratio))
    aspect_candidates = [
        bin_config
        for bin_config in ranked
        if abs(math.log(source_ratio / bin_config.aspect_ratio))
        <= best_distance + 1e-6
    ]
    source_pixels = int(source_height) * int(source_width)
    for bin_config in aspect_candidates:
        if (
            bin_config.max_pixels is None
            or source_pixels <= int(bin_config.max_pixels)
        ):
            return bin_config
    return aspect_candidates[-1]
