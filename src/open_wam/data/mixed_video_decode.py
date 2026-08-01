"""Public stream orchestration facade for mixed-video decode."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from open_wam.configs import MixedVideoDataConfig
from open_wam.data import mixed_video_decode_backends as _backends
from open_wam.data import mixed_video_decode_frames as _frames
from open_wam.data import mixed_video_decode_timeline as _timeline
from open_wam.data.mixed_video_catalog import MixedVideoStreamRecord
from open_wam.data.mixed_video_decode_backends import (
    _iter_chunks_decord,
    _iter_chunks_imageio,
    decode_video_frames,
)
from open_wam.data.mixed_video_decode_frames import (
    resolve_mixed_video_decode_size,
)
from open_wam.data.mixed_video_decode_timeline import (
    _native_span_for_target_chunk,
)

# Compatibility globals used by historical tests and checkout tooling.
imageio = _backends.imageio
_HAS_DECORD = _backends._HAS_DECORD
_TIMESTAMP_BOUNDARY_EPSILON_SECONDS = _backends._TIMESTAMP_BOUNDARY_EPSILON_SECONDS
if _HAS_DECORD:
    decord = _backends.decord
MixedVideoResolvedDecodeSize = _frames.MixedVideoResolvedDecodeSize
_batch_resize_frames = _frames._batch_resize_frames
_center_crop_to_aspect = _frames._center_crop_to_aspect
_letterbox_pad_to_target = _frames._letterbox_pad_to_target
_resize_frame = _frames._resize_frame
_resolve_frame_fit_mode = _frames._resolve_frame_fit_mode
_select_mixed_video_resize_bin = _frames._select_mixed_video_resize_bin
transform_frame = _frames.transform_frame
_resample_video_frames_at_target_indices = (
    _timeline._resample_video_frames_at_target_indices
)
normalized_video_frame_count = _timeline.normalized_video_frame_count
resample_video_frames_to_fps = _timeline.resample_video_frames_to_fps
resolve_mixed_video_observation_fps = _timeline.resolve_mixed_video_observation_fps


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
            data_config if stream.height is None or stream.width is None else None
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
            raise ValueError(f"Invalid frame chunk [{start_frame}, {end_frame}).")
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
