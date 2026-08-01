"""Streaming RGB decode and VAE execution for mixed-video encoding."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import queue
import threading
from typing import Any

import torch

from open_wam.configs import MixedVideoDataConfig, MixedVideoSourceFormat
from open_wam.data.mixed_video_catalog import (
    MixedVideoEpisodeRecord,
    MixedVideoStreamRecord,
)
from open_wam.data.mixed_video_decode import iter_mixed_video_stream_frame_chunks
from open_wam.data.mixed_video_encoding_contracts import MixedVideoLatentEncoder
from open_wam.data.raw_video import ConfiguredCanonicalVideoPreprocessor
from open_wam.models.common.video_geometry import wan_raw_frame_count_to_latent_count


def _encode_episode_latents_streaming(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    canonicalizer: ConfiguredCanonicalVideoPreprocessor,
    assets: MixedVideoLatentEncoder,
    device: torch.device,
    chunk_frames: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode one episode with CPU/GPU overlap double buffering.

    WHY double buffer: CPU video decode and GPU VAE encode are independent
    workloads. By decoding chunk N+1 on CPU while chunk N encodes on GPU,
    we overlap ~60% of decode time with encode time, yielding ~1.5-2x speedup.
    """
    latent_chunks: list[torch.Tensor] = []
    placements_metadata: list[dict[str, Any]] | None = None
    canonical_shape: list[int] | None = None
    raw_chunk_ranges = _streaming_chunk_ranges(
        int(episode.length_frames), max_chunk_frames=chunk_frames
    )
    chunk_iterator = enumerate(
        _iter_episode_view_chunks(
            data_config, episode, raw_chunk_ranges=raw_chunk_ranges
        )
    )

    # WHY queue maxsize=2: one slot for the chunk being GPU-encoded, one for
    # the prefetched next chunk. Larger queues waste CPU memory on decoded
    # frames without additional GPU overlap benefit.
    prefetch_queue: queue.Queue = queue.Queue(maxsize=2)
    prefetch_error: list[Exception] = []

    def _prefetch_worker():
        """Background thread: decode + canonicalize chunks on CPU, push to queue."""
        try:
            for chunk_index, views in chunk_iterator:
                canonical = canonicalizer(views)
                # WHY keep on CPU here: CUDA init from a background thread fails
                # on nodes with older drivers. H2D transfer happens in main thread.
                prefetch_queue.put((chunk_index, canonical.video, canonical.placements))
        except Exception as exc:
            prefetch_error.append(exc)
        finally:
            prefetch_queue.put(None)  # sentinel

    # WHY daemon=True: if main thread crashes, prefetch thread dies immediately
    worker = threading.Thread(target=_prefetch_worker, daemon=True)
    worker.start()

    while True:
        item = prefetch_queue.get()
        if item is None:
            break
        chunk_index, video_cpu, placements = item
        # WHY .to(device) in main thread: avoids CUDA init in background thread
        # which fails on nodes with old drivers (CUDA 12030)
        video_on_device = video_cpu.to(device=device)
        if placements_metadata is None:
            placements_metadata = [asdict(placement) for placement in placements]
            canonical_shape = list(video_on_device.shape[1:])
        latent_chunks.append(
            assets.encode_video(
                video_on_device,
                placements=placements,
                reset_cache=chunk_index == 0,
            ).detach()
        )

    worker.join()
    if prefetch_error:
        raise prefetch_error[0]

    if not latent_chunks:
        raise ValueError(
            f"No latent chunks were produced for mixed-video episode {episode.key!r}."
        )
    metadata = {
        "canonical_shape": canonical_shape,
        "placements": placements_metadata or [],
        "raw_chunk_ranges": [list(item) for item in raw_chunk_ranges],
        "native_length_frames": int(episode.native_length_frames),
        "normalized_length_frames": int(episode.length_frames),
        "target_observation_fps": data_config.target_observation_fps,
        "chunk_frames": int(chunk_frames),
    }
    return torch.cat(latent_chunks, dim=2), metadata


def _iter_episode_view_chunks(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    raw_chunk_ranges: tuple[tuple[int, int], ...],
) -> Iterator[dict[str, torch.Tensor]]:
    streams_by_slot: dict[str, MixedVideoStreamRecord] = {}
    for stream in sorted(episode.streams, key=lambda item: item.stream_index):
        streams_by_slot.setdefault(stream.target_slot, stream)

    stream_iterators: dict[str, Iterator[torch.Tensor]] = {}
    for camera_name in data_config.camera_names:
        stream = streams_by_slot.get(camera_name)
        if stream is None:
            raise KeyError(
                f"Cannot encode mixed-video episode {episode.key!r}: missing RGB stream for slot {camera_name!r}."
            )
        if stream.source_format not in {
            MixedVideoSourceFormat.RGB,
            MixedVideoSourceFormat.RGB_AND_LATENT,
        }:
            raise ValueError(
                f"Cannot encode source={stream.source_id!r}, episode={stream.episode_index}, "
                f"stream={stream.stream_key}: source_format={stream.source_format.value!r} has no RGB input."
            )
        stream_iterators[camera_name] = iter_mixed_video_stream_frame_chunks(
            data_config,
            stream,
            raw_chunk_ranges=raw_chunk_ranges,
        )

    for _ in raw_chunk_ranges:
        yield {
            camera_name: next(stream_iterator)
            for camera_name, stream_iterator in stream_iterators.items()
        }


def _decode_episode_view_chunk(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, torch.Tensor]:
    iterator = _iter_episode_view_chunks(
        data_config,
        episode,
        raw_chunk_ranges=((int(start_frame), int(end_frame)),),
    )
    return next(iterator)


def _streaming_chunk_ranges(
    num_frames: int, *, max_chunk_frames: int
) -> tuple[tuple[int, int], ...]:
    if num_frames <= 0:
        raise ValueError(f"Cannot encode an empty video: num_frames={num_frames}.")
    latent_frames = wan_raw_frame_count_to_latent_count(num_frames)
    encoded_raw_frames = 1 + 4 * (latent_frames - 1)
    if max_chunk_frames == 0 or encoded_raw_frames <= max_chunk_frames:
        return ((0, encoded_raw_frames),)
    if max_chunk_frames < 5:
        raise ValueError("max_chunk_frames must be 0 or at least 5.")
    first_capacity = 1 + 4 * ((max_chunk_frames - 1) // 4)
    stream_capacity = 4 * (max_chunk_frames // 4)
    ranges: list[tuple[int, int]] = []
    first_end = min(encoded_raw_frames, first_capacity)
    ranges.append((0, first_end))
    start = first_end
    while start < encoded_raw_frames:
        end = min(encoded_raw_frames, start + stream_capacity)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


plan_mixed_video_streaming_chunks = _streaming_chunk_ranges


__all__ = ["plan_mixed_video_streaming_chunks"]
