"""Deterministic grouping and validation of mixed-video episodes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from open_wam.configs import MixedVideoDataConfig

from .mixed_video_catalog_contracts import (
    MixedVideoCatalog,
    MixedVideoEpisodeRecord,
    MixedVideoStreamRecord,
)
from .mixed_video_manifest import _load_source_streams


def load_mixed_video_catalog(
    data_config: MixedVideoDataConfig,
) -> MixedVideoCatalog:
    streams: list[MixedVideoStreamRecord] = []
    for source in data_config.video_sources:
        if not source.enabled:
            continue
        streams.extend(_load_source_streams(source, data_config))
    grouped: dict[
        tuple[str, str, int, str],
        list[MixedVideoStreamRecord],
    ] = defaultdict(list)
    for stream in streams:
        grouped[
            (
                stream.source_id,
                stream.dataset_id,
                stream.episode_index,
                stream.clip_id,
            )
        ].append(stream)
    episodes: list[MixedVideoEpisodeRecord] = []
    for (
        source_id,
        dataset_id,
        episode_index,
        clip_id,
    ), episode_streams in grouped.items():
        ordered_streams = sorted(
            episode_streams,
            key=lambda item: (item.stream_index, item.stream_key),
        )
        _validate_unique_episode_target_slots(
            source_id=source_id,
            dataset_id=dataset_id,
            episode_index=episode_index,
            clip_id=clip_id,
            streams=ordered_streams,
        )
        first = ordered_streams[0]
        native_length_frames = min(
            stream.length_frames
            for stream in ordered_streams
            if stream.length_frames > 0
        )
        length_frames = min(
            _stream_normalized_length_frames(stream, data_config)
            for stream in ordered_streams
        )
        latent_lengths = [
            int(stream.latent_length_frames)
            for stream in ordered_streams
            if stream.latent_length_frames is not None
            and stream.latent_length_frames > 0
        ]
        tasks = _merge_tasks(stream.tasks for stream in ordered_streams)
        key = f"{source_id}:{dataset_id}:{episode_index}:{clip_id}"
        episodes.append(
            MixedVideoEpisodeRecord(
                key=key,
                source_id=source_id,
                source_group=first.source_group,
                repo_id=first.repo_id,
                dataset_id=dataset_id,
                episode_index=episode_index,
                clip_id=clip_id,
                native_length_frames=native_length_frames,
                length_frames=length_frames,
                latent_length_frames=min(latent_lengths)
                if latent_lengths
                else None,
                tasks=tasks,
                streams=tuple(ordered_streams),
            )
        )
    episodes.sort(
        key=lambda item: (
            item.source_id,
            item.dataset_id,
            item.episode_index,
            item.clip_id,
        )
    )
    if not episodes:
        raise ValueError("Mixed-video manifests did not produce any usable episodes.")
    return MixedVideoCatalog(episodes=tuple(episodes))


def _validate_unique_episode_target_slots(
    *,
    source_id: str,
    dataset_id: str,
    episode_index: int,
    clip_id: str,
    streams: Sequence[MixedVideoStreamRecord],
) -> None:
    by_slot: dict[str, list[MixedVideoStreamRecord]] = defaultdict(list)
    for stream in streams:
        by_slot[stream.target_slot].append(stream)
    duplicates = {
        slot: slot_streams
        for slot, slot_streams in by_slot.items()
        if len(slot_streams) > 1
    }
    if not duplicates:
        return
    details = []
    for slot, slot_streams in sorted(duplicates.items()):
        rows = [
            f"stream_key={stream.stream_key!r}, "
            f"path={stream.local_path or stream.shard_relative_path!r}, "
            f"from_timestamp={stream.from_timestamp}, "
            f"to_timestamp={stream.to_timestamp}"
            for stream in slot_streams
        ]
        details.append(f"{slot}: {rows}")
    raise ValueError(
        "Mixed-video manifests must not contain duplicate target slots within "
        "one episode group. Use distinct dataset_id/episode_index values for "
        "timestamp clips, or give each row a distinct target slot. "
        f"source_id={source_id!r}, dataset_id={dataset_id!r}, "
        f"episode_index={episode_index}, clip_id={clip_id!r}, "
        f"duplicates={details}"
    )


def _stream_normalized_length_frames(
    stream: MixedVideoStreamRecord,
    data_config: MixedVideoDataConfig,
) -> int:
    return int(stream.clip.normalized_length_frames)


def _merge_tasks(
    task_groups: Iterable[tuple[str, ...]],
) -> tuple[str, ...]:
    merged: list[str] = []
    for tasks in task_groups:
        for task in tasks:
            if task and task not in merged:
                merged.append(task)
    return tuple(merged)


__all__ = ["load_mixed_video_catalog"]
