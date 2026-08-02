"""Immutable stream, episode, and catalog records for mixed video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from open_wam.configs import MixedVideoSourceFormat
from open_wam.contracts import ResolvedVideoClip


@dataclass(frozen=True)
class MixedVideoStreamRecord:
    """One decoded video stream from one source manifest row."""

    source_id: str
    source_group: str | None
    repo_id: str | None
    dataset_id: str
    episode_index: int
    clip_id: str
    stream_index: int
    stream_key: str
    target_slot: str
    source_format: MixedVideoSourceFormat
    manifest_path: Path
    local_path: Path | None
    latent_path: Path | None
    shard_relative_path: str | None
    latent_shard_relative_path: str | None
    latent_key: str
    length_frames: int
    latent_length_frames: int | None
    observation_fps: float | None
    action_fps: float | None
    from_timestamp: float | None
    to_timestamp: float | None
    width: int | None
    height: int | None
    channels: int | None
    tasks: tuple[str, ...]
    clip: ResolvedVideoClip


@dataclass(frozen=True)
class MixedVideoEpisodeRecord:
    """All streams belonging to one video episode across one source."""

    key: str
    source_id: str
    source_group: str | None
    repo_id: str | None
    dataset_id: str
    episode_index: int
    clip_id: str
    native_length_frames: int
    length_frames: int
    latent_length_frames: int | None
    tasks: tuple[str, ...]
    streams: tuple[MixedVideoStreamRecord, ...]


@dataclass(frozen=True)
class MixedVideoCatalog:
    """Resolved streams grouped into deterministic logical episodes."""

    episodes: tuple[MixedVideoEpisodeRecord, ...]


__all__ = [
    "MixedVideoCatalog",
    "MixedVideoEpisodeRecord",
    "MixedVideoStreamRecord",
]
