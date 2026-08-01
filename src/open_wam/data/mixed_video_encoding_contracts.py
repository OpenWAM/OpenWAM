"""Typed contracts for offline mixed-video latent encoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict

import torch

from open_wam.configs import MixedVideoEncodingSplit, MixedVideoLatentEncodingMode
from open_wam.contracts import ViewPlacement


class MixedVideoLatentEncoder(Protocol):
    """Capability required by the mixed-video encoding engine."""

    def encode_video(
        self,
        canonical_video: torch.Tensor,
        *,
        placements: Sequence[ViewPlacement] | None = None,
        reset_cache: bool = True,
    ) -> torch.Tensor:
        del canonical_video, placements, reset_cache
        raise NotImplementedError


class MixedVideoEncodingReport(TypedDict):
    """Serialized artifact and progress report returned by one encode call."""

    output_root: str
    encoded_episodes: int
    encoded_targets: int
    manifest_encoded_episodes: int
    manifest_encoded_targets: int
    newly_encoded_episodes: int
    newly_encoded_targets: int
    reused_episodes: int
    reused_targets: int
    selected_episodes: int
    shard_episodes: int
    split: str
    shard_count: int
    shard_index: int
    source_ids: list[str]
    decode_size_mode: str
    decode_fit_mode: str
    manifest_paths: dict[str, str]
    config_patch_path: str | None
    latent_training_config_path: str | None
    latent_shapes: dict[str, list[int]]


@dataclass(frozen=True)
class MixedVideoEncodedEpisode:
    """One encoded sidecar and its manifest-facing metadata."""

    source_id: str
    dataset_id: str
    episode_index: int
    clip_id: str
    latent_path: Path
    latent_shape: tuple[int, int, int, int]
    raw_length_frames: int
    latent_length_frames: int
    tasks: tuple[str, ...]
    target_slot: str = "observation.images.slot0"
    encoded_slots: tuple[str, ...] = ()
    encoding_mode: MixedVideoLatentEncodingMode = MixedVideoLatentEncodingMode.CANONICAL

    def __post_init__(self) -> None:
        latent_shape = tuple(int(value) for value in self.latent_shape)
        tasks = (self.tasks,) if isinstance(self.tasks, str) else self.tasks
        encoded_slots = (
            (self.encoded_slots,)
            if isinstance(self.encoded_slots, str)
            else self.encoded_slots
        )
        if len(latent_shape) != 4:
            raise ValueError(
                "Mixed-video encoded episodes require a [C, T, H, W] latent shape."
            )
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        object.__setattr__(self, "episode_index", int(self.episode_index))
        object.__setattr__(self, "clip_id", str(self.clip_id))
        object.__setattr__(self, "latent_path", Path(self.latent_path))
        object.__setattr__(self, "latent_shape", latent_shape)
        object.__setattr__(self, "raw_length_frames", int(self.raw_length_frames))
        object.__setattr__(self, "latent_length_frames", int(self.latent_length_frames))
        object.__setattr__(self, "tasks", tuple(str(value) for value in tasks))
        object.__setattr__(self, "target_slot", str(self.target_slot))
        object.__setattr__(
            self, "encoded_slots", tuple(str(value) for value in encoded_slots)
        )
        object.__setattr__(
            self,
            "encoding_mode",
            MixedVideoLatentEncodingMode(self.encoding_mode),
        )


@dataclass(frozen=True)
class MixedVideoEncodingTarget:
    """One canonical or per-view latent sidecar planned for an episode."""

    name: str
    mode: MixedVideoLatentEncodingMode
    target_slot: str
    source_slots: tuple[str, ...]
    latent_path: Path
    compatible_existing_paths: tuple[Path, ...] = ()
    include_in_training_manifest: bool = True

    def __post_init__(self) -> None:
        raw_source_slots = (
            (self.source_slots,)
            if isinstance(self.source_slots, str)
            else self.source_slots
        )
        raw_existing_paths = (
            (self.compatible_existing_paths,)
            if isinstance(self.compatible_existing_paths, (str, Path))
            else self.compatible_existing_paths
        )
        source_slots = tuple(str(value) for value in raw_source_slots)
        if not source_slots:
            raise ValueError(
                "Mixed-video encoding targets require at least one source slot."
            )
        if not self.name:
            raise ValueError("Mixed-video encoding targets require a non-empty name.")
        if not self.target_slot:
            raise ValueError(
                "Mixed-video encoding targets require a non-empty target slot."
            )
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "mode", MixedVideoLatentEncodingMode(self.mode))
        object.__setattr__(self, "target_slot", str(self.target_slot))
        object.__setattr__(self, "source_slots", source_slots)
        object.__setattr__(self, "latent_path", Path(self.latent_path))
        object.__setattr__(
            self,
            "compatible_existing_paths",
            tuple(Path(path) for path in raw_existing_paths),
        )
        object.__setattr__(
            self,
            "include_in_training_manifest",
            bool(self.include_in_training_manifest),
        )


@dataclass(frozen=True)
class MixedVideoEncodingSelection:
    """Deterministic episode subset assigned to one encoding worker."""

    split: MixedVideoEncodingSplit = MixedVideoEncodingSplit.ALL
    source_ids: tuple[str, ...] = ()
    episode_indices: tuple[int, ...] = ()
    max_episodes: int | None = None
    shard_count: int = 1
    shard_index: int = 0

    def __post_init__(self) -> None:
        source_ids = (
            (self.source_ids,) if isinstance(self.source_ids, str) else self.source_ids
        )
        object.__setattr__(self, "split", MixedVideoEncodingSplit(self.split))
        object.__setattr__(
            self, "source_ids", tuple(str(value) for value in source_ids)
        )
        object.__setattr__(
            self, "episode_indices", tuple(int(value) for value in self.episode_indices)
        )
        if self.max_episodes is not None:
            object.__setattr__(self, "max_episodes", int(self.max_episodes))
        object.__setattr__(self, "shard_count", int(self.shard_count))
        object.__setattr__(self, "shard_index", int(self.shard_index))


__all__ = [
    "MixedVideoEncodedEpisode",
    "MixedVideoEncodingReport",
    "MixedVideoEncodingSelection",
    "MixedVideoEncodingTarget",
    "MixedVideoLatentEncoder",
]
