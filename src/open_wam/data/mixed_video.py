from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Iterator, Sequence
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

# WHY: decord's C++ batch decode is 3-5x faster than imageio's Python frame-by-frame
# iteration. We keep imageio as fallback for codec edge cases.
try:
    import decord
    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

from open_wam.configs import (
    CausalPrefixSuffixBucketConfig,
    DataConfig,
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoFrameFitMode,
    MixedVideoMissingStreamPolicy,
    MixedVideoResizeBinConfig,
    MixedVideoRandomMode,
    MixedVideoSourceFormat,
    MixedVideoSourceConfig,
    MixedVideoViewCombinationConfig,
    MixedVideoWeightMode,
)

from .contracts import WAMSample
from .latent_contracts import LatentWAMSample
from open_wam.utils.video_timeline import (
    ResolvedVideoClip,
    normalized_video_frame_count as _timeline_normalized_video_frame_count,
    resolve_video_source_fps,
)


_TIMESTAMP_BOUNDARY_EPSILON_SECONDS = 1e-4


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
class MixedVideoWindowRecord:
    """One fixed-length video-only training window."""

    episode_key: str
    observation_start: int
    observed_prefix_frames: int
    future_suffix_frames: int
    view_combination_name: str | None = None
    view_combination_slots: tuple[str, ...] = ()

    @property
    def valid_video_frames(self) -> int:
        return self.observed_prefix_frames + self.future_suffix_frames


@dataclass(frozen=True)
class MixedVideoCatalog:
    episodes: tuple[MixedVideoEpisodeRecord, ...]


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
    source = resolve_mixed_video_observation_fps(source_fps, missing_observation_fps=missing_source_fps)
    target = float(target_fps)
    if target <= 0:
        raise ValueError("`target_fps` must be positive or None.")
    return _timeline_normalized_video_frame_count(length, source_fps=source, target_fps=target)


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

    source = resolve_mixed_video_observation_fps(source_fps, missing_observation_fps=missing_source_fps)
    resolved_native_total = int(frames.shape[0]) if native_total_frames is None else int(native_total_frames)
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


class MixedVideoTrainSampler(Sampler[int]):
    """Source-balanced sampler for mixed-video training.

    The sampler keeps the nmotions-style "federated" property: one epoch draws
    from all sources according to configured source weights instead of relying
    on global shuffle over a concatenated index.
    """

    def __init__(
        self,
        dataset: MixedVideoWindowDataset,
        *,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.dataset = dataset
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Invalid sampler rank={rank} for world_size={world_size}.")
        self.epoch = 0
        self.num_samples = 0
        self.total_size = 0
        self._epoch_order: tuple[int, ...] = ()
        self._refresh_epoch_order()

    def set_epoch(self, epoch: int) -> None:
        """Refresh the deterministic source-balanced order for one training epoch."""

        self.epoch = int(epoch)
        self._refresh_epoch_order()

    def _refresh_epoch_order(self) -> None:
        base_order = tuple(self.dataset.build_epoch_index_order(epoch=self.epoch))
        if not base_order:
            self.num_samples = 0
            self.total_size = 0
            self._epoch_order = ()
            return
        self.num_samples = int(math.ceil(len(base_order) / self.world_size))
        self.total_size = self.num_samples * self.world_size
        padding_size = self.total_size - len(base_order)
        if padding_size <= 0:
            self._epoch_order = base_order[: self.total_size]
            return
        repeats = (padding_size + len(base_order) - 1) // len(base_order)
        padding = (list(base_order) * repeats)[:padding_size]
        self._epoch_order = tuple(list(base_order) + padding)

    def __iter__(self) -> Iterator[int]:
        yield from self._epoch_order[self.rank : self.total_size : self.world_size]

    def __len__(self) -> int:
        return self.num_samples


class MixedVideoWindowDataset(Dataset[WAMSample]):
    """Manifest-backed multi-source RGB video dataset for video-only training."""

    def __init__(
        self,
        data_config: MixedVideoDataConfig,
        *,
        catalog: MixedVideoCatalog,
        split: str,
        episode_keys: Sequence[str],
    ) -> None:
        self.data_config = data_config
        self.catalog = catalog
        self.split = split
        self.episode_records = {episode.key: episode for episode in catalog.episodes}
        self.episode_keys = tuple(episode_keys)
        self._validate_source_formats()
        self.sample_index = self._build_sample_index()
        self._video_frame_cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        if not self.sample_index:
            raise ValueError(
                f"No valid mixed-video windows were constructed for split='{split}'. "
                f"Check num_frames={data_config.num_frames}, frame_stride={data_config.frame_stride}, "
                f"sample_stride={data_config.sample_stride}, and selected episodes={len(episode_keys)}."
            )

    def _episode_window_length_frames(self, episode: MixedVideoEpisodeRecord) -> int:
        return int(episode.length_frames)

    def _allowed_source_formats(self) -> frozenset[MixedVideoSourceFormat]:
        return frozenset({MixedVideoSourceFormat.RGB, MixedVideoSourceFormat.RGB_AND_LATENT})

    def _configured_stream_slots(self) -> tuple[str, ...]:
        return tuple(self.data_config.camera_names)

    def _source_format_adapter_name(self) -> str:
        return "trainer.batch_adapter=views"

    def _validate_source_formats(self) -> None:
        allowed = self._allowed_source_formats()
        configured_slots = set(self._configured_stream_slots())
        invalid: list[str] = []
        for episode_key in self.episode_keys:
            episode = self.episode_records[episode_key]
            for stream in episode.streams:
                if stream.target_slot not in configured_slots:
                    continue
                if stream.source_format not in allowed:
                    invalid.append(f"{stream.source_id}:{stream.stream_key}={stream.source_format.value}")
        if invalid:
            allowed_values = ", ".join(sorted(format_value.value for format_value in allowed))
            raise ValueError(
                f"Mixed-video source_format incompatible with {self._source_format_adapter_name()}: "
                f"{sorted(set(invalid))}. Allowed source formats: {allowed_values}."
            )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> WAMSample:
        window = self.sample_index[index]
        episode = self.episode_records[window.episode_key]
        frame_indices = [
            window.observation_start + offset * self.data_config.frame_stride
            for offset in range(window.valid_video_frames)
        ]
        views = self._build_views(episode, frame_indices, valid_frame_count=window.valid_video_frames)
        decode_sizes = {
            stream.target_slot: resolve_mixed_video_decode_size(
                self.data_config,
                source_height=stream.height,
                source_width=stream.width,
            )
            for stream in episode.streams
            if stream.target_slot in self.data_config.camera_names
        }
        action_shape = (
            self.data_config.action_schema.action_horizon,
            self.data_config.action_schema.action_dim,
        )
        state_shape = (
            self.data_config.action_schema.state_horizon,
            self.data_config.action_schema.state_dim,
        )
        task_text = episode.tasks[0] if episode.tasks else None
        return WAMSample(
            views=views,
            actions=torch.zeros(action_shape, dtype=torch.float32),
            action_mask=torch.zeros(action_shape, dtype=torch.float32),
            state=torch.zeros(state_shape, dtype=torch.float32),
            state_mask=torch.zeros(state_shape, dtype=torch.float32),
            task_text=task_text,
            metadata={
                "dataset_type": self.data_config.dataset_type,
                "source_id": episode.source_id,
                "source_group": episode.source_group,
                "repo_id": episode.repo_id,
                "dataset_id": episode.dataset_id,
                "episode_index": episode.episode_index,
                "clip_id": episode.clip_id,
                "split": self.split,
                "observation_start": window.observation_start,
                "observation_frame_indices": [int(value) for value in frame_indices],
                "observed_prefix_frames": window.observed_prefix_frames,
                "future_suffix_frames": window.future_suffix_frames,
                "valid_video_frames": window.valid_video_frames,
                "padded_video_frames": self.data_config.num_frames,
                "native_length_frames": episode.native_length_frames,
                "normalized_length_frames": episode.length_frames,
                "target_observation_fps": self.data_config.target_observation_fps,
                "decode_size_mode": self.data_config.decode_size_mode.value,
                "decode_fit_mode": self.data_config.decode_fit_mode.value,
                "decode_height": int(next(iter(views.values())).shape[1]) if views else self.data_config.decode_height,
                "decode_width": int(next(iter(views.values())).shape[2]) if views else self.data_config.decode_width,
                "decode_bins": {
                    slot: resolved.bin_name
                    for slot, resolved in decode_sizes.items()
                },
                "source_video_shapes": {
                    slot: [resolved.source_height, resolved.source_width]
                    for slot, resolved in decode_sizes.items()
                },
                "source_observation_fps": {
                    stream.target_slot: _stream_source_observation_fps(stream, self.data_config)
                    for stream in episode.streams
                    if stream.target_slot in self.data_config.camera_names
                },
                "source_observation_fps_source": {
                    stream.target_slot: stream.clip.source_fps_source
                    for stream in episode.streams
                    if stream.target_slot in self.data_config.camera_names
                },
                "stream_keys": {
                    stream.target_slot: stream.stream_key
                    for stream in episode.streams
                    if stream.target_slot in self.data_config.camera_names
                },
                "tasks": list(episode.tasks),
            },
        )

    def build_train_sampler(self, *, world_size: int = 1, rank: int = 0) -> MixedVideoTrainSampler:
        return MixedVideoTrainSampler(self, world_size=world_size, rank=rank)

    def build_epoch_index_order(self, *, epoch: int = 0) -> tuple[int, ...]:
        source_to_indices: dict[str, list[int]] = defaultdict(list)
        for sample_index, window in enumerate(self.sample_index):
            episode = self.episode_records[window.episode_key]
            source_to_indices[episode.source_id].append(sample_index)
        if not source_to_indices:
            return ()
        source_counts = {
            source_id: len(indices)
            for source_id, indices in source_to_indices.items()
        }
        target_counts = _source_target_counts(
            self.data_config,
            source_counts,
        )
        rng = random.Random(int(self.data_config.sampling_seed) + int(epoch))
        per_source_orders: dict[str, list[int]] = {}
        for source_id, indices in source_to_indices.items():
            order = list(indices)
            if self.data_config.random_mode == MixedVideoRandomMode.WITHIN_SOURCE:
                rng.shuffle(order)
            per_source_orders[source_id] = _repeat_or_trim(order, target_counts[source_id])

        source_cycle = _weighted_source_cycle(target_counts)
        epoch_order: list[int] = []
        source_offsets = {source_id: 0 for source_id in per_source_orders}
        for source_id in source_cycle:
            offset = source_offsets[source_id]
            source_order = per_source_orders[source_id]
            if offset >= len(source_order):
                continue
            epoch_order.append(source_order[offset])
            source_offsets[source_id] = offset + 1
        if self.data_config.random_mode == MixedVideoRandomMode.GLOBAL:
            rng.shuffle(epoch_order)
        return tuple(epoch_order)

    def _build_views(
        self,
        episode: MixedVideoEpisodeRecord,
        frame_indices: Sequence[int],
        *,
        valid_frame_count: int,
    ) -> dict[str, torch.Tensor]:
        streams_by_slot: dict[str, MixedVideoStreamRecord] = {}
        for stream in sorted(episode.streams, key=lambda item: item.stream_index):
            streams_by_slot.setdefault(stream.target_slot, stream)

        views: dict[str, torch.Tensor] = {}
        for camera_name in self.data_config.camera_names:
            stream = streams_by_slot.get(camera_name)
            if stream is None:
                views[camera_name] = self._missing_stream_tensor(camera_name)
                continue
            if stream.source_format not in {
                MixedVideoSourceFormat.RGB,
                MixedVideoSourceFormat.RGB_AND_LATENT,
            }:
                raise ValueError(
                    f"Mixed-video source={stream.source_id!r} is configured as {stream.source_format.value!r} "
                    "and cannot be emitted through the RGB/views batch adapter. Use source_format=rgb or "
                    "rgb_and_latent, or switch trainer.batch_adapter to latents."
                )
            frames = self._load_stream_frames(stream)
            index_tensor = torch.tensor(frame_indices, dtype=torch.long)
            if index_tensor.numel() and int(index_tensor.max().item()) >= int(frames.shape[0]):
                raise IndexError(
                    f"Mixed-video sample requested frame {int(index_tensor.max().item())} from "
                    f"source={stream.source_id}, episode={stream.episode_index}, stream={stream.stream_key}, "
                    f"but decoded stream has {frames.shape[0]} frames."
                )
            selected = frames.index_select(0, index_tensor)
            views[camera_name] = self._pad_view_frames(selected, valid_frame_count=valid_frame_count)
        return views

    def _pad_view_frames(self, frames: torch.Tensor, *, valid_frame_count: int) -> torch.Tensor:
        padded_frames = int(self.data_config.num_frames)
        if frames.shape[0] != int(valid_frame_count):
            raise ValueError(
                f"Mixed-video selected frame count mismatch: got {frames.shape[0]}, expected {valid_frame_count}."
            )
        if frames.shape[0] > padded_frames:
            raise ValueError(
                f"Mixed-video bucket requested {frames.shape[0]} frames, but data.num_frames={padded_frames}."
            )
        if frames.shape[0] == padded_frames:
            return frames.contiguous()
        padding = torch.zeros(
            padded_frames - frames.shape[0],
            frames.shape[1],
            frames.shape[2],
            frames.shape[3],
            dtype=frames.dtype,
            device=frames.device,
        )
        return torch.cat([frames, padding], dim=0).contiguous()

    def _missing_stream_tensor(self, camera_name: str) -> torch.Tensor:
        if self.data_config.missing_stream_policy == MixedVideoMissingStreamPolicy.ERROR:
            raise KeyError(
                f"Mixed-video episode is missing configured stream slot '{camera_name}'. "
                "Use missing_stream_policy=zero_fill if this is expected."
            )
        resolved_size = resolve_mixed_video_decode_size(self.data_config, source_height=None, source_width=None)
        return torch.zeros(
            (
                self.data_config.num_frames,
                resolved_size.height,
                resolved_size.width,
                3,
            ),
            dtype=torch.uint8,
        )

    def _load_stream_frames(self, stream: MixedVideoStreamRecord) -> torch.Tensor:
        cache_key = (stream.source_id, _video_stream_cache_key(stream, self.data_config))
        if cache_key in self._video_frame_cache:
            self._video_frame_cache.move_to_end(cache_key)
            return self._video_frame_cache[cache_key]

        path = _resolve_stream_path(stream, cache_dir=self.data_config.cache_dir)
        resolved_size = resolve_mixed_video_decode_size(
            self.data_config,
            source_height=stream.height,
            source_width=stream.width,
        )
        decoded = decode_video_frames(
            path,
            target_height=resolved_size.height,
            target_width=resolved_size.width,
            center_crop=self.data_config.decode_center_crop,
            allow_upscale=self.data_config.decode_allow_upscale,
            fit_mode=self.data_config.decode_fit_mode,
            source_fps=stream.observation_fps,
            target_fps=self.data_config.target_observation_fps,
            missing_source_fps=self.data_config.missing_observation_fps,
            from_timestamp=stream.from_timestamp,
            to_timestamp=stream.to_timestamp,
            data_config=self.data_config if stream.height is None or stream.width is None else None,
        )
        self._video_frame_cache[cache_key] = decoded
        max_entries = max(1, int(self.data_config.episode_cache_size) * max(1, len(self.data_config.camera_names)))
        while len(self._video_frame_cache) > max_entries:
            self._video_frame_cache.popitem(last=False)
        return decoded

    def _build_sample_index(self) -> tuple[MixedVideoWindowRecord, ...]:
        windows: list[MixedVideoWindowRecord] = []
        for episode_key in self.episode_keys:
            episode = self.episode_records[episode_key]
            episode_length = self._episode_window_length_frames(episode)
            if episode_length <= 0:
                continue
            for start in range(0, episode_length, self.data_config.sample_stride):
                bucket = _select_valid_causal_bucket(
                    self.data_config,
                    episode,
                    start,
                    episode_length=episode_length,
                )
                if bucket is None:
                    continue
                windows.append(
                    MixedVideoWindowRecord(
                        episode_key=episode_key,
                        observation_start=start,
                        observed_prefix_frames=bucket.observed_frames,
                        future_suffix_frames=bucket.future_frames,
                    )
                )
        return tuple(windows)


class MixedVideoLatentWindowDataset(MixedVideoWindowDataset):
    """Manifest-backed latent-first mixed-video dataset.

    This reuses the same mixed-video catalog and source-balanced sampler as the
    RGB path, but loads precomputed VAE latents from manifest sidecars. It is
    the intended path for mixing RGB-origin and latent-origin sources once RGB
    manifests have been encoded by a separate job.
    """

    def __init__(
        self,
        data_config: MixedVideoDataConfig,
        *,
        catalog: MixedVideoCatalog,
        split: str,
        episode_keys: Sequence[str],
    ) -> None:
        super().__init__(data_config, catalog=catalog, split=split, episode_keys=episode_keys)
        self._video_frame_cache.clear()
        self._latent_cache: OrderedDict[tuple[str, str, str], torch.Tensor] = OrderedDict()

    def _allowed_source_formats(self) -> frozenset[MixedVideoSourceFormat]:
        return frozenset({MixedVideoSourceFormat.LATENT, MixedVideoSourceFormat.RGB_AND_LATENT})

    def _configured_stream_slots(self) -> tuple[str, ...]:
        slots = list(self.data_config.latent_camera_names)
        for combination in self.data_config.latent_view_combinations:
            if combination.enabled:
                slots.extend(combination.slots)
        if len(self.data_config.camera_names) == 1:
            slots.append(self.data_config.camera_names[0])
        return tuple(dict.fromkeys(slots))

    def _source_format_adapter_name(self) -> str:
        return "trainer.batch_adapter=latents"

    def _build_sample_index(self) -> tuple[MixedVideoWindowRecord, ...]:
        windows: list[MixedVideoWindowRecord] = []
        for episode_key in self.episode_keys:
            episode = self.episode_records[episode_key]
            combinations = _valid_latent_view_combinations(self.data_config, episode)
            for combination in combinations:
                episode_length = _latent_combination_length_frames(episode, combination.slots)
                if episode_length <= 0:
                    continue
                repeat_count = _latent_view_combination_repeat_count(combination, combinations)
                for start in range(0, episode_length, self.data_config.sample_stride):
                    bucket = _select_valid_causal_bucket(
                        self.data_config,
                        episode,
                        start,
                        episode_length=episode_length,
                    )
                    if bucket is None:
                        continue
                    for _ in range(repeat_count):
                        windows.append(
                            MixedVideoWindowRecord(
                                episode_key=episode_key,
                                observation_start=start,
                                observed_prefix_frames=bucket.observed_frames,
                                future_suffix_frames=bucket.future_frames,
                                view_combination_name=combination.name,
                                view_combination_slots=combination.slots,
                            )
                        )
        return tuple(windows)

    def __getitem__(self, index: int) -> LatentWAMSample:
        window = self.sample_index[index]
        episode = self.episode_records[window.episode_key]
        frame_indices = [
            window.observation_start + offset * self.data_config.frame_stride
            for offset in range(window.valid_video_frames)
        ]
        video_latents, assembly_metadata = self._build_latents(
            episode,
            frame_indices,
            valid_frame_count=window.valid_video_frames,
            view_combination_slots=window.view_combination_slots,
        )
        action_shape = (
            self.data_config.action_schema.action_horizon,
            self.data_config.action_schema.action_dim,
        )
        state_shape = (
            self.data_config.action_schema.state_horizon,
            self.data_config.action_schema.state_dim,
        )
        task_text = episode.tasks[0] if episode.tasks else None
        return LatentWAMSample(
            video_latents=video_latents,
            actions=torch.zeros(action_shape, dtype=torch.float32),
            action_mask=torch.zeros(action_shape, dtype=torch.float32),
            state=torch.zeros(state_shape, dtype=torch.float32),
            state_mask=torch.zeros(state_shape, dtype=torch.float32),
            task_text=task_text,
            metadata={
                "dataset_type": self.data_config.dataset_type,
                "mixed_video_training_input": "latents",
                "source_id": episode.source_id,
                "source_group": episode.source_group,
                "repo_id": episode.repo_id,
                "dataset_id": episode.dataset_id,
                "episode_index": episode.episode_index,
                "clip_id": episode.clip_id,
                "split": self.split,
                "observation_start": window.observation_start,
                "observation_frame_indices": [int(value) for value in frame_indices],
                "observed_prefix_frames": window.observed_prefix_frames,
                "future_suffix_frames": window.future_suffix_frames,
                "valid_video_frames": window.valid_video_frames,
                "padded_video_frames": self.data_config.num_frames,
                "latent_shape": list(video_latents.shape),
                "view_combination_name": window.view_combination_name,
                "view_combination_slots": list(window.view_combination_slots),
                "latent_view_assembly": assembly_metadata,
                "stream_keys": {
                    stream.target_slot: stream.stream_key
                    for stream in episode.streams
                    if stream.target_slot in window.view_combination_slots
                },
                "tasks": list(episode.tasks),
            },
        )

    def _build_latents(
        self,
        episode: MixedVideoEpisodeRecord,
        frame_indices: Sequence[int],
        *,
        valid_frame_count: int,
        view_combination_slots: Sequence[str],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        streams_by_slot: dict[str, MixedVideoStreamRecord] = {}
        for stream in sorted(episode.streams, key=lambda item: item.stream_index):
            streams_by_slot.setdefault(stream.target_slot, stream)
        slots = tuple(str(slot) for slot in view_combination_slots)
        if not slots:
            valid_combinations = _valid_latent_view_combinations(self.data_config, episode)
            if not valid_combinations:
                raise KeyError(f"Mixed-video latent episode {episode.key!r} has no valid latent view combinations.")
            slots = valid_combinations[0].slots
        selected_latents: list[torch.Tensor] = []
        index_tensor = torch.tensor(frame_indices, dtype=torch.long)
        for slot in slots:
            stream = streams_by_slot.get(slot)
            if stream is None:
                raise KeyError(f"Mixed-video latent episode is missing configured stream slot {slot!r}.")
            if stream.source_format not in {
                MixedVideoSourceFormat.LATENT,
                MixedVideoSourceFormat.RGB_AND_LATENT,
            }:
                raise ValueError(
                    f"Mixed-video source={stream.source_id!r} is configured as {stream.source_format.value!r} "
                    "and has no latent sidecar for trainer.batch_adapter=latents. Encode this source first or "
                    "set source_format=rgb_and_latent/latent."
                )
            latents = self._load_stream_latents(stream)
            if index_tensor.numel() and int(index_tensor.max().item()) >= int(latents.shape[1]):
                raise IndexError(
                    f"Mixed-video sample requested latent frame {int(index_tensor.max().item())} from "
                    f"source={stream.source_id}, episode={stream.episode_index}, stream={stream.stream_key}, "
                    f"but decoded latent stream has {latents.shape[1]} frames."
                )
            selected_latents.append(latents.index_select(1, index_tensor))
        assembled, assembly_metadata = assemble_mixed_video_latent_views(
            selected_latents,
            slots=slots,
            canvas_view_count=_latent_view_assembly_canvas_view_count(self.data_config),
        )
        return (
            self._pad_latent_frames(assembled, valid_frame_count=valid_frame_count),
            assembly_metadata,
        )

    def _pad_latent_frames(self, latents: torch.Tensor, *, valid_frame_count: int) -> torch.Tensor:
        padded_frames = int(self.data_config.num_frames)
        if latents.shape[1] != int(valid_frame_count):
            raise ValueError(
                f"Mixed-video selected latent count mismatch: got {latents.shape[1]}, expected {valid_frame_count}."
            )
        if latents.shape[1] > padded_frames:
            raise ValueError(
                f"Mixed-video bucket requested {latents.shape[1]} latent frames, "
                f"but data.num_frames={padded_frames}."
            )
        if latents.shape[1] == padded_frames:
            return latents.contiguous()
        padding = torch.zeros(
            latents.shape[0],
            padded_frames - latents.shape[1],
            latents.shape[2],
            latents.shape[3],
            dtype=latents.dtype,
            device=latents.device,
        )
        return torch.cat([latents, padding], dim=1).contiguous()

    def _load_stream_latents(self, stream: MixedVideoStreamRecord) -> torch.Tensor:
        cache_key = (stream.source_id, _latent_stream_cache_key(stream), stream.latent_key)
        if cache_key in self._latent_cache:
            self._latent_cache.move_to_end(cache_key)
            return self._latent_cache[cache_key]
        path = _resolve_stream_latent_path(stream, cache_dir=self.data_config.cache_dir)
        latents = _load_latent_tensor(path, key=stream.latent_key)
        self._latent_cache[cache_key] = latents
        max_entries = max(1, int(self.data_config.episode_cache_size) * max(1, len(self.data_config.camera_names)))
        while len(self._latent_cache) > max_entries:
            self._latent_cache.popitem(last=False)
        return latents


def assemble_mixed_video_latent_views(
    latents_by_slot: Sequence[torch.Tensor],
    *,
    slots: Sequence[str],
    canvas_view_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Assemble 1-4 same-resolution latent views into a deterministic canvas."""

    latents = tuple(latents_by_slot)
    slot_names = tuple(str(slot) for slot in slots)
    if len(latents) != len(slot_names):
        raise ValueError(f"Expected one latent tensor per slot, got {len(latents)} tensors and {len(slot_names)} slots.")
    if not 1 <= len(latents) <= 4:
        raise ValueError(f"Mixed-video latent view assembly supports 1 to 4 views, got {len(latents)}.")
    first = latents[0]
    if first.ndim != 4:
        raise ValueError(f"Expected latent views shaped [C,T,H,W], got {tuple(first.shape)}.")
    channels, frames, height, width = (int(value) for value in first.shape)
    for slot, latent in zip(slot_names, latents, strict=True):
        if latent.ndim != 4:
            raise ValueError(f"Expected latent view {slot!r} shaped [C,T,H,W], got {tuple(latent.shape)}.")
        if tuple(int(value) for value in latent.shape) != (channels, frames, height, width):
            raise ValueError(
                "Mixed-video latent view assembly requires same-resolution views, "
                f"got first={(channels, frames, height, width)} and {slot!r}={tuple(latent.shape)}."
            )

    resolved_canvas_views = int(canvas_view_count or len(latents))
    if not 1 <= resolved_canvas_views <= 4:
        raise ValueError(f"Mixed-video latent assembly canvas supports 1 to 4 views, got {resolved_canvas_views}.")
    if resolved_canvas_views < len(latents):
        raise ValueError(
            f"Assembly canvas for {resolved_canvas_views} views cannot hold {len(latents)} selected views."
        )
    canvas_height, canvas_width = _latent_assembly_canvas_shape(
        resolved_canvas_views,
        view_height=height,
        view_width=width,
    )
    canvas = first.new_zeros(channels, frames, canvas_height, canvas_width)
    placements = _latent_assembly_placements(
        selected_view_count=len(latents),
        canvas_view_count=resolved_canvas_views,
        view_height=height,
        view_width=width,
    )
    placement_metadata: list[dict[str, Any]] = []
    for slot, latent, (top, left) in zip(slot_names, latents, placements, strict=True):
        canvas[:, :, top : top + height, left : left + width] = latent
        placement_metadata.append(
            {
                "slot": slot,
                "top": int(top),
                "left": int(left),
                "height": int(height),
                "width": int(width),
            }
        )
    return canvas.contiguous(), {
        "slots": list(slot_names),
        "canvas_view_count": resolved_canvas_views,
        "canvas_height": int(canvas_height),
        "canvas_width": int(canvas_width),
        "view_height": int(height),
        "view_width": int(width),
        "placements": placement_metadata,
    }


def _latent_assembly_canvas_shape(
    view_count: int,
    *,
    view_height: int,
    view_width: int,
) -> tuple[int, int]:
    if view_count == 1:
        return int(view_height), int(view_width)
    if view_count == 2:
        return int(view_height), int(view_width) * 2
    if view_count in {3, 4}:
        return int(view_height) * 2, int(view_width) * 2
    raise ValueError(f"Mixed-video latent assembly supports 1 to 4 views, got {view_count}.")


def _latent_assembly_placements(
    *,
    selected_view_count: int,
    canvas_view_count: int,
    view_height: int,
    view_width: int,
) -> tuple[tuple[int, int], ...]:
    canvas_height, canvas_width = _latent_assembly_canvas_shape(
        canvas_view_count,
        view_height=view_height,
        view_width=view_width,
    )
    if selected_view_count == 1:
        return ((max(0, (canvas_height - view_height) // 2), max(0, (canvas_width - view_width) // 2)),)
    if selected_view_count == 2:
        return ((0, 0), (0, view_width))
    if selected_view_count == 3:
        return ((0, 0), (0, view_width), (view_height, max(0, (canvas_width - view_width) // 2)))
    if selected_view_count == 4:
        return ((0, 0), (0, view_width), (view_height, 0), (view_height, view_width))
    raise ValueError(f"Mixed-video latent assembly supports 1 to 4 views, got {selected_view_count}.")


def _latent_view_assembly_canvas_view_count(data_config: MixedVideoDataConfig) -> int:
    enabled = [combo for combo in data_config.latent_view_combinations if combo.enabled]
    if enabled:
        return max(len(combo.slots) for combo in enabled)
    return max(1, min(4, len(data_config.latent_camera_names)))


def _valid_latent_view_combinations(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
) -> tuple[MixedVideoViewCombinationConfig, ...]:
    streams_by_slot = {stream.target_slot: stream for stream in episode.streams}
    configured_slots = tuple(dict.fromkeys(data_config.latent_camera_names or data_config.camera_names))
    present_slots = tuple(slot for slot in configured_slots if slot in streams_by_slot)
    if data_config.latent_view_combinations:
        valid: list[MixedVideoViewCombinationConfig] = []
        for combination in data_config.latent_view_combinations:
            if not combination.enabled:
                continue
            if combination.source_ids and episode.source_id not in combination.source_ids:
                continue
            if all(slot in streams_by_slot for slot in combination.slots):
                valid.append(combination)
        return tuple(valid)
    if not present_slots:
        return ()
    return (
        MixedVideoViewCombinationConfig(
            name="all_available",
            slots=present_slots,
            sampling_weight=1.0,
        ),
    )


def _latent_view_combination_repeat_count(
    combination: MixedVideoViewCombinationConfig,
    combinations: Sequence[MixedVideoViewCombinationConfig],
) -> int:
    positive_weights = [float(item.sampling_weight) for item in combinations if item.enabled]
    if not positive_weights:
        return 1
    scale = min(positive_weights)
    return max(1, int(round(float(combination.sampling_weight) / scale)))


def _latent_combination_length_frames(
    episode: MixedVideoEpisodeRecord,
    slots: Sequence[str],
) -> int:
    streams_by_slot = {stream.target_slot: stream for stream in episode.streams}
    lengths: list[int] = []
    for slot in slots:
        stream = streams_by_slot.get(slot)
        if stream is None:
            raise KeyError(f"Mixed-video latent episode {episode.key!r} is missing slot {slot!r}.")
        if stream.latent_length_frames is None:
            raise ValueError(
                f"Mixed-video episode {episode.key!r}, slot {slot!r} has no latent_length_frames; "
                "latent training requires manifest latent sidecars."
            )
        lengths.append(int(stream.latent_length_frames))
    return min(lengths) if lengths else 0


def build_mixed_video_train_val_datasets(
    data_config: DataConfig,
) -> tuple[MixedVideoWindowDataset, MixedVideoWindowDataset]:
    if not isinstance(data_config, MixedVideoDataConfig):
        raise TypeError("`mixed_video` builder requires MixedVideoDataConfig.")
    catalog = load_mixed_video_catalog(data_config)
    train_keys, val_keys = split_mixed_video_episodes(data_config, catalog)
    return (
        MixedVideoWindowDataset(data_config, catalog=catalog, split="train", episode_keys=train_keys),
        MixedVideoWindowDataset(data_config, catalog=catalog, split="val", episode_keys=val_keys),
    )


def build_mixed_video_latent_train_val_datasets(
    data_config: DataConfig,
) -> tuple[MixedVideoLatentWindowDataset, MixedVideoLatentWindowDataset]:
    if not isinstance(data_config, MixedVideoDataConfig):
        raise TypeError("`mixed_video` latent builder requires MixedVideoDataConfig.")
    catalog = load_mixed_video_catalog(data_config)
    train_keys, val_keys = split_mixed_video_episodes(data_config, catalog)
    return (
        MixedVideoLatentWindowDataset(data_config, catalog=catalog, split="train", episode_keys=train_keys),
        MixedVideoLatentWindowDataset(data_config, catalog=catalog, split="val", episode_keys=val_keys),
    )


def load_mixed_video_catalog(data_config: MixedVideoDataConfig) -> MixedVideoCatalog:
    streams: list[MixedVideoStreamRecord] = []
    for source in data_config.video_sources:
        if not source.enabled:
            continue
        streams.extend(_load_source_streams(source, data_config))
    grouped: dict[tuple[str, str, int, str], list[MixedVideoStreamRecord]] = defaultdict(list)
    for stream in streams:
        grouped[(stream.source_id, stream.dataset_id, stream.episode_index, stream.clip_id)].append(stream)
    episodes: list[MixedVideoEpisodeRecord] = []
    for (source_id, dataset_id, episode_index, clip_id), episode_streams in grouped.items():
        ordered_streams = sorted(episode_streams, key=lambda item: (item.stream_index, item.stream_key))
        _validate_unique_episode_target_slots(
            source_id=source_id,
            dataset_id=dataset_id,
            episode_index=episode_index,
            clip_id=clip_id,
            streams=ordered_streams,
        )
        first = ordered_streams[0]
        native_length_frames = min(stream.length_frames for stream in ordered_streams if stream.length_frames > 0)
        length_frames = min(_stream_normalized_length_frames(stream, data_config) for stream in ordered_streams)
        latent_lengths = [
            int(stream.latent_length_frames)
            for stream in ordered_streams
            if stream.latent_length_frames is not None and stream.latent_length_frames > 0
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
                latent_length_frames=min(latent_lengths) if latent_lengths else None,
                tasks=tasks,
                streams=tuple(ordered_streams),
            )
        )
    episodes.sort(key=lambda item: (item.source_id, item.dataset_id, item.episode_index, item.clip_id))
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
    duplicates = {slot: slot_streams for slot, slot_streams in by_slot.items() if len(slot_streams) > 1}
    if not duplicates:
        return
    details = []
    for slot, slot_streams in sorted(duplicates.items()):
        rows = [
            f"stream_key={stream.stream_key!r}, path={stream.local_path or stream.shard_relative_path!r}, "
            f"from_timestamp={stream.from_timestamp}, to_timestamp={stream.to_timestamp}"
            for stream in slot_streams
        ]
        details.append(f"{slot}: {rows}")
    raise ValueError(
        "Mixed-video manifests must not contain duplicate target slots within one episode group. "
        "Use distinct dataset_id/episode_index values for timestamp clips, or give each row a distinct target slot. "
        f"source_id={source_id!r}, dataset_id={dataset_id!r}, episode_index={episode_index}, "
        f"clip_id={clip_id!r}, duplicates={details}"
    )


def _stream_source_observation_fps(stream: MixedVideoStreamRecord, data_config: MixedVideoDataConfig) -> float:
    return float(stream.clip.source_fps)


def _stream_normalized_length_frames(stream: MixedVideoStreamRecord, data_config: MixedVideoDataConfig) -> int:
    return int(stream.clip.normalized_length_frames)


def split_mixed_video_episodes(
    data_config: MixedVideoDataConfig,
    catalog: MixedVideoCatalog,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    group_to_episode_keys: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for episode in catalog.episodes:
        group_to_episode_keys[_physical_episode_group_key(episode)].append(episode.key)
    group_keys = list(group_to_episode_keys)
    rng = random.Random(int(data_config.split_seed))
    rng.shuffle(group_keys)
    train_count = int(len(group_keys) * float(data_config.train_fraction))
    train_count = min(max(train_count, 1), len(group_keys)) if group_keys else 0
    train_group_list = group_keys[:train_count]
    val_group_list = group_keys[train_count:]
    if data_config.max_train_episodes is not None:
        train_group_list = train_group_list[: data_config.max_train_episodes]
    if data_config.max_val_episodes is not None:
        val_group_list = val_group_list[: data_config.max_val_episodes]
    train_keys = [
        episode_key
        for group_key in train_group_list
        for episode_key in group_to_episode_keys[group_key]
    ]
    val_keys = [
        episode_key
        for group_key in val_group_list
        for episode_key in group_to_episode_keys[group_key]
    ]
    if not val_keys and train_group_list:
        val_keys = list(group_to_episode_keys[train_group_list[0]])
    return tuple(sorted(train_keys)), tuple(sorted(val_keys))


def _physical_episode_group_key(episode: MixedVideoEpisodeRecord) -> tuple[object, ...]:
    path_keys = tuple(sorted({stream.clip.path_key for stream in episode.streams}))
    return (episode.source_id, episode.dataset_id, episode.episode_index, path_keys)


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
            frames.append(torch.as_tensor(np.array(transformed, copy=True), dtype=torch.uint8))
        if not frames:
            raise ValueError(f"Video file has no decodable frames: {path}")
        decoded = torch.stack(frames, dim=0)
        effective_source_fps = source_fps if source_fps is not None and float(source_fps) > 0.0 else fps
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
    """Decode normalized target-frame chunks through the shared mixed-video timeline contract.

    WHY decord path: C++ batch decode is 3-5x faster than imageio's Python
    frame-by-frame iteration. Falls back to imageio for unsupported codecs.
    """

    if not raw_chunk_ranges:
        return
    for start_frame, end_frame in raw_chunk_ranges:
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(f"Invalid frame chunk [{start_frame}, {end_frame}).")
    for previous, current in zip(raw_chunk_ranges, raw_chunk_ranges[1:], strict=False):
        if previous[1] != current[0]:
            raise ValueError(f"Frame chunks must be contiguous for streaming decode: {raw_chunk_ranges!r}.")
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
                f"Frame chunk [{chunk_start}, {chunk_end}) exceeds normalized stream length {target_length} "
                f"for source={stream.source_id}, episode={stream.episode_index}, stream={stream.stream_key}."
            )

    # WHY try decord first: batch C++ decode avoids N Python round-trips per frame;
    # imageio fallback handles rare codec incompatibilities decord can't open.
    if _HAS_DECORD:
        emitted_decord_chunk = False
        try:
            for chunk in _iter_chunks_decord(
                path, data_config=data_config, stream=stream, chunk_specs=chunk_specs,
                resolved_size=resolved_size, source_fps=source_fps, target_fps=target_fps,
            ):
                emitted_decord_chunk = True
                yield chunk
            return
        except Exception:
            if emitted_decord_chunk:
                raise
            pass  # WHY silent fallback: decord may fail on unusual containers (e.g. webm)

    yield from _iter_chunks_imageio(
        path, data_config=data_config, stream=stream, chunk_specs=chunk_specs,
        resolved_size=resolved_size, source_fps=source_fps, target_fps=target_fps,
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
    """Decode video chunks using decord batch reader.

    WHY decord.VideoReader + get_batch: single C++ call decodes N frames at once
    with zero-copy numpy output, vs imageio which iterates one Python frame at a time.
    """
    # WHY cpu(0): GPU decord context would compete with VAE for VRAM
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    container_fps = float(vr.get_avg_fps())
    total_native_frames = len(vr)

    resolved_height = int(resolved_size.height)
    resolved_width = int(resolved_size.width)

    # WHY compute frame offset from timestamp: match imageio's timestamp-based seek
    frame_offset = 0
    if stream.from_timestamp is not None and container_fps > 0:
        frame_offset = 0
        for i in range(total_native_frames):
            ts = float(i) / container_fps
            if ts >= stream.from_timestamp - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS:
                frame_offset = i
                break

    end_frame_limit = total_native_frames
    if stream.to_timestamp is not None and container_fps > 0:
        for i in range(frame_offset, total_native_frames):
            ts = float(i) / container_fps
            if ts >= stream.to_timestamp:
                end_frame_limit = i
                break

    for chunk_start, chunk_end, native_start, native_end in chunk_specs:
        abs_start = frame_offset + native_start
        abs_end = min(frame_offset + native_end, end_frame_limit)
        if abs_end <= abs_start:
            raise ValueError(
                f"Decoded stream shorter than manifest for source={stream.source_id}, "
                f"episode={stream.episode_index}, chunk=[{chunk_start},{chunk_end})."
            )
        # WHY get_batch: one C++ call decodes all needed frames, no Python loop
        indices = list(range(abs_start, abs_end))
        raw_frames = vr.get_batch(indices).asnumpy()  # [N, H, W, 3] uint8

        if stream.height is None or stream.width is None:
            resolved = resolve_mixed_video_decode_size(
                data_config,
                source_height=int(raw_frames.shape[1]),
                source_width=int(raw_frames.shape[2]),
            )
            resolved_height = int(resolved.height)
            resolved_width = int(resolved.width)

        # WHY _batch_resize_frames: processes all N frames in one torch kernel call
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
    """Original imageio decode path, kept as fallback for codec edge cases."""
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
        for chunk_index, (chunk_start, chunk_end, native_start, native_end) in enumerate(chunk_specs):
            while selected_index < native_end and not reader_exhausted:
                try:
                    frame_index, frame = next(reader_iter)
                except StopIteration:
                    reader_exhausted = True
                    break
                if fps > 0.0:
                    timestamp = frame_index / fps
                    # WHY epsilon: LeRobot v3 packed-bundle stores from_timestamp as float64
                    # which may round up from true frame timestamp, causing strict < to
                    # exclude the boundary frame on WAN-aligned (1+4k) episodes.
                    if (
                        stream.from_timestamp is not None
                        and timestamp < stream.from_timestamp - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
                    ):
                        continue
                    if stream.to_timestamp is not None and timestamp >= stream.to_timestamp:
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
            missing = [index for index in range(native_start, native_end) if index not in frames_by_native_index]
            if missing:
                raise ValueError(
                    f"Decoded stream is shorter than manifest metadata for source={stream.source_id}, "
                    f"episode={stream.episode_index}, stream={stream.stream_key}; missing native frames "
                    f"{missing[:5]} for normalized chunk=[{chunk_start}, {chunk_end})."
                )
            native_frames = torch.stack(
                [frames_by_native_index[index] for index in range(native_start, native_end)],
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
        raise ValueError(f"Invalid target frame chunk [{chunk_start}, {chunk_end}).")
    first_position = float(chunk_start) * float(source_fps) / float(target_fps)
    last_position = float(chunk_end - 1) * float(source_fps) / float(target_fps)
    native_start = max(0, min(int(native_length_frames) - 1, int(math.floor(first_position))))
    native_end = max(native_start + 1, min(int(native_length_frames), int(math.ceil(last_position)) + 1))
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
        raise ValueError(f"Expected video frames with leading time dimension, got shape {tuple(frames.shape)}.")
    if target_frame_count <= 0:
        return frames[:0]
    if target_fps is None:
        start = int(target_start_index) - int(native_start_index)
        end = start + int(target_frame_count)
        return frames[start:end]
    if float(source_fps) <= 0 or float(target_fps) <= 0:
        raise ValueError(f"FPS values must be positive, got source={source_fps}, target={target_fps}.")
    if frames.shape[0] == 0:
        raise ValueError("Cannot resample an empty video frame tensor.")
    device = frames.device
    positions = (
        torch.arange(int(target_frame_count), dtype=torch.float32, device=device) + float(target_start_index)
    ) * (float(source_fps) / float(target_fps))
    positions = positions.clamp(min=0.0, max=max(0.0, float(native_total_frames - 1)))
    local_positions = positions - float(native_start_index)
    low = torch.floor(local_positions).to(dtype=torch.long).clamp(min=0, max=frames.shape[0] - 1)
    high = (low + 1).clamp(max=frames.shape[0] - 1)
    alpha = (local_positions - low.to(dtype=torch.float32)).clamp(min=0.0, max=1.0)
    while alpha.ndim < frames.ndim:
        alpha = alpha.unsqueeze(-1)
    source_dtype = frames.dtype
    interpolated = frames[low].to(dtype=torch.float32) * (1.0 - alpha) + frames[high].to(dtype=torch.float32) * alpha
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
    """Resize a batch of frames [N,H,W,3] using torch batch interpolation.

    WHY torch.interpolate instead of per-frame PIL: one kernel call processes
    all N frames in parallel, eliminating N Python→C++ round-trips. Measured
    1.5-2x faster on typical 30-60 frame chunks.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [N,H,W,3] uint8 frames, got {frames.shape}.")
    n, h, w, _ = frames.shape
    if n == 0:
        return frames
    resolved_fit_mode = _resolve_frame_fit_mode(fit_mode, center_crop=center_crop)

    if resolved_fit_mode == MixedVideoFrameFitMode.CENTER_CROP:
        # WHY crop first then resize: matches original per-frame center_crop logic
        target_aspect = target_width / target_height
        current_aspect = w / h
        if abs(current_aspect - target_aspect) > 1e-6:
            if current_aspect > target_aspect:
                crop_w = max(1, int(round(h * target_aspect)))
                left = max(0, (w - crop_w) // 2)
                frames = frames[:, :, left:left + crop_w, :]
            else:
                crop_h = max(1, int(round(w / target_aspect)))
                top = max(0, (h - crop_h) // 2)
                frames = frames[:, top:top + crop_h, :, :]
        n, h, w, _ = frames.shape
        if not allow_upscale and (h < target_height or w < target_width):
            return frames
        if h == target_height and w == target_width:
            return frames
        # WHY permute to NCHW: F.interpolate expects channel-first layout
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        t = F.interpolate(t, size=(target_height, target_width), mode="bilinear", align_corners=False, antialias=True)
        return t.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    if resolved_fit_mode == MixedVideoFrameFitMode.LETTERBOX_PAD:
        if h <= 0 or w <= 0 or target_height <= 0 or target_width <= 0:
            raise ValueError(
                f"Letterbox requires positive dims, got input=({h},{w}) target=({target_height},{target_width})."
            )
        scale = min(float(target_width) / float(w), float(target_height) / float(h))
        if not allow_upscale:
            scale = min(scale, 1.0)
        resized_h = max(1, min(target_height, int(round(float(h) * scale))))
        resized_w = max(1, min(target_width, int(round(float(w) * scale))))
        if resized_h == h and resized_w == w:
            resized = frames
        else:
            t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            t = F.interpolate(t, size=(resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)
            resized = t.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        # WHY np.zeros canvas: letterbox pads to exact target size, matching original PIL path
        canvas = np.zeros((n, target_height, target_width, 3), dtype=np.uint8)
        top_pad = max(0, (target_height - resized_h) // 2)
        left_pad = max(0, (target_width - resized_w) // 2)
        canvas[:, top_pad:top_pad + resized_h, left_pad:left_pad + resized_w] = resized[..., :3]
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
    resolved_fit_mode = _resolve_frame_fit_mode(fit_mode, center_crop=center_crop)
    if resolved_fit_mode == MixedVideoFrameFitMode.CENTER_CROP:
        array = _center_crop_to_aspect(array, target_height=target_height, target_width=target_width)
        return _resize_frame(array, target_height=target_height, target_width=target_width, allow_upscale=allow_upscale)
    if resolved_fit_mode == MixedVideoFrameFitMode.LETTERBOX_PAD:
        return _letterbox_pad_to_target(
            array,
            target_height=target_height,
            target_width=target_width,
            allow_upscale=allow_upscale,
        )
    raise ValueError(f"Unsupported mixed-video frame fit mode: {resolved_fit_mode}")


def _resolve_frame_fit_mode(
    fit_mode: MixedVideoFrameFitMode | str | None,
    *,
    center_crop: bool,
) -> MixedVideoFrameFitMode:
    if fit_mode is not None:
        return fit_mode if isinstance(fit_mode, MixedVideoFrameFitMode) else MixedVideoFrameFitMode(str(fit_mode))
    return MixedVideoFrameFitMode.CENTER_CROP if center_crop else MixedVideoFrameFitMode.LETTERBOX_PAD


def _resize_frame(
    array: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    allow_upscale: bool,
) -> np.ndarray:
    input_height, input_width = int(array.shape[0]), int(array.shape[1])
    if not allow_upscale and (input_height < target_height or input_width < target_width):
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
    if input_height <= 0 or input_width <= 0 or target_height <= 0 or target_width <= 0:
        raise ValueError(
            "Mixed-video letterbox resize expects positive dimensions, "
            f"got input=({input_height}, {input_width}) target=({target_height}, {target_width})."
        )
    scale = min(float(target_width) / float(input_width), float(target_height) / float(input_height))
    if not allow_upscale:
        scale = min(scale, 1.0)
    resized_height = max(1, min(int(target_height), int(round(float(input_height) * scale))))
    resized_width = max(1, min(int(target_width), int(round(float(input_width) * scale))))
    if resized_height == input_height and resized_width == input_width:
        resized = array
    else:
        image = Image.fromarray(array)
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        resized = np.asarray(image.resize((resized_width, resized_height), resampling), dtype=np.uint8)
    canvas = np.zeros((int(target_height), int(target_width), 3), dtype=np.uint8)
    top = max(0, (int(target_height) - int(resized_height)) // 2)
    left = max(0, (int(target_width) - int(resized_width)) // 2)
    canvas[top : top + resized_height, left : left + resized_width] = resized[..., :3]
    return canvas


def _load_source_streams(
    source: MixedVideoSourceConfig,
    data_config: MixedVideoDataConfig,
) -> list[MixedVideoStreamRecord]:
    manifest_path = _resolve_manifest_path(source)
    rows = _read_manifest_csv(manifest_path)
    if not rows:
        raise ValueError(f"Mixed-video manifest is empty: {manifest_path}")
    streams: list[MixedVideoStreamRecord] = []
    for row in rows:
        stream_key = _string_field(row, "stream_key") or _string_field(row, "video_key")
        stream_index = _int_field(row, "stream_index", default=0)
        if not stream_key:
            stream_key = f"stream_{stream_index}"
        if (
            source.include_streams
            and stream_key not in source.include_streams
            and str(stream_index) not in source.include_streams
        ):
            continue
        target_slot = _target_slot_for_stream(row, source, data_config, stream_key, stream_index)
        configured_slots = set(data_config.camera_names) | set(data_config.latent_camera_names)
        for combination in data_config.latent_view_combinations:
            if combination.enabled:
                configured_slots.update(combination.slots)
        if target_slot not in configured_slots:
            continue
        length_frames = _int_field(row, "length_frames", default=0)
        if length_frames <= 0:
            length_frames = _int_field(row, "num_frames", default=0)
        if length_frames <= 0:
            raise ValueError(
                f"Mixed-video manifest row must include positive length_frames: "
                f"{manifest_path}, source={source.source_id}, stream={stream_key}."
            )
        local_path = _local_video_path(row, source, manifest_path)
        latent_path = _local_latent_path(row, source, manifest_path)
        shard_relative_path = _string_field(row, "shard_relative_path")
        latent_shard_relative_path = (
            _string_field(row, "latent_shard_relative_path")
            or _string_field(row, "video_latents_shard_relative_path")
        )
        latent_length_frames = (
            _optional_int_field(row, "latent_length_frames")
            or _optional_int_field(row, "video_latent_frames")
        )
        if latent_length_frames is None and source.source_format == MixedVideoSourceFormat.LATENT:
            latent_length_frames = length_frames
        repo_id = _string_field(row, "repo_id") or source.repo_id
        dataset_id = _string_field(row, "dataset_id") or repo_id or source.source_id
        episode_index = _int_field(row, "episode_index", default=0)
        clip_id = _string_field(row, "clip_id") or _string_field(row, "clip_key") or "default"
        observation_fps = _float_field(row, "observation_fps")
        container_fps = None
        source_has_rgb = source.source_format in {
            MixedVideoSourceFormat.RGB,
            MixedVideoSourceFormat.RGB_AND_LATENT,
        }
        if observation_fps is None and source_has_rgb:
            container_fps = _probe_video_observation_fps(local_path)
        if (
            observation_fps is None
            and container_fps is None
            and source_has_rgb
            and data_config.target_observation_fps is not None
            and local_path is None
            and repo_id is not None
            and shard_relative_path is not None
        ):
            raise ValueError(
                "Remote mixed-video RGB rows must include `observation_fps` when "
                "`target_observation_fps` is enabled. Container FPS probing is only performed for local files; "
                "either add manifest observation_fps, disable FPS normalization, or materialize the video locally. "
                f"manifest={manifest_path}, source={source.source_id}, dataset_id={dataset_id}, "
                f"episode_index={episode_index}, clip_id={clip_id}, stream_key={stream_key}, "
                f"shard_relative_path={shard_relative_path!r}."
            )
        resolved_fps = resolve_video_source_fps(
            observation_fps,
            container_fps=container_fps,
            missing_observation_fps=data_config.missing_observation_fps,
        )
        normalized_length = _timeline_normalized_video_frame_count(
            length_frames,
            source_fps=resolved_fps.value,
            target_fps=data_config.target_observation_fps,
        )
        clip = ResolvedVideoClip(
            clip_id=clip_id,
            source_id=source.source_id,
            dataset_id=dataset_id,
            episode_index=episode_index,
            stream_key=stream_key,
            target_slot=target_slot,
            path_key=_stream_path_key(
                local_path=local_path,
                latent_path=latent_path,
                repo_id=repo_id,
                shard_relative_path=shard_relative_path,
                latent_shard_relative_path=latent_shard_relative_path,
            ),
            native_length_frames=length_frames,
            source_fps=resolved_fps.value,
            source_fps_source=resolved_fps.source,
            target_fps=data_config.target_observation_fps,
            normalized_length_frames=normalized_length,
            from_timestamp=_float_field(row, "from_timestamp"),
            to_timestamp=_float_field(row, "to_timestamp"),
            width=_optional_int_field(row, "width"),
            height=_optional_int_field(row, "height"),
        )
        streams.append(
            MixedVideoStreamRecord(
                source_id=source.source_id,
                source_group=_string_field(row, "source_group") or source.source_group,
                repo_id=repo_id,
                dataset_id=dataset_id,
                episode_index=episode_index,
                clip_id=clip_id,
                stream_index=stream_index,
                stream_key=stream_key,
                target_slot=target_slot,
                source_format=source.source_format,
                manifest_path=manifest_path,
                local_path=local_path,
                latent_path=latent_path,
                shard_relative_path=shard_relative_path,
                latent_shard_relative_path=latent_shard_relative_path,
                latent_key=_string_field(row, "latent_key") or source.latent_key,
                length_frames=length_frames,
                latent_length_frames=latent_length_frames,
                observation_fps=resolved_fps.value,
                action_fps=_float_field(row, "action_fps"),
                from_timestamp=clip.from_timestamp,
                to_timestamp=clip.to_timestamp,
                width=clip.width,
                height=clip.height,
                channels=_optional_int_field(row, "channels"),
                tasks=_parse_tasks(row),
                clip=clip,
            )
        )
    return streams


def _resolve_manifest_path(source: MixedVideoSourceConfig) -> Path:
    manifest = Path(source.manifest_csv).expanduser()
    if manifest.exists():
        return manifest
    if source.local_root is not None:
        candidate = Path(source.local_root).expanduser() / source.manifest_csv
        if candidate.exists():
            return candidate
    return manifest


def _read_manifest_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing mixed-video manifest CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _probe_video_observation_fps(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps", 0.0) or 0.0)
    finally:
        reader.close()
    return fps if fps > 0.0 else None


def _stream_path_key(
    *,
    local_path: Path | None,
    latent_path: Path | None,
    repo_id: str | None,
    shard_relative_path: str | None,
    latent_shard_relative_path: str | None,
) -> str:
    if local_path is not None:
        return str(local_path)
    if latent_path is not None:
        return str(latent_path)
    if repo_id is not None and shard_relative_path is not None:
        return f"{repo_id}:{shard_relative_path}"
    if repo_id is not None and latent_shard_relative_path is not None:
        return f"{repo_id}:{latent_shard_relative_path}"
    return "<unresolved>"


def _target_slot_for_stream(
    row: dict[str, str],
    source: MixedVideoSourceConfig,
    data_config: MixedVideoDataConfig,
    stream_key: str,
    stream_index: int,
) -> str:
    source_mapping = {mapping.source_name: mapping.target_slot for mapping in source.channel_mappings}
    if stream_key in source_mapping:
        return source_mapping[stream_key]
    row_target = _string_field(row, "target_slot") or _string_field(row, "target_slot_key")
    if row_target:
        return row_target
    default_slots = (
        data_config.latent_camera_names
        if source.source_format == MixedVideoSourceFormat.LATENT and data_config.latent_camera_names
        else data_config.camera_names
    )
    if stream_index < len(default_slots):
        return default_slots[stream_index]
    return stream_key


def _local_video_path(
    row: dict[str, str],
    source: MixedVideoSourceConfig,
    manifest_path: Path,
) -> Path | None:
    raw_path = _string_field(row, "local_path") or _string_field(row, "video_path")
    if raw_path is None and source.local_root is not None:
        raw_path = _string_field(row, "shard_relative_path")
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if source.local_root is not None:
        candidates.append(Path(source.local_root).expanduser() / path)
    candidates.append(manifest_path.parent / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _local_latent_path(
    row: dict[str, str],
    source: MixedVideoSourceConfig,
    manifest_path: Path,
) -> Path | None:
    raw_path = (
        _string_field(row, "latent_path")
        or _string_field(row, "video_latents_path")
        or _string_field(row, "latent_local_path")
    )
    if raw_path is None and source.latent_root is not None:
        raw_path = _string_field(row, "latent_shard_relative_path") or _string_field(
            row, "video_latents_shard_relative_path"
        )
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if source.latent_root is not None:
        candidates.append(Path(source.latent_root).expanduser() / path)
    candidates.append(manifest_path.parent / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_stream_path(stream: MixedVideoStreamRecord, *, cache_dir: str | None) -> Path:
    if stream.local_path is not None:
        if not stream.local_path.exists():
            raise FileNotFoundError(
                f"Missing mixed-video file for source={stream.source_id}, "
                f"episode={stream.episode_index}, stream={stream.stream_key}: {stream.local_path}"
            )
        return stream.local_path
    if stream.repo_id is None or stream.shard_relative_path is None:
        raise FileNotFoundError(
            f"Mixed-video stream has neither local_path nor HF repo/shard path: "
            f"source={stream.source_id}, episode={stream.episode_index}, stream={stream.stream_key}."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency exists in normal training envs.
        raise ImportError("huggingface_hub is required for remote mixed-video manifests.") from exc
    return Path(
        hf_hub_download(
            repo_id=stream.repo_id,
            filename=stream.shard_relative_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )


def _resolve_stream_latent_path(stream: MixedVideoStreamRecord, *, cache_dir: str | None) -> Path:
    if stream.latent_path is not None:
        if not stream.latent_path.exists():
            raise FileNotFoundError(
                f"Missing mixed-video latent file for source={stream.source_id}, "
                f"episode={stream.episode_index}, stream={stream.stream_key}: {stream.latent_path}"
            )
        return stream.latent_path
    if stream.repo_id is None or stream.latent_shard_relative_path is None:
        raise FileNotFoundError(
            f"Mixed-video stream has neither latent_path nor HF latent shard path: "
            f"source={stream.source_id}, episode={stream.episode_index}, stream={stream.stream_key}."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency exists in normal training envs.
        raise ImportError("huggingface_hub is required for remote mixed-video latent manifests.") from exc
    return Path(
        hf_hub_download(
            repo_id=stream.repo_id,
            filename=stream.latent_shard_relative_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    )


def _load_latent_tensor(path: Path, *, key: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        tensor = payload
    elif isinstance(payload, dict) and key in payload:
        tensor = payload[key]
    else:
        raise ValueError(f"Expected latent tensor or key {key!r} in latent payload at {path}.")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(f"Expected latent tensor [C,T,H,W] at {path}, got {type(tensor)!r}.")
    return tensor.to(dtype=torch.float32).contiguous()


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
            f"Mixed-video source dimensions must be positive, got height={source_height}, width={source_width}."
        )
    if not bins:
        raise ValueError("At least one mixed-video resize bin is required.")
    source_ratio = float(source_width) / float(source_height)
    ranked = sorted(
        bins,
        key=lambda bin_config: (
            abs(math.log(source_ratio / bin_config.aspect_ratio)),
            float("inf") if bin_config.max_pixels is None else float(bin_config.max_pixels),
        ),
    )
    best_distance = abs(math.log(source_ratio / ranked[0].aspect_ratio))
    aspect_candidates = [
        bin_config
        for bin_config in ranked
        if abs(math.log(source_ratio / bin_config.aspect_ratio)) <= best_distance + 1e-6
    ]
    source_pixels = int(source_height) * int(source_width)
    for bin_config in aspect_candidates:
        if bin_config.max_pixels is None or source_pixels <= int(bin_config.max_pixels):
            return bin_config
    return aspect_candidates[-1]


def _select_valid_causal_bucket(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    observation_start: int,
    *,
    episode_length: int,
) -> CausalPrefixSuffixBucketConfig | None:
    buckets = data_config.sample_construction.effective_causal_prefix_suffix_buckets
    valid_buckets: list[CausalPrefixSuffixBucketConfig] = []
    for bucket in buckets:
        total_frames = _causal_bucket_total_frames(data_config, bucket)
        required_span = (total_frames - 1) * int(data_config.frame_stride) + 1
        if int(observation_start) + required_span <= int(episode_length):
            valid_buckets.append(bucket)
    if not valid_buckets:
        return None
    token = f"{data_config.sampling_seed}:{episode.key}:{observation_start}".encode("utf-8")
    bucket_index = int(hashlib.sha256(token).hexdigest()[:16], 16) % len(valid_buckets)
    return valid_buckets[bucket_index]


def _causal_bucket_total_frames(
    data_config: MixedVideoDataConfig,
    bucket: CausalPrefixSuffixBucketConfig,
) -> int:
    total_frames = int(bucket.observed_frames) + int(bucket.future_frames)
    if total_frames <= 0:
        raise ValueError("Mixed-video causal prefix/suffix buckets must request at least one frame.")
    if total_frames > int(data_config.num_frames):
        raise ValueError(
            f"Mixed-video causal bucket requests {total_frames} frames, "
            f"but data.num_frames={data_config.num_frames}."
        )
    return total_frames


def _source_target_counts(
    data_config: MixedVideoDataConfig,
    source_counts: dict[str, int],
) -> dict[str, int]:
    if data_config.weight_mode == MixedVideoWeightMode.PROPORTIONAL_TO_SIZE:
        return dict(source_counts)
    manual_weights = {
        source.source_id: source.sampling_weight
        for source in data_config.video_sources
        if source.enabled and source.sampling_weight is not None
    }
    if data_config.weight_mode == MixedVideoWeightMode.MANUAL_OVERRIDE:
        if set(manual_weights) != set(source_counts):
            missing = sorted(set(source_counts) - set(manual_weights))
            raise ValueError(f"manual_override mixed-video weighting needs sampling_weight for: {missing}")
        total = sum(source_counts.values())
        weight_sum = sum(float(value) for value in manual_weights.values())
        return {
            source_id: max(1, int(round(total * float(manual_weights[source_id]) / weight_sum)))
            for source_id in source_counts
        }
    scaled = {}
    for source_id, count in source_counts.items():
        scale = float(manual_weights.get(source_id, 1.0))
        scaled[source_id] = max(1, int(round(count * scale)))
    return scaled


def _weighted_source_cycle(target_counts: dict[str, int]) -> tuple[str, ...]:
    remaining = dict(target_counts)
    total = sum(remaining.values())
    order: list[str] = []
    while len(order) < total:
        source_id = max(
            (source for source, count in remaining.items() if count > 0),
            key=lambda source: remaining[source] / max(1, target_counts[source]),
        )
        order.append(source_id)
        remaining[source_id] -= 1
    return tuple(order)


def _repeat_or_trim(values: Sequence[int], target_count: int) -> list[int]:
    if target_count <= len(values):
        return list(values[:target_count])
    repeats = (target_count + len(values) - 1) // len(values)
    return list((list(values) * repeats)[:target_count])


def _video_stream_cache_key(stream: MixedVideoStreamRecord, data_config: MixedVideoDataConfig) -> str:
    path_key = str(stream.local_path) if stream.local_path is not None else f"{stream.repo_id}:{stream.shard_relative_path}"
    signature = {
        "path": path_key,
        "stream_key": stream.stream_key,
        "target_slot": stream.target_slot,
        "length_frames": int(stream.length_frames),
        "observation_fps": stream.observation_fps,
        "from_timestamp": stream.from_timestamp,
        "to_timestamp": stream.to_timestamp,
        "decode_size_mode": data_config.decode_size_mode.value,
        "decode_fit_mode": data_config.decode_fit_mode.value,
        "decode_allow_upscale": bool(data_config.decode_allow_upscale),
        "decode_height": int(data_config.decode_height),
        "decode_width": int(data_config.decode_width),
        "decode_resize_bins": tuple(
            (
                bin_config.name,
                int(bin_config.aspect_width),
                int(bin_config.aspect_height),
                int(bin_config.target_height),
                int(bin_config.target_width),
                bin_config.max_pixels,
            )
            for bin_config in data_config.decode_resize_bins
        ),
        "target_observation_fps": data_config.target_observation_fps,
        "missing_observation_fps": float(data_config.missing_observation_fps),
    }
    payload = repr(sorted(signature.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _latent_stream_cache_key(stream: MixedVideoStreamRecord) -> str:
    if stream.latent_path is not None:
        return str(stream.latent_path)
    return f"{stream.repo_id}:{stream.latent_shard_relative_path}"


def _merge_tasks(task_groups: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    merged: list[str] = []
    for tasks in task_groups:
        for task in tasks:
            if task and task not in merged:
                merged.append(task)
    return tuple(merged)


def _parse_tasks(row: dict[str, str]) -> tuple[str, ...]:
    raw = _string_field(row, "tasks") or _string_field(row, "task") or _string_field(row, "language")
    if raw is None:
        return ()
    cleaned = raw.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    tasks = [item.strip().strip("'\"") for item in cleaned.replace("|", ",").replace(";", ",").split(",")]
    return tuple(item for item in tasks if item)


def _string_field(row: dict[str, str], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _int_field(row: dict[str, str], key: str, *, default: int) -> int:
    value = _string_field(row, key)
    if value is None:
        return default
    return int(float(value))


def _optional_int_field(row: dict[str, str], key: str) -> int | None:
    value = _string_field(row, key)
    if value is None:
        return None
    return int(float(value))


def _float_field(row: dict[str, str], key: str) -> float | None:
    value = _string_field(row, key)
    if value is None:
        return None
    return float(value)
