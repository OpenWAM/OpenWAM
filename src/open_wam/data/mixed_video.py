from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import Dataset

from open_wam.configs import (
    CausalPrefixSuffixBucketConfig,
    DataConfig,
    MixedVideoDataConfig,
    MixedVideoMissingStreamPolicy,
    MixedVideoRandomMode,
    MixedVideoSourceFormat,
    MixedVideoViewCombinationConfig,
    MixedVideoWeightMode,
)

from .contracts import WAMSample
from .distributed_sampling import EpochOrderDistributedSampler
from .latent_contracts import LatentWAMSample
from .latent_view_assembly import assemble_mixed_video_latent_views
from .mixed_video_catalog import (
    MixedVideoCatalog,
    MixedVideoEpisodeRecord,
    MixedVideoStreamRecord,
    load_mixed_video_catalog,
    split_mixed_video_episodes,
)
from .mixed_video_decode import (
    MixedVideoResolvedDecodeSize as MixedVideoResolvedDecodeSize,
    decode_mixed_video_stream_frame_chunk as decode_mixed_video_stream_frame_chunk,
    decode_mixed_video_stream_frames,
    decode_video_frames as decode_video_frames,
    iter_mixed_video_stream_frame_chunks as iter_mixed_video_stream_frame_chunks,
    normalized_video_frame_count as normalized_video_frame_count,
    resample_video_frames_to_fps as resample_video_frames_to_fps,
    resolve_mixed_video_decode_size as resolve_mixed_video_decode_size,
    resolve_mixed_video_observation_fps as resolve_mixed_video_observation_fps,
    transform_frame as transform_frame,
)


_MIXED_VIDEO_DECODE_COMPATIBILITY_EXPORTS = (
    MixedVideoResolvedDecodeSize,
    decode_mixed_video_stream_frame_chunk,
    decode_video_frames,
    iter_mixed_video_stream_frame_chunks,
    normalized_video_frame_count,
    resample_video_frames_to_fps,
    resolve_mixed_video_observation_fps,
    transform_frame,
)


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


class MixedVideoTrainSampler(EpochOrderDistributedSampler):
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
        resolved_world_size = max(1, int(world_size))
        resolved_rank = int(rank)
        if resolved_rank < 0 or resolved_rank >= resolved_world_size:
            raise ValueError(f"Invalid sampler rank={rank} for world_size={world_size}.")
        super().__init__(
            dataset,
            world_size=resolved_world_size,
            rank=resolved_rank,
            empty_dataset_message=None,
            empty_order_message=None,
            cache_order=True,
            geometry_from_order=True,
        )


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
                    stream.target_slot: float(stream.clip.source_fps)
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

        decoded = decode_mixed_video_stream_frames(
            self.data_config,
            stream,
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
