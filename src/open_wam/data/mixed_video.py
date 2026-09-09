from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Callable, Sequence
import hashlib
from typing import Any

import torch
from torch.utils.data import Dataset

from open_wam.configs import (
    BatchingMode,
    DataConfig,
    MixedVideoDataConfig,
    MixedVideoMissingStreamPolicy,
    MixedVideoSourceFormat,
)

from .contracts import WAMSample
from .distributed_sampling import EpochOrderDistributedSampler
from .latent_contracts import LatentWAMSample
from .latent_view_assembly import assemble_mixed_video_latent_views
from .mixed_video_catalog_assembly import (
    load_mixed_video_catalog as load_mixed_video_catalog,
)
from .mixed_video_catalog_contracts import (
    MixedVideoCatalog,
    MixedVideoEpisodeRecord,
    MixedVideoStreamRecord,
)
from .mixed_video_catalog_split import (
    split_mixed_video_episodes as split_mixed_video_episodes,
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
from .mixed_video_latent_storage import MixedVideoLatentRepository
from .mixed_video_planning import (
    MixedVideoWindowPlanner,
    MixedVideoWindowRecord as MixedVideoWindowRecord,
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
        dataset.configure_batch_geometry(world_size=resolved_world_size, batch_size=int(dataset.data_config.train_batch_size))
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
        self._window_planner = MixedVideoWindowPlanner(data_config)
        self._validate_source_formats()
        self.sample_index = self._build_sample_index()
        self._video_frame_cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()
        if not self.sample_index:
            raise ValueError(
                f"No valid mixed-video windows were constructed for split='{split}'. "
                f"Check num_frames={data_config.num_frames}, frame_stride={data_config.frame_stride}, "
                f"sample_stride={data_config.sample_stride}, and selected episodes={len(episode_keys)}."
            )

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

    def configure_batch_geometry(self, *, world_size: int, batch_size: int) -> None:
        """Record the loader geometry the epoch order must be aligned to."""

        self._batch_geometry = (max(1, int(world_size)), max(1, int(batch_size)))


    def build_epoch_index_order(self, *, epoch: int = 0) -> tuple[int, ...]:
        world_size, batch_size = getattr(self, "_batch_geometry", (1, 1))
        if batch_size > 1 and bool(getattr(self.data_config, "shape_bucketed_batching", False)):
            return self._window_planner.build_shape_bucketed_epoch_order(
                sample_index=self.sample_index,
                episode_records=self.episode_records,
                shape_ids=self.window_shape_ids(),
                world_size=world_size,
                batch_size=batch_size,
                epoch=epoch,
            )
        return self._window_planner.build_source_balanced_epoch_order(
            sample_index=self.sample_index,
            episode_records=self.episode_records,
            epoch=epoch,
        )


    def window_shape_ids(self) -> Sequence[int]:
        """Return one small integer per planned window identifying its batch shape."""

        raise ValueError(
            "`shape_bucketed_batching` is implemented for the latent mixed-video dataset only "
            "(trainer.batch_adapter=latents). The RGB path resolves its decoded size lazily per "
            "sample, so per-window shapes are not known at ordering time."
        )


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
        return self._window_planner.build_episode_windows(
            episode_records=self.episode_records,
            episode_keys=self.episode_keys,
        )


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
        self._latent_repository = MixedVideoLatentRepository(data_config)

    def _allowed_source_formats(self) -> frozenset[MixedVideoSourceFormat]:
        return frozenset({MixedVideoSourceFormat.LATENT, MixedVideoSourceFormat.RGB_AND_LATENT})

    def batching_length_hint(self, index: int) -> int:
        return int(self.sample_index[index].valid_video_frames)

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
        return self._window_planner.build_latent_view_windows(
            episode_records=self.episode_records,
            episode_keys=self.episode_keys,
        )

    def window_shape_ids(self) -> Sequence[int]:
        """Map every planned window to a small id identifying its collated shape.

        Two windows may share a batch exactly when `torch.stack` accepts their
        `video_latents` together, i.e. when the assembled canvas has the same
        (H, W). `assemble_latent_views` builds that canvas as a pure function of
        the per-view grid, the number of selected views and the canvas view
        count -- it requires all selected views to be the same resolution and
        raises otherwise -- so those three values are a complete shape key. The
        channel and frame axes need no key: C is fixed by the encoder and T is
        padded by the collator in variable modes and by the dataset in strict mode.

        The grid comes from the manifest's `height`/`width` columns, which for
        latent sources record the LATENT grid (8x8, 16x16, 12x22, 16x22), so no
        sidecar has to be opened to plan an epoch.
        """

        cached = getattr(self, "_window_shape_id_cache", None)
        if cached is not None:
            return cached

        canvas_view_count = _latent_view_assembly_canvas_view_count(self.data_config)
        shape_key_to_id: dict[tuple[int, int, int, int], int] = {}
        episode_slot_grid: dict[tuple[str, str], tuple[int, int]] = {}
        shape_ids = array("i", bytes(4 * len(self.sample_index)))

        for window_position, window in enumerate(self.sample_index):
            episode = self.episode_records[window.episode_key]
            slots = tuple(str(slot) for slot in window.view_combination_slots)
            if not slots:
                valid = self._window_planner.valid_latent_view_combinations(episode)
                if not valid:
                    raise KeyError(
                        f"Mixed-video latent episode {episode.key!r} has no valid latent view combinations."
                    )
                slots = tuple(str(slot) for slot in valid[0].slots)
            grid_key = (window.episode_key, slots[0])
            grid = episode_slot_grid.get(grid_key)
            if grid is None:
                streams_by_slot: dict[str, MixedVideoStreamRecord] = {}
                for stream in sorted(episode.streams, key=lambda item: item.stream_index):
                    streams_by_slot.setdefault(stream.target_slot, stream)
                stream = streams_by_slot.get(slots[0])
                if stream is None:
                    raise KeyError(
                        f"Mixed-video latent episode {episode.key!r} is missing configured stream slot {slots[0]!r}."
                    )
                if stream.height is None or stream.width is None:
                    raise ValueError(
                        f"Shape-bucketed batching needs the latent grid of source={stream.source_id!r}, "
                        f"stream={stream.stream_key!r}, but its manifest row has no height/width. "
                        "Rebuild that manifest with latent grid columns, or set "
                        "`shape_bucketed_batching: false` and train_batch_size: 1."
                    )
                grid = (int(stream.height), int(stream.width))
                episode_slot_grid[grid_key] = grid
            shape_key = (grid[0], grid[1], len(slots), canvas_view_count)
            shape_id = shape_key_to_id.get(shape_key)
            if shape_id is None:
                shape_id = len(shape_key_to_id)
                shape_key_to_id[shape_key] = shape_id
            shape_ids[window_position] = shape_id

        self._window_shape_id_cache = shape_ids
        self._window_shape_key_to_id = dict(shape_key_to_id)
        return shape_ids


    @property
    def batching_shape_hint(self) -> Callable[[int], int] | None:
        """Only opt-in spatial grouping requires manifest grid metadata."""
        if not self.data_config.shape_bucketed_batching:
            return None
        return self.window_shape_ids().__getitem__

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
                "latent_layout": assembly_metadata,
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
            valid_combinations = (
                self._window_planner.valid_latent_view_combinations(
                    episode,
                )
            )
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
            latents = self._latent_repository.load(stream)
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
        if self.data_config.batching.mode is not BatchingMode.STRICT:
            return latents.contiguous()
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

def _latent_view_assembly_canvas_view_count(data_config: MixedVideoDataConfig) -> int:
    enabled = [combo for combo in data_config.latent_view_combinations if combo.enabled]
    if enabled:
        return max(len(combo.slots) for combo in enabled)
    return max(1, min(4, len(data_config.latent_camera_names)))


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
