from __future__ import annotations

import csv
from collections import Counter
from concurrent.futures import Future
from dataclasses import replace as _dataclass_replace
import importlib.util
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

import open_wam.data.mixed_video_decode as mixed_video_decode_module
from open_wam.configs import (
    ActionSchemaConfig,
    BatchAdapterName,
    CausalPrefixSuffixBucketConfig,
    ConsortiumChannelMappingConfig,
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoFrameFitMode,
    MixedVideoLatentEncodingMode,
    MixedVideoRandomMode,
    MixedVideoSourceFormat,
    MixedVideoSourceConfig,
    MixedVideoViewCombinationConfig,
    MixedVideoWeightMode,
    SampleConstructionConfig,
    ViewLayoutConfig,
    WindowSamplingMode,
)
from open_wam.configs import load_experiment_config
from open_wam.data import (
    MixedVideoCatalog as PublicMixedVideoCatalog,
    MixedVideoWindowPlanner as PublicMixedVideoWindowPlanner,
    MixedVideoWindowRecord as PublicMixedVideoWindowRecord,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    collate_wam_samples,
    decode_video_frames as public_decode_video_frames,
    transform_frame as public_transform_frame,
)
from open_wam.data.raw_video import build_canonical_video_preprocessor
from open_wam.data.mixed_video import (
    MixedVideoCatalog as LegacyMixedVideoCatalog,
    MixedVideoEpisodeRecord as LegacyMixedVideoEpisodeRecord,
    MixedVideoLatentWindowDataset,
    MixedVideoResolvedDecodeSize as LegacyMixedVideoResolvedDecodeSize,
    MixedVideoStreamRecord as LegacyMixedVideoStreamRecord,
    MixedVideoWindowDataset,
    MixedVideoWindowRecord as LegacyMixedVideoWindowRecord,
    assemble_mixed_video_latent_views,
    decode_mixed_video_stream_frame_chunk as legacy_decode_stream_frame_chunk,
    decode_video_frames,
    iter_mixed_video_stream_frame_chunks as legacy_iter_stream_frame_chunks,
    load_mixed_video_catalog,
    normalized_video_frame_count,
    resolve_mixed_video_decode_size,
    resample_video_frames_to_fps,
    split_mixed_video_episodes,
    transform_frame,
)
from open_wam.data.mixed_video_catalog import (
    MixedVideoCatalog,
    MixedVideoEpisodeRecord,
    MixedVideoStreamRecord,
    load_mixed_video_catalog as canonical_load_mixed_video_catalog,
    split_mixed_video_episodes as canonical_split_mixed_video_episodes,
)
from open_wam.data.mixed_video_decode import (
    MixedVideoResolvedDecodeSize,
    decode_mixed_video_stream_frame_chunk,
    decode_video_frames as canonical_decode_video_frames,
    iter_mixed_video_stream_frame_chunks,
    normalized_video_frame_count as canonical_normalized_video_frame_count,
    resolve_mixed_video_decode_size as canonical_resolve_mixed_video_decode_size,
    resample_video_frames_to_fps as canonical_resample_video_frames_to_fps,
    transform_frame as canonical_transform_frame,
)
from open_wam.data.mixed_video_latent_storage import (
    MixedVideoLatentRepository,
)
from open_wam.data.mixed_video_planning import (
    MixedVideoWindowPlanner,
    MixedVideoWindowRecord,
)
from open_wam.models.common.video_geometry import WAN_TEMPORAL_CHUNK_SIZE, wan_raw_frame_count_to_latent_count


def _write_video(path: Path, *, num_frames: int, height: int, width: int, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        np.full((height, width, 3), frame_index + offset, dtype=np.uint8)
        for frame_index in range(num_frames)
    ]
    imageio.mimsave(path, frames, fps=10, macro_block_size=1)


def test_mixed_video_catalog_legacy_imports_preserve_identity() -> None:
    assert PublicMixedVideoCatalog is MixedVideoCatalog
    assert LegacyMixedVideoCatalog is MixedVideoCatalog
    assert LegacyMixedVideoEpisodeRecord is MixedVideoEpisodeRecord
    assert LegacyMixedVideoStreamRecord is MixedVideoStreamRecord
    assert load_mixed_video_catalog is canonical_load_mixed_video_catalog
    assert split_mixed_video_episodes is canonical_split_mixed_video_episodes


def test_mixed_video_planning_imports_preserve_identity() -> None:
    assert PublicMixedVideoWindowPlanner is MixedVideoWindowPlanner
    assert PublicMixedVideoWindowRecord is MixedVideoWindowRecord
    assert LegacyMixedVideoWindowRecord is MixedVideoWindowRecord


def test_mixed_video_decode_legacy_imports_preserve_identity() -> None:
    assert LegacyMixedVideoResolvedDecodeSize is MixedVideoResolvedDecodeSize
    assert public_decode_video_frames is canonical_decode_video_frames
    assert decode_video_frames is canonical_decode_video_frames
    assert (
        legacy_decode_stream_frame_chunk
        is decode_mixed_video_stream_frame_chunk
    )
    assert legacy_iter_stream_frame_chunks is iter_mixed_video_stream_frame_chunks
    assert normalized_video_frame_count is canonical_normalized_video_frame_count
    assert (
        resolve_mixed_video_decode_size
        is canonical_resolve_mixed_video_decode_size
    )
    assert (
        resample_video_frames_to_fps
        is canonical_resample_video_frames_to_fps
    )
    assert public_transform_frame is canonical_transform_frame
    assert transform_frame is canonical_transform_frame


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "source_id",
        "dataset_id",
        "episode_index",
        "clip_id",
        "stream_index",
        "stream_key",
        "target_slot_key",
        "video_path",
        "shard_relative_path",
        "latent_path",
        "latent_shard_relative_path",
        "length_frames",
        "latent_length_frames",
        "latent_key",
        "observation_fps",
        "action_fps",
        "from_timestamp",
        "to_timestamp",
        "tasks",
        "width",
        "height",
        "channels",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_pr143_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "source_id",
        "dataset_id",
        "episode_index",
        "stream_index",
        "stream_key",
        "target_slot_key",
        "video_path",
        "length_frames",
        "observation_fps",
        "tasks",
        "width",
        "height",
        "channels",
        "from_timestamp",
        "to_timestamp",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mixed_video_fixture_config(tmp_path: Path) -> MixedVideoDataConfig:
    source_a_root = tmp_path / "source_a"
    source_b_root = tmp_path / "source_b"
    for episode_index in range(2):
        _write_video(
            source_a_root / f"episode_{episode_index}.mp4",
            num_frames=8,
            height=12,
            width=20,
            offset=episode_index * 16,
        )
        _write_video(
            source_b_root / f"episode_{episode_index}.mp4",
            num_frames=8,
            height=20,
            width=12,
            offset=64 + episode_index * 16,
        )
        _write_latents(
            source_a_root / f"episode_{episode_index}_latents.pt",
            episode_offset=episode_index * 1000,
        )
        _write_latents(
            source_b_root / f"episode_{episode_index}_latents.pt",
            episode_offset=2000 + episode_index * 1000,
        )

    manifest_a = source_a_root / "manifest.csv"
    manifest_b = source_b_root / "manifest.csv"
    rows_a = []
    rows_b = []
    for episode_index in range(2):
        common = {
            "episode_index": episode_index,
            "stream_index": 0,
            "stream_key": "front",
            "target_slot_key": "observation.images.slot0",
            "length_frames": 8,
            "observation_fps": 10,
            "action_fps": 10,
            "tasks": f"task {episode_index}",
            "width": 20,
            "height": 12,
            "channels": 3,
        }
        rows_a.append(
            {
                **common,
                "source_id": "source_a",
                "dataset_id": "dataset_a",
                "video_path": f"episode_{episode_index}.mp4",
                "latent_path": f"episode_{episode_index}_latents.pt",
                "latent_length_frames": 8,
                "latent_key": "video_latents",
            }
        )
        rows_b.append(
            {
                **common,
                "source_id": "source_b",
                "dataset_id": "dataset_b",
                "video_path": f"episode_{episode_index}.mp4",
                "latent_path": f"episode_{episode_index}_latents.pt",
                "latent_length_frames": 8,
                "latent_key": "video_latents",
                "width": 12,
                "height": 20,
            }
        )
    _write_manifest(manifest_a, rows_a)
    _write_manifest(manifest_b, rows_b)

    return MixedVideoDataConfig(
        video_sources=(
            MixedVideoSourceConfig(source_id="source_a", manifest_csv=str(manifest_a), local_root=str(source_a_root)),
            MixedVideoSourceConfig(source_id="source_b", manifest_csv=str(manifest_b), local_root=str(source_b_root)),
        ),
        camera_names=("observation.images.slot0",),
        latent_camera_names=("observation.images.slot0",),
        canonical_height=8,
        canonical_width=8,
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=8,
                width=8,
            ),
        ),
        num_frames=4,
        frame_stride=1,
        sample_stride=2,
        episode_cache_size=1,
        train_fraction=1.0,
        action_schema=ActionSchemaConfig(action_dim=1, action_horizon=0, state_dim=1, state_horizon=0),
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=4,
            action_horizon=0,
            state_horizon=0,
            causal_prefix_suffix_buckets=(
                CausalPrefixSuffixBucketConfig(observed_frames=1, future_frames=3),
                CausalPrefixSuffixBucketConfig(observed_frames=2, future_frames=2),
            ),
        ),
        decode_height=8,
        decode_width=8,
        target_observation_fps=None,
        random_mode=MixedVideoRandomMode.NONE,
    )


def _with_wan_encoder_horizon(config: MixedVideoDataConfig, **overrides) -> MixedVideoDataConfig:
    return _dataclass_replace(
        config,
        num_frames=8,
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=8,
            action_horizon=0,
            state_horizon=0,
            causal_prefix_suffix_buckets=(
                CausalPrefixSuffixBucketConfig(observed_frames=4, future_frames=4),
            ),
        ),
        **overrides,
    )


def _two_camera_mixed_video_fixture_config(tmp_path: Path) -> MixedVideoDataConfig:
    root = tmp_path / "two_camera_source"
    _write_video(root / "front.mp4", num_frames=8, height=8, width=8, offset=0)
    _write_video(root / "wrist.mp4", num_frames=8, height=8, width=8, offset=64)
    manifest = root / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "source_id": "two_camera_source",
                "dataset_id": "two_camera_dataset",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "video_path": "front.mp4",
                "latent_path": "front_latents.pt",
                "length_frames": 8,
                "latent_length_frames": 8,
                "latent_key": "video_latents",
                "observation_fps": 10,
                "action_fps": 10,
                "tasks": "two camera task",
                "width": 8,
                "height": 8,
                "channels": 3,
            },
            {
                "source_id": "two_camera_source",
                "dataset_id": "two_camera_dataset",
                "episode_index": 0,
                "stream_index": 1,
                "stream_key": "wrist",
                "target_slot_key": "observation.images.slot1",
                "video_path": "wrist.mp4",
                "latent_path": "wrist_latents.pt",
                "length_frames": 8,
                "latent_length_frames": 8,
                "latent_key": "video_latents",
                "observation_fps": 10,
                "action_fps": 10,
                "tasks": "two camera task",
                "width": 8,
                "height": 8,
                "channels": 3,
            },
        ],
    )
    return MixedVideoDataConfig(
        video_sources=(
            MixedVideoSourceConfig(
                source_id="two_camera_source",
                manifest_csv=str(manifest),
                local_root=str(root),
            ),
        ),
        camera_names=("observation.images.slot0", "observation.images.slot1"),
        latent_camera_names=("observation.images.slot0", "observation.images.slot1"),
        canonical_height=8,
        canonical_width=16,
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=8,
                width=8,
            ),
            ViewLayoutConfig(
                source_name="observation.images.slot1",
                canonical_name="observation.images.slot1",
                top=0,
                left=8,
                height=8,
                width=8,
            ),
        ),
        num_frames=8,
        frame_stride=1,
        sample_stride=2,
        episode_cache_size=1,
        train_fraction=1.0,
        action_schema=ActionSchemaConfig(action_dim=1, action_horizon=0, state_dim=1, state_horizon=0),
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=8,
            action_horizon=0,
            state_horizon=0,
            causal_prefix_suffix_buckets=(
                CausalPrefixSuffixBucketConfig(observed_frames=4, future_frames=4),
            ),
        ),
        decode_height=8,
        decode_width=8,
        target_observation_fps=None,
        random_mode=MixedVideoRandomMode.NONE,
    )


def _write_latents(path: Path, *, episode_offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latents = torch.arange(48 * 8 * 2 * 2, dtype=torch.float32).reshape(48, 8, 2, 2)
    torch.save({"video_latents": latents + float(episode_offset)}, path)


class _FakeLatentEncoderAssets:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.cache_initialized = False

    def encode_video(self, canonical_video, *, placements=None, reset_cache: bool = True):
        self.calls.append(
            {
                "shape": tuple(canonical_video.shape),
                "reset_cache": reset_cache,
                "placements": placements,
            }
        )
        batch, _, frames, _, _ = canonical_video.shape
        if reset_cache:
            latent_frames = wan_raw_frame_count_to_latent_count(int(frames))
            self.cache_initialized = True
        else:
            latent_frames = (
                int(frames) // WAN_TEMPORAL_CHUNK_SIZE
                if self.cache_initialized
                else wan_raw_frame_count_to_latent_count(int(frames))
            )
            self.cache_initialized = True
        return torch.ones(batch, 48, latent_frames, 2, 2, dtype=torch.float32)


def test_mixed_video_dataset_decodes_multiple_sources_to_common_view_shape(tmp_path: Path) -> None:
    config = _mixed_video_fixture_config(tmp_path)
    train_dataset, val_dataset = build_train_val_datasets(config)
    assert isinstance(train_dataset, MixedVideoWindowDataset)
    assert isinstance(train_dataset._window_planner, MixedVideoWindowPlanner)
    assert (
        train_dataset.sample_index
        == train_dataset._window_planner.build_episode_windows(
            episode_records=train_dataset.episode_records,
            episode_keys=train_dataset.episode_keys,
        )
    )
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0

    sample = train_dataset[0]
    assert sample.views["observation.images.slot0"].shape == (4, 8, 8, 3)
    assert sample.actions.shape == (0, 1)
    assert sample.action_mask is not None
    assert sample.action_mask.shape == (0, 1)
    assert sample.metadata["valid_video_frames"] == 4
    assert sample.metadata["observed_prefix_frames"] in {1, 2}
    assert sample.metadata["future_suffix_frames"] in {2, 3}

    batch = collate_wam_samples([sample, train_dataset[1]])
    canonical = build_canonical_video_preprocessor(config)(batch.views)
    assert canonical.video.shape == (2, 3, 4, 8, 8)


def test_mixed_video_loads_pr143_manifest_schema_with_semicolon_tasks_and_mapping(tmp_path: Path) -> None:
    root = tmp_path / "pr143_source"
    _write_video(root / "cam_high.mp4", num_frames=8, height=8, width=8, offset=0)
    _write_video(root / "cam_left_wrist.mp4", num_frames=8, height=8, width=8, offset=64)
    manifest = root / "manifest.csv"
    _write_pr143_manifest(
        manifest,
        [
            {
                "source_id": "robotwin_aug",
                "dataset_id": "task_a",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "observation.images.cam_high",
                "target_slot_key": "observation.images.slot0",
                "video_path": "cam_high.mp4",
                "length_frames": 8,
                "observation_fps": 50,
                "tasks": "pick block; place block",
                "width": 8,
                "height": 8,
                "channels": 3,
                "from_timestamp": "",
                "to_timestamp": "",
            },
            {
                "source_id": "robotwin_aug",
                "dataset_id": "task_a",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "observation.images.cam_left_wrist",
                "target_slot_key": "observation.images.slot0",
                "video_path": "cam_left_wrist.mp4",
                "length_frames": 8,
                "observation_fps": 50,
                "tasks": "pick block; place block",
                "width": 8,
                "height": 8,
                "channels": 3,
                "from_timestamp": "",
                "to_timestamp": "",
            },
        ],
    )
    config = _dataclass_replace(
        _two_camera_mixed_video_fixture_config(tmp_path),
        video_sources=(
            MixedVideoSourceConfig(
                source_id="robotwin_aug",
                manifest_csv=str(manifest),
                local_root=str(root),
                channel_mappings=(
                    ConsortiumChannelMappingConfig(
                        source_name="observation.images.cam_high",
                        target_slot="observation.images.slot0",
                    ),
                    ConsortiumChannelMappingConfig(
                        source_name="observation.images.cam_left_wrist",
                        target_slot="observation.images.slot1",
                    ),
                ),
            ),
        ),
        target_observation_fps=None,
        train_fraction=1.0,
    )

    catalog = load_mixed_video_catalog(config)
    assert len(catalog.episodes) == 1
    assert catalog.episodes[0].tasks == ("pick block", "place block")
    assert {stream.target_slot for stream in catalog.episodes[0].streams} == {
        "observation.images.slot0",
        "observation.images.slot1",
    }


def test_mixed_video_frame_cache_respects_timestamp_windows(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    root = tmp_path / "shared_source"
    _write_video(root / "shared.mp4", num_frames=8, height=8, width=8, offset=0)
    manifest = root / "manifest.csv"
    rows = []
    for episode_index, start_time in enumerate((0.0, 0.4)):
        rows.append(
            {
                "source_id": "shared_source",
                "dataset_id": "shared_dataset",
                "episode_index": episode_index,
                "stream_index": 0,
                "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "video_path": "shared.mp4",
                "length_frames": 4,
                "observation_fps": 10,
                "action_fps": 10,
                "from_timestamp": start_time,
                "to_timestamp": start_time + 0.4,
                "width": 8,
                "height": 8,
                "channels": 3,
            }
        )
    _write_manifest(manifest, rows)
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="shared_source",
                manifest_csv=str(manifest),
                local_root=str(root),
            ),
        ),
        episode_cache_size=4,
        target_observation_fps=None,
        train_fraction=1.0,
    )
    train_dataset, _ = build_train_val_datasets(config)
    samples_by_episode = {}
    for index in range(len(train_dataset)):
        sample = train_dataset[index]
        if sample.metadata["observation_start"] == 0:
            samples_by_episode[int(sample.metadata["episode_index"])] = sample
        if set(samples_by_episode) == {0, 1}:
            break

    first_mean = float(samples_by_episode[0].views["observation.images.slot0"][0].float().mean().item())
    second_mean = float(samples_by_episode[1].views["observation.images.slot0"][0].float().mean().item())
    assert second_mean > first_mean + 2.0


def test_mixed_video_uses_container_fps_when_manifest_fps_missing(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    root = tmp_path / "missing_fps_source"
    _write_video(root / "episode_0.mp4", num_frames=8, height=8, width=8, offset=0)
    manifest = root / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "source_id": "missing_fps_source",
                "dataset_id": "missing_fps_dataset",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "video_path": "episode_0.mp4",
                "length_frames": 8,
                "action_fps": 10,
                "width": 8,
                "height": 8,
                "channels": 3,
            }
        ],
    )
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="missing_fps_source",
                manifest_csv=str(manifest),
                local_root=str(root),
            ),
        ),
        target_observation_fps=5.0,
        missing_observation_fps=30.0,
        train_fraction=1.0,
    )

    catalog = load_mixed_video_catalog(config)
    assert catalog.episodes[0].streams[0].observation_fps == pytest.approx(10.0)
    assert catalog.episodes[0].streams[0].clip.source_fps_source == "container"
    assert catalog.episodes[0].length_frames == 4

    train_dataset, _ = build_train_val_datasets(config)
    sample = train_dataset[0]
    assert sample.metadata["normalized_length_frames"] == 4
    assert sample.metadata["source_observation_fps"]["observation.images.slot0"] == pytest.approx(10.0)
    assert sample.metadata["source_observation_fps_source"]["observation.images.slot0"] == "container"


def test_mixed_video_remote_rgb_requires_manifest_fps_for_normalization(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    manifest = tmp_path / "remote_manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "source_id": "remote_source",
                "dataset_id": "remote_dataset",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "shard_relative_path": "videos/episode_0.mp4",
                "length_frames": 8,
                "width": 8,
                "height": 8,
                "channels": 3,
            }
        ],
    )
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="remote_source",
                manifest_csv=str(manifest),
                repo_id="org/remote_dataset",
            ),
        ),
        target_observation_fps=5.0,
    )

    with pytest.raises(ValueError, match="Remote mixed-video RGB rows must include `observation_fps`"):
        load_mixed_video_catalog(config)


def test_mixed_video_rejects_duplicate_episode_target_slots(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    root = tmp_path / "duplicate_slot_source"
    _write_video(root / "shared.mp4", num_frames=8, height=8, width=8, offset=0)
    manifest = root / "manifest.csv"
    rows = []
    for start_time in (0.0, 0.4):
        rows.append(
            {
                "source_id": "duplicate_slot_source",
                "dataset_id": "duplicate_dataset",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "video_path": "shared.mp4",
                "length_frames": 4,
                "observation_fps": 10,
                "from_timestamp": start_time,
                "to_timestamp": start_time + 0.4,
                "width": 8,
                "height": 8,
                "channels": 3,
            }
        )
    _write_manifest(manifest, rows)
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="duplicate_slot_source",
                manifest_csv=str(manifest),
                local_root=str(root),
            ),
        ),
    )

    with pytest.raises(ValueError, match="duplicate target slots"):
        load_mixed_video_catalog(config)


def test_mixed_video_explicit_clip_ids_split_timestamp_clips(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    root = tmp_path / "clip_id_source"
    _write_video(root / "shared.mp4", num_frames=8, height=8, width=8, offset=0)
    manifest = root / "manifest.csv"
    rows = []
    for clip_id, start_time in (("clip_a", 0.0), ("clip_b", 0.4)):
        rows.append(
            {
                "source_id": "clip_id_source",
                "dataset_id": "clip_dataset",
                "episode_index": 0,
                "clip_id": clip_id,
                "stream_index": 0,
                "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "video_path": "shared.mp4",
                "length_frames": 4,
                "observation_fps": 10,
                "from_timestamp": start_time,
                "to_timestamp": start_time + 0.4,
                "width": 8,
                "height": 8,
                "channels": 3,
            }
        )
    _write_manifest(manifest, rows)
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="clip_id_source",
                manifest_csv=str(manifest),
                local_root=str(root),
            ),
        ),
        train_fraction=1.0,
    )

    catalog = load_mixed_video_catalog(config)
    assert {episode.clip_id for episode in catalog.episodes} == {"clip_a", "clip_b"}
    assert len(catalog.episodes) == 2
    _, val_keys = split_mixed_video_episodes(config, catalog)
    assert set(val_keys) == {episode.key for episode in catalog.episodes}


def test_mixed_video_train_val_split_keeps_physical_clip_group_together(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    root = tmp_path / "clip_split_source"
    _write_video(root / "shared.mp4", num_frames=8, height=8, width=8, offset=0)
    _write_video(root / "other.mp4", num_frames=8, height=8, width=8, offset=32)
    manifest = root / "manifest.csv"
    rows = []
    for clip_id, start_time in (("clip_a", 0.0), ("clip_b", 0.4)):
        rows.append(
            {
                "source_id": "clip_split_source",
                "dataset_id": "clip_dataset",
                "episode_index": 0,
                "clip_id": clip_id,
                "stream_index": 0,
                "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "video_path": "shared.mp4",
                "length_frames": 4,
                "observation_fps": 10,
                "from_timestamp": start_time,
                "to_timestamp": start_time + 0.4,
                "width": 8,
                "height": 8,
                "channels": 3,
            }
        )
    rows.append(
        {
            "source_id": "clip_split_source",
            "dataset_id": "clip_dataset",
            "episode_index": 1,
            "stream_index": 0,
            "stream_key": "front",
            "target_slot_key": "observation.images.slot0",
            "video_path": "other.mp4",
            "length_frames": 8,
            "observation_fps": 10,
            "width": 8,
            "height": 8,
            "channels": 3,
        }
    )
    _write_manifest(manifest, rows)
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="clip_split_source",
                manifest_csv=str(manifest),
                local_root=str(root),
            ),
        ),
        train_fraction=0.5,
        split_seed=0,
    )

    catalog = load_mixed_video_catalog(config)
    train_keys, val_keys = split_mixed_video_episodes(config, catalog)
    clip_group_keys = {episode.key for episode in catalog.episodes if episode.episode_index == 0}

    assert train_keys and val_keys
    assert clip_group_keys <= set(train_keys) or clip_group_keys <= set(val_keys)


def test_mixed_video_split_max_limits_preserve_physical_groups(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    root = tmp_path / "clip_limit_source"
    _write_video(root / "shared.mp4", num_frames=8, height=8, width=8, offset=0)
    _write_video(root / "other.mp4", num_frames=8, height=8, width=8, offset=32)
    manifest = root / "manifest.csv"
    rows = []
    for clip_id, start_time in (("clip_a", 0.0), ("clip_b", 0.4)):
        rows.append(
            {
                "source_id": "clip_limit_source",
                "dataset_id": "clip_dataset",
                "episode_index": 0,
                "clip_id": clip_id,
                "stream_index": 0,
                "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "video_path": "shared.mp4",
                "length_frames": 4,
                "observation_fps": 10,
                "from_timestamp": start_time,
                "to_timestamp": start_time + 0.4,
                "width": 8,
                "height": 8,
                "channels": 3,
            }
        )
    rows.append(
        {
            "source_id": "clip_limit_source",
            "dataset_id": "clip_dataset",
            "episode_index": 1,
            "stream_index": 0,
            "stream_key": "front",
            "target_slot_key": "observation.images.slot0",
            "video_path": "other.mp4",
            "length_frames": 8,
            "observation_fps": 10,
            "width": 8,
            "height": 8,
            "channels": 3,
        }
    )
    _write_manifest(manifest, rows)
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="clip_limit_source",
                manifest_csv=str(manifest),
                local_root=str(root),
            ),
        ),
        train_fraction=1.0,
        max_train_episodes=1,
        split_seed=0,
    )

    catalog = load_mixed_video_catalog(config)
    train_keys, _ = split_mixed_video_episodes(config, catalog)
    episodes_by_key = {episode.key: episode for episode in catalog.episodes}
    train_physical_groups = {
        (episodes_by_key[key].episode_index, episodes_by_key[key].streams[0].clip.path_key)
        for key in train_keys
    }

    assert len(train_physical_groups) == 1
    if any(episodes_by_key[key].episode_index == 0 for key in train_keys):
        assert {episodes_by_key[key].clip_id for key in train_keys} == {"clip_a", "clip_b"}


def test_mixed_video_causal_bucket_loads_selected_span_and_pads_views(tmp_path: Path) -> None:
    config = _dataclass_replace(
        _mixed_video_fixture_config(tmp_path),
        num_frames=6,
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=6,
            action_horizon=0,
            state_horizon=0,
            causal_prefix_suffix_buckets=(
                CausalPrefixSuffixBucketConfig(observed_frames=1, future_frames=3),
            ),
        ),
    )
    train_dataset, _ = build_train_val_datasets(config)
    sample_index = next(
        index
        for index, window in enumerate(train_dataset.sample_index)
        if window.observation_start == 4
    )

    sample = train_dataset[sample_index]

    view = sample.views["observation.images.slot0"]
    assert view.shape == (6, 8, 8, 3)
    assert sample.metadata["valid_video_frames"] == 4
    assert sample.metadata["padded_video_frames"] == 6
    assert sample.metadata["observation_frame_indices"] == [4, 5, 6, 7]
    assert torch.count_nonzero(view[4:]).item() == 0


def test_mixed_video_causal_bucket_sampler_uses_shorter_valid_tail_buckets(tmp_path: Path) -> None:
    config = _dataclass_replace(
        _mixed_video_fixture_config(tmp_path),
        num_frames=6,
        sample_stride=1,
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=6,
            action_horizon=0,
            state_horizon=0,
            causal_prefix_suffix_buckets=(
                CausalPrefixSuffixBucketConfig(observed_frames=3, future_frames=3),
                CausalPrefixSuffixBucketConfig(observed_frames=1, future_frames=1),
            ),
        ),
    )
    train_dataset, _ = build_train_val_datasets(config)

    tail_windows = [
        window
        for window in train_dataset.sample_index
        if train_dataset.episode_records[window.episode_key].source_id == "source_a"
        and train_dataset.episode_records[window.episode_key].episode_index == 0
        and window.observation_start >= 3
    ]

    assert [window.observation_start for window in tail_windows] == [3, 4, 5, 6]
    assert {
        (window.observed_prefix_frames, window.future_suffix_frames)
        for window in tail_windows
    } == {(1, 1)}


def test_mixed_video_causal_bucket_loads_selected_span_and_pads_latents(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    config = _dataclass_replace(
        base_config,
        video_sources=tuple(
            _dataclass_replace(source, source_format=MixedVideoSourceFormat.LATENT)
            for source in base_config.video_sources
        ),
        num_frames=6,
        sample_construction=SampleConstructionConfig(
            mode=WindowSamplingMode.CAUSAL_PREFIX_SUFFIX,
            num_frames=6,
            action_horizon=0,
            state_horizon=0,
            causal_prefix_suffix_buckets=(
                CausalPrefixSuffixBucketConfig(observed_frames=1, future_frames=3),
            ),
        ),
    )
    train_dataset, _ = build_train_val_latent_datasets(config)
    sample_index = next(
        index
        for index, window in enumerate(train_dataset.sample_index)
        if window.observation_start == 4
    )

    sample = train_dataset[sample_index]

    assert sample.video_latents.shape == (48, 6, 2, 2)
    assert sample.metadata["valid_video_frames"] == 4
    assert sample.metadata["padded_video_frames"] == 6
    assert sample.metadata["observation_frame_indices"] == [4, 5, 6, 7]
    assert torch.count_nonzero(sample.video_latents[:, 4:]).item() == 0


def test_mixed_video_aspect_ratio_bins_resolve_vae_friendly_sizes(tmp_path: Path) -> None:
    config = _dataclass_replace(
        _mixed_video_fixture_config(tmp_path),
        decode_size_mode=MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS,
        train_batch_size=1,
        val_batch_size=1,
    )

    assert resolve_mixed_video_decode_size(config, source_height=100, source_width=100).height == 128
    large_square = resolve_mixed_video_decode_size(config, source_height=224, source_width=224)
    assert (large_square.height, large_square.width, large_square.bin_name) == (256, 256, "square_256")
    four_three = resolve_mixed_video_decode_size(config, source_height=480, source_width=640)
    assert (four_three.height, four_three.width) == (256, 352)
    sixteen_nine = resolve_mixed_video_decode_size(config, source_height=1080, source_width=1920)
    assert (sixteen_nine.height, sixteen_nine.width) == (192, 352)
    assert (four_three.height // 16) % 2 == 0
    assert (four_three.width // 16) % 2 == 0
    assert (sixteen_nine.height // 16) % 2 == 0
    assert (sixteen_nine.width // 16) % 2 == 0

    train_dataset, _ = build_train_val_datasets(config)
    assert isinstance(train_dataset, MixedVideoWindowDataset)
    sample = next(
        train_dataset[index]
        for index, window in enumerate(train_dataset.sample_index)
        if train_dataset.episode_records[window.episode_key].source_id == "source_a"
    )

    assert sample.views["observation.images.slot0"].shape == (4, 192, 352, 3)
    assert sample.metadata["decode_size_mode"] == "aspect_ratio_bins"
    assert sample.metadata["decode_fit_mode"] == "letterbox_pad"
    assert sample.metadata["decode_bins"]["observation.images.slot0"] == "sixteen_nine_352x192"

    batch = collate_wam_samples([sample])
    canonical = build_canonical_video_preprocessor(config)(batch.views)
    assert canonical.video.shape == (1, 3, 4, 192, 352)
    assert canonical.metadata["adaptive_canvas"] is True


def test_mixed_video_transform_frame_letterbox_pads_without_cropping() -> None:
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 64
    frame[:, :, 1] = 128
    frame[:, :, 2] = 255

    transformed = transform_frame(
        frame,
        target_height=8,
        target_width=8,
        center_crop=False,
        allow_upscale=True,
        fit_mode=MixedVideoFrameFitMode.LETTERBOX_PAD,
    )

    assert transformed.shape == (8, 8, 3)
    assert np.count_nonzero(transformed[:2]) == 0
    assert np.count_nonzero(transformed[6:]) == 0
    assert np.all(transformed[2:6, :, 0] == 64)
    assert np.all(transformed[2:6, :, 1] == 128)
    assert np.all(transformed[2:6, :, 2] == 255)


def test_mixed_video_rejects_contradictory_legacy_crop_config(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)

    with pytest.raises(ValueError, match="decode_center_crop=True"):
        _dataclass_replace(base_config, decode_center_crop=True)

    with pytest.raises(ValueError, match="decode_fit_mode=center_crop"):
        _dataclass_replace(
            base_config,
            decode_fit_mode=MixedVideoFrameFitMode.CENTER_CROP,
            decode_allow_upscale=False,
        )

    config = _dataclass_replace(
        base_config,
        decode_fit_mode=MixedVideoFrameFitMode.CENTER_CROP,
        decode_center_crop=True,
    )
    assert config.decode_fit_mode == MixedVideoFrameFitMode.CENTER_CROP


def test_mixed_video_resamples_frames_to_target_fps() -> None:
    frames = torch.arange(4, dtype=torch.uint8).reshape(4, 1, 1, 1) * 10

    upsampled = resample_video_frames_to_fps(frames, source_fps=10.0, target_fps=20.0)
    assert upsampled[:, 0, 0, 0].tolist() == [0, 5, 10, 15, 20, 25, 30, 30]

    downsampled = resample_video_frames_to_fps(frames, source_fps=20.0, target_fps=10.0)
    assert downsampled[:, 0, 0, 0].tolist() == [0, 20]

    assert normalized_video_frame_count(4, source_fps=None, target_fps=10.0, missing_source_fps=20.0) == 2


def test_mixed_video_decode_keeps_rounded_timestamp_boundary_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "timestamp_boundary.mp4"
    frames = [
        np.full((8, 8, 3), frame_index * 20, dtype=np.uint8)
        for frame_index in range(8)
    ]
    imageio.mimsave(video_path, frames, fps=10, macro_block_size=1)

    decoded = decode_video_frames(
        video_path,
        target_height=8,
        target_width=8,
        center_crop=False,
        allow_upscale=True,
        source_fps=10.0,
        target_fps=None,
        from_timestamp=0.40000001,
        to_timestamp=0.8,
    )

    assert decoded.shape[0] == 4
    assert float(decoded[0].float().mean().item()) < 90.0


def test_mixed_video_dataset_indexes_normalized_fps_timeline(tmp_path: Path) -> None:
    config = _dataclass_replace(_mixed_video_fixture_config(tmp_path), target_observation_fps=15.0)

    train_dataset, _ = build_train_val_datasets(config)
    assert isinstance(train_dataset, MixedVideoWindowDataset)
    sample = train_dataset[0]

    assert sample.views["observation.images.slot0"].shape == (4, 8, 8, 3)
    assert sample.metadata["native_length_frames"] == 8
    assert sample.metadata["normalized_length_frames"] == 12
    assert sample.metadata["target_observation_fps"] == 15.0
    assert sample.metadata["source_observation_fps"]["observation.images.slot0"] == 10.0


def test_mixed_video_latent_dataset_mixes_rgb_origin_and_latent_sources(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    config = _dataclass_replace(
        base_config,
        video_sources=(
            _dataclass_replace(
                base_config.video_sources[0],
                source_format=MixedVideoSourceFormat.RGB_AND_LATENT,
            ),
            _dataclass_replace(
                base_config.video_sources[1],
                source_format=MixedVideoSourceFormat.LATENT,
            ),
        ),
        train_batch_size=1,
        val_batch_size=1,
    )

    train_dataset, val_dataset = build_train_val_latent_datasets(config)

    assert isinstance(train_dataset, MixedVideoLatentWindowDataset)
    assert isinstance(train_dataset._window_planner, MixedVideoWindowPlanner)
    assert (
        train_dataset.sample_index
        == train_dataset._window_planner.build_latent_view_windows(
            episode_records=train_dataset.episode_records,
            episode_keys=train_dataset.episode_keys,
        )
    )
    assert isinstance(
        train_dataset._latent_repository,
        MixedVideoLatentRepository,
    )
    assert not train_dataset._latent_repository.cache
    assert len(val_dataset) > 0
    source_ids = {
        train_dataset.episode_records[window.episode_key].source_id
        for window in train_dataset.sample_index
    }
    assert {"source_a", "source_b"}.issubset(source_ids)

    source_a_sample = next(
        train_dataset[index]
        for index, window in enumerate(train_dataset.sample_index)
        if train_dataset.episode_records[window.episode_key].source_id == "source_a"
    )
    source_b_sample = next(
        train_dataset[index]
        for index, window in enumerate(train_dataset.sample_index)
        if train_dataset.episode_records[window.episode_key].source_id == "source_b"
    )
    assert source_a_sample.video_latents.shape == (48, 4, 2, 2)
    assert source_b_sample.video_latents.shape == (48, 4, 2, 2)
    assert source_a_sample.metadata["mixed_video_training_input"] == "latents"
    assert source_b_sample.metadata["source_id"] == "source_b"

    batch = collate_latent_wam_samples([source_a_sample, source_b_sample])
    assert batch.video_latents.shape == (2, 48, 4, 2, 2)
    assert batch.metadata[0]["latent_shape"] == [48, 4, 2, 2]


def test_mixed_video_latent_dataset_supports_latent_only_mixtures(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    config = _dataclass_replace(
        base_config,
        video_sources=tuple(
            _dataclass_replace(source, source_format=MixedVideoSourceFormat.LATENT)
            for source in base_config.video_sources
        ),
        train_batch_size=1,
        val_batch_size=1,
    )

    train_dataset, _ = build_train_val_latent_datasets(config)
    source_ids = {
        train_dataset.episode_records[window.episode_key].source_id
        for window in train_dataset.sample_index
    }
    assert {"source_a", "source_b"}.issubset(source_ids)

    samples = [
        next(
            train_dataset[index]
            for index, window in enumerate(train_dataset.sample_index)
            if train_dataset.episode_records[window.episode_key].source_id == source_id
        )
        for source_id in ("source_a", "source_b")
    ]
    batch = collate_latent_wam_samples(samples)
    assert batch.video_latents.shape == (2, 48, 4, 2, 2)
    assert all(metadata["mixed_video_training_input"] == "latents" for metadata in batch.metadata)


def test_mixed_video_latent_dataset_uses_latent_camera_names(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    root = tmp_path / "latent_slot_source"
    _write_latents(root / "episode_0_latents.pt", episode_offset=0)
    manifest = root / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "source_id": "latent_slot_source",
                "dataset_id": "latent_slot_dataset",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "encoded",
                "target_slot_key": "latent.video",
                "latent_path": "episode_0_latents.pt",
                "length_frames": 8,
                "latent_length_frames": 8,
                "latent_key": "video_latents",
                "tasks": "latent slot task",
            }
        ],
    )
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="latent_slot_source",
                manifest_csv=str(manifest),
                local_root=str(root),
                latent_root=str(root),
                source_format=MixedVideoSourceFormat.LATENT,
            ),
        ),
        latent_camera_names=("latent.video",),
        train_batch_size=1,
        val_batch_size=1,
    )

    train_dataset, _ = build_train_val_latent_datasets(config)
    sample = train_dataset[0]

    assert sample.video_latents.shape == (48, 4, 2, 2)
    assert sample.metadata["stream_keys"] == {"latent.video": "encoded"}


def test_mixed_video_latent_view_assembly_supports_weighted_combinations(tmp_path: Path) -> None:
    config = _two_camera_mixed_video_fixture_config(tmp_path)
    root = Path(config.video_sources[0].local_root)
    _write_latents(root / "front_latents.pt", episode_offset=0)
    _write_latents(root / "wrist_latents.pt", episode_offset=1000)
    source = _dataclass_replace(config.video_sources[0], source_format=MixedVideoSourceFormat.RGB_AND_LATENT)
    config = _dataclass_replace(
        config,
        video_sources=(source,),
        latent_view_combinations=(
            MixedVideoViewCombinationConfig(
                name="front_only",
                slots=("observation.images.slot0",),
                sampling_weight=1.0,
            ),
            MixedVideoViewCombinationConfig(
                name="wrist_only",
                slots=("observation.images.slot1",),
                sampling_weight=1.0,
            ),
            MixedVideoViewCombinationConfig(
                name="front_wrist",
                slots=("observation.images.slot0", "observation.images.slot1"),
                sampling_weight=2.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config)
    counts = Counter(window.view_combination_name for window in train_dataset.sample_index)

    assert counts["front_only"] == 1
    assert counts["wrist_only"] == 1
    assert counts["front_wrist"] == 2

    samples = {
        sample.metadata["view_combination_name"]: sample
        for sample in (train_dataset[index] for index in range(len(train_dataset)))
    }
    assert samples["front_only"].video_latents.shape == (48, 8, 2, 4)
    assert samples["front_only"].metadata["latent_view_assembly"]["placements"] == [
        {"slot": "observation.images.slot0", "top": 0, "left": 1, "height": 2, "width": 2}
    ]
    assert samples["front_wrist"].video_latents.shape == (48, 8, 2, 4)
    assert samples["front_wrist"].metadata["latent_view_assembly"]["placements"] == [
        {"slot": "observation.images.slot0", "top": 0, "left": 0, "height": 2, "width": 2},
        {"slot": "observation.images.slot1", "top": 0, "left": 2, "height": 2, "width": 2},
    ]

    batch = collate_latent_wam_samples([samples["front_only"], samples["front_wrist"]])
    assert batch.video_latents.shape == (2, 48, 8, 2, 4)


def test_mixed_video_latent_view_combinations_can_select_manifest_only_slots(tmp_path: Path) -> None:
    config = _two_camera_mixed_video_fixture_config(tmp_path)
    root = Path(config.video_sources[0].local_root)
    _write_latents(root / "front_latents.pt", episode_offset=0)
    _write_latents(root / "wrist_latents.pt", episode_offset=1000)
    source = _dataclass_replace(config.video_sources[0], source_format=MixedVideoSourceFormat.RGB_AND_LATENT)
    config = _dataclass_replace(
        config,
        video_sources=(source,),
        camera_names=("observation.images.slot0",),
        latent_camera_names=("observation.images.slot0",),
        view_layout=config.view_layout[:1],
        latent_view_combinations=(
            MixedVideoViewCombinationConfig(
                name="manifest_wrist_only",
                slots=("observation.images.slot1",),
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config)
    sample = train_dataset[0]

    assert sample.metadata["view_combination_slots"] == ["observation.images.slot1"]
    assert sample.metadata["stream_keys"] == {"observation.images.slot1": "wrist"}
    assert sample.video_latents.shape == (48, 8, 2, 2)


def test_mixed_video_disabled_latent_view_combinations_do_not_load_slots(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    latent_root = tmp_path / "enabled_latent_source"
    rgb_root = tmp_path / "disabled_rgb_source"
    _write_latents(latent_root / "enabled_latents.pt", episode_offset=0)
    _write_video(rgb_root / "disabled.mp4", num_frames=8, height=8, width=8, offset=32)
    latent_manifest = latent_root / "manifest.csv"
    rgb_manifest = rgb_root / "manifest.csv"
    _write_manifest(
        latent_manifest,
        [
            {
                "source_id": "enabled_latent_source",
                "dataset_id": "enabled_latent_dataset",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "enabled",
                "target_slot_key": "observation.images.slot0",
                "latent_path": "enabled_latents.pt",
                "length_frames": 8,
                "latent_length_frames": 8,
                "latent_key": "video_latents",
                "tasks": "enabled latent task",
            }
        ],
    )
    _write_manifest(
        rgb_manifest,
        [
            {
                "source_id": "disabled_rgb_source",
                "dataset_id": "disabled_rgb_dataset",
                "episode_index": 0,
                "stream_index": 0,
                "stream_key": "disabled",
                "target_slot_key": "observation.images.slot1",
                "video_path": "disabled.mp4",
                "length_frames": 8,
                "observation_fps": 10,
                "tasks": "disabled rgb task",
                "width": 8,
                "height": 8,
                "channels": 3,
            }
        ],
    )
    config = _dataclass_replace(
        base_config,
        video_sources=(
            MixedVideoSourceConfig(
                source_id="enabled_latent_source",
                manifest_csv=str(latent_manifest),
                local_root=str(latent_root),
                source_format=MixedVideoSourceFormat.LATENT,
            ),
            MixedVideoSourceConfig(
                source_id="disabled_rgb_source",
                manifest_csv=str(rgb_manifest),
                local_root=str(rgb_root),
                source_format=MixedVideoSourceFormat.RGB,
            ),
        ),
        latent_view_combinations=(
            MixedVideoViewCombinationConfig(
                name="enabled_latent_slot",
                slots=("observation.images.slot0",),
                source_ids=("enabled_latent_source",),
            ),
            MixedVideoViewCombinationConfig(
                name="disabled_rgb_slot",
                slots=("observation.images.slot1",),
                source_ids=("disabled_rgb_source",),
                enabled=False,
            ),
        ),
    )

    catalog = load_mixed_video_catalog(config)
    assert [episode.source_id for episode in catalog.episodes] == ["enabled_latent_source"]

    train_dataset, _ = build_train_val_latent_datasets(config)
    sample = train_dataset[0]
    assert sample.metadata["source_id"] == "enabled_latent_source"
    assert sample.metadata["stream_keys"] == {"observation.images.slot0": "enabled"}


def test_mixed_video_latent_view_assembly_layouts_one_to_four_views() -> None:
    latents = [
        torch.full((2, 3, 4, 5), float(index + 1))
        for index in range(4)
    ]

    one, one_meta = assemble_mixed_video_latent_views(latents[:1], slots=("a",), canvas_view_count=2)
    two, two_meta = assemble_mixed_video_latent_views(latents[:2], slots=("a", "b"), canvas_view_count=2)
    three, three_meta = assemble_mixed_video_latent_views(
        latents[:3],
        slots=("a", "b", "c"),
        canvas_view_count=3,
    )
    four, four_meta = assemble_mixed_video_latent_views(
        latents,
        slots=("a", "b", "c", "d"),
        canvas_view_count=4,
    )

    assert one.shape == (2, 3, 4, 10)
    assert one_meta["placements"][0]["left"] == 2
    assert two.shape == (2, 3, 4, 10)
    assert two_meta["placements"][1]["left"] == 5
    assert three.shape == (2, 3, 8, 10)
    assert three_meta["placements"][2] == {"slot": "c", "top": 4, "left": 2, "height": 4, "width": 5}
    assert four.shape == (2, 3, 8, 10)
    assert four_meta["placements"][3] == {"slot": "d", "top": 4, "left": 5, "height": 4, "width": 5}


def test_mixed_video_decord_fallback_does_not_restart_after_partial_emit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _mixed_video_fixture_config(tmp_path)
    catalog = load_mixed_video_catalog(config)
    stream = catalog.episodes[0].streams[0]
    imageio_calls: list[bool] = []

    def partial_decord(*args, **kwargs):
        del args, kwargs
        yield torch.zeros(4, 8, 8, 3, dtype=torch.uint8)
        raise RuntimeError("decord broke after yielding")

    def imageio_fallback(*args, **kwargs):
        del args, kwargs
        imageio_calls.append(True)
        yield torch.ones(4, 8, 8, 3, dtype=torch.uint8)

    monkeypatch.setattr(mixed_video_decode_module, "_HAS_DECORD", True)
    monkeypatch.setattr(
        mixed_video_decode_module,
        "_iter_chunks_decord",
        partial_decord,
    )
    monkeypatch.setattr(
        mixed_video_decode_module,
        "_iter_chunks_imageio",
        imageio_fallback,
    )

    with pytest.raises(RuntimeError, match="decord broke after yielding"):
        list(
            mixed_video_decode_module.iter_mixed_video_stream_frame_chunks(
                config,
                stream,
                raw_chunk_ranges=((0, 4), (4, 8)),
            )
        )

    assert imageio_calls == []


def test_mixed_video_latent_encoder_writes_manifest_compatible_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    assert encoder._streaming_chunk_ranges(70, max_chunk_frames=65) == ((0, 65), (65, 69))

    base_config = _mixed_video_fixture_config(tmp_path)
    config = _with_wan_encoder_horizon(
        base_config,
        video_sources=(base_config.video_sources[0],),
        train_fraction=1.0,
    )
    assets = _FakeLatentEncoderAssets()
    output_root = tmp_path / "encoded"
    reader_paths: list[object] = []
    original_get_reader = mixed_video_decode_module.imageio.get_reader

    def counted_get_reader(*args, **kwargs):
        reader_paths.append(args[0])
        return original_get_reader(*args, **kwargs)

    monkeypatch.setattr(
        mixed_video_decode_module.imageio,
        "get_reader",
        counted_get_reader,
    )

    report = encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=assets,
        output_root=output_root,
        device=torch.device("cpu"),
        experiment_config=_dataclass_replace(
            load_experiment_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "experiments"
                / "causal_video_prediction_robotwin_smoke.yaml"
            ),
            data=config,
        ),
        split="all",
        source_ids=("source_a",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )

    manifest_path = Path(report["manifest_paths"]["source_a"])
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["source_id"] == "source_a"
    assert rows[0]["length_frames"] == rows[0]["latent_length_frames"] == "2"
    assert rows[0]["raw_length_frames"] == "8"
    config_patch = Path(report["config_patch_path"]).read_text(encoding="utf-8")
    assert "num_frames: 2" in config_patch
    assert "raw 4 + 4 frames -> latent 1 + 1 frames" in config_patch
    latent_training_config = load_experiment_config(Path(report["latent_training_config_path"]))
    assert latent_training_config.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert latent_training_config.backbone.load_wan_vae_frontend is False
    assert latent_training_config.data.num_frames == 2
    assert all(
        source.source_format == MixedVideoSourceFormat.LATENT
        for source in latent_training_config.data.video_sources
    )
    assert [call["shape"][2] for call in assets.calls] == [5]
    assert [call["reset_cache"] for call in assets.calls] == [True]
    assert len(reader_paths) == 1

    with pytest.raises(FileExistsError, match="Preflight failed"):
        encoder.encode_mixed_video_latent_sources(
            data_config=config,
            assets=assets,
            output_root=output_root,
            device=torch.device("cpu"),
            split="all",
            source_ids=("source_a",),
            max_episodes=1,
            chunk_frames=5,
            overwrite=False,
        )
    assert [call["shape"][2] for call in assets.calls] == [5]

    resumed_report = encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=assets,
        output_root=output_root,
        device=torch.device("cpu"),
        split="all",
        source_ids=("source_a",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=False,
        skip_existing=True,
    )
    assert resumed_report["reused_episodes"] == 1
    assert resumed_report["newly_encoded_episodes"] == 0
    assert [call["shape"][2] for call in assets.calls] == [5]

    train_dataset, _ = build_train_val_latent_datasets(latent_training_config.data)
    sample = train_dataset[0]

    assert sample.video_latents.shape == (48, 2, 2, 2)
    assert sample.metadata["mixed_video_training_input"] == "latents"


def test_mixed_video_latent_encoder_surfaces_async_save_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    base_config = _mixed_video_fixture_config(tmp_path)
    config = _with_wan_encoder_horizon(
        base_config,
        video_sources=(base_config.video_sources[0],),
        train_fraction=1.0,
    )
    output_root = tmp_path / "encoded_save_failure"

    def fail_save(*args, **kwargs):
        del args, kwargs
        raise OSError("simulated save failure")

    monkeypatch.setattr(encoder.torch, "save", fail_save)

    with pytest.raises(RuntimeError, match="Failed to write mixed-video latent sidecar"):
        encoder.encode_mixed_video_latent_sources(
            data_config=config,
            assets=_FakeLatentEncoderAssets(),
            output_root=output_root,
            device=torch.device("cpu"),
            split="all",
            source_ids=("source_a",),
            max_episodes=1,
            chunk_frames=5,
            overwrite=True,
        )

    assert not (output_root / "manifests" / "source_a.csv").exists()


def test_mixed_video_latent_encoder_bounds_pending_async_saves(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    class ImmediateExecutor:
        def __init__(self) -> None:
            self.submitted = 0

        def submit(self, *args, **kwargs) -> Future:
            del args, kwargs
            self.submitted += 1
            future: Future = Future()
            future.set_result(None)
            return future

    executor = ImmediateExecutor()
    pending: list[tuple[Path, Future]] = []

    encoder._submit_latent_save(
        save_executor=executor,
        save_futures=pending,
        latent_path=tmp_path / "episode_0.pt",
        payload={"video_latents": torch.zeros(1)},
        max_pending=1,
    )
    encoder._submit_latent_save(
        save_executor=executor,
        save_futures=pending,
        latent_path=tmp_path / "episode_1.pt",
        payload={"video_latents": torch.zeros(1)},
        max_pending=1,
    )

    assert executor.submitted == 2
    assert len(pending) == 1
    assert pending[0][0] == tmp_path / "episode_1.pt"


def test_mixed_video_latent_encoder_writes_trainable_multicamera_config(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    config = _two_camera_mixed_video_fixture_config(tmp_path)
    output_root = tmp_path / "encoded_multicamera_trainable"
    report = encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=_FakeLatentEncoderAssets(),
        output_root=output_root,
        device=torch.device("cpu"),
        experiment_config=_dataclass_replace(
            load_experiment_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "experiments"
                / "causal_video_prediction_robotwin_smoke.yaml"
            ),
            data=config,
        ),
        split="all",
        source_ids=("two_camera_source",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )

    config_patch = Path(report["config_patch_path"]).read_text(encoding="utf-8")
    assert 'camera_names: ["observation.images.slot0"]' in config_patch
    assert 'latent_camera_names: ["observation.images.slot0"]' in config_patch

    latent_training_config = load_experiment_config(Path(report["latent_training_config_path"]))
    assert tuple(latent_training_config.data.camera_names) == ("observation.images.slot0",)
    assert tuple(latent_training_config.data.latent_camera_names) == ("observation.images.slot0",)
    assert len(latent_training_config.data.view_layout) == 1
    assert latent_training_config.trainer.batch_adapter == BatchAdapterName.LATENTS

    train_dataset, _ = build_train_val_latent_datasets(latent_training_config.data)
    sample = train_dataset[0]

    assert sample.video_latents.shape == (48, 2, 2, 2)
    assert sample.metadata["mixed_video_training_input"] == "latents"


def test_mixed_video_canonical_encoder_requires_configured_rgb_slots(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    config = _two_camera_mixed_video_fixture_config(tmp_path)
    source = _dataclass_replace(config.video_sources[0], include_streams=("front",))
    config = _dataclass_replace(config, video_sources=(source,))

    with pytest.raises(KeyError, match="missing RGB streams"):
        encoder.encode_mixed_video_latent_sources(
            data_config=config,
            assets=_FakeLatentEncoderAssets(),
            output_root=tmp_path / "encoded_missing_slot",
            device=torch.device("cpu"),
            split="all",
            source_ids=("two_camera_source",),
            max_episodes=1,
            chunk_frames=5,
            overwrite=True,
        )


def test_mixed_video_encoder_rejects_empty_training_manifest(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    config = _dataclass_replace(
        _two_camera_mixed_video_fixture_config(tmp_path),
        latent_encoding_mode=MixedVideoLatentEncodingMode.PER_VIEW,
    )

    with pytest.raises(ValueError, match="no trainable manifest records"):
        encoder.encode_mixed_video_latent_sources(
            data_config=config,
            assets=_FakeLatentEncoderAssets(),
            output_root=tmp_path / "encoded_empty_manifest",
            device=torch.device("cpu"),
            split="all",
            source_ids=("two_camera_source",),
            max_episodes=0,
            chunk_frames=5,
            overwrite=True,
        )


def test_mixed_video_latent_encoder_writes_per_view_sidecars_and_config(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    config = _dataclass_replace(
        _two_camera_mixed_video_fixture_config(tmp_path),
        latent_encoding_mode=MixedVideoLatentEncodingMode.PER_VIEW,
    )
    assets = _FakeLatentEncoderAssets()
    report = encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=assets,
        output_root=tmp_path / "encoded_per_view",
        device=torch.device("cpu"),
        experiment_config=_dataclass_replace(
            load_experiment_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "experiments"
                / "causal_video_prediction_robotwin_smoke.yaml"
            ),
            data=config,
        ),
        split="all",
        source_ids=("two_camera_source",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )

    manifest_path = Path(report["manifest_paths"]["two_camera_source"])
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    assert {row["target_slot_key"] for row in rows} == {
        "observation.images.slot0",
        "observation.images.slot1",
    }
    assert {row["encoding_mode"] for row in rows} == {"per_view"}
    assert [call["shape"][-2:] for call in assets.calls] == [(8, 8), (8, 8)]

    payloads = [
        torch.load((manifest_path.parent / row["latent_path"]).resolve(), map_location="cpu")
        for row in rows
    ]
    assert {payload["metadata"]["target_slot"] for payload in payloads} == {
        "observation.images.slot0",
        "observation.images.slot1",
    }

    latent_training_config = load_experiment_config(Path(report["latent_training_config_path"]))
    assert tuple(latent_training_config.data.latent_camera_names) == (
        "observation.images.slot0",
        "observation.images.slot1",
    )
    assert latent_training_config.data.latent_encoding_mode == MixedVideoLatentEncodingMode.PER_VIEW

    train_dataset, _ = build_train_val_latent_datasets(latent_training_config.data)
    sample = train_dataset[0]
    assert sample.video_latents.shape == (48, 2, 2, 4)
    assert sample.metadata["view_combination_slots"] == [
        "observation.images.slot0",
        "observation.images.slot1",
    ]


def test_mixed_video_latent_encoder_canonical_and_per_view_manifest_is_trainable(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    config = _dataclass_replace(
        _two_camera_mixed_video_fixture_config(tmp_path),
        latent_encoding_mode=MixedVideoLatentEncodingMode.CANONICAL_AND_PER_VIEW,
    )
    output_root = tmp_path / "encoded_canonical_and_per_view"
    report = encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=_FakeLatentEncoderAssets(),
        output_root=output_root,
        device=torch.device("cpu"),
        experiment_config=_dataclass_replace(
            load_experiment_config(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "experiments"
                / "causal_video_prediction_robotwin_smoke.yaml"
            ),
            data=config,
        ),
        split="all",
        source_ids=("two_camera_source",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )

    latent_root = output_root / "latents" / "two_camera_source" / "two_camera_dataset"
    assert (latent_root / "episode_000000.pt").exists()
    assert (latent_root / "episode_000000__observation.images.slot0.pt").exists()
    assert (latent_root / "episode_000000__observation.images.slot1.pt").exists()
    assert report["encoded_episodes"] == 1
    assert report["encoded_targets"] == 3
    assert report["manifest_encoded_episodes"] == 1
    assert report["manifest_encoded_targets"] == 2
    assert report["newly_encoded_episodes"] == 1
    assert report["newly_encoded_targets"] == 3
    assert report["reused_episodes"] == 0
    assert report["reused_targets"] == 0
    assert len(report["latent_shapes"]) == 3
    assert {
        tuple(key.rsplit(":", 2)[-2:])
        for key in report["latent_shapes"]
    } == {
        ("canonical", "observation.images.slot0"),
        ("per_view", "observation.images.slot0"),
        ("per_view", "observation.images.slot1"),
    }

    manifest_path = Path(report["manifest_paths"]["two_camera_source"])
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert {row["encoding_mode"] for row in rows} == {"per_view"}
    assert {row["target_slot_key"] for row in rows} == {
        "observation.images.slot0",
        "observation.images.slot1",
    }

    latent_training_config = load_experiment_config(Path(report["latent_training_config_path"]))
    assert latent_training_config.data.latent_encoding_mode == MixedVideoLatentEncodingMode.PER_VIEW
    train_dataset, _ = build_train_val_latent_datasets(latent_training_config.data)
    sample = train_dataset[0]
    assert sample.video_latents.shape == (48, 2, 2, 4)


def test_mixed_video_per_view_encoder_reuses_existing_single_view_sidecar(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    fixture_config = _mixed_video_fixture_config(tmp_path)
    base_config = _with_wan_encoder_horizon(
        fixture_config,
        video_sources=(fixture_config.video_sources[0],),
        train_fraction=1.0,
    )
    output_root = tmp_path / "encoded_reuse"
    encoder.encode_mixed_video_latent_sources(
        data_config=base_config,
        assets=_FakeLatentEncoderAssets(),
        output_root=output_root,
        device=torch.device("cpu"),
        split="all",
        source_ids=("source_a",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )

    per_view_config = _dataclass_replace(base_config, latent_encoding_mode=MixedVideoLatentEncodingMode.PER_VIEW)
    resumed = encoder.encode_mixed_video_latent_sources(
        data_config=per_view_config,
        assets=None,
        output_root=output_root,
        device=torch.device("cpu"),
        split="all",
        source_ids=("source_a",),
        max_episodes=1,
        chunk_frames=5,
        skip_existing=True,
    )

    manifest_path = Path(resumed["manifest_paths"]["source_a"])
    row = next(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    assert resumed["reused_episodes"] == 1
    assert row["latent_path"].endswith("episode_000000.pt")
    assert not row["latent_path"].endswith("__observation_images_slot0.pt")


def test_mixed_video_latent_encoder_resamples_rgb_to_target_fps(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    base_config = _mixed_video_fixture_config(tmp_path)
    config = _with_wan_encoder_horizon(
        base_config,
        video_sources=(base_config.video_sources[0],),
        target_observation_fps=15.0,
        train_fraction=1.0,
    )
    assets = _FakeLatentEncoderAssets()
    report = encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=assets,
        output_root=tmp_path / "encoded_fps",
        device=torch.device("cpu"),
        split="all",
        source_ids=("source_a",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )

    manifest_path = Path(report["manifest_paths"]["source_a"])
    row = next(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    assert row["length_frames"] == row["latent_length_frames"] == "3"
    assert row["raw_length_frames"] == "12"
    assert [call["shape"][2] for call in assets.calls] == [5, 4]


def test_mixed_video_encoder_honors_yaml_fit_mode_by_default(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    base_config = _mixed_video_fixture_config(tmp_path)
    center_crop_config = _dataclass_replace(base_config, decode_fit_mode=MixedVideoFrameFitMode.CENTER_CROP)

    resolved = encoder.resolve_encoder_data_config(center_crop_config, decode_size_mode="config", decode_fit_mode="config")

    assert resolved.decode_fit_mode == MixedVideoFrameFitMode.CENTER_CROP


def test_mixed_video_latent_encoder_rejects_stale_transform_signature(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    base_config = _mixed_video_fixture_config(tmp_path)
    config = _with_wan_encoder_horizon(
        base_config,
        video_sources=(base_config.video_sources[0],),
        train_fraction=1.0,
    )
    assets = _FakeLatentEncoderAssets()
    output_root = tmp_path / "encoded_stale_signature"
    encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=assets,
        output_root=output_root,
        device=torch.device("cpu"),
        split="all",
        source_ids=("source_a",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="decode_allow_upscale|transform_signature_hash"):
        encoder.encode_mixed_video_latent_sources(
            data_config=_dataclass_replace(config, decode_allow_upscale=False),
            assets=assets,
            output_root=output_root,
            device=torch.device("cpu"),
            split="all",
            source_ids=("source_a",),
            max_episodes=1,
            chunk_frames=5,
            skip_existing=True,
        )


def test_mixed_video_latent_encoder_rejects_stale_canonical_layout(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    config = _two_camera_mixed_video_fixture_config(tmp_path)
    output_root = tmp_path / "encoded_stale_layout"
    assets = _FakeLatentEncoderAssets()
    encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=assets,
        output_root=output_root,
        device=torch.device("cpu"),
        split="all",
        source_ids=("two_camera_source",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )
    assert [call["shape"][-2:] for call in assets.calls] == [(8, 16)]

    swapped_config = _dataclass_replace(
        config,
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=8,
                height=8,
                width=8,
            ),
            ViewLayoutConfig(
                source_name="observation.images.slot1",
                canonical_name="observation.images.slot1",
                top=0,
                left=0,
                height=8,
                width=8,
            ),
        ),
    )
    with pytest.raises(ValueError, match="transform_signature_hash"):
        encoder.encode_mixed_video_latent_sources(
            data_config=swapped_config,
            assets=_FakeLatentEncoderAssets(),
            output_root=output_root,
            device=torch.device("cpu"),
            split="all",
            source_ids=("two_camera_source",),
            max_episodes=1,
            chunk_frames=5,
            skip_existing=True,
        )


def test_mixed_video_latent_encoder_rejects_stale_stream_fps(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    base_config = _mixed_video_fixture_config(tmp_path)
    config = _with_wan_encoder_horizon(
        base_config,
        video_sources=(base_config.video_sources[0],),
        target_observation_fps=15.0,
        train_fraction=1.0,
    )
    output_root = tmp_path / "encoded_stale_fps"
    encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=_FakeLatentEncoderAssets(),
        output_root=output_root,
        device=torch.device("cpu"),
        split="all",
        source_ids=("source_a",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )

    rows = list(csv.DictReader(Path(config.video_sources[0].manifest_csv).open("r", encoding="utf-8")))
    for row in rows:
        row["observation_fps"] = "10.5"
    changed_manifest = tmp_path / "source_a_changed_fps.csv"
    _write_manifest(changed_manifest, rows)
    changed_config = _dataclass_replace(
        config,
        video_sources=(
            _dataclass_replace(
                config.video_sources[0],
                manifest_csv=str(changed_manifest),
            ),
        ),
    )

    with pytest.raises(ValueError, match="transform_signature_hash"):
        encoder.encode_mixed_video_latent_sources(
            data_config=changed_config,
            assets=_FakeLatentEncoderAssets(),
            output_root=output_root,
            device=torch.device("cpu"),
            split="all",
            source_ids=("source_a",),
            max_episodes=1,
            chunk_frames=5,
            skip_existing=True,
        )


def test_mixed_video_latent_encoder_rejects_stale_sidecar_metadata(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    base_config = _mixed_video_fixture_config(tmp_path)
    config = _with_wan_encoder_horizon(
        base_config,
        video_sources=(base_config.video_sources[0],),
        train_fraction=1.0,
    )
    assets = _FakeLatentEncoderAssets()
    output_root = tmp_path / "encoded_stale"
    report = encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=assets,
        output_root=output_root,
        device=torch.device("cpu"),
        split="all",
        source_ids=("source_a",),
        max_episodes=1,
        chunk_frames=5,
        overwrite=True,
    )
    manifest_path = Path(report["manifest_paths"]["source_a"])
    row = next(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    latent_path = (manifest_path.parent / row["latent_path"]).resolve()
    payload = torch.load(latent_path, map_location="cpu")
    payload["metadata"]["source_id"] = "other_source"
    torch.save(payload, latent_path)

    with pytest.raises(ValueError, match="source_id"):
        encoder.encode_mixed_video_latent_sources(
            data_config=config,
            assets=assets,
            output_root=output_root,
            device=torch.device("cpu"),
            split="all",
            source_ids=("source_a",),
            max_episodes=1,
            chunk_frames=5,
            skip_existing=True,
        )
    assert [call["shape"][2] for call in assets.calls] == [5]


def test_mixed_video_latent_encoder_preflights_manual_shard_report(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    output_root = tmp_path / "encoded_shard_report"
    report_path = output_root / "shard_reports" / "shard_0001.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="shard report already exists"):
        encoder._preflight_shard_report_path(output_root, shard_index=1, overwrite=False)
    encoder._preflight_shard_report_path(output_root, shard_index=1, overwrite=True)


def test_mixed_video_latent_encoder_rejects_patch_incompatible_training_config(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    experiment_config = load_experiment_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "causal_video_prediction_robotwin_smoke.yaml"
    )
    record = encoder.EncodedEpisode(
        source_id="bad_source",
        dataset_id="bad_dataset",
        episode_index=0,
        clip_id="default",
        latent_path=tmp_path / "bad.pt",
        latent_shape=(48, 5, 15, 20),
        raw_length_frames=17,
        latent_length_frames=5,
        tasks=("bad",),
    )

    with pytest.raises(ValueError, match="patch size"):
        encoder._validate_encoded_records_for_backbone([record], experiment_config=experiment_config)


def test_mixed_video_latent_encoder_shards_and_finalizes_from_sidecars(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "encode_mixed_video_latents.py"
    spec = importlib.util.spec_from_file_location("encode_mixed_video_latents_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}.")
    encoder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = encoder
    try:
        spec.loader.exec_module(encoder)
    finally:
        sys.modules.pop(spec.name, None)

    base_config = _mixed_video_fixture_config(tmp_path)
    config = _with_wan_encoder_horizon(
        base_config,
        video_sources=(base_config.video_sources[0],),
        train_fraction=1.0,
    )
    assets = _FakeLatentEncoderAssets()
    output_root = tmp_path / "encoded_sharded"

    for shard_index in range(2):
        report = encoder.encode_mixed_video_latent_sources(
            data_config=config,
            assets=assets,
            output_root=output_root,
            device=torch.device("cpu"),
            selection=encoder.EncoderSelection(
                split="all",
                source_ids=("source_a",),
                max_episodes=2,
                shard_count=2,
                shard_index=shard_index,
            ),
            chunk_frames=5,
            overwrite=False,
            write_manifests=False,
        )
        assert report["encoded_episodes"] == 1
        assert report["manifest_paths"] == {}
        assert report["config_patch_path"] is None

    assert [call["shape"][2] for call in assets.calls] == [5, 5]

    final_report = encoder.encode_mixed_video_latent_sources(
        data_config=config,
        assets=None,
        output_root=output_root,
        device=torch.device("cpu"),
        split="all",
        source_ids=("source_a",),
        max_episodes=2,
        chunk_frames=5,
        skip_existing=True,
    )
    assert final_report["encoded_episodes"] == 2
    assert final_report["reused_episodes"] == 2
    assert [call["shape"][2] for call in assets.calls] == [5, 5]

    manifest_path = Path(final_report["manifest_paths"]["source_a"])
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    assert {row["episode_index"] for row in rows} == {"0", "1"}


def test_mixed_video_latent_dataset_rejects_rgb_only_sources(tmp_path: Path) -> None:
    config = _mixed_video_fixture_config(tmp_path)

    with pytest.raises(ValueError, match="incompatible with trainer.batch_adapter=latents"):
        build_train_val_latent_datasets(config)


def test_mixed_video_rgb_dataset_rejects_latent_only_sources(tmp_path: Path) -> None:
    base_config = _mixed_video_fixture_config(tmp_path)
    config = _dataclass_replace(
        base_config,
        video_sources=(
            _dataclass_replace(base_config.video_sources[0], source_format=MixedVideoSourceFormat.LATENT),
            base_config.video_sources[1],
        ),
    )

    with pytest.raises(ValueError, match="incompatible with trainer.batch_adapter=views"):
        build_train_val_datasets(config)


def test_mixed_video_train_sampler_balances_sources(tmp_path: Path) -> None:
    config = _mixed_video_fixture_config(tmp_path)
    train_dataset, _ = build_train_val_datasets(config)
    assert isinstance(train_dataset, MixedVideoWindowDataset)

    order = train_dataset.build_epoch_index_order()
    ordered_sources = [
        train_dataset.episode_records[train_dataset.sample_index[index].episode_key].source_id
        for index in order
    ]

    assert {"source_a", "source_b"}.issubset(set(ordered_sources))
    assert ordered_sources[0] != ordered_sources[1]


def test_mixed_video_train_sampler_reshuffles_with_epoch(tmp_path: Path) -> None:
    config = _dataclass_replace(
        _mixed_video_fixture_config(tmp_path),
        sample_stride=1,
        random_mode=MixedVideoRandomMode.GLOBAL,
    )
    train_dataset, _ = build_train_val_datasets(config)
    sampler = train_dataset.build_train_sampler(world_size=1, rank=0)

    epoch_zero = list(iter(sampler))
    sampler.set_epoch(1)
    epoch_one = list(iter(sampler))

    assert sorted(epoch_zero) == sorted(epoch_one)
    assert epoch_zero != epoch_one


def test_mixed_video_train_sampler_pads_to_equal_distributed_rank_lengths(tmp_path: Path) -> None:
    config = _dataclass_replace(
        _mixed_video_fixture_config(tmp_path),
        sample_stride=1,
        random_mode=MixedVideoRandomMode.GLOBAL,
    )
    train_dataset, _ = build_train_val_datasets(config)
    world_size = 6
    base_order = train_dataset.build_epoch_index_order()
    assert len(base_order) % world_size != 0

    rank_orders = [
        list(iter(train_dataset.build_train_sampler(world_size=world_size, rank=rank)))
        for rank in range(world_size)
    ]
    rank_lengths = {len(order) for order in rank_orders}
    expected_rank_length = math.ceil(len(base_order) / world_size)
    flattened = [sample_index for order in rank_orders for sample_index in order]

    assert rank_lengths == {expected_rank_length}
    assert len(flattened) == expected_rank_length * world_size
    assert all(0 <= sample_index < len(train_dataset) for sample_index in flattened)
    assert set(base_order).issubset(set(flattened))


def test_mixed_video_train_sampler_sizes_ranks_from_weighted_epoch_order(tmp_path: Path) -> None:
    config = _mixed_video_fixture_config(tmp_path)
    config = _dataclass_replace(
        config,
        video_sources=(
            config.video_sources[0],
            _dataclass_replace(config.video_sources[1], sampling_weight=2.0),
        ),
        weight_mode=MixedVideoWeightMode.PROPORTIONAL_THEN_MANUAL_SCALE,
    )
    train_dataset, _ = build_train_val_datasets(config)
    base_order = list(train_dataset.build_epoch_index_order(epoch=0))
    world_size = 3
    expected_rank_length = math.ceil(len(base_order) / world_size)
    total_size = expected_rank_length * world_size
    repeats = math.ceil(total_size / len(base_order))
    padded_order = (base_order * repeats)[:total_size]

    rank_orders = [
        list(train_dataset.build_train_sampler(world_size=world_size, rank=rank))
        for rank in range(world_size)
    ]

    assert len(base_order) != len(train_dataset)
    assert rank_orders == [
        padded_order[rank:total_size:world_size]
        for rank in range(world_size)
    ]
