from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

import open_wam.data.lerobot_consortium_report as consortium_report_module
from open_wam.configs import (
    ActionSchemaConfig,
    ActionTargetConfig,
    ConsortiumCacheMode,
    ConsortiumChannelMappingConfig,
    ConsortiumCloudCacheConfig,
    ConsortiumLocalCacheConfig,
    ConsortiumMemberConfig,
    LeRobotConsortiumDataConfig,
    ViewLayoutConfig,
)
from open_wam.data import (
    DatasetLoaderSpec,
    build_lerobot_consortium_catalog,
    build_lerobot_consortium_report,
    build_lerobot_consortium_train_val_datasets,
    collate_wam_samples,
    format_lerobot_consortium_report,
    resolve_dataset_loader_spec,
    resolve_lerobot_consortium_train_val_split,
)
from open_wam.lightning.datamodule import OpenWAMDataModule
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_REAL_LEROBOT_TESTS = os.getenv("OPEN_WAM_RUN_REAL_LEROBOT_TESTS") == "1"
REAL_HETEROGENEOUS_SANITY_SET = (
    (
        "DaivdYuan/exumi-insert-pen-lerobot",
        30.0,
        "v3.0",
        ("observation.images.camera0_rgb",),
        (224, 224, 3),
        7,
        7,
        "parquet_meta",
    ),
    (
        "DaivdYuan/umi-bimanual-dish-washing-lerobot",
        30.0,
        "v3.0",
        ("observation.images.camera0_rgb", "observation.images.camera1_rgb"),
        (224, 224, 3),
        14,
        14,
        "parquet_meta",
    ),
    (
        "lerobot/aloha_static_towel",
        50.0,
        "v3.0",
        (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_low",
            "observation.images.cam_right_wrist",
        ),
        (480, 640, 3),
        14,
        14,
        "parquet_meta",
    ),
    (
        "lerobot/aloha_sim_insertion_human",
        50.0,
        "v3.0",
        ("observation.images.top",),
        (480, 640, 3),
        14,
        14,
        "parquet_meta",
    ),
    (
        "physical-intelligence/libero",
        10.0,
        "v2.0",
        ("image", "wrist_image"),
        (256, 256, 3),
        7,
        8,
        "jsonl_meta",
    ),
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _rgb_bytes(height: int, width: int, rgb: tuple[int, int, int]) -> bytes:
    tensor = torch.zeros(height, width, 3, dtype=torch.uint8)
    tensor[..., 0] = rgb[0]
    tensor[..., 1] = rgb[1]
    tensor[..., 2] = rgb[2]
    from PIL import Image
    from io import BytesIO

    image = Image.fromarray(tensor.numpy(), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_local_lerobot_repo(
    repo_root: Path,
    *,
    channel_specs: dict[str, tuple[int, int, tuple[int, int, int]]],
    fps: int,
    action_dim: int,
    state_dim: int,
    episode_lengths: tuple[int, ...] = (6, 6),
) -> None:
    _write_json(
        repo_root / "meta" / "info.json",
        {
            "codebase_version": "v2.0",
            "fps": fps,
            "chunks_size": 1000,
            "total_episodes": len(episode_lengths),
            "total_frames": int(sum(episode_lengths)),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                **{
                    channel_name: {"dtype": "image", "shape": [height, width, 3]}
                    for channel_name, (height, width, _) in channel_specs.items()
                },
                "actions": {"dtype": "float32", "shape": [action_dim]},
                "state": {"dtype": "float32", "shape": [state_dim]},
            },
        },
    )
    _write_jsonl(
        repo_root / "meta" / "episodes.jsonl",
        [
            {"episode_index": episode_index, "length": length, "tasks": [f"task {episode_index}"]}
            for episode_index, length in enumerate(episode_lengths)
        ],
    )
    _write_jsonl(
        repo_root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": "task 0"}, {"task_index": 1, "task": "task 1"}],
    )
    for episode_index, length in enumerate(episode_lengths):
        rows: list[dict[str, object]] = []
        for frame_index in range(length):
            row: dict[str, object] = {
                "frame_index": frame_index,
                "task_index": min(episode_index, 1),
                "actions": [float(frame_index)] * action_dim,
                "state": [float(frame_index)] * state_dim,
            }
            for channel_name, (height, width, rgb) in channel_specs.items():
                row[channel_name] = _rgb_bytes(height, width, rgb)
            rows.append(row)
        table = pa.Table.from_pylist(rows)
        parquet_path = repo_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)


def _make_consortium_config(
    *,
    members: tuple[ConsortiumMemberConfig, ...],
    **overrides,
) -> LeRobotConsortiumDataConfig:
    base_kwargs = dict(
        local_root=None,
        consortium_members=members,
        camera_names=(
            "observation.images.slot0",
            "observation.images.slot1",
        ),
        latent_camera_names=(
            "observation.images.slot0",
            "observation.images.slot1",
        ),
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=32,
                width=32,
            ),
            ViewLayoutConfig(
                source_name="observation.images.slot1",
                canonical_name="observation.images.slot1",
                top=32,
                left=0,
                height=16,
                width=32,
            ),
        ),
        canonical_height=48,
        canonical_width=32,
        num_frames=2,
        frame_stride=1,
        sample_stride=1,
        train_fraction=0.5,
        split_seed=7,
        action_schema=ActionSchemaConfig(action_dim=6, action_horizon=2, state_dim=8, state_horizon=1),
        action_target=ActionTargetConfig(representation="raw", source_key="actions", pose_source_key="state"),
    )
    base_kwargs.update(overrides)
    return LeRobotConsortiumDataConfig(**base_kwargs)


def test_consortium_config_loader_loads_nested_data_config(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    _build_local_lerobot_repo(
        repo_a,
        channel_specs={"cam_high": (12, 12, (255, 0, 0))},
        fps=30,
        action_dim=4,
        state_dim=6,
        episode_lengths=(6,),
    )

    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["name"] = "lerobot_consortium_smoke"
    raw["data"] = {
        "dataset_name": "lerobot_consortium",
        "dataset_type": "lerobot_consortium",
        "camera_names": ["observation.images.slot0", "observation.images.slot1"],
        "latent_camera_names": ["observation.images.slot0", "observation.images.slot1"],
        "view_layout": [
            {"source_name": "observation.images.slot0", "canonical_name": "observation.images.slot0", "top": 0, "left": 0, "height": 32, "width": 32},
            {"source_name": "observation.images.slot1", "canonical_name": "observation.images.slot1", "top": 32, "left": 0, "height": 16, "width": 32},
        ],
        "canonical_height": 48,
        "canonical_width": 32,
        "num_frames": 2,
        "frame_stride": 1,
        "sample_stride": 1,
        "train_fraction": 1.0,
        "channel_selection_mode": "explicit_mapping",
        "channel_mappings": [{"source_name": "cam_high", "target_slot": "observation.images.slot0"}],
        "view_packing_mode": "multicam_as_slots",
        "frame_packing_order": "camera_major",
        "random_mode": "trajectory_global",
        "weight_mode": "proportional_then_manual_scale",
        "sampling_seed": 13,
        "split_mode": "seeded_shuffle_by_episode",
        "local_cache": {"mode": "disabled"},
        "cloud_cache": {"mode": "disabled", "backend": "filesystem"},
        "consortium_members": [
            {"member_id": "repo_a", "local_root": str(repo_a), "sampling_weight": 2.0},
        ],
        "action_schema": {"action_dim": 6, "action_horizon": 2, "state_dim": 8, "state_horizon": 1},
        "action_target": {"representation": "raw", "source_key": "actions", "pose_source_key": "state"},
    }

    config_path = tmp_path / "consortium.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)

    assert isinstance(config.data, LeRobotConsortiumDataConfig)
    assert config.data.dataset_type == "lerobot_consortium"
    assert config.data.channel_selection_mode == "explicit_mapping"
    assert config.data.view_packing_mode == "multicam_as_slots"
    assert config.data.frame_packing_order == "camera_major"
    assert config.data.random_mode == "trajectory_global"
    assert config.data.weight_mode == "proportional_then_manual_scale"
    assert config.data.split_mode == "seeded_shuffle_by_episode"
    assert config.data.local_cache.mode == ConsortiumCacheMode.DISABLED
    assert config.data.cloud_cache.backend == "filesystem"
    assert config.data.consortium_members[0].local_root == str(repo_a)


def test_consortium_catalog_preserves_member_contracts(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _build_local_lerobot_repo(
        repo_a,
        channel_specs={
            "cam_high": (12, 16, (255, 0, 0)),
            "cam_wrist": (8, 8, (0, 255, 0)),
        },
        fps=30,
        action_dim=4,
        state_dim=6,
        episode_lengths=(6, 7),
    )
    _build_local_lerobot_repo(
        repo_b,
        channel_specs={"front": (10, 10, (0, 0, 255))},
        fps=12,
        action_dim=5,
        state_dim=7,
        episode_lengths=(5,),
    )

    config = _make_consortium_config(
        members=(
            ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a), source_group="alpha"),
            ConsortiumMemberConfig(member_id="repo_b", local_root=str(repo_b), source_group="beta"),
        ),
    )
    catalog = build_lerobot_consortium_catalog(config)

    assert [member.member_id for member in catalog.members] == ["repo_a", "repo_b"]
    assert catalog.members[0].observation_fps == 30.0
    assert catalog.members[1].action_fps == 12.0
    assert [channel.source_name for channel in catalog.members[0].visual_channels] == ["cam_high", "cam_wrist"]
    assert catalog.members[1].visual_channels[0].width == 10
    assert catalog.members[0].action_dim == 4
    assert catalog.members[1].state_dim == 7


def test_consortium_split_is_deterministic_and_leak_free(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _build_local_lerobot_repo(repo_a, channel_specs={"cam_high": (12, 12, (255, 0, 0))}, fps=30, action_dim=4, state_dim=6, episode_lengths=(6, 6))
    _build_local_lerobot_repo(repo_b, channel_specs={"front": (12, 12, (0, 0, 255))}, fps=12, action_dim=4, state_dim=6, episode_lengths=(6, 6))

    config = _make_consortium_config(
        members=(
            ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a)),
            ConsortiumMemberConfig(member_id="repo_b", local_root=str(repo_b)),
        ),
        split_mode="hash_by_episode",
        train_fraction=0.5,
    )
    catalog = build_lerobot_consortium_catalog(config)
    split_a = resolve_lerobot_consortium_train_val_split(config, catalog)
    split_b = resolve_lerobot_consortium_train_val_split(config, catalog)

    train_keys_a = {(item.member_id, item.episode_index) for item in split_a.train_episodes}
    val_keys_a = {(item.member_id, item.episode_index) for item in split_a.val_episodes}
    train_keys_b = {(item.member_id, item.episode_index) for item in split_b.train_episodes}
    val_keys_b = {(item.member_id, item.episode_index) for item in split_b.val_episodes}

    assert train_keys_a == train_keys_b
    assert val_keys_a == val_keys_b
    assert train_keys_a.isdisjoint(val_keys_a)


def test_consortium_dataset_explicit_mapping_handles_different_source_names(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _build_local_lerobot_repo(
        repo_a,
        channel_specs={
            "cam_high": (12, 12, (255, 0, 0)),
            "cam_wrist": (8, 8, (0, 255, 0)),
        },
        fps=30,
        action_dim=4,
        state_dim=6,
        episode_lengths=(6,),
    )
    _build_local_lerobot_repo(
        repo_b,
        channel_specs={
            "front": (12, 12, (0, 0, 255)),
            "hand": (8, 8, (255, 255, 0)),
        },
        fps=12,
        action_dim=5,
        state_dim=7,
        episode_lengths=(6,),
    )

    config = _make_consortium_config(
        members=(
            ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a)),
            ConsortiumMemberConfig(member_id="repo_b", local_root=str(repo_b)),
        ),
        train_fraction=1.0,
        channel_selection_mode="explicit_mapping",
        channel_mappings=(
            ConsortiumChannelMappingConfig("cam_high", "observation.images.slot0"),
            ConsortiumChannelMappingConfig("cam_wrist", "observation.images.slot1"),
            ConsortiumChannelMappingConfig("front", "observation.images.slot0"),
            ConsortiumChannelMappingConfig("hand", "observation.images.slot1"),
        ),
    )

    train_dataset, _ = build_lerobot_consortium_train_val_datasets(config)
    repo_b_index = next(index for index, window in enumerate(train_dataset.sample_index) if window.member_id == "repo_b")
    sample = train_dataset[repo_b_index]

    slot0_pixel = tuple(int(value) for value in sample.views["observation.images.slot0"][0, 0, 0].tolist())
    slot1_pixel = tuple(int(value) for value in sample.views["observation.images.slot1"][0, 0, 0].tolist())
    assert slot0_pixel == (0, 0, 255)
    assert slot1_pixel == (255, 255, 0)
    assert sample.metadata["member_id"] == "repo_b"
    assert sample.metadata["resolved_channel_slots"]["observation.images.slot0"] == "front"


def test_consortium_dataset_zero_fills_missing_slots_and_collates(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _build_local_lerobot_repo(
        repo_a,
        channel_specs={
            "cam_high": (12, 12, (255, 0, 0)),
            "cam_wrist": (8, 8, (0, 255, 0)),
        },
        fps=30,
        action_dim=4,
        state_dim=6,
        episode_lengths=(6,),
    )
    _build_local_lerobot_repo(
        repo_b,
        channel_specs={"front": (12, 12, (0, 0, 255))},
        fps=12,
        action_dim=5,
        state_dim=7,
        episode_lengths=(6,),
    )
    config = _make_consortium_config(
        members=(
            ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a)),
            ConsortiumMemberConfig(member_id="repo_b", local_root=str(repo_b)),
        ),
        train_fraction=1.0,
        channel_selection_mode="all_available",
        missing_channel_policy="zero_fill",
    )
    train_dataset, _ = build_lerobot_consortium_train_val_datasets(config)
    repo_b_index = next(index for index, window in enumerate(train_dataset.sample_index) if window.member_id == "repo_b")
    sample = train_dataset[repo_b_index]

    assert tuple(sample.views.keys()) == config.camera_names
    assert sample.actions.shape == (2, 6)
    assert sample.action_mask.shape == (2, 6)
    assert sample.state.shape == (1, 8)
    assert sample.metadata["observation_fps"] == 12.0
    assert torch.count_nonzero(sample.views["observation.images.slot1"]) == 0

    batch = collate_wam_samples([sample, sample])
    assert batch.views["observation.images.slot0"].shape[0] == 2
    assert batch.actions.shape == (2, 2, 6)


def test_consortium_zero_fill_requires_view_layout_for_missing_slot(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    _build_local_lerobot_repo(
        repo_a,
        channel_specs={"cam_high": (12, 12, (255, 0, 0))},
        fps=30,
        action_dim=4,
        state_dim=6,
        episode_lengths=(6,),
    )
    config = _make_consortium_config(
        members=(ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a)),),
        train_fraction=1.0,
        channel_selection_mode="all_available",
        missing_channel_policy="zero_fill",
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=32,
                width=32,
            ),
        ),
    )
    train_dataset, _ = build_lerobot_consortium_train_val_datasets(config)

    with pytest.raises(ValueError, match="Missing view layout configuration for consortium slot"):
        _ = train_dataset[0]


def test_consortium_multicam_as_frames_requires_single_slot_layout() -> None:
    with pytest.raises(ValueError, match="exactly one `camera_names` slot"):
        _make_consortium_config(
            members=(ConsortiumMemberConfig(member_id="repo_a", local_root="/tmp/repo_a"),),
            view_packing_mode="multicam_as_frames",
        )


def test_consortium_multicam_as_frames_builds_camera_local_pseudo_trajectories(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    _build_local_lerobot_repo(
        repo_a,
        channel_specs={
            "cam_high": (12, 12, (255, 0, 0)),
            "cam_wrist": (8, 8, (0, 255, 0)),
        },
        fps=30,
        action_dim=4,
        state_dim=6,
        episode_lengths=(6,),
    )
    config = _make_consortium_config(
        members=(ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a)),),
        camera_names=("observation.images.slot0",),
        latent_camera_names=("observation.images.slot0",),
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=32,
                width=32,
            ),
        ),
        train_fraction=1.0,
        view_packing_mode="multicam_as_frames",
        frame_packing_order="camera_major",
        channel_selection_mode="all_available",
        missing_channel_policy="error",
    )

    train_dataset, _ = build_lerobot_consortium_train_val_datasets(config)

    assert len(train_dataset) == 8
    assert {window.source_camera_name for window in train_dataset.sample_index} == {"cam_high", "cam_wrist"}
    assert all(len(window.channel_selections) == 1 for window in train_dataset.sample_index)
    assert all(
        window.channel_selections[0].target_slot == "observation.images.slot0"
        for window in train_dataset.sample_index
    )

    high_index = next(index for index, window in enumerate(train_dataset.sample_index) if window.source_camera_name == "cam_high")
    wrist_index = next(index for index, window in enumerate(train_dataset.sample_index) if window.source_camera_name == "cam_wrist")
    high_sample = train_dataset[high_index]
    wrist_sample = train_dataset[wrist_index]

    assert tuple(high_sample.views.keys()) == ("observation.images.slot0",)
    assert high_sample.metadata["source_camera_name"] == "cam_high"
    assert high_sample.metadata["view_packing_mode"] == "multicam_as_frames"
    assert high_sample.metadata["resolved_channel_slots"]["observation.images.slot0"] == "cam_high"
    assert tuple(int(value) for value in high_sample.views["observation.images.slot0"][0, 0, 0].tolist()) == (255, 0, 0)

    assert wrist_sample.metadata["source_camera_name"] == "cam_wrist"
    assert wrist_sample.metadata["resolved_channel_slots"]["observation.images.slot0"] == "cam_wrist"
    assert tuple(int(value) for value in wrist_sample.views["observation.images.slot0"][0, 0, 0].tolist()) == (0, 255, 0)


def test_consortium_train_sampler_respects_weight_and_random_modes(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _build_local_lerobot_repo(repo_a, channel_specs={"cam_high": (12, 12, (255, 0, 0))}, fps=30, action_dim=4, state_dim=6, episode_lengths=(6, 6))
    _build_local_lerobot_repo(repo_b, channel_specs={"front": (12, 12, (0, 0, 255))}, fps=12, action_dim=4, state_dim=6, episode_lengths=(6,))
    config = _make_consortium_config(
        members=(
            ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a), sampling_weight=1.0),
            ConsortiumMemberConfig(member_id="repo_b", local_root=str(repo_b), sampling_weight=3.0),
        ),
        train_fraction=1.0,
        random_mode="trajectory_global",
        weight_mode="proportional_then_manual_scale",
        sampling_seed=11,
    )
    train_dataset, _ = build_lerobot_consortium_train_val_datasets(config)
    loader_spec = resolve_dataset_loader_spec(train_dataset, split="train")

    assert loader_spec.sampler is not None
    assert loader_spec.shuffle is False

    epoch0 = train_dataset.build_epoch_index_order(epoch=0)
    epoch0_repeat = train_dataset.build_epoch_index_order(epoch=0)
    epoch1 = train_dataset.build_epoch_index_order(epoch=1)

    assert epoch0 == epoch0_repeat
    assert epoch0 != epoch1

    member_counts = Counter(train_dataset.sample_index[index].member_id for index in epoch0)
    assert member_counts["repo_b"] > 4


def test_consortium_optional_local_and_cloud_cache_write_through(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    _build_local_lerobot_repo(repo_a, channel_specs={"cam_high": (12, 12, (255, 0, 0))}, fps=30, action_dim=4, state_dim=6, episode_lengths=(6,))

    no_cache_config = _make_consortium_config(
        members=(ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a)),),
        train_fraction=1.0,
    )
    build_lerobot_consortium_catalog(no_cache_config)
    assert not (tmp_path / "local_cache").exists()
    assert not (tmp_path / "cloud_cache").exists()

    cache_config = replace(
        no_cache_config,
        local_cache=ConsortiumLocalCacheConfig(mode="write_through", root=str(tmp_path / "local_cache")),
        cloud_cache=ConsortiumCloudCacheConfig(mode="write_through", backend="filesystem", root=str(tmp_path / "cloud_cache")),
    )
    build_lerobot_consortium_catalog(cache_config)

    assert (tmp_path / "local_cache" / "repo_a" / "meta" / "info.json").exists()
    assert (tmp_path / "cloud_cache" / "repo_a" / "meta" / "episodes.jsonl").exists()


def test_consortium_report_summarizes_members_splits_and_previews(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    _build_local_lerobot_repo(
        repo_a,
        channel_specs={
            "cam_high": (12, 12, (255, 0, 0)),
            "cam_wrist": (8, 8, (0, 255, 0)),
        },
        fps=30,
        action_dim=4,
        state_dim=6,
        episode_lengths=(6, 6),
    )
    _build_local_lerobot_repo(
        repo_b,
        channel_specs={"front": (12, 12, (0, 0, 255))},
        fps=12,
        action_dim=5,
        state_dim=7,
        episode_lengths=(6,),
    )
    config = _make_consortium_config(
        members=(
            ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a), source_group="alpha"),
            ConsortiumMemberConfig(member_id="repo_b", local_root=str(repo_b), source_group="beta"),
        ),
        train_fraction=1.0,
        random_mode="trajectory_global",
        weight_mode="proportional_then_manual_scale",
        sampling_seed=11,
        channel_selection_mode="explicit_mapping",
        channel_mappings=(
            ConsortiumChannelMappingConfig("cam_high", "observation.images.slot0"),
            ConsortiumChannelMappingConfig("cam_wrist", "observation.images.slot1"),
            ConsortiumChannelMappingConfig("front", "observation.images.slot0"),
        ),
    )

    report = build_lerobot_consortium_report(config, preview_count=2, sampler_preview_count=5)
    text_report = format_lerobot_consortium_report(report)

    assert report["catalog"]["member_count"] == 2
    assert [member["member_id"] for member in report["catalog"]["members"]] == ["repo_a", "repo_b"]
    assert report["splits"]["train"]["window_count"] == 12
    assert len(report["splits"]["train"]["sample_previews"]) == 2
    assert len(report["splits"]["train"]["sampler_epoch0_preview"]) == 5
    assert report["splits"]["train"]["sample_previews"][0]["member_id"] in {"repo_a", "repo_b"}
    assert "LeRobot Consortium Report" in text_report
    assert "repo_a" in text_report
    assert "repo_b" in text_report


def test_consortium_report_reuses_precomputed_catalog_and_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_a = tmp_path / "repo_a"
    _build_local_lerobot_repo(
        repo_a,
        channel_specs={"cam_high": (12, 12, (255, 0, 0))},
        fps=30,
        action_dim=4,
        state_dim=6,
        episode_lengths=(6,),
    )
    config = _make_consortium_config(
        members=(ConsortiumMemberConfig(member_id="repo_a", local_root=str(repo_a)),),
        train_fraction=1.0,
    )
    expected_catalog = build_lerobot_consortium_catalog(config)
    expected_split = resolve_lerobot_consortium_train_val_split(config, expected_catalog)
    real_builder = consortium_report_module.build_lerobot_consortium_train_val_datasets

    def _wrapped_builder(data_config, *, catalog=None, split=None):
        assert catalog is expected_catalog
        assert split is expected_split
        return real_builder(data_config, catalog=catalog, split=split)

    monkeypatch.setattr(
        consortium_report_module,
        "build_lerobot_consortium_catalog",
        lambda data_config: expected_catalog,
    )
    monkeypatch.setattr(
        consortium_report_module,
        "resolve_lerobot_consortium_train_val_split",
        lambda data_config, catalog: expected_split,
    )
    monkeypatch.setattr(
        consortium_report_module,
        "build_lerobot_consortium_train_val_datasets",
        _wrapped_builder,
    )

    report = build_lerobot_consortium_report(config, preview_count=1, sampler_preview_count=2)

    assert report["catalog"]["member_count"] == 1
    assert report["splits"]["train"]["window_count"] > 0


def test_datamodule_passes_trainer_world_size_and_rank_to_loader_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        datamodule = OpenWAMDataModule(
            _make_consortium_config(
                members=(ConsortiumMemberConfig(member_id="repo_a", local_root="/tmp/repo_a"),),
            )
        )
    except ImportError:
        pytest.skip("Lightning is not installed in this test environment.")

    class _DummyDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            raise IndexError(index)

    captured: dict[str, int | str] = {}

    def _fake_resolve(dataset, *, split: str, world_size: int = 1, rank: int = 0) -> DatasetLoaderSpec:
        captured["split"] = split
        captured["world_size"] = world_size
        captured["rank"] = rank
        return DatasetLoaderSpec(sampler=None, shuffle=True)

    monkeypatch.setattr("open_wam.lightning.datamodule.resolve_dataset_loader_spec", _fake_resolve)
    datamodule.train_dataset = _DummyDataset()
    datamodule._trainer = SimpleNamespace(world_size=4, global_rank=2)

    _ = datamodule.train_dataloader()

    assert captured == {"split": "train", "world_size": 4, "rank": 2}


@pytest.mark.skipif(
    not RUN_REAL_LEROBOT_TESTS,
    reason="Set OPEN_WAM_RUN_REAL_LEROBOT_TESTS=1 to run public-HF LeRobot integration tests.",
)
def test_consortium_loader_reads_real_public_lerobot_repo(tmp_path: Path) -> None:
    config = LeRobotConsortiumDataConfig(
        repo_id=None,
        local_root=None,
        cache_dir=str(tmp_path / "hf_cache"),
        consortium_members=(
            ConsortiumMemberConfig(
                member_id="libero",
                repo_id="physical-intelligence/libero",
                source_group="official_lerobot",
            ),
        ),
        camera_names=("observation.images.slot0", "observation.images.slot1"),
        latent_camera_names=("observation.images.slot0", "observation.images.slot1"),
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.slot0",
                canonical_name="observation.images.slot0",
                top=0,
                left=0,
                height=256,
                width=256,
            ),
            ViewLayoutConfig(
                source_name="observation.images.slot1",
                canonical_name="observation.images.slot1",
                top=256,
                left=0,
                height=256,
                width=256,
            ),
        ),
        canonical_height=512,
        canonical_width=256,
        num_frames=2,
        frame_stride=1,
        sample_stride=1,
        train_fraction=0.95,
        max_train_episodes=1,
        max_val_episodes=1,
        action_schema=ActionSchemaConfig(action_dim=7, action_horizon=2, state_dim=8, state_horizon=1),
        action_target=ActionTargetConfig(representation="raw", source_key="actions", pose_source_key="state"),
        channel_selection_mode="explicit_mapping",
        channel_mappings=(
            ConsortiumChannelMappingConfig("image", "observation.images.slot0"),
            ConsortiumChannelMappingConfig("wrist_image", "observation.images.slot1"),
        ),
        random_mode="none",
        local_cache=ConsortiumLocalCacheConfig(mode="disabled"),
        cloud_cache=ConsortiumCloudCacheConfig(mode="disabled", backend="filesystem"),
    )

    catalog = build_lerobot_consortium_catalog(config)
    assert len(catalog.members) == 1
    member = catalog.members[0]
    assert member.repo_id == "physical-intelligence/libero"
    assert member.observation_fps == 10.0
    assert member.action_fps == 10.0
    assert member.action_dim == 7
    assert member.state_dim == 8
    assert [channel.source_name for channel in member.visual_channels] == ["image", "wrist_image"]

    split = resolve_lerobot_consortium_train_val_split(config, catalog)
    assert len(split.train_episodes) == 1
    assert len(split.val_episodes) == 1
    assert {
        (episode.member_id, episode.episode_index) for episode in split.train_episodes
    }.isdisjoint(
        {
            (episode.member_id, episode.episode_index) for episode in split.val_episodes
        }
    )

    train_dataset, val_dataset = build_lerobot_consortium_train_val_datasets(config)
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0

    sample = train_dataset[0]
    assert tuple(sample.views.keys()) == ("observation.images.slot0", "observation.images.slot1")
    assert sample.views["observation.images.slot0"].shape == (2, 256, 256, 3)
    assert sample.views["observation.images.slot1"].shape == (2, 256, 256, 3)
    assert sample.actions.shape == (2, 7)
    assert sample.state.shape == (1, 8)
    assert sample.metadata["repo_id"] == "physical-intelligence/libero"
    assert sample.metadata["resolved_channel_slots"]["observation.images.slot0"] == "image"
    assert sample.metadata["resolved_channel_slots"]["observation.images.slot1"] == "wrist_image"
    assert sample.metadata["observation_fps"] == 10.0

    val_sample = val_dataset[0]
    assert val_sample.metadata["repo_id"] == "physical-intelligence/libero"
    assert val_sample.actions.shape == (2, 7)


@pytest.mark.skipif(
    not RUN_REAL_LEROBOT_TESTS,
    reason="Set OPEN_WAM_RUN_REAL_LEROBOT_TESTS=1 to run public-HF LeRobot integration tests.",
)
@pytest.mark.parametrize(
    (
        "repo_id",
        "expected_fps",
        "expected_version",
        "expected_image_keys",
        "expected_shape",
        "expected_action_dim",
        "expected_state_dim",
        "expected_meta_layout",
    ),
    REAL_HETEROGENEOUS_SANITY_SET,
)
def test_real_hf_heterogeneous_sanity_set_matches_expected_repo_surface(
    repo_id: str,
    expected_fps: float,
    expected_version: str,
    expected_image_keys: tuple[str, ...],
    expected_shape: tuple[int, int, int],
    expected_action_dim: int,
    expected_state_dim: int,
    expected_meta_layout: str,
) -> None:
    info_path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")
    with open(info_path, "r", encoding="utf-8") as handle:
        info = json.load(handle)

    features = info["features"]
    image_keys = tuple(
        key
        for key, value in features.items()
        if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
    )
    assert float(info["fps"]) == expected_fps
    assert info["codebase_version"] == expected_version
    assert image_keys == expected_image_keys
    assert tuple(features[expected_image_keys[0]]["shape"]) == expected_shape

    action_key = "actions" if "actions" in features else "action"
    state_key = "state" if "state" in features else "observation.state"
    assert int(features[action_key]["shape"][0]) == expected_action_dim
    assert int(features[state_key]["shape"][0]) == expected_state_dim

    repo_files = set(list_repo_files(repo_id=repo_id, repo_type="dataset"))
    if expected_meta_layout == "jsonl_meta":
        assert "meta/episodes.jsonl" in repo_files
        assert "meta/tasks.jsonl" in repo_files
        assert "meta/episodes/chunk-000/file-000.parquet" not in repo_files
        assert "meta/tasks.parquet" not in repo_files
    else:
        assert "meta/episodes/chunk-000/file-000.parquet" in repo_files
        assert "meta/tasks.parquet" in repo_files
        assert "meta/episodes.jsonl" not in repo_files
        assert "meta/tasks.jsonl" not in repo_files
