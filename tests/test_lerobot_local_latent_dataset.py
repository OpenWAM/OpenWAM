from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import pickle
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from open_wam.configs import (
    DataSplit,
    ReplayStatusPolicy,
    RolloutContextPolicy,
    SampleOrderMode,
    SampleStateAnchorMode,
    SampleWeightMode,
    SampleTargetAlignment,
    SegmentContextPolicy,
    WindowSamplingMode,
)
from open_wam.data import (
    LatentCausalPrefixSuffixCandidate as PublicCausalCandidate,
    LatentCausalPrefixSuffixWindowPlan as PublicCausalWindowPlan,
    LatentCausalPrefixSuffixWindowPlanner as PublicCausalWindowPlanner,
    LocalLatentHierarchicalSampleKey as PublicHierarchicalSampleKey,
    LocalLatentHierarchicalSegmentPlan as PublicHierarchicalSegmentPlan,
    LocalLatentRepository as PublicLatentRepository,
    LocalEpisodeWindow as PublicLocalEpisodeWindow,
    LocalRepoBundle as PublicRepoBundle,
    LocalLatentSampleConditioning as PublicSampleConditioning,
    LocalLatentSampleSource as PublicSampleSource,
    LocalLatentSampleSourceLoader as PublicSampleSourceLoader,
    LocalLatentSegment as PublicLocalLatentSegment,
    LocalLatentSegmentAssembler as PublicLocalLatentSegmentAssembler,
    LocalLatentTrainValWindowPlan as PublicTrainValWindowPlan,
    LocalLatentTrainValWindowPlanner as PublicTrainValWindowPlanner,
    LocalLatentUniformSegmentSamplingPlan as PublicUniformSamplingPlan,
    LocalLatentWindowWeightPlan as PublicWindowWeightPlan,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    discover_local_lerobot_repo_bundles,
)
from open_wam.data.lerobot_v2_latent import (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET
    as LegacyConditionSourceFramePolicy,
    LocalEpisodeWindow as LegacyLocalEpisodeWindow,
    LocalLatentEpochOrderSampler,
    LocalLatentWeightedTrainSampler,
    assemble_canonical_latents as legacy_assemble_canonical_latents,
    condition_latent_offset_mismatches as legacy_condition_offset_mismatches,
    discover_local_lerobot_repo_bundles as legacy_discover_repo_bundles,
    latent_filename as legacy_latent_filename,
    load_empty_text_embedding as legacy_load_empty_text_embedding,
    reshape_latent_payload as legacy_reshape_latent_payload,
    resolve_latent_root as legacy_resolve_latent_root,
    scan_local_latent_windows as legacy_scan_local_latent_windows,
)
from open_wam.data.latent_causal_sampling import (
    LatentCausalPrefixSuffixCandidate,
    LatentCausalPrefixSuffixWindowPlan,
    LatentCausalPrefixSuffixWindowPlanner,
)
from open_wam.data.latent_hierarchical_sampling import (
    LocalLatentHierarchicalSampleKey,
    LocalLatentHierarchicalSegmentPlan,
)
from open_wam.data.latent_temporal import (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
)
from open_wam.data.lerobot_v2_latent_storage import (
    LocalEpisodeWindow,
    LocalLatentRepository,
    LocalRepoBundle,
    assemble_canonical_latents,
    condition_latent_offset_mismatches,
    discover_local_lerobot_repo_bundles as discover_storage_repo_bundles,
    latent_filename,
    load_empty_text_embedding,
    reshape_latent_payload,
    resolve_latent_root,
    scan_local_latent_windows,
)
from open_wam.data.lerobot_v2_latent_sampling import (
    LocalLatentUniformSegmentSamplingPlan,
    LocalLatentWindowWeightPlan,
)
from open_wam.data.lerobot_v2_latent_segment import (
    LocalLatentSegment,
    LocalLatentSegmentAssembler,
)
from open_wam.data.lerobot_v2_latent_source import (
    LocalLatentSampleConditioning,
    LocalLatentSampleSource,
    LocalLatentSampleSourceLoader,
)
from open_wam.data.lerobot_v2_latent_split import (
    LocalLatentTrainValWindowPlan,
    LocalLatentTrainValWindowPlanner,
)
from open_wam.data.lerobot_v2_latent_supervision import (
    LocalLatentSupervisionAssembler,
)
from open_wam.training import TrainingRuntime
from open_wam.configs import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lerobot_latent_dataset_facade_exports_canonical_role_objects() -> None:
    from open_wam import data as public_data
    from open_wam.data import lerobot_v2_latent as facade
    from open_wam.data import lerobot_v2_latent_base_dataset as base
    from open_wam.data import lerobot_v2_latent_causal_dataset as causal
    from open_wam.data import lerobot_v2_latent_factory as factory
    from open_wam.data import (
        lerobot_v2_latent_hierarchical_dataset as hierarchical,
    )
    from open_wam.data import lerobot_v2_latent_uniform_dataset as uniform

    owners = {
        base: (
            "FullSegmentLocalLeRobotLatentDataset",
            "LocalLeRobotLatentWindowDataset",
        ),
        causal: ("CausalPrefixSuffixLocalLeRobotLatentDataset",),
        factory: ("build_local_lerobot_latent_train_val_datasets",),
        hierarchical: (
            "HierarchicalFixedSegmentLocalLeRobotLatentDataset",
        ),
        uniform: ("UniformSegmentLocalLeRobotLatentDataset",),
    }

    for owner, names in owners.items():
        for name in names:
            assert getattr(facade, name) is getattr(owner, name)
            legacy_global = (
                "copen_wam.data.lerobot_v2_latent\n" f"{name}\n."
            ).encode("ascii")
            assert pickle.loads(legacy_global) is getattr(owner, name)

    assert (
        public_data.LocalLeRobotLatentWindowDataset
        is base.LocalLeRobotLatentWindowDataset
    )
    assert (
        public_data.build_local_lerobot_latent_train_val_datasets
        is factory.build_local_lerobot_latent_train_val_datasets
    )
    assert issubclass(
        hierarchical.HierarchicalFixedSegmentLocalLeRobotLatentDataset,
        uniform.UniformSegmentLocalLeRobotLatentDataset,
    )
    assert issubclass(
        uniform.UniformSegmentLocalLeRobotLatentDataset,
        base.LocalLeRobotLatentWindowDataset,
    )
    assert issubclass(
        causal.CausalPrefixSuffixLocalLeRobotLatentDataset,
        base.LocalLeRobotLatentWindowDataset,
    )

    wildcard_namespace: dict[str, object] = {}
    exec(
        "from open_wam.data.lerobot_v2_latent import *",
        wildcard_namespace,
    )
    assert len(facade.__all__) == 58
    assert set(wildcard_namespace) - {"__builtins__"} == set(facade.__all__)


def test_lerobot_latent_sampling_facade_exports_canonical_role_objects() -> None:
    from open_wam.data import lerobot_v2_latent_hierarchical_policy as hierarchical
    from open_wam.data import lerobot_v2_latent_sampler_adapters as adapters
    from open_wam.data import lerobot_v2_latent_sampling as facade
    from open_wam.data import lerobot_v2_latent_uniform_policy as uniform
    from open_wam.data import lerobot_v2_latent_weighting as weighting

    owners = {
        hierarchical: (
            "HierarchicalFixedSegmentSamplingPlan",
            "HierarchicalFixedSegmentTaskSpec",
            "HierarchicalFixedSegmentWindowSpec",
            "build_hierarchical_fixed_segment_task_specs",
        ),
        adapters: (
            "HierarchicalFixedSegmentTrainSampler",
            "LocalLatentEpochOrderSampler",
            "LocalLatentWeightedTrainSampler",
            "_WeightedLocalLatentSource",
        ),
        uniform: ("LocalLatentUniformSegmentSamplingPlan",),
        weighting: (
            "LocalLatentWindowWeightPlan",
            "_build_local_latent_sample_weights",
        ),
    }

    for owner, names in owners.items():
        for name in names:
            assert getattr(facade, name) is getattr(owner, name)
            legacy_global = (
                "copen_wam.data.lerobot_v2_latent_sampling\n"
                f"{name}\n."
            ).encode("ascii")
            assert pickle.loads(legacy_global) is getattr(owner, name)


def test_lerobot_latent_storage_owns_compatibility_exports() -> None:
    assert LegacyLocalEpisodeWindow is LocalEpisodeWindow
    assert PublicLocalEpisodeWindow is LocalEpisodeWindow
    assert PublicLatentRepository is LocalLatentRepository
    assert PublicRepoBundle is LocalRepoBundle
    assert discover_local_lerobot_repo_bundles is discover_storage_repo_bundles
    assert legacy_latent_filename is latent_filename
    assert legacy_assemble_canonical_latents is assemble_canonical_latents
    assert legacy_condition_offset_mismatches is condition_latent_offset_mismatches
    assert legacy_load_empty_text_embedding is load_empty_text_embedding
    assert legacy_reshape_latent_payload is reshape_latent_payload
    assert legacy_resolve_latent_root is resolve_latent_root
    assert (
        LegacyConditionSourceFramePolicy
        is CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET
    )
    assert PublicUniformSamplingPlan is LocalLatentUniformSegmentSamplingPlan
    assert PublicWindowWeightPlan is LocalLatentWindowWeightPlan
    assert PublicSampleConditioning is LocalLatentSampleConditioning
    assert PublicSampleSource is LocalLatentSampleSource
    assert PublicSampleSourceLoader is LocalLatentSampleSourceLoader
    assert PublicLocalLatentSegment is LocalLatentSegment
    assert PublicLocalLatentSegmentAssembler is LocalLatentSegmentAssembler
    assert PublicCausalCandidate is LatentCausalPrefixSuffixCandidate
    assert PublicCausalWindowPlan is LatentCausalPrefixSuffixWindowPlan
    assert PublicCausalWindowPlanner is LatentCausalPrefixSuffixWindowPlanner
    assert PublicHierarchicalSampleKey is LocalLatentHierarchicalSampleKey
    assert PublicHierarchicalSegmentPlan is LocalLatentHierarchicalSegmentPlan
    assert PublicTrainValWindowPlan is LocalLatentTrainValWindowPlan
    assert PublicTrainValWindowPlanner is LocalLatentTrainValWindowPlanner
    assert legacy_discover_repo_bundles is discover_storage_repo_bundles
    assert legacy_scan_local_latent_windows is scan_local_latent_windows


def _disable_replay_status(data_config):
    return replace(
        data_config,
        replay_status_path=None,
        val_replay_status_path=None,
        replay_status_policy=ReplayStatusPolicy.INCLUDE_ALL,
        require_replay_status=False,
        val_replay_status_policy=None,
        val_require_replay_status=False,
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


def _build_local_robotwin_latent_repo(
    repo_root: Path,
    *,
    action_key: str = "action",
    state_key: str = "state",
    action_dim: int = 30,
    state_dim: int = 30,
    camera_names: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    total_rows: int = 20,
    latent_num_frames: int = 4,
    include_condition_latent: bool = False,
) -> None:
    _write_json(
        repo_root / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "fps": 10,
            "chunks_size": 1000,
            "total_episodes": 1,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                action_key: {"dtype": "float32"},
                state_key: {"dtype": "float32"},
            },
        },
    )
    _write_jsonl(
        repo_root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "length": total_rows, "tasks": ["pick up block"]}],
    )
    _write_jsonl(
        repo_root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": "pick up block"}],
    )

    rows = []
    for frame_index in range(total_rows):
        rows.append(
            {
                "frame_index": frame_index,
                "task_index": 0,
                action_key: [float(frame_index)] * action_dim,
                state_key: [float(frame_index)] * state_dim,
            }
        )
    table = pa.Table.from_pylist(rows)
    parquet_path = repo_root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)

    default_latent_specs = {
        "cam_high": (16, 20),
        "cam_left_wrist": (8, 10),
        "cam_right_wrist": (8, 10),
    }
    latent_specs = {
        camera_name: default_latent_specs.get(camera_name, (8, 8))
        for camera_name in camera_names
    }
    for camera_index, (camera_name, (latent_height, latent_width)) in enumerate(latent_specs.items()):
        flat_latents = torch.randn(latent_num_frames * latent_height * latent_width, 48)
        payload = {
            "latent": flat_latents,
            "latent_num_frames": latent_num_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "frame_ids": list(range(latent_num_frames)),
        }
        if include_condition_latent:
            payload["condition_latent"] = torch.full_like(flat_latents, float(10 + camera_index))
        latent_path = (
            repo_root
            / "latents"
            / "chunk-000"
            / camera_name
            / f"episode_000000_0_{latent_num_frames}.pth"
        )
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, latent_path)


def _append_second_latent_episode(repo_root: Path, *, total_rows: int = 20) -> None:
    info_path = repo_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = 2
    _write_json(info_path, info)
    _write_jsonl(
        repo_root / "meta" / "episodes.jsonl",
        [
            {"episode_index": 0, "length": total_rows, "tasks": ["pick up block"]},
            {"episode_index": 1, "length": total_rows, "tasks": ["pick up block"]},
        ],
    )
    original_rows = pq.read_table(repo_root / "data" / "chunk-000" / "episode_000000.parquet").to_pylist()
    pq.write_table(
        pa.Table.from_pylist(original_rows),
        repo_root / "data" / "chunk-000" / "episode_000001.parquet",
    )
    for latent_path in (repo_root / "latents" / "chunk-000").glob("*/episode_000000_*.pth"):
        episode_one_path = latent_path.with_name(latent_path.name.replace("episode_000000", "episode_000001"))
        episode_one_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.load(latent_path, map_location="cpu", weights_only=False), episode_one_path)


def _append_second_latent_window_same_episode(repo_root: Path, *, start_frame: int = 2) -> None:
    for latent_path in (repo_root / "latents" / "chunk-000").glob("*/episode_000000_0_*.pth"):
        payload = dict(torch.load(latent_path, map_location="cpu", weights_only=False))
        latent_num_frames = int(payload["latent_num_frames"])
        payload["frame_ids"] = list(range(start_frame, start_frame + latent_num_frames))
        shifted_path = latent_path.with_name(
            f"episode_000000_{start_frame}_{start_frame + latent_num_frames}.pth"
        )
        torch.save(payload, shifted_path)


def _append_latent_episode(
    repo_root: Path,
    *,
    episode_index: int,
    task_index: int,
    task_text: str,
    total_rows: int,
    latent_num_frames: int,
    action_key: str = "action",
    state_key: str = "state",
    action_dim: int = 30,
    state_dim: int = 30,
    camera_names: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist"),
) -> None:
    info_path = repo_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = max(int(info["total_episodes"]), episode_index + 1)
    _write_json(info_path, info)

    episodes_path = repo_root / "meta" / "episodes.jsonl"
    episode_records = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episode_records = [record for record in episode_records if int(record["episode_index"]) != episode_index]
    episode_records.append({"episode_index": episode_index, "length": total_rows, "tasks": [task_text]})
    _write_jsonl(episodes_path, sorted(episode_records, key=lambda record: int(record["episode_index"])))

    tasks_path = repo_root / "meta" / "tasks.jsonl"
    task_records = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task_records = [record for record in task_records if int(record["task_index"]) != task_index]
    task_records.append({"task_index": task_index, "task": task_text})
    _write_jsonl(tasks_path, sorted(task_records, key=lambda record: int(record["task_index"])))

    rows = []
    for frame_index in range(total_rows):
        rows.append(
            {
                "frame_index": frame_index,
                "task_index": task_index,
                action_key: [float(frame_index)] * action_dim,
                state_key: [float(frame_index)] * state_dim,
            }
        )
    parquet_path = repo_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    default_latent_specs = {
        "cam_high": (16, 20),
        "cam_left_wrist": (8, 10),
        "cam_right_wrist": (8, 10),
    }
    for camera_name in camera_names:
        latent_height, latent_width = default_latent_specs.get(camera_name, (8, 8))
        flat_latents = torch.randn(latent_num_frames * latent_height * latent_width, 48)
        payload = {
            "latent": flat_latents,
            "latent_num_frames": latent_num_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "frame_ids": list(range(latent_num_frames)),
        }
        latent_path = (
            repo_root
            / "latents"
            / "chunk-000"
            / camera_name
            / f"episode_{episode_index:06d}_0_{latent_num_frames}.pth"
        )
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, latent_path)


def test_local_lerobot_latent_dataset_builds_canonical_latents(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
    repository = train_dataset._latent_repository
    assert isinstance(repository, LocalLatentRepository)
    assert train_dataset._repo_bundles is repository.repo_bundles
    assert train_dataset._episode_cache is repository.episode_cache
    assert train_dataset._latent_view_cache is repository.latent_view_cache

    window = train_dataset.windows[0]
    metadata = train_dataset._repo_bundles[str(window.repo_root)].metadata
    rows_from_repository = repository.load_episode_rows(
        window.repo_root,
        window.episode_index,
        metadata,
    )
    rows_from_cache = repository.load_episode_rows(
        window.repo_root,
        window.episode_index,
        metadata,
    )
    assert rows_from_cache is rows_from_repository
    latents_from_repository = repository.load_canonical_window_latents(
        window,
        metadata,
    )
    latents_from_cache = repository.load_canonical_window_latents(
        window,
        metadata,
    )
    assert latents_from_cache is latents_from_repository

    sample = train_dataset[0]

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert sample.video_latents.shape == (48, 4, 24, 20)
    assert sample.actions.shape == (8, 30)
    assert sample.state.shape == (1, 30)
    assert sample.text_context is None
    assert sample.metadata["observed_frame_ids"] == [0, 1, 2, 3]
    assert sample.metadata["observation_start"] == 0
    assert sample.metadata["observation_frame_indices"] == [0, 1, 2, 3]
    assert sample.metadata["valid_action_steps"] == 6
    assert sample.metadata["dataset_mean_valid_action_steps"] == pytest.approx(6.0)


def test_local_latent_supervision_owner_builds_expected_sequences(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_supervision"
    _build_local_robotwin_latent_repo(repo_root)
    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )
    train_dataset, _ = build_train_val_latent_datasets(config.data)
    assembler = train_dataset._supervision_assembler
    assert isinstance(assembler, LocalLatentSupervisionAssembler)
    assert assembler.data_config is train_dataset.data_config

    window = train_dataset.windows[0]
    metadata = train_dataset._repo_bundles[str(window.repo_root)].metadata
    rows = train_dataset._latent_repository.load_episode_rows(
        window.repo_root,
        window.episode_index,
        metadata,
    )
    observed_frame_ids = [0, 1, 2, 3]
    owner_targets = assembler.build_lingbot_window_action_targets(
        rows=rows,
        window=window,
        observed_frame_ids=observed_frame_ids,
        latent_num_frames=4,
        leading_zero_action_frames=1,
        leading_zero_action_mask=0.0,
    )
    assert owner_targets[0].shape == (8, 30)
    assert torch.count_nonzero(owner_targets[0][:2]) == 0
    assert torch.count_nonzero(owner_targets[1][:2]) == 0
    assert torch.all(owner_targets[1][2:6] == 1)
    assert torch.count_nonzero(owner_targets[1][6:]) == 0
    assert owner_targets[2]["lingbot_window_action_alignment"][
        "required_action_num"
    ] == 8

    owner_state = assembler.extract_state_history_at_frame(
        rows=rows,
        anchor_frame_index=3,
        state_horizon=3,
    )
    expected_state = torch.tensor(
        [[float(frame_index)] * 30 for frame_index in (1, 2, 3)]
    )
    torch.testing.assert_close(owner_state[0], expected_state)
    assert torch.all(owner_state[1] == 1)

    owner_proprio = assembler.extract_proprio_context_state_sequence(
        rows=rows,
        observed_frame_ids=observed_frame_ids,
        chunk_size=2,
        loss_frame_start=1,
    )
    expected_proprio = torch.tensor(
        [[float(frame_index)] * 30 for frame_index in (0, 2)]
    )
    torch.testing.assert_close(owner_proprio[0], expected_proprio)
    assert torch.all(owner_proprio[1] == 1)


def test_scan_local_latent_windows_requires_complete_multicamera_latents(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)
    (
        repo_root
        / "latents"
        / "chunk-000"
        / "cam_right_wrist"
        / "episode_000000_0_4.pth"
    ).unlink()

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            camera_names=("cam_high", "cam_left_wrist", "cam_right_wrist"),
            latent_camera_names=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        ),
    )

    assert scan_local_latent_windows(repo_root, config.data) == []


def test_local_lerobot_latent_dataset_filters_failed_replay_status(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)
    _append_second_latent_episode(repo_root)
    _write_jsonl(
        repo_root / "meta" / "replay_status.jsonl",
        [
            {"dataset_episode_index": 0, "replay_status": "success", "simulator": {"mujoco_gl": "osmesa"}},
            {"dataset_episode_index": 1, "replay_status": "failure", "simulator": {"mujoco_gl": "osmesa"}},
        ],
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            replay_status_policy=ReplayStatusPolicy.SUCCESSFUL_ONLY,
            require_replay_status=True,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, val_dataset = build_train_val_latent_datasets(config.data)

    assert {window.episode_index for window in train_dataset.windows} == {0}
    assert {window.episode_index for window in val_dataset.windows} == {0}


def test_local_lerobot_latent_dataset_uses_unused_failed_replay_rows_for_val(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)
    _append_second_latent_episode(repo_root)
    _write_jsonl(
        repo_root / "meta" / "replay_status.jsonl",
        [
            {"dataset_episode_index": 0, "replay_status": "success"},
            {"dataset_episode_index": 1, "replay_status": "failure"},
        ],
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            replay_status_policy=ReplayStatusPolicy.SUCCESSFUL_ONLY,
            require_replay_status=True,
            val_replay_status_policy=ReplayStatusPolicy.FAILURE_ONLY,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, val_dataset = build_train_val_latent_datasets(config.data)

    assert {window.episode_index for window in train_dataset.windows} == {0}
    assert {window.episode_index for window in val_dataset.windows} == {1}


def test_local_latent_train_val_window_planner_matches_dataset_builder(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)
    _append_second_latent_episode(repo_root)
    _write_jsonl(
        repo_root / "meta" / "replay_status.jsonl",
        [
            {"dataset_episode_index": 0, "replay_status": "success"},
            {"dataset_episode_index": 1, "replay_status": "failure"},
        ],
    )

    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )
    data_config = replace(
        _disable_replay_status(config.data),
        dataset_type="lerobot_v2_latent_local",
        local_root=str(repo_root),
        train_fraction=1.0,
        replay_status_policy=ReplayStatusPolicy.SUCCESSFUL_ONLY,
        require_replay_status=True,
        val_replay_status_policy=ReplayStatusPolicy.FAILURE_ONLY,
        num_workers=0,
        train_batch_size=1,
        val_batch_size=1,
    )

    planner = LocalLatentTrainValWindowPlanner(data_config)
    plan = planner.plan()
    train_dataset, val_dataset = build_train_val_latent_datasets(data_config)

    assert isinstance(plan, LocalLatentTrainValWindowPlan)
    assert isinstance(plan.train_windows, tuple)
    assert isinstance(plan.val_windows, tuple)
    assert pickle.loads(pickle.dumps(planner)) == planner
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert plan.train_windows == tuple(train_dataset.windows)
    assert plan.val_windows == tuple(val_dataset.windows)
    assert {window.episode_index for window in plan.train_windows} == {0}
    assert {window.episode_index for window in plan.val_windows} == {1}


def test_val_local_root_dataset_uses_val_split_semantics(tmp_path: Path) -> None:
    train_root = tmp_path / "train_robotwin_local_latent"
    val_root = tmp_path / "val_robotwin_local_latent"
    _build_local_robotwin_latent_repo(train_root)
    _build_local_robotwin_latent_repo(val_root)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(train_root),
            val_local_root=str(val_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, val_dataset = build_train_val_latent_datasets(config.data)

    assert train_dataset.data_config.split == DataSplit.TRAIN
    assert val_dataset.data_config.split == DataSplit.VAL
    assert {window.repo_root for window in train_dataset.windows} == {train_root}
    assert {window.repo_root for window in val_dataset.windows} == {val_root}


def test_val_local_root_uses_val_root_replay_status_when_train_path_is_absolute(tmp_path: Path) -> None:
    train_root = tmp_path / "train_robotwin_local_latent"
    val_root = tmp_path / "val_robotwin_local_latent"
    _build_local_robotwin_latent_repo(train_root)
    _build_local_robotwin_latent_repo(val_root)
    _write_jsonl(
        train_root / "meta" / "replay_status.jsonl",
        [{"dataset_episode_index": 0, "replay_status": "success"}],
    )
    _write_jsonl(
        val_root / "meta" / "replay_status.jsonl",
        [{"dataset_episode_index": 0, "replay_status": "failure"}],
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(train_root),
            val_local_root=str(val_root),
            replay_status_path=str(train_root / "meta" / "replay_status.jsonl"),
            replay_status_policy=ReplayStatusPolicy.SUCCESSFUL_ONLY,
            val_replay_status_policy=ReplayStatusPolicy.FAILURE_ONLY,
            require_replay_status=True,
            val_require_replay_status=True,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, val_dataset = build_train_val_latent_datasets(config.data)

    assert {window.repo_root for window in train_dataset.windows} == {train_root}
    assert {window.episode_index for window in train_dataset.windows} == {0}
    assert {window.repo_root for window in val_dataset.windows} == {val_root}
    assert {window.episode_index for window in val_dataset.windows} == {0}


def test_local_lerobot_latent_dataset_weights_long_depleted_tasks(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)
    _append_second_latent_episode(repo_root)
    _append_latent_episode(
        repo_root,
        episode_index=2,
        task_index=1,
        task_text="assemble the long task",
        total_rows=40,
        latent_num_frames=40,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                sample_weight_mode=SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    weight_plan = train_dataset._window_weight_plan
    weights_by_episode = {
        window.episode_index: train_dataset.sample_weights[index]
        for index, window in enumerate(train_dataset.windows)
    }

    assert isinstance(weight_plan, LocalLatentWindowWeightPlan)
    assert train_dataset._window_valid_action_steps is (
        weight_plan.window_valid_action_steps
    )
    assert train_dataset._window_task_texts is weight_plan.window_task_texts
    assert train_dataset._task_demo_counts is weight_plan.task_demo_counts
    assert train_dataset.sample_weights is weight_plan.sample_weights
    assert train_dataset.task_text_for_window_index(2) == "assemble the long task"
    assert weight_plan.task_text_for_window_index(2) == "assemble the long task"
    sample_source = train_dataset._load_sample_source(
        train_dataset.windows[2],
        include_condition_latents=False,
    )
    conditioning = sample_source.conditioning_for_frame(
        0,
        empty_text_embedding=train_dataset.empty_text_embedding,
    )
    assert isinstance(sample_source, LocalLatentSampleSource)
    assert isinstance(conditioning, LocalLatentSampleConditioning)
    assert sample_source.repo_bundle is train_dataset._repo_bundles[
        str(train_dataset.windows[2].repo_root)
    ]
    assert conditioning.task_index == 1
    assert conditioning.task_text == "assemble the long task"
    assert conditioning.text_context is None
    assert conditioning.negative_text_context is None
    restored_source = pickle.loads(pickle.dumps(sample_source))
    restored_conditioning = restored_source.conditioning_for_frame(
        0,
        empty_text_embedding=None,
    )
    assert restored_source.raw_frame_ids == sample_source.raw_frame_ids
    torch.testing.assert_close(
        restored_source.video_latents,
        sample_source.video_latents,
    )
    assert restored_conditioning.task_index == conditioning.task_index
    assert restored_conditioning.task_text == conditioning.task_text
    assert restored_conditioning.text_context is None
    assert restored_conditioning.negative_text_context is None
    assert train_dataset.dataset_mean_valid_action_steps == pytest.approx(18.0)
    assert train_dataset.dataset_mean_task_demo_count == pytest.approx(1.5)
    assert weights_by_episode[0] == pytest.approx(0.25)
    assert weights_by_episode[1] == pytest.approx(0.25)
    assert weights_by_episode[2] == pytest.approx(3.5)
    assert weights_by_episode[2] > weights_by_episode[0]

    sample = train_dataset[2]
    assert sample.metadata["train_sample_weight"] == pytest.approx(weights_by_episode[2])
    assert sample.metadata["eligible_task_demo_count"] == 1
    assert sample.metadata["dataset_mean_eligible_task_demo_count"] == pytest.approx(1.5)
    restored_plan = pickle.loads(pickle.dumps(weight_plan))
    assert restored_plan.sample_weights == weight_plan.sample_weights
    assert restored_plan.sample_weight_metadata(2) == (
        weight_plan.sample_weight_metadata(2)
    )


def test_inverse_task_demo_count_counts_unique_demos_not_windows(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)
    _append_second_latent_window_same_episode(repo_root)
    _append_latent_episode(
        repo_root,
        episode_index=1,
        task_index=1,
        task_text="assemble the other task",
        total_rows=20,
        latent_num_frames=4,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                sample_weight_mode=SampleWeightMode.INVERSE_TASK_DEMO_COUNT,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)

    window_keys = sorted((window.episode_index, window.start_frame) for window in train_dataset.windows)
    assert window_keys == [(0, 0), (0, 2), (1, 0)]
    assert train_dataset.dataset_mean_task_demo_count == pytest.approx(1.0)
    assert train_dataset.sample_weights == pytest.approx((1.0, 1.0, 1.0))

    for index in range(len(train_dataset)):
        sample = train_dataset[index]
        assert sample.metadata["eligible_task_demo_count"] == 1
        assert sample.metadata["dataset_mean_eligible_task_demo_count"] == pytest.approx(1.0)


def test_local_lerobot_latent_weighted_sampler_shards_with_replacement(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)
    _append_second_latent_episode(repo_root)
    _append_latent_episode(
        repo_root,
        episode_index=2,
        task_index=1,
        task_text="assemble the long task",
        total_rows=40,
        latent_num_frames=40,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=123,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                sample_weight_mode=SampleWeightMode.INVERSE_TASK_DEMO_COUNT,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sampler = train_dataset.build_train_sampler(world_size=2, rank=1)

    assert sampler is not None
    assert len(sampler) == 2
    assert all(0 <= index < len(train_dataset) for index in list(sampler))
    sampler.set_epoch(7)
    assert all(0 <= index < len(train_dataset) for index in list(sampler))


def test_uniform_segment_sampling_pads_tail_with_zero_order_hold(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment"
    _build_local_robotwin_latent_repo(repo_root, total_rows=6, latent_num_frames=6)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=4,
                segment_max_frames=4,
                segment_length_stride=1,
                chunk_size=2,
                window_size=4,
                randomize_geometry=False,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[5]

    assert len(train_dataset) == 6
    assert sample.video_latents.shape == (48, 4, 24, 20)
    assert sample.actions.shape == (8, 30)
    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.UNIFORM_SEGMENT
    assert sample.metadata["subwindow_latent_start"] == 5
    assert sample.metadata["subwindow_latent_end"] == 9
    assert sample.metadata["segment_length_frames"] == 4
    assert sample.metadata["segment_valid_latent_frames"] == 1
    assert sample.metadata["segment_padded_latent_frames"] == 3
    assert sample.metadata["tail_padding_mode"] == "zero_hold"
    assert sample.metadata["latent_loss_frame_start"] == 0
    assert sample.metadata["latent_loss_frame_end"] == 1
    assert sample.metadata["sampled_chunk_size"] == 2
    assert sample.metadata["sampled_window_size"] == 4
    assert sample.metadata["history_frames"] == 2
    assert sample.metadata["observed_frame_ids"] == [5, 5, 5, 5]
    assert torch.equal(sample.video_latents[:, 0], sample.video_latents[:, 1])
    assert torch.equal(sample.video_latents[:, 1], sample.video_latents[:, 2])
    assert sample.metadata["valid_action_steps"] == 3


def test_uniform_segment_randomizes_attention_geometry(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_random_geometry"
    _build_local_robotwin_latent_repo(repo_root, total_rows=32, latent_num_frames=32)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=16,
                segment_max_frames=16,
                segment_length_stride=1,
                chunk_size=4,
                window_size=8,
                randomize_geometry=True,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    random.seed(0)
    samples = [train_dataset[index % len(train_dataset)] for index in range(64)]
    chunk_sizes = {int(sample.metadata["sampled_chunk_size"]) for sample in samples}
    window_sizes = {int(sample.metadata["sampled_window_size"]) for sample in samples}

    assert chunk_sizes <= {1, 2, 3, 4}
    assert len(chunk_sizes) > 1
    assert all(4 <= window_size <= 8 for window_size in window_sizes)
    assert len(window_sizes) > 1
    for sample in samples:
        segment_length = int(sample.metadata["segment_length_frames"])
        sampled_chunk_size = int(sample.metadata["sampled_chunk_size"])
        sampled_window_size = int(sample.metadata["sampled_window_size"])
        expected_history_frames = max(
            1,
            min(
                int(math.ceil(sampled_window_size / 2.0)) * sampled_chunk_size,
                max(1, segment_length - sampled_chunk_size),
            ),
        )
        assert segment_length == 16
        assert sample.metadata["history_frames"] == expected_history_frames


def test_uniform_segment_randomizes_start_and_requires_full_segment(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_random_full"
    _build_local_robotwin_latent_repo(repo_root, total_rows=8, latent_num_frames=8)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=2,
                segment_max_frames=4,
                segment_length_stride=1,
                randomize_segment_length=True,
                randomize_segment_start=True,
                require_full_segment=True,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    samples = [train_dataset[0] for _ in range(12)]
    starts = {int(sample.metadata["subwindow_latent_start"]) for sample in samples}
    lengths = {int(sample.metadata["segment_length_frames"]) for sample in samples}

    assert len(starts) > 1
    assert len(lengths) > 1
    for sample in samples:
        start = int(sample.metadata["subwindow_latent_start"])
        length = int(sample.metadata["segment_length_frames"])
        assert start + length <= 8
        assert sample.metadata["segment_valid_latent_frames"] == length
        assert sample.metadata["segment_padded_latent_frames"] == 0
        assert sample.metadata["tail_padding_mode"] == "none"


def test_uniform_segment_preserves_optional_condition_latents(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_condition_latents"
    _build_local_robotwin_latent_repo(
        repo_root,
        total_rows=6,
        latent_num_frames=6,
        include_condition_latent=True,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=4,
                segment_max_frames=4,
                segment_length_stride=1,
                randomize_segment_start=False,
                require_full_segment=True,
                state_anchor_mode=SampleStateAnchorMode.SAMPLE_START_FRAME,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[1]

    assert sample.condition_latents is not None
    assert sample.condition_latents.shape == sample.video_latents.shape
    assert sample.metadata["has_condition_latents"] is True
    assert sample.metadata["condition_latent_layout"]
    assert sample.state[0, 0].item() == pytest.approx(float(sample.metadata["sample_start_frame"]))
    assert sample.metadata["state_anchor_frame"] == sample.metadata["sample_start_frame"]
    assert not torch.equal(sample.condition_latents, sample.video_latents)


def test_uniform_segment_require_full_segment_drops_short_tail_starts(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_full_index"
    _build_local_robotwin_latent_repo(repo_root, total_rows=6, latent_num_frames=6)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=4,
                segment_max_frames=4,
                segment_length_stride=1,
                require_full_segment=True,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)

    assert len(train_dataset) == 3
    assert [latent_start for _, latent_start in train_dataset._virtual_index] == [0, 1, 2]
    sample = train_dataset[2]
    assert sample.metadata["subwindow_latent_start"] == 2
    assert sample.metadata["subwindow_latent_end"] == 6
    assert sample.metadata["segment_padded_latent_frames"] == 0
    assert sample.metadata["tail_padding_mode"] == "none"


def test_uniform_segment_dataset_delegates_to_sampling_plan(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_plan"
    _build_local_robotwin_latent_repo(
        repo_root,
        total_rows=8,
        latent_num_frames=8,
    )

    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=137,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=2,
                segment_max_frames=6,
                segment_length_stride=2,
                segment_locality_block_size=2,
                start_padding_frames=2,
                sample_weight_mode=SampleWeightMode.VALID_ACTION_STEPS,
                randomize_segment_length=True,
                randomize_segment_start=True,
                randomize_geometry=True,
                chunk_size=4,
                window_size=8,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    plan = train_dataset._uniform_segment_sampling_plan

    assert isinstance(plan, LocalLatentUniformSegmentSamplingPlan)
    assert train_dataset._virtual_index is plan.virtual_index
    assert train_dataset._virtual_indices_by_window == {
        window_index: list(indices)
        for window_index, indices in plan.virtual_indices_by_window.items()
    }
    assert train_dataset._task_virtual_start_counts == dict(
        plan.task_virtual_start_counts
    )
    assert train_dataset.sample_weights is plan.sample_weights
    assert train_dataset.build_epoch_index_order(epoch=7) == (
        plan.build_epoch_index_order(epoch=7)
    )
    restored_plan = pickle.loads(pickle.dumps(plan))
    assert restored_plan.virtual_index == plan.virtual_index
    assert restored_plan.sample_weights == plan.sample_weights
    assert restored_plan.build_epoch_index_order(epoch=7) == (
        plan.build_epoch_index_order(epoch=7)
    )

    window_index, virtual_start = train_dataset._virtual_index[0]
    window = train_dataset.windows[window_index]
    start_padding_frames = plan.resolve_start_padding_frames(
        train_dataset.data_config,
        window,
    )
    random.seed(91)
    owner_geometry = plan.sample_segment_geometry(
        index=0,
        source_latent_frames=window.latent_num_frames,
        virtual_latent_start=virtual_start,
        start_padding_frames=start_padding_frames,
    )
    owner_attention = plan.sample_attention_geometry(
        segment_length=owner_geometry[0]
    )
    random.seed(91)
    sample = train_dataset[0]
    assert sample.metadata["segment_length_frames"] == owner_geometry[0]
    assert sample.metadata["subwindow_latent_start"] == owner_geometry[1]
    assert sample.metadata["sampled_chunk_size"] == owner_attention[0]
    assert sample.metadata["sampled_window_size"] == owner_attention[1]


def test_uniform_segment_uses_typed_assembler(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "robotwin_local_latent_typed_segment"
    _build_local_robotwin_latent_repo(
        repo_root,
        total_rows=8,
        latent_num_frames=8,
        include_condition_latent=True,
    )

    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=4,
                segment_max_frames=4,
                segment_length_stride=1,
                start_padding_frames=2,
                require_full_segment=False,
                randomize_segment_length=False,
                randomize_segment_start=False,
                randomize_geometry=False,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    assert isinstance(
        train_dataset._segment_assembler,
        LocalLatentSegmentAssembler,
    )
    assert (
        train_dataset._segment_assembler.supervision_assembler
        is train_dataset._supervision_assembler
    )

    index = next(
        index
        for index, (_, latent_start) in enumerate(
            train_dataset._virtual_index
        )
        if latent_start == -1
    )
    window_index, latent_start = train_dataset._virtual_index[index]
    window = train_dataset.windows[window_index]
    repo_bundle = train_dataset._repo_bundles[str(window.repo_root)]
    rows = train_dataset._latent_repository.load_episode_rows(
        window.repo_root,
        window.episode_index,
        repo_bundle.metadata,
    )
    (
        video_latents,
        _,
        primary_payload,
        condition_latents,
        _,
    ) = train_dataset._latent_repository.load_canonical_window_latents(
        window,
        repo_bundle.metadata,
    )
    raw_frame_ids = [
        int(value)
        for value in list(primary_payload.get("frame_ids", []))
    ] or list(window.observation_frame_indices)
    start_padding_frames = (
        train_dataset._uniform_segment_sampling_plan.resolve_start_padding_frames(
            train_dataset.data_config,
            window,
        )
    )
    segment = train_dataset._segment_assembler.build(
        video_latents=video_latents,
        condition_latents=condition_latents,
        rows=rows,
        raw_frame_ids=raw_frame_ids,
        window=window,
        latent_start=latent_start,
        segment_length=4,
        start_padding_frames=start_padding_frames,
    )
    assert isinstance(segment, LocalLatentSegment)
    assert segment.pre_start_frames == 2
    assert segment.condition_latents is not None

    sample = train_dataset[index]
    torch.testing.assert_close(sample.video_latents, segment.video_latents)
    torch.testing.assert_close(
        sample.condition_latents,
        segment.condition_latents,
    )
    torch.testing.assert_close(sample.actions, segment.actions)
    torch.testing.assert_close(sample.action_mask, segment.action_mask)
    torch.testing.assert_close(sample.state, segment.state)
    torch.testing.assert_close(
        sample.proprio_context_state,
        segment.proprio_context_state,
    )
    assert sample.metadata["sample_start_frame"] == segment.sample_start_frame
    assert sample.metadata["latent_loss_frame_start"] == segment.loss_frame_start
    assert sample.metadata["observed_frame_ids"] == segment.observed_frame_ids
    assert pickle.loads(pickle.dumps(segment)).boundary_metadata == (
        segment.boundary_metadata
    )


def test_uniform_segment_start_padding_repeats_first_latent_and_masks_virtual_actions(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_start_padding"
    _build_local_robotwin_latent_repo(repo_root, total_rows=6, latent_num_frames=6)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=6,
                segment_max_frames=6,
                segment_length_stride=1,
                chunk_size=2,
                window_size=4,
                start_padding_frames=3,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]
    prefix_actions = config.data.action_schema.action_horizon // config.data.num_frames
    virtual_action_steps = 4 * prefix_actions

    assert len(train_dataset) == 9
    assert train_dataset._virtual_index[:4] == ((0, -3), (0, -2), (0, -1), (0, 0))
    assert sample.metadata["subwindow_latent_start"] == -3
    assert sample.metadata["subwindow_latent_end"] == 3
    assert sample.metadata["latent_frame_start"] == -3
    assert sample.metadata["frame_shift"] == -3
    assert sample.metadata["sample_start_frame"] == 0
    assert sample.metadata["observed_frame_ids"] == [0, 0, 0, 0, 1, 2]
    assert sample.metadata["start_padding_frames"] == 3
    assert sample.metadata["segment_pre_start_frames"] == 4
    assert sample.metadata["start_padding_mode"] == "repeat_first_latent"
    assert sample.metadata["loss_frame_start"] == 4
    assert sample.metadata["loss_frame_end"] == 6
    assert sample.metadata["latent_loss_frame_start"] == 4
    assert sample.metadata["latent_loss_frame_end"] == 6
    assert sample.metadata["action_loss_frame_start"] == 4
    assert sample.metadata["action_loss_frame_end"] == 6
    assert sample.metadata["segment_valid_latent_frames"] == 6
    assert sample.metadata["segment_padded_latent_frames"] == 0
    assert sample.metadata["tail_padding_mode"] == "none"
    assert torch.equal(sample.video_latents[:, 0], sample.video_latents[:, 1])
    assert torch.equal(sample.video_latents[:, 1], sample.video_latents[:, 2])
    assert torch.equal(sample.video_latents[:, 2], sample.video_latents[:, 3])
    assert sample.action_mask[:virtual_action_steps].sum().item() == 0
    assert sample.action_mask[virtual_action_steps:].sum().item() == 3 * 30
    assert torch.allclose(sample.actions[virtual_action_steps], torch.zeros(30))
    assert torch.allclose(sample.actions[virtual_action_steps + 1], torch.ones(30))
    assert sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_frames"] == 4
    assert sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_mask"] == 0.0
    assert sample.metadata["valid_action_steps"] == 3

    frame_zero_sample = train_dataset[3]
    assert frame_zero_sample.metadata["subwindow_latent_start"] == 0
    assert frame_zero_sample.metadata["observed_frame_ids"] == [0, 1, 2, 3, 4, 5]
    assert frame_zero_sample.metadata["segment_pre_start_frames"] == 1
    assert frame_zero_sample.metadata["loss_frame_start"] == 1
    assert frame_zero_sample.action_mask[:prefix_actions].sum().item() == 0
    assert frame_zero_sample.metadata["valid_action_steps"] == 6


def test_uniform_segment_randomized_start_samples_head_and_tail_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_random_head_tail"
    _build_local_robotwin_latent_repo(repo_root, total_rows=6, latent_num_frames=6)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=4,
                segment_max_frames=4,
                segment_length_stride=1,
                randomize_segment_start=True,
                start_padding_frames=3,
            ),
        ),
    )

    observed_bounds: list[tuple[int, int]] = []

    def choose_upper_bound(low: int, high: int) -> int:
        observed_bounds.append((low, high))
        return high

    monkeypatch.setattr(random, "randint", choose_upper_bound)
    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert observed_bounds == [(-3, 5)]
    assert sample.metadata["subwindow_latent_start"] == 5
    assert sample.metadata["subwindow_latent_end"] == 9
    assert sample.metadata["segment_valid_latent_frames"] == 1
    assert sample.metadata["segment_padded_latent_frames"] == 3
    assert sample.metadata["tail_padding_mode"] == "zero_hold"
    assert sample.metadata["observed_frame_ids"] == [5, 5, 5, 5]
    assert torch.equal(sample.video_latents[:, 0], sample.video_latents[:, 1])
    assert torch.equal(sample.video_latents[:, 1], sample.video_latents[:, 2])


def test_hierarchical_fixed_segment_samples_padded_start_range_and_masks_targets(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_fixed_segment"
    _build_local_robotwin_latent_repo(repo_root, total_rows=6, latent_num_frames=6)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=7,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=4,
                start_padding_frames=3,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sampling_plan = train_dataset._hierarchical_sampling_plan
    segment_plan = train_dataset._hierarchical_segment_plan

    assert len(train_dataset) == 8
    assert isinstance(segment_plan, LocalLatentHierarchicalSegmentPlan)
    assert segment_plan.sampling_plan is sampling_plan
    assert pickle.loads(pickle.dumps(segment_plan)) == segment_plan
    assert train_dataset._window_start_ranges_by_chunk == (((1, -3, 4, 8),),)
    assert segment_plan.window_start_ranges_by_chunk == (
        train_dataset._window_start_ranges_by_chunk
    )
    assert train_dataset._task_specs is sampling_plan.task_specs
    assert train_dataset._task_weights is sampling_plan.task_weights
    assert train_dataset._task_mass_total == sampling_plan.task_mass_total
    assert train_dataset._task_specs_by_text is sampling_plan.task_specs_by_text
    assert train_dataset._epoch_sample_count == sampling_plan.epoch_sample_count
    assert [
        segment_plan.draw(index)
        for index in range(32)
    ] == [
        sampling_plan.draw(
            index=index,
            split_seed=config.data.split_seed,
            split=config.data.split,
        )
        for index in range(32)
    ]
    sample_key = segment_plan.resolve_sample_key(0)
    assert isinstance(sample_key, LocalLatentHierarchicalSampleKey)
    assert pickle.loads(pickle.dumps(sample_key)) == sample_key
    assert sample_key.as_metadata() == train_dataset.resolve_hierarchical_sample_key(0)
    assert list(train_dataset.iter_hierarchical_eligible_start_keys()) == list(
        sampling_plan.iter_eligible_start_keys()
    )
    tail_sample = next(
        train_dataset[index]
        for index in range(200)
        if train_dataset[index].metadata["subwindow_latent_start"] == 4
    )
    head_sample = next(
        train_dataset[index]
        for index in range(200)
        if train_dataset[index].metadata["subwindow_latent_start"] == -3
    )

    assert tail_sample.metadata["window_sampling_mode"] == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT
    assert tail_sample.metadata["hierarchical_start_min"] == -3
    assert tail_sample.metadata["hierarchical_start_max"] == 4
    assert tail_sample.metadata["hierarchical_start_count"] == 8
    assert tail_sample.metadata["subwindow_latent_end"] == 8
    assert tail_sample.metadata["effective_frame_start"] == 4
    assert tail_sample.metadata["effective_frame_end"] == 6
    assert tail_sample.metadata["effective_segment_frames"] == 2
    assert tail_sample.metadata["tail_padded_frame_count"] == 2
    assert tail_sample.metadata["segment_valid_latent_frames"] == 2
    assert tail_sample.metadata["segment_padded_latent_frames"] == 2
    assert tail_sample.metadata["tail_padding_mode"] == "zero_order_hold"
    assert tail_sample.metadata["latent_loss_frame_start"] == 1
    assert tail_sample.metadata["latent_loss_frame_end"] == 2
    assert tail_sample.metadata["observed_frame_ids"] == [4, 5]
    assert tail_sample.video_latents.shape[1] == 2

    assert head_sample.metadata["subwindow_latent_start"] == -3
    assert head_sample.metadata["effective_frame_start"] == -3
    assert head_sample.metadata["effective_frame_end"] == 1
    assert head_sample.metadata["effective_segment_frames"] == 4
    assert head_sample.metadata["head_padded_frame_count"] == 0
    assert head_sample.metadata["segment_pre_start_frames"] == 3
    assert head_sample.metadata["latent_loss_frame_start"] == 3
    assert head_sample.metadata["latent_loss_frame_end"] == 4
    assert head_sample.action_mask[:6].sum().item() == 0
    assert head_sample.action_mask.sum().item() == 30
    assert head_sample.video_latents.shape[1] == 4
    assert head_sample.metadata["tail_padding_policy"] == "zero_order_hold"
    assert head_sample.metadata["padded_target_policy"] == "mask_loss"


def test_hierarchical_fixed_segment_rejects_multi_sample_compact_batches(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_fixed_segment_batch_guard"
    _build_local_robotwin_latent_repo(repo_root, total_rows=6, latent_num_frames=6)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=7,
            num_workers=0,
            train_batch_size=2,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=4,
                start_padding_frames=3,
            ),
        ),
    )

    with pytest.raises(ValueError, match="compact boundary sampling"):
        build_train_val_latent_datasets(config.data)


def test_hierarchical_fixed_segment_randomizes_chunk_geometry(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_fixed_segment_random_chunk"
    _build_local_robotwin_latent_repo(repo_root, total_rows=16, latent_num_frames=16)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=11,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=8,
                chunk_size=4,
                window_size=30,
                start_padding_frames=3,
                randomize_geometry=True,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    expected_keys = set(train_dataset.iter_hierarchical_eligible_start_keys())
    records = [train_dataset.resolve_hierarchical_sample_key(index) for index in range(256)]

    assert len(train_dataset) == 81
    assert train_dataset._window_start_ranges_by_chunk == (
        (
            (1, -7, 14, 22),
            (2, -7, 13, 21),
            (3, -7, 12, 20),
            (4, -6, 11, 18),
        ),
    )
    assert (0, 14, 1) in expected_keys
    assert (0, 13, 2) in expected_keys
    assert (0, 12, 3) in expected_keys
    assert (0, 11, 4) in expected_keys
    assert (0, 14, 4) not in expected_keys
    assert {int(record["sampled_chunk_size"]) for record in records} == {1, 2, 3, 4}
    assert {int(record["sampled_window_size"]) for record in records} == {30}
    for record in records:
        assert (
            int(record["trajectory_window_index"]),
            int(record["latent_start"]),
            int(record["sampled_chunk_size"]),
        ) in expected_keys
        expected_loss_start = max(
            int(record["sampled_chunk_size"]),
            int(record["supervised_frame_start"]),
        )
        assert int(record["loss_frame_start"]) == expected_loss_start
        assert int(record["chunk_size_for_boundary"]) == int(record["sampled_chunk_size"])

    samples = [train_dataset[index] for index in range(256)]
    assert {int(sample.metadata["sampled_chunk_size"]) for sample in samples} == {1, 2, 3, 4}
    for sample in samples:
        sampled_chunk_size = int(sample.metadata["sampled_chunk_size"])
        expected_loss_start = max(sampled_chunk_size, int(sample.metadata["supervised_start"]))
        assert int(sample.metadata["sampled_window_size"]) == 30
        assert int(sample.metadata["loss_frame_start"]) == expected_loss_start
        assert int(sample.metadata["latent_loss_frame_start"]) == expected_loss_start
        assert int(sample.metadata["action_loss_frame_start"]) == expected_loss_start


def test_hierarchical_fixed_segment_rollout_context_prefix_masks_context(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_fixed_segment_context_prefix"
    _build_local_robotwin_latent_repo(repo_root, total_rows=16, latent_num_frames=16)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=13,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=8,
                chunk_size=2,
                window_size=4,
                start_padding_frames=3,
                randomize_geometry=False,
                context_prefix_policy=SegmentContextPolicy.ROLLOUT_HISTORY,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    assert train_dataset._window_start_ranges_by_chunk == (((2, -3, 15, 19),),)
    sample_index = next(
        index
        for index in range(500)
        if train_dataset.resolve_hierarchical_sample_key(index)["latent_start"] == 5
    )
    startup_sample_index = next(
        index
        for index in range(500)
        if train_dataset.resolve_hierarchical_sample_key(index)["latent_start"] == 0
    )
    padded_start_sample_index = next(
        index
        for index in range(500)
        if train_dataset.resolve_hierarchical_sample_key(index)["latent_start"] == -3
    )
    sample = train_dataset[sample_index]
    startup_sample = train_dataset[startup_sample_index]
    padded_start_sample = train_dataset[padded_start_sample_index]

    assert sample.metadata["virtual_latent_start"] == 5
    assert sample.metadata["logical_frame_start"] == 1
    assert sample.metadata["effective_frame_start"] == 1
    assert sample.metadata["effective_frame_end"] == 13
    assert sample.metadata["target_frame_start"] == 5
    assert sample.metadata["target_frame_end"] == 13
    assert sample.metadata["context_prefix_policy"] == "rollout_history"
    assert sample.metadata["context_prefix_frames_requested"] == 4
    assert sample.metadata["context_prefix_frames_in_sample"] == 4
    assert sample.metadata["context_prefix_real_frames"] == 4
    assert sample.metadata["context_prefix_truncated_frames"] == 0
    assert sample.metadata["supervised_start"] == 4
    assert sample.metadata["latent_loss_frame_start"] == 4
    assert sample.metadata["latent_loss_frame_end"] == 12
    assert sample.metadata["action_loss_frame_start"] == 4
    assert sample.metadata["action_loss_frame_end"] == 12
    assert sample.metadata["history_frames"] == 4
    assert sample.video_latents.shape[1] == 12

    assert padded_start_sample.metadata["virtual_latent_start"] == -3
    assert padded_start_sample.metadata["logical_frame_start"] == -7
    assert padded_start_sample.metadata["effective_frame_start"] == -3
    assert padded_start_sample.metadata["startup_context_frames"] == 3
    assert padded_start_sample.metadata["head_padded_frame_count"] == 4
    assert padded_start_sample.metadata["context_prefix_frames_in_sample"] == 0
    assert padded_start_sample.metadata["context_prefix_real_frames"] == 0
    assert padded_start_sample.metadata["context_prefix_truncated_frames"] == 4
    assert padded_start_sample.metadata["latent_loss_frame_start"] == 4
    assert padded_start_sample.metadata["history_frames"] == 4
    assert padded_start_sample.video_latents.shape[1] == 8
    for offset in (1, 2, 3):
        assert torch.equal(padded_start_sample.video_latents[:, 0], padded_start_sample.video_latents[:, offset])

    assert startup_sample.metadata["virtual_latent_start"] == 0
    assert startup_sample.metadata["logical_frame_start"] == -4
    assert startup_sample.metadata["effective_frame_start"] == 0
    assert startup_sample.metadata["startup_context_frames"] == 0
    assert startup_sample.metadata["head_padded_frame_count"] == 4
    assert startup_sample.metadata["context_prefix_frames_in_sample"] == 0
    assert startup_sample.metadata["context_prefix_real_frames"] == 0
    assert startup_sample.metadata["context_prefix_truncated_frames"] == 4
    assert startup_sample.metadata["segment_valid_latent_frames"] == startup_sample.video_latents.shape[1]
    assert startup_sample.metadata["segment_padded_latent_frames"] == 4
    assert startup_sample.metadata["latent_loss_frame_start"] == 2
    assert startup_sample.metadata["history_frames"] == 2
    assert startup_sample.video_latents.shape[1] == 8


def test_hierarchical_fixed_segment_context_prefix_aligns_loss_to_chunk_boundary(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_fixed_segment_context_chunk_alignment"
    _build_local_robotwin_latent_repo(repo_root, total_rows=20, latent_num_frames=20)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=19,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=16,
                chunk_size=4,
                window_size=4,
                start_padding_frames=3,
                randomize_geometry=False,
                context_prefix_policy=SegmentContextPolicy.ROLLOUT_HISTORY,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample_index = next(
        index
        for index in range(500)
        if train_dataset.resolve_hierarchical_sample_key(index)["latent_start"] == 5
    )
    sample = train_dataset[sample_index]

    assert sample.metadata["context_prefix_frames_requested"] == 8
    assert sample.metadata["context_prefix_frames_in_sample"] == 5
    assert sample.metadata["context_prefix_real_frames"] == 5
    assert sample.metadata["startup_context_frames"] == 0
    assert sample.metadata["supervised_start"] == 5
    assert sample.metadata["latent_loss_frame_start"] == 8
    assert sample.metadata["action_loss_frame_start"] == 8
    assert sample.metadata["history_frames"] == 8
    assert sample.metadata["latent_loss_frame_start"] % sample.metadata["sampled_chunk_size"] == 0


def test_hierarchical_fixed_segment_without_context_keeps_geometry_history_frames(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_fixed_segment_no_context_history"
    _build_local_robotwin_latent_repo(repo_root, total_rows=16, latent_num_frames=16)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=23,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=8,
                chunk_size=2,
                window_size=4,
                start_padding_frames=3,
                randomize_geometry=False,
                context_prefix_policy=SegmentContextPolicy.NONE,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample_index = next(
        index
        for index in range(500)
        if train_dataset.resolve_hierarchical_sample_key(index)["latent_start"] == 5
    )
    sample = train_dataset[sample_index]

    assert sample.metadata["context_prefix_frames_requested"] == 0
    assert sample.metadata["latent_loss_frame_start"] == 2
    assert sample.metadata["history_frames"] == 4


def test_hierarchical_fixed_segment_rollout_parity_uses_one_context_frame(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_fixed_segment_rollout_parity"
    _build_local_robotwin_latent_repo(repo_root, total_rows=6, latent_num_frames=6)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=29,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=4,
                chunk_size=4,
                window_size=30,
                randomize_geometry=False,
                start_padding_frames=0,
                target_alignment=SampleTargetAlignment.NEXT_AFTER_CONTEXT,
                rollout_context_policy=RolloutContextPolicy.ONE_FRAME,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)

    assert len(train_dataset) == 5
    assert train_dataset._window_start_ranges_by_chunk == (((4, 1, 5, 5),),)

    first_target_index = next(
        index
        for index in range(100)
        if train_dataset.resolve_hierarchical_sample_key(index)["latent_start"] == 1
    )
    tail_target_index = next(
        index
        for index in range(100)
        if train_dataset.resolve_hierarchical_sample_key(index)["latent_start"] == 5
    )
    first_target_sample = train_dataset[first_target_index]
    tail_target_sample = train_dataset[tail_target_index]

    assert first_target_sample.metadata["target_alignment"] == "next_after_context"
    assert first_target_sample.metadata["rollout_context_policy"] == "one_frame"
    assert first_target_sample.metadata["virtual_latent_start"] == 1
    assert first_target_sample.metadata["effective_frame_start"] == 0
    assert first_target_sample.metadata["effective_frame_end"] == 5
    assert first_target_sample.metadata["target_frame_start"] == 1
    assert first_target_sample.metadata["target_frame_end"] == 5
    assert first_target_sample.metadata["context_prefix_frames_requested"] == 1
    assert first_target_sample.metadata["context_prefix_frames_in_sample"] == 1
    assert first_target_sample.metadata["latent_loss_frame_start"] == 1
    assert first_target_sample.metadata["latent_loss_frame_end"] == 5
    assert first_target_sample.metadata["history_frames"] == 1
    assert first_target_sample.metadata["observed_frame_ids"] == [0, 1, 2, 3, 4]
    assert first_target_sample.proprio_context_state is not None
    torch.testing.assert_close(
        first_target_sample.proprio_context_state[:, 0],
        torch.tensor([0.0, 4.0]),
    )
    assert first_target_sample.video_latents.shape[1] == 5
    prefix_actions = config.data.action_schema.action_horizon // config.data.num_frames
    assert first_target_sample.action_mask[:prefix_actions].sum().item() == 0
    assert first_target_sample.action_mask[prefix_actions:].sum().item() > 0
    assert first_target_sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_frames"] == 1
    assert first_target_sample.metadata["lingbot_window_action_alignment"]["leading_zero_action_mask"] == 0.0

    assert tail_target_sample.metadata["virtual_latent_start"] == 5
    assert tail_target_sample.metadata["effective_frame_start"] == 4
    assert tail_target_sample.metadata["effective_frame_end"] == 6
    assert tail_target_sample.metadata["target_frame_start"] == 5
    assert tail_target_sample.metadata["target_frame_end"] == 9
    assert tail_target_sample.metadata["tail_padded_frame_count"] == 3
    assert tail_target_sample.metadata["segment_valid_latent_frames"] == 2
    assert tail_target_sample.metadata["segment_padded_latent_frames"] == 3
    assert tail_target_sample.metadata["latent_loss_frame_start"] == 1
    assert tail_target_sample.metadata["latent_loss_frame_end"] == 2
    assert tail_target_sample.metadata["history_frames"] == 1
    assert tail_target_sample.metadata["observed_frame_ids"] == [4, 5]
    assert tail_target_sample.video_latents.shape[1] == 2
    assert tail_target_sample.action_mask[:prefix_actions].sum().item() == 0
    assert tail_target_sample.action_mask[prefix_actions:].sum().item() > 0


def test_hierarchical_fixed_segment_rollout_parity_history_stays_outside_target_budget(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_fixed_segment_rollout_history"
    _build_local_robotwin_latent_repo(repo_root, total_rows=40, latent_num_frames=40)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=31,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=8,
                chunk_size=4,
                window_size=6,
                randomize_geometry=False,
                start_padding_frames=0,
                target_alignment=SampleTargetAlignment.NEXT_AFTER_CONTEXT,
                rollout_context_policy=RolloutContextPolicy.ROLLOUT_HISTORY,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample_index = next(
        index
        for index in range(500)
        if train_dataset.resolve_hierarchical_sample_key(index)["latent_start"] == 20
    )
    sample = train_dataset[sample_index]

    assert sample.metadata["context_prefix_frames_requested"] == 12
    assert sample.metadata["context_prefix_frames_in_sample"] == 12
    assert sample.metadata["effective_frame_start"] == 8
    assert sample.metadata["effective_frame_end"] == 28
    assert sample.metadata["target_frame_start"] == 20
    assert sample.metadata["target_frame_end"] == 28
    assert sample.metadata["latent_loss_frame_start"] == 12
    assert sample.metadata["latent_loss_frame_end"] == 20
    assert sample.metadata["history_frames"] == 12
    assert sample.video_latents.shape[1] == 20
    prefix_actions = config.data.action_schema.action_horizon // config.data.num_frames
    assert sample.action_mask[:prefix_actions].sum().item() == 0
    assert sample.action_mask[prefix_actions : 12 * prefix_actions].sum().item() > 0


def test_uniform_segment_require_full_segment_uses_short_payload_as_full_segment(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_payload_count"
    _build_local_robotwin_latent_repo(repo_root, total_rows=80, latent_num_frames=24)
    for latent_path in (repo_root / "latents" / "chunk-000").glob("*/episode_000000_0_24.pth"):
        payload = dict(torch.load(latent_path, map_location="cpu", weights_only=False))
        payload.pop("frame_ids", None)
        torch.save(payload, latent_path.with_name("episode_000000_0_80.pth"))
        latent_path.unlink()

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=64,
                segment_max_frames=64,
                segment_length_stride=1,
                require_full_segment=True,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)

    assert len(train_dataset) == 1
    assert train_dataset._virtual_index == ((0, 0),)
    sample = train_dataset[0]
    assert sample.metadata["segment_length_frames"] == 24
    assert sample.metadata["subwindow_latent_start"] == 0
    assert sample.metadata["subwindow_latent_end"] == 24
    assert sample.metadata["segment_valid_latent_frames"] == 24
    assert sample.metadata["segment_padded_latent_frames"] == 0
    assert sample.metadata["tail_padding_mode"] == "none"


def test_uniform_segment_action_count_uses_configured_actions_per_latent_frame(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_stride_action_count"
    _build_local_robotwin_latent_repo(repo_root, total_rows=400, latent_num_frames=80)
    for latent_path in (repo_root / "latents" / "chunk-000").glob("*/episode_000000_0_80.pth"):
        payload = dict(torch.load(latent_path, map_location="cpu", weights_only=False))
        payload["frame_ids"] = list(range(0, 320, 4))
        torch.save(payload, latent_path)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=80,
                segment_max_frames=80,
                segment_length_stride=1,
                require_full_segment=True,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]
    expected_actions_per_frame = config.data.action_schema.action_horizon // config.data.num_frames

    assert sample.video_latents.shape[1] == 80
    assert sample.actions.shape == (80 * expected_actions_per_frame, 30)
    assert sample.metadata["lingbot_window_action_alignment"]["frame_stride"] == 4
    assert sample.metadata["lingbot_window_action_alignment"]["prefix_actions"] == expected_actions_per_frame
    assert sample.metadata["lingbot_window_action_alignment"]["required_action_num"] == 80 * expected_actions_per_frame


def test_lingbot_exact_actions_use_wan_causal_latent_anchors(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent_wan_temporal_layout"
    _build_local_robotwin_latent_repo(
        repo_root,
        total_rows=32,
        latent_num_frames=4,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
    )
    for latent_path in (repo_root / "latents" / "chunk-000").glob("*/episode_000000_0_4.pth"):
        payload = dict(torch.load(latent_path, map_location="cpu", weights_only=False))
        payload["frame_ids"] = list(range(15))
        torch.save(payload, latent_path)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.FULL_SEGMENT,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.metadata["observed_frame_ids"] == [0, 4, 8, 12]
    assert sample.metadata["sample_start_frame"] == 0
    assert sample.metadata["sample_end_frame"] == 13
    assert sample.metadata["lingbot_window_action_alignment"]["action_start_offset"] == 0
    assert sample.actions.shape == (16, 7)
    torch.testing.assert_close(sample.actions[:4], torch.zeros(4, 7))
    torch.testing.assert_close(sample.actions[4:8, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_hierarchical_exact_actions_use_wan_causal_latent_anchors(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_hierarchical_wan_temporal_layout"
    _build_local_robotwin_latent_repo(
        repo_root,
        total_rows=64,
        latent_num_frames=8,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
    )
    for latent_path in (repo_root / "latents" / "chunk-000").glob("*/episode_000000_0_8.pth"):
        payload = dict(torch.load(latent_path, map_location="cpu", weights_only=False))
        payload["frame_ids"] = list(range(31))
        torch.save(payload, latent_path)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            split_seed=0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=4,
                chunk_size=2,
                randomize_geometry=False,
                start_padding_frames=0,
                target_alignment=SampleTargetAlignment.LEGACY,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = None
    for index in range(100):
        candidate = train_dataset[index]
        if candidate.metadata["subwindow_latent_start"] == 0:
            sample = candidate
            break
    assert sample is not None

    assert sample.metadata["observed_frame_ids"] == [0, 4, 8, 12]
    assert sample.metadata["sample_start_frame"] == 0
    assert sample.metadata["sample_end_frame"] == 13
    assert sample.metadata["lingbot_window_action_alignment"]["action_start_offset"] == 0
    torch.testing.assert_close(sample.actions[:4], torch.zeros(4, 7))
    torch.testing.assert_close(sample.actions[4:8, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_uniform_segment_frame_shift_uses_latent_frame_not_raw_frame(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_latent_frame_shift"
    _build_local_robotwin_latent_repo(repo_root, total_rows=400, latent_num_frames=80)
    for latent_path in (repo_root / "latents" / "chunk-000").glob("*/episode_000000_0_80.pth"):
        payload = dict(torch.load(latent_path, map_location="cpu", weights_only=False))
        payload["frame_ids"] = list(range(0, 320, 4))
        torch.save(payload, latent_path)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=4,
                segment_max_frames=4,
                segment_length_stride=1,
                require_full_segment=True,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[1]

    assert sample.metadata["subwindow_latent_start"] == 1
    assert sample.metadata["sample_start_frame"] == 4
    assert sample.metadata["latent_frame_start"] == 1
    assert sample.metadata["frame_shift"] == 1

def test_uniform_segment_length_is_deterministic_for_virtual_sample(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_deterministic"
    _build_local_robotwin_latent_repo(repo_root, total_rows=8, latent_num_frames=8)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=123,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=2,
                segment_max_frames=5,
                segment_length_stride=1,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    first_length = train_dataset[3].metadata["segment_length_frames"]

    torch.manual_seed(999)
    for _ in range(10):
        _ = random.randrange(1 << 30)

    assert train_dataset[3].metadata["segment_length_frames"] == first_length


def test_uniform_segment_sampler_round_robins_trajectory_blocks(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_order"
    _build_local_robotwin_latent_repo(repo_root, total_rows=4, latent_num_frames=4)
    _append_latent_episode(
        repo_root,
        episode_index=1,
        task_index=1,
        task_text="longer task",
        total_rows=6,
        latent_num_frames=6,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=2,
                segment_max_frames=2,
                segment_locality_block_size=1,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sampler = train_dataset.build_train_sampler(world_size=1, rank=0)
    order = list(sampler)
    first_windows = {train_dataset._virtual_index[index][0] for index in order[:2]}

    assert len(train_dataset) == 10
    assert sorted(order) == list(range(len(train_dataset)))
    assert first_windows == {0, 1}


def test_uniform_segment_replacement_order_uses_replacement_sampler(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_replacement_order"
    _build_local_robotwin_latent_repo(repo_root, total_rows=4, latent_num_frames=4)
    _append_latent_episode(
        repo_root,
        episode_index=1,
        task_index=1,
        task_text="longer task",
        total_rows=6,
        latent_num_frames=6,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=2,
                segment_max_frames=2,
                segment_locality_block_size=1,
                sample_order_mode=SampleOrderMode.REPLACEMENT,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sampler = train_dataset.build_train_sampler(world_size=1, rank=0)
    order = list(sampler)

    assert isinstance(sampler, LocalLatentWeightedTrainSampler)
    assert not isinstance(sampler, LocalLatentEpochOrderSampler)
    assert len(order) == len(train_dataset)
    assert all(0 <= index < len(train_dataset) for index in order)


def test_uniform_segment_task_virtual_start_power_balances_task_mass(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_uniform_segment_task_power"
    _build_local_robotwin_latent_repo(repo_root, total_rows=4, latent_num_frames=4)
    _append_latent_episode(
        repo_root,
        episode_index=1,
        task_index=1,
        task_text="long task",
        total_rows=16,
        latent_num_frames=16,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.UNIFORM_SEGMENT,
                segment_min_frames=2,
                segment_max_frames=2,
                segment_length_stride=1,
                sample_weight_mode=SampleWeightMode.TASK_VIRTUAL_START_COUNT_POWER,
                sample_weight_length_power=0.5,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    weights_by_task: dict[str, list[float]] = {"pick up block": [], "long task": []}
    for virtual_index, (window_index, _) in enumerate(train_dataset._virtual_index):
        weights_by_task[train_dataset._window_task_texts[window_index]].append(train_dataset.sample_weights[virtual_index])

    short_mass = sum(weights_by_task["pick up block"])
    long_mass = sum(weights_by_task["long task"])

    assert len(weights_by_task["pick up block"]) == 4
    assert len(weights_by_task["long task"]) == 16
    assert weights_by_task["pick up block"][0] > weights_by_task["long task"][0]
    assert long_mass / short_mass == pytest.approx(2.0)

    sample = train_dataset[0]
    assert sample.metadata["train_sample_weight_mode"] == SampleWeightMode.TASK_VIRTUAL_START_COUNT_POWER
    assert sample.metadata["sample_weight_length_power"] == pytest.approx(0.5)
    assert sample.metadata["eligible_task_virtual_start_count"] == 4
    assert sample.metadata["dataset_mean_eligible_task_virtual_start_count"] == pytest.approx(10.0)


def test_hierarchical_fixed_segment_task_power_is_explicit_task_mass(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_task_power"
    _build_local_robotwin_latent_repo(repo_root, total_rows=4, latent_num_frames=4)
    _append_latent_episode(
        repo_root,
        episode_index=1,
        task_index=1,
        task_text="long task",
        total_rows=16,
        latent_num_frames=16,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=2,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    short_task = train_dataset._task_specs_by_text["pick up block"]
    long_task = train_dataset._task_specs_by_text["long task"]
    sample = next(
        train_dataset[index]
        for index in range(200)
        if train_dataset[index].metadata["hierarchical_task_text"] == "pick up block"
    )

    assert len(train_dataset) == 18
    assert short_task.eligible_start_count == 3
    assert long_task.eligible_start_count == 15
    assert long_task.task_mass / short_task.task_mass == pytest.approx(math.sqrt(5.0))
    assert sample.metadata["hierarchical_task_start_power"] == pytest.approx(0.5)
    assert sample.metadata["hierarchical_demo_count_power"] == pytest.approx(0.0)
    assert sample.metadata["hierarchical_trajectory_start_power"] == pytest.approx(1.0)
    assert sample.metadata["hierarchical_task_eligible_start_count"] == 3
    assert sample.metadata["hierarchical_epoch_sample_count"] == 18


def test_hierarchical_fixed_segment_dataloader_samples_stepwise_valid_keys(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent_hierarchical_dataloader_stepwise"
    _build_local_robotwin_latent_repo(repo_root, total_rows=4, latent_num_frames=4)
    _append_latent_episode(
        repo_root,
        episode_index=1,
        task_index=1,
        task_text="long task",
        total_rows=16,
        latent_num_frames=16,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            split_seed=0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
                segment_frames=2,
                task_start_power=0.5,
                demo_count_power=0.0,
                trajectory_start_power=1.0,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sampler = train_dataset.build_train_sampler(world_size=1, rank=0)
    sampler.set_epoch(0)
    loader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_latent_wam_samples,
    )
    seen = {
        (
            metadata["trajectory_window_index"],
            metadata["virtual_latent_start"],
            metadata["sampled_chunk_size"],
        )
        for batch in loader
        for metadata in batch.metadata
    }
    expected = set(train_dataset.iter_hierarchical_eligible_start_keys())

    assert seen
    assert seen.issubset(expected)
    assert len(train_dataset) == 18


def test_standard_policy_full_segment_latent_profile_uses_schema_horizon(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent_full_segment"
    _build_local_robotwin_latent_repo(
        repo_root,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
        total_rows=20,
        latent_num_frames=4,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.actions.shape == (6, 7)
    assert sample.state.shape == (1, 8)
    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.FULL_SEGMENT
    assert sample.metadata["observation_start"] == 0
    assert sample.metadata["observation_frame_indices"] == [0, 1, 2, 3]
    assert sample.metadata["window_start_frame"] == 0
    assert sample.metadata["window_end_frame"] == 4
    assert sample.metadata["dataset_id"] == str(repo_root)


def test_local_lerobot_latent_dataset_loads_empty_text_embedding_as_negative_context(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)
    empty_emb = torch.randn(512, 4096)
    empty_emb_path = tmp_path / "empty_emb.pt"
    torch.save(empty_emb, empty_emb_path)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            empty_text_embedding_path=str(empty_emb_path),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.negative_text_context is not None
    assert torch.equal(sample.negative_text_context, empty_emb)


def test_local_lerobot_latent_dataset_raises_for_missing_configured_empty_text_embedding(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            empty_text_embedding_path=str(tmp_path / "missing_empty_emb.pt"),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    with pytest.raises(FileNotFoundError, match="Configured `data.empty_text_embedding_path` does not exist"):
        _ = build_train_val_latent_datasets(config.data)


def test_local_lerobot_latent_dataset_uses_pose_source_key_for_state(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent"
    _build_local_robotwin_latent_repo(
        repo_root,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.state.shape == (1, 8)
    assert sample.metadata["state_source_key"] == "observation.state"


def test_local_lerobot_latent_dataset_supports_causal_prefix_suffix_sampling(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent_causal"
    _build_local_robotwin_latent_repo(
        repo_root,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
        total_rows=48,
        latent_num_frames=10,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/causal_video_prediction_libero_latent_local.yaml")
    config = replace(
        config,
        data=replace(
            _disable_replay_status(config.data),
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            num_frames=10,
            sample_construction=replace(
                config.data.sample_construction,
                num_frames=8,
                causal_prefix_suffix_buckets=(
                    config.data.sample_construction.causal_prefix_suffix_buckets[0],
                    type(config.data.sample_construction.causal_prefix_suffix_buckets[0])(observed_frames=2, future_frames=4),
                ),
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    planner = train_dataset._causal_sampling_planner
    assert isinstance(planner, LatentCausalPrefixSuffixWindowPlanner)
    assert planner.sample_config is config.data.sample_construction
    assert isinstance(
        pickle.loads(pickle.dumps(planner)),
        LatentCausalPrefixSuffixWindowPlanner,
    )

    candidates = planner.build_candidates(
        raw_frame_ids=tuple(range(10)),
        source_latent_frames=10,
        row_count=48,
    )
    assert candidates
    assert all(
        isinstance(candidate, LatentCausalPrefixSuffixCandidate)
        for candidate in candidates
    )

    previous_rng_state = random.getstate()
    try:
        random.seed(1458)
        plan = planner.plan(
            raw_frame_ids=tuple(range(10)),
            source_latent_frames=10,
            row_count=48,
            sample_index=0,
        )
        planner_next_random = random.random()

        random.seed(1458)
        sample = train_dataset[0]
        dataset_next_random = random.random()
    finally:
        random.setstate(previous_rng_state)

    assert isinstance(plan, LatentCausalPrefixSuffixWindowPlan)
    assert pickle.loads(pickle.dumps(plan)) == plan
    assert dataset_next_random == planner_next_random
    assert sample.metadata["subwindow_latent_start"] == plan.latent_start
    assert sample.metadata["subwindow_latent_end"] == plan.latent_end
    assert sample.metadata["sample_start_frame"] == plan.sample_start_frame
    assert sample.metadata["sample_end_frame"] == plan.sample_end_frame
    assert sample.metadata["observed_frame_ids"] == list(plan.observed_frame_ids)

    assert sample.video_latents.shape == (48, 8, 8, 16)
    assert sample.actions.shape == (0, 7)
    assert sample.state.shape == (0, 8)
    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.CAUSAL_PREFIX_SUFFIX
    assert sample.metadata["valid_video_frames"] in {4, 6}
    assert sample.metadata["observed_prefix_frames"] in {1, 2}
    assert sample.metadata["future_suffix_frames"] in {3, 4}
    assert sample.metadata["observed_prefix_frames"] + sample.metadata["future_suffix_frames"] == sample.metadata["valid_video_frames"]


def test_parallel_stream_runtime_runs_on_local_lerobot_latent_dataset(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        training=replace(config.training, num_steps=1),
        data=replace(
            _disable_replay_status(config.data),
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path / "runs"),
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1
