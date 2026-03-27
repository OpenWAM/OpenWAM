from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from open_wam.data import build_train_val_latent_datasets
from open_wam.training import TrainingRuntime
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


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
        [{"episode_index": 0, "length": 20, "tasks": ["pick up block"]}],
    )
    _write_jsonl(
        repo_root / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": "pick up block"}],
    )

    rows = []
    for frame_index in range(20):
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
    for camera_name, (latent_height, latent_width) in latent_specs.items():
        flat_latents = torch.randn(4 * latent_height * latent_width, 48)
        payload = {
            "latent": flat_latents,
            "latent_num_frames": 4,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "frame_ids": [0, 1, 2, 3],
        }
        latent_path = (
            repo_root
            / "latents"
            / "chunk-000"
            / camera_name
            / "episode_000000_0_4.pth"
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
            config.data,
            dataset_type="lerobot_v2_latent_local",
            local_root=str(repo_root),
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
        ),
    )

    train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert sample.video_latents.shape == (48, 4, 24, 20)
    assert sample.actions.shape == (8, 30)
    assert sample.state.shape == (1, 30)
    assert sample.text_context is None
    assert sample.metadata["observed_frame_ids"] == [0, 1, 2, 3]


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
            config.data,
            local_root=str(repo_root),
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


def test_parallel_stream_runtime_runs_on_local_lerobot_latent_dataset(tmp_path: Path) -> None:
    repo_root = tmp_path / "robotwin_local_latent"
    _build_local_robotwin_latent_repo(repo_root)

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        training=replace(config.training, num_steps=1),
        data=replace(
            config.data,
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
