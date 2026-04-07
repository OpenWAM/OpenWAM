from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from open_wam.configs import WindowSamplingMode
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
    total_rows: int = 20,
    latent_num_frames: int = 4,
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
    for camera_name, (latent_height, latent_width) in latent_specs.items():
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
            / f"episode_000000_0_{latent_num_frames}.pth"
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
    assert sample.metadata["observation_start"] == 0
    assert sample.metadata["observation_frame_indices"] == [0, 1, 2, 3]


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
            config.data,
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
            config.data,
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
            config.data,
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
            config.data,
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


def test_local_lerobot_latent_dataset_supports_random_subwindow_sampling(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent_random"
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
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.RANDOM_SUBWINDOW,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.video_latents.shape == (48, config.data.num_frames, 8, 16)
    assert sample.actions.shape == (config.data.action_schema.action_horizon, 7)
    assert sample.state.shape == (config.data.action_schema.state_horizon, 8)
    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.RANDOM_SUBWINDOW
    assert sample.metadata["segment_start_frame"] == 0
    assert sample.metadata["segment_end_frame"] == 4
    assert sample.metadata["sample_start_frame"] >= sample.metadata["segment_start_frame"]
    assert sample.metadata["sample_end_frame"] <= sample.metadata["segment_end_frame"]
    assert len(sample.metadata["observed_frame_ids"]) == config.data.num_frames


def test_local_lerobot_latent_dataset_supports_contextual_subwindow_sampling(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent_contextual"
    _build_local_robotwin_latent_repo(
        repo_root,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
        total_rows=48,
        latent_num_frames=10,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.CONTEXTUAL_SUBWINDOW,
                num_frames=4,
                action_horizon=16,
                state_horizon=1,
                chunk_size=2,
                window_size=4,
                predict_blocks_per_sample=1,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.CONTEXTUAL_SUBWINDOW
    sampled_chunk_size = int(sample.metadata["sampled_chunk_size"])
    sampled_window_size = int(sample.metadata["sampled_window_size"])
    expected_history_frames = max(1, (sampled_window_size + 1) // 2) * sampled_chunk_size
    expected_current_frames = sampled_chunk_size
    expected_total_frames = expected_history_frames + expected_current_frames

    assert sampled_chunk_size in {1, 2}
    assert sampled_window_size == 4
    assert sample.video_latents.shape == (48, expected_total_frames, 8, 16)
    assert sample.actions.shape == (expected_total_frames * 4, 7)
    assert sample.metadata["history_frames"] == expected_history_frames
    assert sample.metadata["current_frames"] == expected_current_frames
    assert sample.metadata["loss_frame_start"] == expected_history_frames
    assert sample.metadata["loss_frame_end"] == expected_total_frames
    assert sample.metadata["frame_shift"] == sample.metadata["sample_start_frame"]
    assert len(sample.metadata["observed_frame_ids"]) == expected_total_frames


def test_contextual_subwindow_sampling_falls_back_to_geometry_that_fits_segment(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent_contextual_short"
    _build_local_robotwin_latent_repo(
        repo_root,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
        total_rows=96,
        latent_num_frames=42,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.CONTEXTUAL_SUBWINDOW,
                num_frames=4,
                action_horizon=16,
                state_horizon=1,
                chunk_size=4,
                window_size=64,
                predict_blocks_per_sample=1,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.CONTEXTUAL_SUBWINDOW
    assert sample.video_latents.shape[1] <= 42
    assert sample.metadata["sampled_chunk_size"] <= 4
    assert sample.metadata["sampled_window_size"] <= 64
    assert sample.metadata["loss_frame_end"] <= sample.video_latents.shape[1]


def test_contextual_subwindow_sampling_can_use_fixed_geometry(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent_contextual_fixed"
    _build_local_robotwin_latent_repo(
        repo_root,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
        total_rows=96,
        latent_num_frames=42,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                mode=WindowSamplingMode.CONTEXTUAL_SUBWINDOW,
                num_frames=4,
                action_horizon=16,
                state_horizon=1,
                chunk_size=2,
                window_size=8,
                predict_blocks_per_sample=1,
                randomize_geometry=False,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.metadata["sampled_chunk_size"] == 2
    assert sample.metadata["sampled_window_size"] == 8
    assert sample.metadata["history_frames"] == 8
    assert sample.metadata["current_frames"] == 2
    assert sample.video_latents.shape[1] == 10


def test_aligned_subwindow_sampling_uses_sample_construction_horizons_and_stride(tmp_path: Path) -> None:
    repo_root = tmp_path / "libero_local_latent_aligned"
    _build_local_robotwin_latent_repo(
        repo_root,
        state_key="observation.state",
        action_dim=7,
        state_dim=8,
        camera_names=("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
        total_rows=32,
        latent_num_frames=8,
    )

    config = load_experiment_config(REPO_ROOT / "configs/experiments/register_attached_libero_latent_local.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            local_root=str(repo_root),
            empty_text_embedding_path=None,
            train_fraction=1.0,
            num_workers=0,
            train_batch_size=1,
            val_batch_size=1,
            sample_construction=replace(
                config.data.sample_construction,
                num_frames=4,
                action_horizon=3,
                state_horizon=3,
                frame_stride=2,
            ),
        ),
    )

    train_dataset, _ = build_train_val_latent_datasets(config.data)
    sample = train_dataset[0]

    assert sample.video_latents.shape == (48, 4, 8, 16)
    assert sample.actions.shape == (config.data.action_schema.action_horizon, 7)
    assert sample.state.shape == (config.data.action_schema.state_horizon, 8)
    assert sample.action_mask[:3].sum().item() == 3 * 7
    assert sample.action_mask[3:].sum().item() == 0
    assert sample.state_mask.sum().item() == 3 * 8
    assert sample.metadata["window_sampling_mode"] == WindowSamplingMode.ALIGNED_SUBWINDOW
    assert len(sample.metadata["observation_frame_indices"]) == 4
    assert sample.metadata["observation_frame_indices"][1] - sample.metadata["observation_frame_indices"][0] == 2
    assert sample.metadata["state_indices"] == tuple(
        range(sample.metadata["sample_start_frame"], sample.metadata["sample_start_frame"] + 3)
    )


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
            config.data,
            local_root=str(repo_root),
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
    sample = train_dataset[0]

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
