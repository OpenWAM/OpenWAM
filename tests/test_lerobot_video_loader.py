from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import open_wam.data.lerobot_video as lerobot_video_module
from open_wam.configs import (
    ActionMappingConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    RobotWinDataConfig,
    ViewLayoutConfig,
)
from open_wam.data import build_train_val_datasets, collate_wam_samples
from open_wam.data.raw_video import build_canonical_video_preprocessor


ROBOTWIN_CAMERA_NAMES = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def _write_robotwin_lerobot_video_fixture(root: Path, *, num_frames: int = 6) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    for camera_name in ROBOTWIN_CAMERA_NAMES:
        (root / "videos" / "chunk-000" / camera_name).mkdir(parents=True)

    info = {
        "codebase_version": "v2.1",
        "fps": 50,
        "chunks_size": 1000,
        "total_episodes": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"shape": [16], "dtype": "float32"},
            "action": {"shape": [16], "dtype": "float32"},
            **{
                camera_name: {
                    "shape": [16, 16, 3],
                    "dtype": "video",
                }
                for camera_name in ROBOTWIN_CAMERA_NAMES
            },
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": num_frames, "tasks": ["pick up the bottle"]}) + "\n",
        encoding="utf-8",
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick up the bottle"}) + "\n",
        encoding="utf-8",
    )

    rows = []
    for frame_index in range(num_frames):
        rows.append(
            {
                "observation.state": np.arange(16, dtype=np.float32) + frame_index,
                "action": np.arange(16, dtype=np.float32) + frame_index,
                "timestamp": float(frame_index) / 50.0,
                "frame_index": frame_index,
                "episode_index": 0,
                "index": frame_index,
                "task_index": 0,
            }
        )
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")

    for camera_offset, camera_name in enumerate(ROBOTWIN_CAMERA_NAMES):
        frames = [
            np.full((16, 16, 3), frame_index + camera_offset * 32, dtype=np.uint8)
            for frame_index in range(num_frames)
        ]
        imageio.mimsave(
            root / "videos" / "chunk-000" / camera_name / "episode_000000.mp4",
            frames,
            fps=50,
            macro_block_size=1,
        )


def _robotwin_video_config(root: Path) -> RobotWinDataConfig:
    return RobotWinDataConfig(
        dataset_type="lerobot_v2_video",
        local_root=str(root),
        repo_id=None,
        camera_names=ROBOTWIN_CAMERA_NAMES,
        view_layout=(
            ViewLayoutConfig(
                source_name="observation.images.cam_high",
                canonical_name="cam_high",
                top=0,
                left=0,
                height=256,
                width=320,
            ),
            ViewLayoutConfig(
                source_name="observation.images.cam_left_wrist",
                canonical_name="cam_left_wrist",
                top=256,
                left=0,
                height=128,
                width=160,
            ),
            ViewLayoutConfig(
                source_name="observation.images.cam_right_wrist",
                canonical_name="cam_right_wrist",
                top=256,
                left=160,
                height=128,
                width=160,
            ),
        ),
        num_frames=2,
        action_schema=ActionSchemaConfig(
            action_dim=30,
            action_horizon=3,
            state_dim=16,
            state_horizon=1,
        ),
        action_target=ActionTargetConfig(
            representation="raw",
            source_key="action",
            pose_source_key="observation.state",
        ),
        action_mapping=ActionMappingConfig(
            mode="sparse_canvas",
            source_dim=16,
            target_dim=30,
            source_to_target_indices=(0, 1, 2, 3, 4, 5, 6, 28, 7, 8, 9, 10, 11, 12, 13, 29),
            active_target_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 28, 29),
            loss_mask_mode="active_target_indices",
        ),
    )


def test_lerobot_v2_video_loader_decodes_external_mp4_and_maps_robotwin_actions(tmp_path: Path) -> None:
    _write_robotwin_lerobot_video_fixture(tmp_path)
    config = _robotwin_video_config(tmp_path)

    train_dataset, val_dataset = build_train_val_datasets(config)
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0

    sample = train_dataset[0]
    assert set(sample.views) == set(ROBOTWIN_CAMERA_NAMES)
    assert sample.views["observation.images.cam_high"].shape == (2, 16, 16, 3)
    assert sample.actions.shape == (3, 30)
    assert sample.action_mask is not None
    assert sample.action_mask[:, 0:14].sum().item() == 42.0
    assert sample.action_mask[:, 28:30].sum().item() == 6.0
    assert sample.action_mask[:, 14:28].sum().item() == 0.0
    assert sample.metadata["action_mapping_mode"] == "sparse_canvas"
    assert sample.task_text == "pick up the bottle"

    batch = collate_wam_samples([sample])
    canonical = build_canonical_video_preprocessor(config)(batch.views)
    assert canonical.video.shape == (1, 3, 2, 384, 320)


def test_lerobot_v2_video_loader_reuses_decoded_video_cache(tmp_path: Path, monkeypatch) -> None:
    _write_robotwin_lerobot_video_fixture(tmp_path)
    config = _robotwin_video_config(tmp_path)
    open_count = 0
    original_get_reader = lerobot_video_module.imageio.get_reader

    def counting_get_reader(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_get_reader(*args, **kwargs)

    monkeypatch.setattr(lerobot_video_module.imageio, "get_reader", counting_get_reader)
    train_dataset, _ = build_train_val_datasets(config)

    _ = train_dataset[0]
    _ = train_dataset[1]

    assert open_count == len(ROBOTWIN_CAMERA_NAMES)
