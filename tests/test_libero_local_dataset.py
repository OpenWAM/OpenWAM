from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from open_wam.configs import LiberoDataConfig
from open_wam.data import build_train_val_datasets


def _write_demo(group: h5py.Group, *, length: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    obs = group.create_group("obs")
    obs.create_dataset("agentview_rgb", data=rng.integers(0, 255, size=(length, 8, 8, 3), dtype=np.uint8))
    obs.create_dataset("eye_in_hand_rgb", data=rng.integers(0, 255, size=(length, 8, 8, 3), dtype=np.uint8))
    obs.create_dataset("ee_pos", data=rng.normal(size=(length, 3)).astype(np.float32))
    obs.create_dataset("ee_ori", data=rng.normal(size=(length, 3)).astype(np.float32))
    obs.create_dataset("gripper_states", data=rng.normal(size=(length, 2)).astype(np.float32))
    group.create_dataset("actions", data=rng.normal(size=(length, 7)).astype(np.float32))
    group.create_dataset("dones", data=np.zeros((length,), dtype=np.uint8))
    group.create_dataset("rewards", data=np.zeros((length,), dtype=np.uint8))
    group.create_dataset("robot_states", data=rng.normal(size=(length, 9)).astype(np.float32))
    group.create_dataset("states", data=rng.normal(size=(length, 51)).astype(np.float32))


def _write_libero_file(path: Path, *, demo_count: int, seed_offset: int) -> None:
    with h5py.File(path, "w") as handle:
        data_group = handle.create_group("data")
        for index in range(demo_count):
            _write_demo(
                data_group.create_group(f"demo_{index}"),
                length=12 + index,
                seed=seed_offset + index,
            )


def test_local_libero_hdf5_dataset_builds_train_val_windows(tmp_path: Path) -> None:
    dataset_root = tmp_path / "libero_10"
    dataset_root.mkdir()
    _write_libero_file(
        dataset_root / "KITCHEN_SCENE1_put_the_red_mug_on_the_table_demo.hdf5",
        demo_count=2,
        seed_offset=0,
    )
    _write_libero_file(
        dataset_root / "LIVING_ROOM_SCENE2_pick_up_the_book_demo.hdf5",
        demo_count=2,
        seed_offset=10,
    )

    data_config = LiberoDataConfig(
        dataset_type="libero_hdf5",
        repo_id=None,
        local_root=str(dataset_root),
        train_fraction=0.5,
    )
    train_dataset, val_dataset = build_train_val_datasets(data_config)

    assert len(train_dataset) > 0
    assert len(val_dataset) > 0

    sample = train_dataset[0]
    assert sample.views["image"].shape == (data_config.num_frames, 8, 8, 3)
    assert sample.views["wrist_image"].shape == (data_config.num_frames, 8, 8, 3)
    assert sample.actions.shape == (
        data_config.action_schema.action_horizon,
        data_config.action_schema.action_dim,
    )
    assert sample.action_mask is not None
    assert sample.action_mask.shape == sample.actions.shape
    assert sample.state is not None
    assert sample.state.shape == (
        data_config.action_schema.state_horizon,
        data_config.action_schema.state_dim,
    )
    assert sample.state_mask is not None
    assert sample.state_mask.shape == sample.state.shape
    assert sample.task_text in {
        "put the red mug on the table",
        "pick up the book",
    }
    assert sample.metadata["dataset_source"] == "libero_hdf5"
    assert sample.metadata["local_root"] == str(dataset_root.resolve())
