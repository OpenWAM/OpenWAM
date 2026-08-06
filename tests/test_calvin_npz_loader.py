from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from open_wam.configs import (
    ActionMappingConfig,
    ActionSchemaConfig,
    CalvinDataConfig,
)
from open_wam.data import build_train_val_datasets, collate_wam_samples
from open_wam.data.raw_video import build_canonical_video_preprocessor


def _write_calvin_fixture(root: Path, *, num_steps: int = 10) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(num_steps):
        np.savez(
            root / f"episode_{index:07d}.npz",
            rgb_static=np.full((200, 200, 3), index, dtype=np.uint8),
            rgb_gripper=np.full((84, 84, 3), 255 - index, dtype=np.uint8),
            rel_actions=np.arange(7, dtype=np.float32) + index,
            actions=np.arange(7, dtype=np.float32) - index,
            robot_obs=np.arange(15, dtype=np.float32) + index,
            scene_obs=np.arange(24, dtype=np.float32),
        )
    np.save(root / "ep_start_end_ids.npy", np.asarray([[0, 4], [5, 9]], dtype=np.int64))
    lang_dir = root / "lang_annotations"
    lang_dir.mkdir()
    np.save(
        lang_dir / "auto_lang_ann.npy",
        {
            "language": {"ann": ["open the drawer"]},
            "info": {"indx": np.asarray([[0, 9]], dtype=np.int64)},
        },
    )


def test_calvin_npz_loader_emits_raw_7d_actions_and_canonical_views(tmp_path: Path) -> None:
    _write_calvin_fixture(tmp_path)
    config = CalvinDataConfig(
        local_root=str(tmp_path),
        num_frames=2,
        action_schema=ActionSchemaConfig(
            action_dim=7,
            action_horizon=3,
            state_dim=15,
            state_horizon=1,
        ),
        train_fraction=0.5,
        language_annotation_pickle_policy="trusted_legacy",
    )

    train_dataset, val_dataset = build_train_val_datasets(config)
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0

    sample = train_dataset[0]
    assert set(sample.views) == {"rgb_static", "rgb_gripper"}
    assert sample.views["rgb_static"].shape == (2, 200, 200, 3)
    assert sample.views["rgb_gripper"].shape == (2, 84, 84, 3)
    assert sample.actions.shape == (3, 7)
    assert sample.action_mask is not None
    assert sample.action_mask.sum().item() == 21.0
    assert sample.state is not None
    assert sample.state.shape == (1, 15)
    assert sample.task_text == "open the drawer"

    batch = collate_wam_samples([sample])
    canonical = build_canonical_video_preprocessor(config)(batch.views)
    assert canonical.video.shape == (1, 3, 2, 384, 320)


def test_calvin_npz_loader_maps_rel_actions_to_sparse_30d(tmp_path: Path) -> None:
    _write_calvin_fixture(tmp_path)
    config = CalvinDataConfig(
        local_root=str(tmp_path),
        num_frames=2,
        action_schema=ActionSchemaConfig(
            action_dim=30,
            action_horizon=3,
            state_dim=15,
            state_horizon=1,
        ),
        action_mapping=ActionMappingConfig(
            mode="sparse_canvas",
            source_dim=7,
            target_dim=30,
            source_to_target_indices=(0, 1, 2, 3, 4, 5, 28),
            active_target_indices=(0, 1, 2, 3, 4, 5, 28),
            loss_mask_mode="active_target_indices",
            sampler_mask_mode="pin_inactive_channels",
        ),
        train_fraction=0.5,
        language_annotation_pickle_policy="trusted_legacy",
    )

    train_dataset, _ = build_train_val_datasets(config)
    sample = train_dataset[0]

    assert sample.actions.shape == (3, 30)
    assert sample.action_mask is not None
    assert torch.equal(sample.actions[:, 0:6], torch.stack([torch.arange(6) + value for value in sample.actions[:, 0]]))
    assert sample.action_mask[:, 0:6].sum().item() == 18.0
    assert sample.action_mask[:, 28].sum().item() == 3.0
    assert sample.action_mask[:, 6:28].sum().item() == 0.0
    assert sample.action_mask[:, 29].sum().item() == 0.0
    assert sample.metadata["action_mapping_mode"] == "sparse_canvas"


def test_calvin_npz_loader_rejects_pickled_annotations_by_default(
    tmp_path: Path,
) -> None:
    _write_calvin_fixture(tmp_path)
    config = CalvinDataConfig(local_root=str(tmp_path))

    with pytest.raises(ValueError, match="language_annotation_pickle_policy=trusted_legacy"):
        build_train_val_datasets(config)
