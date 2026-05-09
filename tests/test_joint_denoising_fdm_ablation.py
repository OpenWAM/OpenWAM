from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from open_wam.ablations.joint_denoising_fdm.branches import (
    BRANCH_PRESETS,
    apply_action_branch,
    branch_metadata,
    expand_branch_names,
)
from open_wam.ablations.joint_denoising_fdm.counterfactual import (
    _decoded_raw_frames_for_latents,
    _raw_window_frames_for_latents,
    _should_drop_text_conditioning,
)
from open_wam.ablations.joint_denoising_fdm.metrics import (
    latent_mse_per_frame,
    rgb_mse_per_frame,
    summarize_metric_rows,
)
from open_wam.ablations.joint_denoising_fdm.sampling import select_early_middle_windows
from open_wam.ablations.joint_denoising_fdm.types import FdmAblationMode, FdmStartPolicy, FdmWindowSelection


def _load_repo_script(relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class DummyWindow:
    repo_root: str
    episode_index: int
    start_frame: int
    end_frame: int
    observed_frame_ids: tuple[int, ...]
    task_text: str


class DummyDataset:
    def __init__(self) -> None:
        self.windows = [
            DummyWindow("repo", 0, 0, 100, tuple(range(100)), "task b"),
            DummyWindow("repo", 1, 0, 80, tuple(range(80)), "task a"),
            DummyWindow("repo", 2, 0, 90, tuple(range(90)), "task a"),
            DummyWindow("repo", 3, 0, 95, tuple(range(95)), "task b"),
        ]

    def _window_task_text(self, window: DummyWindow) -> str:
        return window.task_text


def test_select_early_middle_windows_is_deterministic_and_chunk_aligned() -> None:
    first = select_early_middle_windows(
        DummyDataset(),
        horizon_frames=8,
        frame_chunk_size=4,
        trajectories_per_task=2,
        seed=7,
    )
    second = select_early_middle_windows(
        DummyDataset(),
        horizon_frames=8,
        frame_chunk_size=4,
        trajectories_per_task=2,
        seed=7,
    )
    assert first == second
    assert len(first) == 4
    assert {item.task_key for item in first} == {"task a", "task b"}
    for selection in first:
        assert selection.generated_frames == 8
        assert selection.t0_frame >= 4
        assert selection.generation_end_frame <= selection.total_video_frames
        assert selection.target_end_frame <= selection.total_video_frames


def test_select_early_middle_windows_rejects_non_chunk_aligned_horizon() -> None:
    with pytest.raises(ValueError, match="exact multiple"):
        select_early_middle_windows(
            DummyDataset(),
            horizon_frames=6,
            frame_chunk_size=4,
            trajectories_per_task=1,
            seed=7,
        )


def test_latest_fit_start_policy_uses_full_target_horizon() -> None:
    selections = select_early_middle_windows(
        DummyDataset(),
        horizon_frames=32,
        frame_chunk_size=4,
        trajectories_per_task=1,
        seed=7,
        start_policy=FdmStartPolicy.LATEST_FIT,
    )
    assert len(selections) == 2
    for selection in selections:
        assert selection.t0_frame == selection.total_video_frames - selection.horizon_frames
        assert selection.target_end_frame == selection.total_video_frames


def test_counterfactual_action_branches_preserve_gripper_and_clip() -> None:
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(-0.2, 0.2, num=8, dtype=np.float32)
    actions[:, 1] = 0.3
    actions[:, 2] = -0.4
    actions[:, 6] = 0.75

    gt = apply_action_branch(actions, branch_name="gt", seed=0)
    assert np.array_equal(gt, actions)

    biased = apply_action_branch(actions, branch_name="bias_x_pos", seed=0)
    assert np.allclose(biased[:, 0], actions[:, 0] + 0.12)
    assert np.allclose(biased[:, 6], actions[:, 6])

    noisy = apply_action_branch(actions, branch_name="noise_small", seed=123)
    assert not np.array_equal(noisy[:, :6], actions[:, :6])
    assert np.allclose(noisy[:, 6], actions[:, 6])
    assert np.max(np.abs(noisy[:, :6])) <= 1.0

    stopped = apply_action_branch(actions, branch_name="stop_motion", seed=0)
    assert np.allclose(stopped[:, :6], 0.0)
    assert np.allclose(stopped[:, 6], actions[:, 6])

    reversed_translation = apply_action_branch(actions, branch_name="reverse_translation", seed=0)
    assert np.allclose(reversed_translation[:, :3], -actions[:, :3])
    assert np.allclose(reversed_translation[:, 6], actions[:, 6])

    swapped = apply_action_branch(actions, branch_name="swap_xy_clockwise", seed=0)
    assert np.allclose(swapped[:, 0], actions[:, 1])
    assert np.allclose(swapped[:, 1], -actions[:, 0])
    assert np.allclose(swapped[:, 6], actions[:, 6])

    saturated = apply_action_branch(actions, branch_name="saturate_z_up", seed=0)
    assert np.allclose(saturated[:, 0], 0.0)
    assert np.allclose(saturated[:, 1], 0.0)
    assert np.allclose(saturated[:, 2], 1.0)
    assert np.allclose(saturated[:, 3:6], 0.0)
    assert np.allclose(saturated[:, 6], actions[:, 6])

    scaled = apply_action_branch(actions, branch_name="scale_demo_0p5", seed=0)
    assert np.allclose(scaled[:, :6], actions[:, :6] * 0.5)
    assert np.allclose(scaled[:, 6], actions[:, 6])

    pulse = apply_action_branch(actions, branch_name="axis_pulse_x_neg", seed=0)
    assert np.allclose(pulse[:4, 0], -0.8)
    assert np.allclose(pulse[4:, 0], 0.0)
    assert np.allclose(pulse[:, 6], actions[:, 6])


def test_branch_presets_expand_and_record_metadata() -> None:
    expanded = expand_branch_names("training_10,gt")
    assert expanded == BRANCH_PRESETS["training_10"]
    metadata = branch_metadata("axis_pulse_x_neg")
    assert metadata["family"] == "axis_pulse"
    assert metadata["strength"] == "strong"


def test_counterfactual_wan_temporal_window_formulas() -> None:
    assert _raw_window_frames_for_latents(4, action_per_frame=4) == 15
    assert _raw_window_frames_for_latents(16, action_per_frame=4) == 63
    assert _decoded_raw_frames_for_latents(4, action_per_frame=4) == 13
    assert _decoded_raw_frames_for_latents(16, action_per_frame=4) == 61


def test_counterfactual_dataset_builder_excludes_manifest_source_episodes(tmp_path: Path) -> None:
    builder = _load_repo_script("scripts/build_libero_fdm_counterfactual_demo_dataset.py")
    excluded_root = tmp_path / "prior_dataset"
    excluded_root.mkdir()
    (excluded_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_episodes": [
                    {"dataset_episode_index": 1},
                    {"dataset_episode_index": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    replay_rows = [
        {
            "replay_status": "success",
            "failure": False,
            "dataset_episode_index": index,
            "metadata_task_index": 0,
            "resolved_init_state_index": 0,
            "parquet_path": f"/tmp/episode_{index}.parquet",
            "task_text": "task",
        }
        for index in (1, 2, 3, 4)
    ]

    excluded = builder._load_excluded_episode_indices([str(excluded_root)])
    selected = builder._select_source_episodes(
        replay_rows,
        task_ids=(0,),
        episodes_per_task=2,
        seed=0,
        excluded_episode_indices=excluded,
    )

    assert excluded == {1, 3}
    assert [episode.dataset_episode_index for episode in selected] == [2, 4]


def test_counterfactual_dataset_builder_plan_only_overwrite_does_not_delete_before_validation(tmp_path: Path) -> None:
    builder = _load_repo_script("scripts/build_libero_fdm_counterfactual_demo_dataset.py")
    output_dir = tmp_path / "outputs"
    run_id = "existing"
    output_root = output_dir / run_id
    output_root.mkdir(parents=True)
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "replay_status": "success",
                "failure": False,
                "dataset_episode_index": 7,
                "metadata_task_index": 0,
                "resolved_init_state_index": 0,
                "parquet_path": "/tmp/episode_7.parquet",
                "task_text": "task",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target-transitions"):
        builder.main(
            [
                "--replay-status-path",
                str(replay_path),
                "--output-dir",
                str(output_dir),
                "--run-id",
                run_id,
                "--task-ids",
                "0",
                "--episodes-per-task",
                "1",
                "--t0-fractions",
                "0.5",
                "--branches",
                "gt",
                "--target-transitions",
                "2",
                "--plan-only",
                "--overwrite",
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_counterfactual_encoder_accepts_single_root_dataset(tmp_path: Path) -> None:
    encoder = _load_repo_script("scripts/encode_libero_fdm_counterfactual_dataset.py")
    dataset_root = tmp_path / "split"
    (dataset_root / "metadata").mkdir(parents=True)
    (dataset_root / "metadata" / "contexts.jsonl").write_text("", encoding="utf-8")
    (dataset_root / "metadata" / "transitions.jsonl").write_text("", encoding="utf-8")

    assert encoder._resolve_shards(dataset_root, None) == [dataset_root]


def test_counterfactual_encoder_rejects_existing_latents_without_overwrite(tmp_path: Path) -> None:
    encoder = _load_repo_script("scripts/encode_libero_fdm_counterfactual_dataset.py")
    dataset_root = tmp_path / "split"
    (dataset_root / "metadata").mkdir(parents=True)
    (dataset_root / "metadata" / "contexts.jsonl").write_text(
        json.dumps({"context_id": 0}) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "metadata" / "transitions.jsonl").write_text(
        json.dumps({"sample_id": 0}) + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "encoded"
    existing = output_root / dataset_root.name / "contexts" / "context_000000_latents.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="--overwrite"):
        encoder._validate_no_existing_encoded_outputs(
            output_root=output_root,
            shard_roots=[dataset_root],
            max_contexts=None,
            max_samples=None,
        )


def test_counterfactual_encoder_uses_reference_asset_streaming_encode() -> None:
    encoder = _load_repo_script("scripts/encode_libero_fdm_counterfactual_dataset.py")

    class FakeAssets:
        def __init__(self) -> None:
            self.calls = []

        def encode_video(self, video, *, placements=None, reset_cache=True):
            self.calls.append((video, placements, reset_cache))
            return torch.ones(video.shape[0], 4, video.shape[2], 8, 16)

    assets = FakeAssets()
    video = torch.zeros(2, 3, 4, 128, 256)

    latents = encoder._encode_libero_side_by_side_video(
        assets,
        video,
        device=torch.device("cpu"),
    )

    assert latents.shape == (2, 4, 4, 8, 16)
    assert len(assets.calls) == 1
    _, placements, reset_cache = assets.calls[0]
    assert reset_cache is True
    assert tuple(placement.canonical_name for placement in placements) == ("image", "wrist_image")


def test_counterfactual_drop_text_flag_applies_to_fdm_modes_only() -> None:
    assert _should_drop_text_conditioning(
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        fdm_drop_text_conditioning=True,
    )
    assert _should_drop_text_conditioning(
        FdmAblationMode.CLEAN_ACTION_FEEDBACK,
        fdm_drop_text_conditioning=True,
    )
    assert not _should_drop_text_conditioning(
        FdmAblationMode.VANILLA_JOINT_ROLLOUT,
        fdm_drop_text_conditioning=True,
    )
    assert not _should_drop_text_conditioning(
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        fdm_drop_text_conditioning=False,
    )


def test_latent_and_rgb_mse_per_frame() -> None:
    predicted_latents = torch.zeros(1, 2, 3, 2, 2)
    target_latents = torch.ones(1, 2, 3, 2, 2)
    assert latent_mse_per_frame(predicted_latents, target_latents) == [1.0, 1.0, 1.0]

    predicted_rgb = np.zeros((2, 2, 2, 3), dtype=np.float32)
    target_rgb = np.ones((2, 2, 2, 3), dtype=np.float32)
    assert rgb_mse_per_frame(predicted_rgb, target_rgb) == [1.0, 1.0]


def test_summarize_metric_rows_groups_by_mode_and_horizon() -> None:
    selection = FdmWindowSelection(
        sample_index=0,
        dataset_index=0,
        task_key="task",
        task_rank=0,
        episode_index=0,
        t0_frame=10,
        horizon_frames=2,
        generated_frames=4,
        context_start_frame=0,
        total_video_frames=20,
        repo_root="repo",
    )
    rows = [
        {
            **selection.__dict__,
            "mode": FdmAblationMode.FORCED_ACTION_JOINT_FDM.value,
            "horizon_index": 0,
            "latent_mse": 1.0,
            "rgb_mse": 0.25,
        },
        {
            **selection.__dict__,
            "mode": FdmAblationMode.FORCED_ACTION_JOINT_FDM.value,
            "horizon_index": 0,
            "latent_mse": 3.0,
            "rgb_mse": 0.75,
        },
    ]
    summary = summarize_metric_rows(rows)
    assert summary == [
        {
            "mode": FdmAblationMode.FORCED_ACTION_JOINT_FDM.value,
            "horizon_index": 0,
            "count": 2,
            "latent_mse_mean": 2.0,
            "latent_mse_std": 1.0,
            "rgb_mse_mean": 0.5,
            "rgb_mse_std": 0.25,
        }
    ]
