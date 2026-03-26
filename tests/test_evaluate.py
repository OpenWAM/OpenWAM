from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

import open_wam.evals.evaluate as evaluate_module
from open_wam.data import WAMSample
from open_wam.evals.evaluate import resolve_evaluation_request, run_evaluation
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_eval_wrapper_resolves_experiment_config() -> None:
    request = resolve_evaluation_request(REPO_ROOT / "configs/evals/contract_only_robotwin.yaml")
    assert request.experiment_config_path == (REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml").resolve()
    assert request.mode == "batch"
    assert request.split == "val"
    assert request.max_batches == 1


def test_run_evaluation_on_contract_only_robotwin() -> None:
    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml",
        max_batches_override=1,
        device_override="cpu",
    )
    summary = run_evaluation(request)
    assert summary.experiment_name == "contract_only_robotwin"
    assert summary.num_batches == 1
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None


def test_run_evaluation_on_parallel_stream_robotwin() -> None:
    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin.yaml",
        max_batches_override=1,
        device_override="cpu",
    )
    summary = run_evaluation(request)
    assert summary.experiment_name == "parallel_stream_robotwin"
    assert summary.num_batches == 1
    assert summary.action_prediction_shape == summary.target_action_shape


@dataclass(frozen=True)
class _EpisodeWindow:
    episode_index: int
    observation_start: int


class _TrajectoryEvalDataset(Dataset[WAMSample]):
    def __init__(self) -> None:
        self.sample_index = (
            _EpisodeWindow(episode_index=0, observation_start=0),
            _EpisodeWindow(episode_index=0, observation_start=1),
            _EpisodeWindow(episode_index=1, observation_start=0),
            _EpisodeWindow(episode_index=1, observation_start=1),
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> WAMSample:
        window = self.sample_index[index]
        base_value = index + 1
        return WAMSample(
            views={
                "cam_high": torch.full((4, 256, 320, 3), fill_value=base_value, dtype=torch.uint8),
                "cam_left_wrist": torch.full((4, 128, 160, 3), fill_value=base_value, dtype=torch.uint8),
                "cam_right_wrist": torch.full((4, 128, 160, 3), fill_value=base_value, dtype=torch.uint8),
            },
            actions=torch.zeros(6, 30, dtype=torch.float32),
            action_mask=torch.ones(6, 30, dtype=torch.float32),
            state=torch.zeros(1, 30, dtype=torch.float32),
            state_mask=torch.ones(1, 30, dtype=torch.float32),
            task_text="synthetic trajectory eval",
            metadata={
                "episode_index": window.episode_index,
                "observation_start": window.observation_start,
            },
        )


def test_run_trajectory_evaluation_carries_across_episode_windows(monkeypatch) -> None:
    dataset = _TrajectoryEvalDataset()
    monkeypatch.setattr(
        evaluate_module,
        "build_train_val_datasets",
        lambda data_config: (dataset, dataset),
    )
    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml",
        mode_override="trajectory",
        max_trajectories_override=2,
        device_override="cpu",
    )
    summary = run_evaluation(request)
    assert summary.experiment_name == "contract_only_robotwin"
    assert summary.mode == "trajectory"
    assert summary.num_trajectories == 2
    assert summary.num_batches == 4
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None
    assert summary.mean_trajectory_action_mse is not None


def test_run_evaluation_loads_pipeline_prefixed_checkpoint(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml")
    pipeline = build_variant_pipeline_from_config(config)
    checkpoint_path = tmp_path / "contract_only_robotwin.ckpt"
    prefixed_state_dict = {f"pipeline.{key}": value for key, value in pipeline.state_dict().items()}
    torch.save({"state_dict": prefixed_state_dict}, checkpoint_path)

    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml",
        max_batches_override=1,
        checkpoint_override=str(checkpoint_path),
        device_override="cpu",
    )
    summary = run_evaluation(request)
    assert summary.experiment_name == "contract_only_robotwin"
    assert summary.checkpoint_path == str(checkpoint_path)
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None
