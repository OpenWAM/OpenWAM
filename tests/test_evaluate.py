from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import Dataset

import open_wam.evals.evaluate as evaluate_module
from open_wam.data import WAMSample
from open_wam.evals.evaluate import resolve_evaluation_request, run_evaluation
from open_wam.models.policy_variants.contracts import DecoderSequenceContext, VideoConditionWindowContext
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_eval_wrapper_resolves_experiment_config() -> None:
    request = resolve_evaluation_request(REPO_ROOT / "configs/evals/contract_only_robotwin.yaml")
    assert request.experiment_config_path == (REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml").resolve()
    assert request.mode == "batch"
    assert request.split == "val"
    assert request.max_batches == 1


def test_eval_wrapper_resolves_checkpoint_path_placeholder(monkeypatch, tmp_path: Path) -> None:
    local_paths_path = tmp_path / "local_paths.yaml"
    with local_paths_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"paths": {"tests": {"eval_checkpoint": "/tmp/eval_checkpoint.ckpt"}}},
            handle,
            sort_keys=False,
        )
    monkeypatch.setenv("OPEN_WAM_LOCAL_PATHS", str(local_paths_path))

    wrapper_path = tmp_path / "eval_wrapper.yaml"
    with wrapper_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "experiment_config": str(REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml"),
                "checkpoint_path": "${paths.tests.eval_checkpoint}",
                "mode": "batch",
                "split": "val",
            },
            handle,
            sort_keys=False,
        )

    request = resolve_evaluation_request(wrapper_path)

    assert request.checkpoint_path == Path("/tmp/eval_checkpoint.ckpt")


def test_method4_video_conditioned_eval_wrappers_resolve_experiment_configs() -> None:
    post_latent_request = resolve_evaluation_request(
        REPO_ROOT / "configs/evals/post_latent_libero_latent_local_video_conditioned_trajectory.yaml"
    )
    post_decoded_request = resolve_evaluation_request(
        REPO_ROOT / "configs/evals/post_decoded_libero_latent_local_video_conditioned_trajectory.yaml"
    )

    assert post_latent_request.experiment_config_path == (
        REPO_ROOT / "configs/experiments/post_latent_libero_latent_local_video_conditioned.yaml"
    ).resolve()
    assert post_decoded_request.experiment_config_path == (
        REPO_ROOT / "configs/experiments/post_decoded_libero_latent_local_video_conditioned.yaml"
    ).resolve()
    assert post_latent_request.mode == "trajectory"
    assert post_decoded_request.mode == "trajectory"
    assert post_latent_request.split == "val"
    assert post_decoded_request.split == "val"
    assert post_latent_request.batch_size == 1
    assert post_decoded_request.batch_size == 1
    assert post_latent_request.max_trajectories == 1
    assert post_decoded_request.max_trajectories == 1


@pytest.mark.parametrize(
    ("wrapper_name", "experiment_name", "checkpoint_alias_suffix"),
    [
        (
            "parallel_stream_libero_lingbot_exact_heng_eval.yaml",
            "parallel_stream_libero_lingbot_exact_heng_compatible.yaml",
            "parallel_stream_libero_lingbot_exact_heng_compatible_0402/checkpoints/checkpoint_step_1100/full_training_state.pt",
        ),
        (
            "parallel_stream_libero_lingbot_joint_denoise_heng_eval.yaml",
            "parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml",
            "parallel_stream_libero_lingbot_joint_denoise_heng_compatible_0402/checkpoints/checkpoint_step_600/full_training_state.pt",
        ),
        (
            "video_sequence_policy_libero_heng_eval.yaml",
            "video_sequence_policy_libero_latent_local_random_subwindow.yaml",
            "video_sequence_policy_libero_latent_local_random_subwindow_0402/checkpoints/checkpoint_step_800/model_state.pt",
        ),
    ],
)
def test_libero_heng_eval_wrappers_resolve_experiment_configs_and_checkpoints(
    wrapper_name: str,
    experiment_name: str,
    checkpoint_alias_suffix: str,
) -> None:
    request = resolve_evaluation_request(REPO_ROOT / "configs/evals" / wrapper_name)

    assert request.experiment_config_path == (REPO_ROOT / "configs/experiments" / experiment_name).resolve()
    assert request.mode == "batch"
    assert request.split == "val"
    assert request.batch_size == 1
    assert request.max_batches == 1
    assert request.checkpoint_path is not None
    assert request.checkpoint_path.as_posix().endswith(checkpoint_alias_suffix)


@pytest.mark.parametrize(
    ("wrapper_name", "experiment_name"),
    [
        ("parallel_stream_robotwin_smoke.yaml", "parallel_stream_robotwin_smoke.yaml"),
        ("register_attached_robotwin_smoke.yaml", "register_attached_robotwin_smoke.yaml"),
        ("video_sequence_policy_robotwin_smoke.yaml", "video_sequence_policy_robotwin_smoke.yaml"),
        ("mot_robotwin_smoke.yaml", "mot_robotwin_smoke.yaml"),
        ("causal_video_prediction_robotwin_smoke.yaml", "causal_video_prediction_robotwin_smoke.yaml"),
    ],
)
def test_robotwin_smoke_eval_wrappers_resolve_experiment_configs(
    wrapper_name: str,
    experiment_name: str,
) -> None:
    request = resolve_evaluation_request(REPO_ROOT / "configs/evals" / wrapper_name)

    assert request.experiment_config_path == (REPO_ROOT / "configs/experiments" / experiment_name).resolve()
    assert request.mode == "batch"
    assert request.split == "val"
    assert request.batch_size == 1
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
    assert summary.mean_video_latent_mse is None


def test_align_eval_action_tensors_tail_aligns_exact_raw_chunk_predictions() -> None:
    prediction = torch.arange(1 * 16 * 7, dtype=torch.float32).view(1, 16, 7)
    target = torch.arange(1 * 180 * 7, dtype=torch.float32).view(1, 180, 7)
    action_mask = torch.ones_like(target)

    source, aligned_prediction, aligned_target, aligned_mask = evaluate_module._align_eval_action_tensors(
        source="raw_chunk_action_pred",
        prediction=prediction,
        target_actions=target,
        action_mask=action_mask,
    )

    assert source == "raw_chunk_action_pred_tail_aligned"
    assert aligned_prediction.shape == (1, 16, 7)
    assert aligned_target.shape == (1, 16, 7)
    assert aligned_mask is not None
    assert torch.equal(aligned_target, target[:, -16:])
    assert torch.equal(aligned_mask, action_mask[:, -16:])


def test_select_eval_video_prediction_aligns_generated_local_future_latents() -> None:
    target = torch.arange(1 * 2 * 6 * 1 * 1, dtype=torch.float32).view(1, 2, 6, 1, 1)
    predicted = target[:, :, 2:5] + 0.5
    sequence_context = DecoderSequenceContext(
        sequence_tokens=torch.zeros(1, 1, 1),
        video_condition_window=VideoConditionWindowContext(
            local_window_tokens=torch.zeros(1, 4, 1, 1),
            observed_frame_count=1,
            metadata={
                "source_family": "generated_future_video_tokens",
                "observed_prefix_frames": 1,
                "observed_prefix_start_index": 1,
            },
        ),
    )

    source, aligned_prediction, aligned_target = evaluate_module._select_eval_video_prediction(
        target_video_latents=target,
        decoder_aux={},
        policy_aux={"predicted_latents": predicted},
        sequence_context=sequence_context,
    )

    assert source == "policy_predicted_local_future_latents"
    assert aligned_prediction is predicted
    assert torch.equal(aligned_target, target[:, :, 2:5])


def test_run_evaluation_on_parallel_stream_robotwin(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw.setdefault("inference", {})
    raw["inference"]["video_num_inference_steps"] = 2
    raw["inference"]["action_num_inference_steps"] = 2
    smoke_path = tmp_path / "parallel_stream_robotwin_eval_smoke.yaml"
    with smoke_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    request = resolve_evaluation_request(
        smoke_path,
        max_batches_override=1,
        device_override="cpu",
    )
    summary = run_evaluation(request)
    assert summary.experiment_name == "parallel_stream_robotwin_smoke"
    assert summary.num_batches == 1
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None


@pytest.mark.parametrize(
    ("config_path", "expected_name"),
    [
        (REPO_ROOT / "configs/evals/parallel_stream_robotwin_smoke.yaml", "parallel_stream_robotwin_smoke"),
        (REPO_ROOT / "configs/evals/register_attached_robotwin_smoke.yaml", "register_attached_robotwin_smoke"),
        (REPO_ROOT / "configs/evals/video_sequence_policy_robotwin_smoke.yaml", "video_sequence_policy_robotwin_smoke"),
        (REPO_ROOT / "configs/evals/mot_robotwin_smoke.yaml", "mot_robotwin_smoke"),
        (REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml", "post_latent_robotwin_video_conditioned"),
        (REPO_ROOT / "configs/experiments/post_decoded_robotwin_video_conditioned.yaml", "post_decoded_robotwin_video_conditioned"),
    ],
)
def test_run_evaluation_on_action_policy_robotwin_variants(config_path: Path, expected_name: str) -> None:
    request = resolve_evaluation_request(
        config_path,
        max_batches_override=1,
        device_override="cpu",
    )
    summary = run_evaluation(request)

    assert summary.experiment_name == expected_name
    assert summary.num_batches == 1
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None


def test_run_evaluation_on_causal_video_prediction_robotwin_wrapper() -> None:
    request = resolve_evaluation_request(
        REPO_ROOT / "configs/evals/causal_video_prediction_robotwin_smoke.yaml",
        device_override="cpu",
    )
    summary = run_evaluation(request)

    assert summary.experiment_name == "causal_video_prediction_robotwin_smoke"
    assert summary.num_batches == 1
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None
    assert summary.mean_video_latent_mse is not None


@dataclass(frozen=True)
class _EpisodeWindow:
    episode_index: int
    observation_start: int
    repo_root: str | None = None


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
    assert summary.mean_video_latent_mse is None
    assert summary.mean_trajectory_video_latent_mse is None


def test_group_dataset_indices_by_episode_uses_repo_root_identity() -> None:
    dataset = _TrajectoryEvalDataset()
    dataset.sample_index = (
        _EpisodeWindow(episode_index=0, observation_start=0, repo_root="/tmp/repo_a"),
        _EpisodeWindow(episode_index=0, observation_start=1, repo_root="/tmp/repo_a"),
        _EpisodeWindow(episode_index=0, observation_start=0, repo_root="/tmp/repo_b"),
        _EpisodeWindow(episode_index=0, observation_start=1, repo_root="/tmp/repo_b"),
    )

    groups = evaluate_module._group_dataset_indices_by_episode(dataset)

    assert groups == [[0, 1], [2, 3]]


def test_resolve_observation_frame_indices_prefers_metadata_list() -> None:
    indices = evaluate_module._resolve_observation_frame_indices(
        {
            "observation_start": 10,
            "observation_frame_indices": [4, 6, 8, 10],
        },
        num_frames=4,
    )

    assert indices == (4, 6, 8, 10)


def test_align_rollout_window_tensor_shifts_overlap_and_seeds_new_frames() -> None:
    previous = torch.tensor([[[10.0, 20.0, 30.0]]])
    current_target = torch.tensor([[[100.0, 200.0, 300.0]]])

    aligned = evaluate_module._align_rollout_window_tensor(
        previous,
        previous_frame_indices=(0, 1, 2),
        current_frame_indices=(1, 2, 3),
        current_target_tensor=current_target,
        frame_dim=2,
    )

    assert torch.equal(aligned, torch.tensor([[[20.0, 30.0, 300.0]]]))


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
    assert summary.mean_video_latent_mse is None
