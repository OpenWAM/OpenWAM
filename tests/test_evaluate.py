from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import Dataset

import open_wam.evals.evaluate as evaluate_module
from open_wam.configs import load_experiment_config
from open_wam.data import WAMSample
from open_wam.evals.evaluate import resolve_evaluation_request, run_evaluation
from open_wam.pipelines import build_variant_pipeline_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_evaluate_pickle_globals_resolve_to_contract_owners() -> None:
    from open_wam.evals.evaluation_contracts import EvaluationRequest, EvaluationSummary

    request_global = pickle.loads(b"copen_wam.evals.evaluate\nEvaluationRequest\n.")
    summary_global = pickle.loads(b"copen_wam.evals.evaluate\nEvaluationSummary\n.")

    assert request_global is EvaluationRequest
    assert summary_global is EvaluationSummary


def test_eval_wrapper_resolves_experiment_config() -> None:
    request = resolve_evaluation_request(REPO_ROOT / "configs/evals/dual_expert_robotwin_smoke_eval.yaml")
    assert request.experiment_config_path == (
        REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    ).resolve()
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
                "experiment_config": str(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"),
                "checkpoint_path": "${paths.tests.eval_checkpoint}",
                "mode": "batch",
                "split": "val",
            },
            handle,
            sort_keys=False,
        )

    request = resolve_evaluation_request(wrapper_path)

    assert request.checkpoint_path == Path("/tmp/eval_checkpoint.ckpt")


@pytest.mark.parametrize(
    ("wrapper_name", "experiment_name"),
    [
        ("parallel_stream_robotwin_smoke_eval.yaml", "parallel_stream_robotwin_smoke.yaml"),
        ("dual_expert_robotwin_smoke_eval.yaml", "dual_expert_robotwin_smoke.yaml"),
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


def test_run_evaluation_on_dual_expert_robotwin() -> None:
    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml",
        max_batches_override=1,
        device_override="cpu",
    )
    summary = run_evaluation(request)
    assert summary.experiment_name == "dual_expert_robotwin_smoke"
    assert summary.num_batches == 1
    assert summary.video_num_inference_steps == 4
    assert summary.action_num_inference_steps == 4
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


def test_rollout_previous_action_prefers_exact_model_space_chunk() -> None:
    decoder_action_pred = torch.zeros(1, 16, 30)
    raw_eval_prediction = torch.ones(1, 16, 7)
    chunk_action_pred = torch.full((1, 16, 30), 2.0)

    previous_action = evaluate_module._select_rollout_previous_action(
        decoder_action_pred=decoder_action_pred,
        policy_aux={
            "raw_chunk_action_pred": raw_eval_prediction,
            "chunk_action_pred": chunk_action_pred,
        },
    )

    assert previous_action is chunk_action_pred
    assert previous_action.shape == (1, 16, 30)


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
    assert summary.video_num_inference_steps == 2
    assert summary.action_num_inference_steps == 2
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None


@pytest.mark.parametrize(
    ("config_path", "expected_name"),
    [
        (REPO_ROOT / "configs/evals/parallel_stream_robotwin_smoke_eval.yaml", "parallel_stream_robotwin_smoke"),
        (REPO_ROOT / "configs/evals/dual_expert_robotwin_smoke_eval.yaml", "dual_expert_robotwin_smoke"),
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
    def __init__(self, *, action_horizon: int = 6) -> None:
        self.action_horizon = action_horizon
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
            actions=torch.zeros(self.action_horizon, 30, dtype=torch.float32),
            action_mask=torch.ones(self.action_horizon, 30, dtype=torch.float32),
            state=torch.zeros(1, 30, dtype=torch.float32),
            state_mask=torch.ones(1, 30, dtype=torch.float32),
            task_text="synthetic trajectory eval",
            metadata={
                "episode_index": window.episode_index,
                "observation_start": window.observation_start,
            },
        )


def test_run_trajectory_evaluation_carries_across_episode_windows(monkeypatch) -> None:
    dataset = _TrajectoryEvalDataset(action_horizon=8)
    monkeypatch.setattr(
        evaluate_module,
        "build_train_val_datasets",
        lambda data_config: (dataset, dataset),
    )
    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml",
        mode_override="trajectory",
        max_trajectories_override=2,
        device_override="cpu",
    )
    summary = run_evaluation(request)
    assert summary.experiment_name == "dual_expert_robotwin_smoke"
    assert summary.mode == "trajectory"
    assert summary.num_trajectories == 2
    assert summary.num_batches == 4
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None
    assert summary.mean_trajectory_action_mse is not None
    assert summary.mean_video_latent_mse is None
    assert summary.mean_trajectory_video_latent_mse is None


def test_run_trajectory_evaluation_resets_dual_expert_observation_conditioned_sessions(
    monkeypatch,
) -> None:
    dataset = _TrajectoryEvalDataset(action_horizon=8)
    monkeypatch.setattr(
        evaluate_module,
        "build_train_val_datasets",
        lambda data_config: (dataset, dataset),
    )
    reset_calls: list[tuple[str | None, ...] | None] = []
    original_rollout_runner = evaluate_module.VariantRolloutRunner

    class _RecordingRolloutRunner(original_rollout_runner):
        def reset(self, **kwargs):
            reset_calls.append(kwargs.get("task_text"))
            return super().reset(**kwargs)

    monkeypatch.setattr(evaluate_module, "VariantRolloutRunner", _RecordingRolloutRunner)

    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml",
        mode_override="trajectory",
        max_trajectories_override=1,
        max_steps_per_trajectory_override=2,
        device_override="cpu",
    )
    summary = run_evaluation(request)

    assert summary.mode == "trajectory"
    assert summary.num_batches == 2
    assert len(reset_calls) == 2


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


def test_trajectory_eval_uses_variant_owned_session_lifecycle() -> None:
    dual_expert_config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    parallel_config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )

    dual_expert = build_variant_pipeline_from_config(dual_expert_config).policy_variant
    parallel = build_variant_pipeline_from_config(parallel_config).policy_variant

    assert evaluate_module._requires_observation_window_session_rebuild(dual_expert)
    assert not evaluate_module._requires_observation_window_session_rebuild(parallel)


def test_run_evaluation_loads_pipeline_prefixed_checkpoint(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    pipeline = build_variant_pipeline_from_config(config)
    checkpoint_path = tmp_path / "dual_expert_robotwin.ckpt"
    prefixed_state_dict = {f"pipeline.{key}": value for key, value in pipeline.state_dict().items()}
    torch.save({"state_dict": prefixed_state_dict}, checkpoint_path)

    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml",
        max_batches_override=1,
        checkpoint_override=str(checkpoint_path),
        device_override="cpu",
    )
    summary = run_evaluation(request)
    assert summary.experiment_name == "dual_expert_robotwin_smoke"
    assert summary.checkpoint_path == str(checkpoint_path)
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None
    assert summary.mean_video_latent_mse is None


def test_apply_checkpoint_runtime_override_uses_checkpoint_local_transformer(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    checkpoint_dir = tmp_path / "checkpoint_step_42"
    transformer_dir = checkpoint_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")
    (transformer_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    model_state = checkpoint_dir / "model_state.pt"
    model_state.write_bytes(b"test")

    resolved_config, resolved = evaluate_module._apply_checkpoint_runtime_override(
        config,
        checkpoint_dir,
    )

    assert resolved == model_state
    assert resolved_config.backbone.runtime_backbone_artifact_path == str(
        transformer_dir.resolve()
    )
    assert str(resolved_config.backbone.reference_core_init_mode) == "full"
    assert config.backbone.runtime_backbone_artifact_path != str(
        transformer_dir.resolve()
    )


@pytest.mark.parametrize("with_config", [False, True])
def test_apply_checkpoint_runtime_override_ignores_incomplete_transformer_export(
    tmp_path: Path,
    with_config: bool,
) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    original_artifact_path = config.backbone.runtime_backbone_artifact_path
    checkpoint_dir = tmp_path / "checkpoint_step_42"
    transformer_dir = checkpoint_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    if with_config:
        (transformer_dir / "config.json").write_text("{}", encoding="utf-8")
    model_state = checkpoint_dir / "model_state.pt"
    model_state.write_bytes(b"test")

    resolved_config, resolved = evaluate_module._apply_checkpoint_runtime_override(
        config,
        checkpoint_dir,
    )

    assert resolved == model_state
    assert resolved_config is config
    assert (
        resolved_config.backbone.runtime_backbone_artifact_path
        == original_artifact_path
    )


def test_run_evaluation_accepts_checkpoint_step_directory(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    pipeline = build_variant_pipeline_from_config(config)
    checkpoint_dir = tmp_path / "checkpoint_step_1"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "model_state.pt"
    prefixed_state_dict = {f"pipeline.{key}": value for key, value in pipeline.state_dict().items()}
    torch.save({"state_dict": prefixed_state_dict}, checkpoint_path)

    request = resolve_evaluation_request(
        REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml",
        max_batches_override=1,
        checkpoint_override=str(checkpoint_dir),
        device_override="cpu",
    )
    summary = run_evaluation(request)

    assert summary.experiment_name == "dual_expert_robotwin_smoke"
    assert summary.checkpoint_path == str(checkpoint_path)
    assert summary.action_prediction_shape == summary.target_action_shape
    assert summary.mean_action_mse is not None
