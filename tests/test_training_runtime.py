from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import yaml

from open_wam.configs import TrainingConfig
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.training import TrainingRuntime
from open_wam.training.loop_policies import StepLoopPolicy
from open_wam.training.state import TrainState
from open_wam.training.step_executor import resolve_sample_loss_weight
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_temp_config(tmp_path: Path, *, source_name: str, output_name: str, mutate) -> Path:
    source_path = REPO_ROOT / "configs/experiments" / source_name
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = output_name
    mutate(raw)
    config_path = tmp_path / f"{output_name}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return config_path


def _build_step_runtime_config(
    config_path: Path,
    *,
    tmp_path: Path,
    batch_adapter: str = "views",
):
    config = load_experiment_config(config_path)
    return replace(
        config,
        training=replace(
            config.training,
            num_steps=1,
        ),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter=batch_adapter,
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            limit_train_batches=1,
            limit_val_batches=1,
            enable_checkpointing=False,
        ),
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "parallel_stream_robotwin_smoke.yaml",
        "register_attached_robotwin_smoke.yaml",
        "video_sequence_policy_robotwin_smoke.yaml",
        "mot_robotwin_smoke.yaml",
    ],
)
def test_composable_runtime_trains_shared_core_method_smokes(tmp_path: Path, config_name: str) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_step_loop_reshuffles_distributed_sampler_each_loader_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = TensorDataset(torch.arange(1))
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    seen_epochs: list[int] = []
    original_set_epoch = sampler.set_epoch

    def record_set_epoch(epoch: int) -> None:
        seen_epochs.append(epoch)
        original_set_epoch(epoch)

    monkeypatch.setattr(sampler, "set_epoch", record_set_epoch)
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.train_loader = loader
    runtime.train_state = TrainState(run_name="step-loop-sampler-test")
    runtime._run_validation = lambda *, limit_batches: None
    runtime._save_checkpoint = lambda *, final: None

    def train_one_batch(batch) -> None:
        del batch
        runtime.train_state.global_step += 1
        runtime.train_state.seen_batches += 1
        runtime.train_state.optimizer_step += 1

    runtime._train_micro_step = train_one_batch

    TrainingRuntime._run_step_loop(runtime, StepLoopPolicy(max_steps=3))

    assert seen_epochs == [0, 1, 2]
    assert runtime.train_state.epoch_index == 3


def test_sample_loss_weight_can_scale_by_valid_action_steps() -> None:
    actions = torch.zeros(1, 6, 7)
    action_mask = torch.zeros_like(actions)
    action_mask[0, :6] = 1.0
    batch = PolicyTrainBatch(
        actions=actions,
        action_mask=action_mask,
        extra={"metadata": ({"dataset_mean_valid_action_steps": 4.0},)},
    )

    weight = resolve_sample_loss_weight(
        training_config=TrainingConfig(sample_loss_weight_mode="valid_action_steps"),
        batch=batch,
    )
    sqrt_weight = resolve_sample_loss_weight(
        training_config=TrainingConfig(sample_loss_weight_mode="sqrt_valid_action_steps"),
        batch=batch,
    )

    assert weight.item() == pytest.approx(1.5)
    assert sqrt_weight.item() == pytest.approx(1.5**0.5)


def test_sample_loss_weight_rejects_reduced_multi_sample_batches() -> None:
    actions = torch.zeros(2, 6, 7)
    action_mask = torch.ones_like(actions)
    batch = PolicyTrainBatch(
        actions=actions,
        action_mask=action_mask,
        extra={
            "metadata": (
                {"dataset_mean_valid_action_steps": 6.0},
                {"dataset_mean_valid_action_steps": 6.0},
            )
        },
    )

    with pytest.raises(ValueError, match="train_batch_size=1"):
        resolve_sample_loss_weight(
            training_config=TrainingConfig(sample_loss_weight_mode="valid_action_steps"),
            batch=batch,
        )


def test_composable_runtime_trains_causal_video_prediction_smoke(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "configs/experiments/causal_video_prediction_robotwin_smoke.yaml"
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path, batch_adapter="latents")

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_training_runtime_initializes_mot_variant_before_strategy_wrap(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml"
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    pipeline = runtime.strategy.unwrap_model(runtime.model)

    assert pipeline.policy_variant._action_expert_initialized is True


def test_composable_runtime_logs_checkpoints_and_resume(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml")
    config = replace(
        config,
        training=replace(
            config.training,
            num_steps=1,
        ),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            enable_checkpointing=True,
            save_interval=1,
            enable_jsonl_logging=True,
            metrics_filename="metrics.jsonl",
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    output_dir = tmp_path / config.name
    checkpoint_root = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"

    assert final_state.optimizer_step == 1
    assert metrics_path.exists()
    assert checkpoint_root.exists()

    resumed_config = replace(
        config,
        training=replace(config.training, num_steps=2),
        trainer=replace(config.trainer, resume_from=str(checkpoint_root)),
    )
    resumed_runtime = TrainingRuntime.from_config(resumed_config)
    resumed_state = resumed_runtime.run()

    assert resumed_state.optimizer_step == 2


def test_composable_runtime_exports_runtime_backbone(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml")
    config = replace(
        config,
        training=replace(config.training, num_steps=1),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            enable_checkpointing=True,
            save_interval=1,
            checkpoint_mode="model_only",
            export_runtime_backbone=True,
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    runtime.run()

    checkpoint_dir = tmp_path / config.name / "checkpoints" / "checkpoint_step_1" / "transformer"
    assert (checkpoint_dir / "diffusion_pytorch_model.safetensors").exists()
    assert (checkpoint_dir / "config.json").exists()


def test_composable_runtime_disable_checkpointing_suppresses_export_runtime_backbone(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml")
    config = replace(
        config,
        training=replace(config.training, num_steps=1),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="single_device",
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            save_interval=None,
            checkpoint_mode="model_only",
            export_runtime_backbone=True,
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    runtime.run()

    checkpoint_root = tmp_path / config.name / "checkpoints"
    assert not list(checkpoint_root.glob("checkpoint_step_*"))


def test_composable_runtime_ddp_strategy_degrades_cleanly_to_single_process(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml")
    config = replace(
        config,
        training=replace(config.training, num_steps=1),
        trainer=replace(
            config.trainer,
            runtime="composable",
            batch_adapter="latents",
            loop_policy="steps",
            strategy="ddp",
            default_root_dir=str(tmp_path),
        ),
    )

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_composable_runtime_trains_post_latent_with_core_layer_visual_readout(tmp_path: Path) -> None:
    def _mutate(raw) -> None:
        raw["policy_variant"]["visual_readout"] = {
            "source_family": "core_layer_tokens",
            "layer_index": 0,
        }
        raw["backbone"]["num_layers"] = 2

    config_path = _write_temp_config(
        tmp_path,
        source_name="post_latent_robotwin.yaml",
        output_name="post_latent_core_layer_runtime",
        mutate=_mutate,
    )
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_composable_runtime_trains_post_decoded_with_multi_layer_visual_readout(tmp_path: Path) -> None:
    def _mutate(raw) -> None:
        raw["policy_variant"]["visual_readout"] = {
            "source_family": "core_multi_layer_tokens",
            "layer_indices": [0, 1],
            "fusion_mode": "concat_project",
        }
        raw["backbone"]["num_layers"] = 2

    config_path = _write_temp_config(
        tmp_path,
        source_name="post_decoded_robotwin.yaml",
        output_name="post_decoded_multi_layer_runtime",
        mutate=_mutate,
    )
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_composable_runtime_trains_method4_with_implicit_video_conditioned_decoder(tmp_path: Path) -> None:
    post_latent_path = REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml"
    post_latent_config = _build_step_runtime_config(post_latent_path, tmp_path=tmp_path)
    post_latent_runtime = TrainingRuntime.from_config(post_latent_config)
    post_latent_state = post_latent_runtime.run()
    assert post_latent_state.optimizer_step == 1

    post_decoded_path = REPO_ROOT / "configs/experiments/post_decoded_robotwin_video_conditioned.yaml"
    post_decoded_config = _build_step_runtime_config(post_decoded_path, tmp_path=tmp_path)
    post_decoded_runtime = TrainingRuntime.from_config(post_decoded_config)
    post_decoded_state = post_decoded_runtime.run()
    assert post_decoded_state.optimizer_step == 1


def test_composable_runtime_trains_method4_current_frame_regression_mode(tmp_path: Path) -> None:
    config_path = REPO_ROOT / "configs/experiments/post_decoded_robotwin_current_frame_regression.yaml"
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1


def test_composable_runtime_trains_video_sequence_policy_with_core_layer_visual_readout(tmp_path: Path) -> None:
    def _mutate(raw) -> None:
        raw["policy_variant"]["name"] = "video_sequence_policy"
        raw["policy_variant"]["attach_site"] = "post_visual_core"
        raw["policy_variant"]["visual_readout"] = {
            "source_family": "core_layer_tokens",
            "layer_index": 0,
        }
        raw["policy_variant"].pop("pooling_mode", None)
        raw["policy_variant"].pop("query_count", None)
        raw["policy_variant"].pop("use_state_projection", None)
        raw["action_decoder"]["name"] = "vpp_decoder"
        raw["action_decoder"]["num_sampling_steps"] = 2
        raw["action_decoder"]["rollout_chunk_steps"] = 2
        raw["backbone"]["num_layers"] = 2

    config_path = _write_temp_config(
        tmp_path,
        source_name="post_latent_robotwin.yaml",
        output_name="video_sequence_policy_core_layer_runtime",
        mutate=_mutate,
    )
    config = _build_step_runtime_config(config_path, tmp_path=tmp_path)

    runtime = TrainingRuntime.from_config(config)
    final_state = runtime.run()

    assert final_state.optimizer_step == 1
