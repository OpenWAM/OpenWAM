from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from open_wam.training import TrainingRuntime
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


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
