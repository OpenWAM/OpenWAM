from __future__ import annotations

from pathlib import Path

from open_wam.configs import BatchAdapterName, LoopPolicyName, StrategyName, TrainerRuntimeName, WandBMode
from open_wam.training import TrainCliOverrides, load_training_cli_config, resolve_experiment_config_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_experiment_config_path_from_config_name() -> None:
    overrides = TrainCliOverrides(config_name="parallel_stream_robotwin_smoke")
    resolved = resolve_experiment_config_path(overrides)
    assert resolved == REPO_ROOT / "configs" / "experiments" / "parallel_stream_robotwin_smoke.yaml"


def test_cli_overrides_map_save_root_and_env_defaults(tmp_path: Path) -> None:
    overrides = TrainCliOverrides(
        config_name="parallel_stream_robotwin_smoke",
        save_root=str(tmp_path / "heng_style_run"),
        dataset_root="/datasets/local_libero",
        latent_root="/datasets/local_libero/latents",
        transformer_subdir="/models/runtime_transformer",
        devices=6,
        enable_wandb=True,
        overrides=(
            "trainer.runtime=composable",
            "trainer.batch_adapter=latents",
            "trainer.loop_policy=steps",
            "trainer.strategy=single_device",
            "training.num_steps=3",
            "trainer.save_interval=50",
        ),
    )

    config = load_training_cli_config(
        overrides,
        env={
            "WANDB_PROJECT": "lingbot-va-posttrain-libero",
            "WANDB_TEAM_NAME": "codefishy-stanford-university",
            "WANDB_MODE": "offline",
        },
    )

    assert config.trainer.default_root_dir == str(tmp_path)
    assert config.trainer.run_name == "heng_style_run"
    assert config.trainer.checkpoint_dir == str(tmp_path / "heng_style_run" / "checkpoints")
    assert config.data.local_root == "/datasets/local_libero"
    assert config.data.latent_root == "/datasets/local_libero/latents"
    assert config.backbone.transformer_subdir == "/models/runtime_transformer"
    assert config.trainer.devices == 6
    assert config.trainer.runtime == TrainerRuntimeName.COMPOSABLE
    assert config.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert config.trainer.loop_policy == LoopPolicyName.STEPS
    assert config.trainer.strategy == StrategyName.SINGLE_DEVICE
    assert config.training.num_steps == 3
    assert config.trainer.save_interval == 50
    assert config.trainer.enable_wandb is True
    assert config.trainer.wandb_project == "lingbot-va-posttrain-libero"
    assert config.trainer.wandb_entity == "codefishy-stanford-university"
    assert config.trainer.wandb_mode == WandBMode.OFFLINE
