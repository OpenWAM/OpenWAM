from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from open_wam.configs import (
    BatchAdapterName,
    ContextConditionLatentSource,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    LoopPolicyName,
    ProprioContextMode,
    RolloutContextPolicy,
    SampleTargetAlignment,
    StrategyName,
    TrainerConfig,
    TrainerRuntimeName,
    WandBMode,
)
from open_wam.training import (
    TrainCliOverrides,
    load_training_cli_config,
    resolve_experiment_config_path,
)
from open_wam.training.cli import parse_train_cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_cli_accepts_ordered_extensions() -> None:
    overrides = parse_train_cli(
        [
            "--config-name",
            "parallel_stream_robotwin_smoke",
            "--extension",
            "acme_open_wam",
            "--extension",
            "research.runtime:install",
        ]
    )

    assert overrides.extensions == (
        "acme_open_wam",
        "research.runtime:install",
    )


def test_resolve_experiment_config_path_from_config_name() -> None:
    overrides = TrainCliOverrides(config_name="parallel_stream_robotwin_smoke")
    resolved = resolve_experiment_config_path(overrides)
    assert resolved == REPO_ROOT / "configs" / "experiments" / "parallel_stream_robotwin_smoke.yaml"


def test_train_entrypoint_help_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "open_wam.training.train", "--help"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "--config-name" in result.stdout


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("runtime", "lightning", "trainer.runtime=lightning"),
        ("strategy", "lightning", "trainer.strategy=lightning"),
    ],
)
def test_legacy_lightning_training_choices_fail_with_migration(
    field: str,
    value: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        TrainerConfig(**{field: value})


def test_cli_overrides_map_save_root_and_env_defaults(tmp_path: Path) -> None:
    overrides = TrainCliOverrides(
        config_name="parallel_stream_robotwin_smoke",
        save_root=str(tmp_path / "reference_style_run"),
        checkpoint_root="/checkpoints/checkpoint_step_400",
        dataset_root="/datasets/local_libero",
        latent_root="/datasets/local_libero/latents",
        devices=6,
        num_steps=4,
        enable_wandb=True,
        overrides=(
            "trainer.runtime=composable",
            "trainer.batch_adapter=latents",
            "trainer.loop_policy=steps",
            "trainer.strategy=single_device",
            "trainer.save_interval=50",
            "trainer.validation_interval=25",
            "training.learning_rate=2e-05",
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
    assert config.trainer.run_name == "reference_style_run"
    assert config.trainer.checkpoint_dir == str(tmp_path / "reference_style_run" / "checkpoints")
    assert config.data.local_root == "/datasets/local_libero"
    assert config.data.latent_root == "/datasets/local_libero/latents"
    assert config.trainer.resume_from == "/checkpoints/checkpoint_step_400/full_training_state.pt"
    assert config.backbone.transformer_subdir == "/checkpoints/checkpoint_step_400/transformer"
    assert config.trainer.devices == 6
    assert config.trainer.runtime == TrainerRuntimeName.COMPOSABLE
    assert config.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert config.trainer.loop_policy == LoopPolicyName.STEPS
    assert config.trainer.strategy == StrategyName.SINGLE_DEVICE
    assert config.training.num_steps == 4
    assert config.training.learning_rate == 2.0e-5
    assert config.trainer.save_interval == 50
    assert config.trainer.validation_interval == 25
    assert config.trainer.enable_wandb is True
    assert config.trainer.wandb_project == "lingbot-va-posttrain-libero"
    assert config.trainer.wandb_entity == "codefishy-stanford-university"
    assert config.trainer.wandb_mode == WandBMode.OFFLINE


def test_cli_overrides_support_nested_adapter_options() -> None:
    config = load_training_cli_config(
        TrainCliOverrides(
            config_name="parallel_stream_robotwin_smoke",
            overrides=(
                "data.adapter_options.rgb-key=observation.images.front",
                "data.adapter_options.decode.timestamp_tolerance_us=100",
            ),
        ),
        env={},
    )

    assert config.data.adapter_options == {
        "rgb-key": "observation.images.front",
        "decode": {"timestamp_tolerance_us": 100},
    }


def test_checkpoint_root_prefers_full_training_state(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint_step_400"
    checkpoint_root.mkdir()
    (checkpoint_root / "model_state.pt").write_bytes(b"model")
    (checkpoint_root / "full_training_state.pt").write_bytes(b"full")

    config = load_training_cli_config(
        TrainCliOverrides(
            config_name="parallel_stream_robotwin_smoke",
            checkpoint_root=str(checkpoint_root),
        ),
        env={},
    )

    assert config.trainer.resume_from == str(checkpoint_root / "full_training_state.pt")


def test_checkpoint_root_falls_back_to_model_state_for_legacy_checkpoint(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint_step_400"
    checkpoint_root.mkdir()
    (checkpoint_root / "model_state.pt").write_bytes(b"model")

    config = load_training_cli_config(
        TrainCliOverrides(
            config_name="parallel_stream_robotwin_smoke",
            checkpoint_root=str(checkpoint_root),
        ),
        env={},
    )

    assert config.trainer.resume_from == str(checkpoint_root / "model_state.pt")


def test_cli_overrides_apply_dependent_sample_construction_fields_together() -> None:
    overrides = TrainCliOverrides(
        config_name="parallel_stream_libero_decoupled_same_step",
        overrides=(
            "data.sample_construction.mode=uniform_segment",
            "data.sample_construction.sample_order_mode=replacement",
            "data.sample_construction.randomize_segment_length=true",
            "data.sample_construction.segment_min_frames=64",
            "data.sample_construction.segment_max_frames=256",
            "data.sample_construction.randomize_geometry=true",
            "data.sample_construction.allow_next_after_context_random_geometry=true",
        ),
    )

    config = load_training_cli_config(overrides, env={})

    sample = config.data.sample_construction
    assert getattr(sample.mode, "value", str(sample.mode)) == "uniform_segment"
    assert getattr(sample.sample_order_mode, "value", str(sample.sample_order_mode)) == "replacement"
    assert sample.randomize_segment_length is True
    assert sample.segment_min_frames == 64
    assert sample.segment_max_frames == 256
    assert sample.randomize_geometry is True
    assert sample.allow_next_after_context_random_geometry is True


def test_cli_contract_override_expands_after_set_overrides() -> None:
    overrides = TrainCliOverrides(
        config_name="parallel_stream_libero_decoupled_same_step",
        overrides=(
            "policy_variant.sequence_contract=rollout_parity_single_frame_perchunk_proprio",
        ),
    )

    config = load_training_cli_config(overrides, env={})

    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
    assert config.data.sample_construction.rollout_context_policy == RolloutContextPolicy.ONE_FRAME
    assert config.data.sample_construction.condition_source_frame_offset == -1


def test_cli_legacy_prefix_contract_override_restores_target_only_sampling() -> None:
    overrides = TrainCliOverrides(
        config_name="parallel_stream_libero_video_then_action",
        overrides=(
            "policy_variant.sequence_contract=legacy_prefix_single_frame_perchunk_proprio",
        ),
    )

    config = load_training_cli_config(overrides, env={})

    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.noisy_video_condition_prob == 0.5
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.LEGACY
    assert config.data.sample_construction.condition_source_frame_offset == -1




def test_cli_legacy_prefix_contract_allows_scheduler_override() -> None:
    overrides = TrainCliOverrides(
        config_name="parallel_stream_libero_video_then_action",
        overrides=(
            "policy_variant.sequence_contract=legacy_prefix_single_frame_perchunk_proprio",
            "policy_variant.joint_timestep_coupling=shared_video_schedule",
        ),
    )

    config = load_training_cli_config(overrides, env={})

    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE


def test_cli_legacy_prefix_contract_allows_noisy_condition_prob_override() -> None:
    overrides = TrainCliOverrides(
        config_name="parallel_stream_libero_video_then_action",
        overrides=(
            "policy_variant.sequence_contract=legacy_prefix_single_frame_perchunk_proprio",
            "policy_variant.noisy_video_condition_prob=0.0",
        ),
    )

    config = load_training_cli_config(overrides, env={})

    assert config.policy_variant.noisy_video_condition_prob == 0.0


def test_cli_legacy_prefix_contract_defaults_joint_coupling_to_independent() -> None:
    overrides = TrainCliOverrides(
        config_name="dual_expert_libero_joint",
        overrides=(
            "policy_variant.sequence_contract=legacy_prefix_single_frame_perchunk_proprio",
        ),
    )

    config = load_training_cli_config(overrides, env={})

    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT


def test_cli_legacy_prefix_contract_allows_joint_coupling_override() -> None:
    overrides = TrainCliOverrides(
        config_name="dual_expert_libero_joint",
        overrides=(
            "policy_variant.sequence_contract=legacy_prefix_single_frame_perchunk_proprio",
            "policy_variant.joint_timestep_coupling=shared_video_schedule",
        ),
    )

    config = load_training_cli_config(overrides, env={})

    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE


def test_cli_legacy_prefix_contract_allows_match_sigma_joint_coupling_override() -> None:
    overrides = TrainCliOverrides(
        config_name="dual_expert_libero_joint",
        overrides=(
            "policy_variant.sequence_contract=legacy_prefix_single_frame_perchunk_proprio",
            "policy_variant.joint_timestep_coupling=match_sigma",
        ),
    )

    config = load_training_cli_config(overrides, env={})

    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA


def test_cli_contract_override_rejects_managed_field_override() -> None:
    overrides = TrainCliOverrides(
        config_name="parallel_stream_libero_decoupled_same_step",
        overrides=(
            "policy_variant.sequence_contract=rollout_parity_single_frame_perchunk_proprio",
            "policy_variant.proprio_context_mode=none",
        ),
    )

    with pytest.raises(ValueError, match="sequence_contract=.*proprio_context_mode"):
        load_training_cli_config(overrides, env={})


def test_cli_overrides_coerce_quoted_bool_strings() -> None:
    config = load_training_cli_config(
        TrainCliOverrides(
            config_name="parallel_stream_libero_generalist_joint_denoising",
            overrides=(
                'policy_variant.generalist_mode_text_token="false"',
                'trainer.enable_wandb="false"',
            ),
        ),
        env={},
    )

    assert config.policy_variant.generalist_mode_text_token is False
    assert config.trainer.enable_wandb is False
