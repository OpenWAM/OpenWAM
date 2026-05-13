from __future__ import annotations

from dataclasses import is_dataclass

import pytest
import yaml

try:
    import lightning.pytorch  # noqa: F401
except ModuleNotFoundError:
    pytest.importorskip("pytorch_lightning")

from open_wam.configs import (
    ActionSchemaConfig,
    CurrentBlockCoupling,
    ExperimentConfig,
    InferenceConfig,
    MLPActionDecoderConfig,
    MoTPolicyConfig,
    MoTRuntimeMode,
    ParallelStreamPolicyConfig,
    PostLatentPolicyConfig,
    RobotWinDataConfig,
    TrainingConfig,
)
from open_wam.lightning import OpenWAMLightningModule
from open_wam.lightning.module import policy_variant_requires_module_mutating_infer_backend
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig


def _build_lightning_config(*, training: TrainingConfig | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=2, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
        ),
        policy_variant=PostLatentPolicyConfig(hidden_size=32, local_video_window_frames=2),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=2),
        training=training or TrainingConfig(),
        inference=InferenceConfig(),
    )


def test_lightning_module_hyperparameters_are_serializable() -> None:
    config = _build_lightning_config()

    module = OpenWAMLightningModule(config)
    hparam_config = module.hparams["config"]

    assert isinstance(hparam_config, dict)
    assert not is_dataclass(hparam_config)
    assert hparam_config["trainer"]["accelerator"] == "cpu"
    yaml.safe_dump(hparam_config)


def test_lightning_module_rejects_sample_loss_weighting() -> None:
    config = _build_lightning_config(training=TrainingConfig(sample_loss_weight_mode="valid_action_steps"))

    with pytest.raises(ValueError, match="composable runtime"):
        OpenWAMLightningModule(config)


@pytest.mark.parametrize(
    ("current_block_coupling", "expected_skip"),
    [
        (CurrentBlockCoupling.VIDEO_THEN_ACTION, True),
        (CurrentBlockCoupling.DECOUPLED_SAME_STEP, True),
        (CurrentBlockCoupling.ACTION_THEN_VIDEO, False),
    ],
)
def test_lightning_skips_infer_validation_for_m5_module_mutating_backend(
    current_block_coupling: CurrentBlockCoupling,
    expected_skip: bool,
) -> None:
    config = ExperimentConfig(
        data=RobotWinDataConfig(
            num_frames=2,
            action_schema=ActionSchemaConfig(action_dim=4, action_horizon=2, state_dim=4, state_horizon=1),
        ),
        backbone=SharedVideoTransformerConfig(
            implementation="shared_transformer",
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            ffn_dim=64,
            text_dim=16,
            freq_dim=8,
        ),
        policy_variant=MoTPolicyConfig(
            hidden_size=32,
            runtime_mode=MoTRuntimeMode.NON_JOINT_TWO_STREAM,
            current_block_coupling=current_block_coupling,
            num_action_layers=1,
        ),
        action_decoder=MLPActionDecoderConfig(hidden_size=32, action_dim=4, action_horizon=2),
        training=TrainingConfig(chunk_size=1, window_size=4),
        inference=InferenceConfig(frame_chunk_size=1),
    )

    module = OpenWAMLightningModule(config)

    assert module._skip_infer_validation_for_module_mutating_backend() is expected_skip


def test_lightning_module_mutating_backend_skip_is_mot_scoped() -> None:
    parallel_policy = ParallelStreamPolicyConfig(
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
    )

    assert policy_variant_requires_module_mutating_infer_backend(parallel_policy) is False
