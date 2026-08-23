from __future__ import annotations

from dataclasses import dataclass, field

from open_wam.configs.action_decoder import ActionDecoderConfig, DualExpertActionDecoderConfig
from open_wam.configs.backbone import LingbotCompatibleVideoBackboneConfig
from open_wam.configs.data_benchmarks import RobotWinDataConfig
from open_wam.configs.data_contracts import DataConfig
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.enums import VideoActionProgram
from open_wam.configs.policy_contracts import PolicyVariantConfig
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.configs.trainer import TrainerConfig
from open_wam.configs.training import TrainingConfig
from open_wam.configs.validation import ValidationConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config boundary that keeps subsystems separated."""

    name: str = "dual_expert_robotwin"
    data: DataConfig = field(default_factory=RobotWinDataConfig)
    backbone: LingbotCompatibleVideoBackboneConfig = field(
        default_factory=LingbotCompatibleVideoBackboneConfig
    )
    policy_variant: PolicyVariantConfig = field(
        default_factory=lambda: DualExpertPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION
        )
    )
    action_decoder: ActionDecoderConfig = field(
        default_factory=lambda: DualExpertActionDecoderConfig(
            hidden_size=256,
            action_dim=30,
            action_horizon=32,
        )
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
