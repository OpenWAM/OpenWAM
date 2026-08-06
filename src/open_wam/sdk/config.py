"""Stable typed configuration and packaged-resource resolution."""

from open_wam.configs.action_decoder import (
    ActionDecoderConfig,
    ExtensionActionDecoderConfig,
)
from open_wam.configs.config_paths import (
    resolve_config_reference,
    resolve_evaluation_config_reference,
    resolve_experiment_config_reference,
)
from open_wam.configs.data_contracts import DataConfig
from open_wam.configs.enums import coerce_fields
from open_wam.configs.experiment import ExperimentConfig
from open_wam.configs.loader import load_experiment_config
from open_wam.configs.policy_contracts import (
    ExtensionPolicyConfig,
    PolicyVariantConfig,
)
from open_wam.configs.serialization import serialize_experiment_config
from open_wam.configs.trainer import TrainerConfig
from open_wam.configs.training import TrainingConfig

__all__ = [
    "ActionDecoderConfig",
    "DataConfig",
    "ExperimentConfig",
    "ExtensionActionDecoderConfig",
    "ExtensionPolicyConfig",
    "PolicyVariantConfig",
    "TrainerConfig",
    "TrainingConfig",
    "load_experiment_config",
    "coerce_fields",
    "resolve_config_reference",
    "resolve_evaluation_config_reference",
    "resolve_experiment_config_reference",
    "serialize_experiment_config",
]
