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
from open_wam.configs.experiment import (
    EXPERIMENT_CONFIG_SCHEMA_VERSION,
    ExperimentConfig,
)
from open_wam.configs.loader import load_experiment_config
from open_wam.configs.policy_contracts import (
    ExtensionPolicyConfig,
    PolicyConditioningRequirements,
    PolicyVariantConfig,
)
from open_wam.configs.resolution import resolve_experiment_config
from open_wam.configs.serialization import serialize_experiment_config
from open_wam.configs.trainer import TrainerConfig
from open_wam.configs.training import TrainingConfig

__all__ = [
    "EXPERIMENT_CONFIG_SCHEMA_VERSION",
    "ActionDecoderConfig",
    "DataConfig",
    "ExperimentConfig",
    "ExtensionActionDecoderConfig",
    "ExtensionPolicyConfig",
    "PolicyConditioningRequirements",
    "PolicyVariantConfig",
    "TrainerConfig",
    "TrainingConfig",
    "coerce_fields",
    "load_experiment_config",
    "resolve_config_reference",
    "resolve_evaluation_config_reference",
    "resolve_experiment_config",
    "resolve_experiment_config_reference",
    "serialize_experiment_config",
]
