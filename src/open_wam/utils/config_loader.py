"""Compatibility export for the package-owned configuration loader.

New code should import :func:`load_experiment_config` from
:mod:`open_wam.configs`.
"""

from open_wam.configs.loader import load_experiment_config
from open_wam.configs.sequence_contracts import (
    apply_video_action_sequence_contract,
    validate_experiment_config_runtime_contract,
    validate_policy_data_sequence_contract,
    validate_video_action_sequence_contract_override_keys,
)

__all__ = [
    "apply_video_action_sequence_contract",
    "load_experiment_config",
    "validate_experiment_config_runtime_contract",
    "validate_policy_data_sequence_contract",
    "validate_video_action_sequence_contract_override_keys",
]
