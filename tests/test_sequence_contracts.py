from __future__ import annotations

from open_wam.configs import (
    apply_parallel_sequence_contract,
    expand_parallel_sequence_contract,
    validate_experiment_config_runtime_contract,
    validate_parallel_sequence_contract_override_keys,
    validate_policy_data_sequence_contract,
)
from open_wam.configs.sequence_contracts import (
    apply_parallel_sequence_contract as OwnedApplyParallelSequenceContract,
    expand_parallel_sequence_contract as OwnedExpandParallelSequenceContract,
    validate_experiment_config_runtime_contract as OwnedValidateExperimentConfigRuntimeContract,
    validate_parallel_sequence_contract_override_keys as OwnedValidateParallelSequenceContractOverrideKeys,
    validate_policy_data_sequence_contract as OwnedValidatePolicyDataSequenceContract,
)
from open_wam.utils.config_loader import (
    apply_parallel_sequence_contract as LegacyApplyParallelSequenceContract,
    validate_experiment_config_runtime_contract as LegacyValidateExperimentConfigRuntimeContract,
    validate_parallel_sequence_contract_override_keys as LegacyValidateParallelSequenceContractOverrideKeys,
    validate_policy_data_sequence_contract as LegacyValidatePolicyDataSequenceContract,
)


def test_sequence_contract_public_and_compatibility_exports_preserve_identity() -> None:
    assert apply_parallel_sequence_contract is OwnedApplyParallelSequenceContract
    assert expand_parallel_sequence_contract is OwnedExpandParallelSequenceContract
    assert (
        validate_experiment_config_runtime_contract
        is OwnedValidateExperimentConfigRuntimeContract
    )
    assert (
        validate_parallel_sequence_contract_override_keys
        is OwnedValidateParallelSequenceContractOverrideKeys
    )
    assert LegacyApplyParallelSequenceContract is OwnedApplyParallelSequenceContract
    assert (
        LegacyValidateExperimentConfigRuntimeContract
        is OwnedValidateExperimentConfigRuntimeContract
    )
    assert (
        LegacyValidateParallelSequenceContractOverrideKeys
        is OwnedValidateParallelSequenceContractOverrideKeys
    )
    assert (
        validate_policy_data_sequence_contract
        is OwnedValidatePolicyDataSequenceContract
    )
    assert (
        LegacyValidatePolicyDataSequenceContract
        is OwnedValidatePolicyDataSequenceContract
    )
