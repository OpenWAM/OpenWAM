from __future__ import annotations

from open_wam.configs import (
    apply_video_action_sequence_contract,
    expand_video_action_sequence_contract,
    validate_experiment_config_runtime_contract,
    validate_policy_data_sequence_contract,
    validate_video_action_sequence_contract_override_keys,
)
from open_wam.configs.sequence_contracts import (
    apply_video_action_sequence_contract as OwnedApplyParallelSequenceContract,
)
from open_wam.configs.sequence_contracts import (
    expand_video_action_sequence_contract as OwnedExpandParallelSequenceContract,
)
from open_wam.configs.sequence_contracts import (
    validate_experiment_config_runtime_contract as OwnedValidateExperimentConfigRuntimeContract,
)
from open_wam.configs.sequence_contracts import (
    validate_policy_data_sequence_contract as OwnedValidatePolicyDataSequenceContract,
)
from open_wam.configs.sequence_contracts import (
    validate_video_action_sequence_contract_override_keys as OwnedValidateParallelSequenceContractOverrideKeys,
)
from open_wam.utils.config_loader import (
    apply_video_action_sequence_contract as LegacyApplyParallelSequenceContract,
)
from open_wam.utils.config_loader import (
    validate_experiment_config_runtime_contract as LegacyValidateExperimentConfigRuntimeContract,
)
from open_wam.utils.config_loader import (
    validate_policy_data_sequence_contract as LegacyValidatePolicyDataSequenceContract,
)
from open_wam.utils.config_loader import (
    validate_video_action_sequence_contract_override_keys as LegacyValidateParallelSequenceContractOverrideKeys,
)


def test_sequence_contract_public_and_compatibility_exports_preserve_identity() -> None:
    assert apply_video_action_sequence_contract is OwnedApplyParallelSequenceContract
    assert expand_video_action_sequence_contract is OwnedExpandParallelSequenceContract
    assert (
        validate_experiment_config_runtime_contract
        is OwnedValidateExperimentConfigRuntimeContract
    )
    assert (
        validate_video_action_sequence_contract_override_keys
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
