from __future__ import annotations

from dataclasses import replace

import yaml

from open_wam.configs import (
    ContextConditionLatentSource,
    ExperimentConfig,
    HistoryStreamVisibility,
    ProprioContextMode,
    VideoActionSequenceContract,
    apply_video_action_sequence_contract,
    expand_video_action_sequence_contract,
    load_experiment_config,
    resolve_experiment_config,
    serialize_experiment_config,
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


def test_direct_sequence_contract_resolution_matches_serialized_round_trip(
    tmp_path,
) -> None:
    base = ExperimentConfig()
    direct = replace(
        base,
        policy_variant=replace(
            base.policy_variant,
            sequence_contract=(
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
            ),
        ),
    )

    resolved = resolve_experiment_config(direct)
    assert resolved.policy_variant.proprio_context_mode is (
        ProprioContextMode.PER_CHUNK_ADDITIVE
    )
    assert resolved.policy_variant.history_stream_visibility is (
        HistoryStreamVisibility.VIDEO_ONLY
    )
    assert resolved.policy_variant.context_condition_latent_source is (
        ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert resolved.data.sample_construction.condition_source_frame_offset == -1

    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(serialize_experiment_config(direct), sort_keys=False),
        encoding="utf-8",
    )
    loaded = load_experiment_config(path)

    assert serialize_experiment_config(loaded) == serialize_experiment_config(resolved)
