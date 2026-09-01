from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from open_wam.configs import (
    ContextConditionLatentSource,
    ExperimentConfig,
    HistoryStreamVisibility,
    ProprioContextMode,
    RolloutContextPolicy,
    SampleTargetAlignment,
    VideoActionSequenceContract,
    apply_video_action_sequence_contract,
    expand_video_action_sequence_contract,
    load_experiment_config,
    materialize_video_action_sequence_contract,
    resolve_experiment_config,
    serialize_experiment_config,
    validate_experiment_config_runtime_contract,
    validate_policy_data_sequence_contract,
    validate_video_action_sequence_contract_override_keys,
)
from open_wam.configs.sequence_contract_specs import (
    VIDEO_ACTION_SEQUENCE_CONTRACT_SPECS,
    get_video_action_sequence_contract_spec,
    video_action_sequence_contract_managed_override_keys,
)
from open_wam.configs.sequence_contracts import (
    apply_video_action_sequence_contract as OwnedApplyParallelSequenceContract,
)
from open_wam.configs.sequence_contracts import (
    expand_video_action_sequence_contract as OwnedExpandParallelSequenceContract,
)
from open_wam.configs.sequence_contracts import (
    materialize_video_action_sequence_contract as OwnedMaterializeVideoActionSequenceContract,
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
        materialize_video_action_sequence_contract
        is OwnedMaterializeVideoActionSequenceContract
    )
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


def test_sequence_contract_spec_registry_is_exhaustive_and_frozen() -> None:
    expected_contracts = set(VideoActionSequenceContract) - {
        VideoActionSequenceContract.DEFAULT
    }
    assert set(VIDEO_ACTION_SEQUENCE_CONTRACT_SPECS) == expected_contracts

    spec = get_video_action_sequence_contract_spec(
        VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert spec is not None
    assert spec.requires_frame_aligned_proprio_context is True
    with pytest.raises(TypeError):
        VIDEO_ACTION_SEQUENCE_CONTRACT_SPECS[
            VideoActionSequenceContract.DEFAULT
        ] = spec  # type: ignore[index]


@pytest.mark.parametrize(
    ("contract", "expected_sample_updates"),
    (
        (
            VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
            {
                "condition_source_frame_offset": -1,
                "start_padding_frames": 0,
                "target_alignment": SampleTargetAlignment.NEXT_AFTER_CONTEXT,
                "rollout_context_policy": RolloutContextPolicy.ONE_FRAME,
            },
        ),
        (
            VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
            {
                "condition_source_frame_offset": -1,
                "start_padding_frames": 0,
                "target_alignment": SampleTargetAlignment.LEGACY,
            },
        ),
    ),
)
def test_raw_and_typed_sequence_contract_materialization_have_strict_parity(
    contract: VideoActionSequenceContract,
    expected_sample_updates: dict[str, object],
) -> None:
    spec = get_video_action_sequence_contract_spec(contract)
    assert spec is not None
    expected_policy_updates = {
        "proprio_context_mode": ProprioContextMode.PER_CHUNK_ADDITIVE,
        "context_condition_latent_source": (
            ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
        ),
        "history_stream_visibility": HistoryStreamVisibility.VIDEO_ONLY,
        "use_condition_latents": True,
        "require_condition_latents": True,
    }
    assert spec.policy_variant_updates() == expected_policy_updates
    assert spec.sample_construction_updates() == expected_sample_updates

    raw = expand_video_action_sequence_contract(
        {
            "policy_variant": {"sequence_contract": contract.value},
            "data": {"sample_construction": {}},
        }
    )
    base = ExperimentConfig()
    assert base.policy_variant.requires_frame_aligned_proprio_context is False
    conflicting_target_alignment = (
        SampleTargetAlignment.LEGACY
        if contract
        is VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
        else SampleTargetAlignment.NEXT_AFTER_CONTEXT
    )
    typed = materialize_video_action_sequence_contract(
        replace(
            base,
            policy_variant=replace(
                base.policy_variant,
                sequence_contract=contract,
                proprio_context_mode=ProprioContextMode.NONE,
                context_condition_latent_source=(
                    ContextConditionLatentSource.VIDEO_LATENTS
                ),
                history_stream_visibility=HistoryStreamVisibility.FULL,
                use_condition_latents=False,
                require_condition_latents=False,
            ),
            data=replace(
                base.data,
                sample_construction=replace(
                    base.data.sample_construction,
                    condition_source_frame_offset=7,
                    start_padding_frames=3,
                    target_alignment=conflicting_target_alignment,
                    rollout_context_policy=RolloutContextPolicy.ROLLOUT_HISTORY,
                ),
            ),
        )
    )

    for key, expected in expected_policy_updates.items():
        assert raw["policy_variant"][key] == expected
        assert getattr(typed.policy_variant, key) == expected
    assert typed.policy_variant.requires_frame_aligned_proprio_context is True
    for key, expected in expected_sample_updates.items():
        assert raw["data"]["sample_construction"][key] == expected
        assert getattr(typed.data.sample_construction, key) == expected

    if (
        contract
        is VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    ):
        assert "rollout_context_policy" not in raw["data"]["sample_construction"]
        assert (
            typed.data.sample_construction.rollout_context_policy
            is RolloutContextPolicy.ROLLOUT_HISTORY
        )


def test_raw_sequence_contract_still_rejects_authored_conflicts() -> None:
    with pytest.raises(
        ValueError,
        match="sequence_contract=.*history_stream_visibility=video_only",
    ):
        expand_video_action_sequence_contract(
            {
                "policy_variant": {
                    "sequence_contract": (
                        VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO.value
                    ),
                    "history_stream_visibility": HistoryStreamVisibility.FULL.value,
                }
            }
        )


def test_contracts_preserve_existing_cli_override_ownership() -> None:
    expected_managed_keys = frozenset(
        {
            "policy_variant.proprio_context_mode",
            "policy_variant.context_condition_latent_source",
            "policy_variant.history_stream_visibility",
            "policy_variant.use_condition_latents",
            "policy_variant.require_condition_latents",
            "data.sample_construction.target_alignment",
            "data.sample_construction.rollout_context_policy",
            "data.sample_construction.condition_source_frame_offset",
            "data.sample_construction.start_padding_frames",
        }
    )
    for contract in (
        VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    ):
        assert (
            video_action_sequence_contract_managed_override_keys(contract)
            == expected_managed_keys
        )
    assert not video_action_sequence_contract_managed_override_keys(
        VideoActionSequenceContract.DEFAULT
    )
    with pytest.raises(ValueError, match="owns.*rollout_context_policy"):
        validate_video_action_sequence_contract_override_keys(
            {
                "data.sample_construction.rollout_context_policy": (
                    RolloutContextPolicy.ROLLOUT_HISTORY.value
                )
            },
            contract_value=(
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
            ),
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
