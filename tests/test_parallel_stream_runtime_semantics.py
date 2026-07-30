from __future__ import annotations

import pytest

from open_wam.configs import (
    CurrentBlockCoupling,
    JointTimestepCoupling,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelRuntimeMode,
    ParallelSequenceContract,
    ParallelStreamPolicyConfig,
)
from open_wam.evals.dynamics import rollout as dynamics_rollout
from open_wam.models.common import (
    chunked_temporal_exact_profile_name_for_coupling,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream import variant
from open_wam.models.policy_variants.parallel_stream.runtime_semantics import (
    attention_profile_name_for_current_block_coupling,
    prefix_visibility_mode_for_policy,
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
    uses_legacy_prefix_per_chunk_proprio_contract,
)


def _policy_config(**overrides: object) -> ParallelStreamPolicyConfig:
    return ParallelStreamPolicyConfig(hidden_size=32, **overrides)


def test_runtime_semantic_consumers_share_the_canonical_resolver() -> None:
    assert (
        reference_runtime.resolve_parallel_current_block_coupling
        is resolve_parallel_current_block_coupling
    )
    assert (
        variant.resolve_parallel_current_block_coupling
        is resolve_parallel_current_block_coupling
    )
    assert (
        dynamics_rollout.resolve_parallel_current_block_coupling
        is resolve_parallel_current_block_coupling
    )


def test_reference_runtime_semantic_names_alias_canonical_contract() -> None:
    assert (
        reference_runtime._attention_profile_name_for_current_block_coupling
        is attention_profile_name_for_current_block_coupling
    )
    assert (
        reference_runtime._prefix_visibility_mode_for_policy
        is prefix_visibility_mode_for_policy
    )
    assert (
        reference_runtime.resolve_parallel_context_condition_latent_source
        is resolve_parallel_context_condition_latent_source
    )
    assert (
        reference_runtime.resolve_parallel_history_stream_visibility
        is resolve_parallel_history_stream_visibility
    )
    assert (
        reference_runtime.resolve_parallel_joint_timestep_coupling
        is resolve_parallel_joint_timestep_coupling
    )
    assert (
        reference_runtime._uses_legacy_prefix_per_chunk_proprio_contract
        is uses_legacy_prefix_per_chunk_proprio_contract
    )


@pytest.mark.parametrize("coupling", list(CurrentBlockCoupling))
def test_explicit_current_block_coupling_is_preserved(
    coupling: CurrentBlockCoupling,
) -> None:
    assert (
        resolve_parallel_current_block_coupling(
            _policy_config(current_block_coupling=coupling)
        )
        == coupling
    )


@pytest.mark.parametrize(
    ("runtime_mode", "expected"),
    [
        (
            ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            CurrentBlockCoupling.JOINT,
        ),
        (ParallelRuntimeMode.LINGBOT_EXACT, CurrentBlockCoupling.VIDEO_THEN_ACTION),
        (
            ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
            CurrentBlockCoupling.VIDEO_THEN_ACTION,
        ),
        (
            ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
            CurrentBlockCoupling.VIDEO_THEN_ACTION,
        ),
    ],
)
def test_current_block_coupling_resolves_runtime_compatibility_default(
    runtime_mode: ParallelRuntimeMode,
    expected: CurrentBlockCoupling,
) -> None:
    assert (
        resolve_parallel_current_block_coupling(
            _policy_config(runtime_mode=runtime_mode)
        )
        == expected
    )


@pytest.mark.parametrize(
    "coupling",
    [
        CurrentBlockCoupling.VIDEO_THEN_ACTION,
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    ],
)
def test_non_joint_programs_force_independent_timestep_clocks(
    coupling: CurrentBlockCoupling,
) -> None:
    config = _policy_config(
        current_block_coupling=coupling,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )

    assert (
        resolve_parallel_joint_timestep_coupling(config)
        == JointTimestepCoupling.INDEPENDENT
    )


@pytest.mark.parametrize(
    "coupling",
    [
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    ],
)
@pytest.mark.parametrize("timestep_coupling", list(JointTimestepCoupling))
def test_joint_like_programs_preserve_configured_timestep_coupling(
    coupling: CurrentBlockCoupling,
    timestep_coupling: JointTimestepCoupling,
) -> None:
    config = _policy_config(
        current_block_coupling=coupling,
        joint_timestep_coupling=timestep_coupling,
    )

    assert resolve_parallel_joint_timestep_coupling(config) == timestep_coupling


@pytest.mark.parametrize(
    "visibility",
    list(ParallelHistoryStreamVisibility),
)
def test_explicit_history_stream_visibility_is_preserved(
    visibility: ParallelHistoryStreamVisibility,
) -> None:
    assert (
        resolve_parallel_history_stream_visibility(
            _policy_config(history_stream_visibility=visibility)
        )
        == visibility
    )


def test_legacy_preserve_video_history_flag_maps_full_visibility() -> None:
    config = _policy_config(
        history_stream_visibility=ParallelHistoryStreamVisibility.FULL,
        preserve_video_pretrain_history=True,
    )

    assert (
        resolve_parallel_history_stream_visibility(config)
        == ParallelHistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
    )
    assert (
        prefix_visibility_mode_for_policy(config)
        == "preserve_video_pretrain_history"
    )


@pytest.mark.parametrize(
    ("visibility", "expected"),
    [
        (ParallelHistoryStreamVisibility.FULL, "full_history"),
        (ParallelHistoryStreamVisibility.VIDEO_ONLY, "video_history_only"),
        (
            ParallelHistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY,
            "preserve_video_pretrain_history",
        ),
    ],
)
def test_history_visibility_maps_to_exact_cache_contract(
    visibility: ParallelHistoryStreamVisibility,
    expected: str,
) -> None:
    assert (
        prefix_visibility_mode_for_policy(
            _policy_config(history_stream_visibility=visibility)
        )
        == expected
    )


@pytest.mark.parametrize(
    "source",
    list(ParallelContextConditionLatentSource),
)
def test_context_condition_latent_source_is_preserved(
    source: ParallelContextConditionLatentSource,
) -> None:
    assert (
        resolve_parallel_context_condition_latent_source(
            _policy_config(context_condition_latent_source=source)
        )
        == source
    )


def test_legacy_prefix_per_chunk_proprio_contract_is_explicit() -> None:
    assert not uses_legacy_prefix_per_chunk_proprio_contract(_policy_config())
    assert uses_legacy_prefix_per_chunk_proprio_contract(
        _policy_config(
            parallel_sequence_contract=(
                ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
            )
        )
    )


@pytest.mark.parametrize("coupling", list(CurrentBlockCoupling))
def test_attention_profile_selection_delegates_shared_profile_contract(
    coupling: CurrentBlockCoupling,
) -> None:
    assert attention_profile_name_for_current_block_coupling(
        coupling
    ) == chunked_temporal_exact_profile_name_for_coupling(coupling.value)
