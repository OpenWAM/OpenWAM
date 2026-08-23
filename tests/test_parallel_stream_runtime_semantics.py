from __future__ import annotations

import pytest

from open_wam.configs import (
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.configs.policy_parallel_stream import parallel_runtime_mode_for_program
from open_wam.models.common import (
    chunked_temporal_exact_profile_name_for_coupling,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime, variant
from open_wam.models.policy_variants.parallel_stream.runtime_semantics import (
    attention_profile_name_for_current_block_coupling,
    prefix_visibility_mode_for_policy,
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
    uses_legacy_prefix_per_chunk_proprio_contract,
)
from scripts.research_dynamics import rollout as dynamics_rollout


def _policy_config(**overrides: object) -> ParallelStreamPolicyConfig:
    overrides.setdefault("program", VideoActionProgram.VIDEO_THEN_ACTION)
    return ParallelStreamPolicyConfig(hidden_size=32, **overrides)


def test_backend_runtime_semantics_do_not_leak_into_generic_research_tools() -> None:
    assert (
        reference_runtime.resolve_parallel_current_block_coupling
        is resolve_parallel_current_block_coupling
    )
    assert (
        variant.resolve_parallel_current_block_coupling
        is resolve_parallel_current_block_coupling
    )
    assert not hasattr(dynamics_rollout, "resolve_parallel_current_block_coupling")


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
def test_program_derives_current_block_coupling(
    coupling: CurrentBlockCoupling,
) -> None:
    assert (
        resolve_parallel_current_block_coupling(
            _policy_config(program=VideoActionProgram(coupling.value))
        )
        == coupling
    )


@pytest.mark.parametrize(
    ("program", "expected_runtime"),
    (
        (VideoActionProgram.ACTION_THEN_VIDEO, ParallelRuntimeMode.LINGBOT_EXACT),
        (
            VideoActionProgram.JOINT,
            ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        ),
    ),
)
def test_runtime_backend_is_derived_from_program(
    program: VideoActionProgram,
    expected_runtime: ParallelRuntimeMode,
) -> None:
    config = _policy_config(program=program)

    assert config.runtime_mode is expected_runtime
    assert resolve_parallel_current_block_coupling(config) is CurrentBlockCoupling(
        program.value
    )


@pytest.mark.parametrize("program", list(VideoActionProgram))
def test_every_program_has_an_explicit_parallel_backend_mapping(
    program: VideoActionProgram,
) -> None:
    assert isinstance(
        parallel_runtime_mode_for_program(program),
        ParallelRuntimeMode,
    )


@pytest.mark.parametrize(
    "program",
    [
        VideoActionProgram.VIDEO_THEN_ACTION,
        VideoActionProgram.ACTION_THEN_VIDEO,
        VideoActionProgram.DECOUPLED_SAME_STEP,
    ],
)
def test_single_noisy_stream_programs_default_to_independent_timestep_clocks(
    program: VideoActionProgram,
) -> None:
    config = _policy_config(program=program)

    assert (
        resolve_parallel_joint_timestep_coupling(config)
        == JointTimestepCoupling.INDEPENDENT
    )


@pytest.mark.parametrize(
    "program",
    [
        VideoActionProgram.VIDEO_THEN_ACTION,
        VideoActionProgram.ACTION_THEN_VIDEO,
        VideoActionProgram.DECOUPLED_SAME_STEP,
    ],
)
def test_single_noisy_stream_programs_reject_inapplicable_clock_coupling(
    program: VideoActionProgram,
) -> None:
    with pytest.raises(ValueError, match="requires.*independent"):
        _policy_config(
            program=program,
            joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
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
        program=VideoActionProgram(coupling.value),
        joint_timestep_coupling=timestep_coupling,
    )

    assert resolve_parallel_joint_timestep_coupling(config) == timestep_coupling


@pytest.mark.parametrize(
    "visibility",
    list(HistoryStreamVisibility),
)
def test_explicit_history_stream_visibility_is_preserved(
    visibility: HistoryStreamVisibility,
) -> None:
    assert (
        resolve_parallel_history_stream_visibility(
            _policy_config(history_stream_visibility=visibility)
        )
        == visibility
    )


def test_history_visibility_projects_legacy_runtime_metadata() -> None:
    config = _policy_config(
        history_stream_visibility=(
            HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
        ),
    )

    assert (
        resolve_parallel_history_stream_visibility(config)
        == HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
    )
    assert (
        prefix_visibility_mode_for_policy(config)
        == "video_queries_video_only"
    )


@pytest.mark.parametrize(
    ("visibility", "expected"),
    [
        (HistoryStreamVisibility.FULL, "full_history"),
        (HistoryStreamVisibility.VIDEO_ONLY, "video_history_only"),
        (
            HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY,
            "video_queries_video_only",
        ),
    ],
)
def test_history_visibility_maps_to_exact_cache_contract(
    visibility: HistoryStreamVisibility,
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
    list(ContextConditionLatentSource),
)
def test_context_condition_latent_source_is_preserved(
    source: ContextConditionLatentSource,
) -> None:
    required_flags = (
        {
            "use_condition_latents": True,
            "require_condition_latents": True,
        }
        if source == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
        else {}
    )
    assert (
        resolve_parallel_context_condition_latent_source(
            _policy_config(
                context_condition_latent_source=source,
                **required_flags,
            )
        )
        == source
    )


def test_legacy_prefix_per_chunk_proprio_contract_is_explicit() -> None:
    assert not uses_legacy_prefix_per_chunk_proprio_contract(_policy_config())
    assert uses_legacy_prefix_per_chunk_proprio_contract(
        _policy_config(
            sequence_contract=(
                VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
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
