from __future__ import annotations

from open_wam.configs.enums import (
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    VideoActionSequenceContract,
)
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.common import (
    chunked_temporal_exact_profile_name_for_coupling,
)

__all__ = [
    "attention_profile_name_for_current_block_coupling",
    "prefix_visibility_mode_for_history_visibility",
    "prefix_visibility_mode_for_policy",
    "resolve_parallel_context_condition_latent_source",
    "resolve_parallel_current_block_coupling",
    "resolve_parallel_history_stream_visibility",
    "resolve_parallel_joint_timestep_coupling",
    "uses_legacy_prefix_per_chunk_proprio_contract",
]

_PREFIX_VISIBILITY_MODE_BY_HISTORY_VISIBILITY = {
    HistoryStreamVisibility.FULL: "full_history",
    HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY: (
        HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY.value
    ),
    HistoryStreamVisibility.VIDEO_ONLY: "video_history_only",
}


def resolve_parallel_history_stream_visibility(
    policy_config: ParallelStreamPolicyConfig,
) -> HistoryStreamVisibility:
    """Return the canonical clean-history stream visibility."""

    return policy_config.history_stream_visibility


def prefix_visibility_mode_for_policy(
    policy_config: ParallelStreamPolicyConfig,
) -> str:
    """Map policy history semantics to the exact-cache visibility contract."""

    return prefix_visibility_mode_for_history_visibility(
        resolve_parallel_history_stream_visibility(policy_config)
    )


def prefix_visibility_mode_for_history_visibility(
    visibility: HistoryStreamVisibility | str,
) -> str:
    """Translate shared history semantics into the exact-cache wire value."""

    return _PREFIX_VISIBILITY_MODE_BY_HISTORY_VISIBILITY[
        HistoryStreamVisibility(visibility)
    ]


def uses_legacy_prefix_per_chunk_proprio_contract(
    policy_config: ParallelStreamPolicyConfig,
) -> bool:
    """Return whether the compatibility prefix/proprio layout is selected."""

    return (
        VideoActionSequenceContract(policy_config.sequence_contract)
        == VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )


def resolve_parallel_context_condition_latent_source(
    policy_config: ParallelStreamPolicyConfig,
) -> ContextConditionLatentSource:
    """Resolve the clean condition-latent source for exact execution."""

    return policy_config.context_condition_latent_source


def resolve_parallel_current_block_coupling(
    policy_config: ParallelStreamPolicyConfig,
) -> CurrentBlockCoupling:
    """Return the low-level coupling derived from the public program."""

    return policy_config.current_block_coupling


def resolve_parallel_joint_timestep_coupling(
    policy_config: ParallelStreamPolicyConfig,
) -> JointTimestepCoupling:
    """Return the validated video/action noise-clock contract."""

    return policy_config.joint_timestep_coupling


def attention_profile_name_for_current_block_coupling(
    coupling: CurrentBlockCoupling,
) -> str:
    """Select the shared exact attention profile for one coupling mode."""

    return chunked_temporal_exact_profile_name_for_coupling(coupling.value)
