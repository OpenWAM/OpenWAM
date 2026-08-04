from __future__ import annotations

from open_wam.configs.enums import (
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    ParallelRuntimeMode,
    VideoActionSequenceContract,
)
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.common import (
    chunked_temporal_exact_profile_name_for_coupling,
)

__all__ = [
    "attention_profile_name_for_current_block_coupling",
    "prefix_visibility_mode_for_policy",
    "resolve_parallel_context_condition_latent_source",
    "resolve_parallel_current_block_coupling",
    "resolve_parallel_history_stream_visibility",
    "resolve_parallel_joint_timestep_coupling",
    "uses_legacy_prefix_per_chunk_proprio_contract",
]


def resolve_parallel_history_stream_visibility(
    policy_config: ParallelStreamPolicyConfig,
) -> HistoryStreamVisibility:
    """Resolve the typed history visibility, including its legacy alias."""

    value = getattr(
        policy_config,
        "history_stream_visibility",
        HistoryStreamVisibility.FULL,
    )
    resolved = HistoryStreamVisibility(value)
    if resolved == HistoryStreamVisibility.FULL and bool(
        getattr(policy_config, "preserve_video_pretrain_history", False)
    ):
        return HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
    return resolved


def prefix_visibility_mode_for_policy(
    policy_config: ParallelStreamPolicyConfig,
) -> str:
    """Map policy history semantics to the exact-cache visibility contract."""

    history_visibility = resolve_parallel_history_stream_visibility(policy_config)
    if history_visibility == HistoryStreamVisibility.VIDEO_ONLY:
        return "video_history_only"
    if (
        history_visibility
        == HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
    ):
        return "preserve_video_pretrain_history"
    return (
        "preserve_video_pretrain_history"
        if bool(getattr(policy_config, "preserve_video_pretrain_history", False))
        else "full_history"
    )


def uses_legacy_prefix_per_chunk_proprio_contract(
    policy_config: ParallelStreamPolicyConfig,
) -> bool:
    """Return whether the compatibility prefix/proprio layout is selected."""

    return (
        VideoActionSequenceContract(
            policy_config.sequence_contract
        )
        == VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )


def resolve_parallel_context_condition_latent_source(
    policy_config: ParallelStreamPolicyConfig,
) -> ContextConditionLatentSource:
    """Resolve the clean condition-latent source for exact execution."""

    return ContextConditionLatentSource(
        getattr(
            policy_config,
            "context_condition_latent_source",
            ContextConditionLatentSource.VIDEO_LATENTS,
        )
    )


def resolve_parallel_current_block_coupling(
    policy_config: ParallelStreamPolicyConfig,
) -> CurrentBlockCoupling:
    """Resolve legacy parallel-stream runtime knobs into an explicit current-block mode."""

    if policy_config.current_block_coupling is not None:
        return CurrentBlockCoupling(policy_config.current_block_coupling)
    if (
        policy_config.runtime_mode
        == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    ):
        return CurrentBlockCoupling.JOINT
    return CurrentBlockCoupling.VIDEO_THEN_ACTION


def resolve_parallel_joint_timestep_coupling(
    policy_config: ParallelStreamPolicyConfig,
) -> JointTimestepCoupling:
    """Resolve how joint-like programs synchronize video and action clocks."""

    if resolve_parallel_current_block_coupling(policy_config) not in {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }:
        return JointTimestepCoupling.INDEPENDENT
    return JointTimestepCoupling(policy_config.joint_timestep_coupling)


def attention_profile_name_for_current_block_coupling(
    coupling: CurrentBlockCoupling,
) -> str:
    """Select the shared exact attention profile for one coupling mode."""

    return chunked_temporal_exact_profile_name_for_coupling(coupling.value)
