"""DualExpert action/video block and timestep coupling semantics."""

from __future__ import annotations

from open_wam.configs import (
    CurrentBlockCoupling,
    DualExpertRuntimeMode,
    JointTimestepCoupling,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig


def resolve_dual_expert_current_block_coupling(config: DualExpertPolicyConfig) -> CurrentBlockCoupling:
    """Resolve dual-expert current-block coupling, defaulting to current behavior."""

    if config.current_block_coupling is None:
        if config.runtime_mode == DualExpertRuntimeMode.JOINT_DENOISE:
            return CurrentBlockCoupling.JOINT
        return CurrentBlockCoupling.VIDEO_THEN_ACTION
    return CurrentBlockCoupling(config.current_block_coupling)


def is_dual_expert_same_step_coupling(coupling: CurrentBlockCoupling) -> bool:
    """Return whether both streams participate in the same denoising step."""

    return coupling in {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }


def resolve_dual_expert_joint_timestep_coupling(
    config: DualExpertPolicyConfig,
    coupling: CurrentBlockCoupling,
) -> JointTimestepCoupling:
    """Resolve dual-expert joint-like action/video timestep coupling."""

    if coupling not in {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }:
        return JointTimestepCoupling.INDEPENDENT
    return JointTimestepCoupling(config.joint_timestep_coupling)


def should_couple_dual_expert_action_to_video_sigmas(
    config: DualExpertPolicyConfig,
    coupling: CurrentBlockCoupling,
) -> bool:
    """Return whether dual-expert rollout should integrate action on the video sigma clock."""

    return resolve_dual_expert_joint_timestep_coupling(config, coupling) in {
        JointTimestepCoupling.MATCH_SIGMA,
        JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    }


__all__ = [
    "is_dual_expert_same_step_coupling",
    "resolve_dual_expert_current_block_coupling",
    "resolve_dual_expert_joint_timestep_coupling",
    "should_couple_dual_expert_action_to_video_sigmas",
]
