from __future__ import annotations

import torch

from open_wam.configs.enums import DynamicsObjective, JointTimestepCoupling
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.contracts import DYNAMICS_ROUTING_SOURCE_METADATA_KEY
from open_wam.models.common.dynamics_objectives import (
    DynamicsTrainingPlan,
    apply_dynamics_training_plan,
)

from .runtime_semantics import resolve_parallel_joint_timestep_coupling
from .training_artifact_contracts import ParallelTrainArtifacts


def _annotate_dynamics_training_artifacts(
    *,
    artifacts: ParallelTrainArtifacts,
    objective: DynamicsObjective,
    joint_timestep_coupling: JointTimestepCoupling,
    routed_objective: DynamicsObjective | None,
    text_dropped: bool,
    training_source: str | None,
    video_condition_source: str,
) -> None:
    artifacts.input_dict[DYNAMICS_ROUTING_SOURCE_METADATA_KEY] = training_source
    artifacts.input_dict["joint_denoise_training_mode"] = objective.value
    artifacts.input_dict["joint_timestep_coupling"] = joint_timestep_coupling.value
    artifacts.input_dict["joint_denoise_training_mode_override"] = (
        None if routed_objective is None else objective.value
    )
    artifacts.input_dict["joint_denoise_text_dropped"] = bool(text_dropped)
    artifacts.input_dict["video_condition_source"] = video_condition_source


def apply_parallel_dynamics_training_plan(
    *,
    artifacts: ParallelTrainArtifacts,
    policy_config: ParallelStreamPolicyConfig,
    video_latents: torch.Tensor,
    action_latents: torch.Tensor,
    action_mask_latents: torch.Tensor | None,
    plan: DynamicsTrainingPlan,
) -> None:
    """Translate one shared dynamics plan into parallel-stream artifacts."""

    if int(video_latents.shape[0]) != 1:
        raise ValueError(
            "Dynamics training applies one objective per runtime batch. "
            "Use train_batch_size=1 to preserve the intended one-mode-per-segment contract."
        )
    objective = plan.objective
    semantics = plan.semantics
    artifacts.dynamics_objective = objective
    joint_timestep_coupling = resolve_parallel_joint_timestep_coupling(policy_config)
    if semantics.is_joint:
        latent_dict = artifacts.input_dict["latent_dict"]
        action_dict = artifacts.input_dict["action_dict"]
        assert isinstance(latent_dict, dict)
        assert isinstance(action_dict, dict)
        text_emb = latent_dict["text_emb"]
        text_dropped = semantics.drop_text_conditioning
        if text_dropped:
            text_emb = torch.zeros_like(text_emb)
            latent_dict["text_emb"] = text_emb
            action_dict["text_emb"] = text_emb
        _annotate_dynamics_training_artifacts(
            artifacts=artifacts,
            objective=objective,
            joint_timestep_coupling=joint_timestep_coupling,
            routed_objective=plan.routed_objective,
            text_dropped=text_dropped,
            training_source=plan.source,
            video_condition_source=artifacts.input_dict.get(
                "video_condition_source",
                "video_latents",
            ),
        )
        if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
            artifacts.input_dict["joint_denoise_shared_sigmas"] = (
                artifacts.latent_scheduler.sigma_for_timesteps(
                    latent_dict["timesteps"][0]
                )
                .detach()
                .clone()
            )
        return

    latent_dict = artifacts.input_dict["latent_dict"]
    action_dict = artifacts.input_dict["action_dict"]
    assert isinstance(latent_dict, dict)
    assert isinstance(action_dict, dict)
    training_tensors = apply_dynamics_training_plan(
        plan,
        clean_video=video_latents,
        noisy_video=latent_dict["noisy_latents"],
        video_targets=latent_dict["targets"],
        video_timesteps=latent_dict["timesteps"],
        video_loss_mask=latent_dict["loss_mask"],
        clean_action=action_latents,
        noisy_action=action_dict["noisy_latents"],
        action_targets=action_dict["targets"],
        action_timesteps=action_dict["timesteps"],
        action_loss_mask=action_dict["loss_mask"],
        clean_action_mask=action_mask_latents,
    )
    latent_dict["noisy_latents"] = training_tensors.noisy_video
    latent_dict["targets"] = training_tensors.video_targets
    latent_dict["timesteps"] = training_tensors.video_timesteps
    latent_dict["loss_mask"] = training_tensors.video_loss_mask
    action_dict["noisy_latents"] = training_tensors.noisy_action
    action_dict["targets"] = training_tensors.action_targets
    action_dict["timesteps"] = training_tensors.action_timesteps
    action_dict["loss_mask"] = training_tensors.action_loss_mask

    text_emb = latent_dict["text_emb"]
    # Conditional dynamics probes intentionally remove task text while keeping
    # mode text tokens and hidden-state proprio payloads handled by the variant.
    text_dropped = semantics.drop_text_conditioning
    if text_dropped:
        text_emb = torch.zeros_like(text_emb)
    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    # Conditional FDM/IDM keep sampled future chunk geometry while exposing
    # only the immediately preceding boundary frame as clean history.
    artifacts.input_dict["window_size"] = semantics.attention_window_size(
        fallback_window_size=int(artifacts.input_dict["window_size"]),
    )
    if semantics.is_conditional:
        artifacts.input_dict["history_stream_visibility"] = (
            semantics.resolve_history_stream_visibility(
                fallback=artifacts.input_dict.get(
                    "history_stream_visibility",
                    policy_config.history_stream_visibility,
                ),
            ).value
        )
        if plan.sequence is None:  # pragma: no cover - shared-plan invariant
            raise RuntimeError(
                "Conditional dynamics plan is missing its sequence layout."
            )
        artifacts.input_dict["conditional_history_policy"] = (
            plan.sequence.history_policy
        )
    artifacts.input_dict["generalist_conditional_history_chunks"] = int(
        semantics.history_frame_count
    )
    _annotate_dynamics_training_artifacts(
        artifacts=artifacts,
        objective=objective,
        joint_timestep_coupling=joint_timestep_coupling,
        routed_objective=plan.routed_objective,
        text_dropped=text_dropped,
        training_source=plan.source,
        video_condition_source="video_latents_target_only",
    )
    if joint_timestep_coupling in {
        JointTimestepCoupling.MATCH_SIGMA,
        JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    }:
        if semantics.video_loss_active:
            shared_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
                latent_dict["timesteps"][0]
            )
        else:
            shared_sigmas = artifacts.action_scheduler.sigma_for_timesteps(
                action_dict["timesteps"][0]
            )
        artifacts.input_dict["joint_denoise_shared_sigmas"] = (
            shared_sigmas.detach().clone()
        )


def apply_parallel_prefix_dynamics_training_plan(
    *,
    artifacts: ParallelTrainArtifacts,
    policy_config: ParallelStreamPolicyConfig,
    plan: DynamicsTrainingPlan,
) -> None:
    """Apply the shared joint plan to external-prefix train artifacts.

    External-prefix assembly is the ordinary planning layout. Conditional
    dynamics uses in-sequence t0 and must take the target-only artifact path.
    """

    latent_dict = artifacts.input_dict["latent_dict"]
    action_dict = artifacts.input_dict["action_dict"]
    assert isinstance(latent_dict, dict)
    assert isinstance(action_dict, dict)
    if int(latent_dict["noisy_latents"].shape[0]) != 1:
        raise ValueError(
            "Dynamics training applies one objective per runtime batch. "
            "Use train_batch_size=1 to preserve the intended one-mode-per-segment contract."
        )
    objective = plan.objective
    semantics = plan.semantics
    artifacts.dynamics_objective = objective
    if semantics.is_conditional:
        raise ValueError(
            "Conditional dynamics cannot use external-prefix planning artifacts; "
            "route the canonical target-only t0 sample instead."
        )

    text_emb = latent_dict["text_emb"]
    text_dropped = semantics.drop_text_conditioning
    if text_dropped:
        text_emb = torch.zeros_like(text_emb)
        latent_dict["text_emb"] = text_emb
        action_dict["text_emb"] = text_emb

    joint_timestep_coupling = resolve_parallel_joint_timestep_coupling(policy_config)
    _annotate_dynamics_training_artifacts(
        artifacts=artifacts,
        objective=objective,
        joint_timestep_coupling=joint_timestep_coupling,
        routed_objective=plan.routed_objective,
        text_dropped=text_dropped,
        training_source=plan.source,
        video_condition_source=artifacts.input_dict.get(
            "video_condition_source",
            "condition_latents_prefix",
        ),
    )
    if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
        prefix_frames = max(
            0, int(artifacts.input_dict.get("prefix_condition_frames", 0))
        )
        video_timesteps = latent_dict["timesteps"][0]
        if prefix_frames:
            video_timesteps = video_timesteps[prefix_frames:]
        artifacts.input_dict["joint_denoise_shared_sigmas"] = (
            artifacts.latent_scheduler.sigma_for_timesteps(video_timesteps)
            .detach()
            .clone()
        )


__all__ = [
    "apply_parallel_dynamics_training_plan",
    "apply_parallel_prefix_dynamics_training_plan",
]
