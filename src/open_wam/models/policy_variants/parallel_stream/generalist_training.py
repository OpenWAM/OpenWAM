from __future__ import annotations

from typing import Any, Protocol

import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.enums import (
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelHistoryStreamVisibility,
)
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.contracts import GENERALIST_TRAINING_SOURCE_METADATA_KEY
from open_wam.models.common.attention_profiles import (
    CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
)
from open_wam.models.common.flow_matching import FlowMatchScheduler
from open_wam.models.common.flow_noise_plan import sample_joint_denoise_timestep_values
from open_wam.models.common.joint_conditioning import (
    JointConditioningModeSemantics,
    resolve_generalist_joint_conditioning_semantics,
    sample_conditioning_mode,
)
from open_wam.models.common.modality_slots import force_clean_noisy_slot

from .latent_conditioning import resolve_full_window_condition_latents
from .runtime_semantics import resolve_parallel_joint_timestep_coupling
from .training_noise import (
    build_parallel_flow_noise_artifacts,
    share_video_scheduler_grid_with_action_scheduler,
)


class ParallelTrainArtifacts(Protocol):
    """Structural train-artifact interface consumed by GJD mode mutation."""

    @property
    def input_dict(self) -> dict[str, Any]: ...

    @property
    def latent_scheduler(self) -> FlowMatchScheduler: ...

    @property
    def action_scheduler(self) -> FlowMatchScheduler: ...


def sample_generalist_joint_denoise_training_mode(
    policy_config: ParallelStreamPolicyConfig,
    *,
    device: torch.device,
) -> JointDenoiseTrainingMode:
    """Sample one FSDP-coordinated parallel-stream GJD training mode."""

    probs = policy_config.joint_denoise_training_mode_probs
    if probs is None:
        return JointDenoiseTrainingMode.JOINT
    return sample_conditioning_mode(
        probs,
        enum_cls=JointDenoiseTrainingMode,
        device=device,
        error_label="Generalist joint-denoise training mode",
    )


def _resolve_training_mode_semantics(
    mode: JointDenoiseTrainingMode | str,
    *,
    drop_text_conditioning: bool | None = None,
) -> JointConditioningModeSemantics:
    return resolve_generalist_joint_conditioning_semantics(
        mode,
        joint_mode=JointDenoiseTrainingMode.JOINT,
        action_conditioned_video_mode=JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION,
        drop_text_conditioning=drop_text_conditioning,
    )


def _annotate_generalist_training_artifacts(
    *,
    artifacts: ParallelTrainArtifacts,
    policy_config: ParallelStreamPolicyConfig,
    mode: JointDenoiseTrainingMode,
    joint_timestep_coupling: JointTimestepCoupling,
    training_mode_override: JointDenoiseTrainingMode | str | None,
    text_dropped: bool,
    training_source: str | None,
    video_condition_source: str,
) -> None:
    artifacts.input_dict["variant_profile"] = policy_config.variant_profile.value
    artifacts.input_dict["generalist_training_paradigm"] = policy_config.generalist_training_paradigm.value
    artifacts.input_dict[GENERALIST_TRAINING_SOURCE_METADATA_KEY] = training_source
    artifacts.input_dict["joint_denoise_training_mode"] = mode.value
    artifacts.input_dict["joint_timestep_coupling"] = joint_timestep_coupling.value
    artifacts.input_dict["joint_denoise_training_mode_override"] = (
        None if training_mode_override is None else mode.value
    )
    artifacts.input_dict["joint_denoise_text_dropped"] = bool(text_dropped)
    artifacts.input_dict["joint_denoise_training_mode_probs"] = {
        mode_key.value: float(prob)
        for mode_key, prob in (policy_config.joint_denoise_training_mode_probs or {}).items()
    }
    artifacts.input_dict["video_condition_source"] = video_condition_source


def apply_generalist_joint_denoise_training_mode(
    *,
    artifacts: ParallelTrainArtifacts,
    policy_config: ParallelStreamPolicyConfig,
    backbone_config: SharedVideoTransformerConfig,
    video_latents: torch.Tensor,
    condition_latents: torch.Tensor | None,
    action_latents: torch.Tensor,
    action_mask_latents: torch.Tensor | None,
    frame_shift: int,
    training_mode_override: JointDenoiseTrainingMode | str | None = None,
    drop_text_conditioning: bool | None = None,
    training_source: str | None = None,
) -> None:
    """Apply one parallel-stream GJD mode to prepared exact train artifacts."""

    if int(video_latents.shape[0]) != 1:
        raise ValueError(
            "`generalist_joint_denoising` currently samples one conditioning mode per runtime batch. "
            "Use train_batch_size=1 to preserve the intended one-mode-per-segment contract."
        )
    mode = (
        JointDenoiseTrainingMode(training_mode_override)
        if training_mode_override is not None
        else sample_generalist_joint_denoise_training_mode(
            policy_config,
            device=video_latents.device,
        )
    )
    semantics = _resolve_training_mode_semantics(
        mode,
        drop_text_conditioning=drop_text_conditioning,
    )
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
        _annotate_generalist_training_artifacts(
            artifacts=artifacts,
            policy_config=policy_config,
            mode=mode,
            joint_timestep_coupling=joint_timestep_coupling,
            training_mode_override=training_mode_override,
            text_dropped=text_dropped,
            training_source=training_source,
            video_condition_source=artifacts.input_dict.get(
                "video_condition_source",
                "condition_latents"
                if condition_latents is not None
                else "video_latents",
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

    num_frames = int(video_latents.shape[2])
    resolved_condition_latents, condition_source = resolve_full_window_condition_latents(
        video_latents,
        condition_latents,
        label="Generalist joint-denoise",
    )
    timestep_plan = sample_joint_denoise_timestep_values(
        video_scheduler=artifacts.latent_scheduler,
        action_scheduler=artifacts.action_scheduler,
        num_frames=num_frames,
        device=video_latents.device,
        coupling=joint_timestep_coupling,
        clean_video=semantics.clean_video_noisy_slot,
        clean_action=semantics.clean_action_noisy_slot,
    )
    if joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
        share_video_scheduler_grid_with_action_scheduler(
            latent_scheduler=artifacts.latent_scheduler,
            action_scheduler=artifacts.action_scheduler,
            device=video_latents.device,
        )

    latent_dict = build_parallel_flow_noise_artifacts(
        video_latents,
        train_scheduler=artifacts.latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
        condition_latent=resolved_condition_latents,
        frame_shift=frame_shift,
        timestep_values=timestep_plan.video_timesteps,
        sigma_values=timestep_plan.video_sigma_values,
    )
    action_dict = build_parallel_flow_noise_artifacts(
        action_latents,
        train_scheduler=artifacts.action_scheduler,
        action_mask=action_mask_latents,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
        frame_shift=frame_shift,
        timestep_values=timestep_plan.action_timesteps,
        sigma_values=timestep_plan.action_sigma_values,
    )

    if semantics.clean_action_noisy_slot:
        force_clean_noisy_slot(
            action_dict,
            action_latents,
            action_mask=action_mask_latents,
        )
        action_dict["loss_mask"] = torch.zeros_like(
            artifacts.input_dict["action_dict"]["loss_mask"]
        )
    else:
        action_dict["loss_mask"] = artifacts.input_dict["action_dict"]["loss_mask"]

    if semantics.clean_video_noisy_slot:
        force_clean_noisy_slot(
            latent_dict,
            video_latents
            if resolved_condition_latents is None
            else resolved_condition_latents,
        )
        latent_dict["loss_mask"] = torch.zeros_like(
            artifacts.input_dict["latent_dict"]["loss_mask"]
        )
    else:
        latent_dict["loss_mask"] = artifacts.input_dict["latent_dict"]["loss_mask"]

    text_emb = artifacts.input_dict["latent_dict"]["text_emb"]
    # Conditional dynamics probes intentionally remove task text while keeping
    # mode text tokens and hidden-state proprio payloads handled by the variant.
    text_dropped = semantics.drop_text_conditioning
    if text_dropped:
        text_emb = torch.zeros_like(text_emb)
    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    action_dict["actions_mask"] = artifacts.input_dict["action_dict"]["actions_mask"]

    artifacts.input_dict["latent_dict"] = latent_dict
    artifacts.input_dict["action_dict"] = action_dict
    # Conditional FDM/IDM keep the sampled GJD chunk geometry while restricting
    # clean history to the immediately previous video chunk.
    artifacts.input_dict["window_size"] = semantics.attention_window_size(
        fallback_window_size=int(artifacts.input_dict["window_size"]),
    )
    if semantics.is_conditional:
        artifacts.input_dict["history_stream_visibility"] = ParallelHistoryStreamVisibility.VIDEO_ONLY.value
        artifacts.input_dict["conditional_history_policy"] = (
            artifacts.input_dict.get("conditional_history_policy")
            or CONDITIONAL_HISTORY_POLICY_PREVIOUS_BOUNDARY_VIDEO_ONLY
        )
    artifacts.input_dict["generalist_conditional_history_chunks"] = int(semantics.conditional_history_chunks)
    _annotate_generalist_training_artifacts(
        artifacts=artifacts,
        policy_config=policy_config,
        mode=mode,
        joint_timestep_coupling=joint_timestep_coupling,
        training_mode_override=training_mode_override,
        text_dropped=text_dropped,
        training_source=training_source,
        video_condition_source=condition_source,
    )
    if timestep_plan.shared_sigma_values is not None:
        artifacts.input_dict["joint_denoise_shared_sigmas"] = (
            timestep_plan.shared_sigma_values.detach().clone()
        )


def apply_generalist_legacy_prefix_joint_training_mode(
    *,
    artifacts: ParallelTrainArtifacts,
    policy_config: ParallelStreamPolicyConfig,
    training_mode_override: JointDenoiseTrainingMode | str | None = None,
    drop_text_conditioning: bool | None = None,
    training_source: str | None = None,
) -> None:
    """Annotate legacy-prefix exact artifacts as pure GJD joint training.

    The legacy-prefix contract provides one clean condition frame plus noisy
    target chunks. That is parity-compatible with joint denoising, but it is not
    enough to express FDM/IDM clean-modality conditioning. Reject those modes
    explicitly instead of silently training a different task.
    """

    latent_dict = artifacts.input_dict["latent_dict"]
    action_dict = artifacts.input_dict["action_dict"]
    assert isinstance(latent_dict, dict)
    assert isinstance(action_dict, dict)
    if int(latent_dict["noisy_latents"].shape[0]) != 1:
        raise ValueError(
            "`generalist_joint_denoising` currently samples one conditioning mode per runtime batch. "
            "Use train_batch_size=1 to preserve the intended one-mode-per-segment contract."
        )
    if training_mode_override is None:
        probs = policy_config.joint_denoise_training_mode_probs or {}
        for mode, prob in probs.items():
            semantics = _resolve_training_mode_semantics(mode)
            if semantics.is_conditional and float(prob) > 0.0:
                raise ValueError(
                    "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` currently supports "
                    "only pure `joint` generalist joint-denoise training. Conditional GJD modes need full clean "
                    "modality target slots, not just a one-frame prefix condition."
                )
        mode = JointDenoiseTrainingMode.JOINT
    else:
        mode = JointDenoiseTrainingMode(training_mode_override)
        semantics = _resolve_training_mode_semantics(mode)
        if semantics.is_conditional:
            raise ValueError(
                "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` cannot force "
                f"`joint_denoise_training_mode={mode.value}`; only `joint` is parity-compatible."
            )

    text_emb = latent_dict["text_emb"]
    semantics = _resolve_training_mode_semantics(
        mode,
        drop_text_conditioning=drop_text_conditioning,
    )
    text_dropped = semantics.drop_text_conditioning
    if text_dropped:
        text_emb = torch.zeros_like(text_emb)
        latent_dict["text_emb"] = text_emb
        action_dict["text_emb"] = text_emb

    joint_timestep_coupling = resolve_parallel_joint_timestep_coupling(policy_config)
    _annotate_generalist_training_artifacts(
        artifacts=artifacts,
        policy_config=policy_config,
        mode=mode,
        joint_timestep_coupling=joint_timestep_coupling,
        training_mode_override=training_mode_override,
        text_dropped=text_dropped,
        training_source=training_source,
        video_condition_source=artifacts.input_dict.get(
            "video_condition_source",
            "condition_latents_prefix",
        ),
    )
    if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
        prefix_frames = max(0, int(artifacts.input_dict.get("prefix_condition_frames", 0)))
        video_timesteps = latent_dict["timesteps"][0]
        if prefix_frames:
            video_timesteps = video_timesteps[prefix_frames:]
        artifacts.input_dict["joint_denoise_shared_sigmas"] = (
            artifacts.latent_scheduler.sigma_for_timesteps(video_timesteps)
            .detach()
            .clone()
        )


__all__ = [
    "ParallelTrainArtifacts",
    "apply_generalist_joint_denoise_training_mode",
    "apply_generalist_legacy_prefix_joint_training_mode",
    "sample_generalist_joint_denoise_training_mode",
]
