from __future__ import annotations

from dataclasses import replace as dataclass_replace

import torch

from open_wam.configs import MoTGeneralistTrainingMode, MoTPolicyConfig
from open_wam.contracts import SampleConstructionMetadata
from open_wam.models.common.flow_matching import VideoFlowMatchTrainArtifacts
from open_wam.models.common.joint_conditioning import (
    resolve_generalist_joint_conditioning_semantics,
    sample_conditioning_mode,
)
from open_wam.models.common.modality_slots import clean_noisy_slot_tensor, zero_loss_mask_like

from ..contracts import PolicyInferContext, PolicyTrainBatch


def sample_generalist_training_mode(
    probs: dict[MoTGeneralistTrainingMode, float],
    *,
    device: torch.device,
) -> MoTGeneralistTrainingMode:
    """Sample one M5 generalist regime per segment."""

    return sample_conditioning_mode(
        probs,
        enum_cls=MoTGeneralistTrainingMode,
        device=device,
        error_label="M5 generalist training mode",
    )


def resolve_generalist_training_metadata(
    batch: PolicyTrainBatch,
) -> tuple[MoTGeneralistTrainingMode | None, bool | None, str | None]:
    """Resolve an optional dataset-forced mode and conditioning metadata."""

    sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
    if sample_metadata is None:
        return None, None, None
    raw_mode = sample_metadata.generalist.mode_override
    mode = None if raw_mode is None else MoTGeneralistTrainingMode(raw_mode)
    return mode, sample_metadata.generalist.drop_text_conditioning, sample_metadata.generalist.source


def apply_generalist_training_mode(
    *,
    sampled_mode: MoTGeneralistTrainingMode,
    video_artifacts: VideoFlowMatchTrainArtifacts,
    noisy_actions: torch.Tensor,
    clean_actions: torch.Tensor,
    noisy_slot_timesteps: torch.Tensor,
    future_loss_mask: torch.Tensor,
    effective_action_mask: torch.Tensor | None,
    clean_action_condition_mask: torch.Tensor | None = None,
) -> tuple[
    VideoFlowMatchTrainArtifacts,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Apply a generalist denoising mode to packed MoT training tensors.

    Conditional modes place the clean modality in its noisy slot, preserve
    real clean history slots, set that modality's timestep to zero, and mask
    its loss. Joint mode returns every input object unchanged.

    ``effective_action_mask`` is the supervised action-loss mask. It may be
    narrower than the raw valid-action mask under fixed-segment sampling, so it
    must not hide clean action conditions from FDM/IDM context.
    """

    semantics = resolve_generalist_joint_conditioning_semantics(
        sampled_mode,
        joint_mode=MoTGeneralistTrainingMode.JOINT,
        action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    )

    if semantics.is_joint:
        return (
            video_artifacts,
            noisy_actions,
            clean_actions,
            noisy_slot_timesteps,
            future_loss_mask,
            effective_action_mask,
        )

    if semantics.clean_action_noisy_slot:
        new_noisy_actions = clean_noisy_slot_tensor(
            clean_actions.clone(),
            action_mask=clean_action_condition_mask,
        )
        new_noisy_slot_timesteps = torch.zeros_like(noisy_slot_timesteps)
        new_action_mask = zero_loss_mask_like(effective_action_mask, fallback_like=noisy_actions)
        return (
            video_artifacts,
            new_noisy_actions,
            clean_actions,
            new_noisy_slot_timesteps,
            future_loss_mask,
            new_action_mask,
        )

    if semantics.clean_video_noisy_slot:
        new_video_artifacts = dataclass_replace(
            video_artifacts,
            noisy_latents=video_artifacts.condition_latents.clone(),
            timesteps=torch.zeros_like(video_artifacts.timesteps),
        )
        new_future_loss_mask = torch.zeros_like(future_loss_mask)
        return (
            new_video_artifacts,
            noisy_actions,
            clean_actions,
            noisy_slot_timesteps,
            new_future_loss_mask,
            effective_action_mask,
        )

    raise ValueError(f"Unsupported MoTGeneralistTrainingMode {sampled_mode!r}.")


def generalist_forces_clean_video_condition(
    sampled_mode: MoTGeneralistTrainingMode | None,
) -> bool:
    """Return whether the selected mode requires a clean video condition."""

    if sampled_mode is None:
        return False
    semantics = resolve_generalist_joint_conditioning_semantics(
        sampled_mode,
        joint_mode=MoTGeneralistTrainingMode.JOINT,
        action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    )
    return semantics.force_clean_video_condition


def generalist_rollout_mode_from_value(
    value: object = "vanilla_joint_rollout",
) -> MoTGeneralistTrainingMode:
    """Map rollout and ablation labels to the M5 generalist mode."""

    raw_value = str(getattr(value, "value", value))
    aliases = {
        "joint": MoTGeneralistTrainingMode.JOINT,
        "vanilla_joint_rollout": MoTGeneralistTrainingMode.JOINT,
        "clean_action_feedback": MoTGeneralistTrainingMode.JOINT,
        "fdm": MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        "forced_action_joint_fdm": MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        "action_conditioned_video": MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        "idm": MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
        "video_conditioned_action": MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    }
    try:
        return aliases[raw_value]
    except KeyError as exc:
        raise ValueError(f"Unsupported M5 GJD rollout mode {raw_value!r}.") from exc


def resolve_generalist_rollout_mode(
    context: PolicyInferContext,
) -> MoTGeneralistTrainingMode:
    """Resolve the requested generalist rollout mode from inference context."""

    return generalist_rollout_mode_from_value(
        context.extra.get(
            "mot_generalist_rollout_mode",
            context.extra.get("action_conditioning_mode", "vanilla_joint_rollout"),
        )
    )


def is_generalist_conditional_rollout(mode: MoTGeneralistTrainingMode) -> bool:
    """Return whether rollout conditions one predicted modality on the other."""

    return mode in {
        MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    }


def generalist_rollout_enabled(policy_config: MoTPolicyConfig) -> bool:
    """Return whether the MoT policy was trained with generalist modes."""

    return getattr(policy_config, "mot_generalist_training_mode_probs", None) is not None


__all__ = [
    "apply_generalist_training_mode",
    "generalist_forces_clean_video_condition",
    "generalist_rollout_enabled",
    "generalist_rollout_mode_from_value",
    "is_generalist_conditional_rollout",
    "resolve_generalist_rollout_mode",
    "resolve_generalist_training_metadata",
    "sample_generalist_training_mode",
]
