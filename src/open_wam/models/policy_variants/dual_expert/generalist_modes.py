from __future__ import annotations

from dataclasses import replace as dataclass_replace

import torch

from open_wam.configs import GeneralistDenoisingMode
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.configs.policy_video_action import (
    fixed_conditioning_mode_for_program,
    resolve_fixed_conditioning_mode,
)
from open_wam.contracts import SampleConstructionMetadata
from open_wam.models.common.flow_training import (
    VideoFlowMatchTrainArtifacts,
)
from open_wam.models.common.joint_conditioning import (
    resolve_generalist_joint_conditioning_semantics,
    sample_conditioning_mode,
)
from open_wam.models.common.modality_slots import (
    clean_noisy_slot_tensor,
    zero_loss_mask_like,
)

from ..contracts import PolicyInferContext, PolicyTrainBatch


def sample_generalist_training_mode(
    probs: dict[GeneralistDenoisingMode, float],
    *,
    device: torch.device,
) -> GeneralistDenoisingMode:
    """Sample one dual-expert generalist regime per segment."""

    return sample_conditioning_mode(
        probs,
        enum_cls=GeneralistDenoisingMode,
        device=device,
        error_label="dual-expert generalist training mode",
    )


def resolve_generalist_training_metadata(
    batch: PolicyTrainBatch,
) -> tuple[GeneralistDenoisingMode | None, bool | None, str | None]:
    """Resolve an optional dataset-forced mode and conditioning metadata."""

    sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
    if sample_metadata is None:
        return None, None, None
    raw_mode = sample_metadata.generalist.mode_override
    mode = None if raw_mode is None else GeneralistDenoisingMode(raw_mode)
    return mode, sample_metadata.generalist.drop_text_conditioning, sample_metadata.generalist.source


def resolve_generalist_training_mode(
    policy_config: DualExpertPolicyConfig,
    batch: PolicyTrainBatch,
    *,
    device: torch.device,
) -> tuple[GeneralistDenoisingMode | None, GeneralistDenoisingMode | None, bool | None, str | None]:
    """Resolve fixed, dataset-forced, or sampled mode in precedence order.

    Standalone conditional programs own their mode. Dataset metadata may
    confirm that mode but cannot redirect it; mixed GJD retains its existing
    per-source forced-mode behavior.
    """

    forced_mode, drop_text, source = resolve_generalist_training_metadata(batch)
    fixed_mode = resolve_fixed_conditioning_mode(policy_config)
    if fixed_mode is not None:
        mode_owner = (
            f"`program = {policy_config.program.value}`"
            if fixed_conditioning_mode_for_program(policy_config.program) is not None
            else "`generalist_denoising_mode_probs`"
        )
        if forced_mode is not None and forced_mode != fixed_mode:
            raise ValueError(
                f"{mode_owner} requires training mode "
                f"{fixed_mode.value!r}, but sample metadata forced {forced_mode.value!r}."
            )
        if forced_mode is not None:
            return fixed_mode, forced_mode, drop_text, source
        probabilities = policy_config.generalist_denoising_mode_probs
        if probabilities is None:  # pragma: no cover - typed config derives the one-hot map.
            raise RuntimeError(
                f"{mode_owner} is missing its fixed mode distribution."
            )
        sampled_mode = sample_generalist_training_mode(probabilities, device=device)
        if sampled_mode != fixed_mode:  # pragma: no cover - typed config validates one-hot ownership.
            raise RuntimeError(
                f"{mode_owner} sampled unexpected mode "
                f"{sampled_mode.value!r}."
            )
        return sampled_mode, forced_mode, drop_text, source
    if forced_mode is not None:
        return forced_mode, forced_mode, drop_text, source
    probabilities = policy_config.generalist_denoising_mode_probs
    if probabilities is None:
        return None, None, drop_text, source
    return (
        sample_generalist_training_mode(probabilities, device=device),
        None,
        drop_text,
        source,
    )


def apply_generalist_training_mode(
    *,
    sampled_mode: GeneralistDenoisingMode,
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
    """Apply a generalist denoising mode to packed DualExpert training tensors.

    Conditional modes place the clean modality in its noisy slot, preserve
    real clean history slots, set that modality's timestep to zero, and mask
    its loss. Joint mode returns every input object unchanged.

    ``effective_action_mask`` is the supervised action-loss mask. It may be
    narrower than the raw valid-action mask under fixed-segment sampling, so it
    must not hide clean action conditions from FDM/IDM context.
    """

    semantics = resolve_generalist_joint_conditioning_semantics(
        sampled_mode,
        joint_mode=GeneralistDenoisingMode.JOINT,
        action_conditioned_video_mode=GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
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

    raise ValueError(f"Unsupported GeneralistDenoisingMode {sampled_mode!r}.")


def generalist_forces_clean_video_condition(
    sampled_mode: GeneralistDenoisingMode | None,
) -> bool:
    """Return whether the selected mode requires a clean video condition."""

    if sampled_mode is None:
        return False
    semantics = resolve_generalist_joint_conditioning_semantics(
        sampled_mode,
        joint_mode=GeneralistDenoisingMode.JOINT,
        action_conditioned_video_mode=GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
    )
    return semantics.force_clean_video_condition


def generalist_rollout_mode_from_value(
    value: object = "vanilla_joint_rollout",
) -> GeneralistDenoisingMode:
    """Map rollout and ablation labels to the dual-expert generalist mode."""

    raw_value = str(getattr(value, "value", value))
    aliases = {
        "joint": GeneralistDenoisingMode.JOINT,
        "vanilla_joint_rollout": GeneralistDenoisingMode.JOINT,
        "clean_action_feedback": GeneralistDenoisingMode.JOINT,
        "fdm": GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
        "forced_action_joint_fdm": GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
        "action_conditioned_video": GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
        "idm": GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
        "video_conditioned_action": GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
    }
    try:
        return aliases[raw_value]
    except KeyError as exc:
        raise ValueError(f"Unsupported dual-expert GJD rollout mode {raw_value!r}.") from exc


def resolve_generalist_rollout_mode(
    context: PolicyInferContext,
    policy_config: DualExpertPolicyConfig | None = None,
) -> GeneralistDenoisingMode:
    """Resolve the requested generalist rollout mode from inference context."""

    requested_value = context.extra.get(
        "dual_expert_generalist_rollout_mode",
        context.extra.get("action_conditioning_mode"),
    )
    fixed_mode = None
    if policy_config is not None:
        fixed_mode = resolve_fixed_conditioning_mode(policy_config)
    if fixed_mode is not None:
        if requested_value is None:
            return fixed_mode
        requested_mode = generalist_rollout_mode_from_value(requested_value)
        if requested_mode != fixed_mode:
            raise ValueError(
                "The configured fixed conditional mode requires "
                f"{fixed_mode.value!r}, but inference requested {requested_mode.value!r}."
            )
        return fixed_mode
    return generalist_rollout_mode_from_value(
        "vanilla_joint_rollout" if requested_value is None else requested_value
    )


def is_generalist_conditional_rollout(mode: GeneralistDenoisingMode) -> bool:
    """Return whether rollout conditions one predicted modality on the other."""

    return mode in {
        GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
        GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
    }


def generalist_rollout_enabled(policy_config: DualExpertPolicyConfig) -> bool:
    """Return whether the DualExpert policy was trained with generalist modes."""

    return policy_config.generalist_denoising_mode_probs is not None


__all__ = [
    "apply_generalist_training_mode",
    "generalist_forces_clean_video_condition",
    "generalist_rollout_enabled",
    "generalist_rollout_mode_from_value",
    "is_generalist_conditional_rollout",
    "resolve_generalist_rollout_mode",
    "resolve_generalist_training_metadata",
    "resolve_generalist_training_mode",
    "sample_generalist_training_mode",
]
