"""Construct parallel-stream training inputs without executing the model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange

from open_wam.configs.backbone import (
    SharedVideoTransformerConfig,
    resolve_stage_attention_mode,
)
from open_wam.configs.enums import (
    CurrentBlockCoupling,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelContextConditionLatentSource,
    ParallelStreamVariantProfile,
)
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common.flow_matching import FlowMatchScheduler
from open_wam.models.common.flow_noise_plan import (
    clean_timestep_values,
    sample_joint_denoise_timestep_values,
)
from open_wam.models.common.modality_slots import (
    force_clean_noisy_slot,
    zero_condition_slot,
)
from open_wam.models.visual_tower.reference_transformer import (
    preferred_reference_dtype,
)

from .generalist_training import (
    apply_generalist_joint_denoise_training_mode as _apply_generalist_joint_denoise_training_mode,
    apply_generalist_legacy_prefix_joint_training_mode as _apply_generalist_legacy_prefix_joint_training_mode,
)
from .latent_conditioning import (
    resolve_full_window_condition_latents as _resolve_full_condition_latents,
    select_first_frame_condition_latents as _select_first_frame_condition_latents,
)
from .runtime_semantics import (
    attention_profile_name_for_current_block_coupling as _attention_profile_name_for_current_block_coupling,
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
)
from .training_noise import (
    build_parallel_flow_noise_artifacts as _add_noise,
    sample_coupled_parallel_timestep_values as _sample_coupled_timestep_values,
    sample_index_matched_timestep_values as _sample_index_matched_timestep_values,
    sample_shared_video_schedule_timestep_values as _sample_shared_video_schedule_timestep_values,
    share_video_scheduler_grid_with_action_scheduler as _share_video_scheduler_grid_with_action_scheduler,
)


@dataclass
class LingbotParallelTrainArtifacts:
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    latent_scheduler: FlowMatchScheduler
    action_scheduler: FlowMatchScheduler


# Generic public name; retain the LingBot name for checkpoint-era import compatibility.
ParallelTrainArtifacts = LingbotParallelTrainArtifacts


def prepare_parallel_exact_train_artifacts(
    *,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    condition_latents: torch.Tensor | None = None,
    chunk_size_override: int | None = None,
    window_size_override: int | None = None,
    loss_frame_start: int | None = None,
    loss_frame_end: int | None = None,
    latent_loss_frame_start: int | None = None,
    latent_loss_frame_end: int | None = None,
    action_loss_frame_start: int | None = None,
    action_loss_frame_end: int | None = None,
    frame_shift: int = 0,
    chunk_origin_frame: int = 0,
    singleton_chunk_frame: int | None = None,
    conditional_history_policy: str | None = None,
    force_clean_video_condition: bool = False,
) -> LingbotParallelTrainArtifacts:
    batch_size, _, num_frames, _, _ = video_latents.shape
    context_condition_source = resolve_parallel_context_condition_latent_source(policy_config)
    if context_condition_source == ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT:
        if condition_latents is None:
            raise ValueError(
                "`context_condition_latent_source=single_frame_condition_latent` requires `condition_latents`."
            )
        resolved_condition_latents = None
        condition_source = "video_latents"
        context_condition_latents, context_condition_source_label = _resolve_full_condition_latents(
            video_latents,
            condition_latents,
            label="Parallel exact context-condition training",
        )
    else:
        context_condition_latents = None
        context_condition_source_label = None
        resolved_condition_latents, condition_source = _resolve_full_condition_latents(
            video_latents,
            condition_latents,
            label="Parallel exact training",
        )
    train_attn_mode = resolve_stage_attention_mode(backbone_config, stage="train", exact_runtime=True)
    # Exact parallel-stream training keeps video and action in the same frame
    # count. Actions are reshaped from `[B, F * A, D]` into
    # `[B, D, F, A, 1]` so the shared exact-runtime backbone can treat them
    # like a narrow latent volume with one "width" slot per action token.
    action_latents = rearrange(
        actions,
        "b (f a) c -> b c f a 1",
        f=num_frames,
        a=policy_config.action_per_frame,
    )
    action_mask_latents = None
    if action_mask is not None:
        action_mask_latents = rearrange(
            action_mask,
            "b (f a) c -> b c f a 1",
            f=num_frames,
            a=policy_config.action_per_frame,
        )

    latent_scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    latent_scheduler.set_timesteps(training_config.video_num_train_timesteps, training=True)
    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    action_scheduler.set_timesteps(training_config.action_num_train_timesteps, training=True)

    joint_timestep_coupling = resolve_parallel_joint_timestep_coupling(policy_config)
    shared_sigma_values: torch.Tensor | None = None
    latent_timestep_values: torch.Tensor | None = None
    action_timestep_values: torch.Tensor | None = None
    if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
        latent_timestep_values, action_timestep_values, shared_sigma_values = _sample_coupled_timestep_values(
            latent_scheduler=latent_scheduler,
            action_scheduler=action_scheduler,
            num_frames=num_frames,
            device=video_latents.device,
        )
    elif joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
        latent_timestep_values, action_timestep_values, shared_sigma_values = _sample_shared_video_schedule_timestep_values(
            latent_scheduler=latent_scheduler,
            num_frames=num_frames,
            device=video_latents.device,
        )
        _share_video_scheduler_grid_with_action_scheduler(
            latent_scheduler=latent_scheduler,
            action_scheduler=action_scheduler,
            device=video_latents.device,
        )
    elif joint_timestep_coupling == JointTimestepCoupling.MATCH_INDEX:
        latent_timestep_values, action_timestep_values = _sample_index_matched_timestep_values(
            latent_scheduler=latent_scheduler,
            action_scheduler=action_scheduler,
            num_frames=num_frames,
            device=video_latents.device,
        )

    # FDM/IDM-style objectives need clean condition streams to be marked as
    # clean-from-start, not "almost denoised" targets. Keep the legacy joint
    # policy augmentation by default, but allow objective-specific callers to
    # force zero condition timesteps for the video condition copy.
    latent_dict = _add_noise(
        video_latents,
        train_scheduler=latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0 if force_clean_video_condition else policy_config.noisy_video_condition_prob,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        condition_latent=resolved_condition_latents,
        frame_shift=frame_shift,
        timestep_values=latent_timestep_values,
        sigma_values=shared_sigma_values,
    )
    action_dict = _add_noise(
        action_latents,
        train_scheduler=action_scheduler,
        action_mask=action_mask_latents,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
        timestep_values=action_timestep_values,
        sigma_values=shared_sigma_values,
    )

    model_dtype = preferred_reference_dtype(video_latents.device)
    if text_emb is None:
        text_emb = torch.zeros(
            batch_size,
            backbone_config.max_text_tokens,
            backbone_config.text_dim,
            device=video_latents.device,
            dtype=model_dtype,
        )
    else:
        text_emb = text_emb.to(device=video_latents.device, dtype=model_dtype)

    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    action_dict["actions_mask"] = (
        action_mask_latents
        if action_mask_latents is not None
        else torch.ones_like(action_latents, device=video_latents.device)
    )
    def _resolve_frame_range(
        *,
        start: int | None,
        end: int | None,
        default_start: int | None = None,
        default_end: int | None = None,
        label: str,
    ) -> tuple[int, int]:
        start_value = default_start if start is None else start
        end_value = default_end if end is None else end
        resolved_start = 0 if start_value is None else int(start_value)
        resolved_end = num_frames if end_value is None else int(end_value)
        if resolved_start < 0 or resolved_end < resolved_start or resolved_end > num_frames:
            raise ValueError(
                f"Invalid {label} frame range for parallel exact training, "
                f"got start={resolved_start}, end={resolved_end}, num_frames={num_frames}."
            )
        return resolved_start, resolved_end

    resolved_loss_frame_start, resolved_loss_frame_end = _resolve_frame_range(
        start=loss_frame_start,
        end=loss_frame_end,
        label="current-loss",
    )
    resolved_latent_loss_frame_start, resolved_latent_loss_frame_end = _resolve_frame_range(
        start=latent_loss_frame_start,
        end=latent_loss_frame_end,
        default_start=loss_frame_start,
        default_end=loss_frame_end,
        label="latent-loss",
    )
    resolved_action_loss_frame_start, resolved_action_loss_frame_end = _resolve_frame_range(
        start=action_loss_frame_start,
        end=action_loss_frame_end,
        default_start=loss_frame_start,
        default_end=loss_frame_end,
        label="action-loss",
    )
    if context_condition_latents is not None:
        if resolved_loss_frame_start <= 0:
            raise ValueError(
                "`context_condition_latent_source=single_frame_condition_latent` requires at least one "
                "pre-target context frame; resolved loss_frame_start=0."
            )
        latent_dict["latent"][:, :, :resolved_loss_frame_start] = context_condition_latents[
            :, :, :resolved_loss_frame_start
        ]
        latent_dict["cond_timesteps"][:, :resolved_loss_frame_start] = 0
        condition_source = f"context_{context_condition_source_label}"
    latent_loss_mask = torch.zeros_like(video_latents, device=video_latents.device)
    latent_loss_mask[:, :, resolved_latent_loss_frame_start:resolved_latent_loss_frame_end] = 1.0
    action_loss_mask = torch.zeros_like(action_latents, device=video_latents.device)
    action_loss_mask[:, :, resolved_action_loss_frame_start:resolved_action_loss_frame_end] = 1.0
    latent_dict["loss_mask"] = latent_loss_mask
    action_dict["loss_mask"] = action_loss_mask

    # LingBot varies the effective chunk and window during training. Those
    # values are carried through as metadata because later layout/mask builders
    # need them to reproduce the same local-attention regime.
    if chunk_size_override is not None:
        sampled_chunk_size = max(1, int(chunk_size_override))
    else:
        chunk_size = max(1, int(training_config.chunk_size))
        sampled_chunk_size = int(torch.randint(1, chunk_size + 1, (1,), device=video_latents.device).item())
    if window_size_override is not None:
        sampled_window_size = max(1, int(window_size_override))
    elif training_config.window_size >= 4:
        sampled_window_size = int(
            torch.randint(4, int(training_config.window_size) + 1, (1,), device=video_latents.device).item()
        )
    else:
        sampled_window_size = max(1, int(training_config.window_size))
    attention_profile_name = None
    if train_attn_mode == "flex":
        attention_profile_name = _attention_profile_name_for_current_block_coupling(
            resolve_parallel_current_block_coupling(policy_config)
        )

    return LingbotParallelTrainArtifacts(
        input_dict={
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": sampled_chunk_size,
            "window_size": sampled_window_size,
            "loss_frame_start": resolved_loss_frame_start,
            "loss_frame_end": resolved_loss_frame_end,
            "latent_loss_frame_start": resolved_latent_loss_frame_start,
            "latent_loss_frame_end": resolved_latent_loss_frame_end,
            "action_loss_frame_start": resolved_action_loss_frame_start,
            "action_loss_frame_end": resolved_action_loss_frame_end,
            "frame_shift": int(frame_shift),
            "chunk_origin_frame": int(chunk_origin_frame),
            "singleton_chunk_frame": None if singleton_chunk_frame is None else int(singleton_chunk_frame),
            "conditional_history_policy": conditional_history_policy,
            "attention_profile_name": attention_profile_name,
            "preserve_video_pretrain_history": bool(
                getattr(policy_config, "preserve_video_pretrain_history", False)
            ),
            "history_stream_visibility": resolve_parallel_history_stream_visibility(policy_config).value,
            "force_clean_video_condition": bool(force_clean_video_condition),
            "joint_timestep_coupling": joint_timestep_coupling.value,
            "coupled_action_video_timesteps": bool(
                joint_timestep_coupling
                in {JointTimestepCoupling.MATCH_SIGMA, JointTimestepCoupling.SHARED_VIDEO_SCHEDULE}
            ),
            "video_condition_source": condition_source,
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )


def prepare_parallel_prefix_condition_exact_train_artifacts(
    *,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    condition_latents: torch.Tensor,
    chunk_size_override: int | None = None,
    window_size_override: int | None = None,
    frame_shift: int = 0,
    chunk_origin_frame: int = 0,
    singleton_chunk_frame: int | None = None,
    conditional_history_policy: str | None = None,
    generalist_training_mode_override: JointDenoiseTrainingMode | str | None = None,
    generalist_drop_text_conditioning: bool | None = None,
    generalist_training_source: str | None = None,
) -> LingbotParallelTrainArtifacts:
    """Build exact train artifacts with one clean single-frame video prefix."""

    if condition_latents.ndim != 5 or int(condition_latents.shape[2]) < 1:
        raise ValueError(
            "Prefix-condition exact training requires `condition_latents` with shape [B, C, F>=1, H, W], "
            f"got {tuple(condition_latents.shape)}."
        )
    if video_latents.shape[0] != condition_latents.shape[0] or video_latents.shape[1] != condition_latents.shape[1]:
        raise ValueError(
            "Prefix-condition exact training expects condition/video latent batch and channel dimensions to match, "
            f"video={tuple(video_latents.shape)}, condition={tuple(condition_latents.shape)}."
        )
    if video_latents.shape[-2:] != condition_latents.shape[-2:]:
        raise ValueError(
            "Prefix-condition exact training expects condition/video latent spatial dimensions to match, "
            f"video={tuple(video_latents.shape)}, condition={tuple(condition_latents.shape)}."
        )

    batch_size, _, target_frames, _, _ = video_latents.shape
    prefix_latent = condition_latents[:, :, :1].to(device=video_latents.device, dtype=video_latents.dtype)
    model_video_latents = torch.cat([prefix_latent, video_latents], dim=2)

    action_latents = rearrange(
        actions,
        "b (f a) c -> b c f a 1",
        f=target_frames,
        a=policy_config.action_per_frame,
    )
    action_mask_latents = None
    if action_mask is not None:
        action_mask_latents = rearrange(
            action_mask,
            "b (f a) c -> b c f a 1",
            f=target_frames,
            a=policy_config.action_per_frame,
        )

    latent_scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    latent_scheduler.set_timesteps(training_config.video_num_train_timesteps, training=True)
    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    action_scheduler.set_timesteps(training_config.action_num_train_timesteps, training=True)

    joint_timestep_coupling = resolve_parallel_joint_timestep_coupling(policy_config)
    target_timestep_plan = sample_joint_denoise_timestep_values(
        video_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
        num_frames=target_frames,
        device=video_latents.device,
        coupling=joint_timestep_coupling,
    )
    if joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
        _share_video_scheduler_grid_with_action_scheduler(
            latent_scheduler=latent_scheduler,
            action_scheduler=action_scheduler,
            device=video_latents.device,
        )
    video_timestep_values = torch.cat(
        [
            clean_timestep_values(num_frames=1, device=video_latents.device),
            target_timestep_plan.video_timesteps,
        ],
        dim=0,
    )
    video_sigma_values = None
    if target_timestep_plan.video_sigma_values is not None:
        video_sigma_values = torch.cat(
            [
                torch.zeros(1, device=video_latents.device, dtype=target_timestep_plan.video_sigma_values.dtype),
                target_timestep_plan.video_sigma_values,
            ],
            dim=0,
        )
    latent_dict = _add_noise(
        model_video_latents,
        train_scheduler=latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=policy_config.noisy_video_condition_prob,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift - 1,
        timestep_values=video_timestep_values,
        sigma_values=video_sigma_values,
    )
    latent_dict["noisy_latents"][:, :, :1] = prefix_latent
    latent_dict["latent"][:, :, :1] = prefix_latent
    latent_dict["targets"][:, :, :1] = 0
    latent_dict["timesteps"][:, :1] = 0
    latent_dict["cond_timesteps"][:, :1] = 0

    action_dict = _add_noise(
        action_latents,
        train_scheduler=action_scheduler,
        action_mask=action_mask_latents,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
        timestep_values=target_timestep_plan.action_timesteps,
        sigma_values=target_timestep_plan.action_sigma_values,
    )

    model_dtype = preferred_reference_dtype(video_latents.device)
    if text_emb is None:
        text_emb = torch.zeros(
            batch_size,
            backbone_config.max_text_tokens,
            backbone_config.text_dim,
            device=video_latents.device,
            dtype=model_dtype,
        )
    else:
        text_emb = text_emb.to(device=video_latents.device, dtype=model_dtype)
    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    action_dict["actions_mask"] = (
        action_mask_latents
        if action_mask_latents is not None
        else torch.ones_like(action_latents, device=video_latents.device)
    )
    latent_loss_mask = torch.ones_like(model_video_latents, device=video_latents.device)
    latent_loss_mask[:, :, :1] = 0
    action_loss_mask = torch.ones_like(action_latents, device=video_latents.device)
    latent_dict["loss_mask"] = latent_loss_mask
    action_dict["loss_mask"] = action_loss_mask

    if chunk_size_override is not None:
        sampled_chunk_size = max(1, int(chunk_size_override))
    else:
        chunk_size = max(1, int(training_config.chunk_size))
        sampled_chunk_size = int(torch.randint(1, chunk_size + 1, (1,), device=video_latents.device).item())
    if window_size_override is not None:
        sampled_window_size = max(1, int(window_size_override))
    elif training_config.window_size >= 4:
        sampled_window_size = int(
            torch.randint(4, int(training_config.window_size) + 1, (1,), device=video_latents.device).item()
        )
    else:
        sampled_window_size = max(1, int(training_config.window_size))
    train_attn_mode = resolve_stage_attention_mode(backbone_config, stage="train", exact_runtime=True)
    attention_profile_name = None
    if train_attn_mode == "flex":
        attention_profile_name = _attention_profile_name_for_current_block_coupling(
            resolve_parallel_current_block_coupling(policy_config)
        )

    artifacts = LingbotParallelTrainArtifacts(
        input_dict={
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": sampled_chunk_size,
            "window_size": sampled_window_size,
            "loss_frame_start": 0,
            "loss_frame_end": target_frames,
            "latent_loss_frame_start": 1,
            "latent_loss_frame_end": int(model_video_latents.shape[2]),
            "action_loss_frame_start": 0,
            "action_loss_frame_end": target_frames,
            "frame_shift": int(frame_shift),
            "chunk_origin_frame": int(chunk_origin_frame),
            "singleton_chunk_frame": None if singleton_chunk_frame is None else int(singleton_chunk_frame),
            "conditional_history_policy": conditional_history_policy,
            "attention_profile_name": attention_profile_name,
            "preserve_video_pretrain_history": bool(
                getattr(policy_config, "preserve_video_pretrain_history", False)
            ),
            "history_stream_visibility": resolve_parallel_history_stream_visibility(policy_config).value,
            "force_clean_video_condition": True,
            "joint_timestep_coupling": joint_timestep_coupling.value,
            "coupled_action_video_timesteps": bool(
                joint_timestep_coupling
                in {JointTimestepCoupling.MATCH_SIGMA, JointTimestepCoupling.SHARED_VIDEO_SCHEDULE}
            ),
            "video_condition_source": "condition_latents_prefix",
            "prefix_condition_frames": 1,
            "per_chunk_proprio_apply_to_video": False,
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )
    if policy_config.variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING:
        _apply_generalist_legacy_prefix_joint_training_mode(
            artifacts=artifacts,
            policy_config=policy_config,
            training_mode_override=generalist_training_mode_override,
            drop_text_conditioning=generalist_drop_text_conditioning,
            training_source=generalist_training_source,
        )
    return artifacts


def prepare_parallel_current_frame_action_chunk_train_artifacts(
    *,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    condition_latents: torch.Tensor | None = None,
    frame_shift: int = 0,
) -> LingbotParallelTrainArtifacts:
    batch_size, _, observed_frames, _, _ = video_latents.shape
    target_frames = int(policy_config.frame_chunk_size)
    if observed_frames < target_frames:
        raise ValueError(
            "Current-frame action-chunk training requires at least `policy_variant.frame_chunk_size` "
            f"latent frames, got observed_frames={observed_frames}, frame_chunk_size={target_frames}."
        )
    required_action_steps = target_frames * int(policy_config.action_per_frame)
    if actions.shape[1] < required_action_steps:
        raise ValueError(
            "Current-frame action-chunk training requires actions for one full generated chunk, "
            f"got action_horizon={actions.shape[1]}, required={required_action_steps}."
        )

    if bool(getattr(policy_config, "require_condition_latents", False)) and condition_latents is None:
        raise ValueError(
            "Current-frame action-chunk training was configured with `require_condition_latents=true`, "
            "but the latent batch did not provide `condition_latents`."
        )
    first_frame_condition_latents, condition_source = _select_first_frame_condition_latents(
        video_latents,
        condition_latents=condition_latents,
        label="Current-frame action-chunk",
    )
    condition_video_latents = first_frame_condition_latents.repeat(1, 1, target_frames, 1, 1)
    selected_actions = actions[:, :required_action_steps]
    action_latents = rearrange(
        selected_actions,
        "b (f a) c -> b c f a 1",
        f=target_frames,
        a=policy_config.action_per_frame,
    )
    action_mask_latents = None
    if action_mask is not None:
        action_mask_latents = rearrange(
            action_mask[:, :required_action_steps],
            "b (f a) c -> b c f a 1",
            f=target_frames,
            a=policy_config.action_per_frame,
        )

    latent_scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    latent_scheduler.set_timesteps(training_config.video_num_train_timesteps, training=True)
    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    action_scheduler.set_timesteps(training_config.action_num_train_timesteps, training=True)

    latent_dict = _add_noise(
        condition_video_latents,
        train_scheduler=latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
        timestep_values=clean_timestep_values(
            num_frames=target_frames,
            device=video_latents.device,
            dtype=latent_scheduler.timesteps.dtype,
        ),
    )
    force_clean_noisy_slot(latent_dict, condition_video_latents)
    action_dict = _add_noise(
        action_latents,
        train_scheduler=action_scheduler,
        action_mask=action_mask_latents,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
    )
    zero_condition_slot(action_dict)

    model_dtype = preferred_reference_dtype(video_latents.device)
    if text_emb is None:
        text_emb = torch.zeros(
            batch_size,
            backbone_config.max_text_tokens,
            backbone_config.text_dim,
            device=video_latents.device,
            dtype=model_dtype,
        )
    else:
        text_emb = text_emb.to(device=video_latents.device, dtype=model_dtype)

    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    action_dict["actions_mask"] = (
        action_mask_latents
        if action_mask_latents is not None
        else torch.ones_like(action_latents, device=video_latents.device)
    )
    latent_dict["loss_mask"] = torch.zeros_like(condition_video_latents, device=video_latents.device)
    action_dict["loss_mask"] = action_dict["actions_mask"].clone()

    return LingbotParallelTrainArtifacts(
        input_dict={
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": target_frames,
            "window_size": target_frames,
            "loss_frame_start": 0,
            "loss_frame_end": target_frames,
            "latent_loss_frame_start": 0,
            "latent_loss_frame_end": 0,
            "action_loss_frame_start": 0,
            "action_loss_frame_end": target_frames,
            "frame_shift": int(frame_shift),
            "attention_profile_name": "none",
            "preserve_video_pretrain_history": False,
            "force_clean_video_condition": True,
            "coupled_action_video_timesteps": False,
            "current_frame_action_chunk": True,
            "current_frame_condition_source": condition_source,
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )


def prepare_parallel_fastwam_first_frame_train_artifacts(
    *,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    condition_latents: torch.Tensor | None = None,
    frame_shift: int = 0,
) -> LingbotParallelTrainArtifacts:
    batch_size, _, num_frames, _, _ = video_latents.shape
    if num_frames <= 1:
        raise ValueError(
            "FastWAM first-frame training requires at least two latent frames "
            f"so future video loss can be supervised, got num_frames={num_frames}."
        )
    required_action_steps = num_frames * int(policy_config.action_per_frame)
    if actions.shape[1] < required_action_steps:
        raise ValueError(
            "FastWAM first-frame training requires actions for the full video window, "
            f"got action_horizon={actions.shape[1]}, required={required_action_steps}."
        )
    if bool(getattr(policy_config, "require_condition_latents", False)) and condition_latents is None:
        raise ValueError(
            "FastWAM first-frame training was configured with `require_condition_latents=true`, "
            "but the latent batch did not provide `condition_latents`."
        )
    first_frame_condition_latents, condition_source = _select_first_frame_condition_latents(
        video_latents,
        condition_latents=condition_latents,
        label="FastWAM",
    )

    selected_actions = actions[:, :required_action_steps]
    action_latents = rearrange(
        selected_actions,
        "b (f a) c -> b c f a 1",
        f=num_frames,
        a=policy_config.action_per_frame,
    )
    action_mask_latents = None
    if action_mask is not None:
        action_mask_latents = rearrange(
            action_mask[:, :required_action_steps],
            "b (f a) c -> b c f a 1",
            f=num_frames,
            a=policy_config.action_per_frame,
        )

    latent_scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    latent_scheduler.set_timesteps(training_config.video_num_train_timesteps, training=True)
    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    action_scheduler.set_timesteps(training_config.action_num_train_timesteps, training=True)

    latent_dict = _add_noise(
        video_latents,
        train_scheduler=latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
    )
    # Match FastWAM's fused first-frame condition: the first video latent is
    # clean, excluded from video loss, and cannot attend future video/action
    # tokens under the FastWAM mask.
    latent_dict["noisy_latents"][:, :, :1] = first_frame_condition_latents
    latent_dict["targets"][:, :, :1] = 0
    latent_dict["timesteps"][:, :1] = 0
    latent_dict["latent"] = torch.zeros_like(video_latents)
    latent_dict["cond_timesteps"] = torch.zeros_like(latent_dict["cond_timesteps"])

    action_dict = _add_noise(
        action_latents,
        train_scheduler=action_scheduler,
        action_mask=action_mask_latents,
        action_mode=True,
        noisy_cond_prob=0.0,
        patch_size=(backbone_config.patch_size_t, backbone_config.patch_size_h, backbone_config.patch_size_w),
        frame_shift=frame_shift,
    )
    zero_condition_slot(action_dict)

    model_dtype = preferred_reference_dtype(video_latents.device)
    if text_emb is None:
        text_emb = torch.zeros(
            batch_size,
            backbone_config.max_text_tokens,
            backbone_config.text_dim,
            device=video_latents.device,
            dtype=model_dtype,
        )
    else:
        text_emb = text_emb.to(device=video_latents.device, dtype=model_dtype)

    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    action_dict["actions_mask"] = (
        action_mask_latents
        if action_mask_latents is not None
        else torch.ones_like(action_latents, device=video_latents.device)
    )
    latent_loss_mask = torch.ones_like(video_latents, device=video_latents.device)
    latent_loss_mask[:, :, :1] = 0
    latent_dict["loss_mask"] = latent_loss_mask
    action_dict["loss_mask"] = action_dict["actions_mask"].clone()

    return LingbotParallelTrainArtifacts(
        input_dict={
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": num_frames,
            "window_size": num_frames,
            "loss_frame_start": 0,
            "loss_frame_end": num_frames,
            "latent_loss_frame_start": 1,
            "latent_loss_frame_end": num_frames,
            "action_loss_frame_start": 0,
            "action_loss_frame_end": num_frames,
            "frame_shift": int(frame_shift),
            "attention_profile_name": "fastwam_first_frame",
            "preserve_video_pretrain_history": False,
            "force_clean_video_condition": True,
            "coupled_action_video_timesteps": False,
            "fastwam_first_frame": True,
            "fastwam_condition_source": condition_source,
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )


def prepare_parallel_action_conditioned_train_artifacts(
    *,
    backbone_config: SharedVideoTransformerConfig,
    policy_config: ParallelStreamPolicyConfig,
    training_config: TrainingConfig,
    video_latents: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    text_emb: torch.Tensor | None,
    condition_latents: torch.Tensor | None = None,
    chunk_size_override: int | None = None,
    window_size_override: int | None = None,
    loss_frame_start: int | None = None,
    loss_frame_end: int | None = None,
    latent_loss_frame_start: int | None = None,
    latent_loss_frame_end: int | None = None,
    action_loss_frame_start: int | None = None,
    action_loss_frame_end: int | None = None,
    frame_shift: int = 0,
    chunk_origin_frame: int = 0,
    singleton_chunk_frame: int | None = None,
    conditional_history_policy: str | None = None,
    force_clean_video_condition: bool = False,
    generalist_training_mode_override: JointDenoiseTrainingMode | str | None = None,
    generalist_drop_text_conditioning: bool | None = None,
    generalist_training_source: str | None = None,
) -> LingbotParallelTrainArtifacts:
    coupling = resolve_parallel_current_block_coupling(policy_config)
    if (
        coupling
        in {
            CurrentBlockCoupling.JOINT,
            CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
            CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
        }
        and policy_config.current_block_coupling is None
        and not policy_config.video_condition_on_action
    ):
        raise ValueError(
            "`lingbot_exact_action_conditioned` requires `video_condition_on_action = true`."
        )
    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        text_emb=text_emb,
        condition_latents=condition_latents,
        chunk_size_override=chunk_size_override,
        window_size_override=window_size_override,
        loss_frame_start=loss_frame_start,
        loss_frame_end=loss_frame_end,
        latent_loss_frame_start=latent_loss_frame_start,
        latent_loss_frame_end=latent_loss_frame_end,
        action_loss_frame_start=action_loss_frame_start,
        action_loss_frame_end=action_loss_frame_end,
        frame_shift=frame_shift,
        chunk_origin_frame=chunk_origin_frame,
        singleton_chunk_frame=singleton_chunk_frame,
        conditional_history_policy=conditional_history_policy,
        force_clean_video_condition=force_clean_video_condition,
    )
    if policy_config.variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING:
        _, _, num_frames, _, _ = video_latents.shape
        action_latents = rearrange(
            actions,
            "b (f a) c -> b c f a 1",
            f=num_frames,
            a=policy_config.action_per_frame,
        )
        action_mask_latents = None
        if action_mask is not None:
            action_mask_latents = rearrange(
                action_mask,
                "b (f a) c -> b c f a 1",
                f=num_frames,
                a=policy_config.action_per_frame,
            )
        _apply_generalist_joint_denoise_training_mode(
            artifacts=artifacts,
            policy_config=policy_config,
            backbone_config=backbone_config,
            video_latents=video_latents,
            condition_latents=condition_latents,
            action_latents=action_latents,
            action_mask_latents=action_mask_latents,
            frame_shift=frame_shift,
            training_mode_override=generalist_training_mode_override,
            drop_text_conditioning=generalist_drop_text_conditioning,
            training_source=generalist_training_source,
        )
    return artifacts


__all__ = [
    "LingbotParallelTrainArtifacts",
    "ParallelTrainArtifacts",
    "prepare_parallel_action_conditioned_train_artifacts",
    "prepare_parallel_current_frame_action_chunk_train_artifacts",
    "prepare_parallel_exact_train_artifacts",
    "prepare_parallel_fastwam_first_frame_train_artifacts",
    "prepare_parallel_prefix_condition_exact_train_artifacts",
]
