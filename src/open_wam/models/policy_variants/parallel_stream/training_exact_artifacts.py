"""Exact and action-conditioned parallel-stream training assembly."""

from __future__ import annotations

import torch
from einops import rearrange

from open_wam.configs.backbone import (
    SharedVideoTransformerConfig,
    resolve_stage_attention_mode,
)
from open_wam.configs.enums import (
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    GeneralistDenoisingMode,
    JointTimestepCoupling,
    ParallelStreamVariantProfile,
)
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common.flow_schedule import FlowMatchScheduler
from open_wam.models.visual_tower.reference_transformer import preferred_reference_dtype

from .generalist_training import (
    apply_generalist_joint_denoise_training_mode as _apply_generalist_joint_denoise_training_mode,
)
from .latent_conditioning import (
    resolve_full_window_condition_latents as _resolve_full_condition_latents,
)
from .runtime_semantics import (
    attention_profile_name_for_current_block_coupling as _attention_profile_name_for_current_block_coupling,
)
from .runtime_semantics import (
    resolve_parallel_context_condition_latent_source,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
)
from .training_artifact_contracts import ParallelTrainArtifacts
from .training_noise import (
    build_parallel_flow_noise_artifacts as _add_noise,
)
from .training_noise import (
    sample_coupled_parallel_timestep_values as _sample_coupled_timestep_values,
)
from .training_noise import (
    sample_index_matched_timestep_values as _sample_index_matched_timestep_values,
)
from .training_noise import (
    sample_shared_video_schedule_timestep_values as _sample_shared_video_schedule_timestep_values,
)
from .training_noise import (
    share_video_scheduler_grid_with_action_scheduler as _share_video_scheduler_grid_with_action_scheduler,
)


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
) -> ParallelTrainArtifacts:
    batch_size, _, num_frames, _, _ = video_latents.shape
    context_condition_source = resolve_parallel_context_condition_latent_source(
        policy_config
    )
    if (
        context_condition_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    ):
        if condition_latents is None:
            raise ValueError(
                "`context_condition_latent_source=single_frame_condition_latent` requires `condition_latents`."
            )
        resolved_condition_latents = None
        condition_source = "video_latents"
        context_condition_latents, context_condition_source_label = (
            _resolve_full_condition_latents(
                video_latents,
                condition_latents,
                label="Parallel exact context-condition training",
            )
        )
    else:
        context_condition_latents = None
        context_condition_source_label = None
        resolved_condition_latents, condition_source = _resolve_full_condition_latents(
            video_latents,
            condition_latents,
            label="Parallel exact training",
        )
    train_attn_mode = resolve_stage_attention_mode(
        backbone_config, stage="train", exact_runtime=True
    )
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
    latent_scheduler.set_timesteps(
        training_config.video_num_train_timesteps, training=True
    )
    action_scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    action_scheduler.set_timesteps(
        training_config.action_num_train_timesteps, training=True
    )

    joint_timestep_coupling = resolve_parallel_joint_timestep_coupling(policy_config)
    shared_sigma_values: torch.Tensor | None = None
    latent_timestep_values: torch.Tensor | None = None
    action_timestep_values: torch.Tensor | None = None
    if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
        latent_timestep_values, action_timestep_values, shared_sigma_values = (
            _sample_coupled_timestep_values(
                latent_scheduler=latent_scheduler,
                action_scheduler=action_scheduler,
                num_frames=num_frames,
                device=video_latents.device,
            )
        )
    elif joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
        latent_timestep_values, action_timestep_values, shared_sigma_values = (
            _sample_shared_video_schedule_timestep_values(
                latent_scheduler=latent_scheduler,
                num_frames=num_frames,
                device=video_latents.device,
            )
        )
        _share_video_scheduler_grid_with_action_scheduler(
            latent_scheduler=latent_scheduler,
            action_scheduler=action_scheduler,
            device=video_latents.device,
        )
    elif joint_timestep_coupling == JointTimestepCoupling.MATCH_INDEX:
        latent_timestep_values, action_timestep_values = (
            _sample_index_matched_timestep_values(
                latent_scheduler=latent_scheduler,
                action_scheduler=action_scheduler,
                num_frames=num_frames,
                device=video_latents.device,
            )
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
        noisy_cond_prob=0.0
        if force_clean_video_condition
        else policy_config.noisy_video_condition_prob,
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
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
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
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
        if (
            resolved_start < 0
            or resolved_end < resolved_start
            or resolved_end > num_frames
        ):
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
    resolved_latent_loss_frame_start, resolved_latent_loss_frame_end = (
        _resolve_frame_range(
            start=latent_loss_frame_start,
            end=latent_loss_frame_end,
            default_start=loss_frame_start,
            default_end=loss_frame_end,
            label="latent-loss",
        )
    )
    resolved_action_loss_frame_start, resolved_action_loss_frame_end = (
        _resolve_frame_range(
            start=action_loss_frame_start,
            end=action_loss_frame_end,
            default_start=loss_frame_start,
            default_end=loss_frame_end,
            label="action-loss",
        )
    )
    if context_condition_latents is not None:
        if resolved_loss_frame_start <= 0:
            raise ValueError(
                "`context_condition_latent_source=single_frame_condition_latent` requires at least one "
                "pre-target context frame; resolved loss_frame_start=0."
            )
        latent_dict["latent"][:, :, :resolved_loss_frame_start] = (
            context_condition_latents[:, :, :resolved_loss_frame_start]
        )
        latent_dict["cond_timesteps"][:, :resolved_loss_frame_start] = 0
        condition_source = f"context_{context_condition_source_label}"
    latent_loss_mask = torch.zeros_like(video_latents, device=video_latents.device)
    latent_loss_mask[
        :, :, resolved_latent_loss_frame_start:resolved_latent_loss_frame_end
    ] = 1.0
    action_loss_mask = torch.zeros_like(action_latents, device=video_latents.device)
    action_loss_mask[
        :, :, resolved_action_loss_frame_start:resolved_action_loss_frame_end
    ] = 1.0
    latent_dict["loss_mask"] = latent_loss_mask
    action_dict["loss_mask"] = action_loss_mask

    # LingBot varies the effective chunk and window during training. Those
    # values are carried through as metadata because later layout/mask builders
    # need them to reproduce the same local-attention regime.
    if chunk_size_override is not None:
        sampled_chunk_size = max(1, int(chunk_size_override))
    else:
        chunk_size = max(1, int(training_config.chunk_size))
        sampled_chunk_size = int(
            torch.randint(1, chunk_size + 1, (1,), device=video_latents.device).item()
        )
    if window_size_override is not None:
        sampled_window_size = max(1, int(window_size_override))
    elif training_config.window_size >= 4:
        sampled_window_size = int(
            torch.randint(
                4,
                int(training_config.window_size) + 1,
                (1,),
                device=video_latents.device,
            ).item()
        )
    else:
        sampled_window_size = max(1, int(training_config.window_size))
    attention_profile_name = None
    if train_attn_mode == "flex":
        attention_profile_name = _attention_profile_name_for_current_block_coupling(
            resolve_parallel_current_block_coupling(policy_config)
        )

    return ParallelTrainArtifacts(
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
            "singleton_chunk_frame": None
            if singleton_chunk_frame is None
            else int(singleton_chunk_frame),
            "conditional_history_policy": conditional_history_policy,
            "attention_profile_name": attention_profile_name,
            "preserve_video_pretrain_history": bool(
                getattr(policy_config, "preserve_video_pretrain_history", False)
            ),
            "history_stream_visibility": resolve_parallel_history_stream_visibility(
                policy_config
            ).value,
            "force_clean_video_condition": bool(force_clean_video_condition),
            "joint_timestep_coupling": joint_timestep_coupling.value,
            "coupled_action_video_timesteps": bool(
                joint_timestep_coupling
                in {
                    JointTimestepCoupling.MATCH_SIGMA,
                    JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
                }
            ),
            "video_condition_source": condition_source,
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
    generalist_training_mode_override: GeneralistDenoisingMode | str | None = None,
    generalist_drop_text_conditioning: bool | None = None,
    generalist_training_source: str | None = None,
) -> ParallelTrainArtifacts:
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
    if (
        policy_config.variant_profile
        == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
    ):
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
    "prepare_parallel_action_conditioned_train_artifacts",
    "prepare_parallel_exact_train_artifacts",
]
