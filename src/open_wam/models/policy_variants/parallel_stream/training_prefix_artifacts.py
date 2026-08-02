"""Single-prefix parallel-stream training artifact assembly."""

from __future__ import annotations

import torch
from einops import rearrange

from open_wam.configs.backbone import (
    SharedVideoTransformerConfig,
    resolve_stage_attention_mode,
)
from open_wam.configs.enums import (
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    ParallelStreamVariantProfile,
)
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common.flow_schedule import FlowMatchScheduler
from open_wam.models.common.flow_noise_plan import (
    clean_timestep_values,
    sample_joint_denoise_timestep_values,
)
from open_wam.models.visual_tower.reference_transformer import preferred_reference_dtype

from .generalist_training import (
    apply_generalist_legacy_prefix_joint_training_mode as _apply_generalist_legacy_prefix_joint_training_mode,
)
from .runtime_semantics import (
    attention_profile_name_for_current_block_coupling as _attention_profile_name_for_current_block_coupling,
    resolve_parallel_current_block_coupling,
    resolve_parallel_history_stream_visibility,
    resolve_parallel_joint_timestep_coupling,
)
from .training_artifact_contracts import LingbotParallelTrainArtifacts
from .training_noise import (
    build_parallel_flow_noise_artifacts as _add_noise,
    share_video_scheduler_grid_with_action_scheduler as _share_video_scheduler_grid_with_action_scheduler,
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
    if (
        video_latents.shape[0] != condition_latents.shape[0]
        or video_latents.shape[1] != condition_latents.shape[1]
    ):
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
    prefix_latent = condition_latents[:, :, :1].to(
        device=video_latents.device, dtype=video_latents.dtype
    )
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
                torch.zeros(
                    1,
                    device=video_latents.device,
                    dtype=target_timestep_plan.video_sigma_values.dtype,
                ),
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
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
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
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
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
    train_attn_mode = resolve_stage_attention_mode(
        backbone_config, stage="train", exact_runtime=True
    )
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
            "force_clean_video_condition": True,
            "joint_timestep_coupling": joint_timestep_coupling.value,
            "coupled_action_video_timesteps": bool(
                joint_timestep_coupling
                in {
                    JointTimestepCoupling.MATCH_SIGMA,
                    JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
                }
            ),
            "video_condition_source": "condition_latents_prefix",
            "prefix_condition_frames": 1,
            "per_chunk_proprio_apply_to_video": False,
        },
        latent_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
    )
    if (
        policy_config.variant_profile
        == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
    ):
        _apply_generalist_legacy_prefix_joint_training_mode(
            artifacts=artifacts,
            policy_config=policy_config,
            training_mode_override=generalist_training_mode_override,
            drop_text_conditioning=generalist_drop_text_conditioning,
            training_source=generalist_training_source,
        )
    return artifacts


__all__ = ["prepare_parallel_prefix_condition_exact_train_artifacts"]
