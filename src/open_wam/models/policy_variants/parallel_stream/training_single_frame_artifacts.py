"""Single-frame parallel-stream training artifact assembly."""

from __future__ import annotations

import torch
from einops import rearrange

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.common.flow_noise_plan import clean_timestep_values
from open_wam.models.common.flow_schedule import FlowMatchScheduler
from open_wam.models.common.modality_slots import (
    force_clean_noisy_slot,
    zero_condition_slot,
)
from open_wam.models.visual_tower.reference_transformer import preferred_reference_dtype

from .latent_conditioning import (
    select_first_frame_condition_latents as _select_first_frame_condition_latents,
)
from .training_artifact_contracts import ParallelTrainArtifacts
from .training_noise import build_parallel_flow_noise_artifacts as _add_noise


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
) -> ParallelTrainArtifacts:
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

    if (
        bool(getattr(policy_config, "require_condition_latents", False))
        and condition_latents is None
    ):
        raise ValueError(
            "Current-frame action-chunk training was configured with `require_condition_latents=true`, "
            "but the latent batch did not provide `condition_latents`."
        )
    first_frame_condition_latents, condition_source = (
        _select_first_frame_condition_latents(
            video_latents,
            condition_latents=condition_latents,
            label="Current-frame action-chunk",
        )
    )
    condition_video_latents = first_frame_condition_latents.repeat(
        1, 1, target_frames, 1, 1
    )
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

    latent_dict = _add_noise(
        condition_video_latents,
        train_scheduler=latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
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
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
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
    latent_dict["loss_mask"] = torch.zeros_like(
        condition_video_latents, device=video_latents.device
    )
    action_dict["loss_mask"] = action_dict["actions_mask"].clone()

    return ParallelTrainArtifacts(
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
) -> ParallelTrainArtifacts:
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
    if (
        bool(getattr(policy_config, "require_condition_latents", False))
        and condition_latents is None
    ):
        raise ValueError(
            "FastWAM first-frame training was configured with `require_condition_latents=true`, "
            "but the latent batch did not provide `condition_latents`."
        )
    first_frame_condition_latents, condition_source = (
        _select_first_frame_condition_latents(
            video_latents,
            condition_latents=condition_latents,
            label="FastWAM",
        )
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

    latent_dict = _add_noise(
        video_latents,
        train_scheduler=latent_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
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
        patch_size=(
            backbone_config.patch_size_t,
            backbone_config.patch_size_h,
            backbone_config.patch_size_w,
        ),
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

    return ParallelTrainArtifacts(
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


__all__ = [
    "prepare_parallel_current_frame_action_chunk_train_artifacts",
    "prepare_parallel_fastwam_first_frame_train_artifacts",
]
