from __future__ import annotations

import torch

from open_wam.models.common.flow_matching import FlowMatchScheduler, sample_timestep_id
from open_wam.models.common.flow_noise_plan import (
    sample_coupled_timestep_values as sample_shared_coupled_timestep_values,
)
from open_wam.models.visual_tower.exact_runtime import build_reference_mesh_id

__all__ = [
    "build_parallel_flow_noise_artifacts",
    "sample_coupled_parallel_timestep_values",
    "sample_index_matched_timestep_values",
    "sample_shared_video_schedule_timestep_values",
    "share_video_scheduler_grid_with_action_scheduler",
]


def build_parallel_flow_noise_artifacts(
    latent: torch.Tensor,
    *,
    train_scheduler: FlowMatchScheduler,
    action_mask: torch.Tensor | None,
    action_mode: bool,
    noisy_cond_prob: float,
    patch_size: tuple[int, int, int],
    condition_latent: torch.Tensor | None = None,
    frame_shift: int = 0,
    timestep_values: torch.Tensor | None = None,
    sigma_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build one exact-stream flow target without executing the transformer."""

    batch_size, _, num_frames, _, _ = latent.shape
    # LingBot samples one timestep per frame, then broadcasts that scalar across
    # every channel/spatial location inside that frame. For video latents the
    # tensor is `[B, C_latent, F, H_latent, W_latent]`; for action latents it is
    # `[B, D_action, F, action_per_frame, 1]`.
    noise = torch.zeros_like(latent).normal_()
    scheduler_timesteps = train_scheduler.timesteps.to(device=latent.device)
    if timestep_values is None:
        timestep_ids = sample_timestep_id(
            batch_size=num_frames,
            num_train_timesteps=train_scheduler.num_train_timesteps,
            device=latent.device,
        )
        timesteps = scheduler_timesteps[timestep_ids]
    else:
        timesteps = timestep_values.to(
            device=latent.device,
            dtype=scheduler_timesteps.dtype,
        )
        if timesteps.ndim != 1 or timesteps.shape[0] != num_frames:
            raise ValueError(
                "Explicit denoise timestep values must be one scalar per frame, "
                f"got shape={tuple(timesteps.shape)} and num_frames={num_frames}."
            )
    if sigma_values is None:
        noisy_latents = train_scheduler.add_noise(
            latent,
            noise,
            timesteps,
            t_dim=2,
        )
    else:
        sigmas = sigma_values.to(device=latent.device, dtype=latent.dtype)
        if sigmas.ndim != 1 or sigmas.shape[0] != num_frames:
            raise ValueError(
                "Explicit denoise sigma values must be one scalar per frame, "
                f"got shape={tuple(sigmas.shape)} and num_frames={num_frames}."
            )
        shape = [1] * noise.ndim
        shape[2] = num_frames
        sigmas = sigmas.view(shape)
        noisy_latents = (1 - sigmas) * latent + sigmas * noise
    targets = train_scheduler.training_target(latent, noise, timesteps)

    patch_f, patch_h, patch_w = patch_size
    if action_mode:
        patch_f = patch_h = patch_w = 1

    # Grid ids stay flattened to match the shared exact-runtime backbone input
    # after patchification:
    # - video: `[B, 4, T_video]` where `T_video = F/p_t * H/p_h * W/p_w`
    # - action: `[B, 4, T_action]` where `T_action = F * action_per_frame`
    latent_grid_id = build_reference_mesh_id(
        latent.shape[-3] // patch_f,
        latent.shape[-2] // patch_h,
        latent.shape[-1] // patch_w,
        t=1 if action_mode else 0,
        f_w=1,
        f_shift=frame_shift,
        action=action_mode,
        device=latent.device,
    )[None].repeat(batch_size, 1, 1)

    condition_source = (
        latent
        if condition_latent is None
        else condition_latent.to(device=latent.device, dtype=latent.dtype)
    )
    if tuple(condition_source.shape) != tuple(latent.shape):
        raise ValueError(
            "Condition latent shape must match the denoising target latent shape, "
            f"got condition={tuple(condition_source.shape)}, target={tuple(latent.shape)}."
        )

    if (
        noisy_cond_prob > 0.0
        and torch.rand(1, device=latent.device).item() < noisy_cond_prob
    ):
        cond_timestep_ids = sample_timestep_id(
            batch_size=num_frames,
            min_timestep_bd=0.5,
            max_timestep_bd=1.0,
            num_train_timesteps=train_scheduler.num_train_timesteps,
            device=latent.device,
        )
        cond_noise = torch.zeros_like(latent).normal_()
        cond_timesteps = scheduler_timesteps[cond_timestep_ids]
        condition_source = train_scheduler.add_noise(
            condition_source,
            cond_noise,
            cond_timesteps,
            t_dim=2,
        )
    else:
        cond_timesteps = torch.zeros_like(timesteps)

    if action_mask is not None:
        noisy_latents = noisy_latents * action_mask.float()
        targets = targets * action_mask.float()
        condition_source = condition_source * action_mask.float()

    return {
        "timesteps": timesteps[None].repeat(batch_size, 1),
        "noisy_latents": noisy_latents,
        "targets": targets,
        "latent": condition_source,
        "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
        "grid_id": latent_grid_id,
    }


def sample_coupled_parallel_timestep_values(
    *,
    latent_scheduler: FlowMatchScheduler,
    action_scheduler: FlowMatchScheduler,
    num_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adapt the shared coupled plan to the parallel runtime's tuple contract."""

    values = sample_shared_coupled_timestep_values(
        video_scheduler=latent_scheduler,
        action_scheduler=action_scheduler,
        num_frames=num_frames,
        device=device,
    )
    return values.video_timesteps, values.action_timesteps, values.sigma_values


def share_video_scheduler_grid_with_action_scheduler(
    *,
    latent_scheduler: FlowMatchScheduler,
    action_scheduler: FlowMatchScheduler,
    device: torch.device,
) -> None:
    """Use the video scheduler grid and weights for both exact streams."""

    action_scheduler.timesteps = latent_scheduler.timesteps.to(device=device)
    action_scheduler.sigmas = latent_scheduler.sigmas.to(device=device)
    if hasattr(latent_scheduler, "linear_timesteps_weights"):
        action_scheduler.linear_timesteps_weights = (
            latent_scheduler.linear_timesteps_weights.to(device=device)
        )


def sample_shared_video_schedule_timestep_values(
    *,
    latent_scheduler: FlowMatchScheduler,
    num_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one video-scheduler index and reuse it for both streams."""

    timestep_ids = sample_timestep_id(
        batch_size=num_frames,
        num_train_timesteps=int(latent_scheduler.timesteps.numel()),
        device=device,
    )
    video_timesteps = latent_scheduler.timesteps.to(device=device)[timestep_ids]
    sigma_values = latent_scheduler.sigmas.to(device=device)[timestep_ids]
    return video_timesteps, video_timesteps, sigma_values


def sample_index_matched_timestep_values(
    *,
    latent_scheduler: FlowMatchScheduler,
    action_scheduler: FlowMatchScheduler,
    num_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one shared scheduler index per frame for action and video."""

    if int(latent_scheduler.timesteps.numel()) != int(
        action_scheduler.timesteps.numel()
    ):
        raise ValueError(
            "Index-matched joint denoising requires equal video/action train timestep grid lengths, "
            f"got video={int(latent_scheduler.timesteps.numel())}, "
            f"action={int(action_scheduler.timesteps.numel())}."
        )
    timestep_ids = sample_timestep_id(
        batch_size=num_frames,
        num_train_timesteps=int(latent_scheduler.timesteps.numel()),
        device=device,
    )
    return (
        latent_scheduler.timesteps.to(device=device)[timestep_ids],
        action_scheduler.timesteps.to(device=device)[timestep_ids],
    )
