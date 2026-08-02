"""Flow reconstruction and supervised loss reduction."""

from __future__ import annotations

import torch

from .flow_schedule import FlowMatchScheduler


def denoised_video_latents_from_flow(
    *,
    noisy_latents: torch.Tensor,
    flow_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
) -> torch.Tensor:
    sigma = scheduler.sigma_for_timesteps(timesteps.flatten()).reshape(timesteps.shape)
    return noisy_latents - sigma[:, None, :, None, None].to(noisy_latents.dtype) * flow_pred


def denoised_actions_from_flow(
    *,
    noisy_actions: torch.Tensor,
    flow_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
) -> torch.Tensor:
    sigma = scheduler.sigma_for_timesteps(timesteps.flatten()).reshape(timesteps.shape)
    return noisy_actions - sigma[:, :, None].to(noisy_actions.dtype) * flow_pred


def reduce_video_flow_match_loss(
    *,
    flow_pred: torch.Tensor,
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
) -> torch.Tensor:
    """Reduce `[B, C_latent, F, H, W]` video diffusion loss frame-wise."""

    per_token_loss = torch.nn.functional.mse_loss(flow_pred.float(), targets.float().detach(), reduction="none")
    timestep_weight = scheduler.training_weight(timesteps.flatten()).reshape(timesteps.shape)
    per_token_loss = per_token_loss * timestep_weight[:, None, :, None, None]
    per_frame_loss = per_token_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    frame_loss_sum = per_frame_loss.sum(dim=1)
    frame_denom = torch.ones_like(per_frame_loss).sum(dim=1)
    return (frame_loss_sum / (frame_denom + 1e-6)).mean()


def reduce_frame_aligned_action_flow_match_loss(
    *,
    flow_pred: torch.Tensor,
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
    action_mask: torch.Tensor | None,
    num_frames: int,
    action_per_frame: int,
) -> torch.Tensor:
    """Reduce `[B, H_action, D_action]` action diffusion loss frame-wise."""

    batch_size, action_horizon, action_dim = flow_pred.shape
    expected_horizon = num_frames * action_per_frame
    if action_horizon != expected_horizon:
        raise ValueError(
            f"Expected frame-aligned action horizon {expected_horizon}, got {action_horizon}."
        )
    per_token_loss = torch.nn.functional.mse_loss(flow_pred.float(), targets.float().detach(), reduction="none")
    per_token_loss = per_token_loss.view(batch_size, num_frames, action_per_frame, action_dim)
    timestep_weight = scheduler.training_weight(timesteps.flatten()).reshape(timesteps.shape)
    per_token_loss = per_token_loss * timestep_weight[:, :, None, None]
    if action_mask is not None:
        mask = action_mask.float().view(batch_size, num_frames, action_per_frame, action_dim)
        per_token_loss = per_token_loss * mask
        frame_denom = mask.sum(dim=(2, 3)).clamp_min(1.0)
    else:
        frame_denom = torch.full(
            (batch_size, num_frames),
            fill_value=float(action_per_frame * action_dim),
            device=per_token_loss.device,
        )
    frame_loss = per_token_loss.sum(dim=(2, 3)) / frame_denom
    return frame_loss.mean()


def reduce_slot_aligned_action_flow_match_loss(
    *,
    flow_pred: torch.Tensor,
    targets: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
    action_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reduce `[B, H_action, D_action]` action diffusion loss slot-wise."""

    per_token_loss = torch.nn.functional.mse_loss(flow_pred.float(), targets.float().detach(), reduction="none")
    timestep_weight = scheduler.training_weight(timesteps.flatten()).reshape(timesteps.shape)
    per_token_loss = per_token_loss * timestep_weight[:, :, None]
    if action_mask is not None:
        per_token_loss = per_token_loss * action_mask.float()
        denom = action_mask.float().sum(dim=-1).clamp_min(1.0)
    else:
        denom = torch.full(
            timesteps.shape,
            fill_value=float(flow_pred.shape[-1]),
            device=per_token_loss.device,
        )
    per_slot_loss = per_token_loss.sum(dim=-1) / denom
    return per_slot_loss.mean()


__all__ = [
    "denoised_video_latents_from_flow",
    "denoised_actions_from_flow",
    "reduce_video_flow_match_loss",
    "reduce_frame_aligned_action_flow_match_loss",
    "reduce_slot_aligned_action_flow_match_loss",
]
