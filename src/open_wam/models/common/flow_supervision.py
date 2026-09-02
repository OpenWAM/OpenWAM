"""Flow reconstruction and supervised loss reduction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from open_wam.contracts import SampleConstructionMetadata

from .flow_schedule import FlowMatchScheduler


def build_video_frame_loss_mask(
    video_latents: torch.Tensor,
    *,
    sample_metadata: object = None,
    prefix_frame_count: int = 0,
    target_frame_count: int | None = None,
    default_target_start: int = 0,
    default_target_end: int | None = None,
    start_key: str = "latent_loss_frame_start",
    end_key: str = "latent_loss_frame_end",
    error_label: str = "video train loss-frame metadata",
) -> torch.Tensor:
    """Build a batch-aware frame mask in target-local coordinates.

    Dataset metadata describes the materialized target sequence. Policies may
    prepend external condition frames before that sequence, so the prefix is
    shifted structurally rather than folded into dataset frame coordinates.
    """

    if video_latents.ndim != 5:
        raise ValueError(
            "Video frame loss masks require [B, C, T, H, W] latents, "
            f"got {tuple(video_latents.shape)}."
        )
    batch_size = int(video_latents.shape[0])
    total_frames = int(video_latents.shape[2])
    prefix_frames = int(prefix_frame_count)
    resolved_target_frames = (
        total_frames - prefix_frames
        if target_frame_count is None
        else int(target_frame_count)
    )
    if prefix_frames < 0 or resolved_target_frames <= 0:
        raise ValueError(
            "Video frame loss masks require a non-negative prefix and positive "
            f"target length, got prefix={prefix_frames}, "
            f"target={resolved_target_frames}."
        )
    if prefix_frames + resolved_target_frames != total_frames:
        raise ValueError(
            "Video frame loss-mask geometry must cover the materialized tensor, "
            f"got prefix={prefix_frames}, target={resolved_target_frames}, "
            f"total={total_frames}."
        )

    metadata_items: tuple[Mapping[str, Any] | None, ...]
    if sample_metadata is None:
        metadata_items = (None,) * batch_size
    elif isinstance(sample_metadata, Mapping):
        if batch_size != 1:
            raise ValueError(
                "One video sample-metadata mapping is only valid for batch size 1; "
                f"got batch_size={batch_size}."
            )
        metadata_items = (sample_metadata,)
    elif isinstance(sample_metadata, (tuple, list)):
        if len(sample_metadata) != batch_size:
            raise ValueError(
                "Video sample metadata must contain one mapping per batch item, "
                f"got metadata={len(sample_metadata)}, batch={batch_size}."
            )
        if any(
            item is not None and not isinstance(item, Mapping)
            for item in sample_metadata
        ):
            raise TypeError(
                "Video sample metadata entries must be mappings or None."
            )
        metadata_items = tuple(sample_metadata)
    else:
        raise TypeError(
            "Video sample metadata must be a mapping, sequence of mappings, or None."
        )

    mask = video_latents.new_zeros(batch_size, 1, total_frames, 1, 1)
    for batch_index, raw_metadata in enumerate(metadata_items):
        # Use an empty typed view when metadata is absent so explicit ranges and
        # defaults obey exactly the same half-open validation contract.
        typed_metadata = SampleConstructionMetadata.from_mapping(
            {} if raw_metadata is None else raw_metadata
        )
        assert typed_metadata is not None
        target_start, target_end = typed_metadata.frame_range_or_default(
            observed_num_frames=resolved_target_frames,
            start_key=start_key,
            end_key=end_key,
            default_start=int(default_target_start),
            default_end=default_target_end,
            error_label=error_label,
        )
        mask[
            batch_index,
            :,
            prefix_frames + target_start : prefix_frames + target_end,
        ] = 1.0
    return mask


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
    "build_video_frame_loss_mask",
    "denoised_video_latents_from_flow",
    "denoised_actions_from_flow",
    "reduce_video_flow_match_loss",
    "reduce_frame_aligned_action_flow_match_loss",
    "reduce_slot_aligned_action_flow_match_loss",
]
