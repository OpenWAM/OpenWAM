"""Train-time flow-matching artifact contracts and builders."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import TrainingConfig

from .flow_schedule import (
    FlowMatchScheduler,
    sample_timestep_id,
    timesteps_matching_sigmas,
)


@dataclass
class ActionFlowMatchTrainArtifacts:
    """Train-time noisy action pack used by diffusion decoders and variants.

    Shapes:
    - `timesteps`: `[B, H_action]`
    - `noisy_actions`: `[B, H_action, D_action]`
    - `targets`: `[B, H_action, D_action]`
    - `action_mask`: optional `[B, H_action, D_action]`
    """

    timesteps: torch.Tensor
    noisy_actions: torch.Tensor
    targets: torch.Tensor
    action_mask: torch.Tensor | None
    scheduler: FlowMatchScheduler


@dataclass
class VideoFlowMatchTrainArtifacts:
    """Train-time noisy video pack for `[B, C_latent, F, H, W]` tensors.

    Shapes:
    - `timesteps`: `[B, F]` (V_noisy copy per-frame timesteps)
    - `noisy_latents`: `[B, C_latent, F, H, W]`
    - `targets`: `[B, C_latent, F, H, W]`
    - `condition_latents`: `[B, C_latent, F, H, W]` (V_clean copy; equals
      GT when no augmentation, slightly noised when `noisy_condition_prob`
      augmentation fires)
    - `condition_timesteps`: `[B, F]` (per-frame timesteps matching
      `condition_latents`; zeros when clean, sampled from the top half of
      the schedule when augmentation fires)
    """

    timesteps: torch.Tensor
    noisy_latents: torch.Tensor
    targets: torch.Tensor
    condition_latents: torch.Tensor
    condition_timesteps: torch.Tensor
    scheduler: FlowMatchScheduler


@dataclass
class FrameAlignedActionFlowMatchTrainArtifacts:
    """Frame-granular noisy actions for LingBot-style parallel-stream training.

    Shapes:
    - `frame_timesteps`: `[B, F]`
    - `slot_timesteps`: `[B, H_action]`
    - `noisy_actions`: `[B, H_action, D_action]`
    - `targets`: `[B, H_action, D_action]`
    - `condition_actions`: `[B, H_action, D_action]`
    - `action_mask`: optional `[B, H_action, D_action]`
    """

    frame_timesteps: torch.Tensor
    slot_timesteps: torch.Tensor
    noisy_actions: torch.Tensor
    targets: torch.Tensor
    condition_actions: torch.Tensor
    action_mask: torch.Tensor | None
    scheduler: FlowMatchScheduler


@dataclass
class BlockCoupledActionFlowMatchTrainArtifacts:
    """DreamZero-style action artifacts coupled to future video block timesteps.

    Shapes:
    - `block_timesteps`: `[B, num_blocks]`
    - `timesteps`: `[B, H_action]`
    - `noisy_actions`: `[B, H_action, D_action]`
    - `targets`: `[B, H_action, D_action]`
    """

    block_timesteps: torch.Tensor
    timesteps: torch.Tensor
    noisy_actions: torch.Tensor
    targets: torch.Tensor
    action_mask: torch.Tensor | None
    scheduler: FlowMatchScheduler


def build_action_flow_match_train_artifacts(
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    training_config: TrainingConfig,
) -> ActionFlowMatchTrainArtifacts:
    """Create LingBot-style noisy actions for `[B, H_action, D_action]` tensors.

    We intentionally sample one timestep per horizon slot and broadcast that
    timestep across the batch. This mirrors LingBot's "one timestep per frame"
    behavior for action latents.
    """

    _, action_horizon, _ = actions.shape
    scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    scheduler.set_timesteps(training_config.action_num_train_timesteps, training=True)
    timestep_ids = sample_timestep_id(
        batch_size=action_horizon,
        num_train_timesteps=training_config.action_num_train_timesteps,
        device=actions.device,
    )
    timesteps = scheduler.timesteps.to(device=actions.device)[timestep_ids]
    noise = torch.randn_like(actions)
    noisy_actions = scheduler.add_noise(actions, noise, timesteps, t_dim=1)
    targets = scheduler.training_target(actions, noise, timesteps)
    if action_mask is not None:
        noisy_actions = noisy_actions * action_mask.float()
        targets = targets * action_mask.float()
    return ActionFlowMatchTrainArtifacts(
        timesteps=timesteps[None].repeat(actions.shape[0], 1),
        noisy_actions=noisy_actions,
        targets=targets,
        action_mask=action_mask,
        scheduler=scheduler,
    )


def build_video_flow_match_train_artifacts(
    video_latents: torch.Tensor,
    *,
    training_config: TrainingConfig,
    noisy_condition_prob: float = 0.0,
    condition_latents: torch.Tensor | None = None,
    timestep_ids: torch.Tensor | None = None,
    clean_prefix_frames: int = 0,
    synchronize_noisy_condition_decision: bool = True,
) -> VideoFlowMatchTrainArtifacts:
    """Create LingBot-style noisy video latents with one timestep per frame.

    The sampled timestep is broadcast across channels and spatial positions of
    each frame, matching LingBot's frame-wise latent diffusion semantics. When
    ``clean_prefix_frames`` is nonzero, those frames remain clean in both video
    streams and are excluded from flow supervision. By default the condition
    augmentation decision is broadcast from rank 0 when distributed training is
    initialized. Sequence-batch callers disable that synchronization because
    their per-rank sample counts may differ within one model forward.
    """

    if video_latents.ndim != 5:
        raise ValueError(
            "Expected video latents with shape [B, C_latent, F, H, W], "
            f"got {tuple(video_latents.shape)}."
        )
    _, _, num_frames, _, _ = video_latents.shape
    scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    scheduler.set_timesteps(training_config.video_num_train_timesteps, training=True)
    batch_size = video_latents.shape[0]
    clean_prefix_frames = int(clean_prefix_frames)
    if clean_prefix_frames < 0 or clean_prefix_frames >= num_frames:
        if clean_prefix_frames != 0:
            raise ValueError(
                "Video flow-match clean_prefix_frames must leave at least one "
                f"denoising target, got prefix={clean_prefix_frames}, frames={num_frames}."
            )
    if timestep_ids is None:
        timestep_ids = sample_timestep_id(
            batch_size=batch_size,
            sample_shape=(num_frames,),
            num_train_timesteps=training_config.video_num_train_timesteps,
            device=video_latents.device,
        )
    else:
        if tuple(timestep_ids.shape) != (batch_size, num_frames):
            raise ValueError(
                "Video flow-match timestep_ids must have shape [B, F], "
                f"got {tuple(timestep_ids.shape)}, expected={(batch_size, num_frames)}."
            )
        timestep_ids = timestep_ids.to(device=video_latents.device, dtype=torch.int64)
    timesteps = scheduler.timesteps.to(device=video_latents.device)[timestep_ids]
    noise = torch.randn_like(video_latents)
    noisy_latents = scheduler.add_noise(video_latents, noise, timesteps, t_dim=2)
    targets = scheduler.training_target(video_latents, noise, timesteps)
    clean_condition_latents = video_latents
    if condition_latents is not None:
        if condition_latents.ndim != 5:
            raise ValueError(
                "Video condition_latents must have shape [B, C_latent, F, H, W], "
                f"got {tuple(condition_latents.shape)}."
            )
        if tuple(condition_latents.shape) != tuple(video_latents.shape):
            raise ValueError(
                "Video condition_latents must match video_latents exactly, "
                f"got condition={tuple(condition_latents.shape)}, "
                f"video={tuple(video_latents.shape)}."
            )
        clean_condition_latents = condition_latents.to(
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
    original_condition_latents = clean_condition_latents
    condition_timesteps = torch.zeros_like(timesteps)
    if noisy_condition_prob > 0.0:
        # Preserve the legacy rank-0 decision by default. Batched preparation
        # uses local decisions: a per-sample collective would mismatch when
        # ranks prepare different sample counts before the same FSDP forward.
        decision = torch.rand(1, device=video_latents.device)
        if (
            synchronize_noisy_condition_decision
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.broadcast(decision, src=0)
        if decision.item() < noisy_condition_prob:
            condition_timestep_ids = sample_timestep_id(
                batch_size=batch_size,
                sample_shape=(num_frames,),
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=training_config.video_num_train_timesteps,
                device=video_latents.device,
            )
            condition_timesteps = scheduler.timesteps.to(device=video_latents.device)[
                condition_timestep_ids
            ]
            condition_noise = torch.randn_like(video_latents)
            clean_condition_latents = scheduler.add_noise(
                clean_condition_latents,
                condition_noise,
                condition_timesteps,
                t_dim=2,
            )
    if clean_prefix_frames > 0:
        prefix = slice(0, clean_prefix_frames)
        noisy_latents[:, :, prefix] = video_latents[:, :, prefix]
        targets[:, :, prefix] = 0
        timesteps[:, prefix] = 0
        if clean_condition_latents is not original_condition_latents:
            clean_condition_latents[:, :, prefix] = original_condition_latents[
                :, :, prefix
            ]
        condition_timesteps[:, prefix] = 0
    return VideoFlowMatchTrainArtifacts(
        timesteps=timesteps,
        noisy_latents=noisy_latents,
        targets=targets,
        condition_latents=clean_condition_latents,
        condition_timesteps=condition_timesteps,
        scheduler=scheduler,
    )


def build_frame_aligned_action_flow_match_train_artifacts(
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    training_config: TrainingConfig,
    num_frames: int,
    action_per_frame: int,
    frame_sigma_values: torch.Tensor | None = None,
    frame_timestep_ids: torch.Tensor | None = None,
    scheduler_override: FlowMatchScheduler | None = None,
) -> FrameAlignedActionFlowMatchTrainArtifacts:
    """Create frame-granular noisy actions for LingBot-style parallel-stream.

    Unlike the generic action helper, timesteps are sampled per frame and then
    broadcast across all `action_per_frame * D_action` values aligned to that
    frame. This mirrors LingBot's action-latent supervision.
    """

    if actions.ndim != 3:
        raise ValueError(
            "Expected actions with shape [B, H_action, D_action], "
            f"got {tuple(actions.shape)}."
        )
    batch_size, action_horizon, action_dim = actions.shape
    expected_horizon = num_frames * action_per_frame
    if action_horizon != expected_horizon:
        raise ValueError(
            "Frame-aligned action diffusion expects `action_horizon == num_frames * action_per_frame`, "
            f"got action_horizon={action_horizon}, num_frames={num_frames}, action_per_frame={action_per_frame}."
        )
    if scheduler_override is None:
        scheduler = FlowMatchScheduler(
            shift=training_config.action_sigma_shift,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=training_config.action_num_train_timesteps,
        )
        scheduler.set_timesteps(training_config.action_num_train_timesteps, training=True)
    else:
        scheduler = scheduler_override
    action_volume = actions.view(batch_size, num_frames, action_per_frame, action_dim).permute(0, 3, 1, 2).unsqueeze(-1)
    action_mask_volume = None
    if action_mask is not None:
        action_mask_volume = action_mask.view(batch_size, num_frames, action_per_frame, action_dim).permute(0, 3, 1, 2).unsqueeze(-1)
    if frame_sigma_values is not None and frame_timestep_ids is not None:
        raise ValueError("Specify only one of `frame_sigma_values` or `frame_timestep_ids`.")
    if frame_sigma_values is None and frame_timestep_ids is None:
        timestep_ids = sample_timestep_id(
            batch_size=batch_size,
            sample_shape=(num_frames,),
            num_train_timesteps=training_config.action_num_train_timesteps,
            device=actions.device,
        )
        frame_timesteps = scheduler.timesteps.to(device=actions.device)[timestep_ids]
    elif frame_timestep_ids is not None:
        if tuple(frame_timestep_ids.shape) != (batch_size, num_frames):
            raise ValueError(
                "Frame-aligned action timestep IDs must have shape [B, F], "
                f"got {tuple(frame_timestep_ids.shape)}, expected={(batch_size, num_frames)}."
            )
        frame_timestep_ids = frame_timestep_ids.to(device=actions.device, dtype=torch.int64)
        frame_timesteps = scheduler.timesteps.to(device=actions.device)[frame_timestep_ids]
    else:
        assert frame_sigma_values is not None
        if tuple(frame_sigma_values.shape) != (batch_size, num_frames):
            raise ValueError(
                "Frame-aligned action sigma values must have shape [B, F], "
                f"got {tuple(frame_sigma_values.shape)}, expected={(batch_size, num_frames)}."
            )
        frame_timesteps = timesteps_matching_sigmas(
            scheduler,
            frame_sigma_values.to(device=actions.device, dtype=scheduler.sigmas.dtype),
        )
    action_noise = torch.randn_like(action_volume)
    noisy_action_volume = scheduler.add_noise(action_volume, action_noise, frame_timesteps, t_dim=2)
    targets_volume = scheduler.training_target(action_volume, action_noise, frame_timesteps)
    if action_mask_volume is not None:
        noisy_action_volume = noisy_action_volume * action_mask_volume.float()
        targets_volume = targets_volume * action_mask_volume.float()
    noisy_actions = noisy_action_volume.squeeze(-1).permute(0, 2, 3, 1).reshape(batch_size, action_horizon, action_dim)
    targets = targets_volume.squeeze(-1).permute(0, 2, 3, 1).reshape(batch_size, action_horizon, action_dim)
    slot_timesteps = frame_timesteps.repeat_interleave(action_per_frame, dim=1)
    return FrameAlignedActionFlowMatchTrainArtifacts(
        frame_timesteps=frame_timesteps,
        slot_timesteps=slot_timesteps,
        noisy_actions=noisy_actions,
        targets=targets,
        condition_actions=actions,
        action_mask=action_mask,
        scheduler=scheduler,
    )


def build_block_coupled_action_flow_match_train_artifacts(
    actions: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    training_config: TrainingConfig,
    future_video_timesteps: torch.Tensor,
    num_frame_per_block: int,
    num_action_per_block: int,
) -> BlockCoupledActionFlowMatchTrainArtifacts:
    """Create DreamZero-style noisy actions coupled to future video block noise.

    `future_video_timesteps` is expected to contain only the future noisy video
    frames, i.e. the clean observed prefix has already been removed. We collapse
    each `num_frame_per_block` run to one block timestep, then repeat that block
    timestep across the aligned action slots.
    """

    if actions.ndim != 3:
        raise ValueError(
            "Expected actions with shape [B, H_action, D_action], "
            f"got {tuple(actions.shape)}."
        )
    if future_video_timesteps.ndim != 2:
        raise ValueError(
            "Expected future video timesteps with shape [B, F_future], "
            f"got {tuple(future_video_timesteps.shape)}."
        )
    batch_size, action_horizon, _ = actions.shape
    if future_video_timesteps.shape[0] != batch_size:
        raise ValueError(
            "Action/video batch size mismatch for coupled noise, "
            f"got actions batch={batch_size}, video batch={future_video_timesteps.shape[0]}."
        )
    if future_video_timesteps.shape[1] % num_frame_per_block != 0:
        raise ValueError(
            "Expected future video frames to be divisible by `num_frame_per_block`, "
            f"got frames={future_video_timesteps.shape[1]}, num_frame_per_block={num_frame_per_block}."
        )
    num_blocks = future_video_timesteps.shape[1] // num_frame_per_block
    expected_horizon = num_blocks * num_action_per_block
    if action_horizon != expected_horizon:
        raise ValueError(
            "DreamZero-style coupled action diffusion expects `action_horizon == num_blocks * num_action_per_block`, "
            f"got action_horizon={action_horizon}, num_blocks={num_blocks}, num_action_per_block={num_action_per_block}."
        )

    scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    scheduler.set_timesteps(training_config.action_num_train_timesteps, training=True)
    block_timesteps = future_video_timesteps.view(batch_size, num_blocks, num_frame_per_block)[:, :, 0]
    slot_timesteps = block_timesteps.repeat_interleave(num_action_per_block, dim=1)
    noise = torch.randn_like(actions)
    noisy_actions = scheduler.add_noise(actions, noise, slot_timesteps, t_dim=1)
    targets = scheduler.training_target(actions, noise, slot_timesteps)
    if action_mask is not None:
        noisy_actions = noisy_actions * action_mask.float()
        targets = targets * action_mask.float()
    return BlockCoupledActionFlowMatchTrainArtifacts(
        block_timesteps=block_timesteps,
        timesteps=slot_timesteps,
        noisy_actions=noisy_actions,
        targets=targets,
        action_mask=action_mask,
        scheduler=scheduler,
    )


__all__ = [
    "ActionFlowMatchTrainArtifacts",
    "VideoFlowMatchTrainArtifacts",
    "FrameAlignedActionFlowMatchTrainArtifacts",
    "BlockCoupledActionFlowMatchTrainArtifacts",
    "build_action_flow_match_train_artifacts",
    "build_video_flow_match_train_artifacts",
    "build_frame_aligned_action_flow_match_train_artifacts",
    "build_block_coupled_action_flow_match_train_artifacts",
]
