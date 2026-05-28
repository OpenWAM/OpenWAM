from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from open_wam.configs import InferenceConfig, TrainingConfig

from .flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


class FlowMatchScheduler:
    """LingBot-style flow-matching scheduler.

    This mirrors the scheduler used in the exact parallel-stream runtime:
    - one discrete training grid of `num_train_timesteps`
    - noisy sample construction `x_t = (1 - sigma) * x + sigma * noise`
    - flow target `noise - x`
    - first-order inference update along the learned flow field
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
    ) -> None:
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.exponential_shift_mu if self.exponential_shift_mu is not None else 0.0
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())
            self.training = True
        else:
            self.training = False

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        t_dim: int = 2,
    ) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=original_samples.device)
        timestep = timestep.to(device=original_samples.device)
        flat_timestep = timestep.reshape(-1)
        timestep_id = torch.argmin(
            (self.timesteps[:, None].to(flat_timestep.device) - flat_timestep[None]).abs(),
            dim=0,
        ).reshape(timestep.shape)
        sigma_values = self.sigmas.to(original_samples.device)[timestep_id].to(original_samples.dtype)
        shape = [1] * noise.ndim
        if timestep.ndim == 0:
            pass
        elif timestep.ndim == 1:
            shape[t_dim] = timestep.shape[0]
        elif timestep.ndim == 2:
            shape[0] = timestep.shape[0]
            shape[t_dim] = timestep.shape[1]
        else:
            raise ValueError(
                "Expected timestep to be scalar, [T], or [B, T], "
                f"got shape {tuple(timestep.shape)}."
            )
        sigma = sigma_values.view(shape)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_id = torch.argmin((self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0)
        return self.linear_timesteps_weights.to(timestep.device)[timestep_id].to(timestep.device)

    def sigma_for_timesteps(self, timestep: torch.Tensor) -> torch.Tensor:
        flat_timestep = timestep.reshape(-1)
        timestep_id = torch.argmin(
            (self.timesteps[:, None].to(flat_timestep.device) - flat_timestep[None]).abs(),
            dim=0,
        ).reshape(timestep.shape)
        return self.sigmas.to(timestep.device)[timestep_id]

    def timestep_matching_sigma(self, sigma: torch.Tensor | float) -> torch.Tensor:
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(float(sigma), dtype=self.timesteps.dtype)
        flat_sigma = sigma.reshape(-1)
        timestep_id = torch.argmin(
            (self.sigmas[:, None].to(flat_sigma.device, dtype=flat_sigma.dtype) - flat_sigma[None]).abs(),
            dim=0,
        ).reshape(sigma.shape)
        return self.timesteps.to(device=flat_sigma.device)[timestep_id]

    def next_sigma(self, timestep_index: int) -> torch.Tensor:
        if int(timestep_index) + 1 >= len(self.sigmas):
            final_sigma = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
            return self.sigmas.new_tensor(final_sigma)
        return self.sigmas[int(timestep_index) + 1]

    def step_with_sigmas(
        self,
        model_output: torch.Tensor,
        *,
        sigma: torch.Tensor | float,
        sigma_next: torch.Tensor | float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(float(sigma), device=sample.device, dtype=sample.dtype)
        if not isinstance(sigma_next, torch.Tensor):
            sigma_next = torch.tensor(float(sigma_next), device=sample.device, dtype=sample.dtype)
        return sample + model_output * (
            sigma_next.to(device=sample.device, dtype=sample.dtype)
            - sigma.to(device=sample.device, dtype=sample.dtype)
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        to_final: bool = False,
    ) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(float(timestep), device=sample.device, dtype=self.timesteps.dtype)
        timestep = timestep.to(device=sample.device)
        if timestep.numel() != 1:
            raise ValueError(f"`FlowMatchScheduler.step` expects a scalar timestep, got shape {tuple(timestep.shape)}.")
        device_timesteps = self.timesteps.to(sample.device)
        device_sigmas = self.sigmas.to(sample.device)
        timestep_id = torch.argmin((device_timesteps - timestep.reshape(())).abs())
        sigma = device_sigmas[timestep_id].to(sample.dtype)
        if to_final:
            sigma_next = torch.tensor(
                1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0,
                device=sample.device,
                dtype=sample.dtype,
            )
        else:
            final_sigma = torch.tensor(
                1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0,
                device=sample.device,
                dtype=sample.dtype,
            )
            next_index = torch.clamp(timestep_id + 1, max=len(self.timesteps) - 1)
            next_grid_sigma = device_sigmas[next_index].to(sample.dtype)
            sigma_next = torch.where(
                timestep_id + 1 >= len(self.timesteps),
                final_sigma,
                next_grid_sigma,
            )
        return sample + model_output * (sigma_next - sigma)


def sample_timestep_id(
    batch_size: int,
    *,
    sample_shape: tuple[int, ...] | None = None,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
    device: torch.device | None = None,
) -> torch.Tensor:
    shape = (batch_size, *(sample_shape or ()))
    u = torch.rand(size=shape, device=device)
    u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    return (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)


def timesteps_matching_sigmas(
    scheduler: FlowMatchScheduler,
    sigma_values: torch.Tensor,
) -> torch.Tensor:
    scheduler_sigmas = scheduler.sigmas.to(device=sigma_values.device, dtype=sigma_values.dtype)
    scheduler_timesteps = scheduler.timesteps.to(device=sigma_values.device)
    flat_sigmas = sigma_values.reshape(-1)
    indices = torch.argmin((scheduler_sigmas[:, None] - flat_sigmas[None]).abs(), dim=0)
    return scheduler_timesteps[indices].reshape(sigma_values.shape)


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
) -> VideoFlowMatchTrainArtifacts:
    """Create LingBot-style noisy video latents with one timestep per frame.

    The sampled timestep is broadcast across channels and spatial positions of
    each frame, matching LingBot's frame-wise latent diffusion semantics.
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
                f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
            )
        clean_condition_latents = condition_latents.to(device=video_latents.device, dtype=video_latents.dtype)
    condition_timesteps = torch.zeros_like(timesteps)
    if noisy_condition_prob > 0.0:
        # Augmentation decision must be identical across ranks under FSDP:
        # different branches produce different autograd-graph shapes, which
        # desynchronizes FSDP's per-rank backward all_gather schedule and
        # triggers NCCL watchdog timeouts. Sample on rank 0 and broadcast.
        decision = torch.rand(1, device=video_latents.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
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
            condition_timesteps = scheduler.timesteps.to(device=video_latents.device)[condition_timestep_ids]
            condition_noise = torch.randn_like(video_latents)
            clean_condition_latents = scheduler.add_noise(
                clean_condition_latents,
                condition_noise,
                condition_timesteps,
                t_dim=2,
            )
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


def build_action_flow_match_inference_scheduler(
    *,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    num_inference_steps_override: int | None = None,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    scheduler.set_timesteps(num_inference_steps_override or inference_config.action_num_inference_steps)
    return scheduler


def build_video_flow_match_inference_scheduler(
    *,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
    num_inference_steps_override: int | None = None,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=training_config.video_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.video_num_train_timesteps,
    )
    scheduler.set_timesteps(num_inference_steps_override or inference_config.video_num_inference_steps)
    return scheduler


def build_flow_unipc_inference_scheduler(
    *,
    num_train_timesteps: int,
    sigma_shift: float,
    num_inference_steps: int,
    device: torch.device,
) -> FlowUniPCMultistepScheduler:
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1.0,
    )
    scheduler.set_timesteps(
        num_inference_steps,
        device=device,
        shift=float(sigma_shift),
    )
    return scheduler


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
