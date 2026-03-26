from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from open_wam.configs import InferenceConfig, TrainingConfig


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
        timestep = timestep.cpu()
        timestep = timestep[None]
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_id = torch.argmin((self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0)
        return self.linear_timesteps_weights.to(timestep.device)[timestep_id].to(timestep.device)

    def sigma_for_timesteps(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_id = torch.argmin((self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0)
        return self.sigmas.to(timestep.device)[timestep_id]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        to_final: bool = False,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = float(self.sigmas[timestep_id])
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
        else:
            sigma_next = float(self.sigmas[timestep_id + 1])
        return sample + model_output * (sigma_next - sigma)


def sample_timestep_id(
    batch_size: int,
    *,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
    device: torch.device | None = None,
) -> torch.Tensor:
    u = torch.rand(size=[batch_size], device=device)
    u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    return (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)


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


def build_action_flow_match_inference_scheduler(
    *,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=training_config.action_sigma_shift,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=training_config.action_num_train_timesteps,
    )
    scheduler.set_timesteps(inference_config.action_num_inference_steps)
    return scheduler
