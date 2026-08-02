"""Flow-matching schedule and timestep primitives."""

from __future__ import annotations

import math

import torch


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


def zero_terminal_next_sigma(scheduler, step_index: int) -> torch.Tensor:
    """Resolve the next integration sigma, ending the schedule at zero."""

    if int(step_index) + 1 >= len(scheduler.sigmas):
        return scheduler.sigmas.new_tensor(0.0)
    return scheduler.sigmas[int(step_index) + 1]


def explicit_sigma_euler_step(
    sample: torch.Tensor,
    flow_pred: torch.Tensor,
    *,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """Apply one explicit-sigma Euler flow step."""

    return sample + flow_pred * (
        sigma_next.to(device=sample.device, dtype=sample.dtype)
        - sigma.to(device=sample.device, dtype=sample.dtype)
    )


def expand_scalar_timestep(
    value: torch.Tensor | float,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Materialize a scalar timestep over a requested stream shape."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar timestep value, got shape {tuple(value.shape)}.")
        return value.to(device=device, dtype=torch.float32).reshape(()).expand(shape).clone()
    return torch.full(shape, float(value), device=device, dtype=torch.float32)


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


__all__ = [
    "FlowMatchScheduler",
    "zero_terminal_next_sigma",
    "explicit_sigma_euler_step",
    "expand_scalar_timestep",
    "sample_timestep_id",
    "timesteps_matching_sigmas",
]
