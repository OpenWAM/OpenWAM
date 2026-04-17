from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from open_wam.configs import DiffusionNoiseSchedule, DiffusionSampler


def _sample_log_logistic(
    *,
    shape: tuple[int, ...],
    sigma_data: float,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    # Matches the default VPP/K-diffusion training density closely: sample a
    # logistic value in log-sigma space, clamp it to [sigma_min, sigma_max].
    uniform = torch.rand(shape, device=device).clamp_(1e-6, 1.0 - 1e-6)
    logistic = torch.log(uniform) - torch.log1p(-uniform)
    log_sigma = math.log(sigma_data) + 0.5 * logistic
    sigma = log_sigma.exp()
    return sigma.clamp_(min=sigma_min, max=sigma_max)


def _build_sigmas(
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    schedule: DiffusionNoiseSchedule,
    rho: float = 7.0,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError(f"`num_steps` must be positive, got {num_steps}.")
    ramp = torch.linspace(0.0, 1.0, steps=num_steps, device=device)
    if schedule == DiffusionNoiseSchedule.EXPONENTIAL:
        sigmas = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), steps=num_steps, device=device))
    elif schedule == DiffusionNoiseSchedule.KARRAS:
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    else:
        raise ValueError(f"Unsupported diffusion noise schedule '{schedule}'.")
    return torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)


class EDMActionGenerationBackend:
    """Generic EDM-style action-generation backend.

    The backend owns:
    - sigma sampling for training
    - Karras-style preconditioning constants
    - deterministic sampling loops for inference

    The actual denoiser network stays outside and is provided as a callable.
    """

    def __init__(
        self,
        *,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        noise_schedule: DiffusionNoiseSchedule | str,
        sampler: DiffusionSampler | str,
        num_sampling_steps: int,
    ) -> None:
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.noise_schedule = DiffusionNoiseSchedule(noise_schedule)
        self.sampler = DiffusionSampler(sampler)
        self.num_sampling_steps = num_sampling_steps

    def get_scalings(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)
        return c_skip, c_out, c_in

    def compute_training_loss(
        self,
        *,
        clean_actions: torch.Tensor,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = clean_actions.shape[0]
        sigmas = _sample_log_logistic(
            shape=(batch_size,),
            sigma_data=self.sigma_data,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            device=clean_actions.device,
        )
        noise = torch.randn_like(clean_actions)
        noised_actions = clean_actions + noise * sigmas[:, None, None]
        c_skip, c_out, c_in = self.get_scalings(sigmas)
        model_output = denoiser(noised_actions * c_in[:, None, None], sigmas)
        target = (clean_actions - c_skip[:, None, None] * noised_actions) / c_out[:, None, None]
        denoised_actions = model_output * c_out[:, None, None] + noised_actions * c_skip[:, None, None]
        loss = F.mse_loss(model_output.float(), target.float())
        return loss, denoised_actions, sigmas, noise

    def sample(
        self,
        *,
        batch_size: int,
        action_horizon: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        sample_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        sigmas = _build_sigmas(
            num_steps=self.num_sampling_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            device=device,
            schedule=self.noise_schedule,
        )
        sample = torch.randn(batch_size, action_horizon, action_dim, device=device, dtype=dtype) * sigmas[0]
        if sample_transform is not None:
            sample = sample_transform(sample)
        for step_index in range(len(sigmas) - 1):
            sigma = sigmas[step_index]
            next_sigma = sigmas[step_index + 1]
            sigma_batch = torch.full((batch_size,), float(sigma), device=device, dtype=torch.float32)
            c_skip, c_out, c_in = self.get_scalings(sigma_batch)
            model_output = denoiser(sample * c_in[:, None, None].to(dtype), sigma_batch)
            denoised = model_output * c_out[:, None, None].to(dtype) + sample * c_skip[:, None, None].to(dtype)
            if next_sigma.item() == 0.0:
                sample = denoised
                if sample_transform is not None:
                    sample = sample_transform(sample)
                continue
            if self.sampler == DiffusionSampler.DDIM:
                sample = denoised + (sample - denoised) * (next_sigma / sigma)
            elif self.sampler == DiffusionSampler.EULER:
                derivative = (sample - denoised) / sigma
                sample = sample + derivative * (next_sigma - sigma)
            else:
                raise ValueError(f"Unsupported diffusion sampler '{self.sampler}'.")
            if sample_transform is not None:
                sample = sample_transform(sample)
        return sample
