from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from open_wam.models.common.flow_matching import sample_timestep_id


class TimestepGridSchedulerLike(Protocol):
    """Minimal scheduler interface for sampling from a discrete timestep grid."""

    num_train_timesteps: int
    timesteps: torch.Tensor
    sigmas: torch.Tensor


class SigmaLookupSchedulerLike(Protocol):
    """Scheduler interface for converting arbitrary timesteps back to sigmas."""

    def sigma_for_timesteps(self, timestep: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CoupledTimestepValues:
    """Per-frame timestep values coupled through shared sigma values."""

    video_timesteps: torch.Tensor
    action_timesteps: torch.Tensor
    sigma_values: torch.Tensor


def clean_timestep_values(
    *,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a per-frame clean timestep vector."""

    return torch.zeros(num_frames, device=device, dtype=dtype)


def sample_timestep_values(
    scheduler: TimestepGridSchedulerLike,
    *,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample one scheduler timestep per frame."""

    grid_length = _validate_timestep_grid(scheduler)
    timestep_ids = sample_timestep_id(
        batch_size=num_frames,
        num_train_timesteps=grid_length,
        device=device,
    )
    return scheduler.timesteps.to(device=device)[timestep_ids]


def sample_coupled_timestep_values(
    *,
    video_scheduler: TimestepGridSchedulerLike,
    action_scheduler: TimestepGridSchedulerLike,
    num_frames: int,
    device: torch.device,
) -> CoupledTimestepValues:
    """Sample video timesteps and map action timesteps to matching sigmas."""

    video_grid_length = _validate_timestep_grid(video_scheduler)
    _validate_timestep_grid(action_scheduler)
    timestep_ids = sample_timestep_id(
        batch_size=num_frames,
        num_train_timesteps=video_grid_length,
        device=device,
    )
    sigma_values = video_scheduler.sigmas.to(device=device)[timestep_ids]
    return CoupledTimestepValues(
        video_timesteps=_timesteps_matching_sigmas(video_scheduler, sigma_values),
        action_timesteps=_timesteps_matching_sigmas(action_scheduler, sigma_values),
        sigma_values=sigma_values,
    )


def _validate_timestep_grid(scheduler: TimestepGridSchedulerLike) -> int:
    timesteps_len = int(scheduler.timesteps.numel())
    sigmas_len = int(scheduler.sigmas.numel())
    if timesteps_len <= 0 or sigmas_len <= 0:
        raise ValueError("Scheduler timestep grid must contain at least one timestep and sigma.")
    if timesteps_len != sigmas_len:
        raise ValueError(
            "Scheduler timestep and sigma grids must have matching lengths, "
            f"got timesteps={timesteps_len}, sigmas={sigmas_len}."
        )
    return timesteps_len


def _timesteps_matching_sigmas(
    scheduler: TimestepGridSchedulerLike,
    sigma_values: torch.Tensor,
) -> torch.Tensor:
    scheduler_sigmas = scheduler.sigmas.to(device=sigma_values.device, dtype=sigma_values.dtype)
    scheduler_timesteps = scheduler.timesteps.to(device=sigma_values.device)
    flat_sigmas = sigma_values.reshape(-1)
    indices = torch.argmin((scheduler_sigmas[:, None] - flat_sigmas[None]).abs(), dim=0)
    return scheduler_timesteps[indices].reshape(sigma_values.shape)


def frame_sigmas_for_timesteps(
    scheduler: SigmaLookupSchedulerLike,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Return scheduler sigma values with the same shape as `timesteps`."""

    return scheduler.sigma_for_timesteps(timesteps)
