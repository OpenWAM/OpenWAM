from __future__ import annotations

import torch


def clean_noisy_slot_tensor(
    clean_latent: torch.Tensor,
    *,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the clean tensor exposed through a noisy slot."""

    if action_mask is None:
        return clean_latent
    resolved_mask = action_mask.to(device=clean_latent.device, dtype=clean_latent.dtype)
    return clean_latent * resolved_mask


def force_clean_noisy_slot(
    artifact_dict: dict[str, torch.Tensor],
    clean_latent: torch.Tensor,
    *,
    action_mask: torch.Tensor | None = None,
) -> None:
    """Place a clean modality tensor into its noisy slot and mask its target."""

    artifact_dict["noisy_latents"] = clean_noisy_slot_tensor(
        clean_latent,
        action_mask=action_mask,
    )
    artifact_dict["targets"] = torch.zeros_like(clean_latent)
    artifact_dict["timesteps"] = torch.zeros_like(artifact_dict["timesteps"])


def zero_condition_slot(
    artifact_dict: dict[str, torch.Tensor],
    *,
    latent_key: str = "latent",
    timestep_key: str = "cond_timesteps",
) -> None:
    """Zero a method-local clean-condition slot in-place."""

    artifact_dict[latent_key] = torch.zeros_like(artifact_dict[latent_key])
    artifact_dict[timestep_key] = torch.zeros_like(artifact_dict[timestep_key])


def zero_loss_mask_like(mask: torch.Tensor | None, *, fallback_like: torch.Tensor) -> torch.Tensor:
    """Return a zero mask preserving the provided mask shape or fallback shape."""

    if mask is None:
        return torch.zeros_like(fallback_like)
    return torch.zeros_like(mask)
