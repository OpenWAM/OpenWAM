from __future__ import annotations

from collections.abc import Iterable

import torch

from open_wam.configs.enums import StrEnum


def add_joint_conditioning_mode_metrics(
    metrics: dict[str, torch.Tensor],
    *,
    namespace: str,
    mode_value: str,
    modes: Iterable[StrEnum],
    action_loss: torch.Tensor,
    latent_loss: torch.Tensor,
    action_loss_active: torch.Tensor,
    latent_loss_active: torch.Tensor,
    action_metric_name: str = "action_loss_sum",
    latent_metric_name: str = "latent_loss_sum",
    action_metric_aliases: tuple[str, ...] = (),
    latent_metric_aliases: tuple[str, ...] = (),
) -> None:
    """Add per-mode count/loss rollups for joint video/action conditioning."""

    metric_device = action_loss.device
    one = torch.ones((), device=metric_device)
    zero = torch.zeros((), device=metric_device)
    for mode in modes:
        active = one if str(mode_value) == mode.value else zero
        prefix = f"{namespace}/{mode.value}"
        metrics[f"{prefix}/count"] = active.detach()
        action_value = (action_loss * active).detach()
        latent_value = (latent_loss * active).detach()
        metrics[f"{prefix}/{action_metric_name}"] = action_value
        metrics[f"{prefix}/{latent_metric_name}"] = latent_value
        for alias in action_metric_aliases:
            metrics[f"{prefix}/{alias}"] = action_value
        for alias in latent_metric_aliases:
            metrics[f"{prefix}/{alias}"] = latent_value
    metrics[f"{namespace}/action_loss_active"] = action_loss_active.to(dtype=torch.float32).detach()
    metrics[f"{namespace}/latent_loss_active"] = latent_loss_active.to(dtype=torch.float32).detach()
