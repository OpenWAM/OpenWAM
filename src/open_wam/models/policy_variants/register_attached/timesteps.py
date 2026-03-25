from __future__ import annotations

import torch

from open_wam.models.policy_variants.common.positions import sinusoidal_embedding


def build_action_register_timestep_context(
    batch_size: int,
    action_horizon: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    action_steps = torch.linspace(0.0, 1.0, action_horizon, device=device)
    return sinusoidal_embedding(action_steps, hidden_size)[None, :, :].expand(batch_size, -1, -1)
