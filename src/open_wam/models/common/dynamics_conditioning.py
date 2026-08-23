"""Architecture-neutral conditioning operations for dynamics objectives."""

from __future__ import annotations

import torch

from open_wam.configs.enums import DynamicsObjective


def append_dynamics_mode_context_token(
    transformer: torch.nn.Module,
    text_context: torch.Tensor,
    objective: DynamicsObjective | str,
) -> tuple[torch.Tensor, int]:
    """Append exactly one objective token through the shared visual core hook."""

    append = getattr(transformer, "append_generalist_mode_context_token", None)
    if not callable(append):
        raise TypeError(
            "Dynamics mode-token conditioning requires the visual core to expose "
            "`append_generalist_mode_context_token`."
        )
    resolved = append(text_context, DynamicsObjective(objective).value)
    token_count = int(resolved.shape[1]) - int(text_context.shape[1])
    if token_count != 1:
        raise ValueError(
            "Dynamics mode-token conditioning must append exactly one token, "
            f"got {token_count}."
        )
    return resolved, token_count


__all__ = ["append_dynamics_mode_context_token"]
