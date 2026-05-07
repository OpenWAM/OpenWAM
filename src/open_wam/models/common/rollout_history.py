from __future__ import annotations

import numpy as np
import torch


def build_executed_action_history_tensor(
    executed_control_actions: list[np.ndarray],
    *,
    start_frame_group: int,
    action_per_frame: int,
    action_dim: int,
) -> torch.Tensor | None:
    """Build MoT/LIBERO warmup history from actions sent to the simulator.

    The helper returns a CPU float32 tensor and prepends zero bootstrap tokens
    for skipped frame groups. Broader rollout-history state should use a
    richer typed contract instead of this visualization warmup helper.
    """

    if action_per_frame <= 0:
        raise ValueError(f"Expected action_per_frame > 0, got {action_per_frame}.")
    if action_dim <= 0:
        raise ValueError(f"Expected action_dim > 0, got {action_dim}.")
    if not executed_control_actions:
        return None
    executed = np.stack(executed_control_actions, axis=0).astype(np.float32, copy=False)
    if executed.ndim != 2 or int(executed.shape[-1]) != int(action_dim):
        raise ValueError(
            "Executed control action history must be [T, D_action], "
            f"got {tuple(executed.shape)}, action_dim={action_dim}."
        )
    skipped_tokens = max(0, int(start_frame_group)) * int(action_per_frame)
    if skipped_tokens > 0:
        bootstrap_actions = np.zeros((skipped_tokens, action_dim), dtype=np.float32)
        executed = np.concatenate([bootstrap_actions, executed], axis=0)
    return torch.from_numpy(executed).unsqueeze(0)
