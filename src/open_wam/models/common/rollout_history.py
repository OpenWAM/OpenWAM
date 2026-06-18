from __future__ import annotations

import numpy as np
import torch


def resolve_execute_action_steps(
    execute_action_steps: int | None,
    *,
    execute_frame_chunk_size: int | None = None,
    action_horizon: int,
    action_per_frame: int,
) -> int:
    action_horizon = int(action_horizon)
    action_per_frame = int(action_per_frame)
    if action_horizon <= 0:
        raise ValueError(f"Expected action_horizon > 0, got {action_horizon}.")
    if action_per_frame <= 0:
        raise ValueError(f"Expected action_per_frame > 0, got {action_per_frame}.")
    if execute_action_steps is not None and execute_frame_chunk_size is not None:
        raise ValueError("Pass only one of --execute-action-steps or --execute-frame-chunk-size.")
    if execute_frame_chunk_size is not None:
        frame_chunk_size = int(execute_frame_chunk_size)
        if frame_chunk_size <= 0:
            raise ValueError(f"Expected --execute-frame-chunk-size to be positive, got {frame_chunk_size}.")
        execute_action_steps = frame_chunk_size * action_per_frame
    if execute_action_steps is None:
        return action_horizon
    resolved = int(execute_action_steps)
    if resolved <= 0:
        raise ValueError(f"Expected --execute-action-steps to be positive, got {resolved}.")
    if resolved > action_horizon:
        raise ValueError(
            "--execute-action-steps cannot exceed the predicted action horizon, "
            f"got execute_action_steps={resolved}, action_horizon={action_horizon}."
        )
    if resolved % action_per_frame != 0:
        raise ValueError(
            "--execute-action-steps must be aligned to action_per_frame so streaming VAE warmup receives "
            "whole latent-frame groups, "
            f"got execute_action_steps={resolved}, action_per_frame={action_per_frame}."
        )
    return resolved


def build_executed_action_history_tensor(
    executed_control_actions: list[np.ndarray],
    *,
    start_frame_group: int,
    action_per_frame: int,
    action_dim: int,
) -> torch.Tensor | None:
    """Build MoT/LIBERO warmup history from actions sent to the simulator.

    The helper returns a CPU float32 tensor for actions actually sent to the
    simulator. Legacy zero-bootstrap rows for skipped frame groups are
    deprecated because they expose synthetic action context to the model.
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
        raise ValueError(
            "Skipped frame-group action bootstrap is deprecated because it would expose synthetic zero "
            "actions as model context. Use first-frame prefix conditioning that executes the full generated "
            "chunk instead."
        )
    return torch.from_numpy(executed).unsqueeze(0)
