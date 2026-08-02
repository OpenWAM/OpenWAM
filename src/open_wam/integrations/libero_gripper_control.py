"""LIBERO gripper command and state-projection policy."""

from __future__ import annotations

import numpy as np
import torch

from open_wam.data.action_gripper import collapse_gripper_state


__all__ = [
    "gripper_command_for_substep",
    "gripper_qpos_tracking_command",
    "project_libero_gripper_state",
]


def gripper_command_for_substep(
    command: float,
    *,
    substep_index: int,
    substeps: int,
    policy: str,
) -> float:
    """Apply one configured gripper-command substep policy."""

    if policy == "repeat":
        return float(command)
    if policy == "first_only":
        return float(command) if int(substep_index) == 0 else 0.0
    if policy == "last_only":
        return (
            float(command)
            if int(substep_index) == int(substeps) - 1
            else 0.0
        )
    raise ValueError(f"Unknown gripper substep policy: {policy!r}.")


def gripper_qpos_tracking_command(
    *,
    current_gripper_positions: np.ndarray,
    target_gripper_positions: np.ndarray,
    tolerance: float = 0.001,
) -> float:
    """Resolve a normalized command that tracks measured gripper qpos."""

    target_values = np.asarray(
        target_gripper_positions,
        dtype=np.float32,
    ).reshape(-1)
    current_values = np.asarray(
        current_gripper_positions,
        dtype=np.float32,
    ).reshape(-1)
    if target_values.size == 1:
        current_value = float(current_values[0])
        target_value = float(target_values[0])
    else:
        current_value = _gripper_opening(current_values)
        target_value = _gripper_opening(target_values)
    if current_value > target_value + float(tolerance):
        return 1.0
    if current_value < target_value - float(tolerance):
        return -1.0
    return 0.0


def project_libero_gripper_state(
    gripper_state: torch.Tensor,
    *,
    gripper_representation: str,
) -> torch.Tensor:
    """Project one measured gripper state into the public representation."""

    if gripper_state.ndim != 1:
        raise ValueError(
            "Expected one gripper state vector, got shape "
            f"{tuple(gripper_state.shape)}."
        )
    if gripper_representation == "action_command":
        raise ValueError(
            "action_command is a control-domain target and cannot be "
            "recovered from env gripper state alone."
        )
    return collapse_gripper_state(
        gripper_state.unsqueeze(0),
        gripper_representation=gripper_representation,
    )[0]


def _gripper_opening(gripper_positions: np.ndarray) -> float:
    values = np.asarray(gripper_positions, dtype=np.float32).reshape(-1)
    if values.size >= 2:
        return float(values[0] - values[1])
    return float(values[0])
