"""Public gripper-state and action-command projections."""

from __future__ import annotations

import torch

from open_wam.configs import GripperRepresentation


def collapse_gripper_state(
    gripper: torch.Tensor,
    *,
    gripper_representation: GripperRepresentation | str,
) -> torch.Tensor:
    """Expose gripper state in the configured public target format."""

    if gripper.ndim != 2:
        raise ValueError(f"Expected gripper sequence with shape [T, D], got {tuple(gripper.shape)}.")

    if gripper_representation == GripperRepresentation.ALL_CHANNELS:
        return gripper

    if gripper_representation == GripperRepresentation.FIRST_CHANNEL:
        return gripper[:, 0:1]

    raise ValueError(f"Unsupported gripper representation: {gripper_representation}")


def extract_public_gripper_targets(
    *,
    state_gripper: torch.Tensor,
    raw_action_sequence: torch.Tensor | None,
    gripper_representation: GripperRepresentation | str,
    gripper_action_index: int,
) -> torch.Tensor:
    """Build the public gripper supervision channel from state or raw action.

    `first_channel` / `all_channels` expose measured gripper state from the
    proprio tensor. `action_command` instead copies the scalar command from the
    raw action tensor, which is the semantically correct 1D LIBERO gripper
    control signal in `[-1, 1]`.
    """

    if gripper_representation in {GripperRepresentation.ALL_CHANNELS, GripperRepresentation.FIRST_CHANNEL}:
        return collapse_gripper_state(
            state_gripper,
            gripper_representation=gripper_representation,
        )

    if gripper_representation == GripperRepresentation.ACTION_COMMAND:
        if raw_action_sequence is None:
            raise ValueError(
                "gripper_representation=action_command requires `raw_action_sequence` so the public "
                "target can use the dataset's native scalar gripper command."
            )
        if raw_action_sequence.ndim != 2:
            raise ValueError(
                f"Expected raw action sequence with shape [T, D], got {tuple(raw_action_sequence.shape)}."
            )
        if raw_action_sequence.shape[0] != state_gripper.shape[0]:
            raise ValueError(
                "Raw action and state sequences must have the same length when building "
                "reference-relative pose targets."
            )
        action_dim = raw_action_sequence.shape[-1]
        resolved_index = gripper_action_index if gripper_action_index >= 0 else action_dim + gripper_action_index
        if resolved_index < 0 or resolved_index >= action_dim:
            raise ValueError(
                f"gripper_action_index={gripper_action_index} resolved outside action dim {action_dim}."
            )
        return raw_action_sequence[:, resolved_index : resolved_index + 1]

    raise ValueError(f"Unsupported gripper representation: {gripper_representation}")


def extract_action_command_gripper_targets(
    *,
    raw_action_sequence: torch.Tensor | None,
    target_length: int,
    gripper_action_index: int,
) -> torch.Tensor:
    """Extract a scalar gripper command from a native action sequence."""

    if raw_action_sequence is None:
        raise ValueError("absolute_joint_position with gripper action command requires `raw_action_sequence`.")
    if raw_action_sequence.ndim != 2:
        raise ValueError(f"Expected raw action sequence with shape [T, D], got {tuple(raw_action_sequence.shape)}.")
    if raw_action_sequence.shape[0] != target_length:
        raise ValueError(
            "Raw action and joint-position sequences must have the same length when appending gripper commands."
        )
    action_dim = raw_action_sequence.shape[-1]
    resolved_index = gripper_action_index if gripper_action_index >= 0 else action_dim + gripper_action_index
    if resolved_index < 0 or resolved_index >= action_dim:
        raise ValueError(f"gripper_action_index={gripper_action_index} resolved outside action dim {action_dim}.")
    return raw_action_sequence[:, resolved_index : resolved_index + 1].to(dtype=torch.float32)


__all__ = [
    "collapse_gripper_state",
    "extract_action_command_gripper_targets",
    "extract_public_gripper_targets",
]
