"""Final pose and joint action-target construction contracts."""

from __future__ import annotations

import torch

from open_wam.configs import (
    ActionNormalizationConfig,
    ActionNormalizationMode,
    ActionTargetStateEncoding,
    GripperRepresentation,
    RotationRepresentation,
)

from .action_gripper import (
    collapse_gripper_state,
    extract_action_command_gripper_targets,
    extract_public_gripper_targets,
)
from .action_normalization import normalize_joint_positions
from .action_pose import (
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_continuous_6d,
    state_sequence_to_pose_sequence,
)


def build_relative_pose_targets(
    state_sequence: torch.Tensor,
    *,
    state_encoding: ActionTargetStateEncoding | str,
    rotation_representation: RotationRepresentation | str,
    include_gripper: bool,
    gripper_representation: GripperRepresentation | str,
    raw_action_sequence: torch.Tensor | None = None,
    gripper_action_index: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float] | str | bool]]:
    """Convert absolute proprio state into reference-anchored pose targets.

    The output is shaped `[T, D_action]` and is suitable for the common WAM
    action contract. The first timestep is the reference pose itself, so its
    pose component is exactly zero translation and identity rotation.
    """

    if state_sequence.ndim != 2:
        raise ValueError(f"Expected state sequence with shape [T, D], got {tuple(state_sequence.shape)}.")

    absolute_pose = state_sequence_to_pose_sequence(state_sequence, state_encoding=state_encoding)
    reference_position = absolute_pose.position[0]
    reference_quaternion = absolute_pose.quaternion[0]

    # Match LingBot's successful supervision convention:
    # - translation is anchored on the reference pose origin
    # - rotation is a true relative rotation `q_ref^-1 * q_t`
    relative_position = absolute_pose.position - reference_position.unsqueeze(0)
    relative_quaternion = quaternion_multiply(
        quaternion_inverse(reference_quaternion).unsqueeze(0).expand_as(absolute_pose.quaternion),
        absolute_pose.quaternion,
    )
    relative_quaternion = normalize_quaternion(relative_quaternion)

    if rotation_representation == RotationRepresentation.QUAT:
        relative_rotation = relative_quaternion
    elif rotation_representation == RotationRepresentation.AXIS_ANGLE:
        relative_rotation = quaternion_to_axis_angle(relative_quaternion)
    elif rotation_representation == RotationRepresentation.CONTINUOUS_6D:
        relative_rotation = quaternion_to_continuous_6d(relative_quaternion)
    else:
        raise ValueError(f"Unsupported rotation representation: {rotation_representation}")

    parts = [relative_position, relative_rotation]
    if include_gripper:
        if absolute_pose.gripper is None:
            raise ValueError("Requested gripper targets, but the selected state encoding has no gripper channels.")
        parts.append(
            extract_public_gripper_targets(
                state_gripper=absolute_pose.gripper,
                raw_action_sequence=raw_action_sequence,
                gripper_representation=gripper_representation,
                gripper_action_index=gripper_action_index,
            )
        )

    targets = torch.cat(parts, dim=-1).to(dtype=torch.float32)
    mask = torch.ones_like(targets, dtype=torch.float32)
    metadata = {
        "reference_position": reference_position.tolist(),
        "reference_quaternion_xyzw": reference_quaternion.tolist(),
        "rotation_representation": rotation_representation,
        "include_gripper": include_gripper,
        "gripper_representation": gripper_representation,
        "gripper_action_index": gripper_action_index,
        "state_encoding": state_encoding,
    }
    return targets, mask, metadata


def build_absolute_joint_position_targets(
    joint_position_sequence: torch.Tensor,
    *,
    include_gripper: bool,
    gripper_representation: GripperRepresentation | str,
    gripper_position_sequence: torch.Tensor | None = None,
    raw_action_sequence: torch.Tensor | None = None,
    gripper_action_index: int = -1,
    normalization: ActionNormalizationConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float] | str | bool | int]]:
    """Build absolute joint-position action targets from proprio state.

    The arm portion is a measured joint target, not a delta. The gripper
    portion can either copy the native scalar action command or expose measured
    gripper qpos for closed-loop gripper tracking during replay.
    """

    if joint_position_sequence.ndim != 2:
        raise ValueError(
            "Expected joint-position sequence with shape [T, D_joint], "
            f"got {tuple(joint_position_sequence.shape)}."
        )
    if joint_position_sequence.shape[0] == 0:
        raise ValueError("Expected at least one joint-position target.")

    normalization = normalization or ActionNormalizationConfig()
    joint_targets = normalize_joint_positions(
        joint_position_sequence.to(dtype=torch.float32),
        normalization=normalization,
    )
    parts = [joint_targets]
    if include_gripper:
        if gripper_representation == GripperRepresentation.ACTION_COMMAND:
            gripper_targets = extract_action_command_gripper_targets(
                raw_action_sequence=raw_action_sequence,
                target_length=joint_position_sequence.shape[0],
                gripper_action_index=gripper_action_index,
            )
        elif gripper_representation in {GripperRepresentation.FIRST_CHANNEL, GripperRepresentation.ALL_CHANNELS}:
            if gripper_position_sequence is None:
                raise ValueError(
                    "absolute_joint_position with gripper_representation="
                    f"{gripper_representation} requires `gripper_position_sequence`."
                )
            if gripper_position_sequence.shape[0] != joint_position_sequence.shape[0]:
                raise ValueError(
                    "Joint-position and gripper-position sequences must have the same length when building "
                    "absolute joint-position targets."
                )
            gripper_targets = collapse_gripper_state(
                gripper_position_sequence.to(dtype=torch.float32),
                gripper_representation=gripper_representation,
            )
        else:
            raise ValueError(f"Unsupported gripper representation: {gripper_representation}")
        parts.append(gripper_targets)

    targets = torch.cat(parts, dim=-1).to(dtype=torch.float32)
    mask = torch.ones_like(targets, dtype=torch.float32)
    metadata = {
        "action_target_family": "absolute_joint_position",
        "joint_position_dim": int(joint_position_sequence.shape[-1]),
        "include_gripper": include_gripper,
        "gripper_representation": str(gripper_representation),
        "gripper_action_index": gripper_action_index,
        "joint_position_normalization_mode": str(normalization.mode),
        "joint_position_normalized": normalization.mode != ActionNormalizationMode.NONE,
    }
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        metadata["joint_position_lower"] = list(normalization.lower)
        metadata["joint_position_upper"] = list(normalization.upper)
    return targets, mask, metadata


def expected_pose_target_dim(
    *,
    rotation_representation: RotationRepresentation | str,
    include_gripper: bool,
    gripper_representation: GripperRepresentation | str,
) -> int:
    """Return the public action dimension implied by one pose-target config."""

    if rotation_representation == RotationRepresentation.QUAT:
        dim = 3 + 4
    elif rotation_representation == RotationRepresentation.AXIS_ANGLE:
        dim = 3 + 3
    elif rotation_representation == RotationRepresentation.CONTINUOUS_6D:
        dim = 3 + 6
    else:
        raise ValueError(f"Unsupported rotation representation: {rotation_representation}")

    if include_gripper:
        if gripper_representation == GripperRepresentation.ALL_CHANNELS:
            dim += 2
        elif gripper_representation in {GripperRepresentation.FIRST_CHANNEL, GripperRepresentation.ACTION_COMMAND}:
            dim += 1
        else:
            raise ValueError(f"Unsupported gripper representation: {gripper_representation}")
    return dim


def expected_joint_position_target_dim(
    *,
    joint_dim: int,
    include_gripper: bool,
    gripper_representation: GripperRepresentation | str,
) -> int:
    """Return the target dimension implied by absolute joint-position control."""

    dim = int(joint_dim)
    if include_gripper:
        if gripper_representation in {GripperRepresentation.FIRST_CHANNEL, GripperRepresentation.ACTION_COMMAND}:
            dim += 1
        elif gripper_representation == GripperRepresentation.ALL_CHANNELS:
            dim += 2
        else:
            raise ValueError(f"Unsupported gripper representation: {gripper_representation}")
    return dim


__all__ = [
    "build_absolute_joint_position_targets",
    "build_relative_pose_targets",
    "expected_joint_position_target_dim",
    "expected_pose_target_dim",
]
