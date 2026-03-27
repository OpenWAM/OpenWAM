from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import ActionTargetStateEncoding, GripperRepresentation, RotationRepresentation


@dataclass(frozen=True)
class PoseSequence:
    """Absolute EEF pose sequence parsed from a state trajectory.

    Attributes:
        position:
            Cartesian positions, `[T, 3]`.
        quaternion:
            Unit quaternions in `xyzw` order, `[T, 4]`.
        gripper:
            Optional gripper state, `[T, D_gripper]`.
    """

    position: torch.Tensor
    quaternion: torch.Tensor
    gripper: torch.Tensor | None = None


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


def reconstruct_absolute_pose_targets(
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
    relative_pose_targets: torch.Tensor,
    *,
    rotation_representation: RotationRepresentation | str,
) -> PoseSequence:
    """Recover absolute pose from a reference-anchored pose target."""

    if relative_pose_targets.ndim != 2:
        raise ValueError(f"Expected relative pose targets with shape [T, D], got {tuple(relative_pose_targets.shape)}.")

    rel_position = relative_pose_targets[:, :3]
    if rotation_representation == RotationRepresentation.QUAT:
        if relative_pose_targets.shape[-1] < 7:
            raise ValueError("Quaternion pose targets require at least 7 dims: `[xyz, xyzw]`.")
        rel_quaternion = normalize_quaternion(relative_pose_targets[:, 3:7])
        gripper_start = 7
    elif rotation_representation == RotationRepresentation.AXIS_ANGLE:
        if relative_pose_targets.shape[-1] < 6:
            raise ValueError("Axis-angle pose targets require at least 6 dims: `[xyz, axis_angle]`.")
        rel_quaternion = axis_angle_to_quaternion(relative_pose_targets[:, 3:6])
        gripper_start = 6
    else:
        raise ValueError(f"Unsupported rotation representation: {rotation_representation}")

    abs_position = rel_position + reference_position.unsqueeze(0)
    abs_quaternion = quaternion_multiply(
        reference_quaternion.unsqueeze(0).expand_as(rel_quaternion),
        rel_quaternion,
    )
    abs_quaternion = normalize_quaternion(abs_quaternion)
    gripper = relative_pose_targets[:, gripper_start:] if relative_pose_targets.shape[-1] > gripper_start else None
    return PoseSequence(position=abs_position, quaternion=abs_quaternion, gripper=gripper)


def state_sequence_to_pose_sequence(
    state_sequence: torch.Tensor,
    *,
    state_encoding: ActionTargetStateEncoding | str,
) -> PoseSequence:
    """Parse a raw proprio sequence into absolute EEF pose tensors."""

    if state_encoding == ActionTargetStateEncoding.EEF_POS_AXISANGLE_GRIPPER_2D:
        if state_sequence.shape[-1] < 8:
            raise ValueError(
                "Expected state encoding `eef_pos_axisangle_gripper_2d` to expose at least 8 dims "
                f"but received {state_sequence.shape[-1]}."
            )
        position = state_sequence[:, 0:3]
        axis_angle = state_sequence[:, 3:6]
        quaternion = axis_angle_to_quaternion(axis_angle)
        gripper = state_sequence[:, 6:8]
        return PoseSequence(position=position, quaternion=quaternion, gripper=gripper)

    if state_encoding == ActionTargetStateEncoding.EEF_POS_QUAT_GRIPPER_1D:
        if state_sequence.shape[-1] < 8:
            raise ValueError(
                "Expected state encoding `eef_pos_quat_gripper_1d` to expose at least 8 dims "
                f"but received {state_sequence.shape[-1]}."
            )
        position = state_sequence[:, 0:3]
        quaternion = normalize_quaternion(state_sequence[:, 3:7])
        gripper = state_sequence[:, 7:8]
        return PoseSequence(position=position, quaternion=quaternion, gripper=gripper)

    raise ValueError(f"Unsupported pose-state encoding: {state_encoding}")


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors `[T, 3]` into `xyzw` quaternions `[T, 4]`."""

    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle tensor with last dim 3, got {axis_angle.shape[-1]}.")

    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angle = angle * 0.5
    sin_half = torch.sin(half_angle)

    # The zero-angle branch is common near steady-state manipulation. Use a
    # first-order limit so the conversion stays numerically stable.
    safe_axis = axis_angle / angle.clamp_min(1e-8)
    xyz = safe_axis * sin_half
    w = torch.cos(half_angle)

    identity_quaternion = torch.zeros_like(torch.cat([xyz, w], dim=-1))
    identity_quaternion[..., 3] = 1.0
    quaternion = torch.cat([xyz, w], dim=-1)
    quaternion = torch.where(angle > 1e-8, quaternion, identity_quaternion)
    return normalize_quaternion(quaternion)


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized `xyzw` quaternions to axis-angle vectors `[T, 3]`."""

    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {quaternion.shape[-1]}.")

    normalized = normalize_quaternion(quaternion)
    xyz = normalized[..., 0:3]
    w = normalized[..., 3:4].clamp(min=-1.0, max=1.0)
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    half_angle = torch.atan2(sin_half, w)
    angle = 2.0 * half_angle
    safe_axis = xyz / sin_half.clamp_min(1e-8)
    axis_angle = safe_axis * angle
    return torch.where(sin_half > 1e-8, axis_angle, torch.zeros_like(axis_angle))


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


def quaternion_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    """Invert normalized `xyzw` quaternions."""

    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {quaternion.shape[-1]}.")

    conjugate = quaternion.clone()
    conjugate[..., 0:3] = -conjugate[..., 0:3]
    denom = (quaternion * quaternion).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return conjugate / denom


def quaternion_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Hamilton product for `xyzw` quaternions."""

    if lhs.shape[-1] != 4 or rhs.shape[-1] != 4:
        raise ValueError("Quaternion multiplication expects tensors ending in 4 dims.")

    x1, y1, z1, w1 = lhs.unbind(dim=-1)
    x2, y2, z2, w2 = rhs.unbind(dim=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize `xyzw` quaternions along the last dimension."""

    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
