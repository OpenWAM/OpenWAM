"""Pose records, rotation representations, and quaternion geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import ActionTargetStateEncoding, RotationRepresentation


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
    elif rotation_representation == RotationRepresentation.CONTINUOUS_6D:
        if relative_pose_targets.shape[-1] < 9:
            raise ValueError("Continuous-6D pose targets require at least 9 dims: `[xyz, rotation_6d]`.")
        rel_quaternion = rotation_matrix_to_quaternion(continuous_6d_to_rotation_matrix(relative_pose_targets[:, 3:9]))
        gripper_start = 9
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


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized `xyzw` quaternions to rotation matrices."""

    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {quaternion.shape[-1]}.")

    quat = normalize_quaternion(quaternion)
    x, y, z, w = quat.unbind(dim=-1)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    matrix = torch.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype, device=quat.device)
    matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[..., 0, 1] = 2.0 * (xy - zw)
    matrix[..., 0, 2] = 2.0 * (xz + yw)
    matrix[..., 1, 0] = 2.0 * (xy + zw)
    matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[..., 1, 2] = 2.0 * (yz - xw)
    matrix[..., 2, 0] = 2.0 * (xz - yw)
    matrix[..., 2, 1] = 2.0 * (yz + xw)
    matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def quaternion_to_continuous_6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to the continuous 6D rotation representation."""

    matrix = quaternion_to_rotation_matrix(quaternion)
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def continuous_6d_to_rotation_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert continuous 6D rotations to orthonormal rotation matrices."""

    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected continuous-6D tensor with last dim 6, got {rotation_6d.shape[-1]}.")

    first = _normalize_vectors(rotation_6d[..., 0:3])
    second_raw = rotation_6d[..., 3:6] - (first * rotation_6d[..., 3:6]).sum(dim=-1, keepdim=True) * first
    second = _normalize_vectors(_replace_degenerate_second_axis(first, second_raw))
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized `xyzw` quaternions."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices ending in [3, 3], got {tuple(matrix.shape)}.")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]
    qw = 0.5 * torch.sqrt((1.0 + m00 + m11 + m22).clamp_min(0.0))
    qx = 0.5 * _copy_sign(torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(0.0)), m21 - m12)
    qy = 0.5 * _copy_sign(torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(0.0)), m02 - m20)
    qz = 0.5 * _copy_sign(torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(0.0)), m10 - m01)
    return normalize_quaternion(torch.stack([qx, qy, qz, qw], dim=-1))


def pose_blocks_relative_to_anchor(
    actions: torch.Tensor,
    *,
    anchor: torch.Tensor,
    block_dims: int,
) -> torch.Tensor:
    """Express repeated [xyz, rot6, passthrough] blocks in their anchor frames.

    For each block, dp = R_anchor.T @ (p - p_anchor) and
    dR = R_anchor.T @ R. Channels after rot6 are copied, not differenced.
    """

    if block_dims < 9:
        raise ValueError(f"Pose blocks need at least 9 dims for `[xyz, rot6]`, got {block_dims}.")
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape [T, D], got {tuple(actions.shape)}.")
    total = actions.shape[-1]
    if total % block_dims:
        raise ValueError(f"Action dim {total} is not a whole number of {block_dims}-wide pose blocks.")
    if anchor.shape[-1] != total:
        raise ValueError(
            f"Anchor dim {anchor.shape[-1]} must match the action dim {total} so blocks correspond."
        )

    out = actions.clone()
    for start in range(0, total, block_dims):
        a_pos = anchor[start : start + 3]
        a_rot = continuous_6d_to_rotation_matrix(anchor[start + 3 : start + 9])
        a_rot_t = a_rot.transpose(-1, -2)
        pos = actions[:, start : start + 3]
        rot = continuous_6d_to_rotation_matrix(actions[:, start + 3 : start + 9])
        out[:, start : start + 3] = (pos - a_pos) @ a_rot_t.transpose(-1, -2)
        rel = a_rot_t @ rot
        out[:, start + 3 : start + 9] = torch.cat([rel[..., :, 0], rel[..., :, 1]], dim=-1)
    return out


def _normalize_vectors(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)


def _replace_degenerate_second_axis(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    fallback_seed = torch.zeros_like(first)
    fallback_seed[..., 0] = 1.0
    y_seed = torch.zeros_like(first)
    y_seed[..., 1] = 1.0
    near_x_axis = (first * fallback_seed).sum(dim=-1, keepdim=True).abs() > 0.9
    fallback_seed = torch.where(near_x_axis, y_seed, fallback_seed)
    fallback = torch.cross(first, fallback_seed, dim=-1)
    return torch.where(norm > 1e-8, second, fallback)


def _copy_sign(value: torch.Tensor, sign_source: torch.Tensor) -> torch.Tensor:
    sign = torch.where(sign_source < 0.0, -torch.ones_like(value), torch.ones_like(value))
    return value * sign


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


__all__ = [
    "PoseSequence",
    "axis_angle_to_quaternion",
    "continuous_6d_to_rotation_matrix",
    "normalize_quaternion",
    "pose_blocks_relative_to_anchor",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_to_axis_angle",
    "quaternion_to_continuous_6d",
    "quaternion_to_rotation_matrix",
    "reconstruct_absolute_pose_targets",
    "rotation_matrix_to_quaternion",
    "state_sequence_to_pose_sequence",
]
