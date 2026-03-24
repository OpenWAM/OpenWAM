from __future__ import annotations

from dataclasses import dataclass

import torch


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
    state_encoding: str,
    rotation_representation: str,
    include_gripper: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float] | str | bool]]:
    """Convert absolute proprio state into reference-anchored pose targets.

    The output is shaped `[T, D_action]` and is suitable for the common WAM
    action contract. The first timestep is the reference pose itself, so its
    pose component is exactly zero translation and identity rotation.
    """

    if state_sequence.ndim != 2:
        raise ValueError(f"Expected state sequence with shape [T, D], got {tuple(state_sequence.shape)}.")

    if rotation_representation != "quat":
        raise ValueError(f"Unsupported rotation representation: {rotation_representation}")

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

    parts = [relative_position, relative_quaternion]
    if include_gripper:
        if absolute_pose.gripper is None:
            raise ValueError("Requested gripper targets, but the selected state encoding has no gripper channels.")
        parts.append(absolute_pose.gripper)

    targets = torch.cat(parts, dim=-1).to(dtype=torch.float32)
    mask = torch.ones_like(targets, dtype=torch.float32)
    metadata = {
        "reference_position": reference_position.tolist(),
        "reference_quaternion_xyzw": reference_quaternion.tolist(),
        "rotation_representation": rotation_representation,
        "include_gripper": include_gripper,
        "state_encoding": state_encoding,
    }
    return targets, mask, metadata


def reconstruct_absolute_pose_targets(
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
    relative_pose_targets: torch.Tensor,
) -> PoseSequence:
    """Recover absolute pose from the reference-anchored `[xyz, xyzw]` target."""

    if relative_pose_targets.ndim != 2 or relative_pose_targets.shape[-1] < 7:
        raise ValueError(
            "Expected relative pose targets with shape [T, >=7] where the first seven dims are `[xyz, xyzw]`."
        )

    rel_position = relative_pose_targets[:, :3]
    rel_quaternion = normalize_quaternion(relative_pose_targets[:, 3:7])
    abs_position = rel_position + reference_position.unsqueeze(0)
    abs_quaternion = quaternion_multiply(
        reference_quaternion.unsqueeze(0).expand_as(rel_quaternion),
        rel_quaternion,
    )
    abs_quaternion = normalize_quaternion(abs_quaternion)
    gripper = relative_pose_targets[:, 7:] if relative_pose_targets.shape[-1] > 7 else None
    return PoseSequence(position=abs_position, quaternion=abs_quaternion, gripper=gripper)


def state_sequence_to_pose_sequence(state_sequence: torch.Tensor, *, state_encoding: str) -> PoseSequence:
    """Parse a raw proprio sequence into absolute EEF pose tensors."""

    if state_encoding == "eef_pos_axisangle_gripper_2d":
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

    if state_encoding == "eef_pos_quat_gripper_1d":
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
