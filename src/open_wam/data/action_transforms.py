from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import (
    ActionNormalizationConfig,
    ActionNormalizationMode,
    ActionTargetStateEncoding,
    GripperRepresentation,
    RotationRepresentation,
)


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


def normalize_joint_positions(
    joint_positions: torch.Tensor,
    *,
    normalization: ActionNormalizationConfig,
) -> torch.Tensor:
    """Normalize joint-position channels using a configured numeric contract."""

    if normalization.mode == ActionNormalizationMode.NONE:
        return joint_positions
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _normalization_bounds(normalization, joint_positions)
        normalized = normalize_joint_positions_by_limits(joint_positions, lower=lower, upper=upper)
    elif normalization.mode == ActionNormalizationMode.QUANTILES:
        lower, upper = _quantile_bounds(normalization, joint_positions)
        normalized = normalize_joint_positions_by_limits(joint_positions, lower=lower, upper=upper)
    else:
        raise ValueError(f"Unsupported joint-position normalization mode: {normalization.mode}")

    if normalization.clip_min is not None or normalization.clip_max is not None:
        min_value = -torch.inf if normalization.clip_min is None else float(normalization.clip_min)
        max_value = torch.inf if normalization.clip_max is None else float(normalization.clip_max)
        normalized = normalized.clamp(min=min_value, max=max_value)
    return normalized


def denormalize_joint_positions(
    normalized_joint_positions: torch.Tensor,
    *,
    normalization: ActionNormalizationConfig,
) -> torch.Tensor:
    """Invert joint-position normalization for rollout adapters."""

    if normalization.mode == ActionNormalizationMode.NONE:
        return normalized_joint_positions
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _normalization_bounds(normalization, normalized_joint_positions)
        return denormalize_joint_positions_by_limits(normalized_joint_positions, lower=lower, upper=upper)
    if normalization.mode == ActionNormalizationMode.QUANTILES:
        lower, upper = _quantile_bounds(normalization, normalized_joint_positions)
        return denormalize_joint_positions_by_limits(normalized_joint_positions, lower=lower, upper=upper)
    raise ValueError(f"Unsupported joint-position normalization mode: {normalization.mode}")


def normalize_action_targets(
    actions: torch.Tensor,
    *,
    normalization: ActionNormalizationConfig,
) -> torch.Tensor:
    """Normalize one final action-target tensor with an invertible contract."""

    if normalization.mode == ActionNormalizationMode.NONE:
        return actions
    if normalization.mode == ActionNormalizationMode.GAUSSIAN:
        mean, std = _gaussian_stats(normalization, actions)
        normalized = (actions - mean) / std.clamp_min(1e-6)
    elif normalization.mode == ActionNormalizationMode.QUANTILES:
        lower, upper = _quantile_bounds(normalization, actions)
        normalized = normalize_joint_positions_by_limits(actions, lower=lower, upper=upper)
    elif normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _normalization_bounds(normalization, actions)
        normalized = normalize_joint_positions_by_limits(actions, lower=lower, upper=upper)
    else:
        raise ValueError(f"Unsupported action-target normalization mode: {normalization.mode}")

    if normalization.clip_min is not None or normalization.clip_max is not None:
        min_value = -torch.inf if normalization.clip_min is None else float(normalization.clip_min)
        max_value = torch.inf if normalization.clip_max is None else float(normalization.clip_max)
        normalized = normalized.clamp(min=min_value, max=max_value)
    return normalized


def denormalize_action_targets(
    actions: torch.Tensor,
    *,
    normalization: ActionNormalizationConfig,
) -> torch.Tensor:
    """Invert `normalize_action_targets` for rollout adapters."""

    if normalization.mode == ActionNormalizationMode.NONE:
        return actions
    if normalization.mode == ActionNormalizationMode.GAUSSIAN:
        mean, std = _gaussian_stats(normalization, actions)
        return actions * std.clamp_min(1e-6) + mean
    if normalization.mode == ActionNormalizationMode.QUANTILES:
        lower, upper = _quantile_bounds(normalization, actions)
        return denormalize_joint_positions_by_limits(actions, lower=lower, upper=upper)
    if normalization.mode == ActionNormalizationMode.JOINT_LIMITS:
        lower, upper = _normalization_bounds(normalization, actions)
        return denormalize_joint_positions_by_limits(actions, lower=lower, upper=upper)
    raise ValueError(f"Unsupported action-target normalization mode: {normalization.mode}")


def normalize_joint_positions_by_limits(
    joint_positions: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map absolute joint positions from configured limits to roughly `[-1, 1]`."""

    center = (upper + lower) * 0.5
    scale = (upper - lower).clamp_min(1e-6) * 0.5
    return (joint_positions - center) / scale


def denormalize_joint_positions_by_limits(
    normalized_joint_positions: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map normalized joint-position channels back to physical joint units."""

    center = (upper + lower) * 0.5
    scale = (upper - lower).clamp_min(1e-6) * 0.5
    return normalized_joint_positions * scale + center


def _gaussian_stats(
    normalization: ActionNormalizationConfig,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(normalization.mean, dtype=reference.dtype, device=reference.device)
    std = torch.as_tensor(normalization.std, dtype=reference.dtype, device=reference.device)
    if mean.numel() != reference.shape[-1] or std.numel() != reference.shape[-1]:
        raise ValueError(
            "Gaussian action-target normalization stats must match the last action dimension, "
            f"got mean={mean.numel()}, std={std.numel()}, action_dim={reference.shape[-1]}."
        )
    return mean, std


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


def _normalization_bounds(
    normalization: ActionNormalizationConfig,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.tensor(normalization.lower, dtype=tensor.dtype, device=tensor.device)
    upper = torch.tensor(normalization.upper, dtype=tensor.dtype, device=tensor.device)
    if lower.numel() != tensor.shape[-1] or upper.numel() != tensor.shape[-1]:
        raise ValueError(
            "Joint-limit normalization bounds must match joint dimension, "
            f"got lower={lower.numel()}, upper={upper.numel()}, dim={tensor.shape[-1]}."
        )
    return lower, upper


def _quantile_bounds(
    normalization: ActionNormalizationConfig,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q01 = torch.tensor(normalization.q01, dtype=tensor.dtype, device=tensor.device)
    q99 = torch.tensor(normalization.q99, dtype=tensor.dtype, device=tensor.device)
    if q01.numel() != tensor.shape[-1] or q99.numel() != tensor.shape[-1]:
        raise ValueError(
            "Quantile normalization bounds must match joint dimension, "
            f"got q01={q01.numel()}, q99={q99.numel()}, dim={tensor.shape[-1]}."
        )
    return q01, q99


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
