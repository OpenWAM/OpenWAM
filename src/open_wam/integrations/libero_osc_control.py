"""LIBERO OSC pose-target translation and rotation geometry."""

from __future__ import annotations

import numpy as np
import torch

from open_wam.data.action_pose import (
    PoseSequence,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_axis_angle,
    rotation_matrix_to_quaternion,
)
from open_wam.integrations.libero_gripper_control import (
    project_libero_gripper_state,
)
from open_wam.integrations.simulator_configs import LiberoControlConfig


__all__ = [
    "compute_osc_pose_action",
    "integrated_eef6d_target_to_osc_action",
    "integrated_eef6d_target_to_osc_action_from_arrays",
    "quaternion_angular_error_degrees",
    "quaternion_xyzw_to_rotation_matrix",
]


def compute_osc_pose_action(
    *,
    current_pose: PoseSequence,
    desired_pose: PoseSequence,
    control_config: LiberoControlConfig,
    gripper_representation: str,
) -> np.ndarray:
    """Convert one desired absolute pose into a normalized OSC_POSE action."""

    position_error = desired_pose.position - current_pose.position
    delta_quaternion = quaternion_multiply(
        desired_pose.quaternion.unsqueeze(0),
        quaternion_inverse(current_pose.quaternion).unsqueeze(0),
    )[0]
    delta_axis_angle = quaternion_to_axis_angle(
        normalize_quaternion(delta_quaternion.unsqueeze(0))
    )[0]
    position_command = torch.clamp(
        position_error / control_config.max_pos_delta_m,
        min=-1.0,
        max=1.0,
    )
    rotation_command = torch.clamp(
        delta_axis_angle / control_config.max_rot_delta_rad,
        min=-1.0,
        max=1.0,
    )

    if desired_pose.gripper is None:
        gripper_command = torch.tensor([0.0], dtype=torch.float32)
    elif gripper_representation == "action_command":
        gripper_command = (
            desired_pose.gripper[0:1]
            .clamp(min=-1.0, max=1.0)
            .to(dtype=torch.float32)
        )
    elif current_pose.gripper is None:
        gripper_command = torch.tensor([0.0], dtype=torch.float32)
    else:
        current_public = project_libero_gripper_state(
            current_pose.gripper,
            gripper_representation=gripper_representation,
        )
        if gripper_representation == "all_channels":
            current_value = current_pose.gripper[0] - current_pose.gripper[1]
            desired_value = desired_pose.gripper[0] - desired_pose.gripper[1]
            open_threshold = control_config.gripper_open_threshold
            close_threshold = control_config.gripper_close_threshold
            tolerance = control_config.gripper_position_tolerance
        elif gripper_representation == "first_channel":
            current_value = current_public[0]
            desired_value = desired_pose.gripper[0]
            open_threshold = control_config.gripper_open_threshold * 0.5
            close_threshold = control_config.gripper_close_threshold * 0.5
            tolerance = control_config.gripper_position_tolerance * 0.5
        else:
            raise ValueError(
                "Unsupported gripper representation: "
                f"{gripper_representation}"
            )

        error_value = desired_value - current_value
        if desired_value >= open_threshold:
            gripper_command = torch.tensor([-1.0], dtype=torch.float32)
        elif desired_value <= close_threshold:
            gripper_command = torch.tensor([1.0], dtype=torch.float32)
        elif torch.abs(error_value) <= tolerance:
            gripper_command = torch.tensor([0.0], dtype=torch.float32)
        else:
            gripper_command = torch.clamp(
                -error_value / control_config.max_gripper_delta,
                min=-1.0,
                max=1.0,
            ).reshape(1)

    action = torch.cat(
        [position_command, rotation_command, gripper_command],
        dim=0,
    )
    return action.detach().cpu().numpy().astype(np.float32)


def integrated_eef6d_target_to_osc_action(
    *,
    previous_target: PoseSequence,
    target: np.ndarray,
    position_scale: float,
    rotation_scale: float,
) -> tuple[np.ndarray, PoseSequence]:
    """Recover one OSC action from a pseudo-absolute EEF-6D target."""

    action, target_position, target_rotation = (
        integrated_eef6d_target_to_osc_action_from_arrays(
            previous_position=previous_target.position.detach().cpu().numpy(),
            previous_rotation_matrix=quaternion_xyzw_to_rotation_matrix(
                previous_target.quaternion.detach().cpu().numpy()
            ),
            target=target,
            position_scale=position_scale,
            rotation_scale=rotation_scale,
        )
    )
    next_target = PoseSequence(
        position=torch.as_tensor(target_position, dtype=torch.float32),
        quaternion=rotation_matrix_to_quaternion(
            torch.as_tensor(target_rotation, dtype=torch.float32).unsqueeze(0)
        )[0],
        gripper=torch.as_tensor([float(action[6])], dtype=torch.float32),
    )
    return action, next_target


def integrated_eef6d_target_to_osc_action_from_arrays(
    *,
    previous_position: np.ndarray,
    previous_rotation_matrix: np.ndarray,
    target: np.ndarray,
    position_scale: float,
    rotation_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Array-only implementation shared by stateful simulator adapters."""

    payload = np.asarray(target, dtype=np.float32).reshape(-1)
    if payload.shape[0] < 10:
        raise ValueError(
            "Expected integrated EEF6D target with at least 10 dims, got "
            f"{payload.shape[0]}."
        )
    if (
        abs(float(position_scale)) <= 1e-12
        or abs(float(rotation_scale)) <= 1e-12
    ):
        raise ValueError(
            "Integrated EEF position and rotation scales must be nonzero."
        )
    target_position = payload[0:3].astype(np.float32, copy=True)
    target_rotation = _continuous_6d_to_rotation_matrix_np(
        payload[3:9]
    ).astype(np.float32, copy=False)
    delta_axis_angle = _relative_rotation_matrix_to_axis_angle_np(
        target_rotation,
        np.asarray(previous_rotation_matrix, dtype=np.float32),
    )
    position_command = np.clip(
        (
            target_position
            - np.asarray(previous_position, dtype=np.float32)
        )
        / float(position_scale),
        -1.0,
        1.0,
    )
    rotation_command = np.clip(
        delta_axis_angle / float(rotation_scale),
        -1.0,
        1.0,
    )
    action = np.concatenate(
        [
            position_command.astype(np.float32, copy=False),
            rotation_command.astype(np.float32, copy=False),
            np.asarray(
                [float(np.clip(payload[9], -1.0, 1.0))],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    return (
        action.astype(np.float32, copy=False),
        target_position,
        target_rotation.astype(np.float32, copy=False),
    )


def quaternion_xyzw_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert one normalized-or-unnormalized XYZW quaternion to a matrix."""

    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    x, y, z, w = quat
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float32,
    )


def quaternion_angular_error_degrees(
    lhs_xyzw: torch.Tensor,
    rhs_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Return unsigned geodesic quaternion error in degrees."""

    lhs = normalize_quaternion(lhs_xyzw)
    rhs = normalize_quaternion(rhs_xyzw)
    dot = (lhs * rhs).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.arccos(dot))


def _continuous_6d_to_rotation_matrix_np(
    rotation_6d: np.ndarray,
) -> np.ndarray:
    rot = np.asarray(rotation_6d, dtype=np.float64).reshape(6)
    first = _normalize_np(rot[0:3])
    second_raw = rot[3:6] - float(np.dot(first, rot[3:6])) * first
    if float(np.linalg.norm(second_raw)) <= 1e-8:
        seed = np.asarray(
            [0.0, 1.0, 0.0]
            if abs(float(first[0])) > 0.9
            else [1.0, 0.0, 0.0],
            dtype=np.float64,
        )
        second_raw = np.cross(first, seed)
    second = _normalize_np(second_raw)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1).astype(np.float32)


def _normalize_np(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def _relative_rotation_matrix_to_axis_angle_np(
    target: np.ndarray,
    previous: np.ndarray,
) -> np.ndarray:
    delta = (
        np.asarray(target, dtype=np.float64)
        @ np.asarray(previous, dtype=np.float64).T
    )
    return _rotation_matrix_to_axis_angle_np(delta)


def _rotation_matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(mat))
    angle = float(
        np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    )
    vee = np.asarray(
        [
            mat[2, 1] - mat[1, 2],
            mat[0, 2] - mat[2, 0],
            mat[1, 0] - mat[0, 1],
        ],
        dtype=np.float64,
    )
    if angle <= 1e-6:
        return (0.5 * vee).astype(np.float32)
    return (
        vee / max(2.0 * float(np.sin(angle)), 1e-12) * angle
    ).astype(np.float32)
