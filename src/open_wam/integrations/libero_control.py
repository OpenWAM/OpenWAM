"""LIBERO observation, controller, and action-translation semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from open_wam.data.action_transforms import (
    PoseSequence,
    collapse_gripper_state,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_axis_angle,
    rotation_matrix_to_quaternion,
)


__all__ = [
    "LiberoControlConfig",
    "absolute_joint_position_to_libero_joint_delta_action",
    "compute_osc_pose_action",
    "disable_libero_joint_position_controller_interpolator",
    "extract_gripper_positions_from_obs",
    "extract_joint_positions_from_obs",
    "extract_pose_from_obs",
    "gripper_command_for_substep",
    "gripper_qpos_tracking_command",
    "integrated_eef6d_target_to_osc_action",
    "integrated_eef6d_target_to_osc_action_from_arrays",
    "project_libero_gripper_state",
    "quaternion_angular_error_degrees",
    "quaternion_xyzw_to_rotation_matrix",
    "resolve_libero_joint_delta_limit",
    "resolve_libero_joint_limit_array",
    "resolve_libero_joint_scale_array",
    "set_libero_joint_position_controller_gain",
    "step_libero_absolute_joint_position_goal",
]


@dataclass(frozen=True)
class LiberoControlConfig:
    """Closed-loop gains for converting public targets to OSC actions.

    `OSC_POSE` expects `[dx, dy, dz, dax, day, daz, gripper]`. The first six
    channels are normalized and internally scaled by robosuite to +/- 0.05 m
    and +/- 0.5 rad. Public WAM targets are reference-relative absolute EEF
    targets, so replay reconstructs an absolute target, computes world-frame
    pose error, and normalizes that error for the controller.
    """

    max_pos_delta_m: float = 0.05
    max_rot_delta_rad: float = 0.5
    max_gripper_delta: float = 0.005
    control_substeps_per_target: int = 8
    env_control_hz: int = 20
    action_command_delay_steps: int = 1
    gripper_open_threshold: float = 0.060
    gripper_close_threshold: float = 0.030
    gripper_position_tolerance: float = 0.002


def extract_pose_from_obs(obs: dict[str, Any]) -> PoseSequence:
    """Parse a LIBERO observation into the common pose contract."""

    quaternion_xyzw = torch.tensor(
        obs["robot0_eef_quat"],
        dtype=torch.float32,
    )
    return PoseSequence(
        position=torch.tensor(obs["robot0_eef_pos"], dtype=torch.float32),
        quaternion=normalize_quaternion(quaternion_xyzw),
        gripper=torch.tensor(
            obs["robot0_gripper_qpos"],
            dtype=torch.float32,
        ),
    )


def extract_joint_positions_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Extract Panda arm qpos from a LIBERO observation."""

    if "robot0_joint_pos" not in obs:
        raise KeyError("LIBERO observation does not expose `robot0_joint_pos`.")
    joint_positions = np.asarray(
        obs["robot0_joint_pos"],
        dtype=np.float32,
    ).reshape(-1)
    if joint_positions.size == 0:
        raise ValueError("LIBERO `robot0_joint_pos` is empty.")
    return joint_positions


def extract_gripper_positions_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Extract Panda gripper qpos from a LIBERO observation."""

    if "robot0_gripper_qpos" not in obs:
        raise KeyError(
            "LIBERO observation does not expose `robot0_gripper_qpos`."
        )
    values = np.asarray(
        obs["robot0_gripper_qpos"],
        dtype=np.float32,
    ).reshape(-1)
    if values.size == 0:
        raise ValueError("LIBERO `robot0_gripper_qpos` is empty.")
    return values


def resolve_libero_joint_delta_limit(
    env: Any,
    *,
    fallback: float | tuple[float, ...] = 0.05,
    joint_dim: int = 7,
) -> np.ndarray:
    """Infer JOINT_POSITION delta scaling from the controller config."""

    fallback_limit = resolve_libero_joint_limit_array(
        fallback,
        joint_dim=joint_dim,
    )
    robots = getattr(getattr(env, "env", env), "robots", None)
    if not robots:
        return fallback_limit
    controller = getattr(robots[0], "controller", None)
    if controller is None:
        return fallback_limit
    output_max = getattr(controller, "output_max", None)
    output_min = getattr(controller, "output_min", None)
    if output_max is None:
        return fallback_limit
    max_values = np.asarray(output_max, dtype=np.float32).reshape(-1)
    if max_values.size < joint_dim:
        return fallback_limit
    if output_min is not None:
        min_values = np.asarray(output_min, dtype=np.float32).reshape(-1)
        if min_values.size >= joint_dim:
            max_values = np.maximum(
                np.abs(max_values[:joint_dim]),
                np.abs(min_values[:joint_dim]),
            )
        else:
            max_values = np.abs(max_values[:joint_dim])
    else:
        max_values = np.abs(max_values[:joint_dim])
    if np.any(max_values <= 0.0):
        return fallback_limit
    return max_values.astype(np.float32)


def absolute_joint_position_to_libero_joint_delta_action(
    *,
    target_joint_positions: np.ndarray,
    current_joint_positions: np.ndarray,
    gripper_command: float = 0.0,
    joint_delta_limit_rad: float | tuple[float, ...] | np.ndarray = 0.05,
) -> np.ndarray:
    """Convert absolute qpos targets to normalized JOINT_POSITION actions."""

    target = np.asarray(target_joint_positions, dtype=np.float32).reshape(-1)
    current = np.asarray(current_joint_positions, dtype=np.float32).reshape(-1)
    if target.shape != current.shape:
        raise ValueError(
            "Target/current joint shapes must match, got "
            f"{target.shape} and {current.shape}."
        )
    limits = resolve_libero_joint_limit_array(
        joint_delta_limit_rad,
        joint_dim=target.shape[0],
    )
    arm_action = np.clip((target - current) / limits, -1.0, 1.0)
    gripper = np.asarray(
        [float(np.clip(gripper_command, -1.0, 1.0))],
        dtype=np.float32,
    )
    return np.concatenate(
        [arm_action.astype(np.float32), gripper],
        axis=0,
    )


def step_libero_absolute_joint_position_goal(
    env: Any,
    *,
    target_joint_positions: np.ndarray,
    gripper_command: float = 0.0,
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    """Step a JOINT_POSITION env with an absolute joint-position goal."""

    target = np.asarray(target_joint_positions, dtype=np.float32).reshape(-1)
    robot = _first_libero_robot(env)
    controller = getattr(robot, "controller", None)
    if controller is None or not hasattr(controller, "set_goal"):
        raise ValueError(
            "LIBERO env does not expose a robosuite arm controller with "
            "`set_goal`."
        )
    control_dim = int(getattr(controller, "control_dim", target.shape[0]))
    if control_dim < target.shape[0]:
        raise ValueError(
            f"Controller control_dim={control_dim} is smaller than target "
            f"joint dim={target.shape[0]}."
        )

    action_dim = int(getattr(robot, "action_dim", control_dim + 1))
    action = np.zeros(action_dim, dtype=np.float32)
    action[:control_dim] = 0.0
    if action_dim > control_dim:
        action[control_dim:] = float(
            np.clip(gripper_command, -1.0, 1.0)
        )

    original_set_goal = controller.set_goal

    def _set_absolute_goal(
        action_arg: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del action_arg, args, kwargs
        return original_set_goal(
            np.zeros(control_dim, dtype=np.float32),
            set_qpos=target,
        )

    controller.set_goal = _set_absolute_goal
    try:
        return env.step(action)
    finally:
        controller.set_goal = original_set_goal


def set_libero_joint_position_controller_gain(env: Any, *, kp: float) -> None:
    """Override JOINT_POSITION gains for deterministic goal tracking."""

    controller = getattr(_first_libero_robot(env), "controller", None)
    if controller is None:
        raise ValueError(
            "LIBERO env robot does not expose a controller for gain override."
        )
    joint_dim = int(getattr(controller, "control_dim", 7))
    controller.kp = np.full(joint_dim, float(kp), dtype=np.float64)
    controller.kd = 2.0 * np.sqrt(controller.kp)


def disable_libero_joint_position_controller_interpolator(env: Any) -> None:
    """Disable the JOINT_POSITION interpolator for exact goal tracking."""

    controller = getattr(_first_libero_robot(env), "controller", None)
    if controller is None:
        raise ValueError(
            "LIBERO env robot does not expose a controller for interpolator "
            "override."
        )
    controller.interpolator = None


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


def resolve_libero_joint_limit_array(
    value: float | tuple[float, ...] | np.ndarray,
    *,
    joint_dim: int,
) -> np.ndarray:
    """Resolve positive per-joint delta limits from a scalar or vector."""

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.full(joint_dim, float(array[0]), dtype=np.float32)
    if array.size != joint_dim:
        raise ValueError(
            f"Expected {joint_dim} joint delta limits, got {array.size}."
        )
    if np.any(array <= 0.0):
        raise ValueError("Joint delta limits must be positive.")
    return array.astype(np.float32)


def resolve_libero_joint_scale_array(
    value: float | tuple[float, ...] | np.ndarray,
    *,
    joint_dim: int,
) -> np.ndarray:
    """Resolve nonzero per-joint integration scales from scalar or vector."""

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.full(joint_dim, float(array[0]), dtype=np.float32)
    if array.size != joint_dim:
        raise ValueError(
            f"Expected {joint_dim} joint integration scales, got {array.size}."
        )
    if np.any(np.isclose(array, 0.0)):
        raise ValueError("Joint integration scales must be nonzero.")
    return array.astype(np.float32)


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


def _first_libero_robot(env: Any) -> Any:
    robots = getattr(env, "robots", None)
    if robots is None:
        inner_env = getattr(env, "env", None)
        robots = getattr(inner_env, "robots", None)
    if not robots:
        raise ValueError("LIBERO env does not expose any robot handles.")
    return robots[0]


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
