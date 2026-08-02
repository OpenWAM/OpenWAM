"""LIBERO joint-position action translation and controller hooks."""

from __future__ import annotations

from typing import Any

import numpy as np


__all__ = [
    "absolute_joint_position_to_libero_joint_delta_action",
    "disable_libero_joint_position_controller_interpolator",
    "resolve_libero_joint_delta_limit",
    "resolve_libero_joint_limit_array",
    "resolve_libero_joint_scale_array",
    "set_libero_joint_position_controller_gain",
    "step_libero_absolute_joint_position_goal",
]


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


def _first_libero_robot(env: Any) -> Any:
    robots = getattr(env, "robots", None)
    if robots is None:
        inner_env = getattr(env, "env", None)
        robots = getattr(inner_env, "robots", None)
    if not robots:
        raise ValueError("LIBERO env does not expose any robot handles.")
    return robots[0]
