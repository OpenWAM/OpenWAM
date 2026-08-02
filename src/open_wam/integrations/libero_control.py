"""Compatibility facade for LIBERO observation and control contracts."""

from open_wam.integrations.libero_gripper_control import (
    gripper_command_for_substep,
    gripper_qpos_tracking_command,
    project_libero_gripper_state,
)
from open_wam.integrations.libero_joint_control import (
    absolute_joint_position_to_libero_joint_delta_action,
    disable_libero_joint_position_controller_interpolator,
    resolve_libero_joint_delta_limit,
    resolve_libero_joint_limit_array,
    resolve_libero_joint_scale_array,
    set_libero_joint_position_controller_gain,
    step_libero_absolute_joint_position_goal,
)
from open_wam.integrations.libero_observations import (
    extract_gripper_positions_from_obs,
    extract_joint_positions_from_obs,
    extract_pose_from_obs,
)
from open_wam.integrations.libero_osc_control import (
    compute_osc_pose_action,
    integrated_eef6d_target_to_osc_action,
    integrated_eef6d_target_to_osc_action_from_arrays,
    quaternion_angular_error_degrees,
    quaternion_xyzw_to_rotation_matrix,
)
from open_wam.integrations.simulator_configs import LiberoControlConfig


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
