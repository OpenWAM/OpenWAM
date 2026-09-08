"""Stable facade for action-target transforms and pose geometry."""

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

from .action_gripper import (
    collapse_gripper_state as collapse_gripper_state,
    extract_action_command_gripper_targets as extract_action_command_gripper_targets,
    extract_public_gripper_targets as extract_public_gripper_targets,
)
from .action_normalization import (
    _gaussian_stats,
    _normalization_bounds,
    _quantile_bounds,
    denormalize_action_targets as denormalize_action_targets,
    denormalize_joint_positions as denormalize_joint_positions,
    denormalize_joint_positions_by_limits as denormalize_joint_positions_by_limits,
    normalize_action_targets as normalize_action_targets,
    normalize_joint_positions as normalize_joint_positions,
    normalize_joint_positions_by_limits as normalize_joint_positions_by_limits,
)
from .action_pose import (
    PoseSequence as PoseSequence,
    _copy_sign,
    _normalize_vectors,
    _replace_degenerate_second_axis,
    axis_angle_to_quaternion as axis_angle_to_quaternion,
    continuous_6d_to_rotation_matrix as continuous_6d_to_rotation_matrix,
    normalize_quaternion as normalize_quaternion,
    pose_blocks_relative_to_anchor as pose_blocks_relative_to_anchor,
    quaternion_inverse as quaternion_inverse,
    quaternion_multiply as quaternion_multiply,
    quaternion_to_axis_angle as quaternion_to_axis_angle,
    quaternion_to_continuous_6d as quaternion_to_continuous_6d,
    quaternion_to_rotation_matrix as quaternion_to_rotation_matrix,
    reconstruct_absolute_pose_targets as reconstruct_absolute_pose_targets,
    rotation_matrix_to_quaternion as rotation_matrix_to_quaternion,
    state_sequence_to_pose_sequence as state_sequence_to_pose_sequence,
)
from .action_target_builders import (
    build_absolute_joint_position_targets as build_absolute_joint_position_targets,
    build_relative_pose_targets as build_relative_pose_targets,
    expected_joint_position_target_dim as expected_joint_position_target_dim,
    expected_pose_target_dim as expected_pose_target_dim,
)


_COMPATIBILITY_EXPORTS = (
    _copy_sign,
    _gaussian_stats,
    _normalization_bounds,
    _normalize_vectors,
    _quantile_bounds,
    _replace_degenerate_second_axis,
)


# Preserve the historical wildcard-import surface.
__all__ = [
    "ActionNormalizationConfig",
    "ActionNormalizationMode",
    "ActionTargetStateEncoding",
    "GripperRepresentation",
    "PoseSequence",
    "RotationRepresentation",
    "annotations",
    "axis_angle_to_quaternion",
    "build_absolute_joint_position_targets",
    "build_relative_pose_targets",
    "collapse_gripper_state",
    "continuous_6d_to_rotation_matrix",
    "dataclass",
    "denormalize_action_targets",
    "denormalize_joint_positions",
    "denormalize_joint_positions_by_limits",
    "expected_joint_position_target_dim",
    "expected_pose_target_dim",
    "extract_action_command_gripper_targets",
    "extract_public_gripper_targets",
    "normalize_action_targets",
    "normalize_joint_positions",
    "normalize_joint_positions_by_limits",
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
    "torch",
]
