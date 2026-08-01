"""Optional external environment integrations.

Importing `open_wam.integrations` should not eagerly import simulator-specific
modules. Attributes are loaded lazily so basic package imports work without
LIBERO, RoboTwin, or CALVIN extras installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "CalvinBenchmarkAdapter": "open_wam.integrations.calvin_env",
    "CalvinEnvConfig": "open_wam.integrations.calvin_env",
    "OpenWAMCalvinCustomModel": "open_wam.integrations.calvin_env",
    "LiberoControlConfig": "open_wam.integrations.libero_control",
    "LiberoBenchmarkAdapter": "open_wam.integrations.libero_env",
    "LiberoEnvConfig": "open_wam.integrations.libero_env",
    "LiberoTaskSpec": "open_wam.integrations.libero_tasks",
    "LiberoTrackingResult": "open_wam.integrations.libero_tracking",
    "PlannedControlStep": "open_wam.integrations.realtime_control",
    "PlannedFrameAction": "open_wam.integrations.realtime_control",
    "RealtimeSchedulerDefaults": "open_wam.integrations.realtime_control",
    "LIBERO_ROLLOUT_VIEW_KEYS": "open_wam.integrations.libero_rollout",
    "build_libero_state_history": "open_wam.integrations.libero_rollout",
    "extract_libero_rollout_observation": "open_wam.integrations.libero_rollout",
    "initialize_libero_observation_window": "open_wam.integrations.libero_rollout",
    "libero_observation_window_to_views": "open_wam.integrations.libero_rollout",
    "pose_from_libero_observation": "open_wam.integrations.libero_rollout",
    "reconstruct_libero_pose_targets": "open_wam.integrations.libero_rollout",
    "absolute_joint_position_to_libero_joint_delta_action": "open_wam.integrations.libero_control",
    "build_libero_control_env": "open_wam.integrations.libero_runtime",
    "build_libero_offscreen_env": "open_wam.integrations.libero_runtime",
    "build_live_rollout_summary": "open_wam.integrations.realtime_control",
    "compute_osc_pose_action": "open_wam.integrations.libero_control",
    "drop_control_steps_from": "open_wam.integrations.realtime_control",
    "drop_partial_stale_control_chunk": "open_wam.integrations.realtime_control",
    "disable_libero_joint_position_controller_interpolator": "open_wam.integrations.libero_control",
    "ensure_local_libero_config": "open_wam.integrations.libero_tasks",
    "extract_gripper_positions_from_obs": "open_wam.integrations.libero_control",
    "extract_joint_positions_from_obs": "open_wam.integrations.libero_control",
    "extract_pose_from_obs": "open_wam.integrations.libero_control",
    "frame_index_to_action_start": "open_wam.integrations.realtime_control",
    "future_control_depth": "open_wam.integrations.realtime_control",
    "future_control_steps": "open_wam.integrations.realtime_control",
    "infer_task_local_episode_rank": "open_wam.integrations.libero_tasks",
    "integrated_eef6d_target_to_osc_action": "open_wam.integrations.libero_control",
    "load_libero_task_init_states": "open_wam.integrations.libero_tasks",
    "make_planned_frame_actions": "open_wam.integrations.realtime_control",
    "merge_future_control_steps": "open_wam.integrations.realtime_control",
    "merge_future_frame_actions": "open_wam.integrations.realtime_control",
    "missing_control_action_indices": "open_wam.integrations.realtime_control",
    "planned_frame_actions_to_control_steps": "open_wam.integrations.realtime_control",
    "resolve_libero_joint_delta_limit": "open_wam.integrations.libero_control",
    "resolve_libero_task": "open_wam.integrations.libero_tasks",
    "resolve_libero_task_by_id": "open_wam.integrations.libero_tasks",
    "required_control_action_indices": "open_wam.integrations.realtime_control",
    "resolve_realtime_planner_mode": "open_wam.integrations.realtime_control",
    "resolve_realtime_scheduler_defaults": "open_wam.integrations.realtime_control",
    "select_realtime_planner_job": "open_wam.integrations.realtime_control",
    "set_libero_joint_position_controller_gain": "open_wam.integrations.libero_control",
    "step_libero_absolute_joint_position_goal": "open_wam.integrations.libero_control",
    "should_submit_frame_grouped_planner": "open_wam.integrations.realtime_control",
    "should_submit_realtime_planner_job": "open_wam.integrations.realtime_control",
    "should_submit_sequence_planner": "open_wam.integrations.realtime_control",
    "summarize_scalars": "open_wam.integrations.realtime_control",
    "track_relative_targets_in_libero_env": "open_wam.integrations.libero_tracking",
    "RobotwinBenchmarkAdapter": "open_wam.integrations.robotwin_env",
    "RobotwinEnvConfig": "open_wam.integrations.robotwin_env",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
