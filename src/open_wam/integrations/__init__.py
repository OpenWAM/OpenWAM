"""Optional external environment integrations.

Importing `open_wam.integrations` should not eagerly import simulator-specific
modules. Attributes are loaded lazily so basic package imports work without
LIBERO, RoboTwin, or CALVIN extras installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from open_wam.runtime.optional_dependencies import load_optional_module


_EXPORTS: dict[str, str] = {
    "CalvinBenchmarkAdapter": "open_wam.integrations.calvin_env",
    "CalvinEnvConfig": "open_wam.integrations.simulator_configs",
    "OpenWAMCalvinCustomModel": "open_wam.integrations.calvin_env",
    "LiberoControlConfig": "open_wam.integrations.simulator_configs",
    "LiberoBenchmarkAdapter": "open_wam.integrations.libero_env",
    "LiberoEnvConfig": "open_wam.integrations.simulator_configs",
    "LiberoRendererConfig": "open_wam.integrations.libero_rendering",
    "LiberoTaskSpec": "open_wam.integrations.libero_tasks",
    "LiberoTrackingResult": "open_wam.integrations.libero_tracking",
    "PlannedControlStep": "open_wam.integrations.realtime_contracts",
    "PlannedFrameAction": "open_wam.integrations.realtime_contracts",
    "RealtimeSchedulerDefaults": "open_wam.integrations.realtime_contracts",
    "LIBERO_ROLLOUT_VIEW_KEYS": "open_wam.integrations.libero_rollout",
    "build_libero_state_history": "open_wam.integrations.libero_rollout",
    "activate_libero_renderer": "open_wam.integrations.libero_rendering",
    "extract_libero_rollout_observation": "open_wam.integrations.libero_rollout",
    "initialize_libero_observation_window": "open_wam.integrations.libero_rollout",
    "libero_observation_window_to_views": "open_wam.integrations.libero_rollout",
    "pose_from_libero_observation": "open_wam.integrations.libero_rollout",
    "reconstruct_libero_pose_targets": "open_wam.integrations.libero_rollout",
    "absolute_joint_position_to_libero_joint_delta_action": "open_wam.integrations.libero_joint_control",
    "build_libero_control_env": "open_wam.integrations.libero_runtime",
    "build_libero_offscreen_env": "open_wam.integrations.libero_runtime",
    "build_live_rollout_summary": "open_wam.integrations.realtime_control",
    "compute_osc_pose_action": "open_wam.integrations.libero_osc_control",
    "drop_control_steps_from": "open_wam.integrations.realtime_plan_queue",
    "drop_partial_stale_control_chunk": "open_wam.integrations.realtime_plan_queue",
    "disable_libero_joint_position_controller_interpolator": "open_wam.integrations.libero_joint_control",
    "ensure_local_libero_config": "open_wam.integrations.libero_tasks",
    "extract_gripper_positions_from_obs": "open_wam.integrations.libero_observations",
    "extract_joint_positions_from_obs": "open_wam.integrations.libero_observations",
    "extract_pose_from_obs": "open_wam.integrations.libero_observations",
    "frame_index_to_action_start": "open_wam.integrations.realtime_scheduling",
    "future_control_depth": "open_wam.integrations.realtime_plan_queue",
    "future_control_steps": "open_wam.integrations.realtime_plan_queue",
    "infer_task_local_episode_rank": "open_wam.integrations.libero_tasks",
    "integrated_eef6d_target_to_osc_action": "open_wam.integrations.libero_osc_control",
    "load_libero_benchmark_init_state_counts": "open_wam.integrations.libero_tasks",
    "load_libero_task_init_states": "open_wam.integrations.libero_tasks",
    "make_planned_frame_actions": "open_wam.integrations.realtime_control",
    "merge_future_control_steps": "open_wam.integrations.realtime_plan_queue",
    "merge_future_frame_actions": "open_wam.integrations.realtime_plan_queue",
    "missing_control_action_indices": "open_wam.integrations.realtime_plan_queue",
    "planned_frame_actions_to_control_steps": "open_wam.integrations.realtime_control",
    "resolve_libero_benchmark_tasks": "open_wam.integrations.libero_tasks",
    "resolve_libero_joint_delta_limit": "open_wam.integrations.libero_joint_control",
    "resolve_libero_renderer_config": "open_wam.integrations.libero_rendering",
    "resolve_libero_task": "open_wam.integrations.libero_tasks",
    "resolve_libero_task_by_id": "open_wam.integrations.libero_tasks",
    "required_control_action_indices": "open_wam.integrations.realtime_plan_queue",
    "resolve_realtime_planner_mode": "open_wam.integrations.realtime_scheduling",
    "resolve_realtime_scheduler_defaults": "open_wam.integrations.realtime_scheduling",
    "select_realtime_planner_job": "open_wam.integrations.realtime_scheduling",
    "set_libero_joint_position_controller_gain": "open_wam.integrations.libero_joint_control",
    "step_libero_absolute_joint_position_goal": "open_wam.integrations.libero_joint_control",
    "should_submit_frame_grouped_planner": "open_wam.integrations.realtime_scheduling",
    "should_submit_realtime_planner_job": "open_wam.integrations.realtime_scheduling",
    "should_submit_sequence_planner": "open_wam.integrations.realtime_scheduling",
    "summarize_scalars": "open_wam.integrations.realtime_control",
    "track_relative_targets_in_libero_env": "open_wam.integrations.libero_tracking",
    "RobotwinBenchmarkAdapter": "open_wam.integrations.robotwin_env",
    "RobotwinEnvConfig": "open_wam.integrations.simulator_configs",
}

_OPTIONAL_RUNTIME_MODULES = {
    "open_wam.integrations.calvin_env",
    "open_wam.integrations.libero_control",
    "open_wam.integrations.libero_env",
    "open_wam.integrations.libero_gripper_control",
    "open_wam.integrations.libero_joint_control",
    "open_wam.integrations.libero_observations",
    "open_wam.integrations.libero_osc_control",
    "open_wam.integrations.libero_rollout",
    "open_wam.integrations.libero_tracking",
    "open_wam.integrations.realtime_control",
    "open_wam.integrations.robotwin_env",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    if module_name in _OPTIONAL_RUNTIME_MODULES:
        module = load_optional_module(
            module_name,
            public_name=f"open_wam.integrations.{name}",
            extra="sim",
        )
    else:
        module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
