"""Optional external environment integrations.

Importing `open_wam.integrations` should not eagerly import simulator-specific
modules. Attributes are loaded lazily so basic package imports work without
LIBERO, RoboTwin, or CALVIN extras installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "BenchmarkActionSchema": "open_wam.integrations.contracts",
    "BenchmarkAdapterContract": "open_wam.integrations.contracts",
    "BenchmarkObservationSchema": "open_wam.integrations.contracts",
    "CalvinBenchmarkAdapter": "open_wam.integrations.calvin_env",
    "CalvinEnvConfig": "open_wam.integrations.calvin_env",
    "OpenWAMCalvinCustomModel": "open_wam.integrations.calvin_env",
    "LiberoControlConfig": "open_wam.integrations.libero_control",
    "LiberoBenchmarkAdapter": "open_wam.integrations.libero_env",
    "LiberoEnvConfig": "open_wam.integrations.libero_env",
    "LiberoTaskSpec": "open_wam.integrations.libero_tasks",
    "LiberoTrackingResult": "open_wam.integrations.libero_tracking",
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
    "compute_osc_pose_action": "open_wam.integrations.libero_control",
    "disable_libero_joint_position_controller_interpolator": "open_wam.integrations.libero_control",
    "ensure_local_libero_config": "open_wam.integrations.libero_tasks",
    "extract_gripper_positions_from_obs": "open_wam.integrations.libero_control",
    "extract_joint_positions_from_obs": "open_wam.integrations.libero_control",
    "extract_pose_from_obs": "open_wam.integrations.libero_control",
    "infer_task_local_episode_rank": "open_wam.integrations.libero_tasks",
    "integrated_eef6d_target_to_osc_action": "open_wam.integrations.libero_control",
    "load_libero_task_init_states": "open_wam.integrations.libero_tasks",
    "resolve_libero_joint_delta_limit": "open_wam.integrations.libero_control",
    "resolve_libero_task": "open_wam.integrations.libero_tasks",
    "resolve_libero_task_by_id": "open_wam.integrations.libero_tasks",
    "set_libero_joint_position_controller_gain": "open_wam.integrations.libero_control",
    "step_libero_absolute_joint_position_goal": "open_wam.integrations.libero_control",
    "track_relative_targets_in_libero_env": "open_wam.integrations.libero_tracking",
    "RobotwinBenchmarkAdapter": "open_wam.integrations.robotwin_env",
    "RobotwinEnvConfig": "open_wam.integrations.robotwin_env",
    "SimBenchmarkAdapter": "open_wam.integrations.sim_benchmark",
    "SimRolloutResult": "open_wam.integrations.sim_benchmark",
    "SimStepResult": "open_wam.integrations.sim_benchmark",
    "SimulatorBackend": "open_wam.simulators",
    "SimulatorCapabilities": "open_wam.simulators",
    "SimulatorObservation": "open_wam.simulators",
    "SimulatorStepResult": "open_wam.simulators",
    "build_state_history_tensor": "open_wam.integrations.sim_benchmark",
    "build_view_history_batch": "open_wam.integrations.sim_benchmark",
    "run_closed_loop_sim_rollout": "open_wam.integrations.sim_benchmark",
    "source_action_from_model_action": "open_wam.integrations.sim_benchmark",
    "summarize_sim_rollout": "open_wam.integrations.sim_benchmark",
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
