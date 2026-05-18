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
    "LiberoControlConfig": "open_wam.integrations.libero_env",
    "LiberoBenchmarkAdapter": "open_wam.integrations.libero_env",
    "LiberoEnvConfig": "open_wam.integrations.libero_env",
    "LiberoTaskSpec": "open_wam.integrations.libero_env",
    "LiberoTrackingResult": "open_wam.integrations.libero_env",
    "absolute_joint_position_to_libero_joint_delta_action": "open_wam.integrations.libero_env",
    "build_libero_control_env": "open_wam.integrations.libero_env",
    "build_libero_offscreen_env": "open_wam.integrations.libero_env",
    "compute_osc_pose_action": "open_wam.integrations.libero_env",
    "disable_libero_joint_position_controller_interpolator": "open_wam.integrations.libero_env",
    "ensure_local_libero_config": "open_wam.integrations.libero_env",
    "extract_gripper_positions_from_obs": "open_wam.integrations.libero_env",
    "extract_joint_positions_from_obs": "open_wam.integrations.libero_env",
    "extract_pose_from_obs": "open_wam.integrations.libero_env",
    "infer_task_local_episode_rank": "open_wam.integrations.libero_env",
    "integrated_eef6d_target_to_osc_action": "open_wam.integrations.libero_env",
    "load_libero_task_init_states": "open_wam.integrations.libero_env",
    "resolve_libero_joint_delta_limit": "open_wam.integrations.libero_env",
    "resolve_libero_task": "open_wam.integrations.libero_env",
    "resolve_libero_task_by_id": "open_wam.integrations.libero_env",
    "set_libero_joint_position_controller_gain": "open_wam.integrations.libero_env",
    "step_libero_absolute_joint_position_goal": "open_wam.integrations.libero_env",
    "track_relative_targets_in_libero_env": "open_wam.integrations.libero_env",
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
