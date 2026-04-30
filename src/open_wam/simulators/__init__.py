"""Shared simulator contracts and rollout utilities.

Importing this package is intentionally light. Torch-dependent rollout helpers
are loaded lazily so config/CLI surfaces can import simulator contracts without
pulling the model stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "EpisodeSpec": "open_wam.simulators.contracts",
    "LegacyAdapterSimulatorBackend": "open_wam.simulators.contracts",
    "SimulatorBackend": "open_wam.simulators.contracts",
    "SimulatorCapabilities": "open_wam.simulators.contracts",
    "SimulatorObservation": "open_wam.simulators.contracts",
    "SimulatorStepResult": "open_wam.simulators.contracts",
    "ensure_simulator_backend": "open_wam.simulators.contracts",
    "SimActionCommitMode": "open_wam.simulators.rollout",
    "SimPolicyInferContext": "open_wam.simulators.rollout",
    "SimRolloutResult": "open_wam.simulators.rollout",
    "SimStepResult": "open_wam.simulators.rollout",
    "build_state_history_tensor": "open_wam.simulators.rollout",
    "build_view_history_batch": "open_wam.simulators.rollout",
    "normalize_quaternion_xyzw": "open_wam.simulators.rollout",
    "run_closed_loop_sim_rollout": "open_wam.simulators.rollout",
    "source_action_from_model_action": "open_wam.simulators.rollout",
    "summarize_sim_rollout": "open_wam.simulators.rollout",
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
