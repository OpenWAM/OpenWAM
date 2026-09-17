"""Shared simulator contracts and rollout utilities.

Importing this package is intentionally light. Torch-dependent rollout helpers
are loaded lazily so config/CLI surfaces can import simulator contracts without
pulling the model stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from open_wam.runtime.optional_dependencies import load_optional_module


_EXPORTS: dict[str, str] = {
    "EpisodeSpec": "open_wam.simulators.contracts",
    "ObservationAdapterBackend": "open_wam.simulators.contracts",
    "SimulatorBackend": "open_wam.simulators.contracts",
    "ControlCommand": "open_wam.runtime.control",
    "SimulatorCapabilities": "open_wam.simulators.contracts",
    "SimulatorObservation": "open_wam.simulators.contracts",
    "ControlTransition": "open_wam.runtime.control",
    "SimActionCommitMode": "open_wam.simulators.rollout",
    "SimRolloutResult": "open_wam.simulators.rollout",
    "normalize_quaternion_xyzw": "open_wam.simulators.rollout",
    "run_closed_loop_sim_rollout": "open_wam.simulators.rollout",
    "summarize_sim_rollout": "open_wam.simulators.rollout",
}

_ROLLOUT_MODULE = "open_wam.simulators.rollout"

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    if module_name == _ROLLOUT_MODULE:
        module = load_optional_module(
            module_name,
            public_name=f"open_wam.simulators.{name}",
            extra="sim",
        )
    else:
        module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
