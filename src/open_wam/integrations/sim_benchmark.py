"""Compatibility exports for the shared simulator runtime.

New code should import from `open_wam.simulators`. This module remains so
existing scripts/tests that imported `open_wam.integrations.sim_benchmark`
continue to work.
"""

from __future__ import annotations

from open_wam.simulators import (  # noqa: F401
    EpisodeSpec,
    LegacyAdapterSimulatorBackend,
    SimActionCommitMode,
    SimPolicyInferContext,
    SimRolloutResult,
    SimStepResult,
    SimulatorBackend,
    SimulatorCapabilities,
    SimulatorObservation,
    SimulatorStepResult,
    build_state_history_tensor,
    build_view_history_batch,
    ensure_simulator_backend,
    normalize_quaternion_xyzw,
    run_closed_loop_sim_rollout,
    source_action_from_model_action,
    summarize_sim_rollout,
)

SimBenchmarkAdapter = SimulatorBackend

__all__ = [
    "EpisodeSpec",
    "LegacyAdapterSimulatorBackend",
    "SimActionCommitMode",
    "SimBenchmarkAdapter",
    "SimPolicyInferContext",
    "SimRolloutResult",
    "SimStepResult",
    "SimulatorBackend",
    "SimulatorCapabilities",
    "SimulatorObservation",
    "SimulatorStepResult",
    "build_state_history_tensor",
    "build_view_history_batch",
    "ensure_simulator_backend",
    "normalize_quaternion_xyzw",
    "run_closed_loop_sim_rollout",
    "source_action_from_model_action",
    "summarize_sim_rollout",
]
