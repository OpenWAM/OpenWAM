"""Stable simulator backend and factory-registration contracts."""

from open_wam.simulators.contracts import (
    EpisodeSpec,
    SimulatorBackend,
    SimulatorCapabilities,
    SimulatorObservation,
    SimulatorStepResult,
)
from open_wam.simulators.registry import (
    SimulatorAdapterFactory,
    SimulatorFactoryContext,
    register_simulator_adapter,
    registered_simulator_adapters,
)

__all__ = [
    "EpisodeSpec",
    "SimulatorAdapterFactory",
    "SimulatorBackend",
    "SimulatorCapabilities",
    "SimulatorFactoryContext",
    "SimulatorObservation",
    "SimulatorStepResult",
    "register_simulator_adapter",
    "registered_simulator_adapters",
]
