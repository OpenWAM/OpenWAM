"""Stable simulator backend and factory-registration contracts."""

from open_wam.simulators.contracts import (
    EpisodeSpec,
    SimulatorBackend,
    SimulatorCapabilities,
    SimulatorObservation,
)
from open_wam.runtime.control import ControlCommand, ControlTransition
from open_wam.simulators.registry import (
    SimulatorAdapterFactory,
    SimulatorFactoryContext,
    register_simulator_adapter,
    registered_simulator_adapters,
)

__all__ = [
    "ControlCommand",
    "EpisodeSpec",
    "SimulatorAdapterFactory",
    "SimulatorBackend",
    "SimulatorCapabilities",
    "SimulatorFactoryContext",
    "SimulatorObservation",
    "ControlTransition",
    "register_simulator_adapter",
    "registered_simulator_adapters",
]
