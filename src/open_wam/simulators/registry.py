"""Application-extensible simulator construction registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from open_wam.registry import Registry

from .contracts import SimulatorBackend


@dataclass(frozen=True)
class SimulatorFactoryContext:
    """Dependency-light inputs supplied to one simulator adapter factory."""

    benchmark: str
    options: Mapping[str, str]
    local_paths: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.benchmark or self.benchmark != self.benchmark.strip():
            raise ValueError(
                "Simulator benchmark identifiers must be non-empty and have no "
                "surrounding whitespace."
            )
        object.__setattr__(
            self,
            "options",
            _immutable_string_mapping("options", self.options),
        )
        object.__setattr__(
            self,
            "local_paths",
            _immutable_string_mapping("local_paths", self.local_paths),
        )


SimulatorAdapterFactory = Callable[[SimulatorFactoryContext], SimulatorBackend]


SIMULATOR_ADAPTER_FACTORIES = Registry[str, SimulatorAdapterFactory](
    "simulator adapter factory"
)


def _immutable_string_mapping(
    name: str,
    values: Mapping[str, str],
) -> Mapping[str, str]:
    copied = dict(values)
    invalid = [
        key
        for key, value in copied.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid:
        raise TypeError(
            f"Simulator factory {name} must contain only string keys and values."
        )
    return MappingProxyType(copied)


def register_simulator_adapter(
    benchmark: str,
    factory: SimulatorAdapterFactory,
    *,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Register one application-owned simulator adapter factory."""

    normalized = benchmark.strip()
    if not normalized or normalized != benchmark:
        raise ValueError(
            "Simulator benchmark identifiers must be non-empty and have no "
            "surrounding whitespace."
        )
    if not callable(factory):
        raise TypeError("Simulator adapter factory must be callable.")
    SIMULATOR_ADAPTER_FACTORIES.register(
        normalized,
        factory,
        description=description,
        replace=replace,
    )


def build_simulator_adapter(context: SimulatorFactoryContext) -> SimulatorBackend:
    """Build the adapter selected by ``context.benchmark``."""

    return SIMULATOR_ADAPTER_FACTORIES.require(context.benchmark)(context)


def registered_simulator_adapters() -> tuple[str, ...]:
    """Return simulator identifiers registered in this process."""

    return SIMULATOR_ADAPTER_FACTORIES.keys()


__all__ = [
    "SIMULATOR_ADAPTER_FACTORIES",
    "SimulatorAdapterFactory",
    "SimulatorFactoryContext",
    "build_simulator_adapter",
    "register_simulator_adapter",
    "registered_simulator_adapters",
]
