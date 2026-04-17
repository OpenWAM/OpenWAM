from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar


K = TypeVar("K")
T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry(Generic[K, T]):
    key: K
    value: T
    description: str | None = None


class Registry(Generic[K, T]):
    """Small typed registry for extension points.

    Registries are intentionally simple: they map public keys to builders or
    classes, reject accidental replacement by default, and expose stable keys
    for docs/tests.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: dict[K, RegistryEntry[K, T]] = {}

    def register(
        self,
        key: K,
        value: T,
        *,
        description: str | None = None,
        replace: bool = False,
    ) -> None:
        if key in self._entries and not replace:
            raise ValueError(f"{self.name} registry already has an entry for {key!r}.")
        self._entries[key] = RegistryEntry(key=key, value=value, description=description)

    def get(self, key: K) -> T | None:
        entry = self._entries.get(key)
        return None if entry is None else entry.value

    def require(self, key: K) -> T:
        value = self.get(key)
        if value is None:
            supported = ", ".join(str(item) for item in self.keys())
            raise KeyError(f"Unsupported {self.name} registry key {key!r}. Registered keys: {supported}")
        return value

    def keys(self) -> tuple[K, ...]:
        return tuple(self._entries)

    def entries(self) -> tuple[RegistryEntry[K, T], ...]:
        return tuple(self._entries.values())


Builder = Callable[..., T]


class BuilderRegistry(Registry[K, Builder[T]]):
    """Registry whose values are callables."""

    def build(self, key: K, *args, **kwargs) -> T:
        return self.require(key)(*args, **kwargs)


def registry_keys(registry: Registry[K, object]) -> Iterable[K]:
    """Return registry keys through a function useful in tests/docs."""

    return registry.keys()
