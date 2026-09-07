"""Explicit loading for application-owned OpenWAM extensions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from threading import RLock


DEFAULT_EXTENSION_HOOK = "register_open_wam"


@dataclass(frozen=True)
class LoadedExtension:
    """One extension module and registration hook loaded in this process."""

    module_name: str
    hook_name: str

    @property
    def spec(self) -> str:
        return f"{self.module_name}:{self.hook_name}"


_LOAD_LOCK = RLock()
_LOADED_EXTENSIONS: dict[str, LoadedExtension] = {}
_LOADING_EXTENSIONS: set[str] = set()


def load_extension_module(spec: str) -> LoadedExtension:
    """Import and invoke ``module[:hook]`` exactly once per process."""

    module_name, hook_name = _parse_extension_spec(spec)
    normalized_spec = f"{module_name}:{hook_name}"
    with _LOAD_LOCK:
        loaded = _LOADED_EXTENSIONS.get(normalized_spec)
        if loaded is not None:
            return loaded
        if normalized_spec in _LOADING_EXTENSIONS:
            raise RuntimeError(f"Recursive OpenWAM extension load detected for {normalized_spec!r}.")
        _LOADING_EXTENSIONS.add(normalized_spec)
        try:
            module = import_module(module_name)
            try:
                hook = getattr(module, hook_name)
            except AttributeError as exc:
                raise AttributeError(
                    f"OpenWAM extension module {module_name!r} has no registration hook "
                    f"{hook_name!r}."
                ) from exc
            if not callable(hook):
                raise TypeError(
                    f"OpenWAM extension hook {normalized_spec!r} must be callable."
                )
            hook()
            loaded = LoadedExtension(module_name=module_name, hook_name=hook_name)
            _LOADED_EXTENSIONS[normalized_spec] = loaded
            return loaded
        finally:
            _LOADING_EXTENSIONS.remove(normalized_spec)


def load_extension_modules(specs: Iterable[str]) -> tuple[LoadedExtension, ...]:
    """Load extension specs in operator-provided order."""

    return tuple(load_extension_module(spec) for spec in specs)


def loaded_extensions() -> tuple[LoadedExtension, ...]:
    """Return extensions successfully loaded in this process."""

    with _LOAD_LOCK:
        return tuple(_LOADED_EXTENSIONS.values())


def _parse_extension_spec(spec: str) -> tuple[str, str]:
    normalized = spec.strip()
    if not normalized:
        raise ValueError("OpenWAM extension spec must be a non-empty module path.")
    module_name, separator, hook_name = normalized.partition(":")
    module_name = module_name.strip()
    hook_name = hook_name.strip() if separator else DEFAULT_EXTENSION_HOOK
    if not module_name:
        raise ValueError(f"OpenWAM extension spec {spec!r} is missing a module path.")
    if not hook_name:
        raise ValueError(f"OpenWAM extension spec {spec!r} is missing a hook name.")
    return module_name, hook_name


__all__ = [
    "DEFAULT_EXTENSION_HOOK",
    "LoadedExtension",
    "load_extension_module",
    "load_extension_modules",
    "loaded_extensions",
]
