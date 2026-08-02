"""Actionable import errors for optional Open-WAM runtime components."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def load_optional_module(
    module_name: str,
    *,
    public_name: str,
    extra: str,
) -> ModuleType:
    """Import one optional module without hiding missing package internals."""

    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "open_wam" or exc.name.startswith("open_wam.")):
            raise
        missing = exc.name or "an optional runtime module"
        raise ImportError(
            f"`{public_name}` requires optional dependencies. Install with "
            f"`pip install 'open-wam[{extra}]'` or `uv sync --extra {extra}`. "
            f"Missing module: {missing}."
        ) from exc


__all__ = ["load_optional_module"]
