"""Vendored LingBot reference modules."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _shim_root() -> Path:
    return Path(__file__).resolve().parents[2] / "_shims"


def _install_shim_module(module_name: str, relative_path: str) -> None:
    module_path = (_shim_root() / relative_path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import flash-attn shim from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _ensure_flash_attn_shims() -> None:
    for module_name, relative_path in (
        ("flash_attn_interface", "flash_attn_interface.py"),
        ("flash_attn", "flash_attn.py"),
    ):
        if module_name in sys.modules:
            continue
        try:
            importlib.import_module(module_name)
        except ImportError:
            _install_shim_module(module_name, relative_path)


_ensure_flash_attn_shims()

__all__ = ["WanTransformer3DModel"]


def __getattr__(name: str):
    if name == "WanTransformer3DModel":
        from .model import WanTransformer3DModel

        globals()[name] = WanTransformer3DModel
        return WanTransformer3DModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
