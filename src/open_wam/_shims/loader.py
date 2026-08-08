"""Package-relative loaders for optional dependency shims."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


_SHIM_ROOT = Path(__file__).resolve().parent
_FLASH_ATTN_SHIMS = (
    ("flash_attn_interface", "flash_attn_interface.py"),
    ("flash_attn", "flash_attn.py"),
)


def ensure_flash_attn_shims() -> None:
    """Install bundled FlashAttention compatibility modules when unavailable."""

    for module_name, filename in _FLASH_ATTN_SHIMS:
        if module_name in sys.modules:
            continue
        try:
            importlib.import_module(module_name)
        except ImportError:
            _install_shim_module(module_name, _SHIM_ROOT / filename)


def _install_shim_module(module_name: str, module_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import compatibility shim from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
