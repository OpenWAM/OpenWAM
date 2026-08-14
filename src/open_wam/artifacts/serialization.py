"""Safe deserialization boundaries for tensor-bearing artifacts.

Open-WAM artifacts are data, not executable Python object graphs. The normal
loader therefore uses PyTorch's restricted weights-only unpickler. A second
loader admits only the NumPy reconstruction globals needed by historical
LIBERO init-state files while retaining the weights-only restrictions.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import torch


class UnsafeArtifactError(RuntimeError):
    """Raised when an artifact requires unrestricted pickle deserialization."""


def load_tensor_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load tensors and primitive containers without executing pickle globals."""

    resolved_path = Path(path).expanduser()
    try:
        return torch.load(
            resolved_path,
            map_location=map_location,
            weights_only=True,
        )
    except pickle.UnpicklingError as exc:
        raise UnsafeArtifactError(
            f"Artifact {resolved_path} is not compatible with Open-WAM's safe "
            "tensor format. Convert it to tensors and primitive containers; "
            "Open-WAM will not retry with unrestricted pickle loading."
        ) from exc


def load_numpy_compatible_torch_artifact(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Load a legacy tensor artifact with restricted NumPy reconstruction."""

    with torch.serialization.safe_globals(_numpy_safe_globals()):
        return load_tensor_artifact(path, map_location=map_location)


def load_trusted_numpy_pickle_artifact(
    path: str | Path,
    *,
    trust_reason: str,
) -> Any:
    """Load an intrinsically pickled NumPy artifact after explicit acknowledgement.

    This boundary exists only for upstream formats that cannot be represented
    by NumPy's non-pickle loader. Callers must expose an opt-in policy and pass
    a concrete reason; normal Open-WAM artifacts must use safe tensor formats.
    """

    if not trust_reason.strip():
        raise ValueError("Trusted NumPy pickle loading requires a non-empty reason.")
    return np.load(Path(path).expanduser(), allow_pickle=True)


def _legacy_numpy_global_alias(
    value: Callable[..., Any],
) -> Callable[..., Any]:
    @wraps(value)
    def alias(*args: Any, **kwargs: Any) -> Any:
        return value(*args, **kwargs)

    alias.__module__ = "numpy.core.multiarray"
    return alias


def _numpy_safe_globals() -> tuple[Any, ...]:
    # ``hasattr(np, "_core")`` is not a usable NumPy-2 test: under NumPy 1.26
    # the ``numpy._core`` shim only becomes an attribute of ``numpy`` once some
    # other import has pulled in a ``numpy._core`` submodule, so the branch
    # taken depends on import order. Import the module directly instead.
    try:
        from numpy._core import multiarray
    except ImportError:  # NumPy without the ``_core`` shim.
        from numpy.core import multiarray
    numpy_globals = (multiarray._reconstruct, multiarray.scalar)
    safe_values: list[Any] = [*numpy_globals, np.ndarray, np.dtype]
    # LIBERO published its init states under NumPy 1.x, so those pickles name
    # ``numpy.core.multiarray``. NumPy 2 moved the callables to ``numpy._core``;
    # callable aliases preserve the old names without relying on named
    # safe-global tuples, which PyTorch only supports starting in 2.6.
    if multiarray._reconstruct.__module__ != "numpy.core.multiarray":
        safe_values.extend(
            _legacy_numpy_global_alias(value) for value in numpy_globals
        )
    for dtype_name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ):
        safe_values.append(type(np.dtype(dtype_name)))
    return tuple(dict.fromkeys(safe_values))


__all__ = [
    "UnsafeArtifactError",
    "load_numpy_compatible_torch_artifact",
    "load_tensor_artifact",
    "load_trusted_numpy_pickle_artifact",
]
