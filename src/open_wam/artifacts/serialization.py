"""Safe deserialization boundaries for tensor-bearing artifacts.

Open-WAM artifacts are data, not executable Python object graphs. The normal
loader therefore uses PyTorch's restricted weights-only unpickler. A second
loader admits only the NumPy reconstruction globals needed by historical
LIBERO init-state files while retaining the weights-only restrictions.
"""

from __future__ import annotations

import pickle
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


def _numpy_safe_globals() -> tuple[Any, ...]:
    numpy_core = np._core if hasattr(np, "_core") else np.core
    multiarray = numpy_core.multiarray
    safe_values: list[Any] = [
        multiarray._reconstruct,
        multiarray.scalar,
        np.ndarray,
        np.dtype,
    ]
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
