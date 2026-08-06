"""Safe deserialization boundaries for Open-WAM artifacts."""

from .serialization import (
    UnsafeArtifactError,
    load_numpy_compatible_torch_artifact,
    load_tensor_artifact,
    load_trusted_numpy_pickle_artifact,
)

__all__ = [
    "UnsafeArtifactError",
    "load_numpy_compatible_torch_artifact",
    "load_tensor_artifact",
    "load_trusted_numpy_pickle_artifact",
]
