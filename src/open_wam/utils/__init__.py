"""Utilities for config loading, seeding, and experiment bootstrapping.

Exports are resolved lazily so minimal installs can use config/artifact helpers
without importing optional Torch/Numpy-backed runtime modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "ArtifactManifestEntry": "open_wam.utils.artifacts",
    "load_artifact_manifest": "open_wam.utils.artifacts",
    "validate_artifact_layout": "open_wam.utils.artifacts",
    "load_experiment_config": "open_wam.utils.config_loader",
    "load_local_path_registry": "open_wam.utils.local_paths",
    "read_yaml_with_local_paths": "open_wam.utils.local_paths",
    "resolve_transformer_dir_override": "open_wam.utils.cli",
    "validate_positive_step_override": "open_wam.utils.cli",
    "seed_everywhere": "open_wam.utils.seeding",
}

__all__ = [
    "ArtifactManifestEntry",
    "load_experiment_config",
    "load_artifact_manifest",
    "load_local_path_registry",
    "read_yaml_with_local_paths",
    "resolve_transformer_dir_override",
    "seed_everywhere",
    "validate_artifact_layout",
    "validate_positive_step_override",
]


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
