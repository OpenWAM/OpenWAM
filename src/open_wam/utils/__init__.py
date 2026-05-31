"""Utilities for config loading, seeding, and experiment bootstrapping.

Exports are resolved lazily so minimal installs can use config/artifact helpers
without importing optional Torch/Numpy-backed runtime modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "ArtifactManifestEntry": "open_wam.utils.artifacts",
    "apply_config_overrides": "open_wam.utils.config_overrides",
    "load_artifact_manifest": "open_wam.utils.artifacts",
    "validate_artifact_layout": "open_wam.utils.artifacts",
    "find_checkpoint_resolved_config": "open_wam.utils.checkpoint_runtime",
    "load_experiment_config": "open_wam.utils.config_loader",
    "load_local_path_registry": "open_wam.utils.local_paths",
    "merge_checkpoint_runtime_config": "open_wam.utils.checkpoint_runtime",
    "merge_runtime_config_from_checkpoint": "open_wam.utils.checkpoint_runtime",
    "read_yaml_with_local_paths": "open_wam.utils.local_paths",
    "parse_override_assignments": "open_wam.utils.config_overrides",
    "resolve_transformer_dir_override": "open_wam.utils.cli",
    "resolve_checkpoint_file": "open_wam.utils.checkpoint_runtime",
    "validate_positive_step_override": "open_wam.utils.cli",
    "seed_everywhere": "open_wam.utils.seeding",
}

__all__ = [
    "ArtifactManifestEntry",
    "apply_config_overrides",
    "find_checkpoint_resolved_config",
    "load_experiment_config",
    "load_artifact_manifest",
    "load_local_path_registry",
    "merge_checkpoint_runtime_config",
    "merge_runtime_config_from_checkpoint",
    "parse_override_assignments",
    "read_yaml_with_local_paths",
    "resolve_checkpoint_file",
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
