"""Versioned component contract for standalone runtime-backbone exports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_wam.configs.enums import TrainingComponentSelector
from open_wam.configs.runtime_backbone_components import (
    validate_runtime_backbone_components,
)

RUNTIME_BACKBONE_MANIFEST_FILENAME = "runtime_backbone_manifest.json"
RUNTIME_BACKBONE_MANIFEST_SCHEMA_V1 = "open_wam.runtime_backbone.v1"


@dataclass(frozen=True)
class RuntimeBackboneManifest:
    """Semantic contents and exact tensor inventory of one transformer export."""

    components: tuple[TrainingComponentSelector, ...]
    state_keys: tuple[str, ...]
    schema_version: str = RUNTIME_BACKBONE_MANIFEST_SCHEMA_V1

    def __post_init__(self) -> None:
        components = validate_runtime_backbone_components(
            self.components,
            scope="Runtime-backbone manifest components",
        )
        state_keys = tuple(self.state_keys)
        if not state_keys:
            raise ValueError("Runtime-backbone manifest requires non-empty state keys.")
        if any(not isinstance(key, str) or not key for key in state_keys):
            raise TypeError(
                "Runtime-backbone manifest state keys must be non-empty strings."
            )
        if len(set(state_keys)) != len(state_keys):
            raise ValueError("Runtime-backbone manifest state keys must be unique.")
        if self.schema_version != RUNTIME_BACKBONE_MANIFEST_SCHEMA_V1:
            raise ValueError(
                "Unsupported runtime-backbone manifest schema "
                f"{self.schema_version!r}; expected {RUNTIME_BACKBONE_MANIFEST_SCHEMA_V1!r}."
            )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "state_keys", tuple(sorted(state_keys)))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RuntimeBackboneManifest:
        components = raw.get("components")
        state_keys = raw.get("state_keys")
        if not isinstance(components, list) or not isinstance(state_keys, list):
            raise TypeError(
                "Runtime-backbone manifest requires list-valued `components` and `state_keys`."
            )
        if any(not isinstance(value, str) for value in (*components, *state_keys)):
            raise TypeError(
                "Runtime-backbone manifest components and state keys must be strings."
            )
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            components=tuple(TrainingComponentSelector(value) for value in components),
            state_keys=tuple(state_keys),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "components": [component.value for component in self.components],
            "state_keys": list(self.state_keys),
        }


def load_runtime_backbone_manifest(
    transformer_dir: Path,
) -> RuntimeBackboneManifest | None:
    path = transformer_dir / RUNTIME_BACKBONE_MANIFEST_FILENAME
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Runtime-backbone manifest must be a JSON object: {path}.")
    return RuntimeBackboneManifest.from_mapping(raw)


def write_runtime_backbone_manifest(
    transformer_dir: Path,
    manifest: RuntimeBackboneManifest,
) -> Path:
    path = transformer_dir / RUNTIME_BACKBONE_MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "RUNTIME_BACKBONE_MANIFEST_FILENAME",
    "RUNTIME_BACKBONE_MANIFEST_SCHEMA_V1",
    "RuntimeBackboneManifest",
    "load_runtime_backbone_manifest",
    "write_runtime_backbone_manifest",
]
