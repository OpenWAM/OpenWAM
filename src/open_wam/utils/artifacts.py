from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ArtifactManifestEntry:
    artifact_id: str
    method_family: str
    variant: str
    benchmark: str | None
    config: str
    local_path_alias: str | None
    expected_layout: dict[str, Any]
    download_url: str | None
    checksum: str | None
    license: str | None
    source: str | None
    notes: str | None


def load_artifact_manifest(path: str | Path) -> tuple[ArtifactManifestEntry, ...]:
    """Load an Open-WAM artifact manifest."""

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Expected `artifacts` list in {path}.")
    return tuple(_coerce_artifact_entry(item, source_path=path) for item in artifacts)


def validate_artifact_layout(root: str | Path, expected_layout: dict[str, Any]) -> tuple[str, ...]:
    """Return missing paths for an artifact root and expected layout mapping."""

    root = Path(root)
    missing: list[str] = []
    for key in ("root_files", "directories", "transformer_files"):
        values = expected_layout.get(key, ())
        if values is None:
            continue
        if not isinstance(values, list):
            raise ValueError(f"expected_layout.{key} must be a list when provided.")
        for relative in values:
            if not isinstance(relative, str):
                raise ValueError(f"expected_layout.{key} entries must be strings.")
            if not (root / relative).exists():
                missing.append(relative)
    return tuple(missing)


def _coerce_artifact_entry(raw: Any, *, source_path: Path) -> ArtifactManifestEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected artifact entries in {source_path} to be mappings.")
    required = ("artifact_id", "method_family", "variant", "config", "expected_layout")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Artifact entry in {source_path} is missing required fields: {missing}")
    if not isinstance(raw["expected_layout"], dict):
        raise ValueError(f"Artifact {raw.get('artifact_id')!r} expected_layout must be a mapping.")
    return ArtifactManifestEntry(
        artifact_id=str(raw["artifact_id"]),
        method_family=str(raw["method_family"]),
        variant=str(raw["variant"]),
        benchmark=None if raw.get("benchmark") is None else str(raw["benchmark"]),
        config=str(raw["config"]),
        local_path_alias=None if raw.get("local_path_alias") is None else str(raw["local_path_alias"]),
        expected_layout=dict(raw["expected_layout"]),
        download_url=None if raw.get("download_url") is None else str(raw["download_url"]),
        checksum=None if raw.get("checksum") is None else str(raw["checksum"]),
        license=None if raw.get("license") is None else str(raw["license"]),
        source=None if raw.get("source") is None else str(raw["source"]),
        notes=None if raw.get("notes") is None else str(raw["notes"]),
    )
