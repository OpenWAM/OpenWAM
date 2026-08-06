"""Reproducibility metadata for public runtime results."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from enum import StrEnum
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from open_wam import __version__
from open_wam.contracts import find_repo_root


OPEN_WAM_PROVENANCE_SCHEMA_V1 = "open_wam.provenance.v1"


class ProvenanceMode(StrEnum):
    """Cost/detail level for artifact identities in runtime results."""

    STANDARD = "standard"
    FULL = "full"


def collect_runtime_provenance(
    *,
    config_path: str | Path | None,
    resolved_config: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    mode: ProvenanceMode | str = ProvenanceMode.STANDARD,
    argv: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Collect deterministic identities without importing model dependencies."""

    resolved_mode = ProvenanceMode(mode)
    config_identity = _file_identity(config_path, include_sha256=True)
    if resolved_config is not None:
        config_identity["resolved_sha256"] = _mapping_sha256(resolved_config)
    return {
        "schema_version": OPEN_WAM_PROVENANCE_SCHEMA_V1,
        "mode": resolved_mode.value,
        "source": _git_identity(),
        "command_argv": list(sys.argv if argv is None else argv),
        "config": config_identity,
        "checkpoint": _file_identity(
            checkpoint_path,
            include_sha256=resolved_mode is ProvenanceMode.FULL,
        ),
        "dataset": _dataset_identity(dataset_root),
        "environment": _environment_identity(),
    }


def _file_identity(
    path: str | Path | None,
    *,
    include_sha256: bool,
) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    resolved = Path(path).expanduser().resolve()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": None,
    }
    if not resolved.is_file():
        return identity
    stat = resolved.stat()
    identity.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    )
    if include_sha256:
        identity["sha256"] = _sha256_file(resolved)
    return identity


def _dataset_identity(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "metadata": []}
    root = Path(path).expanduser().resolve()
    metadata_paths = (
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "metadata.json",
        "manifest.json",
    )
    return {
        "path": str(root),
        "exists": root.exists(),
        "metadata": [
            _file_identity(root / relative, include_sha256=True)
            for relative in metadata_paths
            if (root / relative).is_file()
        ],
    }


def _git_identity(package_file: str | Path | None = None) -> dict[str, Any]:
    source_file = Path(__file__ if package_file is None else package_file).resolve()
    root = find_repo_root(source_file)
    imported_package_root = source_file.parents[1]
    source_package_roots = (
        (root / "src" / "open_wam").resolve(),
        (root / "open_wam").resolve(),
    )
    if imported_package_root not in source_package_roots or not (
        root / ".git"
    ).exists():
        return {"root": None, "commit": None, "dirty": None}
    commit = _run_git(root, "rev-parse", "HEAD")
    status = _run_git(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "root": str(root),
        "commit": commit or None,
        "dirty": None if status is None else bool(status),
    }


def _run_git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _environment_identity() -> dict[str, Any]:
    package_names = ("torch", "diffusers", "transformers", "accelerate")
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return {
        "open_wam": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "OPEN_WAM_PROVENANCE_SCHEMA_V1",
    "ProvenanceMode",
    "collect_runtime_provenance",
]
