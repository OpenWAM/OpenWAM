"""Source discovery, resolution, and caching for LeRobot consortium data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from open_wam.configs import ConsortiumCacheMode, LeRobotConsortiumDataConfig


__all__ = [
    "CloudConsortiumCache",
    "ConsortiumSourceResolver",
    "ConsortiumSourceSpec",
    "LocalConsortiumCache",
    "NoopConsortiumCache",
    "discover_local_lerobot_consortium_members",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _strip_file_uri(value: str | None) -> str | None:
    if value is None:
        return None
    if value.startswith("file://"):
        return value[len("file://") :]
    return value


@dataclass(frozen=True)
class ConsortiumSourceSpec:
    member_id: str
    repo_id: str | None
    local_root: str | None


class NoopConsortiumCache:
    """Resolve physical source files without an Open-WAM-managed cache."""

    def resolve(self, *, source: ConsortiumSourceSpec, relative_path: str, cache_dir: str | None) -> Path:
        if source.local_root is not None:
            return Path(source.local_root).expanduser().resolve() / relative_path
        if source.repo_id is None:
            raise ValueError(f"Cannot resolve consortium source for member '{source.member_id}' without repo_id.")
        return Path(
            hf_hub_download(
                repo_id=source.repo_id,
                filename=relative_path,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        )


class _FilesystemConsortiumCache:
    """Shared filesystem mechanics for configured consortium caches."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def path_for(self, *, source: ConsortiumSourceSpec, relative_path: str) -> Path:
        return self.root / source.member_id / relative_path

    def has(self, *, source: ConsortiumSourceSpec, relative_path: str) -> bool:
        return self.path_for(source=source, relative_path=relative_path).exists()

    def resolve(self, *, source: ConsortiumSourceSpec, relative_path: str) -> Path:
        return self.path_for(source=source, relative_path=relative_path)

    def store(self, *, source: ConsortiumSourceSpec, relative_path: str, source_path: Path) -> Path:
        target = self.path_for(source=source, relative_path=relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return target


class LocalConsortiumCache(_FilesystemConsortiumCache):
    """Filesystem cache for resolved consortium source files."""

    def __init__(self, root: str) -> None:
        super().__init__(Path(root))


class CloudConsortiumCache(_FilesystemConsortiumCache):
    """Filesystem-backed cache for an optional mounted cloud root."""

    def __init__(self, root: str) -> None:
        super().__init__(Path(_strip_file_uri(root) or root))


class ConsortiumSourceResolver:
    """Resolve consortium source files with optional local/cloud caches."""

    def __init__(self, data_config: LeRobotConsortiumDataConfig) -> None:
        self.data_config = data_config
        self.noop = NoopConsortiumCache()
        self.local_cache = (
            LocalConsortiumCache(data_config.local_cache.root)
            if data_config.local_cache.mode != ConsortiumCacheMode.DISABLED and data_config.local_cache.root
            else None
        )
        self.cloud_cache = (
            CloudConsortiumCache(data_config.cloud_cache.root)
            if data_config.cloud_cache.mode != ConsortiumCacheMode.DISABLED and data_config.cloud_cache.root
            else None
        )

    def resolve(self, *, source: ConsortiumSourceSpec, relative_path: str) -> Path:
        if self.local_cache is not None and self.local_cache.has(source=source, relative_path=relative_path):
            return self.local_cache.resolve(source=source, relative_path=relative_path)
        if self.cloud_cache is not None and self.cloud_cache.has(source=source, relative_path=relative_path):
            resolved = self.cloud_cache.resolve(source=source, relative_path=relative_path)
            if self.local_cache is not None and self.data_config.local_cache.mode == ConsortiumCacheMode.WRITE_THROUGH:
                return self.local_cache.store(source=source, relative_path=relative_path, source_path=resolved)
            return resolved

        source_path = self.noop.resolve(source=source, relative_path=relative_path, cache_dir=self.data_config.cache_dir)

        if self.cloud_cache is not None and self.data_config.cloud_cache.mode == ConsortiumCacheMode.WRITE_THROUGH:
            cached = self.cloud_cache.store(source=source, relative_path=relative_path, source_path=source_path)
            if self.local_cache is not None and self.data_config.local_cache.mode == ConsortiumCacheMode.WRITE_THROUGH:
                return self.local_cache.store(source=source, relative_path=relative_path, source_path=cached)
            return cached
        if self.local_cache is not None and self.data_config.local_cache.mode == ConsortiumCacheMode.WRITE_THROUGH:
            return self.local_cache.store(source=source, relative_path=relative_path, source_path=source_path)
        return source_path

    def read_json(self, *, source: ConsortiumSourceSpec, relative_path: str) -> dict[str, Any]:
        return _read_json(self.resolve(source=source, relative_path=relative_path))

    def read_jsonl(self, *, source: ConsortiumSourceSpec, relative_path: str) -> list[dict[str, Any]]:
        return _read_jsonl(self.resolve(source=source, relative_path=relative_path))

    def read_parquet_rows(self, *, source: ConsortiumSourceSpec, relative_path: str) -> list[dict[str, Any]]:
        return pq.read_table(self.resolve(source=source, relative_path=relative_path)).to_pylist()


def discover_local_lerobot_consortium_members(local_root: str | None) -> tuple[ConsortiumSourceSpec, ...]:
    """Discover one local LeRobot repository or its immediate children."""

    if local_root is None:
        return ()
    root = Path(local_root).expanduser().resolve()
    if not root.exists():
        return ()
    candidates: list[Path] = []
    if (root / "meta" / "info.json").exists():
        candidates.append(root)
    else:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "meta" / "info.json").exists() and (child / "meta" / "episodes.jsonl").exists():
                candidates.append(child)
    return tuple(
        ConsortiumSourceSpec(member_id=candidate.name, repo_id=None, local_root=str(candidate))
        for candidate in candidates
    )
