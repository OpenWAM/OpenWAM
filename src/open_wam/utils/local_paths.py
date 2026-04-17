from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from open_wam.runtime.paths import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__))
LOCAL_PATHS_ENV_VAR = "OPEN_WAM_LOCAL_PATHS"
LOCAL_PATHS_SAMPLE_PATH = REPO_ROOT / "configs" / "local_paths.sample.yaml"
LOCAL_PATHS_PATH = REPO_ROOT / "configs" / "local_paths.yaml"
_LOCAL_PATH_PATTERN = re.compile(r"\$\{paths\.([A-Za-z0-9_.-]+)\}")


def read_yaml_with_local_paths(path: str | Path, *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Read one YAML mapping and expand `${paths.*}` placeholders."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    registry = load_local_path_registry(env=env)
    return _resolve_local_path_aliases(data, registry=registry, source_path=path)


def load_local_path_registry(*, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Load sample defaults, then overlay optional local overrides."""

    resolved_env = env or os.environ
    raw_registry: dict[str, str] = {}
    for path in _iter_local_path_files(resolved_env):
        raw = _read_registry_yaml(path)
        raw_registry.update(_flatten_registry(raw))
    return _resolve_registry_aliases(raw_registry, source_path=LOCAL_PATHS_PATH)


def _iter_local_path_files(env: Mapping[str, str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    if LOCAL_PATHS_SAMPLE_PATH.exists():
        paths.append(LOCAL_PATHS_SAMPLE_PATH)
    override = env.get(LOCAL_PATHS_ENV_VAR)
    if override:
        override_path = Path(override).expanduser()
        if not override_path.exists():
            raise FileNotFoundError(
                f"{LOCAL_PATHS_ENV_VAR} points to '{override_path}', but that file does not exist."
            )
        paths.append(override_path)
    elif LOCAL_PATHS_PATH.exists():
        paths.append(LOCAL_PATHS_PATH)
    return tuple(paths)


def _read_registry_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")
    scoped = raw.get("paths", raw)
    if not isinstance(scoped, dict):
        raise ValueError(f"Expected `paths` mapping in {path}, got {type(scoped).__name__}")
    return scoped


def _flatten_registry(raw: Mapping[str, Any], *, prefix: tuple[str, ...] = ()) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in raw.items():
        key_str = str(key)
        next_prefix = (*prefix, key_str)
        if isinstance(value, Mapping):
            flattened.update(_flatten_registry(value, prefix=next_prefix))
            continue
        if not isinstance(value, str):
            dotted = ".".join(next_prefix)
            raise ValueError(f"Expected local path alias '{dotted}' to resolve to a string path.")
        flattened[".".join(next_prefix)] = value
    return flattened


def _resolve_registry_aliases(raw_registry: dict[str, str], *, source_path: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    resolving: set[str] = set()

    def resolve_key(key: str) -> str:
        if key in resolved:
            return resolved[key]
        if key in resolving:
            raise ValueError(f"Detected a cycle while resolving local path alias '{key}' in {source_path}.")
        if key not in raw_registry:
            raise KeyError(key)
        resolving.add(key)
        raw_value = raw_registry[key]

        def replace(match: re.Match[str]) -> str:
            nested_key = match.group(1)
            if nested_key not in raw_registry:
                raise ValueError(
                    f"Local path alias '{key}' in {source_path} references unknown alias '{nested_key}'."
                )
            return resolve_key(nested_key)

        resolved_value = _LOCAL_PATH_PATTERN.sub(replace, raw_value)
        resolving.remove(key)
        resolved[key] = resolved_value
        return resolved_value

    for key in raw_registry:
        resolve_key(key)
    return resolved


def _resolve_local_path_aliases(value: Any, *, registry: Mapping[str, str], source_path: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_local_path_aliases(item, registry=registry, source_path=source_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_local_path_aliases(item, registry=registry, source_path=source_path) for item in value]
    if not isinstance(value, str) or "${paths." not in value:
        return value

    def replace(match: re.Match[str]) -> str:
        alias = match.group(1)
        resolved = registry.get(alias)
        if resolved is None:
            raise ValueError(
                "Could not resolve local path placeholder "
                f"'${{paths.{alias}}}' while reading {source_path}. "
                "Create `configs/local_paths.yaml` from `configs/local_paths.sample.yaml` "
                f"or point {LOCAL_PATHS_ENV_VAR} at your local registry file."
            )
        return resolved

    return _LOCAL_PATH_PATTERN.sub(replace, value)
