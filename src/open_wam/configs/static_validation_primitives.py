"""Primitive enum, scalar, path, and YAML validation helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from .enums import BackboneImplementation, StrEnum
from .static_validation_contracts import _IssueBuilder


LOCAL_PATH_PATTERN = re.compile(r"\$\{paths\.([A-Za-z0-9_.-]+)\}")
ENUM_VALUE_ALIASES: dict[type[StrEnum], dict[str, str]] = {
    BackboneImplementation: {
        "lingbot_replica": BackboneImplementation.SHARED_TRANSFORMER.value,
    }
}


def _validate_local_path_placeholders(value: Any, issues: "_IssueBuilder", *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = str(key) if not path else f"{path}.{key}"
            _validate_local_path_placeholders(item, issues, path=child_path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_local_path_placeholders(item, issues, path=f"{path}[{index}]")
        return
    if not isinstance(value, str) or "${paths." not in value:
        return
    matches = LOCAL_PATH_PATTERN.findall(value)
    if not matches:
        issues.error(path, "Malformed local path placeholder. Expected `${paths.alias}`.")
    for alias in matches:
        if ".." in alias or alias.startswith(".") or alias.endswith("."):
            issues.error(path, f"Invalid local path alias syntax: {alias!r}.")


def _validate_enum(
    mapping: Mapping[str, Any],
    key: str,
    enum_cls: type[StrEnum],
    issues: "_IssueBuilder",
    path_prefix: str,
) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if isinstance(value, enum_cls):
        return
    if not isinstance(value, str):
        issues.error(_join_path(path_prefix, key), f"Expected a string enum value for {enum_cls.__name__}.")
        return
    value = ENUM_VALUE_ALIASES.get(enum_cls, {}).get(value, value)
    valid = {item.value for item in enum_cls}
    if value not in valid:
        issues.error(
            _join_path(path_prefix, key),
            f"Invalid {enum_cls.__name__} value {value!r}. Expected one of {sorted(valid)}.",
        )


def _validate_positive_ints(
    mapping: Mapping[str, Any],
    issues: "_IssueBuilder",
    path_prefix: str,
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key not in mapping or mapping[key] is None:
            continue
        value = _optional_int(mapping[key])
        if value is None or value <= 0:
            issues.error(_join_path(path_prefix, key), "Expected a positive integer.")


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _join_path(prefix: str, key: str) -> str:
    return key if not prefix else f"{prefix}.{key}"


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return raw


def _resolve_relative(source_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    local_candidate = (source_path.parent / candidate).resolve()
    if local_candidate.exists():
        return local_candidate
    return (_find_repo_root(source_path) / candidate).resolve()


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()
