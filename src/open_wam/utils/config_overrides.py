from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, Mapping

import yaml

from open_wam.configs import ExperimentConfig


def parse_override_assignments(tokens: tuple[str, ...] | list[str]) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for token in tokens:
        key, raw_value = _split_override_token(token)
        assignments[key] = yaml.safe_load(raw_value)
    return assignments


def apply_config_overrides(config: ExperimentConfig, overrides: Mapping[str, Any]) -> ExperimentConfig:
    updated = config
    grouped_overrides: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, value in overrides.items():
        parts = tuple(part.replace("-", "_") for part in key.split("."))
        grouped_overrides.setdefault(parts[:-1], {})[parts[-1]] = value
    for parent_path, values in sorted(grouped_overrides.items(), key=lambda item: len(item[0]), reverse=True):
        updated = _replace_dataclass_fields(updated, list(parent_path), values)
    return updated


def _split_override_token(token: str) -> tuple[str, str]:
    key, raw_value = token.split("=", 1)
    key = key.strip().replace("-", "_")
    if not key:
        raise ValueError(f"Override key is empty in token {token!r}.")
    return key, raw_value


def _replace_dataclass_fields(node: object, path: list[str], values: Mapping[str, Any]):
    if not is_dataclass(node):
        raise ValueError(f"Cannot override nested path on non-dataclass node {type(node).__name__}.")
    if path:
        field_name = path[0].replace("-", "_")
        if not hasattr(node, field_name):
            raise ValueError(f"{type(node).__name__} has no field {field_name!r}.")
        current_value = getattr(node, field_name)
        nested_value = _replace_dataclass_fields(current_value, path[1:], values)
        return replace(node, **{field_name: nested_value})
    updates: dict[str, Any] = {}
    for field_name, value in values.items():
        normalized_field_name = field_name.replace("-", "_")
        if not hasattr(node, normalized_field_name):
            raise ValueError(f"{type(node).__name__} has no field {normalized_field_name!r}.")
        updates[normalized_field_name] = _coerce_override_value(getattr(node, normalized_field_name), value)
    return replace(node, **updates)


def _coerce_override_value(current_value: Any, value: Any) -> Any:
    if isinstance(current_value, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current_value, bool) and isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Cannot coerce override value {value!r} to bool.")
    if isinstance(current_value, bool) and isinstance(value, int):
        return bool(value)
    if isinstance(current_value, float) and isinstance(value, int):
        return float(value)
    if isinstance(current_value, float) and isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Cannot coerce override value {value!r} to float.") from exc
    return value
