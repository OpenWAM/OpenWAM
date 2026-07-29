"""Shared coercion helpers for typed configuration boundaries."""

from __future__ import annotations

from typing import Any, TypeVar

from .enums import StrEnum


EnumT = TypeVar("EnumT", bound=StrEnum)

__all__ = [
    "coerce_bool",
    "coerce_enum",
    "coerce_enum_tuple",
    "coerce_optional_enum",
    "coerce_strict_chunk_size",
    "raw_enum_value",
]


def raw_enum_value(value: Any) -> Any:
    """Return an enum's YAML value while leaving open-ended values unchanged."""

    if isinstance(value, StrEnum):
        return value.value
    return value


def coerce_enum(enum_cls: type[EnumT], value: EnumT | str) -> EnumT:
    """Return an enum member from either its member or YAML string value."""

    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def coerce_optional_enum(
    enum_cls: type[EnumT],
    value: EnumT | str | None,
) -> EnumT | None:
    """Coerce an optional enum-backed configuration value."""

    if value is None:
        return None
    return coerce_enum(enum_cls, value)


def coerce_enum_tuple(
    enum_cls: type[EnumT],
    values: tuple[EnumT | str, ...] | list[EnumT | str],
) -> tuple[EnumT, ...]:
    """Coerce a YAML sequence into an immutable enum tuple."""

    return tuple(coerce_enum(enum_cls, value) for value in values)


def coerce_bool(value: Any, *, field_name: str) -> bool:
    """Coerce common YAML/CLI boolean spellings or fail with field context."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"`{field_name}` must be boolean, got {value!r}.")


def coerce_strict_chunk_size(name: str, value: Any) -> int:
    """Coerce strict rollout chunk-size fields with a clear config error."""

    try:
        if isinstance(value, bool):
            raise TypeError
        coerced = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "`sample_construction.target_alignment=next_after_context` requires integer chunk-size fields; "
            f"got {name}={value!r}."
        ) from None
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            "`sample_construction.target_alignment=next_after_context` requires integer chunk-size fields; "
            f"got {name}={value!r}."
        )
    return coerced
