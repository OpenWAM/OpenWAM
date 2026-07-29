from __future__ import annotations

from collections.abc import Callable
from typing import Any

from open_wam.configs import ActionDecoderName
from open_wam.registry import BuilderRegistry


POLICY_VARIANT_BUILDERS = BuilderRegistry[type[Any], Any]("policy variant builder")
ACTION_DECODER_BUILDERS = BuilderRegistry[ActionDecoderName, Any]("action decoder builder")
_EXTENSION_POLICY_VARIANT_BUILDERS = BuilderRegistry[str, Any]("extension policy variant builder")
_EXTENSION_ACTION_DECODER_BUILDERS = BuilderRegistry[str, Any]("extension action decoder builder")


def _validate_extension_type(extension_type: str, *, role: str) -> str:
    if not isinstance(extension_type, str) or not extension_type.strip():
        raise ValueError(f"{role} extension type must be a non-empty string.")
    if extension_type != extension_type.strip():
        raise ValueError(f"{role} extension type must not have surrounding whitespace.")
    return extension_type


def register_policy_variant(
    extension_type: str,
    builder: Callable[[Any], Any],
    *,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Register an application-owned policy builder by open string identifier."""

    key = _validate_extension_type(extension_type, role="Policy variant")
    if not callable(builder):
        raise TypeError("Policy variant extension builder must be callable.")
    _EXTENSION_POLICY_VARIANT_BUILDERS.register(
        key,
        builder,
        description=description,
        replace=replace,
    )


def register_action_decoder(
    extension_type: str,
    builder: Callable[[Any], Any],
    *,
    description: str | None = None,
    replace: bool = False,
) -> None:
    """Register an application-owned action-decoder builder."""

    key = _validate_extension_type(extension_type, role="Action decoder")
    if not callable(builder):
        raise TypeError("Action decoder extension builder must be callable.")
    _EXTENSION_ACTION_DECODER_BUILDERS.register(
        key,
        builder,
        description=description,
        replace=replace,
    )


def registered_policy_variants() -> tuple[str, ...]:
    """Return application-owned policy identifiers registered in this process."""

    return _EXTENSION_POLICY_VARIANT_BUILDERS.keys()


def registered_action_decoders() -> tuple[str, ...]:
    """Return application-owned action-decoder identifiers registered in this process."""

    return _EXTENSION_ACTION_DECODER_BUILDERS.keys()


__all__ = [
    "ACTION_DECODER_BUILDERS",
    "POLICY_VARIANT_BUILDERS",
    "register_action_decoder",
    "register_policy_variant",
    "registered_action_decoders",
    "registered_policy_variants",
]
