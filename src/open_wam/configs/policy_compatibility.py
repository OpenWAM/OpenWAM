"""Input-boundary compatibility for retired video/action config names.

Canonical config dataclasses and runtime code do not depend on historical
architecture labels. This module is the single place that accepts those
labels while loading old YAML, checkpoint configs, and CLI overrides.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeVar

from .enums import ActionDecoderName, JointTimestepCoupling, PolicyVariantName

_T = TypeVar("_T")


LEGACY_VIDEO_ACTION_POLICY_FIELD_ALIASES = {
    "parallel_sequence_contract": "sequence_contract",
    "joint_denoise_training_mode_probs": "generalist_denoising_mode_probs",
    "mot_generalist_training_mode_probs": "generalist_denoising_mode_probs",
}
LEGACY_VIDEO_ACTION_POLICY_FIELDS = frozenset(
    {
        *LEGACY_VIDEO_ACTION_POLICY_FIELD_ALIASES,
        "couple_action_to_video_timesteps",
    }
)


class DeprecatedPolicyConfigFieldWarning(FutureWarning):
    """A policy config used a supported but retired public field name."""


def _plain_config_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain_config_value(item) for item in value)
    return value


def _warn_legacy_field(*, legacy_name: str, canonical_name: str, stacklevel: int) -> None:
    warnings.warn(
        f"Policy config field `{legacy_name}` is deprecated; use `{canonical_name}`.",
        DeprecatedPolicyConfigFieldWarning,
        stacklevel=stacklevel,
    )


def normalize_video_action_policy_fields(
    raw_policy: Mapping[str, Any],
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Return a canonical copy of one raw video/action policy mapping."""

    normalized = dict(raw_policy)
    if normalized.get("name") == "mot":
        normalized["name"] = PolicyVariantName.DUAL_EXPERT
        if warn:
            warnings.warn(
                "Policy architecture `mot` is deprecated; use `dual_expert`.",
                DeprecatedPolicyConfigFieldWarning,
                stacklevel=3,
            )
    for legacy_name, canonical_name in LEGACY_VIDEO_ACTION_POLICY_FIELD_ALIASES.items():
        if legacy_name not in normalized:
            continue
        legacy_value = normalized.pop(legacy_name)
        if (
            canonical_name in normalized
            and _plain_config_value(normalized[canonical_name])
            != _plain_config_value(legacy_value)
        ):
            raise ValueError(
                f"Conflicting policy config fields `{canonical_name}` and deprecated "
                f"`{legacy_name}`: {normalized[canonical_name]!r} != {legacy_value!r}."
            )
        normalized[canonical_name] = legacy_value
        if warn:
            _warn_legacy_field(
                legacy_name=legacy_name,
                canonical_name=canonical_name,
                stacklevel=3,
            )

    legacy_timestep_key = "couple_action_to_video_timesteps"
    if legacy_timestep_key in normalized:
        legacy_value = normalized.pop(legacy_timestep_key)
        canonical_name = "joint_timestep_coupling"
        if legacy_value is not None:
            if not isinstance(legacy_value, bool):
                raise TypeError(
                    f"Deprecated `{legacy_timestep_key}` must be a boolean, "
                    f"got {legacy_value!r}."
                )
            canonical_value = (
                JointTimestepCoupling.MATCH_SIGMA
                if legacy_value
                else JointTimestepCoupling.INDEPENDENT
            )
            if (
                canonical_name in normalized
                and _plain_config_value(normalized[canonical_name])
                != canonical_value.value
            ):
                raise ValueError(
                    f"Conflicting policy config fields `{canonical_name}` and deprecated "
                    f"`{legacy_timestep_key}`."
                )
            normalized[canonical_name] = canonical_value
        if warn:
            _warn_legacy_field(
                legacy_name=legacy_timestep_key,
                canonical_name=canonical_name,
                stacklevel=3,
            )
    return normalized


def normalize_video_action_decoder_fields(
    raw_decoder: Mapping[str, Any],
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Canonicalize retired architecture names in an action decoder mapping."""

    normalized = dict(raw_decoder)
    raw_name = _plain_config_value(normalized.get("name"))
    legacy_decoder_names = {
        "mot_decoder": ActionDecoderName.DUAL_EXPERT,
        "lingbot_parallel_decoder": ActionDecoderName.PARALLEL_STREAM,
    }
    canonical_name = legacy_decoder_names.get(raw_name)
    if canonical_name is not None:
        normalized["name"] = canonical_name
        if warn:
            warnings.warn(
                f"Action decoder `{raw_name}` is deprecated; use `{canonical_name.value}`.",
                DeprecatedPolicyConfigFieldWarning,
                stacklevel=3,
            )
    return normalized


def normalize_video_action_config_fields(
    raw_config: Mapping[str, Any],
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Canonicalize shared video/action keys in a root experiment mapping."""

    normalized = dict(raw_config)
    raw_policy = normalized.get("policy_variant")
    if isinstance(raw_policy, Mapping):
        normalized["policy_variant"] = normalize_video_action_policy_fields(
            raw_policy,
            warn=warn,
        )
    raw_decoder = normalized.get("action_decoder")
    if isinstance(raw_decoder, Mapping):
        normalized["action_decoder"] = normalize_video_action_decoder_fields(
            raw_decoder,
            warn=warn,
        )
    return normalized


def normalize_video_action_override_keys(
    overrides: Mapping[str, Any],
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Canonicalize shared video/action keys in flattened CLI overrides."""

    normalized = dict(overrides)
    policy_prefix = "policy_variant."
    raw_policy = {
        key.removeprefix(policy_prefix): value
        for key, value in normalized.items()
        if key.startswith(policy_prefix)
    }
    canonical_policy = normalize_video_action_policy_fields(raw_policy, warn=warn)
    for key in tuple(normalized):
        if key.startswith(policy_prefix):
            del normalized[key]
    normalized.update(
        {f"{policy_prefix}{key}": value for key, value in canonical_policy.items()}
    )

    decoder_prefix = "action_decoder."
    raw_decoder = {
        key.removeprefix(decoder_prefix): value
        for key, value in normalized.items()
        if key.startswith(decoder_prefix)
    }
    canonical_decoder = normalize_video_action_decoder_fields(raw_decoder, warn=warn)
    for key in tuple(normalized):
        if key.startswith(decoder_prefix):
            del normalized[key]
    normalized.update(
        {f"{decoder_prefix}{key}": value for key, value in canonical_decoder.items()}
    )
    return normalized


def resolve_legacy_policy_field(
    *,
    canonical_value: _T,
    legacy_value: _T | None,
    canonical_default: _T,
    canonical_name: str,
    legacy_name: str,
) -> _T:
    """Resolve one constructor-level compatibility alias without ambiguity."""

    if legacy_value is None:
        return canonical_value
    if canonical_value != canonical_default and canonical_value != legacy_value:
        raise ValueError(
            f"Conflicting policy config fields `{canonical_name}` and deprecated "
            f"`{legacy_name}`: {canonical_value!r} != {legacy_value!r}."
        )
    _warn_legacy_field(
        legacy_name=legacy_name,
        canonical_name=canonical_name,
        stacklevel=4,
    )
    return legacy_value


__all__ = [
    "DeprecatedPolicyConfigFieldWarning",
    "LEGACY_VIDEO_ACTION_POLICY_FIELD_ALIASES",
    "LEGACY_VIDEO_ACTION_POLICY_FIELDS",
    "normalize_video_action_config_fields",
    "normalize_video_action_decoder_fields",
    "normalize_video_action_override_keys",
    "normalize_video_action_policy_fields",
]
