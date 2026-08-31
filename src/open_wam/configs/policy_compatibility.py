"""Input-boundary compatibility for retired video/action config names.

Canonical config dataclasses and runtime code do not depend on historical
architecture labels. Naming aliases remain accepted with warnings. Retired
semantic fields are translated only for immutable checkpoint metadata.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from enum import Enum
from typing import Any

from .enums import (
    ActionDecoderName,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    PolicyVariantName,
    VideoActionSequenceContract,
)

LEGACY_VIDEO_ACTION_POLICY_FIELD_ALIASES = {
    "parallel_sequence_contract": "sequence_contract",
}
_RETIRED_NOOP_VIDEO_ACTION_POLICY_FIELDS = {
    "use_state_conditioning": (
        "use `proprio_context_mode` or a `sequence_contract` that owns it"
    ),
    "use_text_conditioning": (
        "use the policy program/objective text-conditioning semantics"
    ),
}
LEGACY_VIDEO_ACTION_POLICY_FIELDS = frozenset(
    {
        *LEGACY_VIDEO_ACTION_POLICY_FIELD_ALIASES,
        *_RETIRED_NOOP_VIDEO_ACTION_POLICY_FIELDS,
        "couple_action_to_video_timesteps",
        "preserve_video_pretrain_history",
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


def _warn_ignored_noop_field(
    *, legacy_name: str, replacement: str, stacklevel: int
) -> None:
    warnings.warn(
        f"Policy config field `{legacy_name}` is deprecated and had no runtime "
        f"effect; ignoring its checkpoint value to preserve historical behavior. "
        f"For authored configs, {replacement}.",
        DeprecatedPolicyConfigFieldWarning,
        stacklevel=stacklevel,
    )


def _normalize_video_action_policy_name(
    raw_policy: Mapping[str, Any],
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Return a copy with naming-only policy aliases canonicalized."""

    normalized = dict(raw_policy)
    if normalized.get("name") == "mot":
        normalized["name"] = PolicyVariantName.DUAL_EXPERT
        if warn:
            warnings.warn(
                "Policy architecture `mot` is deprecated; use `dual_expert`.",
                DeprecatedPolicyConfigFieldWarning,
                stacklevel=3,
            )
    return normalized


def _retired_semantic_fields(raw_policy: Mapping[str, Any]) -> tuple[str, ...]:
    semantic_fields = LEGACY_VIDEO_ACTION_POLICY_FIELDS.difference(
        _RETIRED_NOOP_VIDEO_ACTION_POLICY_FIELDS
    )
    return tuple(sorted(semantic_fields.intersection(raw_policy)))


def normalize_video_action_policy_fields(
    raw_policy: Mapping[str, Any],
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Canonicalize names and reject retired authored semantic fields."""

    noop_fields = tuple(
        sorted(_RETIRED_NOOP_VIDEO_ACTION_POLICY_FIELDS.keys() & raw_policy.keys())
    )
    if noop_fields:
        fields = ", ".join(f"policy_variant.{name}" for name in noop_fields)
        replacements = "; ".join(
            _RETIRED_NOOP_VIDEO_ACTION_POLICY_FIELDS[name] for name in noop_fields
        )
        status = (
            "is a retired no-op field"
            if len(noop_fields) == 1
            else "are retired no-op fields"
        )
        subject_pronoun = "it" if len(noop_fields) == 1 else "they"
        pronoun = "it" if len(noop_fields) == 1 else "them"
        raise ValueError(
            f"{fields} {status} in authored configs because "
            f"{subject_pronoun} never controlled runtime behavior. "
            f"Remove {pronoun} and {replacements}. "
            "Historical checkpoint metadata is accepted only with "
            "`checkpoint_runtime_compat=True`."
        )
    retired_fields = _retired_semantic_fields(raw_policy)
    if retired_fields:
        fields = ", ".join(f"policy_variant.{name}" for name in retired_fields)
        raise ValueError(
            f"{fields} are retired in authored configs. Use the canonical "
            "`sequence_contract`, `joint_timestep_coupling`, and "
            "`history_stream_visibility` fields; historical fields are accepted "
            "only when loading checkpoint metadata with "
            "`checkpoint_runtime_compat=True`."
        )
    return _normalize_video_action_policy_name(raw_policy, warn=warn)


def _migrate_checkpoint_video_action_policy_fields(
    raw_policy: Mapping[str, Any],
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Translate retired semantic fields in immutable checkpoint metadata."""

    normalized = _normalize_video_action_policy_name(raw_policy, warn=warn)
    for legacy_name, replacement in _RETIRED_NOOP_VIDEO_ACTION_POLICY_FIELDS.items():
        if legacy_name not in normalized:
            continue
        normalized.pop(legacy_name)
        if warn:
            _warn_ignored_noop_field(
                legacy_name=legacy_name,
                replacement=replacement,
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

    legacy_history_key = "preserve_video_pretrain_history"
    if legacy_history_key in normalized:
        legacy_value = normalized.pop(legacy_history_key)
        if not isinstance(legacy_value, bool):
            raise TypeError(
                f"Deprecated `{legacy_history_key}` must be a boolean, "
                f"got {legacy_value!r}."
            )
        contract = VideoActionSequenceContract(
            _plain_config_value(
                normalized.get(
                    "sequence_contract",
                    VideoActionSequenceContract.DEFAULT,
                )
            )
        )
        canonical_name = "history_stream_visibility"
        if (
            legacy_value
            and contract == VideoActionSequenceContract.DEFAULT
            and canonical_name not in normalized
        ):
            normalized[canonical_name] = (
                HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
            )
        if warn:
            _warn_legacy_field(
                legacy_name=legacy_history_key,
                canonical_name="history_stream_visibility",
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


def migrate_checkpoint_video_action_config_fields(
    raw_config: Mapping[str, Any],
    *,
    warn: bool = True,
) -> dict[str, Any]:
    """Canonicalize names and retired fields in checkpoint metadata only."""

    normalized = dict(raw_config)
    raw_policy = normalized.get("policy_variant")
    if isinstance(raw_policy, Mapping):
        normalized["policy_variant"] = (
            _migrate_checkpoint_video_action_policy_fields(raw_policy, warn=warn)
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


__all__ = [
    "DeprecatedPolicyConfigFieldWarning",
    "LEGACY_VIDEO_ACTION_POLICY_FIELD_ALIASES",
    "LEGACY_VIDEO_ACTION_POLICY_FIELDS",
    "migrate_checkpoint_video_action_config_fields",
    "normalize_video_action_config_fields",
    "normalize_video_action_decoder_fields",
    "normalize_video_action_override_keys",
    "normalize_video_action_policy_fields",
]
