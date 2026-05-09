from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

from .enums import StrEnum

ModeEnumT = TypeVar("ModeEnumT", bound=StrEnum)


@dataclass(frozen=True)
class ProbabilityMapIssue:
    """Static-validation issue for one enum-backed probability map."""

    path_suffix: str | None
    message: str


GENERALIST_JOINT_CONDITIONING_DEFAULT_PROBS: dict[str, float] = {
    "joint": 0.6,
    "action_conditioned_video": 0.2,
    "video_conditioned_action": 0.2,
}

GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY = "generalist_training_mode_override"
GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY = "generalist_drop_text_conditioning"
GENERALIST_TRAINING_SOURCE_METADATA_KEY = "generalist_training_source"
GENERALIST_TRAINING_BUCKET_METADATA_KEY = "generalist_training_bucket"

JOINT_ONLY_CONDITIONING_DEFAULT_PROBS: dict[str, float] = {
    "joint": 1.0,
    "action_conditioned_video": 0.0,
    "video_conditioned_action": 0.0,
}


def default_video_action_conditioning_mode_probs(
    enum_cls: type[ModeEnumT],
    *,
    generalist: bool,
) -> dict[ModeEnumT, float]:
    """Return M1/M5 defaults for the three video/action conditioning modes.

    `enum_cls` must expose `joint`, `action_conditioned_video`, and
    `video_conditioned_action`. This helper is intentionally for the mirrored
    M1/M5 generalist semantics, not arbitrary enum-backed probabilities.
    """

    source = (
        GENERALIST_JOINT_CONDITIONING_DEFAULT_PROBS
        if generalist
        else JOINT_ONLY_CONDITIONING_DEFAULT_PROBS
    )
    return {enum_cls(mode): float(prob) for mode, prob in source.items()}


def default_conditioning_mode_probs(
    enum_cls: type[ModeEnumT],
    *,
    generalist: bool,
) -> dict[ModeEnumT, float]:
    """Compatibility alias for M1/M5 video/action conditioning defaults."""

    return default_video_action_conditioning_mode_probs(enum_cls, generalist=generalist)


def coerce_probability_map(
    raw_value: object,
    *,
    enum_cls: type[ModeEnumT],
    field_name: str,
) -> dict[ModeEnumT, float]:
    """Coerce and normalize an enum-backed probability map.

    Missing enum values default to 0. The returned probabilities always sum to
    one. Error messages intentionally include `field_name` so legacy M1/M5
    config tests keep the same public failure surface.
    """

    if not isinstance(raw_value, dict):
        raise ValueError(f"`{field_name}` must be a mapping from mode to probability.")

    probs = {mode: 0.0 for mode in enum_cls}
    for raw_mode, raw_prob in raw_value.items():
        try:
            mode = enum_cls(raw_mode)
        except ValueError as exc:
            valid_modes = ", ".join(mode.value for mode in enum_cls)
            raise ValueError(
                f"`{field_name}` contains invalid mode {raw_mode!r}; "
                f"expected one of: {valid_modes}."
            ) from exc
        if isinstance(raw_prob, bool):
            raise ValueError(
                f"`{field_name}` entries must be finite numeric probabilities, "
                f"got {mode.value}={raw_prob!r}."
            )
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"`{field_name}` entries must be finite numeric probabilities, "
                f"got {mode.value}={raw_prob!r}."
            ) from exc
        if not math.isfinite(prob):
            raise ValueError(
                f"`{field_name}` entries must be finite numeric probabilities, "
                f"got {mode.value}={raw_prob!r}."
            )
        if prob < 0.0:
            raise ValueError(
                f"`{field_name}` entries must be non-negative, "
                f"got {mode.value}={prob}."
            )
        probs[mode] = prob

    total = sum(probs.values())
    if total <= 0.0:
        raise ValueError(f"`{field_name}` must contain at least one positive probability.")
    return {mode: prob / total for mode, prob in probs.items()}


def probability_map_static_issues(
    raw_probs: object,
    *,
    enum_cls: type[StrEnum],
) -> tuple[ProbabilityMapIssue, ...]:
    """Return static-validation issues for an enum-backed probability map."""

    if raw_probs is None:
        return ()
    if not isinstance(raw_probs, Mapping):
        return (ProbabilityMapIssue(None, "Expected a mapping of mode to probability."),)

    issues: list[ProbabilityMapIssue] = []
    total = 0.0
    for raw_mode, raw_prob in raw_probs.items():
        if not isinstance(raw_mode, str):
            issues.append(ProbabilityMapIssue(None, "Expected string mode keys."))
            continue
        if raw_mode not in {mode.value for mode in enum_cls}:
            issues.append(
                ProbabilityMapIssue(
                    raw_mode,
                    f"Invalid {enum_cls.__name__} value {raw_mode!r}.",
                )
            )
            continue
        if isinstance(raw_prob, bool):
            issues.append(ProbabilityMapIssue(raw_mode, "Expected a numeric probability."))
            continue
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError):
            issues.append(ProbabilityMapIssue(raw_mode, "Expected a numeric probability."))
            continue
        if not math.isfinite(prob):
            issues.append(ProbabilityMapIssue(raw_mode, "Expected a finite probability."))
            continue
        if prob < 0.0:
            issues.append(ProbabilityMapIssue(raw_mode, "Expected a non-negative probability."))
            continue
        total += prob

    if total <= 0.0:
        issues.append(ProbabilityMapIssue(None, "Expected at least one positive probability."))
    return tuple(issues)
