"""Config contract for semantic runtime-backbone artifact components."""

from __future__ import annotations

from collections.abc import Iterable

from .enums import TrainingComponentSelector

RUNTIME_BACKBONE_COMPONENT_SELECTORS = frozenset(
    {
        TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,
        TrainingComponentSelector.VISUAL_TOWER_SHARED_VIDEO_BACKBONE,
        TrainingComponentSelector.VISUAL_TOWER_SHARED_ACTION_RUNTIME,
        TrainingComponentSelector.VISUAL_TOWER_SHARED_RUNTIME_ADAPTERS,
    }
)


def validate_runtime_backbone_components(
    values: Iterable[TrainingComponentSelector | str],
    *,
    scope: str,
) -> tuple[TrainingComponentSelector, ...]:
    """Normalize and validate one complete or scoped component selection."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{scope} must be a sequence of component selectors.")
    components = tuple(TrainingComponentSelector(value) for value in values)
    if not components:
        raise ValueError(f"{scope} must contain at least one component.")
    invalid = tuple(
        component
        for component in components
        if component not in RUNTIME_BACKBONE_COMPONENT_SELECTORS
    )
    if invalid:
        rendered = ", ".join(component.value for component in invalid)
        raise ValueError(
            f"{scope} accepts only runtime-backbone semantic groups, got: {rendered}."
        )
    if len(set(components)) != len(components):
        raise ValueError(f"{scope} must not contain duplicates.")
    if (
        TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE in components
        and len(components) != 1
    ):
        raise ValueError(
            "The complete runtime-backbone component cannot be combined with "
            "narrow component selectors."
        )
    return components


def is_complete_runtime_backbone_selection(
    components: Iterable[TrainingComponentSelector],
) -> bool:
    """Return whether a validated selection denotes the complete backbone."""

    return tuple(components) == (
        TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,
    )


__all__ = [
    "RUNTIME_BACKBONE_COMPONENT_SELECTORS",
    "is_complete_runtime_backbone_selection",
    "validate_runtime_backbone_components",
]
