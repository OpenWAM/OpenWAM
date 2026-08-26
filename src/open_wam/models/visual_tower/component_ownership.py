"""State ownership utilities for semantic visual-tower components."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from open_wam.configs.enums import TrainingComponentSelector

from .contracts import VisualComponent, VisualComponentTopology


def owned_state_dict_keys(
    module: nn.Module,
    components: Iterable[VisualComponent],
    *,
    available_keys: Iterable[str] | None = None,
) -> frozenset[str]:
    """Return serializable state keys owned by the selected components."""

    owned_tensor_ids: set[int] = set()
    for component in components:
        if isinstance(component, nn.Parameter):
            owned_tensor_ids.add(id(component))
            continue
        owned_tensor_ids.update(
            id(parameter) for parameter in component.parameters(recurse=True)
        )
        owned_tensor_ids.update(
            id(buffer) for buffer in component.buffers(recurse=True)
        )

    keys = {
        name
        for name, parameter in module.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
        if id(parameter) in owned_tensor_ids
    }
    keys.update(
        name
        for name, buffer in module.named_buffers(
            recurse=True,
            remove_duplicate=False,
        )
        if id(buffer) in owned_tensor_ids
    )
    serializable_keys = (
        set(module.state_dict())
        if available_keys is None
        else {str(key) for key in available_keys}
    )
    return frozenset(keys & serializable_keys)


def resolve_visual_components(
    topology: VisualComponentTopology,
    selectors: Iterable[TrainingComponentSelector],
) -> tuple[VisualComponent, ...]:
    """Resolve narrow runtime-backbone selectors through visual ownership."""

    components: list[VisualComponent] = []
    for selector in selectors:
        if selector == TrainingComponentSelector.VISUAL_TOWER_SHARED_VIDEO_BACKBONE:
            selected = topology.shared_video_backbone
        elif selector == TrainingComponentSelector.VISUAL_TOWER_SHARED_ACTION_RUNTIME:
            selected = topology.shared_action_runtime
        elif selector == TrainingComponentSelector.VISUAL_TOWER_SHARED_RUNTIME_ADAPTERS:
            selected = topology.shared_runtime_adapters
        else:
            raise ValueError(
                "Scoped runtime-backbone artifacts require semantic visual component "
                f"selectors, got {selector.value!r}."
            )
        components.extend(selected)

    unique: list[VisualComponent] = []
    seen_ids: set[int] = set()
    for component in components:
        if id(component) in seen_ids:
            continue
        seen_ids.add(id(component))
        unique.append(component)
    return tuple(unique)


__all__ = ["owned_state_dict_keys", "resolve_visual_components"]
