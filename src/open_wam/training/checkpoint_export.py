"""Architecture-neutral runtime-backbone export composition."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from open_wam.configs.enums import TrainingComponentSelector
from open_wam.configs.runtime_backbone_components import (
    is_complete_runtime_backbone_selection,
    validate_runtime_backbone_components,
)
from open_wam.models.policy_variants.contracts import PolicyModuleTopology
from open_wam.models.visual_tower.component_ownership import (
    owned_state_dict_keys,
    resolve_visual_components,
)


def merge_state_dict_overlay(
    *,
    base_state_dict: dict[str, torch.Tensor],
    overlay_state_dict: dict[str, torch.Tensor],
    map_key: Callable[[str], str | None],
    exclusive_target_prefixes: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Project one policy-owned module state into an exported backbone state."""

    conflicting_prefixes = tuple(
        prefix
        for prefix in exclusive_target_prefixes
        if any(key.startswith(prefix) for key in base_state_dict)
    )
    if conflicting_prefixes:
        raise ValueError(
            "Runtime backbone state already owns keys reserved for a policy "
            f"overlay: {', '.join(conflicting_prefixes)}."
        )

    merged = dict(base_state_dict)
    for key, tensor in overlay_state_dict.items():
        target_key = map_key(key)
        if target_key is None:
            continue
        if target_key in merged:
            raise ValueError(
                f"Runtime backbone state overlay would replace existing key {target_key!r}."
            )
        merged[target_key] = tensor
    return merged


def resolve_runtime_backbone_export_keys(
    *,
    backbone: nn.Module,
    topology: PolicyModuleTopology,
    selectors: Iterable[TrainingComponentSelector],
) -> frozenset[str] | None:
    """Resolve canonical scoped-export keys before distributed wrapping.

    ``None`` denotes a complete runtime-backbone export. Narrow exports retain
    canonical state-dict names here so later FSDP and activation-checkpoint
    wrappers cannot change semantic component ownership.
    """

    resolved_selectors = validate_runtime_backbone_components(
        selectors,
        scope="Runtime-backbone export components",
    )
    if is_complete_runtime_backbone_selection(resolved_selectors):
        return None

    selected_components = resolve_visual_components(
        topology.visual_components,
        resolved_selectors,
    )
    selected_keys = set(owned_state_dict_keys(backbone, selected_components))
    for overlay in topology.runtime_backbone_state_overlays:
        for source_key in owned_state_dict_keys(
            overlay.module,
            selected_components,
        ):
            target_key = overlay.map_key(source_key)
            if target_key is not None:
                selected_keys.add(target_key)
    if not selected_keys:
        raise ValueError("Runtime-backbone export selectors resolved no state tensors.")
    return frozenset(selected_keys)


__all__ = ["merge_state_dict_overlay", "resolve_runtime_backbone_export_keys"]
