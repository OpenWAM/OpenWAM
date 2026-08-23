"""Architecture-neutral state-dict overlay composition."""

from __future__ import annotations

from collections.abc import Callable

import torch


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


__all__ = ["merge_state_dict_overlay"]
