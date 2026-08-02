from __future__ import annotations

import torch

__all__ = []


def _remap_packed_video_blocks_into_backbone(
    *,
    backbone_state_dict: dict[str, torch.Tensor],
    stack_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Move ``packed_blocks.{i}.video_block.*`` entries under ``blocks.{i}.*``.

    After ownership transfer in ``MoTPolicyVariant.attach_visual_tower``, the
    visual_tower core no longer owns its blocks; running ``state_dict()`` on
    the core therefore drops every ``blocks.{i}.*`` weight. The packed stack
    holds the canonical video block weights under
    ``packed_blocks.{i}.video_block.*``; this helper re-keys them so the
    exported runtime backbone state dict is a drop-in replacement for the
    pre-surgery layout. ``action_block.*`` entries are intentionally skipped —
    they belong to the action expert export path, not the video runtime
    backbone.
    """

    if any(key.startswith("blocks.") for key in backbone_state_dict):
        raise ValueError(
            "Runtime backbone state dict already contains `blocks.*` keys; "
            "packed-coupling remap would clobber them. Investigate why "
            "visual_tower.core kept its block weights despite the packed "
            "stack being attached."
        )
    remapped: dict[str, torch.Tensor] = dict(backbone_state_dict)
    prefix = "packed_blocks."
    video_marker = ".video_block."
    for key, tensor in stack_state_dict.items():
        if not key.startswith(prefix):
            continue
        marker_index = key.find(video_marker, len(prefix))
        if marker_index == -1:
            # action_block.* (or any other future child) — not part of the
            # video runtime backbone export.
            continue
        block_index_str = key[len(prefix) : marker_index]
        if not block_index_str.isdigit():
            continue
        suffix = key[marker_index + len(video_marker) :]
        remapped[f"blocks.{block_index_str}.{suffix}"] = tensor
    return remapped
