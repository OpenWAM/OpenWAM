"""Module ownership for the Dual Expert packed and split layouts."""

from __future__ import annotations

from torch import nn

from open_wam.models.policy_variants.contracts import (
    PolicyModuleTopology,
    PolicyStateDictOverlay,
)
from open_wam.models.visual_tower import VisualTower

from .packed_block import DualExpertPackedBlockStack


def map_packed_video_block_export_key(key: str) -> str | None:
    """Map one packed video-block key back to the shared-backbone layout."""

    prefix = "packed_blocks."
    video_marker = ".video_block."
    if not key.startswith(prefix):
        return None
    marker_index = key.find(video_marker, len(prefix))
    if marker_index == -1:
        return None
    block_index = key[len(prefix) : marker_index]
    if not block_index.isdigit():
        return None
    suffix = key[marker_index + len(video_marker) :]
    return f"blocks.{block_index}.{suffix}"


def build_dual_expert_module_topology(
    *,
    visual_tower: VisualTower,
    action_expert: nn.Module,
    packed_block_stack: DualExpertPackedBlockStack | None,
) -> PolicyModuleTopology:
    """Describe Dual Expert module placement after optional block packing."""

    if packed_block_stack is None:
        return PolicyModuleTopology(
            visual_runtime_modules=(visual_tower.core,),
            action_expert_modules=(action_expert,),
            fsdp_block_stacks=(visual_tower.core, action_expert),
        )

    packed_blocks = tuple(packed_block_stack.packed_blocks)
    return PolicyModuleTopology(
        visual_runtime_modules=(
            visual_tower.core,
            *(block.video_block for block in packed_blocks),
        ),
        action_expert_modules=(
            action_expert,
            *(block.action_block for block in packed_blocks),
        ),
        fsdp_atomic_modules=packed_blocks,
        runtime_backbone_state_overlays=(
            PolicyStateDictOverlay(
                module=packed_block_stack,
                map_key=map_packed_video_block_export_key,
                exclusive_target_prefixes=("blocks.",),
            ),
        ),
    )


__all__ = [
    "build_dual_expert_module_topology",
    "map_packed_video_block_export_key",
]
