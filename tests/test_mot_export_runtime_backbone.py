"""Verify ``_export_runtime_backbone`` remap for the M5 packed-coupling path.

After ``MoTPolicyVariant.attach_visual_tower`` transfers ownership of
``visual_tower.core.blocks`` and ``action_expert.blocks`` into a
``MoTPackedBlockStack``, the bare ``visual_tower.core.state_dict()`` no longer
contains any ``blocks.*`` entries. The exporter therefore needs to re-key
``packed_blocks.{i}.video_block.*`` back into ``blocks.{i}.*`` so the saved
``transformer/`` directory is a drop-in replacement for the pre-surgery
layout consumed by LingBot loaders / method-1 visualization scripts.
"""

from __future__ import annotations

import torch

from open_wam.training.checkpoints import _remap_packed_video_blocks_into_backbone


def _make_backbone_state_dict() -> dict[str, torch.Tensor]:
    return {
        "patch_embed.proj.weight": torch.arange(8.0).reshape(2, 4),
        "patch_embed.proj.bias": torch.tensor([0.5, -0.5]),
        "norm_out.weight": torch.tensor([1.0, 1.0, 1.0]),
        "scale_shift_table": torch.full((1, 2, 3), 0.25),
    }


def _make_stack_state_dict() -> dict[str, torch.Tensor]:
    return {
        # Block 0 video weights — should land at blocks.0.*
        "packed_blocks.0.video_block.attn1.to_q.weight": torch.full((4, 4), 1.0),
        "packed_blocks.0.video_block.attn1.to_k.weight": torch.full((4, 4), 2.0),
        "packed_blocks.0.video_block.ffn.net.0.proj.weight": torch.full((8, 4), 3.0),
        # Block 0 action weights — should be ignored by the video remap.
        "packed_blocks.0.action_block.attn1.to_q.weight": torch.full((2, 2), -1.0),
        "packed_blocks.0.action_block.ffn.net.0.proj.weight": torch.full((4, 2), -2.0),
        # Block 1 video weights — should land at blocks.1.*
        "packed_blocks.1.video_block.attn1.to_q.weight": torch.full((4, 4), 4.0),
        "packed_blocks.1.video_block.attn2.to_v.weight": torch.full((4, 4), 5.0),
    }


def test_remap_promotes_video_blocks_into_blocks_namespace() -> None:
    backbone_state = _make_backbone_state_dict()
    stack_state = _make_stack_state_dict()
    remapped = _remap_packed_video_blocks_into_backbone(
        backbone_state_dict=backbone_state,
        stack_state_dict=stack_state,
    )
    assert remapped["blocks.0.attn1.to_q.weight"] is stack_state[
        "packed_blocks.0.video_block.attn1.to_q.weight"
    ]
    assert remapped["blocks.0.attn1.to_k.weight"] is stack_state[
        "packed_blocks.0.video_block.attn1.to_k.weight"
    ]
    assert remapped["blocks.0.ffn.net.0.proj.weight"] is stack_state[
        "packed_blocks.0.video_block.ffn.net.0.proj.weight"
    ]
    assert remapped["blocks.1.attn1.to_q.weight"] is stack_state[
        "packed_blocks.1.video_block.attn1.to_q.weight"
    ]
    assert remapped["blocks.1.attn2.to_v.weight"] is stack_state[
        "packed_blocks.1.video_block.attn2.to_v.weight"
    ]


def test_remap_skips_action_block_entries() -> None:
    backbone_state = _make_backbone_state_dict()
    stack_state = _make_stack_state_dict()
    remapped = _remap_packed_video_blocks_into_backbone(
        backbone_state_dict=backbone_state,
        stack_state_dict=stack_state,
    )
    for key in remapped:
        assert "action_block" not in key, key
        assert "packed_blocks" not in key, key


def test_remap_preserves_non_block_backbone_entries() -> None:
    backbone_state = _make_backbone_state_dict()
    stack_state = _make_stack_state_dict()
    remapped = _remap_packed_video_blocks_into_backbone(
        backbone_state_dict=backbone_state,
        stack_state_dict=stack_state,
    )
    for key, original in backbone_state.items():
        assert remapped[key] is original


def test_remap_does_not_mutate_inputs() -> None:
    backbone_state = _make_backbone_state_dict()
    stack_state = _make_stack_state_dict()
    backbone_keys_before = set(backbone_state)
    stack_keys_before = set(stack_state)
    _remap_packed_video_blocks_into_backbone(
        backbone_state_dict=backbone_state,
        stack_state_dict=stack_state,
    )
    assert set(backbone_state) == backbone_keys_before
    assert set(stack_state) == stack_keys_before


def test_remap_rejects_backbone_state_with_existing_block_keys() -> None:
    backbone_state = _make_backbone_state_dict()
    backbone_state["blocks.0.attn1.to_q.weight"] = torch.zeros(4, 4)
    stack_state = _make_stack_state_dict()
    try:
        _remap_packed_video_blocks_into_backbone(
            backbone_state_dict=backbone_state,
            stack_state_dict=stack_state,
        )
    except ValueError as exc:
        assert "blocks.*" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when backbone state already contains blocks.* keys."
        )


def test_remap_skips_keys_with_unexpected_layout() -> None:
    backbone_state = _make_backbone_state_dict()
    # Mix in a rogue key that does not match the packed_blocks.{i}.video_block.*
    # pattern; it must be silently dropped (it isn't part of the video backbone).
    stack_state = {
        "packed_blocks.foo.video_block.attn1.to_q.weight": torch.zeros(4, 4),
        "packed_blocks.0.something_else.attn1.to_q.weight": torch.zeros(4, 4),
        "packed_blocks.0.video_block.attn1.to_q.weight": torch.full((4, 4), 7.0),
    }
    remapped = _remap_packed_video_blocks_into_backbone(
        backbone_state_dict=backbone_state,
        stack_state_dict=stack_state,
    )
    assert remapped["blocks.0.attn1.to_q.weight"] is stack_state[
        "packed_blocks.0.video_block.attn1.to_q.weight"
    ]
    assert "blocks.foo.attn1.to_q.weight" not in remapped
    assert "blocks.0.something_else.attn1.to_q.weight" not in remapped
