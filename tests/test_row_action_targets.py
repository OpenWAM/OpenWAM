from __future__ import annotations

from typing import Any

import pytest
import torch

from open_wam.configs import (
    ActionMappingConfig,
    ActionNormalizationConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    GenericDataConfig,
)
from open_wam.data import build_row_action_targets, resolve_row_key


def _pack_sequence(
    *,
    sequence: torch.Tensor,
    target_dim: int,
    target_length: int,
    left_pad: bool = False,
    sequence_name: str = "sequence",
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence.ndim != 2:
        raise ValueError(
            f"Expected {sequence_name} tensor with shape [T, D], got {tuple(sequence.shape)}."
        )
    output = torch.zeros(target_length, target_dim, dtype=torch.float32)
    mask = torch.zeros_like(output)
    clipped = sequence[:target_length]
    start = target_length - len(clipped) if left_pad else 0
    output[start : start + len(clipped), : sequence.shape[-1]] = clipped
    mask[start : start + len(clipped), : sequence.shape[-1]] = 1.0
    return output, mask


def _extract_sequence(
    *,
    rows: list[dict[str, Any]],
    key: str,
    target_dim: int,
    target_length: int,
    left_pad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.stack(
        [
            torch.tensor(row[resolve_row_key(row, key)], dtype=torch.float32)
            for row in rows
        ]
    )
    return _pack_sequence(
        sequence=sequence,
        target_dim=target_dim,
        target_length=target_length,
        left_pad=left_pad,
        sequence_name=key,
    )


def test_resolve_row_key_supports_singular_and_plural_schema_aliases() -> None:
    row = {"action": [1.0], "states": [2.0], "exact": [3.0]}

    assert resolve_row_key(row, "actions") == "action"
    assert resolve_row_key(row, "state") == "states"
    assert resolve_row_key(row, "exact") == "exact"
    with pytest.raises(KeyError, match="missing"):
        resolve_row_key(row, "missing")


def test_raw_targets_delegate_extraction_then_normalize_and_map() -> None:
    extraction_calls: list[tuple[str, int, int, bool]] = []

    def recording_extractor(
        *,
        rows: list[dict[str, Any]],
        key: str,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        extraction_calls.append((key, target_dim, target_length, left_pad))
        return _extract_sequence(
            rows=rows,
            key=key,
            target_dim=target_dim,
            target_length=target_length,
            left_pad=left_pad,
        )

    config = GenericDataConfig(
        action_schema=ActionSchemaConfig(
            action_dim=4,
            action_horizon=3,
            state_dim=8,
        ),
        action_target=ActionTargetConfig(
            representation="raw",
            source_key="actions",
            normalization=ActionNormalizationConfig(
                mode="gaussian",
                mean=(1.0, 0.0),
                std=(2.0, 2.0),
            ),
        ),
        action_mapping=ActionMappingConfig(
            mode="sparse_canvas",
            source_dim=2,
            target_dim=4,
            source_to_target_indices=(1, 3),
            active_target_indices=(1, 3),
        ),
    )

    actions, mask, metadata = build_row_action_targets(
        data_config=config,
        action_rows=[
            {"action": [3.0, -2.0]},
            {"action": [1.0, 2.0]},
        ],
        target_state_rows=[],
        extract_sequence=recording_extractor,
        pack_sequence=_pack_sequence,
        reference_source_subject="test rows",
    )

    assert extraction_calls == [("actions", 2, 3, False)]
    assert torch.equal(
        actions,
        torch.tensor(
            [
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(
        mask,
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert metadata["action_mapping_source_to_target_indices"] == [1, 3]
    assert metadata["action_target_normalization_mode"] == "gaussian"


def test_relative_pose_targets_use_adapter_owned_sequence_packing() -> None:
    packed_names: list[str] = []

    def recording_packer(
        *,
        sequence: torch.Tensor,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
        sequence_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed_names.append(sequence_name)
        return _pack_sequence(
            sequence=sequence,
            target_dim=target_dim,
            target_length=target_length,
            left_pad=left_pad,
            sequence_name=sequence_name,
        )

    config = GenericDataConfig(
        action_schema=ActionSchemaConfig(
            action_dim=7,
            action_horizon=2,
            state_dim=8,
        ),
        action_target=ActionTargetConfig(
            representation="eef_pose_relative_to_reference",
            source_key="actions",
            pose_source_key="state",
            state_encoding="eef_pos_axisangle_gripper_2d",
            rotation_representation="axis_angle",
            include_gripper=True,
            gripper_representation="first_channel",
        ),
    )

    actions, mask, metadata = build_row_action_targets(
        data_config=config,
        action_rows=[
            {"action": [0.0] * 7},
            {"action": [0.0] * 7},
        ],
        target_state_rows=[
            {"state": [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.2, 0.8]},
            {"state": [2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.4, 0.6]},
        ],
        extract_sequence=_extract_sequence,
        pack_sequence=recording_packer,
        reference_source_subject="test rows",
    )

    assert packed_names == ["sequence"]
    assert torch.allclose(
        actions,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.4],
            ]
        ),
    )
    assert torch.equal(mask, torch.ones(2, 7))
    assert metadata["reference_position"] == [1.0, 2.0, 3.0]
    assert metadata["action_mapping_applied"] is False


def test_absolute_joint_targets_append_action_command_and_preserve_mask() -> None:
    config = GenericDataConfig(
        action_schema=ActionSchemaConfig(
            action_dim=3,
            action_horizon=2,
            state_dim=4,
        ),
        action_target=ActionTargetConfig(
            representation="absolute_joint_position",
            source_key="actions",
            joint_position_source_key="robot0_joint_pos",
            include_gripper=True,
            gripper_representation="action_command",
            gripper_action_index=-1,
        ),
    )

    actions, mask, metadata = build_row_action_targets(
        data_config=config,
        action_rows=[
            {"action": [10.0, 0.5]},
            {"action": [11.0, -0.5]},
        ],
        target_state_rows=[
            {"robot0_joint_pos": [1.0, 2.0]},
            {"robot0_joint_pos": [3.0, 4.0]},
        ],
        extract_sequence=_extract_sequence,
        pack_sequence=_pack_sequence,
        reference_source_subject="test rows",
    )

    assert torch.equal(
        actions,
        torch.tensor(
            [
                [1.0, 2.0, 0.5],
                [3.0, 4.0, -0.5],
            ]
        ),
    )
    assert torch.equal(mask, torch.ones(2, 3))
    assert metadata["action_target_family"] == "absolute_joint_position"
    assert metadata["joint_position_source_key"] == "robot0_joint_pos"
    assert metadata["action_mapping_applied"] is False
