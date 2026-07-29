from __future__ import annotations

from typing import Any, Protocol

import torch

from open_wam.configs import (
    ActionTargetReferenceSource,
    ActionTargetRepresentation,
    DataConfig,
    GripperRepresentation,
)

from .action_mapping import (
    action_mapping_is_active,
    apply_action_mapping,
    resolve_action_source_dim,
)
from .action_transforms import (
    build_absolute_joint_position_targets,
    build_relative_pose_targets,
    expected_joint_position_target_dim,
    expected_pose_target_dim,
    normalize_action_targets,
)
from .sequence_packing import pack_temporal_sequence


class RowSequenceExtractor(Protocol):
    """Adapter-owned conversion from source rows to one padded sequence."""

    def __call__(
        self,
        *,
        rows: list[dict[str, Any]],
        key: str,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class SequencePacker(Protocol):
    """Adapter-owned sequence padding and mask construction."""

    def __call__(
        self,
        *,
        sequence: torch.Tensor,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
        sequence_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def resolve_row_key(row: dict[str, Any], key: str) -> str:
    """Resolve the singular/plural aliases used by LeRobot row schemas."""

    if key in row:
        return key
    if key.endswith("s") and key[:-1] in row:
        return key[:-1]
    plural_candidate = f"{key}s"
    if plural_candidate in row:
        return plural_candidate
    raise KeyError(key)


def build_row_action_targets(
    *,
    data_config: DataConfig,
    action_rows: list[dict[str, Any]],
    target_state_rows: list[dict[str, Any]],
    extract_sequence: RowSequenceExtractor,
    pack_sequence: SequencePacker = pack_temporal_sequence,
    reference_source_subject: str = "Row-oriented datasets",
    relative_sequence_name: str = "sequence",
    include_pose_dimension_context: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build common WAM action supervision from row-oriented robot data.

    Source adapters retain row loading through ``extract_sequence`` and may
    override the shared sequence packer when their storage contract requires
    it. This transform owns representation conversion, action-channel mapping,
    normalization, and target metadata only.

    Relative-EEF and absolute-joint rows must span the configured action
    horizon because their source-validity masks are copied without temporal
    padding. Raw targets retain the adapter callback's padding policy.
    """

    action_target = data_config.action_target
    action_mapping = data_config.action_mapping
    target_dim = data_config.action_schema.action_dim
    target_length = data_config.action_schema.action_horizon

    if action_target.representation == ActionTargetRepresentation.RAW:
        source_dim = resolve_action_source_dim(action_mapping, fallback_dim=target_dim)
        actions, action_mask = extract_sequence(
            rows=action_rows,
            key=action_target.source_key,
            target_dim=source_dim,
            target_length=target_length,
        )
        actions = normalize_action_targets(
            actions,
            normalization=action_target.normalization,
        )
        mapped = apply_action_mapping(
            actions,
            action_mask,
            action_mapping,
            target_dim=target_dim,
        )
        metadata = dict(mapped.metadata)
        metadata["action_target_normalization_mode"] = str(action_target.normalization.mode)
        return mapped.actions, mapped.action_mask, metadata

    if action_target.representation == ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
        if action_target.reference_source != ActionTargetReferenceSource.ANCHOR_STATE:
            raise ValueError(
                f"{reference_source_subject} currently support only "
                f"`reference_source=anchor_state`, got {action_target.reference_source}."
            )
        pose_source = torch.stack(
            [
                torch.tensor(row[resolve_row_key(row, action_target.pose_source_key)], dtype=torch.float32)
                for row in target_state_rows
            ],
            dim=0,
        )
        raw_action_sequence = torch.stack(
            [
                torch.tensor(row[resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                for row in action_rows
            ],
            dim=0,
        )
        relative_targets, relative_mask, metadata = build_relative_pose_targets(
            pose_source,
            state_encoding=action_target.state_encoding,
            rotation_representation=action_target.rotation_representation,
            include_gripper=action_target.include_gripper,
            gripper_representation=action_target.gripper_representation,
            raw_action_sequence=raw_action_sequence,
            gripper_action_index=action_target.gripper_action_index,
        )
        expected_dim = expected_pose_target_dim(
            rotation_representation=action_target.rotation_representation,
            include_gripper=action_target.include_gripper,
            gripper_representation=action_target.gripper_representation,
        )
        target_or_source_dim = resolve_action_source_dim(action_mapping, fallback_dim=target_dim)
        if target_or_source_dim != expected_dim:
            message = (
                "Configured action_dim does not match the derived pose-target dimension: "
                f"configured_dim={target_or_source_dim}, expected={expected_dim}"
            )
            if include_pose_dimension_context:
                message += (
                    " for "
                    f"[rotation_representation={action_target.rotation_representation}, "
                    f"gripper_representation={action_target.gripper_representation}]."
                )
            else:
                message += "."
            raise ValueError(message)
        metadata.update(
            {
                "reference_source": action_target.reference_source,
                "pose_source_key": action_target.pose_source_key,
                "gripper_source_key": action_target.source_key,
            }
        )
        actions, action_mask = pack_sequence(
            sequence=relative_targets,
            target_dim=target_or_source_dim,
            target_length=target_length,
            sequence_name=relative_sequence_name,
        )
        if relative_mask.shape[-1] != relative_targets.shape[-1]:
            raise ValueError("Relative target mask shape must match the relative target tensor shape.")
        action_mask[:, : relative_mask.shape[-1]] = relative_mask
        mapped = apply_action_mapping(
            actions,
            action_mask,
            action_mapping,
            target_dim=target_dim,
        )
        metadata.update(mapped.metadata)
        metadata["action_mapping_applied"] = action_mapping_is_active(action_mapping)
        return mapped.actions, mapped.action_mask, metadata

    if action_target.representation == ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION:
        joint_position_source = torch.stack(
            [
                torch.tensor(
                    row[resolve_row_key(row, action_target.joint_position_source_key)],
                    dtype=torch.float32,
                )
                for row in target_state_rows
            ],
            dim=0,
        )
        raw_action_sequence = torch.stack(
            [
                torch.tensor(row[resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                for row in action_rows
            ],
            dim=0,
        )
        gripper_position_sequence = None
        if (
            action_target.include_gripper
            and action_target.gripper_representation != GripperRepresentation.ACTION_COMMAND
        ):
            gripper_position_sequence = torch.stack(
                [
                    torch.tensor(
                        row[resolve_row_key(row, action_target.gripper_position_source_key)],
                        dtype=torch.float32,
                    )
                    for row in target_state_rows
                ],
                dim=0,
            )
        joint_targets, joint_mask, metadata = build_absolute_joint_position_targets(
            joint_position_source,
            include_gripper=action_target.include_gripper,
            gripper_representation=action_target.gripper_representation,
            gripper_position_sequence=gripper_position_sequence,
            raw_action_sequence=raw_action_sequence,
            gripper_action_index=action_target.gripper_action_index,
            normalization=action_target.joint_position_normalization,
        )
        expected_dim = expected_joint_position_target_dim(
            joint_dim=joint_position_source.shape[-1],
            include_gripper=action_target.include_gripper,
            gripper_representation=action_target.gripper_representation,
        )
        target_or_source_dim = resolve_action_source_dim(action_mapping, fallback_dim=target_dim)
        if target_or_source_dim != expected_dim:
            raise ValueError(
                "Configured action_dim does not match the derived absolute-joint target dimension: "
                f"configured_dim={target_or_source_dim}, expected={expected_dim}."
            )
        metadata.update(
            {
                "joint_position_source_key": action_target.joint_position_source_key,
                "gripper_source_key": action_target.source_key,
            }
        )
        actions, action_mask = pack_sequence(
            sequence=joint_targets,
            target_dim=target_or_source_dim,
            target_length=target_length,
            sequence_name="absolute_joint_position_targets",
        )
        if joint_mask.shape[-1] != joint_targets.shape[-1]:
            raise ValueError("Absolute-joint target mask shape must match the target tensor shape.")
        action_mask[:, : joint_mask.shape[-1]] = joint_mask
        mapped = apply_action_mapping(
            actions,
            action_mask,
            action_mapping,
            target_dim=target_dim,
        )
        metadata.update(mapped.metadata)
        metadata["action_mapping_applied"] = action_mapping_is_active(action_mapping)
        return mapped.actions, mapped.action_mask, metadata

    raise ValueError(f"Unsupported action target representation: {action_target.representation}")


__all__ = [
    "RowSequenceExtractor",
    "SequencePacker",
    "build_row_action_targets",
    "resolve_row_key",
]
