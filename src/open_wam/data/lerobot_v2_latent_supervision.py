from __future__ import annotations

from functools import partial
import math
from typing import Any

import torch

from open_wam.configs import (
    ActionTargetRepresentation,
    DataConfig,
    GripperRepresentation,
)

from .action_normalization import normalize_action_targets
from .action_pose import pose_blocks_relative_to_anchor
from .action_target_builders import (
    build_absolute_joint_position_targets,
    expected_joint_position_target_dim,
)
from .lerobot_v2_latent_storage import LocalEpisodeWindow
from .row_action_targets import build_row_action_targets, resolve_row_key
from .sequence_packing import pack_temporal_sequence


_TRUNCATING_SEQUENCE_PACKER = partial(
    pack_temporal_sequence,
    truncate_to_target_length=True,
)

_POSE_BLOCK_DIMS = 10  # [xyz(3), rot6(6), gripper(1)] per arm.


class LocalLatentSupervisionAssembler:
    """Build local latent action and proprio supervision from decoded rows."""

    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config

    def build_standard_policy_window_action_targets(
        self,
        *,
        rows: list[dict[str, Any]],
        observation_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_horizon = int(self.data_config.action_schema.action_horizon)
        action_rows = rows[observation_start : observation_start + action_horizon]
        target_state_rows = rows[observation_start : observation_start + action_horizon]
        actions, action_mask, metadata = self.build_action_targets(
            action_rows=action_rows,
            target_state_rows=target_state_rows,
        )
        metadata = dict(metadata)
        metadata["latent_window_profile"] = self.data_config.latent_window_profile
        return actions, action_mask, metadata

    def build_lingbot_window_action_targets(
        self,
        *,
        rows: list[dict[str, Any]],
        window: LocalEpisodeWindow,
        observed_frame_ids: list[int],
        latent_num_frames: int,
        leading_zero_action_frames: int = 1,
        leading_zero_action_mask: float = 1.0,
        proprio_chunk_size: int = 1,
        proprio_loss_frame_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_target = self.data_config.action_target
        if action_target.representation not in {
            ActionTargetRepresentation.RAW,
            ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION,
        }:
            raise ValueError(
                "Long-window local latent datasets currently support only "
                "`action_target.representation=raw` or `absolute_joint_position` for LingBot-compatible exact "
                "training."
            )
        if latent_num_frames <= 0:
            raise ValueError("Expected at least one latent frame in the local latent window.")
        if not observed_frame_ids:
            raise ValueError("Expected non-empty frame_ids metadata for the local latent window.")

        frame_stride = 1
        if len(observed_frame_ids) > 1:
            frame_stride = max(1, int(observed_frame_ids[1] - observed_frame_ids[0]))
        prefix_actions = int(self.data_config.action_schema.action_horizon // max(1, self.data_config.num_frames))
        required_action_num = latent_num_frames * prefix_actions
        leading_zero_action_frames = max(0, int(leading_zero_action_frames))
        leading_action_steps = leading_zero_action_frames * prefix_actions

        action_start_offset = max(0, int(observed_frame_ids[0] - window.start_frame))
        raw_window_rows = rows[window.start_frame : window.end_frame]
        aligned_rows = raw_window_rows[action_start_offset:]
        if action_target.representation == ActionTargetRepresentation.RAW:
            source_actions = torch.stack(
                [
                    torch.tensor(row[resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                    for row in aligned_rows
                ],
                dim=0,
            )
            source_actions = normalize_action_targets(
                source_actions,
                normalization=action_target.normalization,
            )
            source_mask = torch.ones_like(source_actions, dtype=torch.float32)
            action_dim = source_actions.shape[-1]
            target_family_metadata: dict[str, Any] = {
                "action_target_normalization_mode": str(action_target.normalization.mode),
            }
            block_dims = action_target.relative_pose_block_dims
            if block_dims:
                source_actions = self._anchor_relative_chunk_targets(
                    source_actions,
                    rows=rows,
                    observed_frame_ids=observed_frame_ids,
                    latent_num_frames=latent_num_frames,
                    prefix_actions=prefix_actions,
                    leading_action_steps=leading_action_steps,
                    block_dims=int(block_dims),
                    proprio_chunk_size=int(proprio_chunk_size),
                    proprio_loss_frame_start=int(proprio_loss_frame_start),
                )
            target_family_metadata["relative_pose_block_dims"] = int(block_dims) if block_dims else None
        else:
            joint_position_source = torch.stack(
                [
                    torch.tensor(row[resolve_row_key(row, action_target.joint_position_source_key)], dtype=torch.float32)
                    for row in aligned_rows
                ],
                dim=0,
            )
            raw_action_sequence = torch.stack(
                [
                    torch.tensor(row[resolve_row_key(row, action_target.source_key)], dtype=torch.float32)
                    for row in aligned_rows
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
                        for row in aligned_rows
                    ],
                    dim=0,
                )
            source_actions, source_mask, target_family_metadata = build_absolute_joint_position_targets(
                joint_position_source,
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
                gripper_position_sequence=gripper_position_sequence,
                raw_action_sequence=raw_action_sequence,
                gripper_action_index=action_target.gripper_action_index,
                normalization=action_target.joint_position_normalization,
            )
            action_dim = source_actions.shape[-1]
            expected_dim = expected_joint_position_target_dim(
                joint_dim=joint_position_source.shape[-1],
                include_gripper=action_target.include_gripper,
                gripper_representation=action_target.gripper_representation,
            )
            if action_dim != expected_dim:
                raise ValueError(
                    "Derived absolute-joint target dim mismatch: "
                    f"derived={action_dim}, expected={expected_dim}."
                )
        if action_dim != self.data_config.action_schema.action_dim:
            raise ValueError(
                "Configured action_dim does not match local latent supervision: "
                f"configured={self.data_config.action_schema.action_dim}, source={action_dim}."
            )

        leading_fill = torch.zeros(leading_action_steps, action_dim, dtype=torch.float32)
        leading_mask = torch.ones_like(leading_fill, dtype=torch.float32)
        if action_target.representation == ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION and leading_action_steps > 0:
            leading_fill = source_actions[0:1].expand(leading_action_steps, -1).contiguous()
            leading_mask = source_mask[0:1].expand(leading_action_steps, -1).contiguous()

        padded_actions = torch.cat(
            [
                leading_fill,
                source_actions,
            ],
            dim=0,
        )
        padded_mask = torch.cat([leading_mask, source_mask], dim=0)
        if padded_actions.shape[0] < required_action_num:
            padded_actions = torch.cat(
                [
                    padded_actions,
                    torch.zeros(required_action_num - padded_actions.shape[0], action_dim, dtype=torch.float32),
                ],
                dim=0,
            )
            padded_mask = torch.cat(
                [
                    padded_mask,
                    torch.zeros(required_action_num - padded_mask.shape[0], action_dim, dtype=torch.float32),
                ],
                dim=0,
            )
        actions = padded_actions[:required_action_num].contiguous()

        action_mask = padded_mask[:required_action_num].contiguous()
        if leading_action_steps > 0 and float(leading_zero_action_mask) <= 0.0:
            action_mask[:leading_action_steps] = 0.0
        if source_actions.shape[0] + leading_action_steps < required_action_num:
            action_mask[source_actions.shape[0] + leading_action_steps :] = 0.0
        return actions, action_mask, {
            "lingbot_window_action_alignment": {
                "latent_num_frames": latent_num_frames,
                "raw_frame_count": len(observed_frame_ids),
                "frame_stride": frame_stride,
                "prefix_actions": prefix_actions,
                "required_action_num": required_action_num,
                "action_start_offset": action_start_offset,
                "leading_zero_action_frames": leading_zero_action_frames,
                "leading_zero_action_steps": leading_action_steps,
                "leading_zero_action_mask": float(leading_zero_action_mask),
            },
            **target_family_metadata,
        }

    def _anchor_relative_chunk_targets(
        self,
        source_actions: torch.Tensor,
        *,
        rows: list[dict[str, Any]],
        observed_frame_ids: list[int],
        latent_num_frames: int,
        prefix_actions: int,
        leading_action_steps: int,
        block_dims: int,
        proprio_chunk_size: int,
        proprio_loss_frame_start: int,
    ) -> torch.Tensor:
        """Use each chunk's absolute observation state as its action anchor.

        Chunk size and loss origin must match the dataset's proprio context
        geometry. The context may be lagged motion, but the anchor must remain
        an absolute pose or the action transform silently uses the wrong frame.
        """

        if prefix_actions <= 0:
            raise ValueError("Anchor-relative action targets need a positive per-frame action count.")
        if leading_action_steps % prefix_actions:
            raise ValueError(
                "Anchor-relative action targets need the leading zero pad to land on a chunk "
                f"boundary, got leading_action_steps={leading_action_steps} with "
                f"prefix_actions={prefix_actions}."
            )
        chunk_size = max(1, int(proprio_chunk_size))
        anchors, _ = self.extract_proprio_context_state_sequence(
            rows=rows,
            observed_frame_ids=observed_frame_ids,
            chunk_size=chunk_size,
            loss_frame_start=int(proprio_loss_frame_start),
            apply_history_lag=False,
        )
        chunk_count = int(anchors.shape[0])
        if chunk_count <= 0:
            raise ValueError("Anchor-relative action targets need at least one proprio anchor.")
        if int(anchors.shape[-1]) != int(source_actions.shape[-1]):
            raise ValueError(
                "`relative_pose_block_dims` needs the anchor state and the action target to share "
                f"a channel layout, got state_dim={int(anchors.shape[-1])} and "
                f"action_dim={int(source_actions.shape[-1])}."
            )
        relative = source_actions.clone()
        leading_frames = leading_action_steps // prefix_actions
        total_steps = int(source_actions.shape[0])
        for offset in range(0, total_steps, prefix_actions):
            frame_index = leading_frames + offset // prefix_actions
            if frame_index >= int(latent_num_frames):
                break  # These excess source steps are dropped by required_action_num.
            chunk_index = (frame_index - int(proprio_loss_frame_start)) // chunk_size
            chunk_index = max(0, min(chunk_count - 1, chunk_index))
            stop = min(offset + prefix_actions, total_steps)
            relative[offset:stop] = pose_blocks_relative_to_anchor(
                source_actions[offset:stop], anchor=anchors[chunk_index], block_dims=block_dims
            )
        return relative

    def build_action_targets(
        self,
        *,
        action_rows: list[dict[str, Any]],
        target_state_rows: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        return build_row_action_targets(
            data_config=self.data_config,
            action_rows=action_rows,
            target_state_rows=target_state_rows,
            extract_sequence=self.extract_sequence,
            pack_sequence=_TRUNCATING_SEQUENCE_PACKER,
            reference_source_subject="Local latent LeRobot datasets",
        )

    def extract_sequence(
        self,
        *,
        rows: list[dict[str, Any]],
        key: str,
        target_dim: int,
        target_length: int,
        left_pad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows:
            return (
                torch.zeros(target_length, target_dim, dtype=torch.float32),
                torch.zeros(target_length, target_dim, dtype=torch.float32),
            )
        sequence = torch.stack(
            [torch.tensor(row[resolve_row_key(row, key)], dtype=torch.float32) for row in rows],
            dim=0,
        )
        return pack_temporal_sequence(
            sequence=sequence,
            target_dim=target_dim,
            target_length=target_length,
            left_pad=left_pad,
            sequence_name=key,
            truncate_to_target_length=True,
        )

    def extract_state_history_at_frame(
        self,
        *,
        rows: list[dict[str, Any]],
        anchor_frame_index: int,
        state_horizon: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_horizon = int(
            self.data_config.action_schema.state_horizon if state_horizon is None else state_horizon
        )
        anchor = max(0, min(int(anchor_frame_index), len(rows) - 1)) if rows else 0
        state_start = max(0, anchor - resolved_horizon + 1)
        return self.extract_sequence(
            rows=rows[state_start : anchor + 1],
            key=self.data_config.action_target.pose_source_key,
            target_dim=self.data_config.action_schema.state_dim,
            target_length=resolved_horizon,
            left_pad=True,
        )

    def extract_state_at_frame(
        self,
        *,
        rows: list[dict[str, Any]],
        frame_index: int,
        apply_history_lag: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return absolute state, or inv(T_now) @ T_history for proprio context.

        Action-target anchors opt out of history encoding. Pose and passthrough
        channels may use separate lags without mixing frames within a pose.
        """
        state, state_mask = self.extract_state_history_at_frame(
            rows=rows,
            anchor_frame_index=frame_index,
            state_horizon=1,
        )
        state, state_mask = state[0], state_mask[0]
        if not apply_history_lag:
            return state, state_mask
        target = self.data_config.action_target
        base_lag = int(target.proprio_history_lag or 0)
        pose_lag = target.proprio_history_lag_pose
        grip_lag = target.proprio_history_lag_gripper
        pose_lag = base_lag if pose_lag is None else int(pose_lag)
        grip_lag = base_lag if grip_lag is None else int(grip_lag)
        if pose_lag <= 0 and grip_lag <= 0:
            return state, state_mask

        # Absolute-action configurations still need the state's pose-block
        # layout for history encoding; null relative targets do not disable it.
        block = target.relative_pose_block_dims
        if not block:
            state_dim = int(self.data_config.action_schema.state_dim)
            if state_dim <= 0 or state_dim % _POSE_BLOCK_DIMS:
                raise ValueError(
                    f"`proprio_history_lag` needs a pose-block width. state_dim="
                    f"{state_dim} is not a whole number of {_POSE_BLOCK_DIMS}-wide "
                    "blocks, so set `relative_pose_block_dims` explicitly."
                )
            block = _POSE_BLOCK_DIMS
        block = int(block)

        def _relative_at(lag: int) -> tuple[torch.Tensor, torch.Tensor]:
            if lag <= 0:
                return state, state_mask
            earlier, earlier_mask = self.extract_state_history_at_frame(
                rows=rows, anchor_frame_index=max(0, int(frame_index) - lag), state_horizon=1
            )
            relative = pose_blocks_relative_to_anchor(
                earlier[0].unsqueeze(0), anchor=state, block_dims=block
            )[0]
            return relative, torch.minimum(state_mask, earlier_mask[0])

        pose_rel, pose_mask = _relative_at(pose_lag)
        if grip_lag == pose_lag:
            return pose_rel, pose_mask
        grip_rel, grip_mask = _relative_at(grip_lag)
        merged = pose_rel.clone()
        merged_mask = pose_mask.clone()
        for start in range(0, merged.shape[-1], block):
            passthrough = slice(start + 9, min(start + block, merged.shape[-1]))
            if passthrough.start < passthrough.stop:
                merged[passthrough] = grip_rel[passthrough]
                merged_mask[passthrough] = grip_mask[passthrough]
        return merged, merged_mask

    def extract_proprio_context_state_sequence(
        self,
        *,
        rows: list[dict[str, Any]],
        observed_frame_ids: list[int],
        chunk_size: int,
        loss_frame_start: int = 0,
        apply_history_lag: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not observed_frame_ids:
            raise ValueError("Per-chunk proprio context requires non-empty observed_frame_ids.")
        resolved_chunk_size = max(1, int(chunk_size))
        chunk_count = int(math.ceil(len(observed_frame_ids) / float(resolved_chunk_size)))
        states: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for chunk_index in range(chunk_count):
            local_context_index = max(
                0,
                min(
                    len(observed_frame_ids) - 1,
                    int(loss_frame_start) + chunk_index * resolved_chunk_size - 1,
                ),
            )
            frame_index = int(observed_frame_ids[local_context_index])
            state, state_mask = self.extract_state_at_frame(
                rows=rows, frame_index=frame_index, apply_history_lag=apply_history_lag
            )
            states.append(state)
            masks.append(state_mask)
        return torch.stack(states, dim=0), torch.stack(masks, dim=0)

    def extract_proprio_context_frames(
        self,
        *,
        rows: list[dict[str, Any]],
        observed_frame_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_dim = int(self.data_config.action_schema.state_dim)
        if state_dim <= 0:
            raise ValueError("Per-frame proprio context requires positive data.action_schema.state_dim.")
        if not observed_frame_ids:
            return (
                torch.zeros(0, state_dim, dtype=torch.float32),
                torch.zeros(0, state_dim, dtype=torch.float32),
            )
        if not rows:
            return (
                torch.zeros(len(observed_frame_ids), state_dim, dtype=torch.float32),
                torch.zeros(len(observed_frame_ids), state_dim, dtype=torch.float32),
            )
        states: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for frame_index in observed_frame_ids:
            state, state_mask = self.extract_state_history_at_frame(
                rows=rows,
                anchor_frame_index=int(frame_index),
                state_horizon=1,
            )
            states.append(state[-1])
            masks.append(state_mask[-1])
        return torch.stack(states, dim=0), torch.stack(masks, dim=0)
