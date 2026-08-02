"""Closed-loop LIBERO trajectory replay and tracking metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from open_wam.data.action_pose import PoseSequence, reconstruct_absolute_pose_targets
from open_wam.integrations.libero_control import (
    LiberoControlConfig,
    compute_osc_pose_action,
    extract_pose_from_obs,
    project_libero_gripper_state,
    quaternion_angular_error_degrees,
)
from open_wam.integrations.libero_runtime import build_libero_offscreen_env
from open_wam.integrations.libero_tasks import (
    LiberoTaskSpec,
    load_libero_task_init_states,
    resolve_libero_task,
)


__all__ = [
    "LiberoTrackingResult",
    "track_relative_targets_in_libero_env",
]


@dataclass(frozen=True)
class LiberoTrackingResult:
    """Trajectory rollout and tracking metrics from a LIBERO env replay."""

    task_spec: LiberoTaskSpec
    init_state_index: int
    desired_pose: PoseSequence
    tracked_pose: PoseSequence
    position_error_per_target: torch.Tensor
    rotation_error_deg_per_target: torch.Tensor
    gripper_error_per_target: torch.Tensor
    camera_frames: dict[str, list[np.ndarray]]
    rendered_target_indices: list[int]


def track_relative_targets_in_libero_env(
    *,
    task_text: str,
    relative_pose_targets: torch.Tensor,
    rotation_representation: str,
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
    gripper_representation: str = "first_channel",
    init_state_index: int = 0,
    control_config: LiberoControlConfig | None = None,
    camera_obs_keys: tuple[str, ...] = (
        "agentview_image",
        "robot0_eye_in_hand_image",
    ),
    camera_height: int = 256,
    camera_width: int = 256,
    project_root: Path | None = None,
) -> LiberoTrackingResult:
    """Replay one public WAM trajectory in the real LIBERO simulator."""

    if control_config is None:
        control_config = LiberoControlConfig()

    task_spec = resolve_libero_task(task_text, project_root=project_root)
    init_states = load_libero_task_init_states(
        task_spec,
        project_root=project_root,
    )
    init_state_index = int(
        np.clip(init_state_index, 0, len(init_states) - 1)
    )

    env = build_libero_offscreen_env(
        task_spec,
        camera_height=camera_height,
        camera_width=camera_width,
        horizon=max(
            5000,
            int(
                relative_pose_targets.shape[0]
                * control_config.control_substeps_per_target
                + 32
            ),
        ),
        ignore_done=True,
        project_root=project_root,
    )
    try:
        obs = env.reset()
        obs = env.set_init_state(init_states[init_state_index])
        desired_pose = reconstruct_absolute_pose_targets(
            reference_position=reference_position,
            reference_quaternion=reference_quaternion,
            relative_pose_targets=relative_pose_targets,
            rotation_representation=rotation_representation,
        )
        aligned_gripper_targets = _align_replay_gripper_targets(
            desired_pose.gripper,
            gripper_representation=gripper_representation,
            delay_steps=control_config.action_command_delay_steps,
        )

        tracked_positions: list[torch.Tensor] = []
        tracked_quaternions: list[torch.Tensor] = []
        tracked_gripper: list[torch.Tensor] = []
        rendered_target_indices: list[int] = []
        camera_frames: dict[str, list[np.ndarray]] = {
            camera_key: [] for camera_key in camera_obs_keys
        }

        for target_index in range(relative_pose_targets.shape[0]):
            target_pose = PoseSequence(
                position=desired_pose.position[target_index],
                quaternion=desired_pose.quaternion[target_index],
                gripper=(
                    None
                    if aligned_gripper_targets is None
                    else aligned_gripper_targets[target_index]
                ),
            )
            for _ in range(control_config.control_substeps_per_target):
                current_pose = extract_pose_from_obs(obs)
                action = compute_osc_pose_action(
                    current_pose=current_pose,
                    desired_pose=target_pose,
                    control_config=control_config,
                    gripper_representation=gripper_representation,
                )
                obs, _, _, _ = env.step(action)
                rendered_target_indices.append(target_index)
                for camera_key in camera_obs_keys:
                    camera_frames[camera_key].append(
                        np.array(obs[camera_key], copy=True)
                    )

            final_pose = extract_pose_from_obs(obs)
            tracked_positions.append(final_pose.position)
            tracked_quaternions.append(final_pose.quaternion)
            if final_pose.gripper is not None:
                if gripper_representation == "action_command":
                    tracked_gripper.append(
                        torch.tensor(
                            [float(action[-1])],
                            dtype=torch.float32,
                        )
                    )
                    continue
                tracked_gripper.append(
                    project_libero_gripper_state(
                        final_pose.gripper,
                        gripper_representation=gripper_representation,
                    )
                )

        tracked_pose = PoseSequence(
            position=torch.stack(tracked_positions, dim=0),
            quaternion=torch.stack(tracked_quaternions, dim=0),
            gripper=(
                torch.stack(tracked_gripper, dim=0)
                if tracked_gripper
                else None
            ),
        )
        position_error_per_target = torch.linalg.vector_norm(
            tracked_pose.position - desired_pose.position,
            dim=-1,
        )
        rotation_error_deg_per_target = quaternion_angular_error_degrees(
            tracked_pose.quaternion,
            desired_pose.quaternion,
        )
        if (
            desired_pose.gripper is not None
            and tracked_pose.gripper is not None
        ):
            gripper_error_per_target = torch.linalg.vector_norm(
                tracked_pose.gripper - aligned_gripper_targets,
                dim=-1,
            )
        else:
            gripper_error_per_target = torch.zeros_like(
                position_error_per_target
            )

        return LiberoTrackingResult(
            task_spec=task_spec,
            init_state_index=init_state_index,
            desired_pose=desired_pose,
            tracked_pose=tracked_pose,
            position_error_per_target=position_error_per_target,
            rotation_error_deg_per_target=rotation_error_deg_per_target,
            gripper_error_per_target=gripper_error_per_target,
            camera_frames=camera_frames,
            rendered_target_indices=rendered_target_indices,
        )
    finally:
        env.close()


def _align_replay_gripper_targets(
    gripper_targets: torch.Tensor | None,
    *,
    gripper_representation: str,
    delay_steps: int,
) -> torch.Tensor | None:
    """Align causal action commands to the state targets they produce."""

    if gripper_targets is None:
        return None
    if gripper_representation != "action_command":
        return gripper_targets
    if delay_steps < 0:
        raise ValueError(
            "Expected non-negative action_command_delay_steps, got "
            f"{delay_steps}."
        )

    aligned = torch.zeros_like(gripper_targets)
    if delay_steps == 0:
        aligned.copy_(gripper_targets)
        return aligned
    if delay_steps >= gripper_targets.shape[0]:
        return aligned

    aligned[delay_steps:] = gripper_targets[:-delay_steps]
    return aligned
