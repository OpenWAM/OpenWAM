from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from open_wam.configs import ActionTargetStateEncoding
from open_wam.data.action_pose import (
    PoseSequence,
    normalize_quaternion,
    quaternion_to_axis_angle,
    reconstruct_absolute_pose_targets,
)


LIBERO_ROLLOUT_VIEW_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def extract_libero_rollout_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Translate one raw LIBERO observation into Open-WAM rollout fields."""

    return {
        LIBERO_ROLLOUT_VIEW_KEYS[0]: np.ascontiguousarray(
            observation["agentview_image"][::-1]
        ),
        LIBERO_ROLLOUT_VIEW_KEYS[1]: np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1]
        ),
        "robot0_eef_pos": np.asarray(
            observation["robot0_eef_pos"],
            dtype=np.float32,
        ).copy(),
        "robot0_eef_quat": np.asarray(
            observation["robot0_eef_quat"],
            dtype=np.float32,
        ).copy(),
        "robot0_gripper_qpos": np.asarray(
            observation["robot0_gripper_qpos"],
            dtype=np.float32,
        ).copy(),
    }


def initialize_libero_observation_window(
    env: Any,
    init_state: Any,
    *,
    num_frames: int,
    init_steps: int = 5,
) -> list[dict[str, np.ndarray]]:
    """Reset one LIBERO env and collect its zero-action startup window."""

    if num_frames <= 0:
        raise ValueError(f"Expected positive num_frames, got {num_frames}.")
    if init_steps <= 0:
        raise ValueError(f"Expected positive init_steps, got {init_steps}.")
    env.reset()
    env.set_init_state(init_state)
    observations: list[dict[str, np.ndarray]] = []
    for _ in range(max(init_steps, num_frames)):
        observation, _, _, _ = env.step([0.0] * 7)
        observations.append(extract_libero_rollout_observation(observation))
    if not observations:
        raise RuntimeError(
            "LIBERO env did not return an observation during initialization."
        )
    return observations[-num_frames:]


def libero_observation_window_to_views(
    observations: Sequence[Mapping[str, np.ndarray]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack canonical LIBERO rollout images into the public view mapping."""

    if not observations:
        raise ValueError("Cannot build LIBERO views from an empty observation window.")
    return {
        key: torch.from_numpy(
            np.stack([observation[key] for observation in observations], axis=0)
        ).to(device=device)
        for key in LIBERO_ROLLOUT_VIEW_KEYS
    }


def build_libero_state_history(
    observations: Sequence[Mapping[str, np.ndarray]],
    *,
    state_horizon: int,
    state_encoding: ActionTargetStateEncoding | str,
) -> torch.Tensor:
    """Build policy proprio history from canonical LIBERO observations."""

    if state_horizon <= 0:
        raise ValueError(f"Expected positive state_horizon, got {state_horizon}.")
    if not observations:
        raise ValueError("Cannot build state inputs from an empty observation window.")
    resolved_encoding = ActionTargetStateEncoding(state_encoding)
    state_records = list(observations[-state_horizon:])
    if len(state_records) < state_horizon:
        state_records = [state_records[0]] * (state_horizon - len(state_records)) + state_records
    return torch.stack(
        [
            _build_libero_state_vector(
                observation,
                state_encoding=resolved_encoding,
            )
            for observation in state_records
        ],
        dim=0,
    )


def _build_libero_state_vector(
    observation: Mapping[str, np.ndarray],
    *,
    state_encoding: ActionTargetStateEncoding,
) -> torch.Tensor:
    position = torch.from_numpy(
        np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    )
    quaternion = normalize_quaternion(
        torch.from_numpy(
            np.asarray(observation["robot0_eef_quat"], dtype=np.float32)
        ).unsqueeze(0)
    )[0]
    gripper = torch.from_numpy(
        np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
    )

    if state_encoding is ActionTargetStateEncoding.EEF_POS_AXISANGLE_GRIPPER_2D:
        axis_angle = quaternion_to_axis_angle(quaternion.unsqueeze(0))[0]
        return torch.cat([position, axis_angle, gripper], dim=0)
    if state_encoding is ActionTargetStateEncoding.EEF_POS_QUAT_GRIPPER_1D:
        return torch.cat([position, quaternion, gripper[:1]], dim=0)
    raise ValueError(
        f"Unsupported LIBERO rollout state encoding: {state_encoding.value}"
    )


def pose_from_libero_observation(
    observation: Mapping[str, np.ndarray],
) -> PoseSequence:
    return PoseSequence(
        position=torch.from_numpy(
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
        ),
        quaternion=normalize_quaternion(
            torch.from_numpy(
                np.asarray(observation["robot0_eef_quat"], dtype=np.float32)
            ).unsqueeze(0)
        )[0],
        gripper=torch.from_numpy(
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
        ),
    )


def reconstruct_libero_pose_targets(
    action_prediction: np.ndarray,
    *,
    reference_observation: Mapping[str, np.ndarray],
    rotation_representation: str,
) -> PoseSequence:
    """Recover absolute EEF targets from one reference-relative action chunk."""

    relative_pose_targets = torch.from_numpy(
        np.asarray(action_prediction, dtype=np.float32)
    )
    reference_pose = pose_from_libero_observation(reference_observation)
    return reconstruct_absolute_pose_targets(
        reference_position=reference_pose.position,
        reference_quaternion=reference_pose.quaternion,
        relative_pose_targets=relative_pose_targets,
        rotation_representation=rotation_representation,
    )
