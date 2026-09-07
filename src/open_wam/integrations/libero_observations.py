"""LIBERO observation parsing into OpenWAM state contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from open_wam.data.action_pose import PoseSequence, normalize_quaternion


__all__ = [
    "extract_gripper_positions_from_obs",
    "extract_joint_positions_from_obs",
    "extract_pose_from_obs",
]


def extract_pose_from_obs(obs: dict[str, Any]) -> PoseSequence:
    """Parse a LIBERO observation into the common pose contract."""

    quaternion_xyzw = torch.tensor(
        obs["robot0_eef_quat"],
        dtype=torch.float32,
    )
    return PoseSequence(
        position=torch.tensor(obs["robot0_eef_pos"], dtype=torch.float32),
        quaternion=normalize_quaternion(quaternion_xyzw),
        gripper=torch.tensor(
            obs["robot0_gripper_qpos"],
            dtype=torch.float32,
        ),
    )


def extract_joint_positions_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Extract Panda arm qpos from a LIBERO observation."""

    if "robot0_joint_pos" not in obs:
        raise KeyError("LIBERO observation does not expose `robot0_joint_pos`.")
    joint_positions = np.asarray(
        obs["robot0_joint_pos"],
        dtype=np.float32,
    ).reshape(-1)
    if joint_positions.size == 0:
        raise ValueError("LIBERO `robot0_joint_pos` is empty.")
    return joint_positions


def extract_gripper_positions_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Extract Panda gripper qpos from a LIBERO observation."""

    if "robot0_gripper_qpos" not in obs:
        raise KeyError(
            "LIBERO observation does not expose `robot0_gripper_qpos`."
        )
    values = np.asarray(
        obs["robot0_gripper_qpos"],
        dtype=np.float32,
    ).reshape(-1)
    if values.size == 0:
        raise ValueError("LIBERO `robot0_gripper_qpos` is empty.")
    return values
