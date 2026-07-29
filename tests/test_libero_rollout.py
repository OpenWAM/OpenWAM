from __future__ import annotations

import numpy as np
import pytest
import torch

from open_wam.configs import ActionTargetStateEncoding
from open_wam.integrations.libero_rollout import build_libero_state_history


def _observation(position: tuple[float, float, float]) -> dict[str, np.ndarray]:
    return {
        "robot0_eef_pos": np.asarray(position, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.02, -0.02], dtype=np.float32),
    }


def test_build_libero_state_history_left_pads_axis_angle_state() -> None:
    state = build_libero_state_history(
        [_observation((1.0, 2.0, 3.0))],
        state_horizon=2,
        state_encoding=ActionTargetStateEncoding.EEF_POS_AXISANGLE_GRIPPER_2D,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.02, -0.02],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.02, -0.02],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(state, expected)


def test_build_libero_state_history_keeps_latest_quaternion_states() -> None:
    state = build_libero_state_history(
        [
            _observation((0.0, 0.0, 0.0)),
            _observation((1.0, 2.0, 3.0)),
            _observation((4.0, 5.0, 6.0)),
        ],
        state_horizon=2,
        state_encoding=ActionTargetStateEncoding.EEF_POS_QUAT_GRIPPER_1D,
    )

    assert state.shape == (2, 8)
    torch.testing.assert_close(
        state[:, :3],
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        state[:, 3:7],
        torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.float32),
    )
    torch.testing.assert_close(state[:, 7], torch.tensor([0.02, 0.02]))


def test_build_libero_state_history_rejects_non_pose_encoding() -> None:
    with pytest.raises(ValueError, match="Unsupported LIBERO rollout state encoding"):
        build_libero_state_history(
            [_observation((0.0, 0.0, 0.0))],
            state_horizon=1,
            state_encoding=ActionTargetStateEncoding.IDENTITY,
        )
