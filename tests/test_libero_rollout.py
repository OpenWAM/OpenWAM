from __future__ import annotations

import numpy as np
import pytest
import torch

from open_wam.configs import ActionTargetStateEncoding
from open_wam.integrations.libero_rollout import (
    LIBERO_ROLLOUT_VIEW_KEYS,
    build_libero_state_history,
    extract_libero_rollout_observation,
    initialize_libero_observation_window,
    libero_observation_window_to_views,
    pose_from_libero_observation,
    reconstruct_libero_pose_targets,
)


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


def test_extract_and_stack_libero_rollout_observations() -> None:
    agent = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    wrist = agent + 20
    raw = {
        "agentview_image": agent,
        "robot0_eye_in_hand_image": wrist,
        **_observation((1.0, 2.0, 3.0)),
    }

    observation = extract_libero_rollout_observation(raw)
    views = libero_observation_window_to_views(
        [observation, observation],
        device=torch.device("cpu"),
    )

    np.testing.assert_array_equal(
        observation[LIBERO_ROLLOUT_VIEW_KEYS[0]],
        agent[::-1],
    )
    np.testing.assert_array_equal(
        observation[LIBERO_ROLLOUT_VIEW_KEYS[1]],
        wrist[::-1],
    )
    assert observation[LIBERO_ROLLOUT_VIEW_KEYS[0]].flags.c_contiguous
    assert views[LIBERO_ROLLOUT_VIEW_KEYS[0]].shape == (2, 2, 3, 3)


def test_initialize_libero_observation_window_preserves_startup_contract() -> None:
    class _Env:
        def __init__(self) -> None:
            self.step_count = 0
            self.init_state = None

        def reset(self) -> None:
            self.step_count = 0

        def set_init_state(self, init_state) -> None:
            self.init_state = init_state

        def step(self, action):
            assert action == [0.0] * 7
            self.step_count += 1
            image = np.full((2, 2, 3), self.step_count, dtype=np.uint8)
            return (
                {
                    "agentview_image": image,
                    "robot0_eye_in_hand_image": image,
                    **_observation((float(self.step_count), 0.0, 0.0)),
                },
                0.0,
                False,
                {},
            )

    env = _Env()
    observations = initialize_libero_observation_window(
        env,
        "init",
        num_frames=2,
    )

    assert env.init_state == "init"
    assert env.step_count == 5
    assert [float(item["robot0_eef_pos"][0]) for item in observations] == [
        4.0,
        5.0,
    ]


def test_reconstruct_libero_pose_targets_uses_reference_pose() -> None:
    observation = _observation((1.0, 2.0, 3.0))
    relative = np.asarray(
        [
            [0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    reference = pose_from_libero_observation(observation)
    reconstructed = reconstruct_libero_pose_targets(
        relative,
        reference_observation=observation,
        rotation_representation="axis_angle",
    )

    torch.testing.assert_close(
        reference.position,
        torch.tensor([1.0, 2.0, 3.0]),
    )
    torch.testing.assert_close(
        reconstructed.position,
        torch.tensor([[1.5, 1.5, 4.0], [2.0, 2.0, 2.0]]),
    )
