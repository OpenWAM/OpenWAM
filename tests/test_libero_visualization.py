from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_wam.evals.libero_visualization import (
    LIBERO_OBS_KEYS,
    extract_observation,
    extract_proprio_context_tensor,
    initialize_raw_observation,
    observations_to_views,
    prepare_exact_runtime_inputs,
)


def _raw_observation(offset: int = 0) -> dict[str, np.ndarray]:
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3) + offset
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image[::-1].copy(),
        "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.4, 0.5], dtype=np.float32),
    }


def test_initialize_and_extract_observation_preserve_exact_startup_contract() -> None:
    class _Environment:
        def __init__(self) -> None:
            self.actions: list[list[float]] = []
            self.reset_calls = 0
            self.init_state = None

        def reset(self) -> None:
            self.reset_calls += 1

        def set_init_state(self, init_state: object) -> None:
            self.init_state = init_state

        def step(self, action: list[float]):
            self.actions.append(action)
            return _raw_observation(len(self.actions)), 0.0, False, {}

    env = _Environment()
    raw = initialize_raw_observation(env, "state-3")
    observation = extract_observation(raw)

    assert env.reset_calls == 1
    assert env.init_state == "state-3"
    assert env.actions == [[0.0] * 7] * 5
    assert set(observation) == set(LIBERO_OBS_KEYS)
    assert np.array_equal(
        observation[LIBERO_OBS_KEYS[0]],
        raw["agentview_image"][::-1],
    )
    assert np.array_equal(
        observation[LIBERO_OBS_KEYS[1]],
        raw["robot0_eye_in_hand_image"][::-1],
    )
    assert all(view.flags.c_contiguous for view in observation.values())


def test_observations_to_views_stacks_time_without_value_conversion() -> None:
    observations = [
        extract_observation(_raw_observation(0)),
        extract_observation(_raw_observation(1)),
    ]

    views = observations_to_views(observations, device=torch.device("cpu"))

    assert set(views) == set(LIBERO_OBS_KEYS)
    assert tuple(views[LIBERO_OBS_KEYS[0]].shape) == (2, 4, 5, 3)
    assert views[LIBERO_OBS_KEYS[0]].dtype is torch.uint8
    assert torch.equal(
        views[LIBERO_OBS_KEYS[0]][1],
        torch.from_numpy(observations[1][LIBERO_OBS_KEYS[0]]),
    )
    with pytest.raises(ValueError, match="empty observation sequence"):
        observations_to_views([], device=torch.device("cpu"))


def test_proprio_context_uses_exact_eef_axisangle_gripper_layout() -> None:
    enabled_config = SimpleNamespace(
        policy_variant=SimpleNamespace(proprio_context_mode="per_chunk_additive"),
        data=SimpleNamespace(
            action_target=SimpleNamespace(
                state_encoding="eef_pos_axisangle_gripper_2d"
            ),
            action_schema=SimpleNamespace(state_dim=8),
        ),
    )
    disabled_config = SimpleNamespace(
        policy_variant=SimpleNamespace(proprio_context_mode="none"),
    )

    state = extract_proprio_context_tensor(
        _raw_observation(),
        config=enabled_config,
        device=torch.device("cpu"),
    )

    assert state is not None
    torch.testing.assert_close(
        state,
        torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.4, 0.5]]),
    )
    assert (
        extract_proprio_context_tensor(
            _raw_observation(),
            config=disabled_config,
            device=torch.device("cpu"),
        )
        is None
    )


def test_prepare_exact_runtime_inputs_delegates_to_shared_visual_frontend() -> None:
    class _VisualTower:
        def __init__(self) -> None:
            self.call: dict[str, object] | None = None

        def run_frontend(self, video: torch.Tensor, **kwargs):
            self.call = {"video": video, **kwargs}
            return SimpleNamespace(
                video_latents=torch.full((1, 2, 3, 1, 1), 4.0),
                conditioning=SimpleNamespace(
                    text_context=torch.full((1, 2, 3), 5.0),
                    negative_text_context=torch.full((1, 2, 3), 6.0),
                ),
            )

    visual_tower = _VisualTower()
    canonical_video = torch.full((1, 3, 2, 4, 5), 2.0)
    pipeline = SimpleNamespace(
        canonicalize=lambda views: SimpleNamespace(
            video=canonical_video,
            placements=("left", "right"),
        ),
        visual_tower=visual_tower,
    )
    runner = SimpleNamespace(pipeline=pipeline)
    text_context = torch.ones(1, 2, 3)
    negative_text_context = torch.zeros(1, 2, 3)

    prepared = prepare_exact_runtime_inputs(
        runner,
        views={"unused": torch.zeros(1)},
        task_text=("task",),
        frontend_device=torch.device("cpu"),
        runtime_device=torch.device("cpu"),
        text_context=text_context,
        negative_text_context=negative_text_context,
        preserve_stream_cache=True,
    )

    assert visual_tower.call is not None
    assert visual_tower.call["video"] is canonical_video
    assert visual_tower.call["placements"] == ("left", "right")
    assert visual_tower.call["task_text"] == ("task",)
    assert visual_tower.call["text_context"] is text_context
    assert visual_tower.call["negative_text_context"] is negative_text_context
    assert visual_tower.call["preserve_stream_cache"] is True
    assert torch.equal(prepared["video_latents"], torch.full((1, 2, 3, 1, 1), 4.0))
    assert torch.equal(prepared["text_context"], torch.full((1, 2, 3), 5.0))
    assert torch.equal(prepared["negative_text_context"], torch.full((1, 2, 3), 6.0))
