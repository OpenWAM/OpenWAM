from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_libero_mot_visualization as mot_viz


def _obs(index: int) -> dict[str, np.ndarray]:
    frame = np.full((2, 2, 3), index, dtype=np.uint8)
    return {
        mot_viz.LIBERO_OBS_KEYS[0]: frame,
        mot_viz.LIBERO_OBS_KEYS[1]: frame + 1,
    }


def test_select_model_obs_window_uses_one_frame_for_chunk0() -> None:
    window = [_obs(index) for index in range(15)]

    selected = mot_viz._select_model_obs_window(
        window,
        chunk_index=0,
        startup_model_obs_frames=1,
    )

    assert len(selected) == 1
    assert np.array_equal(selected[0][mot_viz.LIBERO_OBS_KEYS[0]], window[-1][mot_viz.LIBERO_OBS_KEYS[0]])


def test_select_model_obs_window_keeps_full_window_after_chunk0() -> None:
    window = [_obs(index) for index in range(15)]

    selected = mot_viz._select_model_obs_window(
        window,
        chunk_index=1,
        startup_model_obs_frames=1,
    )

    assert selected == window


@pytest.mark.parametrize("startup_frames", [0, 16])
def test_select_model_obs_window_rejects_invalid_startup_frames(startup_frames: int) -> None:
    window = [_obs(index) for index in range(15)]

    with pytest.raises(ValueError):
        mot_viz._select_model_obs_window(
            window,
            chunk_index=0,
            startup_model_obs_frames=startup_frames,
        )


class _FakeEnv:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.init_state = None
        self.step_calls = 0

    def reset(self):
        self.reset_calls += 1

    def set_init_state(self, init_state):
        self.init_state = init_state

    def step(self, action):
        self.step_calls += 1
        image = np.full((2, 2, 3), self.step_calls, dtype=np.uint8)
        obs = {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image + 1,
        }
        return obs, 0.0, False, {}


def test_init_single_env_defaults_to_method1_five_step_startup() -> None:
    env = _FakeEnv()

    window = mot_viz._init_single_env(env, init_state={"state": 1}, num_frames=1, init_steps=5)

    assert env.reset_calls == 1
    assert env.init_state == {"state": 1}
    assert env.step_calls == 5
    assert len(window) == 1
    assert np.all(window[0][mot_viz.LIBERO_OBS_KEYS[0]] == 5)


def test_init_single_env_keeps_enough_observations_when_init_steps_is_short() -> None:
    env = _FakeEnv()

    window = mot_viz._init_single_env(env, init_state=None, num_frames=3, init_steps=2)

    assert env.step_calls == 3
    assert len(window) == 3
    assert [int(obs[mot_viz.LIBERO_OBS_KEYS[0]][0, 0, 0]) for obs in window] == [1, 2, 3]


@pytest.mark.parametrize("init_steps", [0, -1])
def test_init_single_env_rejects_invalid_init_steps(init_steps: int) -> None:
    with pytest.raises(ValueError):
        mot_viz._init_single_env(_FakeEnv(), init_state=None, num_frames=1, init_steps=init_steps)


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.visual_tower = SimpleNamespace(
            core=SimpleNamespace(
                patch_embedding_mlp=SimpleNamespace(
                    weight=torch.zeros((), dtype=torch.float32),
                ),
            ),
            run_frontend=self._run_frontend,
        )

    def canonicalize(self, views):
        self.calls.append("canonicalize")
        batch = next(iter(views.values())).shape[0]
        return SimpleNamespace(
            video=torch.zeros(batch, 3, 4, 4, 4),
            placements=("placement",),
        )

    def _run_frontend(self, canonical_video, *, placements, task_text, text_context, negative_text_context, preserve_stream_cache):
        self.calls.append(f"run_frontend:{preserve_stream_cache}")
        batch = canonical_video.shape[0]
        return SimpleNamespace(
            video_latents=torch.ones(batch, 48, 1, 2, 2),
            conditioning=SimpleNamespace(
                text_context=torch.ones(batch, 2, 3),
                negative_text_context=torch.zeros(batch, 2, 3),
            ),
        )

    def prepare_visual_outputs_from_latents(self, video_latents, **kwargs):
        self.calls.append(f"from_latents:{tuple(video_latents.shape)}")
        return SimpleNamespace(
            frontend=SimpleNamespace(
                video_latents=video_latents,
                conditioning=SimpleNamespace(
                    text_context=kwargs.get("text_context"),
                    negative_text_context=kwargs.get("negative_text_context"),
                ),
            )
        )


def test_build_executed_action_history_uses_clipped_controls_and_bootstrap_zeros() -> None:
    executed = [
        np.array([2.0, -2.0, 0.5], dtype=np.float32),
        np.array([0.25, 0.5, -0.25], dtype=np.float32),
    ]

    history = mot_viz._build_executed_action_history_tensor(
        executed,
        start_frame_group=1,
        action_per_frame=2,
        action_dim=3,
    )

    assert history is not None
    assert history.shape == (1, 4, 3)
    assert torch.equal(history[0, :2], torch.zeros(2, 3))
    assert torch.equal(history[0, 2:], torch.from_numpy(np.stack(executed, axis=0)))


def test_build_executed_action_history_returns_none_when_nothing_executed() -> None:
    assert mot_viz._build_executed_action_history_tensor(
        [],
        start_frame_group=0,
        action_per_frame=2,
        action_dim=3,
    ) is None


def test_prepare_mot_visual_outputs_streaming_path_uses_run_frontend() -> None:
    pipeline = _FakePipeline()
    views = {
        mot_viz.LIBERO_OBS_KEYS[0]: torch.zeros(1, 4, 4, 3),
        mot_viz.LIBERO_OBS_KEYS[1]: torch.zeros(1, 4, 4, 3),
    }

    outputs = mot_viz._prepare_mot_visual_outputs(
        pipeline,
        views=views,
        task_text=("prompt",),
        frontend_device=torch.device("cpu"),
        runtime_device=torch.device("cpu"),
        use_streaming_frontend=True,
    )

    assert pipeline.calls == ["canonicalize", "run_frontend:False", "from_latents:(1, 48, 1, 2, 2)"]
    assert outputs.frontend.video_latents.shape[2] == 1
