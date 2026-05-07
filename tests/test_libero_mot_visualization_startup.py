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


def test_append_predicted_latent_chunk_honors_frame_cap() -> None:
    chunks: list[torch.Tensor] = []
    mot_viz._append_predicted_latent_chunk(
        chunks,
        torch.ones(1, 2, 3, 4, 4),
        max_imagined_latent_frames=5,
    )
    mot_viz._append_predicted_latent_chunk(
        chunks,
        torch.ones(1, 2, 4, 4, 4) * 2.0,
        max_imagined_latent_frames=5,
    )

    assert [int(chunk.shape[2]) for chunk in chunks] == [3, 2]
    assert torch.all(chunks[1] == 2.0)


def test_append_predicted_latent_chunk_zero_cap_disables_collection() -> None:
    chunks: list[torch.Tensor] = []

    mot_viz._append_predicted_latent_chunk(
        chunks,
        torch.ones(1, 2, 3, 4, 4),
        max_imagined_latent_frames=0,
    )

    assert chunks == []


@pytest.mark.parametrize(
    ("current_block_coupling", "expected"),
    [
        ("video_then_action", True),
        ("decoupled_same_step", True),
        ("joint", False),
        ("video_noisy_to_action", False),
        (None, False),
    ],
)
def test_should_restore_mot_legacy_blocks_only_for_legacy_couplings(
    current_block_coupling: str | None,
    expected: bool,
) -> None:
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(current_block_coupling=current_block_coupling)
    )

    assert mot_viz._should_restore_mot_legacy_blocks(config) is expected


def test_comparison_video_frame_builder_resamples_imagined_frames_without_materializing_alignment() -> None:
    real_obs = [_obs(1), _obs(2), _obs(3)]
    imagined_video = np.stack(
        [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 127, dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
            np.full((2, 2, 3), 64, dtype=np.uint8),
            np.full((2, 2, 3), 32, dtype=np.uint8),
        ],
        axis=0,
    )

    frames = list(
        mot_viz._iter_comparison_video_frames(
            real_obs_list=real_obs,
            imagined_video=imagined_video,
        )
    )

    assert len(frames) == len(real_obs)
    assert all(frame.flags["C_CONTIGUOUS"] for frame in frames)


def test_write_video_frames_streams_to_imageio_writer(monkeypatch, tmp_path) -> None:
    written: list[np.ndarray] = []

    class _Writer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def append_data(self, frame):
            written.append(np.array(frame, copy=True))

    def _fake_get_writer(path, *, fps):
        assert path == tmp_path / "out.mp4"
        assert fps == 7.0
        return _Writer()

    monkeypatch.setattr(mot_viz.imageio, "get_writer", _fake_get_writer)

    mot_viz._write_video_frames(
        tmp_path / "out.mp4",
        [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)],
        fps=7.0,
    )

    assert len(written) == 2


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
