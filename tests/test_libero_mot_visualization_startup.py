from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_wam.evals import libero_mot_rollout as mot_viz
from open_wam.models.policy_variants.mot.runtime_routing import (
    should_use_mot_legacy_split_cache_inference,
)


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
            "robot0_eef_pos": np.asarray([float(self.step_calls), 0.0, 0.0], dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([0.0, 0.0], dtype=np.float32),
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


def test_maybe_merge_checkpoint_runtime_config_skips_by_default(monkeypatch, tmp_path: Path) -> None:
    config = object()

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("checkpoint runtime config merge should be opt-in")

    monkeypatch.setattr(mot_viz, "merge_runtime_config_from_checkpoint", _raise_if_called)

    merged, resolved_config = mot_viz._maybe_merge_checkpoint_runtime_config(
        config,
        tmp_path / "checkpoint_step_1",
        merge_enabled=False,
    )

    assert merged is config
    assert resolved_config is None


def test_maybe_merge_checkpoint_runtime_config_merges_when_requested(monkeypatch, tmp_path: Path) -> None:
    config = object()
    merged_config = object()
    resolved_path = tmp_path / "checkpoint_step_1" / "resolved_config.yaml"

    def _fake_merge(base_config, checkpoint_path):
        assert base_config is config
        assert checkpoint_path == tmp_path / "checkpoint_step_1"
        return merged_config, resolved_path

    monkeypatch.setattr(mot_viz, "merge_runtime_config_from_checkpoint", _fake_merge)

    merged, resolved_config = mot_viz._maybe_merge_checkpoint_runtime_config(
        config,
        tmp_path / "checkpoint_step_1",
        merge_enabled=True,
    )

    assert merged is merged_config
    assert resolved_config == resolved_path


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


class _FakeReferenceAssets:
    has_vae = True

    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def encode_video(self, canonical_video, *, placements=None, reset_cache: bool = True):
        self.calls.append(
            {
                "shape": tuple(canonical_video.shape),
                "placements": placements,
                "reset_cache": reset_cache,
            }
        )
        return self.output


def test_build_executed_action_history_rejects_bootstrap_zero_actions() -> None:
    executed = [
        np.array([2.0, -2.0, 0.5], dtype=np.float32),
        np.array([0.25, 0.5, -0.25], dtype=np.float32),
    ]

    with pytest.raises(ValueError, match="deprecated"):
        mot_viz._build_executed_action_history_tensor(
            executed,
            start_frame_group=1,
            action_per_frame=2,
            action_dim=3,
        )


def test_build_executed_action_history_returns_none_when_nothing_executed() -> None:
    assert mot_viz._build_executed_action_history_tensor(
        [],
        start_frame_group=0,
        action_per_frame=2,
        action_dim=3,
    ) is None


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
def test_should_use_mot_legacy_split_cache_inference_only_for_legacy_couplings(
    current_block_coupling: str | None,
    expected: bool,
) -> None:
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(current_block_coupling=current_block_coupling)
    )

    assert should_use_mot_legacy_split_cache_inference(config) is expected


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


def test_build_infer_context_threads_mot_generalist_rollout_mode() -> None:
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    config = SimpleNamespace(
        data=SimpleNamespace(
            action_schema=SimpleNamespace(state_horizon=1),
            action_target=SimpleNamespace(state_encoding="eef_pos_axisangle_gripper_2d"),
        )
    )

    context = mot_viz._build_infer_context(
        "task",
        action_device=torch.device("cpu"),
        model_obs_window=[obs],
        config=config,
        runtime_device=torch.device("cpu"),
        mot_inference_window_size=30,
        mot_action_only_rollout=False,
        mot_generalist_rollout_mode="joint",
    )

    assert context.extra["task_text"] == ("task",)
    assert context.extra["mot_inference_window_size"] == 30
    assert context.extra["action_conditioning_mode"] == "joint"
    assert context.extra["mot_generalist_rollout_mode"] == "joint"
    assert "mot_action_only_rollout" not in context.extra
    assert context.state.shape == (1, 1, 8)


def test_build_infer_context_rejects_offline_diagnostic_mot_generalist_modes() -> None:
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    config = SimpleNamespace(
        data=SimpleNamespace(
            action_schema=SimpleNamespace(state_horizon=1),
            action_target=SimpleNamespace(state_encoding="eef_pos_axisangle_gripper_2d"),
        )
    )

    with pytest.raises(ValueError, match="offline diagnostic mode"):
        mot_viz._build_infer_context(
            "task",
            action_device=torch.device("cpu"),
            model_obs_window=[obs],
            config=config,
            runtime_device=torch.device("cpu"),
            mot_inference_window_size=30,
            mot_action_only_rollout=False,
            mot_generalist_rollout_mode="video_conditioned_action",
        )


def test_standalone_offline_visualization_encoding_uses_shared_reference_assets() -> None:
    canonical_video = torch.zeros(1, 3, 3, 8, 8)
    encoded = torch.ones(1, 48, 1, 2, 2)
    placements = ("placement",)

    assets = _FakeReferenceAssets(encoded)

    result = mot_viz._encode_video_window_offline(
        assets,
        canonical_video=canonical_video,
        placements=placements,
        device=torch.device("cpu"),
    )

    assert result is encoded
    assert assets.calls == [
        {
            "shape": tuple(canonical_video.shape),
            "placements": placements,
            "reset_cache": True,
        }
    ]
