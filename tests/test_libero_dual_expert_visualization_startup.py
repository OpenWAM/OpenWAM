from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_wam.evals import libero_dual_expert_rollout as dual_expert_viz
from open_wam.evals import libero_dual_expert_inputs as dual_expert_inputs
from open_wam.evals import libero_dual_expert_runtime as dual_expert_runtime
from open_wam.models.policy_variants.dual_expert.runtime_routes import (
    should_use_dual_expert_split_cache_inference,
)
from open_wam.models.policy_variants.contracts import (
    DecoderArtifactEnvelope,
    PolicyInferOutput,
    PolicyInferState,
    PolicyInferenceOutputRequest,
)
from open_wam.models.policy_variants.dual_expert.decoder_artifacts import (
    DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
    DualExpertInferArtifacts,
)


def test_dual_expert_runtime_loading_contract_has_one_canonical_owner() -> None:
    public_names = (
        "CURRENT_FRONTEND_ENCODE_MODE",
        "DEPRECATED_FRONTEND_ENCODE_MODE",
        "DUAL_EXPERT_GJD_ACTION_ROUTES",
        "DualExpertLiberoLoadOptions",
        "DualExpertLiberoRuntime",
        "load_dual_expert_libero_runtime",
        "print_rollout_event",
    )
    for name in public_names:
        assert getattr(dual_expert_viz, name) is getattr(dual_expert_runtime, name)


def test_policy_debug_summarizes_typed_decoder_artifacts() -> None:
    action_pred = torch.zeros(1, 16, 7)
    predicted_latents = torch.zeros(1, 48, 4, 8, 16)
    output = PolicyInferOutput(
        policy_features=torch.zeros(1, 1, 1),
        next_state=PolicyInferState(),
        decoder_artifacts=DecoderArtifactEnvelope(
            contract=DUAL_EXPERT_DECODER_ARTIFACT_CONTRACT,
            payload=DualExpertInferArtifacts(
                action_pred=action_pred,
                predicted_latents=predicted_latents,
                condition_mode="teacher_forcing_cond_video",
                program="video_then_action",
            ),
        ),
        aux={"architecture": "dual_expert"},
    )

    summary = dual_expert_viz._summarize_policy_debug(output)

    assert summary["architecture"] == "dual_expert"
    assert summary["dual_expert_infer_artifacts"] == {
        "action_pred_shape": [1, 16, 7],
        "predicted_latents_shape": [1, 48, 4, 8, 16],
        "condition_mode": "teacher_forcing_cond_video",
        "program": "video_then_action",
    }


def _obs(index: int) -> dict[str, np.ndarray]:
    frame = np.full((2, 2, 3), index, dtype=np.uint8)
    return {
        dual_expert_viz.LIBERO_OBS_KEYS[0]: frame,
        dual_expert_viz.LIBERO_OBS_KEYS[1]: frame + 1,
    }


def test_select_model_obs_window_uses_one_frame_for_chunk0() -> None:
    window = [_obs(index) for index in range(15)]

    selected = dual_expert_viz._select_model_obs_window(
        window,
        chunk_index=0,
        startup_model_obs_frames=1,
    )

    assert len(selected) == 1
    assert np.array_equal(selected[0][dual_expert_viz.LIBERO_OBS_KEYS[0]], window[-1][dual_expert_viz.LIBERO_OBS_KEYS[0]])


def test_select_model_obs_window_keeps_full_window_after_chunk0() -> None:
    window = [_obs(index) for index in range(15)]

    selected = dual_expert_viz._select_model_obs_window(
        window,
        chunk_index=1,
        startup_model_obs_frames=1,
    )

    assert selected == window


@pytest.mark.parametrize("startup_frames", [0, 16])
def test_select_model_obs_window_rejects_invalid_startup_frames(startup_frames: int) -> None:
    window = [_obs(index) for index in range(15)]

    with pytest.raises(ValueError):
        dual_expert_viz._select_model_obs_window(
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

    window = dual_expert_viz._init_single_env(env, init_state={"state": 1}, num_frames=1, init_steps=5)

    assert env.reset_calls == 1
    assert env.init_state == {"state": 1}
    assert env.step_calls == 5
    assert len(window) == 1
    assert np.all(window[0][dual_expert_viz.LIBERO_OBS_KEYS[0]] == 5)


def test_init_single_env_keeps_enough_observations_when_init_steps_is_short() -> None:
    env = _FakeEnv()

    window = dual_expert_viz._init_single_env(env, init_state=None, num_frames=3, init_steps=2)

    assert env.step_calls == 3
    assert len(window) == 3
    assert [int(obs[dual_expert_viz.LIBERO_OBS_KEYS[0]][0, 0, 0]) for obs in window] == [1, 2, 3]


@pytest.mark.parametrize("init_steps", [0, -1])
def test_init_single_env_rejects_invalid_init_steps(init_steps: int) -> None:
    with pytest.raises(ValueError):
        dual_expert_viz._init_single_env(_FakeEnv(), init_state=None, num_frames=1, init_steps=init_steps)


def test_maybe_merge_checkpoint_runtime_config_skips_by_default(monkeypatch, tmp_path: Path) -> None:
    config = object()

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("checkpoint runtime config merge should be opt-in")

    monkeypatch.setattr(dual_expert_runtime, "merge_runtime_config_from_checkpoint", _raise_if_called)

    merged, resolved_config = dual_expert_runtime._maybe_merge_checkpoint_runtime_config(
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

    monkeypatch.setattr(dual_expert_runtime, "merge_runtime_config_from_checkpoint", _fake_merge)

    merged, resolved_config = dual_expert_runtime._maybe_merge_checkpoint_runtime_config(
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
        dual_expert_viz._build_executed_action_history_tensor(
            executed,
            start_frame_group=1,
            action_per_frame=2,
            action_dim=3,
        )


def test_build_executed_action_history_returns_none_when_nothing_executed() -> None:
    assert dual_expert_viz._build_executed_action_history_tensor(
        [],
        start_frame_group=0,
        action_per_frame=2,
        action_dim=3,
    ) is None


def test_build_execution_commit_converts_partial_actions_to_model_frames() -> None:
    commit = dual_expert_viz._build_execution_commit(
        generation_frame_start=1,
        speculative_frame_count=4,
        executed_action_count=12,
        action_per_frame=4,
        start_frame_group=0,
        terminal=False,
    )

    assert commit is not None
    assert commit.speculative_span.start_frame == 1
    assert commit.speculative_span.end_frame == 5
    assert commit.executed_span.start_frame == 1
    assert commit.executed_span.end_frame == 4


def test_build_execution_commit_rejects_partial_action_frame() -> None:
    with pytest.raises(ValueError, match="complete model-frame action groups"):
        dual_expert_viz._build_execution_commit(
            generation_frame_start=1,
            speculative_frame_count=4,
            executed_action_count=11,
            action_per_frame=4,
            start_frame_group=0,
            terminal=False,
        )


@pytest.mark.parametrize(
    ("executed_action_count", "start_frame_group", "terminal"),
    [(0, 0, False), (12, 1, False), (12, 0, True)],
)
def test_build_execution_commit_skips_non_reconcilable_execution(
    executed_action_count: int,
    start_frame_group: int,
    terminal: bool,
) -> None:
    assert dual_expert_viz._build_execution_commit(
        generation_frame_start=1,
        speculative_frame_count=4,
        executed_action_count=executed_action_count,
        action_per_frame=4,
        start_frame_group=start_frame_group,
        terminal=terminal,
    ) is None


@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("video_then_action", True),
        ("decoupled_same_step", True),
        ("joint", False),
        ("video_noisy_to_action", False),
    ],
)
def test_should_use_dual_expert_split_cache_inference_is_program_driven(
    program: str,
    expected: bool,
) -> None:
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="dual_expert", program=program)
    )

    assert should_use_dual_expert_split_cache_inference(config) is expected


def test_prepare_dual_expert_visual_outputs_streaming_path_uses_run_frontend() -> None:
    pipeline = _FakePipeline()
    views = {
        dual_expert_viz.LIBERO_OBS_KEYS[0]: torch.zeros(1, 4, 4, 3),
        dual_expert_viz.LIBERO_OBS_KEYS[1]: torch.zeros(1, 4, 4, 3),
    }

    outputs = dual_expert_viz._prepare_dual_expert_visual_outputs(
        pipeline,
        views=views,
        task_text=("prompt",),
        frontend_device=torch.device("cpu"),
        runtime_device=torch.device("cpu"),
        use_streaming_frontend=True,
    )

    assert pipeline.calls == ["canonicalize", "run_frontend:False", "from_latents:(1, 48, 1, 2, 2)"]
    assert outputs.frontend.video_latents.shape[2] == 1


def test_build_infer_context_uses_joint_dynamics_by_default() -> None:
    obs = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    config = SimpleNamespace(
        data=SimpleNamespace(
            action_schema=SimpleNamespace(state_horizon=1),
            action_target=SimpleNamespace(state_encoding="eef_pos_axisangle_gripper_2d"),
        ),
        inference=SimpleNamespace(
            frame_chunk_size=4,
            attention_window_size=64,
        ),
    )

    output_request = PolicyInferenceOutputRequest.video_only()
    context = dual_expert_viz._build_infer_context(
        "task",
        action_device=torch.device("cpu"),
        model_obs_window=[obs],
        config=config,
        runtime_device=torch.device("cpu"),
        dual_expert_inference_window_size=30,
        dual_expert_action_only_rollout=False,
        output_request=output_request,
    )

    assert context.extra["task_text"] == ("task",)
    assert context.temporal_geometry is not None
    assert context.temporal_geometry.frame_chunk_size == 4
    assert context.temporal_geometry.attention_window_size == 30
    assert "dual_expert_inference_window_size" not in context.extra
    assert "action_conditioning_mode" not in context.extra
    assert "dual_expert_action_only_rollout" not in context.extra
    assert context.output_request is output_request
    assert context.state.shape == (1, 1, 8)


def test_standalone_offline_visualization_encoding_uses_shared_reference_assets() -> None:
    canonical_video = torch.zeros(1, 3, 3, 8, 8)
    encoded = torch.ones(1, 48, 1, 2, 2)
    placements = ("placement",)

    assets = _FakeReferenceAssets(encoded)

    result = dual_expert_inputs._encode_video_window_offline(
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
