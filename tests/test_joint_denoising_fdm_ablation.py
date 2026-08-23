from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    DynamicsObjective,
    ParallelStreamPolicyConfig,
    VideoActionProgram,
)
from open_wam.data.counterfactual_actions import (
    BRANCH_PRESETS,
    apply_action_branch,
    branch_metadata,
    expand_branch_names,
)
from scripts.research_dynamics.counterfactual import (
    CounterfactualCase,
    _decoded_raw_frames_for_latents,
    _raw_window_frames_for_latents,
    _render_counterfactual_branch,
    _should_drop_text_conditioning,
)
from scripts.research_dynamics.metrics import (
    action_mse_per_frame,
    build_metric_rows,
    latent_mse_per_frame,
    rgb_mse_per_frame,
    summarize_metric_rows,
)
from scripts.research_dynamics.rollout import (
    DualExpertDynamicsRollout,
    ParallelStreamDynamicsRollout,
    build_diagnostic_dynamics_request,
    build_dynamics_rollout_adapter,
    resolve_action_per_frame,
    resolve_dynamics_rollout_frame_chunk_size,
    should_drop_task_text_for_fdm_mode,
)
from scripts.research_dynamics.sampling import (
    select_counterfactual_target_only_windows,
    select_early_middle_windows,
)
from scripts.research_dynamics.types import (
    FdmAblationMode,
    FdmStartPolicy,
    FdmWindowSelection,
    dynamics_objective_for_ablation_mode,
)


@pytest.mark.parametrize(
    ("mode", "objective", "rollout_chunk_size"),
    [
        (
            FdmAblationMode.VANILLA_JOINT_ROLLOUT,
            DynamicsObjective.JOINT,
            4,
        ),
        (
            FdmAblationMode.CLEAN_ACTION_FEEDBACK,
            DynamicsObjective.JOINT,
            4,
        ),
        (
            FdmAblationMode.FORCED_ACTION_JOINT_FDM,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            1,
        ),
        (
            FdmAblationMode.VIDEO_CONDITIONED_ACTION,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            1,
        ),
    ],
)
def test_research_modes_map_to_one_core_objective_and_rollout_geometry(
    mode: FdmAblationMode,
    objective: DynamicsObjective,
    rollout_chunk_size: int,
) -> None:
    assert dynamics_objective_for_ablation_mode(mode) is objective
    assert (
        resolve_dynamics_rollout_frame_chunk_size(
            mode,
            configured_frame_chunk_size=4,
        )
        == rollout_chunk_size
    )


@pytest.mark.parametrize(
    "mode",
    (
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        FdmAblationMode.CLEAN_ACTION_FEEDBACK,
        FdmAblationMode.VANILLA_JOINT_ROLLOUT,
    ),
)
def test_diagnostic_request_compilation_is_model_agnostic(
    mode: FdmAblationMode,
) -> None:
    action = torch.randn(1, 4, 7)
    video = torch.randn(1, 48, 1, 4, 4)

    request = build_diagnostic_dynamics_request(
        mode,
        model_action_chunk=action,
        video_condition_latents=video,
    )

    assert request.objective == dynamics_objective_for_ablation_mode(mode)
    assert (request.clean_action is action) is (
        mode == FdmAblationMode.FORCED_ACTION_JOINT_FDM
    )
    assert (request.clean_video is video) is (
        mode == FdmAblationMode.VIDEO_CONDITIONED_ACTION
    )
    assert (request.history_action is action) is (
        mode != FdmAblationMode.VANILLA_JOINT_ROLLOUT
    )


def test_diagnostic_idm_request_can_commit_generated_actions_explicitly() -> None:
    video = torch.randn(1, 48, 1, 4, 4)

    request = build_diagnostic_dynamics_request(
        FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        model_action_chunk=None,
        video_condition_latents=video,
        allow_generated_action_commit=True,
    )

    assert request.clean_video is video
    assert request.history_action is None

    with pytest.raises(ValueError, match="generated-action commit"):
        build_diagnostic_dynamics_request(
            FdmAblationMode.VIDEO_CONDITIONED_ACTION,
            model_action_chunk=None,
            video_condition_latents=video,
        )


def _load_repo_script(relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class _FakeCounterfactualSim:
    def __init__(self, env: _FakeCounterfactualEnv) -> None:
        self.env = env

    def get_state(self) -> SimpleNamespace:
        return SimpleNamespace(
            flatten=lambda: np.asarray([float(self.env.state_index)], dtype=np.float64)
        )

    def set_state_from_flattened(self, value: np.ndarray) -> None:
        self.env.set_state_from_flattened_calls += 1
        self.env.state_index = int(np.asarray(value).reshape(-1)[0])

    def forward(self) -> None:
        pass


class _FakeCounterfactualEnv:
    def __init__(self) -> None:
        self.state_index = 0
        self.sim = _FakeCounterfactualSim(self)
        self.reset_calls = 0
        self.set_init_state_calls = 0
        self.set_state_from_flattened_calls = 0

    def reset(self) -> dict[str, np.ndarray]:
        self.reset_calls += 1
        self.state_index = 0
        return self._get_observations()

    def set_init_state(self, init_state: np.ndarray) -> dict[str, np.ndarray]:
        self.set_init_state_calls += 1
        self.state_index = int(np.asarray(init_state).reshape(-1)[0])
        return self._get_observations()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
        del action
        self.state_index += 1
        return self._get_observations(), 0.0, False, {}

    def _get_observations(self) -> dict[str, np.ndarray]:
        value = int(self.state_index)
        return {
            "agentview_image": np.full((4, 4, 3), value, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full((4, 4, 3), value + 100, dtype=np.uint8),
            "robot0_eef_pos": np.asarray([float(value), 0.0, 0.0], dtype=np.float32),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray(
                [float(value), -float(value)], dtype=np.float32
            ),
        }


@dataclass(frozen=True)
class DummyWindow:
    repo_root: str
    episode_index: int
    start_frame: int
    end_frame: int
    observed_frame_ids: tuple[int, ...]
    task_text: str


class DummyDataset:
    def __init__(self) -> None:
        self.windows = [
            DummyWindow("repo", 0, 0, 100, tuple(range(100)), "task b"),
            DummyWindow("repo", 1, 0, 80, tuple(range(80)), "task a"),
            DummyWindow("repo", 2, 0, 90, tuple(range(90)), "task a"),
            DummyWindow("repo", 3, 0, 95, tuple(range(95)), "task b"),
        ]

    def task_text_for_window_index(self, index: int) -> str:
        return self.windows[index].task_text

    def _window_task_text(self, window: DummyWindow) -> str:
        del window
        raise AssertionError("The explicit indexed task contract should win.")


class _FakeDualExpertVisualTower:
    config = SimpleNamespace(max_text_tokens=4, text_dim=8)

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_runtime_state(self) -> None:
        self.reset_calls += 1


class _FakeDualExpertPolicyVariant:
    action_horizon = 4
    action_dim = 7
    inference_config = SimpleNamespace(frame_chunk_size=2)
    config = SimpleNamespace(
        name="dual_expert",
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        current_block_coupling=CurrentBlockCoupling.JOINT,
    )


class _FakeDualExpertPipeline:
    def __init__(self) -> None:
        self.visual_tower = _FakeDualExpertVisualTower()
        self.policy_variant = _FakeDualExpertPolicyVariant()


class _FakeDualExpertRunner:
    def __init__(self) -> None:
        self.pipeline = _FakeDualExpertPipeline()
        self.seen_contexts = []
        self.seen_video_latents = []

    def reset(self, *, task_text=None, text_context=None, negative_text_context=None):
        return SimpleNamespace(
            policy_state=None,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
        )

    def infer_step(self, *, session, context, video_latents, canonical_video=None):
        del canonical_video
        self.seen_contexts.append(context)
        self.seen_video_latents.append(video_latents)
        predicted_latents = torch.zeros_like(video_latents)
        action_pred = torch.zeros(1, 4, 7)
        return SimpleNamespace(
            session=session,
            infer_output=SimpleNamespace(
                decoder_output=SimpleNamespace(
                    action_pred=action_pred,
                    aux={"predicted_latents": predicted_latents},
                ),
                policy_output=SimpleNamespace(aux={"runtime_mode": "fake"}),
            ),
        )


def test_select_early_middle_windows_is_deterministic_and_chunk_aligned() -> None:
    first = select_early_middle_windows(
        DummyDataset(),
        horizon_frames=8,
        frame_chunk_size=4,
        trajectories_per_task=2,
        seed=7,
    )
    second = select_early_middle_windows(
        DummyDataset(),
        horizon_frames=8,
        frame_chunk_size=4,
        trajectories_per_task=2,
        seed=7,
    )
    assert first == second
    assert len(first) == 4
    assert {item.task_key for item in first} == {"task a", "task b"}
    for selection in first:
        assert selection.generated_frames == 8
        assert selection.t0_frame >= 4
        assert selection.generation_end_frame <= selection.total_video_frames
        assert selection.target_end_frame <= selection.total_video_frames


def test_select_early_middle_windows_rejects_non_chunk_aligned_horizon() -> None:
    with pytest.raises(ValueError, match="exact multiple"):
        select_early_middle_windows(
            DummyDataset(),
            horizon_frames=6,
            frame_chunk_size=4,
            trajectories_per_task=1,
            seed=7,
        )


def test_dual_expert_dynamics_rollout_adapter_threads_conditional_mode_controls() -> (
    None
):
    runner = _FakeDualExpertRunner()
    rollout = DualExpertDynamicsRollout(runner)
    video_context = torch.randn(1, 3, 3, 2, 2)
    action_context = torch.randn(1, 6, 7)
    session = rollout.reset_and_warmup(
        task_text=("task",),
        video_context=video_context,
        action_context=action_context,
        text_context=torch.randn(1, 4, 8),
        negative_text_context=None,
        context_start_frame=5,
        mode=FdmAblationMode.FORCED_ACTION_JOINT_FDM,
    )

    assert runner.pipeline.visual_tower.reset_calls == 1
    assert session.policy_state.step_index == 1
    assert session.policy_state.cursor.current_start_frame == 8
    torch.testing.assert_close(
        session.policy_state.variant_state.past_clean_latents, video_context
    )
    torch.testing.assert_close(
        session.policy_state.variant_state.past_clean_actions, action_context
    )

    raw_action_chunk = torch.randn(1, 4, 7)
    rollout.infer_chunk(
        session=session,
        mode=FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        raw_action_chunk=raw_action_chunk,
    )
    fdm_request = runner.seen_contexts[-1].dynamics
    assert fdm_request.objective == DynamicsObjective.ACTION_CONDITIONED_VIDEO
    assert fdm_request.clean_action is raw_action_chunk
    assert fdm_request.history_action is raw_action_chunk
    assert fdm_request.clean_video is None

    video_condition = torch.randn(1, 3, 2, 2, 2)
    rollout.infer_chunk(
        session=session,
        mode=FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        raw_action_chunk=raw_action_chunk,
        video_condition_latents=video_condition,
    )
    idm_request = runner.seen_contexts[-1].dynamics
    assert idm_request.objective == DynamicsObjective.VIDEO_CONDITIONED_ACTION
    assert idm_request.clean_video is video_condition
    assert idm_request.history_action is raw_action_chunk
    assert idm_request.clean_action is None
    assert runner.seen_video_latents[-1] is video_condition


@pytest.mark.parametrize(
    "program",
    (
        VideoActionProgram.FORWARD_DYNAMICS,
        VideoActionProgram.INVERSE_DYNAMICS,
    ),
)
def test_dynamics_rollout_adapters_accept_fixed_programs(
    program: VideoActionProgram,
) -> None:
    dual_runner = _FakeDualExpertRunner()
    dual_runner.pipeline.policy_variant.config = SimpleNamespace(
        name="dual_expert",
        program=program,
        current_block_coupling=CurrentBlockCoupling.JOINT,
    )
    DualExpertDynamicsRollout(dual_runner)

    parallel_runner = SimpleNamespace(
        policy_variant=SimpleNamespace(
            config=ParallelStreamPolicyConfig(
                program=program,
                hidden_size=32,
            )
        )
    )
    ParallelStreamDynamicsRollout(parallel_runner)


def test_dynamics_rollout_adapters_reject_planning_programs() -> None:
    dual_runner = _FakeDualExpertRunner()
    dual_runner.pipeline.policy_variant.config = SimpleNamespace(
        name="dual_expert",
        program=VideoActionProgram.JOINT,
        current_block_coupling=CurrentBlockCoupling.JOINT,
    )
    with pytest.raises(ValueError, match="requires a GJD"):
        DualExpertDynamicsRollout(dual_runner)

    parallel_runner = SimpleNamespace(
        policy_variant=SimpleNamespace(
            config=ParallelStreamPolicyConfig(
                program=VideoActionProgram.JOINT,
                hidden_size=32,
            )
        )
    )
    with pytest.raises(ValueError, match="requires a GJD"):
        ParallelStreamDynamicsRollout(parallel_runner)


@pytest.mark.parametrize(
    ("architecture", "runner_name", "adapter_name"),
    (
        ("dual_expert", "VariantRolloutRunner", "DualExpertDynamicsRollout"),
        ("parallel_stream", "LingbotExactRunner", "ParallelStreamDynamicsRollout"),
    ),
)
def test_dynamics_adapter_builder_applies_one_checkpoint_and_device_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    architecture: str,
    runner_name: str,
    adapter_name: str,
) -> None:
    import open_wam.pipelines as pipeline_api
    import scripts.research_dynamics.rollout as rollout_module

    class FakePipeline:
        def __init__(self) -> None:
            self.to_calls: list[dict[str, object]] = []
            self.eval_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls.append({"args": args, "kwargs": kwargs})
            return self

        def eval(self):
            self.eval_calls += 1
            return self

    pipeline = FakePipeline()
    load_calls: list[tuple[object, Path, object]] = []
    runner_calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        pipeline_api,
        "build_variant_pipeline_from_config",
        lambda config: pipeline,
    )
    monkeypatch.setattr(
        rollout_module,
        "_load_pipeline_checkpoint",
        lambda value, path, *, compatibility: load_calls.append(
            (value, path, compatibility)
        ),
    )
    monkeypatch.setattr(
        pipeline_api,
        runner_name,
        lambda value: runner_calls.append((runner_name, value)) or value,
    )
    monkeypatch.setattr(
        rollout_module,
        adapter_name,
        lambda value: (adapter_name, value),
    )

    checkpoint = tmp_path / "model_state.pt"
    compatibility = "allow_checkpoint_superset"
    result = build_dynamics_rollout_adapter(
        config=SimpleNamespace(
            policy_variant=SimpleNamespace(name=architecture),
        ),
        checkpoint_file=checkpoint,
        runtime_device=torch.device("cpu"),
        runtime_dtype=torch.bfloat16,
        checkpoint_compatibility=compatibility,
    )

    assert load_calls == [(pipeline, checkpoint, compatibility)]
    assert pipeline.to_calls == [
        {
            "args": (),
            "kwargs": {
                "device": torch.device("cpu"),
                "dtype": torch.bfloat16,
            },
        }
    ]
    assert pipeline.eval_calls == 1
    assert runner_calls == [(runner_name, pipeline)]
    assert result == (adapter_name, pipeline)


def test_latest_fit_start_policy_uses_full_target_horizon() -> None:
    selections = select_early_middle_windows(
        DummyDataset(),
        horizon_frames=32,
        frame_chunk_size=4,
        trajectories_per_task=1,
        seed=7,
        start_policy=FdmStartPolicy.LATEST_FIT,
    )
    assert len(selections) == 2
    for selection in selections:
        assert (
            selection.t0_frame
            == selection.total_video_frames - selection.horizon_frames
        )
        assert selection.target_end_frame == selection.total_video_frames


def test_latest_fit_start_policy_respects_target_start_offset() -> None:
    selections = select_early_middle_windows(
        DummyDataset(),
        horizon_frames=32,
        frame_chunk_size=4,
        trajectories_per_task=1,
        seed=7,
        start_policy=FdmStartPolicy.LATEST_FIT,
        target_start_offset_frames=1,
    )
    assert len(selections) == 2
    for selection in selections:
        assert (
            selection.t0_frame
            == selection.total_video_frames - selection.horizon_frames - 1
        )
        assert selection.target_start_frame == selection.t0_frame + 1
        assert selection.target_end_frame == selection.total_video_frames


def test_latest_fit_start_policy_can_reserve_tail_margin_without_mode_offset() -> None:
    selections = select_early_middle_windows(
        DummyDataset(),
        horizon_frames=32,
        frame_chunk_size=4,
        trajectories_per_task=1,
        seed=7,
        start_policy=FdmStartPolicy.LATEST_FIT,
        fit_target_start_offset_frames=1,
    )
    assert len(selections) == 2
    for selection in selections:
        assert (
            selection.t0_frame
            == selection.total_video_frames - selection.horizon_frames - 1
        )
        assert selection.target_start_offset_frames == 0
        assert selection.target_start_frame == selection.t0_frame


def test_counterfactual_action_branches_preserve_gripper_and_clip() -> None:
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 0] = np.linspace(-0.2, 0.2, num=8, dtype=np.float32)
    actions[:, 1] = 0.3
    actions[:, 2] = -0.4
    actions[:, 6] = 0.75

    gt = apply_action_branch(actions, branch_name="gt", seed=0)
    assert np.array_equal(gt, actions)

    biased = apply_action_branch(actions, branch_name="bias_x_pos", seed=0)
    assert np.allclose(biased[:, 0], actions[:, 0] + 0.12)
    assert np.allclose(biased[:, 6], actions[:, 6])

    noisy = apply_action_branch(actions, branch_name="noise_small", seed=123)
    assert not np.array_equal(noisy[:, :6], actions[:, :6])
    assert np.allclose(noisy[:, 6], actions[:, 6])
    assert np.max(np.abs(noisy[:, :6])) <= 1.0

    stopped = apply_action_branch(actions, branch_name="stop_motion", seed=0)
    assert np.allclose(stopped[:, :6], 0.0)
    assert np.allclose(stopped[:, 6], actions[:, 6])

    reversed_translation = apply_action_branch(
        actions, branch_name="reverse_translation", seed=0
    )
    assert np.allclose(reversed_translation[:, :3], -actions[:, :3])
    assert np.allclose(reversed_translation[:, 6], actions[:, 6])

    swapped = apply_action_branch(actions, branch_name="swap_xy_clockwise", seed=0)
    assert np.allclose(swapped[:, 0], actions[:, 1])
    assert np.allclose(swapped[:, 1], -actions[:, 0])
    assert np.allclose(swapped[:, 6], actions[:, 6])

    saturated = apply_action_branch(actions, branch_name="saturate_z_up", seed=0)
    assert np.allclose(saturated[:, 0], 0.0)
    assert np.allclose(saturated[:, 1], 0.0)
    assert np.allclose(saturated[:, 2], 1.0)
    assert np.allclose(saturated[:, 3:6], 0.0)
    assert np.allclose(saturated[:, 6], actions[:, 6])

    scaled = apply_action_branch(actions, branch_name="scale_demo_0p5", seed=0)
    assert np.allclose(scaled[:, :6], actions[:, :6] * 0.5)
    assert np.allclose(scaled[:, 6], actions[:, 6])

    pulse = apply_action_branch(actions, branch_name="axis_pulse_x_neg", seed=0)
    assert np.allclose(pulse[:4, 0], -0.8)
    assert np.allclose(pulse[4:, 0], 0.0)
    assert np.allclose(pulse[:, 6], actions[:, 6])


def test_branch_presets_expand_and_record_metadata() -> None:
    expanded = expand_branch_names("training_10,gt")
    assert expanded == BRANCH_PRESETS["training_10"]
    metadata = branch_metadata("axis_pulse_x_neg")
    assert metadata["family"] == "axis_pulse"
    assert metadata["strength"] == "strong"


def test_counterfactual_wan_temporal_window_formulas() -> None:
    assert _raw_window_frames_for_latents(4, action_per_frame=4) == 13
    assert _raw_window_frames_for_latents(16, action_per_frame=4) == 61
    assert _decoded_raw_frames_for_latents(4, action_per_frame=4) == 17
    assert _decoded_raw_frames_for_latents(16, action_per_frame=4) == 65
    assert _decoded_raw_frames_for_latents(32, action_per_frame=4) == 129


def test_counterfactual_dataset_builder_excludes_manifest_source_episodes(
    tmp_path: Path,
) -> None:
    builder = _load_repo_script(
        "scripts/build_libero_fdm_counterfactual_demo_dataset.py"
    )
    excluded_root = tmp_path / "prior_dataset"
    excluded_root.mkdir()
    (excluded_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_episodes": [
                    {"dataset_episode_index": 1},
                    {"dataset_episode_index": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    replay_rows = [
        {
            "replay_status": "success",
            "failure": False,
            "dataset_episode_index": index,
            "metadata_task_index": 0,
            "resolved_init_state_index": 0,
            "parquet_path": f"/tmp/episode_{index}.parquet",
            "task_text": "task",
        }
        for index in (1, 2, 3, 4)
    ]

    excluded = builder._load_excluded_episode_indices([str(excluded_root)])
    selected = builder._select_source_episodes(
        replay_rows,
        task_ids=(0,),
        episodes_per_task=2,
        seed=0,
        excluded_episode_indices=excluded,
    )

    assert excluded == {1, 3}
    assert [episode.dataset_episode_index for episode in selected] == [2, 4]


def test_counterfactual_dataset_builder_samples_random_t0_with_partial_context() -> (
    None
):
    builder = _load_repo_script(
        "scripts/build_libero_fdm_counterfactual_demo_dataset.py"
    )

    first = builder._select_t0_frames(
        total_video_frames=24,
        horizon_frames=8,
        context_window_frames=16,
        t0_fractions=(0.2, 0.4, 0.6, 0.8),
        sampling_mode=builder.T0SamplingMode.UNIFORM_RANDOM,
        samples_per_episode=5,
        min_context_frames=1,
        min_separation_frames=2,
        seed=17,
        episode_index=3,
    )
    second = builder._select_t0_frames(
        total_video_frames=24,
        horizon_frames=8,
        context_window_frames=16,
        t0_fractions=(0.2, 0.4, 0.6, 0.8),
        sampling_mode=builder.T0SamplingMode.UNIFORM_RANDOM,
        samples_per_episode=5,
        min_context_frames=1,
        min_separation_frames=2,
        seed=17,
        episode_index=3,
    )

    assert first == second
    assert len(first) == 5
    frames = [frame for requested, frame in first]
    assert all(requested is None for requested, _ in first)
    assert frames == sorted(frames)
    assert min(frames) >= 1
    assert max(frames) <= 15
    assert min(abs(left - right) for left, right in zip(frames, frames[1:])) >= 2
    assert any(frame < 16 for frame in frames)


def test_counterfactual_dataset_builder_relaxes_random_t0_separation_for_short_episodes() -> (
    None
):
    builder = _load_repo_script(
        "scripts/build_libero_fdm_counterfactual_demo_dataset.py"
    )

    selected = builder._select_t0_frames(
        total_video_frames=50,
        horizon_frames=32,
        context_window_frames=16,
        t0_fractions=(0.2, 0.4, 0.6, 0.8),
        sampling_mode=builder.T0SamplingMode.UNIFORM_RANDOM,
        samples_per_episode=4,
        min_context_frames=1,
        min_separation_frames=8,
        seed=0,
        episode_index=12,
    )

    frames = [frame for requested, frame in selected]
    assert all(requested is None for requested, _ in selected)
    assert len(frames) == 4
    assert len(set(frames)) == 4
    assert frames == sorted(frames)
    assert min(frames) >= 1
    assert max(frames) <= 17
    assert min(abs(left - right) for left, right in zip(frames, frames[1:])) < 8


def test_counterfactual_dataset_builder_random_t0_count_and_defaults() -> None:
    builder = _load_repo_script(
        "scripts/build_libero_fdm_counterfactual_demo_dataset.py"
    )
    args = builder._parse_args(
        [
            "--replay-status-path",
            "/tmp/replay_status.jsonl",
            "--output-dir",
            "/tmp/counterfactual-output",
            "--t0-sampling-mode",
            "uniform_random",
            "--t0-samples-per-episode",
            "7",
            "--context-window-frames",
            "16",
        ]
    )

    assert (
        builder._resolve_t0_count(
            args,
            t0_fractions=(0.2, 0.4, 0.6, 0.8),
            mode=builder.T0SamplingMode.UNIFORM_RANDOM,
        )
        == 7
    )
    assert (
        builder._resolve_t0_min_context_frames(
            args, mode=builder.T0SamplingMode.UNIFORM_RANDOM
        )
        == 1
    )


def test_counterfactual_dataset_builder_defaults_to_16_context_32_future() -> None:
    builder = _load_repo_script(
        "scripts/build_libero_fdm_counterfactual_demo_dataset.py"
    )

    args = builder._parse_args(
        [
            "--replay-status-path",
            "/tmp/replay_status.jsonl",
            "--output-dir",
            "/tmp/counterfactual-output",
        ]
    )

    assert args.context_window_frames == 16
    assert args.horizon_frames == 32
    assert args.segment_frames == 48


def test_counterfactual_dataset_builder_writes_canonical_view_and_state_payload(
    tmp_path: Path,
) -> None:
    builder = _load_repo_script(
        "scripts/build_libero_fdm_counterfactual_demo_dataset.py"
    )
    obs = {
        "agentview_image": np.full((4, 4, 3), 10, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((4, 4, 3), 20, dtype=np.uint8),
        "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.1, 0.2], dtype=np.float32),
    }
    extracted = builder._extract_obs(obs)
    sequence = builder._obs_sequence_to_payload(
        [extracted],
        frame_index=np.asarray([7], dtype=np.int64),
        source_timestamps=np.arange(10, dtype=np.float64) / 60.0,
        output_fps=60.0,
    )
    path = tmp_path / "sample.npz"

    builder._write_counterfactual_npz(
        path,
        sequence=sequence,
        extra={"future_actions": np.ones((4, 7), dtype=np.float32)},
    )
    payload = np.load(path)

    assert set(payload.files) == {
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
        "observation.state",
        "frame_index",
        "timestamp",
        "future_actions",
    }
    assert payload["observation.images.agentview_rgb"].shape == (1, 4, 4, 3)
    assert payload["observation.images.eye_in_hand_rgb"].shape == (1, 4, 4, 3)
    np.testing.assert_allclose(
        payload["observation.state"][0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.1, 0.2]
    )
    np.testing.assert_array_equal(payload["frame_index"], [7])
    assert payload["timestamp"].dtype == np.float32
    np.testing.assert_allclose(payload["timestamp"], [7.0 / 60.0])


def test_counterfactual_dataset_builder_renders_pre_action_observation_frames(
    tmp_path: Path,
) -> None:
    builder = _load_repo_script(
        "scripts/build_libero_fdm_counterfactual_demo_dataset.py"
    )
    env = _FakeCounterfactualEnv()
    actions = np.zeros((20, 7), dtype=np.float32)
    source_timestamps = np.arange(20, dtype=np.float32) / 60.0
    episode = builder.SourceEpisode(
        dataset_episode_index=0,
        task_id=0,
        task_text="task",
        init_state_index=0,
        parquet_path=tmp_path / "episode.parquet",
    )

    context = builder._render_context_at_t0(
        env=env,
        init_state=np.asarray([0.0], dtype=np.float32),
        episode=episode,
        actions=actions,
        t0_frame=2,
        requested_t0_fraction=None,
        total_video_frames=5,
        context_window_frames=2,
        action_per_frame=4,
        output_fps=60.0,
        source_timestamps=source_timestamps,
        output_root=tmp_path,
        context_id=0,
    )
    context_payload = np.load(context.context_path)

    np.testing.assert_array_equal(context_payload["frame_index"], np.arange(5))
    np.testing.assert_array_equal(
        context_payload["observation.state"][:, 0], np.arange(5, dtype=np.float32)
    )
    assert context.simulator_state.tolist() == [8.0]
    assert env.state_index == 8
    set_init_state_calls_after_context = env.set_init_state_calls

    target = builder._render_future_from_state(
        env=env,
        flattened_state=np.asarray([8.0], dtype=np.float64),
        future_actions=actions[:8],
        target_raw_frames=5,
        start_action_index=8,
        output_fps=60.0,
        source_timestamps=source_timestamps,
    )

    np.testing.assert_array_equal(target.frame_index, np.arange(8, 13))
    np.testing.assert_array_equal(
        target.state[:, 0], np.arange(8, 13, dtype=np.float32)
    )
    assert env.state_index == 12
    assert env.set_init_state_calls == set_init_state_calls_after_context
    assert env.set_state_from_flattened_calls == 1


def test_counterfactual_ablation_render_includes_cached_t0_observation(
    tmp_path: Path,
) -> None:
    env = _FakeCounterfactualEnv()
    actions = np.zeros((32, 7), dtype=np.float32)
    case = CounterfactualCase(
        case_index=0,
        episode_index=0,
        task_text="task",
        task_id=0,
        init_state_index=0,
        parquet_path=tmp_path / "episode.parquet",
        t0_frame=2,
        context_start_frame=1,
        action_length=int(actions.shape[0]),
    )

    branch = _render_counterfactual_branch(
        env=env,
        init_state=np.asarray([0.0], dtype=np.float32),
        case=case,
        actions=actions,
        branch_name="gt",
        horizon_frames=1,
        generated_frames=1,
        action_per_frame=4,
        seed=0,
        output_root=tmp_path,
        video_fps=60.0,
    )

    assert branch.target_rgb.shape[0] == 5
    assert len(branch.future_obs) == 5
    assert env.state_index == 12
    np.testing.assert_allclose(
        branch.target_rgb[:, 0, 0, 0], np.arange(8, 13, dtype=np.float32) / 255.0
    )


def test_counterfactual_dataset_builder_plan_only_overwrite_does_not_delete_before_validation(
    tmp_path: Path,
) -> None:
    builder = _load_repo_script(
        "scripts/build_libero_fdm_counterfactual_demo_dataset.py"
    )
    output_dir = tmp_path / "outputs"
    run_id = "existing"
    output_root = output_dir / run_id
    output_root.mkdir(parents=True)
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "replay_status": "success",
                "failure": False,
                "dataset_episode_index": 7,
                "metadata_task_index": 0,
                "resolved_init_state_index": 0,
                "parquet_path": "/tmp/episode_7.parquet",
                "task_text": "task",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target-transitions"):
        builder.main(
            [
                "--replay-status-path",
                str(replay_path),
                "--output-dir",
                str(output_dir),
                "--run-id",
                run_id,
                "--task-ids",
                "0",
                "--episodes-per-task",
                "1",
                "--t0-fractions",
                "0.5",
                "--branches",
                "gt",
                "--target-transitions",
                "2",
                "--plan-only",
                "--overwrite",
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_counterfactual_encoder_accepts_single_root_dataset(tmp_path: Path) -> None:
    encoder = _load_repo_script("scripts/encode_libero_fdm_counterfactual_dataset.py")
    dataset_root = tmp_path / "split"
    (dataset_root / "metadata").mkdir(parents=True)
    (dataset_root / "metadata" / "contexts.jsonl").write_text("", encoding="utf-8")
    (dataset_root / "metadata" / "transitions.jsonl").write_text("", encoding="utf-8")

    assert encoder._resolve_shards(dataset_root, None) == [dataset_root]


def test_counterfactual_encoder_rejects_existing_latents_without_overwrite(
    tmp_path: Path,
) -> None:
    encoder = _load_repo_script("scripts/encode_libero_fdm_counterfactual_dataset.py")
    dataset_root = tmp_path / "split"
    (dataset_root / "metadata").mkdir(parents=True)
    (dataset_root / "metadata" / "contexts.jsonl").write_text(
        json.dumps({"context_id": 0}) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "metadata" / "transitions.jsonl").write_text(
        json.dumps({"sample_id": 0}) + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "encoded"
    existing = (
        output_root / dataset_root.name / "contexts" / "context_000000_latents.pt"
    )
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"stale")

    with pytest.raises(FileExistsError, match="--overwrite"):
        encoder._validate_no_existing_encoded_outputs(
            output_root=output_root,
            shard_roots=[dataset_root],
            max_contexts=None,
            max_samples=None,
        )


def test_counterfactual_encoder_uses_reference_asset_streaming_encode() -> None:
    encoder = _load_repo_script("scripts/encode_libero_fdm_counterfactual_dataset.py")

    class FakeAssets:
        def __init__(self) -> None:
            self.calls = []

        def encode_video(self, video, *, placements=None, reset_cache=True):
            self.calls.append((video, placements, reset_cache))
            return torch.ones(video.shape[0], 4, video.shape[2], 8, 16)

    assets = FakeAssets()
    video = torch.zeros(2, 3, 4, 128, 256)

    latents = encoder._encode_libero_side_by_side_video(
        assets,
        video,
        device=torch.device("cpu"),
    )

    assert latents.shape == (2, 4, 4, 8, 16)
    assert len(assets.calls) == 1
    _, placements, reset_cache = assets.calls[0]
    assert reset_cache is True
    assert tuple(placement.canonical_name for placement in placements) == (
        "image",
        "wrist_image",
    )


def test_counterfactual_encoder_loads_canonical_separate_views(tmp_path: Path) -> None:
    encoder = _load_repo_script("scripts/encode_libero_fdm_counterfactual_dataset.py")
    path = tmp_path / "payload.npz"
    np.savez(
        path,
        **{
            "observation.images.agentview_rgb": np.zeros(
                (2, 128, 128, 3), dtype=np.uint8
            ),
            "observation.images.eye_in_hand_rgb": np.ones(
                (2, 128, 128, 3), dtype=np.uint8
            ),
        },
    )

    rgb = encoder._load_counterfactual_rgb(np.load(path), legacy_key="target_rgb")

    assert rgb.shape == (2, 128, 256, 3)
    assert np.all(rgb[:, :, :128] == 0)
    assert np.all(rgb[:, :, 128:] == 1)


def test_counterfactual_encoder_builds_single_frame_condition_latents() -> None:
    encoder = _load_repo_script("scripts/encode_libero_fdm_counterfactual_dataset.py")

    assert encoder._condition_source_frame_indices(
        raw_frame_count=17,
        latent_frames=5,
        source_frame_offset=-1,
    ) == [0, 4, 8, 12, 16]

    class FakeAssets:
        def __init__(self) -> None:
            self.calls = []

        def encode_video(self, video, *, placements=None, reset_cache=True):
            self.calls.append((video.detach().clone(), placements, reset_cache))
            values = video[:, :, 0].mean(dim=(1, 2, 3))
            return values[:, None, None, None, None].expand(video.shape[0], 2, 1, 2, 4)

    rgb = np.zeros((17, 128, 256, 3), dtype=np.uint8)
    rgb[:, :, :, 0] = np.arange(17, dtype=np.uint8)[:, None, None]
    assets = FakeAssets()

    latents = encoder._encode_condition_latents_for_rgb(
        assets,
        rgb,
        latent_frames=5,
        source_frame_offset=-1,
        condition_batch_size=2,
        device=torch.device("cpu"),
        output_dtype=torch.float32,
    )

    assert latents.shape == (2, 5, 2, 4)
    assert len(assets.calls) == 3
    torch.testing.assert_close(
        latents[0, :, 0, 0],
        torch.tensor([0, 4, 8, 12, 16], dtype=torch.float32) / (3.0 * 255.0),
    )


def test_counterfactual_drops_text_for_fdm_mode_even_without_flag() -> None:
    assert should_drop_task_text_for_fdm_mode(FdmAblationMode.FORCED_ACTION_JOINT_FDM)
    assert should_drop_task_text_for_fdm_mode(FdmAblationMode.VIDEO_CONDITIONED_ACTION)
    assert not should_drop_task_text_for_fdm_mode(FdmAblationMode.CLEAN_ACTION_FEEDBACK)
    assert not should_drop_task_text_for_fdm_mode(FdmAblationMode.VANILLA_JOINT_ROLLOUT)
    assert _should_drop_text_conditioning(
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        fdm_drop_text_conditioning=True,
    )
    assert _should_drop_text_conditioning(
        FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        fdm_drop_text_conditioning=False,
    )
    assert _should_drop_text_conditioning(
        FdmAblationMode.CLEAN_ACTION_FEEDBACK,
        fdm_drop_text_conditioning=True,
    )
    assert not _should_drop_text_conditioning(
        FdmAblationMode.VANILLA_JOINT_ROLLOUT,
        fdm_drop_text_conditioning=True,
    )
    assert _should_drop_text_conditioning(
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        fdm_drop_text_conditioning=False,
    )


def test_fdm_rollout_warmup_receives_rollout_mode() -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        pipeline = SimpleNamespace(
            visual_tower=SimpleNamespace(
                config=SimpleNamespace(max_text_tokens=5, text_dim=6),
            )
        )
        policy_variant = SimpleNamespace(
            config=ParallelStreamPolicyConfig(
                program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
                hidden_size=32,
            )
        )

        def reset(self, **_kwargs):
            return SimpleNamespace(
                policy_state=SimpleNamespace(
                    cache={},
                    cursor=SimpleNamespace(block_index=0, chunk_size=2),
                )
            )

        def warmup_cache(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(session="warmed")

    rollout = ParallelStreamDynamicsRollout(FakeRunner())
    text_context = torch.randn(1, 5, 6)
    negative_text_context = torch.randn(1, 5, 6)
    hidden_proprio_history = torch.randn(1, 2, 4)
    session = rollout.reset_and_warmup(
        task_text=("task",),
        video_context=torch.zeros(1, 4, 2, 2, 2),
        action_context=torch.zeros(1, 4, 3),
        text_context=text_context,
        negative_text_context=negative_text_context,
        context_start_frame=3,
        mode=FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        drop_text_conditioning=True,
        hidden_proprio_history=hidden_proprio_history,
    )

    assert session == "warmed"
    request = captured["dynamics"]
    assert request.objective == DynamicsObjective.ACTION_CONDITIONED_VIDEO
    assert torch.count_nonzero(captured["text_context"]) == 0
    assert torch.count_nonzero(captured["negative_text_context"]) == 0
    assert captured["hidden_proprio_history"] is hidden_proprio_history


def test_fdm_cli_accepts_training_style_set_overrides() -> None:
    from scripts.research_dynamics.cli import _parse_args, _resolve_runtime_dtype

    args = _parse_args(
        [
            "--checkpoint",
            "/tmp/checkpoint",
            "--runtime-dtype",
            "bfloat16",
            "--set",
            "policy_variant.generalist_mode_text_token=true",
            "--set",
            "policy_variant.proprio_context_mode=per_chunk_additive",
        ]
    )

    assert args.set_overrides == [
        "policy_variant.generalist_mode_text_token=true",
        "policy_variant.proprio_context_mode=per_chunk_additive",
    ]
    assert args.runtime_dtype == "bfloat16"
    assert args.mode is None
    assert _resolve_runtime_dtype(args.runtime_dtype) is torch.bfloat16


@pytest.mark.parametrize(
    ("program", "expected_mode"),
    [
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        ),
    ],
)
def test_fdm_cli_derives_omitted_mode_from_fixed_program(
    program: VideoActionProgram,
    expected_mode: FdmAblationMode,
) -> None:
    from scripts.research_dynamics.cli import _resolve_requested_diagnostic_modes

    config = SimpleNamespace(
        policy_variant=ParallelStreamPolicyConfig(
            program=program,
        )
    )

    assert _resolve_requested_diagnostic_modes(config, None) == (expected_mode,)


def test_fdm_cli_rejects_explicit_mode_conflicting_with_fixed_program() -> None:
    from scripts.research_dynamics.cli import _resolve_requested_diagnostic_modes

    config = SimpleNamespace(
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.FORWARD_DYNAMICS,
        )
    )

    with pytest.raises(
        ValueError,
        match="forward_dynamics.*forced_action_joint_fdm.*video_conditioned_action",
    ):
        _resolve_requested_diagnostic_modes(
            config,
            [FdmAblationMode.VIDEO_CONDITIONED_ACTION.value],
        )


def test_fdm_cli_keeps_all_default_modes_for_nonfixed_gjd() -> None:
    from scripts.research_dynamics.cli import _resolve_requested_diagnostic_modes

    config = SimpleNamespace(
        policy_variant=ParallelStreamPolicyConfig(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        )
    )

    assert _resolve_requested_diagnostic_modes(config, None) == tuple(FdmAblationMode)


def test_fdm_local_path_overrides_are_explicit(tmp_path: Path) -> None:
    from scripts.research_dynamics.cli import _resolve_existing_path_override

    assert _resolve_existing_path_override(explicit=None) is None

    existing = tmp_path / "asset"
    existing.mkdir()
    assert _resolve_existing_path_override(explicit=str(existing)) == existing.resolve()

    with pytest.raises(
        FileNotFoundError, match="Configured override path does not exist"
    ):
        _resolve_existing_path_override(explicit=str(tmp_path / "missing"))


def test_fdm_cli_resolves_action_per_frame_for_m1_and_m5_configs() -> None:
    m1_config = SimpleNamespace(
        policy_variant=SimpleNamespace(action_per_frame=3),
        action_decoder=SimpleNamespace(action_horizon=12),
        inference=SimpleNamespace(frame_chunk_size=4),
    )
    assert resolve_action_per_frame(m1_config) == 3

    m5_config = SimpleNamespace(
        policy_variant=SimpleNamespace(),
        action_decoder=SimpleNamespace(action_horizon=16),
        inference=SimpleNamespace(frame_chunk_size=4),
    )
    assert resolve_action_per_frame(m5_config) == 4

    invalid_config = SimpleNamespace(
        policy_variant=SimpleNamespace(),
        action_decoder=SimpleNamespace(action_horizon=10),
        inference=SimpleNamespace(frame_chunk_size=4),
    )
    with pytest.raises(ValueError, match="must be positive and divisible"):
        resolve_action_per_frame(invalid_config)


def test_fdm_cli_drops_latent_rows_when_rgb_temporal_resolution_differs() -> None:
    from scripts.research_dynamics.cli import _latent_mse_for_metric_rows

    assert (
        _latent_mse_for_metric_rows(
            latent_mse=[0.1, 0.2],
            rgb_mse=[0.3, 0.4, 0.5],
            action_mse=None,
        )
        is None
    )
    assert _latent_mse_for_metric_rows(
        latent_mse=[0.1, 0.2],
        rgb_mse=[0.3, 0.4],
        action_mse=None,
    ) == [0.1, 0.2]


def test_fdm_eval_selects_counterfactual_target_only_windows() -> None:
    class FakeCounterfactualDataset:
        encoded_root = Path("/encoded/cf")
        transition_rows = [
            {
                "sample_id": 0,
                "context_id": 10,
                "task_id": 1,
                "task_text": "task one",
                "dataset_episode_index": 3,
                "branch": "gt",
                "branch_family": "demo",
                "target_video_latent_shape": [48, 8, 8, 16],
                "t0_frame": 20,
            },
            {
                "sample_id": 1,
                "context_id": 11,
                "task_id": 0,
                "task_text": "task zero",
                "dataset_episode_index": 4,
                "branch": "stop_motion",
                "branch_family": "counterfactual",
                "target_video_latent_shape": [48, 32, 8, 16],
                "t0_frame": 8,
            },
        ]

        def build_balanced_source_indices(self) -> tuple[int, ...]:
            return (1, 0)

    selections = select_counterfactual_target_only_windows(
        FakeCounterfactualDataset(),
        horizon_frames=16,
        frame_chunk_size=4,
        target_start_offset_frames=1,
    )

    assert len(selections) == 1
    selection = selections[0]
    assert selection.dataset_index == 1
    assert selection.t0_frame == 0
    assert selection.target_start_frame == 1
    assert selection.context_start_frame == 0
    assert selection.total_video_frames == 32
    assert selection.task_key == "task:0:branch:stop_motion"
    assert selection.source_metadata["sample_id"] == 1
    assert selection.source_metadata["task_text"] == "task zero"


def test_fdm_cli_rejects_unsupported_counterfactual_modes() -> None:
    from scripts.research_dynamics.cli import _validate_counterfactual_eval_modes

    _validate_counterfactual_eval_modes(
        (
            FdmAblationMode.FORCED_ACTION_JOINT_FDM,
            FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        )
    )
    with pytest.raises(ValueError, match="target-only"):
        _validate_counterfactual_eval_modes((FdmAblationMode.VANILLA_JOINT_ROLLOUT,))


def test_counterfactual_cli_accepts_training_style_set_overrides() -> None:
    from scripts.research_dynamics.counterfactual import _parse_args

    args = _parse_args(
        [
            "--checkpoint",
            "/tmp/checkpoint",
            "--replay-status-path",
            "/tmp/replay_status.jsonl",
            "--episode-indices",
            "0",
            "--set",
            "policy_variant.generalist_mode_text_token=true",
            "--set",
            "policy_variant.proprio_context_mode=per_chunk_additive",
        ]
    )

    assert args.set_overrides == [
        "policy_variant.generalist_mode_text_token=true",
        "policy_variant.proprio_context_mode=per_chunk_additive",
    ]


def test_fdm_eval_threads_per_frame_proprio_to_warmup_and_chunks(
    tmp_path: Path,
) -> None:
    from scripts.research_dynamics.cli import _run_one_selection_mode

    captured_warmup: dict[str, object] = {}
    captured_chunks: list[torch.Tensor | None] = []

    class FakeRollout:
        action_per_frame = 2
        frame_chunk_size = 2

        def reset_and_warmup(self, **kwargs):
            captured_warmup.update(kwargs)
            return "session0"

        def infer_chunk(self, **kwargs):
            captured_chunks.append(kwargs.get("proprio_state"))
            return SimpleNamespace(
                session=f"session{len(captured_chunks)}",
                predicted_latents=kwargs["video_condition_latents"].detach().clone(),
                raw_action_sequence=torch.zeros(1, 2, 3),
                debug={"rollout_window_size": 3},
            )

    sample = SimpleNamespace(
        video_latents=torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1),
        actions=torch.zeros(16, 3),
        task_text="task",
        text_context=None,
        negative_text_context=None,
        proprio_context_state=torch.arange(64, dtype=torch.float32).reshape(8, 8),
        proprio_context_state_mask=torch.ones(8, 8),
    )
    selection = FdmWindowSelection(
        sample_index=0,
        dataset_index=0,
        task_key="task",
        task_rank=0,
        episode_index=0,
        t0_frame=2,
        horizon_frames=4,
        generated_frames=4,
        context_start_frame=0,
        total_video_frames=8,
        repo_root="repo",
    )

    result = _run_one_selection_mode(
        fdm_rollout=FakeRollout(),
        selection=selection,
        sample=sample,
        mode=FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        runtime_device=torch.device("cpu"),
        decode_device=torch.device("cpu"),
        seed=0,
        write_video=False,
        video_dir=tmp_path,
        video_fps=8.0,
    )

    torch.testing.assert_close(
        captured_warmup["proprio_state"], sample.proprio_context_state[2].unsqueeze(0)
    )
    assert captured_warmup["video_context"].shape[2] == 3
    assert len(captured_chunks) == 4
    torch.testing.assert_close(
        captured_chunks[0], sample.proprio_context_state[2].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_chunks[1], sample.proprio_context_state[3].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_chunks[2], sample.proprio_context_state[4].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_chunks[3], sample.proprio_context_state[5].unsqueeze(0)
    )
    assert result["summary"]["metric_target"] == "action"


def test_fdm_eval_target_only_offset_predicts_future_from_current_action(
    tmp_path: Path,
) -> None:
    from scripts.research_dynamics.cli import _run_one_selection_mode

    captured_warmup: dict[str, object] = {}
    captured_actions: list[torch.Tensor] = []
    captured_videos: list[torch.Tensor] = []
    captured_proprio: list[torch.Tensor | None] = []

    class FakeRollout(DualExpertDynamicsRollout):
        action_per_frame = 2
        frame_chunk_size = 2
        runner = SimpleNamespace(pipeline=None)

        def __init__(self) -> None:
            pass

        def reset_and_warmup(self, **kwargs):
            captured_warmup.update(kwargs)
            return "session0"

        def infer_chunk(self, **kwargs):
            captured_actions.append(kwargs["raw_action_chunk"].detach().clone())
            captured_videos.append(kwargs["video_condition_latents"].detach().clone())
            captured_proprio.append(kwargs.get("proprio_state"))
            return SimpleNamespace(
                session=f"session{len(captured_actions)}",
                predicted_latents=kwargs["video_condition_latents"].detach().clone(),
                raw_action_sequence=kwargs["raw_action_chunk"].detach().clone(),
                debug={"rollout_window_size": 3},
            )

    video_latents = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1)
    actions = torch.arange(60, dtype=torch.float32).reshape(20, 3)
    sample = SimpleNamespace(
        video_latents=video_latents,
        actions=actions,
        task_text="task",
        text_context=None,
        negative_text_context=None,
        proprio_context_state=torch.arange(80, dtype=torch.float32).reshape(10, 8),
        proprio_context_state_mask=torch.ones(10, 8),
    )
    selection = FdmWindowSelection(
        sample_index=0,
        dataset_index=0,
        task_key="task",
        task_rank=0,
        episode_index=0,
        t0_frame=2,
        horizon_frames=4,
        generated_frames=4,
        context_start_frame=0,
        total_video_frames=10,
        repo_root="repo",
        target_start_offset_frames=1,
    )

    result = _run_one_selection_mode(
        fdm_rollout=FakeRollout(),
        selection=selection,
        sample=sample,
        mode=FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        runtime_device=torch.device("cpu"),
        decode_device=torch.device("cpu"),
        seed=0,
        write_video=False,
        video_dir=tmp_path,
        video_fps=8.0,
    )

    assert captured_warmup["video_context"].shape[2] == 3
    assert torch.equal(
        captured_warmup["action_context"],
        torch.zeros_like(captured_warmup["action_context"]),
    )
    torch.testing.assert_close(
        captured_warmup["proprio_state"], sample.proprio_context_state[2].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_warmup["hidden_proprio_history"],
        sample.proprio_context_state[:3].unsqueeze(0),
    )
    assert len(captured_videos) == 4
    torch.testing.assert_close(
        captured_videos[0], sample.video_latents[:, 3:4].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_videos[1], sample.video_latents[:, 4:5].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_videos[2], sample.video_latents[:, 5:6].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_videos[3], sample.video_latents[:, 6:7].unsqueeze(0)
    )
    torch.testing.assert_close(captured_actions[0], sample.actions[4:6].unsqueeze(0))
    torch.testing.assert_close(captured_actions[1], sample.actions[6:8].unsqueeze(0))
    torch.testing.assert_close(captured_actions[2], sample.actions[8:10].unsqueeze(0))
    torch.testing.assert_close(captured_actions[3], sample.actions[10:12].unsqueeze(0))
    torch.testing.assert_close(
        captured_proprio[0], sample.proprio_context_state[2].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_proprio[1], sample.proprio_context_state[3].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_proprio[2], sample.proprio_context_state[4].unsqueeze(0)
    )
    torch.testing.assert_close(
        captured_proprio[3], sample.proprio_context_state[5].unsqueeze(0)
    )
    assert result["summary"]["target_start_frame"] == 3
    assert result["summary"]["action_source_start_frame"] == 2
    assert result["metric_rows"][0]["future_frame"] == 3


def test_fdm_eval_m5_vanilla_ignores_selection_fit_target_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.research_dynamics.cli as cli_module
    from scripts.research_dynamics.cli import _run_one_selection_mode

    captured_warmup: dict[str, object] = {}

    class FakeRollout(DualExpertDynamicsRollout):
        action_per_frame = 2
        frame_chunk_size = 2
        runner = SimpleNamespace(pipeline=None)

        def __init__(self) -> None:
            pass

        def reset_and_warmup(self, **kwargs):
            captured_warmup.update(kwargs)
            return "session0"

        def infer_chunk(self, **kwargs):
            return SimpleNamespace(
                session="session",
                predicted_latents=torch.zeros(1, 1, 2, 1, 1),
                raw_action_sequence=kwargs["raw_action_chunk"].detach().clone(),
                debug={"rollout_window_size": 3},
            )

    monkeypatch.setattr(
        cli_module,
        "decode_latent_video",
        lambda pipeline, latents, *, decode_device: np.zeros(
            (latents.shape[2], 2, 2, 3), dtype=np.float32
        ),
    )
    sample = SimpleNamespace(
        video_latents=torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1),
        actions=torch.arange(60, dtype=torch.float32).reshape(20, 3),
        task_text="task",
        text_context=None,
        negative_text_context=None,
        proprio_context_state=None,
        proprio_context_state_mask=None,
    )
    selection = FdmWindowSelection(
        sample_index=0,
        dataset_index=0,
        task_key="task",
        task_rank=0,
        episode_index=0,
        t0_frame=2,
        horizon_frames=4,
        generated_frames=4,
        context_start_frame=0,
        total_video_frames=10,
        repo_root="repo",
        target_start_offset_frames=1,
    )

    result = _run_one_selection_mode(
        fdm_rollout=FakeRollout(),
        selection=selection,
        sample=sample,
        mode=FdmAblationMode.VANILLA_JOINT_ROLLOUT,
        runtime_device=torch.device("cpu"),
        decode_device=torch.device("cpu"),
        seed=0,
        write_video=False,
        video_dir=tmp_path,
        video_fps=8.0,
    )

    assert captured_warmup["video_context"].shape[2] == 2
    torch.testing.assert_close(
        captured_warmup["action_context"], sample.actions[:4].unsqueeze(0)
    )
    assert result["summary"]["target_start_frame"] == 2
    assert result["summary"]["target_start_offset_frames"] == 0
    assert result["metric_rows"][0]["future_frame"] == 2


@pytest.mark.parametrize(
    "mode",
    [
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        FdmAblationMode.VIDEO_CONDITIONED_ACTION,
    ],
)
def test_m5_gjd_offline_rollout_seeds_per_chunk_proprio_history(
    mode: FdmAblationMode,
) -> None:
    from open_wam.configs import DynamicsObjective, ProprioContextMode
    from open_wam.pipelines import VariantRolloutRunner

    dual_expert_training_tests = _load_repo_script(
        "tests/test_dual_expert_generalist_training.py"
    )

    pipeline, _, _, text_context = (
        dual_expert_training_tests._build_tiny_generalist_pipeline(
            DynamicsObjective.JOINT,
            proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        )
    )
    object.__setattr__(
        pipeline.policy_variant.inference_config, "action_num_inference_steps", 25
    )
    rollout = DualExpertDynamicsRollout(VariantRolloutRunner(pipeline))
    video_context = torch.randn(1, 48, 2, 8, 8)
    action_context = torch.randn(1, 4, 4)
    hidden_proprio_history = torch.randn(1, 2, 4)
    current_proprio = torch.randn(1, 4)
    session = rollout.reset_and_warmup(
        task_text=("task",),
        video_context=video_context,
        action_context=action_context,
        text_context=text_context,
        negative_text_context=None,
        context_start_frame=0,
        mode=mode,
        proprio_state=current_proprio,
        hidden_proprio_history=hidden_proprio_history,
    )

    torch.testing.assert_close(
        session.policy_state.variant_state.past_hidden_proprio_states,
        hidden_proprio_history,
    )
    raw_action_chunk = torch.randn(1, rollout.action_per_frame, 4)
    video_condition = (
        torch.randn(1, 48, 1, 8, 8)
        if mode == FdmAblationMode.VIDEO_CONDITIONED_ACTION
        else None
    )

    output = rollout.infer_chunk(
        session=session,
        mode=mode,
        raw_action_chunk=raw_action_chunk,
        video_condition_latents=video_condition,
        proprio_state=current_proprio,
    )

    assert output.predicted_latents.shape == (1, 48, 1, 8, 8)
    assert (
        output.session.policy_state.variant_state.past_hidden_proprio_states is not None
    )
    assert (
        output.session.policy_state.variant_state.past_hidden_proprio_states.shape[1]
        >= 1
    )


def test_idm_rollout_threads_video_condition_and_drops_text() -> None:
    captured: dict[str, object] = {}
    reference_transformer = torch.nn.Linear(1, 1, bias=False)

    class FakeVisualTower:
        def get_runtime_backbone(self, *, action_dim: int):
            assert action_dim == 3
            return reference_transformer

    class FakeActionAdapter:
        def to_model_action_sequence(
            self,
            raw_action_chunk,
            *,
            action_space,
            device,
            dtype,
        ):
            del action_space
            return raw_action_chunk.to(device=device, dtype=dtype)

    class FakeRunner:
        def __init__(self) -> None:
            self.pipeline = SimpleNamespace(visual_tower=FakeVisualTower())
            self.policy_variant = SimpleNamespace(
                config=ParallelStreamPolicyConfig(
                    program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
                    hidden_size=32,
                ),
                inference_config=SimpleNamespace(frame_chunk_size=4),
                action_dim=3,
                exact_action_adapter=FakeActionAdapter(),
            )

        def infer_chunk(self, **kwargs):
            captured.update(kwargs)
            request = kwargs["dynamics"]
            return SimpleNamespace(
                session=kwargs["session"],
                predicted_latents=request.clean_video.detach().clone(),
                chunk_action_pred=torch.ones(1, 4, 3),
                raw_chunk_action_pred=torch.ones(1, 4, 3),
                debug={"rollout_window_size": 3},
            )

    rollout = ParallelStreamDynamicsRollout(FakeRunner())
    session = SimpleNamespace()
    video_condition_latents = torch.randn(1, 4, 1, 2, 2)

    output = rollout.infer_chunk(
        session=session,
        mode=FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        raw_action_chunk=torch.full((1, 4, 3), 7.0),
        video_condition_latents=video_condition_latents,
        drop_text_conditioning=True,
    )

    assert output.predicted_latents.shape == video_condition_latents.shape
    request = captured["dynamics"]
    assert request.objective == DynamicsObjective.VIDEO_CONDITIONED_ACTION
    assert torch.equal(request.clean_video, video_condition_latents)
    assert request.clean_action is None
    assert request.history_action is not None
    assert torch.all(request.history_action == 7.0)
    assert output.debug["drop_text_conditioning"] is True
    assert torch.all(output.raw_action_sequence == 1.0)

    rollout.infer_chunk(
        session=session,
        mode=FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        raw_action_chunk=None,
        video_condition_latents=video_condition_latents,
        allow_generated_action_commit=True,
    )
    assert captured["dynamics"].history_action is None


def test_latent_and_rgb_mse_per_frame() -> None:
    predicted_latents = torch.zeros(1, 2, 3, 2, 2)
    target_latents = torch.ones(1, 2, 3, 2, 2)
    assert latent_mse_per_frame(predicted_latents, target_latents) == [1.0, 1.0, 1.0]

    predicted_actions = torch.zeros(1, 6, 2)
    target_actions = torch.ones(1, 6, 2)
    assert action_mse_per_frame(
        predicted_actions, target_actions, action_per_frame=2
    ) == [1.0, 1.0, 1.0]

    predicted_rgb = np.zeros((2, 2, 2, 3), dtype=np.float32)
    target_rgb = np.ones((2, 2, 2, 3), dtype=np.float32)
    assert rgb_mse_per_frame(predicted_rgb, target_rgb) == [1.0, 1.0]


def test_summarize_metric_rows_groups_by_mode_and_horizon() -> None:
    selection = FdmWindowSelection(
        sample_index=0,
        dataset_index=0,
        task_key="task",
        task_rank=0,
        episode_index=0,
        t0_frame=10,
        horizon_frames=2,
        generated_frames=4,
        context_start_frame=0,
        total_video_frames=20,
        repo_root="repo",
    )
    rows = [
        {
            **selection.__dict__,
            "mode": FdmAblationMode.FORCED_ACTION_JOINT_FDM.value,
            "horizon_index": 0,
            "latent_mse": 1.0,
            "rgb_mse": 0.25,
        },
        {
            **selection.__dict__,
            "mode": FdmAblationMode.FORCED_ACTION_JOINT_FDM.value,
            "horizon_index": 0,
            "latent_mse": 3.0,
            "rgb_mse": 0.75,
        },
    ]
    summary = summarize_metric_rows(rows)
    assert summary == [
        {
            "mode": FdmAblationMode.FORCED_ACTION_JOINT_FDM.value,
            "horizon_index": 0,
            "count": 2,
            "latent_mse_mean": 2.0,
            "latent_mse_std": 1.0,
            "rgb_mse_mean": 0.5,
            "rgb_mse_std": 0.25,
        }
    ]


def test_idm_metric_rows_report_action_without_video_scores() -> None:
    selection = FdmWindowSelection(
        sample_index=0,
        dataset_index=0,
        task_key="task",
        task_rank=0,
        episode_index=0,
        t0_frame=10,
        horizon_frames=2,
        generated_frames=4,
        context_start_frame=0,
        total_video_frames=20,
        repo_root="repo",
    )

    rows = build_metric_rows(
        selection=selection,
        mode=FdmAblationMode.VIDEO_CONDITIONED_ACTION,
        action_mse=[0.25, 0.5],
    )
    summary = summarize_metric_rows(rows)

    assert rows[0]["latent_mse"] is None
    assert rows[0]["rgb_mse"] is None
    assert rows[0]["action_mse"] == 0.25
    assert summary[0]["latent_mse_mean"] is None
    assert summary[0]["action_mse_mean"] == 0.25
