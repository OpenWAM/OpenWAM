from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from open_wam.configs import ActionTargetRepresentation
from open_wam.models.action_decoders import ActionDecoder
from open_wam.models.policy_variants import (
    PolicyGenerationActionOrigin,
    PolicyRolloutContract,
)


class _RealtimeTestActionDecoder(ActionDecoder):
    def __init__(self, *, rollout_chunk_steps: int) -> None:
        super().__init__()
        self.rollout_chunk_steps = int(rollout_chunk_steps)

    def forward_train(self, policy_output, batch):
        raise NotImplementedError

    def forward_infer(self, policy_output, previous_state=None):
        raise NotImplementedError


class _ZeroOriginRolloutPolicy:
    """Minimal policy test double implementing the public rollout contract."""

    rollout_contract = PolicyRolloutContract(
        generation_action_origin=PolicyGenerationActionOrigin.ZERO,
    )

    @staticmethod
    def build_rollout_infer_extra(*, runtime_device):
        if runtime_device is None:
            return {}
        return {"action_device": str(runtime_device)}


def _load_sandbox_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_libero_realtime_sandbox.py"
    module_name = f"run_libero_realtime_sandbox_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
    return module


def test_realtime_helpers_use_source_visualization_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected_module = (repo_root / "src" / "open_wam" / "evals" / "libero_visualization.py").resolve()
    sandbox = _load_sandbox_module()

    assert Path(sandbox.exact_viz.__file__).resolve() == expected_module
    assert Path(sandbox.realtime_runtime.exact_viz.__file__).resolve() == expected_module


def _build_exact_history_frame_payload() -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], list[np.ndarray]]:
    current_obs = {"image": np.full((1, 1, 3), 9, dtype=np.uint8)}
    frame_obs_sequence = [
        {"image": np.full((1, 1, 3), 2 + index, dtype=np.uint8)}
        for index in range(4)
    ]
    frame_actions = [
        np.full((7,), float(index), dtype=np.float32)
        for index in range(4)
    ]
    return current_obs, frame_obs_sequence, frame_actions


def _strict_split_cache_dual_expert_config(
    sandbox,
    *,
    action_horizon: int = 16,
    frame_chunk_size: int = 4,
    program: str = "video_then_action",
):
    return SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="dual_expert",
            program=str(program),
        ),
        data=SimpleNamespace(
            sample_construction=SimpleNamespace(
                target_alignment="next_after_context",
                rollout_context_policy="one_frame",
            ),
            action_schema=SimpleNamespace(
                action_horizon=int(action_horizon),
                action_dim=7,
                state_horizon=1,
            ),
            action_target=SimpleNamespace(
                state_encoding="eef_pos_axisangle_gripper_2d",
                representation=ActionTargetRepresentation.RAW,
                rotation_representation="axis_angle",
                gripper_representation="action_command",
            ),
        ),
        inference=SimpleNamespace(
            frame_chunk_size=int(frame_chunk_size),
            action_num_inference_steps=20,
            video_num_inference_steps=20,
            guidance_scale=1.0,
            action_guidance_scale=1.0,
        ),
    )


def _minimal_obs_record(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "robot0_eef_pos": np.full(3, value, dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


def test_realtime_profiles_apply_long_libero_defaults_and_scheduler_defaults() -> None:
    sandbox = _load_sandbox_module()
    args = SimpleNamespace(
        eval_profile="libero_10hz_full",
        realtime_scheduler_profile="blocking_control",
        max_actions=80,
        env_horizon=None,
        target_action_hz=10.0,
        video_fps=None,
        deadline_miss_policy="hold_state",
        planner_mode="async_buffer",
        sequence_empty_plan_policy="fallback",
        fallback_history_policy="include_fallback_history",
        startup_open_loop_chunks=0,
        replan_low_watermark_actions=0,
    )

    sandbox._apply_realtime_cli_profiles(args, [])

    assert args.max_actions == 3000
    assert args.env_horizon == 5000
    assert args.video_fps is None
    assert args.realtime_scheduler_profile is sandbox.RealtimeSchedulerProfile.BLOCKING_CONTROL
    assert args.planner_mode is sandbox.RealtimePlannerMode.HISTORY_ONLY
    assert args.sequence_empty_plan_policy is sandbox.RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN
    assert args.fallback_history_policy is sandbox.FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY


def test_realtime_profiles_preserve_explicit_low_level_overrides() -> None:
    sandbox = _load_sandbox_module()
    args = SimpleNamespace(
        eval_profile="libero_10hz_full",
        realtime_scheduler_profile="async_history_first",
        max_actions=120,
        env_horizon=None,
        target_action_hz=10.0,
        video_fps=None,
        deadline_miss_policy="hold_state",
        planner_mode="history_only",
        sequence_empty_plan_policy="fallback",
        fallback_history_policy="include_fallback_history",
        startup_open_loop_chunks=0,
        replan_low_watermark_actions=7,
    )

    sandbox._apply_realtime_cli_profiles(
        args,
        ["--max-actions", "120", "--planner-mode", "history_only", "--replan-low-watermark-actions", "7"],
    )

    assert args.max_actions == 120
    assert args.realtime_scheduler_profile is sandbox.RealtimeSchedulerProfile.ASYNC_HISTORY_FIRST
    assert args.planner_mode is sandbox.RealtimePlannerMode.HISTORY_ONLY
    assert args.replan_low_watermark_actions == 7
    assert args.startup_open_loop_chunks == 1


def test_exact_startup_bootstrap_padding_defaults_to_single_init() -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        data=SimpleNamespace(sample_construction=SimpleNamespace(start_padding_frames=0)),
    )

    assert sandbox._resolve_exact_startup_bootstrap_padding(config, cli_value=None, checkpoint_path=None) is False
    with pytest.raises(ValueError, match="deprecated"):
        sandbox._resolve_exact_startup_bootstrap_padding(config, cli_value=True, checkpoint_path=None)

    config.data.sample_construction.start_padding_frames = 3

    assert sandbox._resolve_exact_startup_bootstrap_padding(config, cli_value=None, checkpoint_path=None) is False
    assert sandbox._resolve_exact_startup_bootstrap_padding(config, cli_value=False, checkpoint_path=None) is False


def test_exact_startup_bootstrap_padding_ignores_checkpoint_training_marker_by_default(tmp_path: Path) -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        data=SimpleNamespace(sample_construction=SimpleNamespace(start_padding_frames=3)),
    )
    checkpoint_dir = tmp_path / "checkpoint_step_400"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "model_state.pt"
    checkpoint_file.write_bytes(b"stub")
    resolved_config = checkpoint_dir / "resolved_config.yaml"
    resolved_config.write_text(
        "data:\n"
        "  sample_construction:\n"
        "    mode: full_segment\n",
        encoding="utf-8",
    )

    assert sandbox._resolve_exact_startup_bootstrap_padding(
        config,
        cli_value=None,
        checkpoint_path=checkpoint_file,
    ) is False
    with pytest.raises(ValueError, match="deprecated"):
        sandbox._resolve_exact_startup_bootstrap_padding(
            config,
            cli_value=True,
            checkpoint_path=checkpoint_file,
        )

    resolved_config.write_text(
        "data:\n"
        "  sample_construction:\n"
        "    start_padding_frames: 3\n",
        encoding="utf-8",
    )

    assert sandbox._resolve_exact_startup_bootstrap_padding(
        config,
        cli_value=None,
        checkpoint_path=checkpoint_file,
    ) is False


def test_exact_startup_bootstrap_padding_defaults_to_single_init_for_unknown_checkpoint(
    tmp_path: Path,
) -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        data=SimpleNamespace(sample_construction=SimpleNamespace(start_padding_frames=3)),
    )
    checkpoint_file = tmp_path / "model_state.pt"
    checkpoint_file.write_bytes(b"stub")

    assert sandbox._resolve_exact_startup_bootstrap_padding(
        config,
        cli_value=None,
        checkpoint_path=checkpoint_file,
    ) is False
    with pytest.raises(ValueError, match="deprecated"):
        sandbox._resolve_exact_startup_bootstrap_padding(
            config,
            cli_value=True,
            checkpoint_path=checkpoint_file,
        )


def test_exact_startup_bootstrap_padding_transformer_export_defaults_to_single_init(tmp_path: Path) -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        data=SimpleNamespace(sample_construction=SimpleNamespace(start_padding_frames=3)),
    )
    transformer_dir = tmp_path / "checkpoint_step_2000" / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / "config.json").write_text("{}", encoding="utf-8")
    checkpoint_dir = transformer_dir.parent

    assert sandbox._resolve_exact_startup_bootstrap_padding(
        config,
        cli_value=None,
        checkpoint_path=checkpoint_dir,
    ) is False

    (checkpoint_dir / "resolved_config.yaml").write_text(
        "data:\n  sample_construction:\n    start_padding_frames: 3\n",
        encoding="utf-8",
    )

    assert sandbox._resolve_exact_startup_bootstrap_padding(
        config,
        cli_value=None,
        checkpoint_path=checkpoint_dir,
    ) is False
    with pytest.raises(ValueError, match="deprecated"):
        sandbox._resolve_exact_startup_bootstrap_padding(
            config,
            cli_value=True,
            checkpoint_path=checkpoint_dir,
        )


def test_finalize_rollout_outputs_lean_skips_videos_and_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _load_sandbox_module()

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("lean artifact profile should not render videos or trace files")

    monkeypatch.setattr(
        sandbox.rollout_artifacts,
        "build_libero_realtime_video_frames",
        fail_if_called,
    )
    monkeypatch.setattr(
        sandbox.rollout_artifacts,
        "build_libero_fallback_timeline_video_frames",
        fail_if_called,
    )
    monkeypatch.setattr(sandbox.rollout_artifacts.imageio, "mimsave", fail_if_called)

    summary = sandbox._finalize_rollout_outputs(
        summary={"target_action_hz": 10.0},
        action_records=[{"action_index": 0}],
        action_video_records=[],
        replan_records=[{"event": "replan"}],
        extension_records=[{"event": "extension"}],
        component_report={"checkpoint_file": "/tmp/model_state.pt"},
        output_dir=tmp_path,
        benchmark="libero_10",
        task_id=0,
        prompt="pick up the black bowl",
        episode_idx=0,
        suffix="lean",
        video_fps=10.0,
        action_per_frame=1,
        write_fallback_timeline_video=False,
        artifact_profile="lean",
    )

    summary_path = Path(summary["summary_path"])
    assert summary["artifact_profile"] == "lean"
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["artifact_profile"] == "lean"
    assert "video_path" not in summary
    assert "action_trace_path" not in summary
    assert not list(tmp_path.rglob("*.mp4"))
    assert not list(tmp_path.rglob("*_actions.jsonl"))
    assert not list(tmp_path.rglob("*_replans.jsonl"))
    assert not list(tmp_path.rglob("*_extensions.jsonl"))
    assert not list(tmp_path.rglob("*_load_report.json"))


def test_exact_fallback_history_policy_freeze_until_clean_chunk_delays_full_clean_chunk() -> None:
    sandbox = _load_sandbox_module()
    pending_history = []
    state = sandbox.realtime_history.FrameFallbackHistoryState(
        policy=sandbox.FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
    )
    current_obs, frame_obs_sequence, frame_actions = _build_exact_history_frame_payload()

    assert (
        sandbox.realtime_history.append_frame_history_record(
            pending_history=pending_history,
            state=state,
            absolute_frame_index=4,
            current_obs=current_obs,
            frame_obs_sequence=frame_obs_sequence,
            frame_actions=frame_actions,
            frame_action_sources=["fallback_hold_last"] * 4,
            frame_chunk_size=4,
        )
        == "fallback"
    )
    assert state.quarantine_active

    for absolute_frame_index in range(5, 9):
        assert (
            sandbox.realtime_history.append_frame_history_record(
                pending_history=pending_history,
                state=state,
                absolute_frame_index=absolute_frame_index,
                current_obs=current_obs,
                frame_obs_sequence=frame_obs_sequence,
                frame_actions=frame_actions,
                frame_action_sources=["open_loop_extension"] * 4,
                frame_chunk_size=4,
            )
            == "washout"
        )

    assert not state.quarantine_active
    assert [record["absolute_frame_index"] for record in pending_history] == [5, 6, 7, 8]
    assert state.hidden_fallback_frames == 1
    assert state.hidden_washout_frames == 4

    assert (
        sandbox.realtime_history.append_frame_history_record(
            pending_history=pending_history,
            state=state,
            absolute_frame_index=9,
            current_obs=current_obs,
            frame_obs_sequence=frame_obs_sequence,
            frame_actions=frame_actions,
            frame_action_sources=["history_replan"] * 4,
            frame_chunk_size=4,
        )
        == "included"
    )
    assert [record["absolute_frame_index"] for record in pending_history] == [5, 6, 7, 8, 9]


def test_fallback_model_timeline_advancement_is_policy_driven() -> None:
    sandbox = _load_sandbox_module()

    assert sandbox.realtime_history.action_advances_model_timeline(
        "history_replan",
        fallback_history_policy=sandbox.FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
    )
    assert not sandbox.realtime_history.action_advances_model_timeline(
        "fallback_hold_state",
        fallback_history_policy=sandbox.FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
    )
    assert sandbox.realtime_history.action_advances_model_timeline(
        "fallback_hold_state",
        fallback_history_policy=sandbox.FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY,
    )


def test_sequence_buffer_tail_promotes_after_prebuffer_actions_are_consumed() -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="dual_expert", program="video_then_action"),
        data=SimpleNamespace(
            action_schema=SimpleNamespace(action_horizon=16),
        ),
        inference=SimpleNamespace(frame_chunk_size=4),
    )

    assert not sandbox.realtime_runtime.sequence_buffer_tail_ready_for_history_promotion(
        config=config,
        next_action_index=27,
        buffer_tail_generation_action_start=32,
        history_generation_action_start=16,
    )
    assert sandbox.realtime_runtime.sequence_buffer_tail_ready_for_history_promotion(
        config=config,
        next_action_index=28,
        buffer_tail_generation_action_start=32,
        history_generation_action_start=16,
    )
    assert not sandbox.realtime_runtime.sequence_buffer_tail_ready_for_history_promotion(
        config=config,
        next_action_index=28,
        buffer_tail_generation_action_start=32,
        history_generation_action_start=32,
    )


def test_dual_expert_async_history_submit_rewinds_speculative_action_tail_only() -> None:
    sandbox = _load_sandbox_module()
    dual_expert_config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="dual_expert", program="video_then_action"),
    )
    non_dual_expert_config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="parallel_stream", runtime_mode="lingbot_exact"),
    )

    assert (
        sandbox.realtime_runtime.resolve_sequence_action_cache_rewind_frame(
            config=dual_expert_config,
            planner_mode="async_history_first",
            use_observation_update=True,
            condition_frame_start=8,
        )
        == 8
    )
    assert (
        sandbox.realtime_runtime.resolve_sequence_action_cache_rewind_frame(
            config=dual_expert_config,
            planner_mode="history_only",
            use_observation_update=True,
            condition_frame_start=8,
        )
        is None
    )
    assert (
        sandbox.realtime_runtime.resolve_sequence_action_cache_rewind_frame(
            config=dual_expert_config,
            planner_mode="async_history_first",
            use_observation_update=False,
            condition_frame_start=8,
        )
        is None
    )
    assert (
        sandbox.realtime_runtime.resolve_sequence_action_cache_rewind_frame(
            config=non_dual_expert_config,
            planner_mode="async_history_first",
            use_observation_update=True,
            condition_frame_start=8,
        )
        is None
    )


def test_sequence_fallback_history_freezes_until_full_clean_action_chunk() -> None:
    sandbox = _load_sandbox_module()
    model_obs_window = [{"image": np.array([0], dtype=np.uint8)}]
    state = sandbox.realtime_history.ActionFallbackHistoryState(
        policy=sandbox.FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK,
    )

    assert (
        sandbox.realtime_history.append_action_observation(
            model_obs_window=model_obs_window,
            state=state,
            current_obs={"image": np.array([1], dtype=np.uint8)},
            action_source="fallback_hold_state",
            clean_actions_required=3,
            max_window_frames=2,
        )
        == "fallback"
    )
    assert [int(obs["image"][0]) for obs in model_obs_window] == [0]
    assert state.quarantine_active

    for value in (2, 3):
        assert (
            sandbox.realtime_history.append_action_observation(
                model_obs_window=model_obs_window,
                state=state,
                current_obs={"image": np.array([value], dtype=np.uint8)},
                action_source="history_replan",
                clean_actions_required=3,
                max_window_frames=2,
            )
            == "washout"
        )
        assert [int(obs["image"][0]) for obs in model_obs_window] == [0]

    assert (
        sandbox.realtime_history.append_action_observation(
            model_obs_window=model_obs_window,
            state=state,
            current_obs={"image": np.array([4], dtype=np.uint8)},
            action_source="history_replan",
            clean_actions_required=3,
            max_window_frames=2,
        )
        == "washout"
    )

    assert not state.quarantine_active
    assert [int(obs["image"][0]) for obs in model_obs_window] == [3, 4]
    assert state.hidden_fallback_actions == 1
    assert state.hidden_washout_actions == 3


def test_exact_realtime_job_seed_tracks_session_step_index() -> None:
    sandbox = _load_sandbox_module()
    session = SimpleNamespace(policy_state=SimpleNamespace(step_index=4))

    assert sandbox.realtime_runtime.job_seed_for_session(10, session) == 14
    assert sandbox.realtime_runtime.job_seed_for_session(None, session) is None


def test_exact_realtime_job_seed_uses_isolated_torch_rng() -> None:
    sandbox = _load_sandbox_module()
    torch = sandbox.torch
    original_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(1234)
        before_state = torch.random.get_rng_state()
        with sandbox.realtime_runtime.isolated_torch_rng(99, torch.device("cpu")):
            first = torch.rand(3)
        with sandbox.realtime_runtime.isolated_torch_rng(99, torch.device("cpu")):
            second = torch.rand(3)

        assert torch.equal(torch.random.get_rng_state(), before_state)
        assert torch.allclose(first, second)
    finally:
        torch.random.set_rng_state(original_state)


def test_exact_planner_no_submit_does_not_snapshot_runtime_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _load_sandbox_module()

    def fail_snapshot(**_: object) -> None:
        raise AssertionError("cache snapshot should not be taken when no planner job can be submitted")

    monkeypatch.setattr(
        sandbox.realtime_speculation,
        "snapshot_visual_runtime",
        fail_snapshot,
    )

    future, snapshot = sandbox.realtime_runtime.submit_planner_job_with_snapshot(
        executor=None,
        planner_mode="history_only",
        pending_history=[],
        future_buffer_depth=0,
        runner=None,
        history_base_session=None,
        current_chunk_session=None,
        prompt="task",
        config=None,
        frontend_device=sandbox.torch.device("cpu"),
        runtime_device=sandbox.torch.device("cpu"),
        buffer_tail_session=None,
        seed_base=0,
    )

    assert future is None
    assert snapshot is None


def test_exact_fallback_hold_state_zeroes_delta_channels_and_preserves_gripper() -> None:
    sandbox = _load_sandbox_module()
    last_action = np.array([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, -1.0], dtype=np.float32)

    fallback = sandbox.realtime_runtime.build_fallback_frame_actions(
        action_dim=7,
        action_per_frame=2,
        policy="hold_state",
        last_action=last_action,
    )

    np.testing.assert_allclose(
        fallback,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_exact_fallback_hold_state_zeroes_action_command_tail_when_configured() -> None:
    sandbox = _load_sandbox_module()
    last_action = np.array([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, -1.0], dtype=np.float32)

    fallback = sandbox.realtime_runtime.build_fallback_frame_actions(
        action_dim=7,
        action_per_frame=2,
        policy="hold_state",
        last_action=last_action,
        preserve_absolute_tail_from=None,
    )

    np.testing.assert_allclose(fallback, np.zeros((2, 7), dtype=np.float32))


def test_fallback_absolute_tail_start_skips_action_command_gripper() -> None:
    sandbox = _load_sandbox_module()

    assert (
        sandbox.realtime_history.fallback_absolute_tail_start(
            SimpleNamespace(
                data=SimpleNamespace(
                    action_target=SimpleNamespace(
                        include_gripper=True,
                        gripper_representation="action_command",
                    )
                )
            )
        )
        is None
    )
    assert (
        sandbox.realtime_history.fallback_absolute_tail_start(
            SimpleNamespace(
                data=SimpleNamespace(
                    action_target=SimpleNamespace(
                        include_gripper=True,
                        gripper_representation="first_channel",
                    )
                )
            )
        )
        == 6
    )


def test_exact_runtime_cache_snapshot_restores_rejected_speculation() -> None:
    sandbox = _load_sandbox_module()
    snapshot = sandbox.VisualRuntimeStateSnapshot(
        frontend_state=[sandbox.torch.tensor([2.0])],
    )

    class FakeVisualTower:
        def __init__(self) -> None:
            self.state = [sandbox.torch.tensor([9.0])]

        def snapshot_runtime_state(self, *, cache_name=None):
            assert cache_name == "open_wam_exact"
            return snapshot

        def restore_runtime_state(self, restored_snapshot) -> None:
            assert restored_snapshot is snapshot
            self.state = [value.clone() for value in restored_snapshot.frontend_state]

    visual_tower = FakeVisualTower()
    runner = SimpleNamespace(
        pipeline=SimpleNamespace(visual_tower=visual_tower)
    )
    session = SimpleNamespace(
        policy_state=SimpleNamespace(cache={"cache_name": "open_wam_exact"}),
    )

    captured = sandbox.realtime_speculation.snapshot_visual_runtime(
        runner=runner,
        config=SimpleNamespace(),
        session=session,
    )

    sandbox.realtime_speculation.restore_visual_runtime_if_rejected(
        {"trace": {"accepted_chunk": False}},
        runner=runner,
        snapshot=captured,
    )

    assert float(visual_tower.state[0][0]) == 2.0


def test_exact_fallback_hold_last_repeats_full_raw_action() -> None:
    sandbox = _load_sandbox_module()
    last_action = np.array([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, -1.0], dtype=np.float32)

    fallback = sandbox.realtime_runtime.build_fallback_frame_actions(
        action_dim=7,
        action_per_frame=2,
        policy="hold_last",
        last_action=last_action,
    )

    np.testing.assert_allclose(fallback, np.repeat(last_action[None, :], 2, axis=0))


def test_exact_startup_sessions_reject_legacy_frame_zero_generation() -> None:
    sandbox = _load_sandbox_module()
    startup_session = SimpleNamespace(policy_state=SimpleNamespace(step_index=0))
    session = SimpleNamespace(
        policy_state=SimpleNamespace(
            step_index=1,
            cache={"frame_start": 0},
            cursor=SimpleNamespace(block_index=0),
            decoder_state="decoder",
        ),
        task_text=("task",),
        text_context="text",
        negative_text_context="negative",
    )
    first_chunk = SimpleNamespace(
        session=session,
        debug={"generation_frame_start": 0},
    )

    with pytest.raises(ValueError, match="generation_frame_start < 1"):
        sandbox.realtime_runtime.resolve_exact_startup_sessions(
            config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact")),
            startup_session=startup_session,
            first_chunk=first_chunk,
            frame_chunk_size=4,
        )


def test_exact_startup_sessions_strict_first_chunk_advances_tail_from_frame_one() -> None:
    sandbox = _load_sandbox_module()
    startup_session = SimpleNamespace(policy_state=SimpleNamespace(step_index=0))
    session = SimpleNamespace(
        policy_state=SimpleNamespace(
            step_index=1,
            cache={"frame_start": 1},
            cursor=SimpleNamespace(block_index=0),
            decoder_state="decoder",
        ),
        task_text=("task",),
        text_context="text",
        negative_text_context="negative",
    )
    first_chunk = SimpleNamespace(
        session=session,
        debug={"generation_frame_start": 1},
    )

    _, _, buffer_tail_session = sandbox.realtime_runtime.resolve_exact_startup_sessions(
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact")),
        startup_session=startup_session,
        first_chunk=first_chunk,
        frame_chunk_size=4,
    )

    assert buffer_tail_session.policy_state.cache["frame_start"] == 5
    assert buffer_tail_session.policy_state.cursor.current_start_frame == 5


def test_exact_action_conditioned_startup_sessions_keep_reset_history_base() -> None:
    sandbox = _load_sandbox_module()
    startup_session = SimpleNamespace(policy_state=SimpleNamespace(step_index=0))
    chunk_session = SimpleNamespace(
        policy_state=SimpleNamespace(
            step_index=1,
            cache={"frame_start": 1},
            cursor=SimpleNamespace(block_index=0),
            decoder_state="decoder",
        ),
        task_text=("task",),
        text_context="text",
        negative_text_context="negative",
    )
    first_chunk = SimpleNamespace(
        session=chunk_session,
        debug={"generation_frame_start": 1},
    )

    history_session, current_session, buffer_tail_session = sandbox.realtime_runtime.resolve_exact_startup_sessions(
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact_action_conditioned")),
        startup_session=startup_session,
        first_chunk=first_chunk,
        frame_chunk_size=4,
    )

    assert history_session is startup_session
    assert current_session is chunk_session
    assert buffer_tail_session.policy_state.step_index == 1


def test_exact_chunk_to_planned_steps_rejects_legacy_frame_zero_generation() -> None:
    sandbox = _load_sandbox_module()
    raw_actions = sandbox.torch.arange(16, dtype=sandbox.torch.float32).view(1, 16, 1)
    chunk = SimpleNamespace(
        raw_chunk_action_pred=raw_actions,
        debug={"generation_frame_start": 0},
        session=SimpleNamespace(policy_state=SimpleNamespace(step_index=3)),
    )

    with pytest.raises(ValueError, match="generation_frame_start < 1"):
        sandbox.realtime_runtime.exact_chunk_to_planned_steps(
            chunk=chunk,
            action_per_frame=4,
            frame_chunk_size=4,
            source="startup_plan",
            ready_monotonic_s=1.5,
        )


def test_exact_chunk_to_planned_steps_strict_frame_one_starts_at_action_zero() -> None:
    sandbox = _load_sandbox_module()
    raw_actions = sandbox.torch.arange(16, dtype=sandbox.torch.float32).view(1, 16, 1)
    chunk = SimpleNamespace(
        raw_chunk_action_pred=raw_actions,
        debug={"generation_frame_start": 1},
        session=SimpleNamespace(policy_state=SimpleNamespace(step_index=3)),
    )

    planned = sandbox.realtime_runtime.exact_chunk_to_planned_steps(
        chunk=chunk,
        action_per_frame=4,
        frame_chunk_size=4,
        source="startup_plan",
        ready_monotonic_s=1.5,
    )

    assert len(planned) == 16
    assert [step.absolute_action_index for step in planned] == list(range(16))
    assert all(step.generation_frame_start == 1 for step in planned)
    np.testing.assert_allclose(planned[0].raw_action, np.array([0.0], dtype=np.float32))


def test_exact_startup_conditioning_history_rejects_legacy_frame_zero_generation() -> None:
    sandbox = _load_sandbox_module()
    raw_actions = sandbox.torch.arange(16, dtype=sandbox.torch.float32).view(1, 16, 1)
    initial_latents = sandbox.torch.ones(1, 2, 1, 1, 1)
    chunk = SimpleNamespace(
        raw_chunk_action_pred=raw_actions,
        debug={"generation_frame_start": 0},
    )

    with pytest.raises(ValueError, match="generation_frame_start < 1"):
        sandbox.realtime_runtime.build_exact_startup_conditioning_history_record(
            chunk=chunk,
            initial_video_latents=initial_latents,
            initial_obs={"image": np.zeros((2, 2, 3), dtype=np.uint8)},
            action_per_frame=4,
            frame_chunk_size=4,
        )


def test_exact_startup_conditioning_history_omits_override_actions_for_frame_zero_prefix() -> None:
    sandbox = _load_sandbox_module()
    raw_actions = sandbox.torch.arange(16, dtype=sandbox.torch.float32).view(1, 16, 1)
    initial_latents = sandbox.torch.ones(1, 2, 1, 1, 1)
    chunk = SimpleNamespace(
        raw_chunk_action_pred=raw_actions,
        debug={"generation_frame_start": 1},
    )

    record = sandbox.realtime_runtime.build_exact_startup_conditioning_history_record(
        chunk=chunk,
        initial_video_latents=initial_latents,
        initial_obs={"image": np.zeros((2, 2, 3), dtype=np.uint8)},
        action_per_frame=4,
        frame_chunk_size=4,
        conditioning_frame_index=0,
        raw_actions_override=np.zeros((4, 1), dtype=np.float32),
    )

    assert record["absolute_frame_index"] == 0
    assert record["raw_actions_valid"] is False
    assert record["raw_action_dim"] == 1
    assert record["raw_actions"].shape == (0, 1)


def test_exact_startup_conditioning_history_omits_actions_for_frame_zero_prefix() -> None:
    sandbox = _load_sandbox_module()
    raw_actions = sandbox.torch.arange(16, dtype=sandbox.torch.float32).view(1, 16, 1)
    initial_latents = sandbox.torch.ones(1, 2, 1, 1, 1)
    chunk = SimpleNamespace(
        raw_chunk_action_pred=raw_actions,
        debug={"generation_frame_start": 1},
    )

    record = sandbox.realtime_runtime.build_exact_startup_conditioning_history_record(
        chunk=chunk,
        initial_video_latents=initial_latents,
        initial_obs={"image": np.zeros((2, 2, 3), dtype=np.uint8)},
        action_per_frame=4,
        frame_chunk_size=4,
        conditioning_frame_index=0,
    )

    assert record["absolute_frame_index"] == 0
    assert record["raw_actions_valid"] is False
    assert record["raw_action_dim"] == 1
    assert record["raw_actions"].shape == (0, 1)


def test_exact_history_worker_copy_preserves_raw_obs_sequences_and_latents() -> None:
    sandbox = _load_sandbox_module()
    record = {
        "absolute_frame_index": 3,
        "obs": {"image": np.ones((1, 1, 3), dtype=np.uint8)},
        "obs_sequence": [
            {"image": np.full((1, 1, 3), 2, dtype=np.uint8)},
            {"image": np.full((1, 1, 3), 3, dtype=np.uint8)},
        ],
        "raw_actions": np.ones((4, 7), dtype=np.float32),
        "video_latents": sandbox.torch.ones(1, 2, 1, 1, 1),
    }

    copied = sandbox.realtime_runtime.copy_history_record_for_worker(record)
    views = sandbox.realtime_runtime._history_records_to_obs_sequence([copied])
    raw_count = sandbox.realtime_runtime._count_history_raw_observations([copied])
    latents = sandbox.realtime_runtime._history_records_to_precomputed_video_latents([copied])

    assert len(views) == 2
    assert raw_count == 2
    assert latents is not None
    assert tuple(latents.shape) == (1, 2, 1, 1, 1)
    copied["obs_sequence"][0]["image"][0, 0, 0] = 99
    assert record["obs_sequence"][0]["image"][0, 0, 0] == 2


def test_exact_history_obs_sequence_expands_raw_observation_sequences() -> None:
    sandbox = _load_sandbox_module()
    record = {
        "absolute_frame_index": 3,
        "obs": {"image": np.full((1, 1, 3), 7, dtype=np.uint8)},
        "obs_sequence": [
            {"image": np.full((1, 1, 3), 2, dtype=np.uint8)},
            {"image": np.full((1, 1, 3), 3, dtype=np.uint8)},
        ],
        "raw_actions": np.ones((4, 7), dtype=np.float32),
    }

    views = sandbox.realtime_runtime._history_records_to_obs_sequence([record])
    raw_count = sandbox.realtime_runtime._count_history_raw_observations([record])

    assert len(views) == 2
    np.testing.assert_array_equal(views[0]["image"], record["obs_sequence"][0]["image"])
    np.testing.assert_array_equal(views[1]["image"], record["obs_sequence"][1]["image"])
    assert raw_count == 2


def test_exact_history_action_history_skips_invalid_startup_rows() -> None:
    sandbox = _load_sandbox_module()
    history = [
        {
            "absolute_frame_index": 0,
            "obs": {},
            "raw_actions": np.zeros((0, 7), dtype=np.float32),
            "raw_actions_valid": False,
            "raw_action_dim": 7,
        },
        {
            "absolute_frame_index": 1,
            "obs": {},
            "raw_actions": np.ones((4, 7), dtype=np.float32),
        },
    ]

    action_history = sandbox.realtime_runtime._history_records_to_action_history(
        history,
        config=SimpleNamespace(data=SimpleNamespace(action_schema=SimpleNamespace(action_dim=7))),
    )

    assert action_history.shape == (4, 7)
    np.testing.assert_allclose(action_history, np.ones((4, 7), dtype=np.float32))


def test_exact_history_records_feed_wan_aligned_streaming_chunks() -> None:
    from open_wam.models.common.video_geometry import wan_safe_temporal_frame_count

    sandbox = _load_sandbox_module()
    pending_history = []
    state = sandbox.realtime_history.FrameFallbackHistoryState(
        policy=sandbox.FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY,
    )
    current_obs, frame_obs_sequence, frame_actions = _build_exact_history_frame_payload()

    decision = sandbox.realtime_history.append_frame_history_record(
        pending_history=pending_history,
        state=state,
        absolute_frame_index=4,
        current_obs=current_obs,
        frame_obs_sequence=frame_obs_sequence,
        frame_actions=frame_actions,
        frame_action_sources=["history_replan"] * 4,
        frame_chunk_size=4,
    )

    assert decision == "included"
    assert len(pending_history) == 1
    assert len(pending_history[0]["obs_sequence"]) == 4
    assert pending_history[0]["raw_actions"].shape == (4, 7)
    assert wan_safe_temporal_frame_count(4, cache_initialized=True) == 4

    views = sandbox.realtime_runtime._history_records_to_obs_sequence(pending_history)
    raw_count = sandbox.realtime_runtime._count_history_raw_observations(pending_history)

    assert len(views) == 4
    assert raw_count == 4


def test_exact_future_result_drops_partial_stale_chunks() -> None:
    sandbox = _load_sandbox_module()
    frame_cls = sandbox.realtime_runtime.PlannedFrameAction
    trace = {"job_kind": "open_loop_extension"}
    result = sandbox.realtime_runtime.FramePlannerJobResult(
        job_kind="open_loop_extension",
        planned_frames=[
            frame_cls(
                absolute_frame_index=2,
                generation_frame_start=2,
                frame_offset=0,
                raw_actions=np.ones((4, 7), dtype=np.float32),
                source="open_loop_extension",
            ),
            frame_cls(
                absolute_frame_index=3,
                generation_frame_start=2,
                frame_offset=1,
                raw_actions=np.ones((4, 7), dtype=np.float32) * 2,
                source="open_loop_extension",
            ),
        ],
        buffer_tail_session="tail",
        trace=trace,
        submitted_through_frame=None,
    )

    application = sandbox.realtime_runtime.apply_frame_planner_result(
        result,
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact")),
        plan_by_action={},
        next_action_to_execute=6,
        pending_history=[],
        history_base_session="history",
        current_chunk_session="current",
        buffer_tail_session=None,
        replan_records=[],
        extension_records=[],
    )

    assert application.plan_by_action == {}
    assert application.buffer_tail_session is None
    assert trace["stale_planned_actions"] == 8
    assert trace["future_planned_actions"] == 0
    assert trace["chunk_boundary_dropped_actions"] == 6
    assert not trace["accepted_chunk"]
    assert trace["planned_action_indices"] == list(range(4, 12))


def test_exact_future_result_keeps_full_future_chunks() -> None:
    sandbox = _load_sandbox_module()
    frame_cls = sandbox.realtime_runtime.PlannedFrameAction
    trace = {"job_kind": "open_loop_extension"}
    result = sandbox.realtime_runtime.FramePlannerJobResult(
        job_kind="open_loop_extension",
        planned_frames=[
            frame_cls(
                absolute_frame_index=2,
                generation_frame_start=2,
                frame_offset=0,
                raw_actions=np.ones((4, 7), dtype=np.float32),
                source="open_loop_extension",
            ),
            frame_cls(
                absolute_frame_index=3,
                generation_frame_start=2,
                frame_offset=1,
                raw_actions=np.ones((4, 7), dtype=np.float32) * 2,
                source="open_loop_extension",
            ),
        ],
        buffer_tail_session="tail",
        trace=trace,
        submitted_through_frame=None,
    )

    application = sandbox.realtime_runtime.apply_frame_planner_result(
        result,
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact")),
        plan_by_action={},
        next_action_to_execute=4,
        pending_history=[],
        history_base_session="history",
        current_chunk_session="current",
        buffer_tail_session=None,
        replan_records=[],
        extension_records=[],
    )

    assert sorted(application.plan_by_action) == list(range(4, 12))
    assert application.buffer_tail_session == "tail"
    assert trace["stale_planned_actions"] == 0
    assert trace["future_planned_actions"] == 8
    assert trace["chunk_boundary_dropped_actions"] == 0
    assert trace["accepted_chunk"]
    assert trace["planned_action_indices"] == list(range(4, 12))


def test_exact_history_replan_anchors_planned_frames_to_next_observed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = _load_sandbox_module()

    warmup_session = SimpleNamespace(
        policy_state=SimpleNamespace(
            step_index=1,
            cache={"frame_start": 13},
            cursor=SimpleNamespace(block_index=0),
            decoder_state="decoder",
        ),
        task_text=("task",),
        text_context="text",
        negative_text_context="negative",
    )
    chunk_session = SimpleNamespace(
        policy_state=SimpleNamespace(
            step_index=2,
            cache={"frame_start": 13},
            cursor=SimpleNamespace(block_index=1),
            decoder_state="decoder",
        ),
        task_text=("task",),
        text_context="text",
        negative_text_context="negative",
    )

    class _FakeRunner:
        def warmup_cache(self, **kwargs):
            assert kwargs["frame_start_override"] == 0
            assert tuple(kwargs["action_history"].shape) == (1, 4, 7)
            return SimpleNamespace(session=warmup_session)

        def infer_chunk(self, **kwargs):
            return SimpleNamespace(
                session=chunk_session,
                raw_chunk_action_pred=sandbox.torch.arange(16, dtype=sandbox.torch.float32).view(1, 16, 1),
                debug={"generation_frame_start": 13},
            )

    monkeypatch.setattr(
        sandbox.realtime_runtime,
        "_prepare_history_runtime_inputs",
        lambda *args, **kwargs: {
            "video_latents": sandbox.torch.zeros(1, 48, 13, 1, 1),
            "text_context": "text",
            "negative_text_context": "negative",
        },
    )
    monkeypatch.setattr(sandbox.realtime_runtime, "synchronize_devices", lambda *args, **kwargs: None)

    result = sandbox.realtime_runtime.run_replan_job(
        runner=_FakeRunner(),
        session=SimpleNamespace(
            policy_state=SimpleNamespace(step_index=1, cache={"frame_start": 0}),
            text_context="text",
            negative_text_context="negative",
        ),
        prompt="task",
        history_records=[
            {
                "absolute_frame_index": 0,
                "obs": {},
                "raw_actions": np.zeros((0, 7), dtype=np.float32),
                "raw_actions_valid": False,
                "raw_action_dim": 7,
            },
            {"absolute_frame_index": 3, "obs": {}, "raw_actions": np.zeros((4, 7), dtype=np.float32)},
        ],
        config=SimpleNamespace(
            inference=SimpleNamespace(frame_chunk_size=4),
            policy_variant=SimpleNamespace(action_per_frame=4),
        ),
        frontend_device=sandbox.torch.device("cpu"),
        runtime_device=sandbox.torch.device("cpu"),
        job_seed=None,
    )

    assert isinstance(result, sandbox.realtime_runtime.FramePlannerJobResult)
    assert [frame.absolute_frame_index for frame in result.planned_frames] == [4, 5, 6, 7]
    assert all(frame.generation_frame_start == 4 for frame in result.planned_frames)
    assert result.trace["history_frame_start"] == 0
    assert result.trace["generation_frame_start"] == 4
    assert result.trace["model_generation_frame_start"] == 13
    assert result.trace["session_frame_start_after_model"] == 13
    assert result.buffer_tail_session.policy_state.cache["frame_start"] == 8


def test_exact_history_replan_result_advances_base_session_to_chunk_session() -> None:
    sandbox = _load_sandbox_module()
    frame_cls = sandbox.realtime_runtime.PlannedFrameAction
    result = sandbox.realtime_runtime.FramePlannerJobResult(
        job_kind="history_replan",
        planned_frames=[
            frame_cls(
                absolute_frame_index=1,
                generation_frame_start=1,
                frame_offset=0,
                raw_actions=np.ones((4, 7), dtype=np.float32),
                source="history_replan",
            ),
        ],
        warmup_session="warmup",
        session="chunk",
        buffer_tail_session="tail",
        submitted_through_frame=3,
        trace={"job_kind": "history_replan"},
    )

    application = sandbox.realtime_runtime.apply_frame_planner_result(
        result,
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact")),
        plan_by_action={},
        next_action_to_execute=0,
        pending_history=[
            {"absolute_frame_index": 0},
            {"absolute_frame_index": 2},
            {"absolute_frame_index": 4},
        ],
        history_base_session="old_history",
        current_chunk_session="old_chunk",
        buffer_tail_session=None,
        replan_records=[],
        extension_records=[],
    )

    assert application.history_base_session == "chunk"
    assert application.current_chunk_session == "chunk"
    assert application.buffer_tail_session == "tail"
    assert sorted(application.plan_by_action) == [0, 1, 2, 3]
    assert application.pending_history == [{"absolute_frame_index": 4}]


def test_exact_rejected_history_replan_keeps_pending_history() -> None:
    sandbox = _load_sandbox_module()
    frame_cls = sandbox.realtime_runtime.PlannedFrameAction
    result = sandbox.realtime_runtime.FramePlannerJobResult(
        job_kind="history_replan",
        planned_frames=[
            frame_cls(
                absolute_frame_index=1,
                generation_frame_start=1,
                frame_offset=0,
                raw_actions=np.ones((4, 7), dtype=np.float32),
                source="history_replan",
            ),
            frame_cls(
                absolute_frame_index=2,
                generation_frame_start=1,
                frame_offset=1,
                raw_actions=np.ones((4, 7), dtype=np.float32),
                source="history_replan",
            ),
        ],
        warmup_session="warmup",
        session="chunk",
        buffer_tail_session="tail",
        submitted_through_frame=3,
        trace={"job_kind": "history_replan"},
    )
    pending = [
        {"absolute_frame_index": 0},
        {"absolute_frame_index": 2},
        {"absolute_frame_index": 4},
    ]

    application = sandbox.realtime_runtime.apply_frame_planner_result(
        result,
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact")),
        plan_by_action={},
        next_action_to_execute=6,
        pending_history=pending,
        history_base_session="old_history",
        current_chunk_session="old_chunk",
        buffer_tail_session=None,
        replan_records=[],
        extension_records=[],
    )

    assert application.history_base_session == "old_history"
    assert application.current_chunk_session == "old_chunk"
    assert application.buffer_tail_session is None
    assert application.plan_by_action == {}
    assert application.pending_history == pending
    assert not result.trace["accepted_chunk"]


def test_exact_action_conditioned_history_replan_keeps_warmup_session_base() -> None:
    sandbox = _load_sandbox_module()
    frame_cls = sandbox.realtime_runtime.PlannedFrameAction
    result = sandbox.realtime_runtime.FramePlannerJobResult(
        job_kind="history_replan",
        planned_frames=[
            frame_cls(
                absolute_frame_index=1,
                generation_frame_start=1,
                frame_offset=0,
                raw_actions=np.ones((4, 7), dtype=np.float32),
                source="history_replan",
            ),
        ],
        warmup_session="warmup",
        session="chunk",
        buffer_tail_session="tail",
        submitted_through_frame=1,
        trace={"job_kind": "history_replan"},
    )

    application = sandbox.realtime_runtime.apply_frame_planner_result(
        result,
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact_action_conditioned")),
        plan_by_action={},
        next_action_to_execute=0,
        pending_history=[],
        history_base_session="old_history",
        current_chunk_session="old_chunk",
        buffer_tail_session=None,
        replan_records=[],
        extension_records=[],
    )

    assert application.history_base_session == "warmup"
    assert application.current_chunk_session == "chunk"


def test_apply_sequence_replan_result_records_trace_and_merges_future_steps() -> None:
    sandbox = _load_sandbox_module()
    step_cls = sandbox.PlannedControlStep
    replan_records = []
    existing = {
        1: step_cls(absolute_action_index=1, generation_action_start=0, source="old"),
        2: step_cls(absolute_action_index=2, generation_action_start=0, source="old"),
    }
    result = sandbox.realtime_runtime.SequenceReplanJobResult(
        session="session",
        runtime_cache_snapshot=None,
        next_generation_action_start=5,
        planned_steps=[
            step_cls(absolute_action_index=0, generation_action_start=0, source="stale"),
            step_cls(absolute_action_index=2, generation_action_start=2, source="new"),
            step_cls(absolute_action_index=3, generation_action_start=2, source="new"),
        ],
        trace={"job_kind": "history_replan"},
    )

    session, next_start, merged = sandbox.realtime_runtime.apply_sequence_replan_result(
        result=result,
        replan_records=replan_records,
        plan_by_action=existing,
        next_action_to_execute=1,
    )

    assert session == "session"
    assert next_start == 5
    assert replan_records == [{"job_kind": "history_replan"}]
    assert list(merged) == [1, 2, 3]
    assert merged[1].source == "old"
    assert merged[2].source == "new"
    assert merged[3].source == "new"


def test_initial_generation_start_matches_warmup_window() -> None:
    sandbox = _load_sandbox_module()

    initial_obs_window = [{"image": np.zeros((2, 2, 3), dtype=np.uint8)} for _ in range(15)]

    assert (
        sandbox.rollout_runtime.resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=None,
        )
        == 15
    )


def test_rollout_runtime_can_override_initial_generation_start() -> None:
    sandbox = _load_sandbox_module()

    initial_obs_window = [{"image": np.zeros((2, 2, 3), dtype=np.uint8)} for _ in range(15)]

    assert (
        sandbox.rollout_runtime.resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=0,
            rollout_starts_at_action_zero=False,
        )
        == 0
    )
    assert (
        sandbox.rollout_runtime.resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=7,
            rollout_starts_at_action_zero=False,
        )
        == 7
    )
    assert (
        sandbox.rollout_runtime.resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=None,
            rollout_starts_at_action_zero=False,
        )
        == 15
    )


def test_generated_future_rollout_defaults_initial_generation_start_to_zero() -> None:
    sandbox = _load_sandbox_module()

    initial_obs_window = [{"image": np.zeros((2, 2, 3), dtype=np.uint8)} for _ in range(15)]

    assert (
        sandbox.rollout_runtime.resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=None,
            rollout_starts_at_action_zero=True,
        )
        == 0
    )


def test_dual_expert_rollout_defaults_initial_generation_start_to_zero() -> None:
    sandbox = _load_sandbox_module()

    policy_variant = _ZeroOriginRolloutPolicy()
    initial_obs_window = [
        {"image": np.zeros((2, 2, 3), dtype=np.uint8)} for _ in range(15)
    ]

    assert (
        sandbox.rollout_runtime.uses_zero_based_generation_start(policy_variant) is True
    )
    assert (
        sandbox.rollout_runtime.resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=None,
            rollout_starts_at_action_zero=sandbox.rollout_runtime.uses_zero_based_generation_start(
                policy_variant
            ),
        )
        == 0
    )


def test_dual_expert_program_routes_preserve_observation_conditioned_session() -> None:
    sandbox = _load_sandbox_module()
    calls = []

    class Runner:
        def reset(self, *, task_text=None, text_context=None, negative_text_context=None):
            calls.append(
                {
                    "task_text": task_text,
                    "text_context": text_context,
                    "negative_text_context": negative_text_context,
                }
            )
            return SimpleNamespace(
                task_text=task_text,
                text_context=text_context,
                negative_text_context=negative_text_context,
            )

    session = SimpleNamespace(
        task_text=("task",),
        text_context="text",
        negative_text_context="negative",
    )
    dual_expert_config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="dual_expert", program="video_then_action")
    )
    dual_expert_native_packed_config = SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="dual_expert",
            program="joint",
        )
    )
    parallel_config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="parallel_stream")
    )
    parallel_config = SimpleNamespace(policy_variant=SimpleNamespace(name="parallel_stream"))

    resolved = sandbox.realtime_runtime.resolve_observation_conditioned_replan_session(
        runner=Runner(),
        session=session,
        config=dual_expert_config,
    )

    assert resolved is session
    assert calls == []
    assert sandbox.realtime_runtime.uses_dual_expert_split_cache_sequence(dual_expert_config)
    assert not sandbox.realtime_runtime.uses_dual_expert_split_cache_sequence(dual_expert_native_packed_config)
    assert not sandbox.realtime_runtime.should_use_sequence_open_loop_extension(
        config=dual_expert_native_packed_config,
        planner_mode="async_history_first",
        remaining_buffer_actions=4,
    )
    assert (
        sandbox.realtime_runtime.resolve_sequence_action_cache_rewind_frame(
            config=dual_expert_native_packed_config,
            planner_mode="async_history_first",
            use_observation_update=True,
            condition_frame_start=8,
        )
        is None
    )

    resolved_native = sandbox.realtime_runtime.resolve_observation_conditioned_replan_session(
        runner=Runner(),
        session=session,
        config=dual_expert_native_packed_config,
    )

    assert resolved_native is session
    assert calls == []

    assert (
        sandbox.realtime_runtime.resolve_observation_conditioned_replan_session(
            runner=Runner(),
            session=session,
            config=parallel_config,
        )
        is session
    )


def test_dual_expert_startup_open_loop_requires_history_control_route() -> None:
    sandbox = _load_sandbox_module()
    dual_expert_split_cache_config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="dual_expert", program="video_then_action")
    )
    dual_expert_native_packed_config = SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="dual_expert",
            program="joint",
        )
    )

    assert (
        sandbox.realtime_runtime.validate_sequence_startup_open_loop_support(
            config=dual_expert_split_cache_config,
            startup_open_loop_chunks=1,
        ).supports_realtime_history_controls
        is True
    )
    sandbox.realtime_runtime.validate_sequence_startup_open_loop_support(
        config=dual_expert_native_packed_config,
        startup_open_loop_chunks=0,
    )

    with pytest.raises(
        ValueError, match="does not support startup open-loop extension"
    ):
        sandbox.realtime_runtime.validate_sequence_startup_open_loop_support(
            config=dual_expert_native_packed_config,
            startup_open_loop_chunks=1,
        )


def test_dual_expert_startup_open_loop_validation_runs_before_pipeline_setup(monkeypatch) -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="dual_expert",
            program="joint",
        )
    )

    def fail_pipeline_build(*args, **kwargs):
        raise AssertionError("pipeline setup should not run for invalid startup open-loop route")

    monkeypatch.setattr(sandbox, "build_variant_pipeline_from_config", fail_pipeline_build)

    with pytest.raises(ValueError, match="does not support startup open-loop extension"):
        sandbox._run_sequence_policy_realtime_rollout(
            config=config,
            checkpoint_path=Path("/tmp/checkpoint.pt"),
            rollout_label="test",
            benchmark="libero_10",
            task_id=0,
            episode_idx=0,
            max_actions=1,
            env_horizon=None,
            target_action_hz=1.0,
            video_fps=None,
            planner_mode="async_history_first",
            deadline_miss_policy="fallback",
            deadline_tolerance_ms=0.0,
            output_dir=Path("/tmp"),
            suffix="",
            seed=0,
            runtime_device=sandbox.torch.device("cpu"),
            runtime_devices=(sandbox.torch.device("cpu"),),
            runtime_prep_device=sandbox.torch.device("cpu"),
            runtime_output_device=sandbox.torch.device("cpu"),
            frontend_device=sandbox.torch.device("cpu"),
            decode_device=sandbox.torch.device("cpu"),
            sequence_buffer_threshold=1,
            sequence_empty_plan_policy="fallback",
            fallback_history_policy=sandbox.FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY,
            startup_open_loop_chunks=1,
            replan_low_watermark_actions=0,
            video_num_inference_steps=None,
            action_num_inference_steps=None,
            guidance_scale=None,
            action_guidance_scale=None,
            initial_generation_action_start=None,
            write_fallback_timeline_video=False,
            artifact_profile="debug",
        )



def test_sequence_rollout_infer_extra_matches_dual_expert_contract() -> None:
    sandbox = _load_sandbox_module()

    dual_expert_extra = sandbox.rollout_runtime.build_sequence_rollout_infer_extra(
        policy_variant=_ZeroOriginRolloutPolicy(),
        prompt="task",
        runtime_device=sandbox.torch.device("cpu"),
    )

    assert dual_expert_extra == {
        "task_text": ("task",),
        "action_device": "cpu",
    }


def test_sequence_raw_action_targets_materialize_without_pose_conversion() -> None:
    sandbox = _load_sandbox_module()
    action_pred = np.array(
        [
            [0.10, -0.20, 0.30, 0.40, -0.50, 1.60, -1.25],
            [-0.15, 0.25, -0.35, -0.45, 0.55, -0.65, 1.00],
        ],
        dtype=np.float32,
    )

    planned_steps = sandbox.realtime_runtime.sequence_chunk_to_planned_steps(
        action_pred=action_pred,
        reference_obs={},
        generation_action_start=7,
        source="test_plan",
        planner_step_index=3,
        ready_monotonic_s=12.5,
        action_target_representation=ActionTargetRepresentation.RAW,
        rotation_representation="axis_angle",
    )
    materialized = sandbox.realtime_runtime.materialize_sequence_control_action(
        planned_steps[0],
        current_obs={},
        control_config=sandbox.LiberoControlConfig(),
        gripper_representation="action_command",
    )

    assert [step.absolute_action_index for step in planned_steps] == [7, 8]
    assert planned_steps[0].desired_position is None
    assert planned_steps[0].desired_quaternion is None
    assert planned_steps[0].desired_gripper is None
    np.testing.assert_allclose(planned_steps[0].raw_action, action_pred[0])
    np.testing.assert_allclose(materialized, np.clip(action_pred[0], -1.0, 1.0))


def test_native_packed_dual_expert_realtime_does_not_drop_startup_actions() -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="dual_expert",
            program="action_then_video",
        ),
        data=SimpleNamespace(action_schema=SimpleNamespace(action_horizon=4)),
        inference=SimpleNamespace(frame_chunk_size=2),
    )
    action_pred = np.arange(4, dtype=np.float32).reshape(4, 1)

    planned_steps = sandbox.realtime_runtime.sequence_chunk_to_planned_steps(
        action_pred=action_pred,
        reference_obs={},
        generation_action_start=0,
        execution_action_offset=sandbox.realtime_runtime.resolve_sequence_execution_action_offset(config),
        source="native_packed_startup",
        planner_step_index=0,
        ready_monotonic_s=1.0,
        action_target_representation=ActionTargetRepresentation.RAW,
        rotation_representation="axis_angle",
    )
    merged = sandbox.merge_future_control_steps(
        {},
        planned_steps,
        next_action_to_execute=0,
    )

    assert sandbox.realtime_runtime.resolve_sequence_actions_per_frame(config) == 2
    assert sandbox.realtime_runtime.resolve_sequence_execution_action_offset(config) == 0
    assert [step.absolute_action_index for step in planned_steps] == [0, 1, 2, 3]
    assert sorted(merged) == [0, 1, 2, 3]
    assert sandbox.realtime_runtime.resolve_sequence_condition_frame_start(config=config, generation_action_start=2) == 1


def test_strict_split_cache_dual_expert_realtime_does_not_drop_startup_actions() -> None:
    sandbox = _load_sandbox_module()
    config = _strict_split_cache_dual_expert_config(sandbox, action_horizon=16, frame_chunk_size=4)
    action_pred = np.arange(16, dtype=np.float32).reshape(16, 1)

    planned_steps = sandbox.realtime_runtime.sequence_chunk_to_planned_steps(
        action_pred=action_pred,
        reference_obs={},
        generation_action_start=0,
        execution_action_offset=sandbox.realtime_runtime.resolve_sequence_execution_action_offset(config),
        source="strict_split_cache_startup",
        planner_step_index=0,
        ready_monotonic_s=1.0,
        action_target_representation=ActionTargetRepresentation.RAW,
        rotation_representation="axis_angle",
    )
    merged = sandbox.merge_future_control_steps(
        {},
        planned_steps,
        next_action_to_execute=0,
    )

    assert sandbox.realtime_runtime.resolve_sequence_actions_per_frame(config) == 4
    assert sandbox.realtime_runtime.resolve_sequence_execution_action_offset(config) == 0
    assert [step.absolute_action_index for step in planned_steps] == list(range(16))
    assert sorted(merged) == list(range(16))


def test_default_contract_split_cache_realtime_keeps_one_frame_execution_offset() -> (
    None
):
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="dual_expert",
            program="video_then_action",
        ),
        data=SimpleNamespace(action_schema=SimpleNamespace(action_horizon=4)),
        inference=SimpleNamespace(frame_chunk_size=2),
    )

    assert sandbox.realtime_runtime.resolve_sequence_actions_per_frame(config) == 2
    assert sandbox.realtime_runtime.resolve_sequence_execution_action_offset(config) == 2
    assert sandbox.realtime_runtime.resolve_sequence_condition_frame_start(config=config, generation_action_start=2) == 1


def test_strict_split_cache_dual_expert_startup_uses_one_model_observation() -> None:
    sandbox = _load_sandbox_module()
    config = _strict_split_cache_dual_expert_config(sandbox)
    initial_obs_window = [_minimal_obs_record(float(index)) for index in range(13)]

    startup_window = sandbox.realtime_runtime.build_sequence_startup_observation_window(config, initial_obs_window)

    assert len(startup_window) == 1
    np.testing.assert_allclose(startup_window[0]["robot0_eef_pos"], initial_obs_window[-1]["robot0_eef_pos"])
    assert startup_window[0]["robot0_eef_pos"] is not initial_obs_window[-1]["robot0_eef_pos"]


def test_strict_split_cache_dual_expert_startup_env_init_uses_single_frame() -> None:
    sandbox = _load_sandbox_module()
    config = _strict_split_cache_dual_expert_config(sandbox)

    assert sandbox.realtime_runtime.uses_strict_dual_expert_split_cache_startup(config)
    assert sandbox.realtime_runtime.uses_strict_dual_expert_one_frame_history(config)
    assert sandbox.realtime_runtime.resolve_sequence_startup_environment_frames(config, raw_window_frames=13) == 1
    assert sandbox.realtime_runtime.resolve_sequence_model_observation_window_frames(config, raw_window_frames=13) == 1


def test_strict_native_packed_dual_expert_startup_env_init_uses_single_frame() -> None:
    sandbox = _load_sandbox_module()
    config = _strict_split_cache_dual_expert_config(sandbox, program="joint")
    initial_obs_window = [_minimal_obs_record(float(index)) for index in range(13)]

    assert not sandbox.realtime_runtime.uses_strict_dual_expert_split_cache_startup(config)
    assert sandbox.realtime_runtime.uses_strict_dual_expert_one_frame_history(config)
    assert sandbox.realtime_runtime.resolve_sequence_startup_environment_frames(config, raw_window_frames=13) == 1
    assert sandbox.realtime_runtime.resolve_sequence_model_observation_window_frames(config, raw_window_frames=13) == 1

    startup_window = sandbox.realtime_runtime.build_sequence_startup_observation_window(config, initial_obs_window)
    assert len(startup_window) == 1
    np.testing.assert_allclose(startup_window[0]["robot0_eef_pos"], initial_obs_window[-1]["robot0_eef_pos"])


def test_strict_split_cache_dual_expert_model_history_keeps_latest_observation() -> None:
    sandbox = _load_sandbox_module()
    config = _strict_split_cache_dual_expert_config(sandbox, action_horizon=16, frame_chunk_size=4)
    model_obs_window = sandbox.realtime_runtime.build_sequence_startup_observation_window(
        config,
        [_minimal_obs_record(float(index)) for index in range(4)],
    )
    state = sandbox.realtime_history.ActionFallbackHistoryState(
        policy=sandbox.FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY,
    )

    assert len(model_obs_window) == 1
    assert (
        sandbox.realtime_history.append_action_observation(
            model_obs_window=model_obs_window,
            state=state,
            current_obs=_minimal_obs_record(4.0),
            action_source="startup_plan",
            clean_actions_required=16,
            max_window_frames=sandbox.realtime_runtime.resolve_sequence_model_observation_window_frames(config, raw_window_frames=13),
        )
        == "included"
    )

    assert len(model_obs_window) == 1
    np.testing.assert_allclose(model_obs_window[0]["robot0_eef_pos"], [4.0, 4.0, 4.0])
    condition_frame_start = sandbox.realtime_runtime.resolve_sequence_condition_frame_start(
        config=config,
        generation_action_start=16,
    )
    generation_frame_start = condition_frame_start + len(model_obs_window)
    assert condition_frame_start == 4
    assert generation_frame_start == 5
    assert sandbox.frame_index_to_action_start(generation_frame_start, 4) == 16


def test_strict_split_cache_dual_expert_realtime_init_calls_env_with_single_frame(monkeypatch) -> None:
    sandbox = _load_sandbox_module()
    config = _strict_split_cache_dual_expert_config(sandbox)
    config.data.num_frames = 4
    config.data.action_schema.action_dim = 7
    config.data.action_target.gripper_representation = "action_command"
    config.backbone = SimpleNamespace(
        runtime_backbone_artifact_path="/tmp/transformer",
        reference_assets_device_policy="runtime",
    )
    init_calls = []
    replan_obs_lengths = []

    class FakeVisualTower:
        def configure_runtime_devices(self, *args, **kwargs):
            del args, kwargs

    class FakePipeline:
        visual_tower = FakeVisualTower()
        action_decoder = SimpleNamespace(rollout_chunk_steps=16)
        policy_variant = _ZeroOriginRolloutPolicy()

        def to(self, device):
            del device
            return self

        def eval(self):
            return self

    class FakeRunner:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def reset(self, *, task_text=None, text_context=None, negative_text_context=None):
            return SimpleNamespace(
                task_text=task_text,
                text_context=text_context,
                negative_text_context=negative_text_context,
                policy_state=SimpleNamespace(step_index=0, decoder_state=None),
            )

    class FakeEnv:
        env = SimpleNamespace(timestep=5)

        def close(self):
            pass

    def fake_init_single_env(env, init_state, *, num_frames):
        del env, init_state
        init_calls.append(int(num_frames))
        return [_minimal_obs_record(5.0)]

    def fake_replan_job(**kwargs):
        replan_obs_lengths.append(len(kwargs["obs_window"]))
        return sandbox.realtime_runtime.SequenceReplanJobResult(
            session=kwargs["session"],
            runtime_cache_snapshot=None,
            planned_steps=[],
            next_generation_action_start=16,
            trace={"job_kind": "startup_plan"},
        )

    monkeypatch.setattr(sandbox, "_print_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(sandbox, "build_variant_pipeline_from_config", lambda cfg: FakePipeline())
    monkeypatch.setattr(sandbox, "VariantRolloutRunner", FakeRunner)
    monkeypatch.setattr(sandbox, "ensure_dual_expert_inference_backend", lambda pipeline, cfg: {"backend": "test"})
    monkeypatch.setattr(
        sandbox.runtime_checkpoints,
        "load_pipeline_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            missing_keys=(),
            unexpected_keys=(),
        ),
    )
    monkeypatch.setattr(
        sandbox,
        "resolve_libero_task_by_id",
        lambda benchmark, task_id, project_root: SimpleNamespace(
            task_language="task"
        ),
    )
    monkeypatch.setattr(
        sandbox,
        "load_libero_task_init_states",
        lambda task_spec, project_root: ["init"],
    )
    monkeypatch.setattr(sandbox, "_construct_realtime_libero_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(
        sandbox.libero_rollout,
        "initialize_libero_observation_window",
        fake_init_single_env,
    )
    monkeypatch.setattr(
        sandbox.libero_rollout,
        "libero_observation_window_to_views",
        lambda obs_window, *, device: {},
    )
    monkeypatch.setattr(
        sandbox.rollout_runtime,
        "prepare_rollout_observation_inputs",
        lambda *args, **kwargs: {
            "text_context": "text",
            "negative_text_context": "negative",
        },
    )
    monkeypatch.setattr(sandbox.realtime_runtime, "synchronize_devices", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sandbox.realtime_runtime,
        "run_sequence_replan_job",
        fake_replan_job,
    )
    monkeypatch.setattr(sandbox, "build_live_rollout_summary", lambda **kwargs: {"summary": True})
    monkeypatch.setattr(sandbox, "_finalize_rollout_outputs", lambda **kwargs: kwargs["summary"])

    summary = sandbox._run_sequence_policy_realtime_rollout(
        config=config,
        checkpoint_path=Path("/tmp/checkpoint.pt"),
        rollout_label="test",
        benchmark="libero_10",
        task_id=0,
        episode_idx=0,
        max_actions=0,
        env_horizon=None,
        target_action_hz=1.0,
        video_fps=None,
        planner_mode="async_history_first",
        deadline_miss_policy="fallback",
        deadline_tolerance_ms=0.0,
        output_dir=Path("/tmp"),
        suffix="",
        seed=0,
        runtime_device=sandbox.torch.device("cpu"),
        runtime_devices=(sandbox.torch.device("cpu"),),
        runtime_prep_device=sandbox.torch.device("cpu"),
        runtime_output_device=sandbox.torch.device("cpu"),
        frontend_device=sandbox.torch.device("cpu"),
        decode_device=sandbox.torch.device("cpu"),
        sequence_buffer_threshold=1,
        sequence_empty_plan_policy="fallback",
        fallback_history_policy=sandbox.FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY,
        startup_open_loop_chunks=0,
        replan_low_watermark_actions=0,
        video_num_inference_steps=None,
        action_num_inference_steps=None,
        guidance_scale=None,
        action_guidance_scale=None,
        initial_generation_action_start=None,
        write_fallback_timeline_video=False,
        artifact_profile="debug",
    )

    assert init_calls == [1]
    assert replan_obs_lengths == [1]
    assert summary["raw_window_frames"] == 13
    assert summary["startup_env_init_frames"] == 1


def test_default_contract_split_cache_startup_keeps_full_model_observation_window() -> (
    None
):
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="dual_expert",
            program="video_then_action",
        )
    )
    initial_obs_window = [_minimal_obs_record(float(index)) for index in range(13)]

    startup_window = sandbox.realtime_runtime.build_sequence_startup_observation_window(config, initial_obs_window)

    assert len(startup_window) == len(initial_obs_window)
    np.testing.assert_allclose(startup_window[0]["robot0_eef_pos"], initial_obs_window[0]["robot0_eef_pos"])
    assert startup_window[0]["robot0_eef_pos"] is not initial_obs_window[0]["robot0_eef_pos"]
    assert sandbox.realtime_runtime.resolve_sequence_startup_environment_frames(config, raw_window_frames=13) == 13


def test_strict_split_cache_dual_expert_startup_rejects_multi_latent_context() -> None:
    sandbox = _load_sandbox_module()
    config = _strict_split_cache_dual_expert_config(sandbox)

    with pytest.raises(ValueError, match="exactly one latent context frame"):
        sandbox.realtime_runtime.validate_sequence_startup_inputs(
            config=config,
            source="startup_plan",
            generation_action_start=0,
            video_latents=sandbox.torch.zeros(1, 48, 4, 1, 1),
        )


def test_strict_split_cache_dual_expert_startup_replan_trace_reports_origin(monkeypatch) -> None:
    sandbox = _load_sandbox_module()
    config = _strict_split_cache_dual_expert_config(sandbox)
    captured_video_latents = {}

    monkeypatch.setattr(
        sandbox.libero_rollout,
        "libero_observation_window_to_views",
        lambda obs_window, *, device: {"obs_count": len(obs_window)},
    )
    monkeypatch.setattr(
        sandbox.rollout_runtime,
        "prepare_rollout_observation_inputs",
        lambda *args, **kwargs: {
            "video_latents": sandbox.torch.zeros(1, 48, 1, 1, 1),
            "text_context": None,
            "negative_text_context": None,
        },
    )
    monkeypatch.setattr(sandbox.realtime_runtime, "synchronize_devices", lambda *args, **kwargs: None)

    class Runner(sandbox.VariantRolloutRunner):
        def __init__(self) -> None:
            super().__init__(
                SimpleNamespace(
                    action_decoder=_RealtimeTestActionDecoder(rollout_chunk_steps=16),
                    policy_variant=_ZeroOriginRolloutPolicy(),
                )
            )

        def infer_step(self, *, session, context, video_latents, canonical_video=None):
            del context, canonical_video
            captured_video_latents["shape"] = tuple(video_latents.shape)
            next_session = SimpleNamespace(policy_state=SimpleNamespace(step_index=3, decoder_state=None))
            policy_output = SimpleNamespace(
                aux={
                    "generation_frame_start": 1,
                    "dual_expert_cache_debug": {
                        "chunk_origin_frame": 1,
                        "current_action_frame_start": 1,
                    },
                },
            )
            decoder_output = SimpleNamespace(
                action_pred=sandbox.torch.zeros(1, 16, 7),
                aux={},
            )
            return SimpleNamespace(
                session=next_session,
                infer_output=SimpleNamespace(
                    policy_output=policy_output,
                    decoder_output=decoder_output,
                ),
            )

    result = sandbox.realtime_runtime.run_sequence_replan_job(
        runner=Runner(),
        session=SimpleNamespace(text_context=None, negative_text_context=None, task_text=("task",)),
        obs_window=[_minimal_obs_record(0.0)],
        config=config,
        options=sandbox.realtime_runtime.SequenceReplanJobOptions(
            prompt="task",
            task_id=0,
            episode_idx=0,
            frontend_device=sandbox.torch.device("cpu"),
            runtime_device=sandbox.torch.device("cpu"),
            generation_action_start=0,
            source="startup_plan",
        ),
    )

    assert captured_video_latents["shape"][2] == 1
    assert result.trace["execution_action_offset"] == 0
    assert result.trace["model_generation_frame_start"] == 1
    assert result.trace["dual_expert_chunk_origin_frame"] == 1
    assert result.trace["dual_expert_current_action_frame_start"] == 1
    assert result.trace["planned_action_ids"] == list(range(16))
    assert [step.absolute_action_index for step in result.planned_steps] == list(range(16))


def test_realtime_common_inference_overrides_preserve_config_values_by_default() -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        inference=SimpleNamespace(
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        )
    )

    sandbox._apply_common_inference_overrides(
        config,
        video_num_inference_steps=None,
        action_num_inference_steps=None,
        guidance_scale=None,
        action_guidance_scale=None,
    )

    assert config.inference.video_num_inference_steps == 20
    assert config.inference.action_num_inference_steps == 50
    assert config.inference.guidance_scale == 5.0
    assert config.inference.action_guidance_scale == 1.0


def test_realtime_common_inference_overrides_apply_explicit_smoke_values() -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        inference=SimpleNamespace(
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        )
    )

    sandbox._apply_common_inference_overrides(
        config,
        video_num_inference_steps=2,
        action_num_inference_steps=3,
        guidance_scale=1.5,
        action_guidance_scale=0.75,
    )

    assert config.inference.video_num_inference_steps == 2
    assert config.inference.action_num_inference_steps == 3
    assert config.inference.guidance_scale == 1.5
    assert config.inference.action_guidance_scale == 0.75


@pytest.mark.parametrize(
    ("video_steps", "action_steps"),
    [
        (0, None),
        (None, 0),
        (-1, None),
        (None, -1),
    ],
)
def test_realtime_common_inference_overrides_reject_nonpositive_step_values(
    video_steps: int | None,
    action_steps: int | None,
) -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(
        inference=SimpleNamespace(
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        )
    )

    with pytest.raises(ValueError, match="must be positive"):
        sandbox._apply_common_inference_overrides(
            config,
            video_num_inference_steps=video_steps,
            action_num_inference_steps=action_steps,
            guidance_scale=None,
            action_guidance_scale=None,
        )

    assert config.inference.video_num_inference_steps == 20
    assert config.inference.action_num_inference_steps == 50


def test_exact_realtime_inference_overrides_preserve_config_values_by_default() -> None:
    sandbox = _load_sandbox_module()
    runner = SimpleNamespace(
        policy_variant=SimpleNamespace(
            inference_config=SimpleNamespace(
                video_num_inference_steps=20,
                action_num_inference_steps=50,
                guidance_scale=5.0,
                action_guidance_scale=1.0,
            )
        )
    )

    sandbox.realtime_runtime.apply_inference_overrides(
        runner,
        video_num_inference_steps=None,
        action_num_inference_steps=None,
        guidance_scale=None,
        action_guidance_scale=None,
    )

    assert runner.policy_variant.inference_config.video_num_inference_steps == 20
    assert runner.policy_variant.inference_config.action_num_inference_steps == 50
    assert runner.policy_variant.inference_config.guidance_scale == 5.0
    assert runner.policy_variant.inference_config.action_guidance_scale == 1.0


@pytest.mark.parametrize(
    ("video_steps", "action_steps"),
    [
        (0, None),
        (None, 0),
        (-1, None),
        (None, -1),
    ],
)
def test_exact_realtime_inference_overrides_reject_nonpositive_step_values(
    video_steps: int | None,
    action_steps: int | None,
) -> None:
    sandbox = _load_sandbox_module()
    runner = SimpleNamespace(
        policy_variant=SimpleNamespace(
            inference_config=SimpleNamespace(
                video_num_inference_steps=20,
                action_num_inference_steps=50,
                guidance_scale=5.0,
                action_guidance_scale=1.0,
            )
        )
    )

    with pytest.raises(ValueError, match="must be positive"):
        sandbox.realtime_runtime.apply_inference_overrides(
            runner,
            video_num_inference_steps=video_steps,
            action_num_inference_steps=action_steps,
            guidance_scale=None,
            action_guidance_scale=None,
        )

    assert runner.policy_variant.inference_config.video_num_inference_steps == 20
    assert runner.policy_variant.inference_config.action_num_inference_steps == 50
