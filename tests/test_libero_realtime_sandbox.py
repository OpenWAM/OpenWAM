from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import numpy as np
import pytest


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


def test_frame_index_to_action_start_matches_exact_realtime_convention() -> None:
    sandbox = _load_sandbox_module()

    assert sandbox._frame_index_to_action_start(1, 4) == 0
    assert sandbox._frame_index_to_action_start(2, 4) == 4
    assert sandbox._frame_index_to_action_start(3, 4) == 8


def test_required_frame_action_indices_stop_at_rollout_limit() -> None:
    sandbox = _load_sandbox_module()

    assert sandbox._required_frame_action_indices(
        next_action_index=8,
        max_actions=10,
        action_per_frame=4,
    ) == [8, 9]


def test_merge_future_step_actions_drops_stale_steps_and_prefers_newer_future() -> None:
    sandbox = _load_sandbox_module()
    step_cls = sandbox.PlannedControlStep

    existing = {
        0: step_cls(absolute_action_index=0, generation_action_start=0, source="old"),
        1: step_cls(absolute_action_index=1, generation_action_start=0, source="old"),
        2: step_cls(absolute_action_index=2, generation_action_start=0, source="old"),
    }
    incoming = [
        step_cls(absolute_action_index=1, generation_action_start=1, source="new"),
        step_cls(absolute_action_index=3, generation_action_start=1, source="new"),
    ]

    merged = sandbox._merge_future_step_actions(existing, incoming, next_action_to_execute=1)

    assert list(merged) == [1, 2, 3]
    assert merged[1].source == "new"
    assert merged[2].source == "old"
    assert merged[3].source == "new"


def test_missing_plan_action_indices_reports_required_gaps() -> None:
    sandbox = _load_sandbox_module()
    step_cls = sandbox.PlannedControlStep
    plan_by_action = {
        4: step_cls(absolute_action_index=4, generation_action_start=4, source="plan"),
        6: step_cls(absolute_action_index=6, generation_action_start=4, source="plan"),
    }

    missing = sandbox._missing_plan_action_indices(plan_by_action, [4, 5, 6, 7])

    assert missing == [5, 7]


def test_exact_wait_mode_prefers_history_replans_when_history_exists() -> None:
    sandbox = _load_sandbox_module()

    assert (
        sandbox._resolve_exact_realtime_planner_mode(
            planner_mode="async_buffer",
            sequence_empty_plan_policy="wait_for_replan",
            pending_history=[{"absolute_frame_index": 1}],
        )
        == "history_only"
    )
    assert (
        sandbox._resolve_exact_realtime_planner_mode(
            planner_mode="async_buffer",
            sequence_empty_plan_policy="wait_for_replan",
            pending_history=[],
        )
        == "async_buffer"
    )
    assert (
        sandbox._resolve_exact_realtime_planner_mode(
            planner_mode="async_buffer",
            sequence_empty_plan_policy="fallback",
            pending_history=[{"absolute_frame_index": 1}],
        )
        == "async_buffer"
    )


def test_exact_wait_mode_submits_replans_only_when_buffer_is_empty() -> None:
    sandbox = _load_sandbox_module()

    assert sandbox._should_submit_exact_realtime_planner(
        future_buffer_depth_frames=0,
        sequence_empty_plan_policy="wait_for_replan",
    )
    assert not sandbox._should_submit_exact_realtime_planner(
        future_buffer_depth_frames=2,
        sequence_empty_plan_policy="wait_for_replan",
    )
    assert sandbox._should_submit_exact_realtime_planner(
        future_buffer_depth_frames=2,
        sequence_empty_plan_policy="fallback",
    )


def test_exact_realtime_job_seed_tracks_session_step_index() -> None:
    sandbox = _load_sandbox_module()
    session = SimpleNamespace(policy_state=SimpleNamespace(step_index=4))

    assert sandbox.exact_sandbox._job_seed_for_session(10, session) == 14
    assert sandbox.exact_sandbox._job_seed_for_session(None, session) is None


def test_exact_startup_sessions_replan_from_first_chunk_state() -> None:
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

    history_session, current_session, buffer_tail_session = sandbox.exact_sandbox._resolve_exact_startup_sessions(
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact")),
        startup_session=startup_session,
        first_chunk=first_chunk,
        frame_chunk_size=4,
    )

    assert history_session is session
    assert current_session is session
    assert buffer_tail_session.policy_state.step_index == 1
    assert buffer_tail_session.policy_state.cache["frame_start"] == 4
    assert buffer_tail_session.policy_state.cursor.current_start_frame == 4


def test_exact_action_conditioned_startup_sessions_keep_reset_history_base() -> None:
    sandbox = _load_sandbox_module()
    startup_session = SimpleNamespace(policy_state=SimpleNamespace(step_index=0))
    chunk_session = SimpleNamespace(
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
        session=chunk_session,
        debug={"generation_frame_start": 0},
    )

    history_session, current_session, buffer_tail_session = sandbox.exact_sandbox._resolve_exact_startup_sessions(
        config=SimpleNamespace(policy_variant=SimpleNamespace(runtime_mode="lingbot_exact_action_conditioned")),
        startup_session=startup_session,
        first_chunk=first_chunk,
        frame_chunk_size=4,
    )

    assert history_session is startup_session
    assert current_session is chunk_session
    assert buffer_tail_session.policy_state.step_index == 1


def test_exact_chunk_to_planned_steps_overwrites_conditioning_frame_actions() -> None:
    sandbox = _load_sandbox_module()
    raw_actions = sandbox.torch.arange(16, dtype=sandbox.torch.float32).view(1, 16, 1)
    chunk = SimpleNamespace(
        raw_chunk_action_pred=raw_actions,
        debug={"generation_frame_start": 0},
        session=SimpleNamespace(policy_state=SimpleNamespace(step_index=3)),
    )

    planned = sandbox._exact_chunk_to_planned_steps(
        chunk=chunk,
        action_per_frame=4,
        frame_chunk_size=4,
        source="startup_plan",
        ready_monotonic_s=1.5,
    )
    merged = sandbox._merge_future_step_actions({}, planned, next_action_to_execute=0)

    assert len(planned) == 12
    assert [step.absolute_action_index for step in planned] == list(range(12))
    assert all(step.generation_frame_start == 0 for step in planned)
    assert list(merged) == list(range(12))
    assert merged[0].raw_action is not None
    np.testing.assert_allclose(merged[0].raw_action, np.array([4.0], dtype=np.float32))
    assert merged[11].raw_action is not None
    np.testing.assert_allclose(merged[11].raw_action, np.array([15.0], dtype=np.float32))


def test_exact_startup_conditioning_history_preserves_skipped_frame_and_latent() -> None:
    sandbox = _load_sandbox_module()
    raw_actions = sandbox.torch.arange(16, dtype=sandbox.torch.float32).view(1, 16, 1)
    initial_latents = sandbox.torch.ones(1, 2, 1, 1, 1)
    chunk = SimpleNamespace(
        raw_chunk_action_pred=raw_actions,
        debug={"generation_frame_start": 0},
    )

    record = sandbox._exact_startup_conditioning_history_record(
        chunk=chunk,
        initial_video_latents=initial_latents,
        initial_obs={"image": np.zeros((2, 2, 3), dtype=np.uint8)},
        action_per_frame=4,
        frame_chunk_size=4,
    )

    assert record["absolute_frame_index"] == 0
    assert record["obs_sequence"] == []
    sandbox.torch.testing.assert_close(record["video_latents"], initial_latents)
    np.testing.assert_allclose(record["raw_actions"], np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32))


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

    copied = sandbox.exact_sandbox._copy_history_record_for_worker(record)
    views = sandbox.exact_sandbox._history_records_to_obs_sequence([copied])
    latents = sandbox.exact_sandbox._history_records_to_precomputed_video_latents([copied])

    assert len(views) == 2
    assert latents is not None
    assert tuple(latents.shape) == (1, 2, 1, 1, 1)
    copied["obs_sequence"][0]["image"][0, 0, 0] = 99
    assert record["obs_sequence"][0]["image"][0, 0, 0] == 2


def test_exact_future_result_records_stale_and_future_planned_actions() -> None:
    sandbox = _load_sandbox_module()
    frame_cls = sandbox.exact_sandbox.PlannedFrameAction
    trace = {"job_kind": "open_loop_extension"}
    result = {
        "job_kind": "open_loop_extension",
        "planned_frames": [
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
        "buffer_tail_session": "tail",
        "trace": trace,
    }

    _, _, _, merged, _ = sandbox._consume_exact_future_result(
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

    assert sorted(merged) == [6, 7, 8, 9, 10, 11]
    assert trace["stale_planned_actions"] == 2
    assert trace["future_planned_actions"] == 6
    assert trace["planned_action_indices"] == list(range(4, 12))


def test_exact_history_replan_result_advances_base_session_to_chunk_session() -> None:
    sandbox = _load_sandbox_module()
    result = {
        "job_kind": "history_replan",
        "planned_frames": [],
        "warmup_session": "warmup",
        "session": "chunk",
        "buffer_tail_session": "tail",
        "submitted_through_frame": 3,
        "trace": {"job_kind": "history_replan"},
    }

    history_session, chunk_session, buffer_tail_session, merged, pending_history = sandbox._consume_exact_future_result(
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

    assert history_session == "chunk"
    assert chunk_session == "chunk"
    assert buffer_tail_session == "tail"
    assert merged == {}
    assert pending_history == [{"absolute_frame_index": 4}]


def test_exact_action_conditioned_history_replan_keeps_warmup_session_base() -> None:
    sandbox = _load_sandbox_module()
    result = {
        "job_kind": "history_replan",
        "planned_frames": [],
        "warmup_session": "warmup",
        "session": "chunk",
        "buffer_tail_session": "tail",
        "submitted_through_frame": 1,
        "trace": {"job_kind": "history_replan"},
    }

    history_session, chunk_session, _, _, _ = sandbox._consume_exact_future_result(
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

    assert history_session == "warmup"
    assert chunk_session == "chunk"


def test_apply_sequence_replan_result_records_trace_and_merges_future_steps() -> None:
    sandbox = _load_sandbox_module()
    step_cls = sandbox.PlannedControlStep
    replan_records = []
    existing = {
        1: step_cls(absolute_action_index=1, generation_action_start=0, source="old"),
        2: step_cls(absolute_action_index=2, generation_action_start=0, source="old"),
    }
    result = {
        "session": "session",
        "next_generation_action_start": 5,
        "planned_steps": [
            step_cls(absolute_action_index=0, generation_action_start=0, source="stale"),
            step_cls(absolute_action_index=2, generation_action_start=2, source="new"),
            step_cls(absolute_action_index=3, generation_action_start=2, source="new"),
        ],
        "trace": {"job_kind": "history_replan"},
    }

    session, next_start, merged = sandbox._apply_sequence_replan_result(
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


def test_video_sequence_visualization_commits_full_decoder_chunk() -> None:
    sandbox = _load_sandbox_module()
    decoder_output = SimpleNamespace(
        action_pred=sandbox.torch.arange(6, dtype=sandbox.torch.float32).view(1, 6, 1),
        aux={
            "current_action": sandbox.torch.tensor([0.0], dtype=sandbox.torch.float32),
            "current_action_index": sandbox.torch.tensor(0.0),
            "rollout_chunk_steps": sandbox.torch.tensor(6.0),
        },
        next_state=SimpleNamespace(
            step_within_chunk=1,
            aux={"rollout_chunk_steps": 6},
        ),
    )

    planned_chunk, metadata = sandbox.video_viz._decoder_output_to_rollout_action_plan(decoder_output)

    assert planned_chunk.shape == (6, 1)
    np.testing.assert_allclose(planned_chunk[:, 0], np.arange(6, dtype=np.float32))
    assert metadata["decoder_rollout_commit_end_index"] == 6
    session = SimpleNamespace(
        policy_state=SimpleNamespace(
            decoder_state=SimpleNamespace(step_within_chunk=1),
        )
    )
    sandbox.video_viz._advance_decoder_state_to_rollout_commit(session, metadata)
    assert session.policy_state.decoder_state.step_within_chunk == 6


def test_video_sequence_visualization_clamps_raw_control_actions() -> None:
    sandbox = _load_sandbox_module()

    action = sandbox.video_viz._materialize_raw_control_action(np.array([1.5, -2.0, 0.25], dtype=np.float32))

    np.testing.assert_allclose(action, np.array([1.0, -1.0, 0.25], dtype=np.float32))


def test_video_sequence_visualization_can_override_rollout_chunk_steps() -> None:
    sandbox = _load_sandbox_module()
    config = SimpleNamespace(action_decoder=SimpleNamespace(rollout_chunk_steps=6))

    sandbox.video_viz._apply_rollout_chunk_steps_override(config, 1)

    assert config.action_decoder.rollout_chunk_steps == 1


def test_video_sequence_visualization_initial_generation_start_matches_warmup_window() -> None:
    sandbox = _load_sandbox_module()

    initial_obs_window = [{"image": np.zeros((2, 2, 3), dtype=np.uint8)} for _ in range(15)]

    assert sandbox.video_viz._initial_generation_action_start(initial_obs_window) == 15


def test_video_sequence_visualization_can_override_initial_generation_start() -> None:
    sandbox = _load_sandbox_module()

    initial_obs_window = [{"image": np.zeros((2, 2, 3), dtype=np.uint8)} for _ in range(15)]

    assert (
        sandbox.video_viz._resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=0,
            rollout_starts_at_action_zero=False,
        )
        == 0
    )
    assert (
        sandbox.video_viz._resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=7,
            rollout_starts_at_action_zero=False,
        )
        == 7
    )
    assert (
        sandbox.video_viz._resolve_initial_generation_action_start(
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
        sandbox.video_viz._resolve_initial_generation_action_start(
            initial_obs_window,
            initial_generation_action_start=None,
            rollout_starts_at_action_zero=True,
        )
        == 0
    )


def test_mot_realtime_replan_resets_observation_conditioned_session() -> None:
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
    mot_config = SimpleNamespace(policy_variant=SimpleNamespace(name="mot"))
    method4_config = SimpleNamespace(policy_variant=SimpleNamespace(name="post_latent"))

    resolved = sandbox._resolve_observation_conditioned_replan_session(
        runner=Runner(),
        session=session,
        config=mot_config,
    )

    assert resolved is not session
    assert calls == [
        {
            "task_text": ("task",),
            "text_context": "text",
            "negative_text_context": "negative",
        }
    ]
    assert (
        sandbox._resolve_observation_conditioned_replan_session(
            runner=Runner(),
            session=session,
            config=method4_config,
        )
        is session
    )


def test_method4_realtime_replan_uses_absolute_action_start_for_video_condition(monkeypatch) -> None:
    sandbox = _load_sandbox_module()
    captured_extra = {}

    monkeypatch.setattr(
        sandbox.video_viz,
        "_obs_window_to_rollout_views",
        lambda obs_window, *, device: {},
    )
    monkeypatch.setattr(
        sandbox.video_viz,
        "_prepare_rollout_inputs",
        lambda *args, **kwargs: {
            "video_latents": sandbox.torch.zeros(1, 1, 1, 1, 1),
            "text_context": None,
            "negative_text_context": None,
        },
    )
    monkeypatch.setattr(sandbox.exact_sandbox, "_synchronize_devices", lambda *args, **kwargs: None)

    class Runner:
        pipeline = SimpleNamespace(action_decoder=SimpleNamespace(rollout_chunk_steps=6))

        def infer_step(self, *, session, context, video_latents, canonical_video=None):
            del video_latents, canonical_video
            captured_extra.update(context.extra)
            next_session = SimpleNamespace(policy_state=SimpleNamespace(step_index=4, decoder_state=None))
            video_window = SimpleNamespace(
                metadata={
                    "frame_start": context.extra["video_condition_frame_start"],
                    "sample_seed": context.extra["video_condition_sample_seed"],
                }
            )
            policy_output = SimpleNamespace(
                aux={
                    "video_condition_source": "generated_future_video_tokens",
                    "video_condition_uses_future_ground_truth": False,
                },
                decoder_sequence_context=SimpleNamespace(video_condition_window=video_window),
            )
            decoder_output = SimpleNamespace(
                action_pred=sandbox.torch.zeros(1, 2, 7),
                aux={},
            )
            infer_output = SimpleNamespace(
                policy_output=policy_output,
                decoder_output=decoder_output,
            )
            return SimpleNamespace(session=next_session, infer_output=infer_output)

    config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="post_latent"),
        data=SimpleNamespace(
            action_schema=SimpleNamespace(state_horizon=1),
            action_target=SimpleNamespace(
                state_encoding="eef_pos_axisangle_gripper_2d",
                representation=sandbox.ActionTargetRepresentation.RAW,
                rotation_representation="axis_angle",
            ),
        ),
        inference=SimpleNamespace(action_num_inference_steps=20, video_num_inference_steps=20),
    )
    obs_window = [
        {
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }
    ]

    result = sandbox._run_sequence_replan_job(
        runner=Runner(),
        session=SimpleNamespace(text_context=None, negative_text_context=None, task_text=("task",)),
        obs_window=obs_window,
        prompt="task",
        task_id=1,
        episode_idx=7,
        config=config,
        frontend_device=sandbox.torch.device("cpu"),
        runtime_device=sandbox.torch.device("cpu"),
        generation_action_start=42,
        source="test",
    )

    assert captured_extra["video_condition_observed_prefix_anchor"] == "end"
    assert captured_extra["video_condition_frame_start"] == 42
    assert captured_extra["video_condition_sample_seed"] == sandbox.video_viz.derive_video_condition_sample_seed(
        {
            "task_index": 1,
            "episode_index": 7,
            "anchor_frame_index": 42,
            "action_start_index": 42,
        }
    )
    assert result["trace"]["video_condition_frame_start"] == 42
    assert result["trace"]["video_condition_sample_seed"] == captured_extra["video_condition_sample_seed"]
    assert [step.absolute_action_index for step in result["planned_steps"]] == [42, 43]


def test_sequence_rollout_infer_extra_matches_sandbox_and_viz_contract() -> None:
    sandbox = _load_sandbox_module()
    method4_config = SimpleNamespace(policy_variant=SimpleNamespace(name="post_latent"))
    mot_config = SimpleNamespace(policy_variant=SimpleNamespace(name="mot"))

    method4_extra = sandbox.video_viz._build_sequence_rollout_infer_extra(
        config=method4_config,
        prompt="task",
        generation_action_start=42,
        runtime_device=sandbox.torch.device("cpu"),
        task_id=1,
        episode_idx=7,
    )
    mot_extra = sandbox.video_viz._build_sequence_rollout_infer_extra(
        config=mot_config,
        prompt="task",
        generation_action_start=7,
        runtime_device=sandbox.torch.device("cpu"),
    )

    assert method4_extra == {
        "task_text": ("task",),
        "video_condition_frame_start": 42,
        "video_condition_sample_seed": sandbox.video_viz.derive_video_condition_sample_seed(
            {
                "task_index": 1,
                "episode_index": 7,
                "anchor_frame_index": 42,
                "action_start_index": 42,
            }
        ),
        "video_condition_observed_prefix_anchor": "end",
    }
    assert mot_extra == {
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

    planned_steps = sandbox._sequence_chunk_to_planned_steps(
        action_pred=action_pred,
        reference_obs={},
        generation_action_start=7,
        source="test_plan",
        planner_step_index=3,
        ready_monotonic_s=12.5,
        action_target_representation=sandbox.ActionTargetRepresentation.RAW,
        rotation_representation="axis_angle",
    )
    materialized = sandbox._materialize_sequence_control_action(
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


def test_decoder_current_action_commits_configured_rollout_chunk() -> None:
    sandbox = _load_sandbox_module()
    decoder_output = SimpleNamespace(
        action_pred=sandbox.torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        next_state=SimpleNamespace(step_within_chunk=1, aux={"rollout_chunk_steps": 2}),
        aux={"current_action": sandbox.torch.tensor([[1.0, 2.0]])},
    )

    planned, metadata = sandbox._decoder_output_to_rollout_action_plan(decoder_output)

    assert metadata["action_plan_source"] == "decoder_current_action_rollout_chunk"
    assert metadata["decoder_rollout_commit_start_index"] == 0
    assert metadata["decoder_rollout_commit_end_index"] == 2
    assert metadata["decoder_rollout_committed_actions"] == 2
    assert planned.shape == (2, 2)
    np.testing.assert_allclose(planned, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_decoder_cached_current_action_commits_remaining_rollout_chunk() -> None:
    sandbox = _load_sandbox_module()
    decoder_output = SimpleNamespace(
        action_pred=sandbox.torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        next_state=SimpleNamespace(step_within_chunk=2, aux={"rollout_chunk_steps": 2}),
        aux={"current_action": sandbox.torch.tensor([[3.0, 4.0]])},
    )

    planned, metadata = sandbox._decoder_output_to_rollout_action_plan(decoder_output)

    assert metadata["action_plan_source"] == "decoder_current_action_rollout_chunk"
    assert metadata["decoder_rollout_commit_start_index"] == 1
    assert metadata["decoder_rollout_commit_end_index"] == 2
    assert metadata["decoder_rollout_committed_actions"] == 1
    assert planned.shape == (1, 2)
    np.testing.assert_allclose(planned, np.array([[3.0, 4.0]], dtype=np.float32))


def test_advance_decoder_state_to_rollout_commit_marks_planned_actions_consumed() -> None:
    sandbox = _load_sandbox_module()
    decoder_state = SimpleNamespace(step_within_chunk=1)
    session = SimpleNamespace(policy_state=SimpleNamespace(decoder_state=decoder_state))

    sandbox._advance_decoder_state_to_rollout_commit(
        session,
        {
            "decoder_rollout_commit_end_index": 2,
        },
    )

    assert decoder_state.step_within_chunk == 2


def test_decoder_rollout_plan_falls_back_to_full_chunk_without_current_action() -> None:
    sandbox = _load_sandbox_module()
    decoder_output = SimpleNamespace(
        action_pred=sandbox.torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        aux={},
    )

    planned, metadata = sandbox._decoder_output_to_rollout_action_plan(decoder_output)

    assert metadata["action_plan_source"] == "decoder_action_chunk"
    assert metadata["decoder_rollout_committed_actions"] == 2
    assert planned.shape == (2, 2)
    np.testing.assert_allclose(planned, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_loaded_mot_action_expert_is_marked_initialized() -> None:
    sandbox = _load_sandbox_module()
    pipeline = SimpleNamespace(policy_variant=SimpleNamespace(_action_expert_initialized=False))

    sandbox.video_viz._mark_loaded_lazy_components_initialized(
        pipeline,
        {"policy_variant.action_expert.layers.0.weight": sandbox.torch.ones(1)},
    )

    assert pipeline.policy_variant._action_expert_initialized is True


def test_missing_mot_action_expert_weights_are_not_marked_initialized() -> None:
    sandbox = _load_sandbox_module()
    pipeline = SimpleNamespace(policy_variant=SimpleNamespace(_action_expert_initialized=False))

    sandbox.video_viz._mark_loaded_lazy_components_initialized(
        pipeline,
        {"policy_variant.action_expert.layers.0.weight": sandbox.torch.ones(1)},
        missing_keys=["policy_variant.action_expert.layers.0.weight"],
    )

    assert pipeline.policy_variant._action_expert_initialized is False


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

    sandbox.exact_sandbox._apply_inference_overrides(
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
        sandbox.exact_sandbox._apply_inference_overrides(
            runner,
            video_num_inference_steps=video_steps,
            action_num_inference_steps=action_steps,
            guidance_scale=None,
            action_guidance_scale=None,
        )

    assert runner.policy_variant.inference_config.video_num_inference_steps == 20
    assert runner.policy_variant.inference_config.action_num_inference_steps == 50
