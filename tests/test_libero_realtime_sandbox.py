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

    assert list(merged) == list(range(12))
    assert merged[0].raw_action is not None
    np.testing.assert_allclose(merged[0].raw_action, np.array([4.0], dtype=np.float32))
    assert merged[11].raw_action is not None
    np.testing.assert_allclose(merged[11].raw_action, np.array([15.0], dtype=np.float32))


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


def test_sequence_raw_action_targets_materialize_without_pose_conversion() -> None:
    sandbox = _load_sandbox_module()
    action_pred = np.array(
        [
            [0.10, -0.20, 0.30, 0.40, -0.50, 0.60, -1.00],
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
    np.testing.assert_allclose(materialized, action_pred[0])


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
