from __future__ import annotations

import numpy as np

import open_wam.integrations as integrations
from open_wam.configs import (
    FallbackHistoryPolicy,
    RealtimeEmptyPlanPolicy,
    RealtimePlannerJob,
    RealtimePlannerMode,
    RealtimeSchedulerProfile,
)
from open_wam.integrations.realtime_control import (
    PlannedControlStep,
    PlannedFrameAction,
    build_live_rollout_summary,
    drop_control_steps_from,
    drop_partial_stale_control_chunk,
    frame_index_to_action_start,
    future_control_depth,
    future_control_steps,
    make_planned_frame_actions,
    merge_future_control_steps,
    merge_future_frame_actions,
    missing_control_action_indices,
    planned_frame_actions_to_control_steps,
    required_control_action_indices,
    resolve_realtime_planner_mode,
    resolve_realtime_scheduler_defaults,
    select_realtime_planner_job,
    should_submit_frame_grouped_planner,
    should_submit_realtime_planner_job,
    should_submit_sequence_planner,
    summarize_scalars,
)


def _control_step(
    action_index: int,
    *,
    generation_action_start: int = 0,
    source: str = "plan",
) -> PlannedControlStep:
    return PlannedControlStep(
        absolute_action_index=action_index,
        generation_action_start=generation_action_start,
        source=source,
    )


def test_realtime_control_contract_is_lazily_exported_by_integrations() -> None:
    assert integrations.PlannedControlStep is PlannedControlStep
    assert integrations.merge_future_control_steps is merge_future_control_steps


def test_realtime_scheduler_profiles_resolve_to_typed_optional_overrides() -> None:
    manual = resolve_realtime_scheduler_defaults(RealtimeSchedulerProfile.MANUAL)
    blocking = resolve_realtime_scheduler_defaults("blocking_control")
    frozen = resolve_realtime_scheduler_defaults("freeze_until_clean_chunk")
    asynchronous = resolve_realtime_scheduler_defaults("async_history_first")

    assert manual.to_override_mapping() == {}
    assert blocking.planner_mode == RealtimePlannerMode.HISTORY_ONLY
    assert blocking.empty_plan_policy == RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN
    assert blocking.fallback_history_policy == FallbackHistoryPolicy.INCLUDE_FALLBACK_HISTORY
    assert blocking.replan_low_watermark_actions == 0
    assert frozen.fallback_history_policy == FallbackHistoryPolicy.FREEZE_UNTIL_CLEAN_CHUNK
    assert asynchronous.planner_mode == RealtimePlannerMode.ASYNC_HISTORY_FIRST
    assert asynchronous.startup_open_loop_chunks == 1
    assert "replan_low_watermark_actions" not in asynchronous.to_override_mapping()


def test_realtime_planner_job_selection_preserves_mode_priority() -> None:
    select = select_realtime_planner_job

    assert select(
        planner_mode="history_only",
        history_count=0,
        future_buffer_depth=0,
        has_buffer_tail_session=True,
    ) is None
    assert select(
        planner_mode="history_only",
        history_count=1,
        future_buffer_depth=7,
        has_buffer_tail_session=False,
    ) == RealtimePlannerJob.HISTORY_REPLAN
    assert select(
        planner_mode="async_buffer",
        history_count=1,
        future_buffer_depth=3,
        has_buffer_tail_session=True,
    ) == RealtimePlannerJob.BUFFER_EXTENSION
    assert select(
        planner_mode="async_buffer",
        history_count=1,
        future_buffer_depth=4,
        has_buffer_tail_session=True,
    ) == RealtimePlannerJob.HISTORY_REPLAN
    assert select(
        planner_mode="async_history_first",
        history_count=1,
        future_buffer_depth=0,
        has_buffer_tail_session=True,
    ) == RealtimePlannerJob.HISTORY_REPLAN
    assert select(
        planner_mode="async_mix",
        history_count=1,
        future_buffer_depth=3,
        has_buffer_tail_session=True,
    ) == RealtimePlannerJob.BUFFER_EXTENSION
    assert select(
        planner_mode="async_mix",
        history_count=2,
        future_buffer_depth=2,
        has_buffer_tail_session=True,
    ) == RealtimePlannerJob.HISTORY_REPLAN
    assert should_submit_realtime_planner_job(
        planner_mode="async_buffer",
        history_count=0,
        future_buffer_depth=6,
        has_buffer_tail_session=True,
    )


def test_realtime_outer_scheduler_gates_preserve_blocking_and_low_watermarks() -> None:
    assert resolve_realtime_planner_mode(
        planner_mode="async_buffer",
        empty_plan_policy="wait_for_replan",
        has_history=True,
    ) == RealtimePlannerMode.HISTORY_ONLY
    assert resolve_realtime_planner_mode(
        planner_mode="async_buffer",
        empty_plan_policy="wait_for_replan",
        has_history=False,
    ) == RealtimePlannerMode.ASYNC_BUFFER
    assert should_submit_frame_grouped_planner(
        future_buffer_depth_actions=0,
        future_buffer_depth_frames=0,
        empty_plan_policy="wait_for_replan",
    )
    assert not should_submit_frame_grouped_planner(
        future_buffer_depth_actions=8,
        future_buffer_depth_frames=2,
        empty_plan_policy="wait_for_replan",
    )
    assert should_submit_frame_grouped_planner(
        future_buffer_depth_actions=10,
        future_buffer_depth_frames=3,
        empty_plan_policy="fallback",
        replan_low_watermark_actions=10,
    )
    assert not should_submit_frame_grouped_planner(
        future_buffer_depth_actions=11,
        future_buffer_depth_frames=3,
        empty_plan_policy="fallback",
        replan_low_watermark_actions=10,
    )
    assert should_submit_sequence_planner(
        planner_mode="async_history_first",
        future_buffer_depth_actions=12,
        empty_plan_policy="fallback",
        sequence_buffer_threshold=3,
        replan_low_watermark_actions=12,
    )
    assert not should_submit_sequence_planner(
        planner_mode="async_history_first",
        future_buffer_depth_actions=13,
        empty_plan_policy="fallback",
        sequence_buffer_threshold=3,
        replan_low_watermark_actions=12,
    )
    assert not should_submit_sequence_planner(
        planner_mode="history_only",
        future_buffer_depth_actions=1,
        empty_plan_policy="fallback",
        sequence_buffer_threshold=3,
        replan_low_watermark_actions=12,
    )


def test_make_planned_frame_actions_assigns_absolute_frame_ids() -> None:
    frame_actions = np.zeros((3, 4, 7), dtype=np.float32)
    planned = make_planned_frame_actions(
        frame_actions,
        generation_frame_start=5,
        source="startup_plan",
        planner_step_index=9,
        ready_monotonic_s=1.25,
    )

    assert [plan.absolute_frame_index for plan in planned] == [5, 6, 7]
    assert [plan.frame_offset for plan in planned] == [0, 1, 2]
    assert all(plan.source == "startup_plan" for plan in planned)
    assert all(plan.planner_step_index == 9 for plan in planned)
    assert all(plan.ready_monotonic_s == 1.25 for plan in planned)


def test_merge_future_frame_actions_drops_stale_and_replaces_future() -> None:
    existing = {
        plan.absolute_frame_index: plan
        for plan in make_planned_frame_actions(
            np.array(
                [
                    np.full((4, 7), 1.0, dtype=np.float32),
                    np.full((4, 7), 2.0, dtype=np.float32),
                    np.full((4, 7), 3.0, dtype=np.float32),
                ],
                dtype=np.float32,
            ),
            generation_frame_start=1,
        )
    }
    incoming = make_planned_frame_actions(
        np.array(
            [
                np.full((4, 7), 20.0, dtype=np.float32),
                np.full((4, 7), 30.0, dtype=np.float32),
                np.full((4, 7), 40.0, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
        generation_frame_start=2,
    )

    merged = merge_future_frame_actions(existing, incoming, next_frame_to_execute=3)

    assert list(merged) == [3, 4]
    assert float(merged[3].raw_actions[0, 0]) == 30.0
    assert float(merged[4].raw_actions[0, 0]) == 40.0


def test_frame_index_to_action_start_uses_conditioning_frame_convention() -> None:
    assert [frame_index_to_action_start(index, 4) for index in range(5)] == [0, 0, 4, 8, 12]


def test_planned_frame_actions_expand_to_independent_control_steps() -> None:
    source_actions = np.asarray(
        [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]],
        dtype=np.float64,
    )
    planned_frame = PlannedFrameAction(
        absolute_frame_index=3,
        generation_frame_start=2,
        frame_offset=1,
        raw_actions=source_actions,
        source="frame_plan",
        planner_step_index=7,
        ready_monotonic_s=1.25,
    )

    planned_steps = planned_frame_actions_to_control_steps([planned_frame])

    assert [step.absolute_action_index for step in planned_steps] == [8, 9, 10, 11]
    assert [step.generation_action_start for step in planned_steps] == [4, 4, 4, 4]
    assert all(step.generation_frame_start == 2 for step in planned_steps)
    assert all(step.source == "frame_plan" for step in planned_steps)
    assert all(step.planner_step_index == 7 for step in planned_steps)
    assert all(step.ready_monotonic_s == 1.25 for step in planned_steps)
    assert all(step.raw_action is not None and step.raw_action.dtype == np.float32 for step in planned_steps)
    np.testing.assert_array_equal(planned_steps[0].raw_action, np.asarray([10.0, 11.0], dtype=np.float32))
    source_actions[0, 0] = -1.0
    assert planned_steps[0].raw_action is not None
    assert float(planned_steps[0].raw_action[0]) == 10.0


def test_merge_future_control_steps_drops_stale_and_prefers_newer_predictions() -> None:
    existing = {
        0: _control_step(0, source="old"),
        1: _control_step(1, source="old"),
        2: _control_step(2, source="old"),
    }
    incoming = [
        _control_step(1, generation_action_start=1, source="new"),
        _control_step(3, generation_action_start=1, source="new"),
    ]

    merged = merge_future_control_steps(existing, incoming, next_action_to_execute=1)

    assert list(merged) == [1, 2, 3]
    assert merged[1].source == "new"
    assert merged[2].source == "old"
    assert merged[3].source == "new"


def test_control_plan_queries_respect_rollout_cursor_and_limit() -> None:
    plan = {
        4: _control_step(4),
        6: _control_step(6),
    }

    required = required_control_action_indices(
        next_action_index=8,
        max_actions=10,
        action_per_frame=4,
    )

    assert required == [8, 9]
    assert missing_control_action_indices(plan, [4, 5, 6, 7]) == [5, 7]
    assert future_control_depth(plan, next_action_to_execute=4) == 3
    assert future_control_depth({}, next_action_to_execute=4) == 0


def test_control_plan_slices_steps_at_replacement_and_execution_boundaries() -> None:
    planned_steps = [_control_step(index) for index in range(8)]
    plan_by_action = {step.absolute_action_index: step for step in planned_steps}

    future = future_control_steps(planned_steps, next_action_to_execute=5)
    prefix = drop_control_steps_from(plan_by_action, replace_from_action=5)

    assert [step.absolute_action_index for step in future] == [5, 6, 7]
    assert list(prefix) == [0, 1, 2, 3, 4]


def test_partial_stale_control_chunks_are_atomic_unless_suffix_is_large_enough() -> None:
    planned_steps = [_control_step(index, source="history_replan") for index in range(16)]

    rejected, dropped, accepted_partial = drop_partial_stale_control_chunk(
        planned_steps,
        next_action_to_execute=8,
    )
    accepted, accepted_dropped, accepted_partial_count = drop_partial_stale_control_chunk(
        planned_steps,
        next_action_to_execute=8,
        min_future_actions_to_accept_stale_chunk=8,
    )
    fully_stale, fully_stale_dropped, fully_stale_partial = drop_partial_stale_control_chunk(
        planned_steps[:4],
        next_action_to_execute=8,
    )

    assert rejected == []
    assert dropped == 8
    assert accepted_partial == 0
    assert [step.absolute_action_index for step in accepted] == list(range(8, 16))
    assert accepted_dropped == 0
    assert accepted_partial_count == 8
    assert fully_stale == planned_steps[:4]
    assert fully_stale_dropped == 0
    assert fully_stale_partial == 0


def test_build_live_rollout_summary_reports_rates_and_stage_stats() -> None:
    action_records = [
        {
            "absolute_frame_index": 1,
            "source": "startup_plan",
            "lateness_s": 0.001,
            "env_step_s": 0.020,
            "generation_lag_frames": 1,
        },
        {
            "absolute_frame_index": 1,
            "source": "history_replan",
            "lateness_s": 0.003,
            "env_step_s": 0.021,
            "generation_lag_frames": 1,
        },
        {
            "absolute_frame_index": 2,
            "source": "open_loop_extension",
            "lateness_s": 0.001,
            "env_step_s": 0.019,
            "generation_lag_frames": 2,
        },
        {
            "absolute_frame_index": 2,
            "source": "fallback_hold_last",
            "lateness_s": 0.002,
            "env_step_s": 0.020,
            "generation_lag_frames": None,
        },
    ]
    replan_records = [
        {
            "prepare_s": 0.010,
            "warmup_s": 0.020,
            "infer_s": 0.030,
            "total_latency_s": 0.060,
        },
        {
            "prepare_s": 0.011,
            "warmup_s": 0.019,
            "infer_s": 0.031,
            "total_latency_s": 0.061,
        },
    ]

    summary = build_live_rollout_summary(
        action_records=action_records,
        replan_records=replan_records,
        target_action_hz=10.0,
        live_wall_time_s=0.4,
        startup_prepare_s=2.0,
        startup_infer_s=0.3,
        deadline_tolerance_s=0.002,
    )

    assert summary["total_actions"] == 4
    assert summary["planned_actions"] == 3
    assert summary["startup_plan_actions"] == 1
    assert summary["history_replan_actions"] == 1
    assert summary["observation_conditioned_actions"] == 2
    assert summary["open_loop_extension_actions"] == 1
    assert summary["fallback_actions"] == 1
    assert summary["total_frames"] == 2
    assert summary["total_action_steps"] == 4
    assert summary["achieved_action_hz"] == 10.0
    assert summary["planned_action_hz"] == 7.5
    assert summary["observation_conditioned_action_hz"] == 5.0
    assert summary["open_loop_extension_action_hz"] == 2.5
    assert summary["deadline_hit_rate"] == 0.75
    assert summary["action_lateness_s"]["count"] == 4
    assert summary["generation_lag_actions"]["count"] == 0
    assert summary["generation_lag_frames"]["count"] == 3
    assert summary["replan_total_latency_s"]["count"] == 2
    assert summary["replan_infer_s"]["max"] == 0.031


def test_build_live_rollout_summary_accepts_action_aligned_records() -> None:
    action_records = [
        {
            "absolute_action_index": 0,
            "absolute_frame_index": None,
            "source": "startup_plan",
            "lateness_s": 0.001,
            "env_step_s": 0.020,
            "generation_lag_actions": 0,
            "generation_lag_frames": None,
        },
        {
            "absolute_action_index": 1,
            "absolute_frame_index": None,
            "source": "history_replan",
            "lateness_s": 0.001,
            "env_step_s": 0.021,
            "generation_lag_actions": 1,
            "generation_lag_frames": None,
        },
    ]

    summary = build_live_rollout_summary(
        action_records=action_records,
        replan_records=[],
        target_action_hz=10.0,
        live_wall_time_s=0.2,
        startup_prepare_s=1.0,
        startup_infer_s=0.1,
    )

    assert summary["total_actions"] == 2
    assert summary["total_frames"] == 0
    assert summary["total_action_steps"] == 2
    assert summary["generation_lag_actions"]["count"] == 2
    assert summary["generation_lag_frames"]["count"] == 0


def test_summarize_scalars_accepts_numpy_arrays() -> None:
    summary = summarize_scalars(np.asarray([1.0, 2.0, 3.0], dtype=np.float64))

    assert summary["count"] == 3
    assert summary["mean"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
