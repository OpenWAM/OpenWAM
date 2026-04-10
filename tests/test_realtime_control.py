from __future__ import annotations

import numpy as np

from open_wam.integrations.realtime_control import (
    build_live_rollout_summary,
    make_planned_frame_actions,
    merge_future_frame_actions,
    summarize_scalars,
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
