from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import pytest


def _load_ablation_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "deprecated" / "run_libero_realtime_ablation.py"
    module_name = f"run_libero_realtime_ablation_test_{uuid.uuid4().hex}"
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


def test_parse_target_action_hz_csv_rejects_nonpositive_values() -> None:
    ablation = _load_ablation_module()

    assert ablation._parse_float_csv("1,2.5,20") == [1.0, 2.5, 20.0]

    try:
        ablation._parse_float_csv("1,0")
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected nonpositive target Hz to be rejected.")


def test_default_profiles_encode_blocking_and_live_semantics() -> None:
    ablation = _load_ablation_module()

    naive = ablation.DEFAULT_PROFILES["naive_blocking"]
    live = ablation.DEFAULT_PROFILES["live_async_hold"]
    history_first = ablation.DEFAULT_PROFILES["live_async_history_first_hold"]
    recommended = ablation.DEFAULT_PROFILES["live_async_history_first_startup_hold"]

    assert naive.planner_mode == "history_only"
    assert naive.sequence_empty_plan_policy == "wait_for_replan"
    assert live.planner_mode == "async_buffer"
    assert live.sequence_empty_plan_policy == "fallback"
    assert live.deadline_miss_policy == "hold_state"
    assert history_first.planner_mode == "async_history_first"
    assert history_first.sequence_empty_plan_policy == "fallback"
    assert recommended.planner_mode == "async_history_first"
    assert recommended.sequence_empty_plan_policy == "fallback"
    assert recommended.startup_open_loop_chunks == 1


def test_local_path_alias_expansion_supports_nested_aliases() -> None:
    ablation = _load_ablation_module()
    local_paths = {
        "checkpoints.root": "/tmp/checkpoints",
        "checkpoints.step": "/tmp/checkpoints/checkpoint_step_1",
    }

    assert (
        ablation._resolve_path_token("${paths.checkpoints.step}/transformer", local_path_registry=local_paths)
        == "/tmp/checkpoints/checkpoint_step_1/transformer"
    )


def test_load_local_path_registry_uses_shared_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    ablation = _load_ablation_module()
    expected = {"checkpoints.step": "/tmp/checkpoints/checkpoint_step_1"}
    monkeypatch.setattr(ablation, "load_local_path_registry", lambda: expected)

    assert ablation._load_local_path_registry() == expected


def test_resolve_path_token_reports_shared_registry_sources() -> None:
    ablation = _load_ablation_module()

    with pytest.raises(KeyError, match="OPEN_WAM_LOCAL_PATHS"):
        ablation._resolve_path_token(
            "${paths.checkpoints.missing}",
            local_path_registry={},
        )


def test_build_jobs_materializes_profile_flags_and_suffixes(tmp_path: Path) -> None:
    ablation = _load_ablation_module()
    case = ablation.RolloutCase(
        name="case_a",
        config=Path("configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"),
        checkpoint=None,
    )
    args = SimpleNamespace(
        benchmark="libero_10",
        task_id=0,
        episode_idx=0,
        max_actions=4,
        video_fps=15.0,
        output_dir=str(tmp_path),
        suffix="smoke",
        sequence_buffer_threshold=3,
        fallback_history_policy="include_fallback_history",
        replan_low_watermark_actions=0,
        reference_assets_device_policy="runtime",
        seed=0,
        runtime_device="cuda:0",
        runtime_devices=None,
        runtime_prep_device=None,
        runtime_output_device=None,
        frontend_device="cuda:0",
        decode_device="cuda:0",
        video_num_inference_steps=20,
        action_num_inference_steps=50,
        guidance_scale=None,
        action_guidance_scale=None,
        write_fallback_timeline_video=False,
    )

    jobs = ablation._build_jobs(
        cases=[case],
        profiles=[ablation.DEFAULT_PROFILES["naive_blocking"]],
        target_action_hz_values=[1.0, 10.0],
        args=args,
        local_path_registry={},
    )

    assert len(jobs) == 2
    command = jobs[0].command
    assert "--planner-mode" in command
    assert command[command.index("--planner-mode") + 1] == "history_only"
    assert command[command.index("--sequence-empty-plan-policy") + 1] == "wait_for_replan"
    assert command[command.index("--deadline-miss-policy") + 1] == "hold_state"
    assert command[command.index("--fallback-history-policy") + 1] == "include_fallback_history"
    assert command[command.index("--replan-low-watermark-actions") + 1] == "0"
    assert command[command.index("--suffix") + 1] == "smoke_case_a_naive_blocking_1hz"
    assert "--write-fallback-timeline-video" not in command


def test_build_jobs_appends_nondefault_fallback_history_policy_to_suffix(tmp_path: Path) -> None:
    ablation = _load_ablation_module()
    case = ablation.RolloutCase(
        name="case_a",
        config=Path("configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"),
        checkpoint=None,
    )
    args = SimpleNamespace(
        benchmark="libero_10",
        task_id=0,
        episode_idx=0,
        max_actions=4,
        video_fps=15.0,
        output_dir=str(tmp_path),
        suffix="smoke",
        sequence_buffer_threshold=3,
        fallback_history_policy="freeze_until_clean_chunk",
        replan_low_watermark_actions=10,
        reference_assets_device_policy="runtime",
        seed=0,
        runtime_device="cuda:0",
        runtime_devices=None,
        runtime_prep_device=None,
        runtime_output_device=None,
        frontend_device="cuda:0",
        decode_device="cuda:0",
        video_num_inference_steps=20,
        action_num_inference_steps=50,
        guidance_scale=None,
        action_guidance_scale=None,
        write_fallback_timeline_video=True,
    )

    jobs = ablation._build_jobs(
        cases=[case],
        profiles=[ablation.DEFAULT_PROFILES["live_async_hold"]],
        target_action_hz_values=[2.0],
        args=args,
        local_path_registry={},
    )

    assert len(jobs) == 1
    command = jobs[0].command
    assert command[command.index("--fallback-history-policy") + 1] == "freeze_until_clean_chunk"
    assert command[command.index("--replan-low-watermark-actions") + 1] == "10"
    assert "--write-fallback-timeline-video" in command
    assert (
        command[command.index("--suffix") + 1]
        == "smoke_case_a_live_async_hold_2hz_freeze_until_clean_chunk_lowwatermark_10a"
    )


def test_build_jobs_resolves_auto_low_watermark_per_verified_case(tmp_path: Path) -> None:
    ablation = _load_ablation_module()
    args = SimpleNamespace(
        benchmark="libero_10",
        task_id=0,
        episode_idx=0,
        max_actions=4,
        video_fps=15.0,
        output_dir=str(tmp_path),
        suffix="recommended",
        sequence_buffer_threshold=3,
        fallback_history_policy="freeze_until_clean_chunk",
        replan_low_watermark_actions="auto",
        reference_assets_device_policy="runtime",
        seed=0,
        runtime_device="cuda:0",
        runtime_devices=None,
        runtime_prep_device=None,
        runtime_output_device=None,
        frontend_device="cuda:0",
        decode_device="cuda:0",
        video_num_inference_steps=None,
        action_num_inference_steps=None,
        guidance_scale=None,
        action_guidance_scale=None,
        write_fallback_timeline_video=False,
    )

    jobs = ablation._build_jobs(
        cases=[
            ablation.DEFAULT_CASES["method1_exact_step400"],
            ablation.DEFAULT_CASES["method2_joint_step600"],
        ],
        profiles=[ablation.DEFAULT_PROFILES["live_async_history_first_startup_hold"]],
        target_action_hz_values=[2.0],
        args=args,
        local_path_registry={
            "checkpoints.parallel_stream_exact_libero_step_400": "/tmp/m1",
            "checkpoints.parallel_stream_joint_libero_step_600_0402": "/tmp/m2",
        },
    )

    assert len(jobs) == 2
    method1_command = jobs[0].command
    method2_command = jobs[1].command
    assert method1_command[method1_command.index("--startup-open-loop-chunks") + 1] == "1"
    assert method1_command[method1_command.index("--replan-low-watermark-actions") + 1] == "8"
    assert method2_command[method2_command.index("--replan-low-watermark-actions") + 1] == "12"
    assert "lowwatermark_8a" in method1_command[method1_command.index("--suffix") + 1]
    assert "lowwatermark_12a" in method2_command[method2_command.index("--suffix") + 1]


def test_auto_low_watermark_rejects_unverified_cases() -> None:
    ablation = _load_ablation_module()

    with pytest.raises(ValueError, match="only defined for verified built-in cases"):
        ablation._resolve_case_replan_low_watermark_actions(
            case=ablation.DEFAULT_CASES["method3_vsp_step800"],
            spec="auto",
        )


def test_extract_last_json_object_prefers_rollout_summary() -> None:
    ablation = _load_ablation_module()
    text = """
    {"phase": "load"}
    {
      "target_action_hz": 10.0,
      "replan_infer_s": {"count": 0},
      "executed_actions": 8,
      "summary_path": "/tmp/summary.json"
    }
    """

    summary = ablation._extract_last_json_object(text)

    assert summary is not None
    assert summary["executed_actions"] == 8
    assert summary["summary_path"] == "/tmp/summary.json"
