from __future__ import annotations
from open_wam.configs import ExperimentConfig, InferenceConfig

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from open_wam.configs import (
    ActionTargetRepresentation,
    ActionMappingConfig,
    ActionNormalizationConfig,
)
from open_wam.models.action_decoders import ActionDecoder


class _RealtimeTestActionDecoder(ActionDecoder):
    def __init__(self, *, rollout_chunk_steps: int) -> None:
        super().__init__()
        self.rollout_chunk_steps = int(rollout_chunk_steps)

    def forward_train(self, policy_output, batch):
        raise NotImplementedError

    def forward_infer(self, policy_output, previous_state=None):
        raise NotImplementedError


def _load_sandbox_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_libero_realtime_sandbox.py"
    )
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


def _build_exact_history_frame_payload() -> tuple[
    dict[str, np.ndarray], list[dict[str, np.ndarray]], list[np.ndarray]
]:
    current_obs = {"image": np.full((1, 1, 3), 9, dtype=np.uint8)}
    frame_obs_sequence = [
        {"image": np.full((1, 1, 3), 2 + index, dtype=np.uint8)} for index in range(4)
    ]
    frame_actions = [
        np.full((7,), float(index), dtype=np.float32) for index in range(4)
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
                normalization=ActionNormalizationConfig(),
                rotation_representation="axis_angle",
                gripper_representation="action_command",
            ),
            action_mapping=ActionMappingConfig(),
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
        startup_open_loop_chunks=0,
        replan_low_watermark_actions=0,
    )

    sandbox._apply_realtime_cli_profiles(args, [])

    assert args.max_actions == 3000
    assert args.env_horizon == 5000
    assert args.video_fps is None
    assert (
        args.realtime_scheduler_profile
        is sandbox.RealtimeSchedulerProfile.BLOCKING_CONTROL
    )
    assert args.planner_mode is sandbox.RealtimePlannerMode.HISTORY_ONLY
    assert (
        args.sequence_empty_plan_policy
        is sandbox.RealtimeEmptyPlanPolicy.WAIT_FOR_REPLAN
    )


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
        startup_open_loop_chunks=0,
        replan_low_watermark_actions=7,
    )

    sandbox._apply_realtime_cli_profiles(
        args,
        [
            "--max-actions",
            "120",
            "--planner-mode",
            "history_only",
            "--replan-low-watermark-actions",
            "7",
        ],
    )

    assert args.max_actions == 120
    assert (
        args.realtime_scheduler_profile
        is sandbox.RealtimeSchedulerProfile.ASYNC_HISTORY_FIRST
    )
    assert args.planner_mode is sandbox.RealtimePlannerMode.HISTORY_ONLY
    assert args.replan_low_watermark_actions == 7
    assert args.startup_open_loop_chunks == 1


def test_finalize_rollout_outputs_lean_skips_videos_and_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _load_sandbox_module()

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "lean artifact profile should not render videos or trace files"
        )

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
    assert (
        json.loads(summary_path.read_text(encoding="utf-8"))["artifact_profile"]
        == "lean"
    )
    assert "video_path" not in summary
    assert "action_trace_path" not in summary
    assert not list(tmp_path.rglob("*.mp4"))
    assert not list(tmp_path.rglob("*_actions.jsonl"))
    assert not list(tmp_path.rglob("*_replans.jsonl"))
    assert not list(tmp_path.rglob("*_extensions.jsonl"))
    assert not list(tmp_path.rglob("*_load_report.json"))


def test_realtime_common_inference_overrides_preserve_config_values_by_default() -> (
    None
):
    sandbox = _load_sandbox_module()
    config = ExperimentConfig(
        inference=InferenceConfig(
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        )
    )

    updated = sandbox._apply_common_inference_overrides(
        config,
        video_num_inference_steps=None,
        action_num_inference_steps=None,
        guidance_scale=None,
        action_guidance_scale=None,
    )

    assert updated.inference.video_num_inference_steps == 20
    assert updated.inference.action_num_inference_steps == 50
    assert updated.inference.guidance_scale == 5.0
    assert updated.inference.action_guidance_scale == 1.0


def test_realtime_common_inference_overrides_apply_explicit_smoke_values() -> None:
    sandbox = _load_sandbox_module()
    config = ExperimentConfig(
        inference=InferenceConfig(
            video_num_inference_steps=20,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        )
    )

    updated = sandbox._apply_common_inference_overrides(
        config,
        video_num_inference_steps=2,
        action_num_inference_steps=3,
        guidance_scale=1.5,
        action_guidance_scale=0.75,
    )

    assert updated.inference.video_num_inference_steps == 2
    assert updated.inference.action_num_inference_steps == 3
    assert updated.inference.guidance_scale == 1.5
    assert updated.inference.action_guidance_scale == 0.75


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
    config = ExperimentConfig(
        inference=InferenceConfig(
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
