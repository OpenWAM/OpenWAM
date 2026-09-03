from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import (
    DualExpertPolicyConfig,
    ParallelStreamPolicyConfig,
    VideoActionProgram,
)
from open_wam.evals import libero_dual_expert_runtime as runtime
from open_wam.models.policy_variants import PolicyOutputModality


def test_generic_video_producer_is_only_admitted_for_explicit_producer_role() -> None:
    config = SimpleNamespace(
        policy_variant=SimpleNamespace(name="causal_video_prediction")
    )

    with pytest.raises(ValueError, match="requires a `dual_expert` policy"):
        runtime._validate_libero_policy_runtime_config(
            config,
            runtime_role=runtime.LiberoPolicyRuntimeRole.NATIVE_POLICY,
        )

    runtime._validate_libero_policy_runtime_config(
        config,
        runtime_role=runtime.LiberoPolicyRuntimeRole.VIDEO_PRODUCER,
    )


def test_conditional_consumer_role_requires_declared_clean_video() -> None:
    with pytest.raises(ValueError, match="requires provided conditioning modalities"):
        runtime._validate_runtime_role_inputs(
            runtime.LiberoPolicyRuntimeRole.VIDEO_CONDITIONED_ACTION_CONSUMER,
            action_route=runtime.DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION,
            provided_modalities=(),
        )

    runtime._validate_runtime_role_inputs(
        runtime.LiberoPolicyRuntimeRole.VIDEO_CONDITIONED_ACTION_CONSUMER,
        action_route=runtime.DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION,
        provided_modalities=(PolicyOutputModality.VIDEO,),
    )


@pytest.mark.parametrize(
    "runtime_role",
    (
        runtime.LiberoPolicyRuntimeRole.NATIVE_POLICY,
        runtime.LiberoPolicyRuntimeRole.VIDEO_PRODUCER,
    ),
)
def test_only_action_consumer_may_declare_provided_conditioning_inputs(
    runtime_role: runtime.LiberoPolicyRuntimeRole,
) -> None:
    action_route = (
        runtime.DualExpertActionRoute.JOINT
        if runtime_role is runtime.LiberoPolicyRuntimeRole.NATIVE_POLICY
        else runtime.DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION
    )
    with pytest.raises(
        ValueError,
        match="Only the video-conditioned action consumer",
    ):
        runtime._validate_runtime_role_inputs(
            runtime_role,
            action_route=action_route,
            provided_modalities=(PolicyOutputModality.VIDEO,),
        )


def test_runtime_role_must_match_action_route() -> None:
    with pytest.raises(ValueError, match="requires an explicit video-producer"):
        runtime._validate_runtime_role_inputs(
            runtime.LiberoPolicyRuntimeRole.NATIVE_POLICY,
            action_route=runtime.DualExpertActionRoute.GENERATED_VIDEO_THEN_ACTION,
            provided_modalities=(),
        )
    with pytest.raises(ValueError, match="requires the generated-video action"):
        runtime._validate_runtime_role_inputs(
            runtime.LiberoPolicyRuntimeRole.VIDEO_PRODUCER,
            action_route=runtime.DualExpertActionRoute.JOINT,
            provided_modalities=(),
        )


def test_live_sim_rejects_standalone_conditional_program_default() -> None:
    with pytest.raises(ValueError, match="action_conditioned_video.*offline diagnostic program"):
        runtime._validate_live_sim_dynamics_program(
            DualExpertPolicyConfig(program=VideoActionProgram.FORWARD_DYNAMICS),
        )


def test_live_sim_rejects_generic_producer_that_needs_clean_actions() -> None:
    with pytest.raises(
        ValueError,
        match="action_conditioned_video.*offline diagnostic program",
    ):
        runtime._validate_live_sim_dynamics_program(
            ParallelStreamPolicyConfig(program=VideoActionProgram.FORWARD_DYNAMICS),
        )


def test_live_sim_does_not_infer_rollout_mode_from_training_routes() -> None:
    runtime._validate_live_sim_dynamics_program(
        DualExpertPolicyConfig(
            program=VideoActionProgram.GENERALIST_JOINT_DENOISING
        ),
    )


def test_live_sim_allows_fixed_idm_only_when_clean_video_is_provided() -> None:
    policy = DualExpertPolicyConfig(program=VideoActionProgram.INVERSE_DYNAMICS)

    with pytest.raises(ValueError, match="video_conditioned_action"):
        runtime._validate_live_sim_dynamics_program(policy)

    runtime._validate_live_sim_dynamics_program(
        policy,
        provided_modalities=(PolicyOutputModality.VIDEO,),
    )


class _FakePipeline:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.policy_variant = SimpleNamespace()
        self.training = True

    def to(self, *, device: torch.device):
        self.calls.append(("pipeline.to", str(device)))
        return self

    def eval(self):
        self.calls.append("pipeline.eval")
        self.training = False
        return self


def test_load_dual_expert_libero_runtime_preserves_composition_order_and_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    config_path = tmp_path / "resolved_config.yaml"
    checkpoint_path = tmp_path / "checkpoint_step_10" / "model_state.pt"
    config = SimpleNamespace(
        policy_variant=DualExpertPolicyConfig(
            program=VideoActionProgram.VIDEO_THEN_ACTION
        ),
        backbone=SimpleNamespace(
            runtime_backbone_artifact_path="/unused/transformer"
        ),
        data=SimpleNamespace(
            num_frames=4,
            action_schema=SimpleNamespace(action_horizon=16),
        ),
    )
    pipeline = _FakePipeline(calls)
    runner = object()

    def _load_config(path: Path, **kwargs):
        calls.append(("load_config", path, kwargs))
        return config

    def _resolve_checkpoint(**kwargs):
        calls.append(("resolve_checkpoint", kwargs))
        return checkpoint_path

    def _require_paradigm(value, **kwargs):
        calls.append(("require_paradigm", value, kwargs))

    def _build_pipeline(value):
        calls.append(("build_pipeline", value))
        return pipeline

    def _load_checkpoint(value, path, **kwargs):
        calls.append(("load_checkpoint", value, path, kwargs))
        return SimpleNamespace(missing_keys=(), unexpected_keys=())

    def _ensure_backend(value, cfg):
        calls.append(("ensure_backend", value, cfg))
        return {
            "block_restore_performed": False,
            "route": "split_cache",
        }

    def _build_runner(value):
        calls.append(("build_runner", value))
        return runner

    def _component_report(value, model, **kwargs):
        calls.append(("component_report", value, model, kwargs))
        return {"base": "report"}

    def _log(label: str, payload: dict[str, object]) -> None:
        calls.append(("log", label, payload.copy()))

    monkeypatch.setattr(runtime, "load_experiment_config", _load_config)
    monkeypatch.setattr(runtime, "_resolve_dual_expert_checkpoint_path", _resolve_checkpoint)
    monkeypatch.setattr(runtime, "require_current_libero_policy_paradigm", _require_paradigm)
    monkeypatch.setattr(runtime, "build_variant_pipeline_from_config", _build_pipeline)
    monkeypatch.setattr(runtime, "load_pipeline_checkpoint", _load_checkpoint)
    monkeypatch.setattr(runtime, "ensure_dual_expert_inference_backend", _ensure_backend)
    monkeypatch.setattr(runtime, "VariantRolloutRunner", _build_runner)
    monkeypatch.setattr(runtime, "_build_component_report", _component_report)
    monkeypatch.setattr(runtime, "_print_log", _log)

    loaded = runtime.load_dual_expert_libero_runtime(
        runtime.DualExpertLiberoLoadOptions(
            config=config_path,
            checkpoint=checkpoint_path,
            merge_checkpoint_runtime_config=False,
            set_overrides=(),
            source="test loader",
            checkpoint_error="checkpoint required",
            raw_window_frames=13,
            startup_model_obs_frames=1,
            startup_env_init_steps=5,
            dual_expert_inference_window_size=30,
            dual_expert_rollout_frame_chunk_size=None,
            dual_expert_action_only_rollout=False,
            dual_expert_gjd_action_route="joint",
            execute_action_steps=None,
            execute_frame_chunk_size=None,
            frontend_encode_mode=runtime.CURRENT_FRONTEND_ENCODE_MODE,
            reset_policy_state_each_chunk=False,
            runtime_device="cpu",
            action_device="cpu",
            frontend_device="cpu",
            decode_device="cpu",
            allow_deprecated_libero_config=False,
            allow_deprecated_frontend_encode_mode=False,
            component_report_extra={
                "caller": "single",
                "runtime_role": "must_not_override_authoritative_role",
            },
        )
    )

    assert loaded.config is config
    assert loaded.checkpoint_path == checkpoint_path
    assert loaded.pipeline is pipeline
    assert loaded.runner is runner
    assert loaded.raw_window_frames == 13
    assert loaded.startup_model_obs_frames == 1
    assert loaded.startup_env_init_steps == 5
    assert loaded.use_lingbot_streaming_vae is True
    assert loaded.runtime_device == torch.device("cpu")
    assert loaded.action_device == torch.device("cpu")
    assert loaded.frontend_device == torch.device("cpu")
    assert loaded.decode_device == torch.device("cpu")
    assert loaded.component_report == {
        "base": "report",
        "dual_expert_inference_backend": {
            "block_restore_performed": False,
            "route": "split_cache",
        },
        "checkpoint_file": str(checkpoint_path.resolve()),
        "checkpoint_runtime_config_path": None,
        "checkpoint_runtime_config_merged": False,
        "pipeline_training_mode": False,
        "runtime_role": runtime.LiberoPolicyRuntimeRole.NATIVE_POLICY.value,
        "frontend_encode_mode": runtime.CURRENT_FRONTEND_ENCODE_MODE,
        "dual_expert_rollout_frame_chunk_size": None,
        "execute_action_steps": None,
        "execute_frame_chunk_size": None,
        "caller": "single",
    }
    assert [
        entry[0] if isinstance(entry, tuple) else entry
        for entry in calls
    ] == [
        "load_config",
        "resolve_checkpoint",
        "require_paradigm",
        "build_pipeline",
        "load_checkpoint",
        "pipeline.to",
        "ensure_backend",
        "pipeline.eval",
        "build_runner",
        "component_report",
        "log",
    ]
    assert calls[-1][1] == "load_report"
    assert calls[-1][2] == loaded.component_report
    assert calls[0] == (
        "load_config",
        config_path,
        {"checkpoint_runtime_compat": False},
    )
