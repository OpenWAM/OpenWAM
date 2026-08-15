from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import GeneralistDenoisingMode
from open_wam.evals import libero_dual_expert_runtime as runtime


def test_live_sim_rejects_standalone_conditional_program_default() -> None:
    with pytest.raises(ValueError, match="fixed conditional mode.*offline diagnostic mode"):
        runtime._validate_live_sim_dual_expert_generalist_rollout_mode(
            None,
            policy_config=SimpleNamespace(program="forward_dynamics"),
        )


def test_live_sim_cannot_override_standalone_conditional_program_to_joint() -> None:
    with pytest.raises(ValueError, match="fixed conditional mode.*offline diagnostic mode"):
        runtime._validate_live_sim_dual_expert_generalist_rollout_mode(
            "joint",
            policy_config=SimpleNamespace(program="forward_dynamics"),
        )


def test_live_sim_rejects_one_hot_gjd_conditional_default() -> None:
    with pytest.raises(ValueError, match="fixed conditional mode.*offline diagnostic mode"):
        runtime._validate_live_sim_dual_expert_generalist_rollout_mode(
            None,
            policy_config=SimpleNamespace(
                program="generalist_joint_denoising",
                generalist_denoising_mode_probs={
                    GeneralistDenoisingMode.JOINT: 0.0,
                    GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO: 1.0,
                    GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION: 0.0,
                },
            ),
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
        policy_variant=SimpleNamespace(name="dual_expert", program="video_then_action"),
        backbone=SimpleNamespace(transformer_subdir="/unused/transformer"),
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
            dual_expert_generalist_rollout_mode="joint",
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
            component_report_extra={"caller": "single"},
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
        {"checkpoint_runtime_compat": True},
    )
