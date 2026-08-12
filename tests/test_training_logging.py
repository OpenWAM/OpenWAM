from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import open_wam.training.logging as logging_module
from open_wam.configs import load_experiment_config
from open_wam.training.logging import WandBLogSink
from open_wam.training.run_tracking import (
    build_default_wandb_project,
    build_run_title,
    build_run_tracking_metadata,
    build_wandb_group,
    build_wandb_job_type,
    build_wandb_tags,
    resolve_wandb_project,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wandb_log_sink_can_use_contiguous_global_step(monkeypatch) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class _FakeWandB:
        @staticmethod
        def init(**kwargs):
            calls.append(("init", (), kwargs))
            return object()

        @staticmethod
        def define_metric(*args, **kwargs):
            calls.append(("define_metric", args, kwargs))

        @staticmethod
        def log(payload, **kwargs):
            calls.append(("log", (payload,), kwargs))

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandB)
    monkeypatch.setenv("OPEN_WAM_WANDB_CONTIGUOUS_STEPS", "1")

    sink = WandBLogSink(
        project="project",
        entity=None,
        mode="offline",
        run_name="run",
        group="group",
        job_type="train",
        tags=("tag",),
        config_payload={"config": True},
    )
    sink.log_metrics(step=7, phase="train", metrics={"loss": 1.5})

    define_metric_calls = [call for call in calls if call[0] == "define_metric"]
    assert define_metric_calls == [
        ("define_metric", ("trainer/global_step",), {}),
        ("define_metric", ("*",), {"step_metric": "trainer/global_step"}),
    ]
    log_calls = [call for call in calls if call[0] == "log"]
    assert len(log_calls) == 1
    payload = log_calls[0][1][0]
    assert payload == {
        "train/loss": 1.5,
        "trainer/global_step": 7,
        "train/global_step": 7,
    }
    assert log_calls[0][2] == {}


def test_run_tracking_metadata_uses_policy_architectures(tmp_path: Path) -> None:
    cases = [
        ("parallel_stream_robotwin_smoke.yaml", "parallel_stream"),
        ("post_latent_robotwin_video_conditioned.yaml", "post_latent"),
        ("dual_expert_robotwin_smoke.yaml", "dual_expert"),
        ("causal_video_prediction_robotwin_smoke.yaml", "causal_video_prediction"),
    ]

    for config_name, expected_architecture in cases:
        config = load_experiment_config(REPO_ROOT / "configs/experiments" / config_name)
        output_dir = tmp_path / config.name

        metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=output_dir)

        assert metadata["framework"] == "open_wam"
        assert metadata["experiment_name"] == config.name
        assert metadata["run_name"] == config.name
        assert metadata["run_slug"] == config.name
        assert metadata["tracking_schema_version"] == 2
        assert metadata["architecture"] == expected_architecture
        assert metadata["policy_variant"] == expected_architecture
        assert metadata["run_title"] == build_run_title(metadata)
        assert metadata["dataset_name"] == config.data.dataset_name
        assert metadata["dataset_type"] == config.data.dataset_type
        assert metadata["runtime"] == str(config.trainer.runtime)
        assert metadata["output_dir"] == str(output_dir)
        assert metadata["checkpoint_dir"] == str(output_dir / "checkpoints")


def test_build_log_sink_passes_standardized_wandb_tracking_context(monkeypatch, tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_robotwin_video_conditioned.yaml")
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            enable_wandb=True,
            wandb_project="open-wam",
            wandb_mode="offline",
        ),
    )

    captured: dict[str, object] = {}

    class _FakeWandBLogSink:
        def __init__(
            self,
            *,
            project,
            entity,
            mode,
            run_name,
            group,
            job_type,
            tags,
            config_payload,
        ) -> None:
            captured["project"] = project
            captured["entity"] = entity
            captured["mode"] = mode
            captured["run_name"] = run_name
            captured["group"] = group
            captured["job_type"] = job_type
            captured["tags"] = tuple(tags)
            captured["config_payload"] = config_payload

        def log_metrics(self, *, step: int, phase: str, metrics: dict[str, float]) -> None:
            del step, phase, metrics

        def log_event(self, *, name: str, payload: dict[str, object]) -> None:
            del name, payload

        def close(self) -> None:
            return None

    monkeypatch.setattr(logging_module, "WandBLogSink", _FakeWandBLogSink)

    output_dir = tmp_path / "track-run"
    sink = logging_module.build_log_sink(
        config=config,
        output_dir=output_dir,
        run_name="track-run",
    )
    sink.close()

    assert captured["project"] == "open-wam"
    assert captured["mode"] == "offline"
    assert captured["run_name"] == "robotwin · post_decoded · track-run"
    assert captured["group"] == "robotwin/post_decoded"
    assert captured["job_type"] == "policy_train"
    assert "framework:open_wam" in captured["tags"]
    assert "architecture:post_decoded" in captured["tags"]
    assert "variant:post_decoded" in captured["tags"]
    assert "decoder:video_conditioned_action_decoder" in captured["tags"]
    assert "dataset:robotwin" in captured["tags"]

    config_payload = captured["config_payload"]
    assert isinstance(config_payload, dict)
    assert config_payload["tracking"]["architecture"] == "post_decoded"
    assert config_payload["tracking"]["program"] is None
    assert config_payload["tracking"]["policy_variant"] == "post_decoded"
    assert config_payload["tracking"]["action_decoder"] == "video_conditioned_action_decoder"
    assert config_payload["tracking"]["wandb_group"] == "robotwin/post_decoded"
    assert config_payload["tracking"]["wandb_job_type"] == "policy_train"
    assert config_payload["tracking"]["output_dir"] == str(output_dir)


def test_wandb_group_job_type_and_tags_follow_tracking_metadata(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    metadata = build_run_tracking_metadata(config, run_name="dual_expert-run", output_dir=tmp_path / "dual_expert-run")

    assert build_wandb_group(metadata) == "robotwin/dual_expert"
    assert build_wandb_job_type(metadata) == "policy_train"
    assert build_run_title(metadata) == "robotwin · dual_expert · dual_expert-run"
    tags = build_wandb_tags(metadata)
    assert tags[:4] == (
        "framework:open_wam",
        "dataset:robotwin",
        "dataset_type:synthetic_robotwin",
        "workload:policy_train",
    )
    assert "decoder:dual_expert_decoder" in tags
    assert "architecture:dual_expert" in tags
    assert "segment_frames:None" not in tags


def test_generated_video_condition_source_is_tracked(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_libero_latent_local_generated_video_conditioned.yaml"
    )
    metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=tmp_path / config.name)

    assert metadata["train_video_condition_source"] == "generated_future"
    assert "train_video_condition:generated_future" in build_wandb_tags(metadata)


def test_legacy_sample_construction_does_not_emit_rollout_context_tag(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=tmp_path / config.name)

    assert metadata["target_alignment"] == "legacy"
    assert metadata["rollout_context_policy"] == "one_frame"
    assert "rollout_context:one_frame" not in build_wandb_tags(metadata)


def test_parallel_stream_program_and_segment_sampling_are_tracked(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )
    metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=tmp_path / config.name)

    assert metadata["runtime_mode"] == "lingbot_exact"
    assert metadata["architecture"] == "parallel_stream"
    assert metadata["program"] == "video_then_action"
    assert metadata["current_block_coupling"] == "video_then_action"
    assert metadata["reference_profile"] == "libero"
    assert metadata["sample_construction_mode"] == "uniform_segment"
    assert metadata["segment_frames"] is None
    assert metadata["segment_min_frames"] == 1000
    assert metadata["segment_max_frames"] == 1000
    assert metadata["start_padding_frames"] == 0
    assert metadata["target_alignment"] == "legacy"
    assert metadata["rollout_context_policy"] == "one_frame"
    assert metadata["task_start_power"] == 0.0
    assert metadata["demo_count_power"] == 0.0
    assert metadata["trajectory_start_power"] == 0.0
    assert metadata["sample_order_mode"] == "replacement"
    tags = build_wandb_tags(metadata)
    assert "coupling:video_then_action" in tags
    assert "sample:uniform_segment" in tags
    assert "segment_frames:1000" in tags
    assert "sample_order:replacement" in tags
    assert not any(tag.startswith("target_alignment:") for tag in tags)
    assert not any(tag.startswith("rollout_context:") for tag in tags)
    assert metadata["gjd_ablation"] is None
    assert build_wandb_group(metadata) == "libero/parallel_stream/video_then_action"
    assert "gjd:" not in build_run_title(metadata)
    assert not any(tag.startswith("gjd:") for tag in tags)


def test_non_default_sample_order_is_tracked(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    config = replace(
        config,
        data=replace(
            config.data,
            sample_construction=replace(
                config.data.sample_construction,
                sample_order_mode="replacement",
            ),
        ),
    )

    metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=tmp_path / config.name)

    assert metadata["sample_order_mode"] == "replacement"
    assert "sample_order:replacement" in build_wandb_tags(metadata)


def test_parallel_stream_generalist_joint_denoising_tracking_metadata(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    config = replace(
        config,
        policy_variant=replace(config.policy_variant, generalist_mode_text_token=True),
    )
    metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=tmp_path / config.name)

    assert metadata["architecture"] == "parallel_stream"
    assert metadata["program"] == "generalist_joint_denoising"
    assert metadata["variant_profile"] == "generalist_joint_denoising"
    assert metadata["gjd_ablation"] == "mode_token"
    assert metadata["generalist_mode_text_token"] is True
    assert metadata["generalist_denoising_mode_probs"] == {
        "joint": 0.6,
        "action_conditioned_video": 0.2,
        "video_conditioned_action": 0.2,
    }
    assert (
        build_wandb_group(metadata)
        == "libero/parallel_stream/generalist_joint_denoising/mode_token"
    )
    assert "gjd:mode_token" in build_run_title(metadata)
    tags = build_wandb_tags(metadata)
    assert "variant_profile:generalist_joint_denoising" in tags
    assert "gjd:parallel_stream:mode_token" in tags
    assert "generalist_mode_text_token" in tags


def test_dual_expert_generalist_joint_denoising_tracking_metadata(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    config = replace(
        config,
        policy_variant=replace(config.policy_variant, generalist_mode_text_token=True),
    )
    metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=tmp_path / config.name)

    assert metadata["architecture"] == "dual_expert"
    assert metadata["program"] == "generalist_joint_denoising"
    assert metadata["gjd_ablation"] == "mode_token"
    assert metadata["generalist_mode_text_token"] is True
    assert metadata["generalist_denoising_mode_probs"] == {
        "joint": 0.6,
        "action_conditioned_video": 0.2,
        "video_conditioned_action": 0.2,
    }
    assert (
        build_wandb_group(metadata)
        == "libero/dual_expert/generalist_joint_denoising/mode_token"
    )
    assert "gjd:mode_token" in build_run_title(metadata)
    tags = build_wandb_tags(metadata)
    assert "gjd:dual_expert:mode_token" in tags
    assert "generalist_mode_text_token" in tags


def test_pure_conditional_gjd_tracking_preserves_gjd_identity(tmp_path: Path) -> None:
    base = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    cases = (
        ("pure_fdm", "action_conditioned_video"),
        ("pure_idm", "video_conditioned_action"),
    )

    for ablation, selected_mode in cases:
        probabilities = {
            "joint": 0.0,
            "action_conditioned_video": float(selected_mode == "action_conditioned_video"),
            "video_conditioned_action": float(selected_mode == "video_conditioned_action"),
        }
        config = replace(
            base,
            policy_variant=replace(
                base.policy_variant,
                generalist_denoising_mode_probs=probabilities,
                generalist_mode_text_token=False,
            ),
        )
        metadata = build_run_tracking_metadata(
            config,
            run_name=ablation,
            output_dir=tmp_path / ablation,
        )

        assert metadata["program"] == "generalist_joint_denoising"
        assert metadata["gjd_ablation"] == ablation
        assert metadata["fixed_conditioning_mode"] == selected_mode
        assert f"gjd:dual_expert:{ablation}" in build_wandb_tags(metadata)


def test_dual_expert_conditional_dynamics_tracking_uses_program_identity(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )

    metadata = build_run_tracking_metadata(
        config,
        run_name=config.name,
        output_dir=tmp_path / config.name,
    )

    assert metadata["architecture"] == "dual_expert"
    assert metadata["program"] == "forward_dynamics"
    assert metadata["fixed_conditioning_mode"] == "action_conditioned_video"
    assert metadata["gjd_ablation"] is None
    assert build_wandb_group(metadata) == "libero/dual_expert/forward_dynamics"
    tags = build_wandb_tags(metadata)
    assert "program:forward_dynamics" in tags
    assert "conditioning_mode:action_conditioned_video" in tags
    assert not any(tag.startswith("gjd:") for tag in tags)


def test_wandb_project_defaults_to_dataset_and_workload_bin(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    config = replace(config, trainer=replace(config.trainer, enable_wandb=True, wandb_project=None))
    metadata = build_run_tracking_metadata(config, run_name="dual_expert-run", output_dir=tmp_path / "dual_expert-run")

    assert build_default_wandb_project(metadata) == "openwam-robotwin-policy-train"
    assert resolve_wandb_project(config, metadata) == "openwam-robotwin-policy-train"
