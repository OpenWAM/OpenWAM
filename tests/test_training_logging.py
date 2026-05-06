from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import open_wam.training.runtime as runtime_module
from open_wam.training.run_tracking import (
    build_default_wandb_project,
    build_run_title,
    build_run_tracking_metadata,
    build_wandb_group,
    build_wandb_job_type,
    build_wandb_tags,
    resolve_wandb_project,
)
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_tracking_metadata_normalizes_method_families(tmp_path: Path) -> None:
    cases = [
        ("parallel_stream_robotwin_smoke.yaml", "method_1", "parallel_stream"),
        ("register_attached_robotwin_smoke.yaml", "method_2", "register_attached"),
        ("video_sequence_policy_robotwin_smoke.yaml", "method_3", "video_sequence_policy"),
        ("post_latent_robotwin_video_conditioned.yaml", "method_4", "post_latent"),
        ("mot_robotwin_smoke.yaml", "method_5", "mot"),
        ("causal_video_prediction_robotwin_smoke.yaml", "causal_video_prediction", "causal_video_prediction"),
    ]

    for config_name, expected_method_family, expected_job_type in cases:
        config = load_experiment_config(REPO_ROOT / "configs/experiments" / config_name)
        output_dir = tmp_path / config.name

        metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=output_dir)

        assert metadata["framework"] == "open_wam"
        assert metadata["experiment_name"] == config.name
        assert metadata["run_name"] == config.name
        assert metadata["run_slug"] == config.name
        assert metadata["method_family"] == expected_method_family
        assert metadata["method_label"] in {"m1", "m2", "m3", "m4", "m5", "causal"}
        assert metadata["policy_variant"] == expected_job_type
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

    monkeypatch.setattr(runtime_module, "WandBLogSink", _FakeWandBLogSink)

    output_dir = tmp_path / "track-run"
    sink = runtime_module.build_log_sink(
        config=config,
        output_dir=output_dir,
        run_name="track-run",
    )
    sink.close()

    assert captured["project"] == "open-wam"
    assert captured["mode"] == "offline"
    assert captured["run_name"] == "robotwin · m4 · post_decoded · track-run"
    assert captured["group"] == "robotwin/m4/post_decoded"
    assert captured["job_type"] == "policy_train"
    assert "framework:open_wam" in captured["tags"]
    assert "method:m4" in captured["tags"]
    assert "method_family:method_4" in captured["tags"]
    assert "variant:post_decoded" in captured["tags"]
    assert "decoder:video_conditioned_action_decoder" in captured["tags"]
    assert "dataset:robotwin" in captured["tags"]

    config_payload = captured["config_payload"]
    assert isinstance(config_payload, dict)
    assert config_payload["tracking"]["method_family"] == "method_4"
    assert config_payload["tracking"]["method_label"] == "m4"
    assert config_payload["tracking"]["policy_variant"] == "post_decoded"
    assert config_payload["tracking"]["action_decoder"] == "video_conditioned_action_decoder"
    assert config_payload["tracking"]["wandb_group"] == "robotwin/m4/post_decoded"
    assert config_payload["tracking"]["wandb_job_type"] == "policy_train"
    assert config_payload["tracking"]["output_dir"] == str(output_dir)


def test_wandb_group_job_type_and_tags_follow_tracking_metadata(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml")
    metadata = build_run_tracking_metadata(config, run_name="mot-run", output_dir=tmp_path / "mot-run")

    assert build_wandb_group(metadata) == "robotwin/m5/mot"
    assert build_wandb_job_type(metadata) == "policy_train"
    assert build_run_title(metadata) == "robotwin · m5 · mot · mot-run"
    tags = build_wandb_tags(metadata)
    assert tags[:4] == (
        "framework:open_wam",
        "dataset:robotwin",
        "dataset_type:synthetic_robotwin",
        "workload:policy_train",
    )
    assert "decoder:mot_decoder" in tags
    assert "method:m5" in tags
    assert "segment_frames:None" not in tags


def test_method4_generated_video_condition_source_is_tracked(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_libero_latent_local_generated_video_conditioned.yaml"
    )
    metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=tmp_path / config.name)

    assert metadata["train_video_condition_source"] == "generated_future"
    assert "train_video_condition:generated_future" in build_wandb_tags(metadata)


def test_method1_coupling_and_segment_sampling_are_tracked(tmp_path: Path) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml"
    )
    metadata = build_run_tracking_metadata(config, run_name=config.name, output_dir=tmp_path / config.name)

    assert metadata["runtime_mode"] == "lingbot_exact"
    assert metadata["current_block_coupling"] == "video_then_action"
    assert metadata["reference_profile"] == "libero"
    assert metadata["sample_construction_mode"] == "uniform_segment"
    assert metadata["segment_min_frames"] == 128
    assert metadata["segment_max_frames"] == 128
    assert metadata["start_padding_frames"] == 3
    assert metadata["sample_weight_mode"] == "task_virtual_start_count_power"
    tags = build_wandb_tags(metadata)
    assert "coupling:video_then_action" in tags
    assert "sample:uniform_segment" in tags
    assert "segment_frames:128" in tags
    assert "start_padding_frames:3" in tags
    assert "sample_weight:task_virtual_start_count_power" in tags


def test_wandb_project_defaults_to_dataset_and_workload_bin(tmp_path: Path) -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml")
    config = replace(config, trainer=replace(config.trainer, enable_wandb=True, wandb_project=None))
    metadata = build_run_tracking_metadata(config, run_name="mot-run", output_dir=tmp_path / "mot-run")

    assert build_default_wandb_project(metadata) == "openwam-robotwin-policy-train"
    assert resolve_wandb_project(config, metadata) == "openwam-robotwin-policy-train"
