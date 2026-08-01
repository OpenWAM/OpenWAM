from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from open_wam.configs.enums import (
    DeadlineMissPolicy,
    RealtimeSchedulerProfile,
    ReferenceAssetsDevicePolicy,
    RolloutArtifactProfile,
)
from open_wam.evals import sampled_eval_planning as planning
from open_wam.evals.sampled_eval_sampling import DatasetEpisode


def test_sampled_eval_planning_contract_is_explicit_and_enum_backed() -> None:
    assert set(planning.__all__) == {
        "SAMPLED_EVAL_DEFAULT_CONFIG",
        "SAMPLED_EVAL_METHODS",
        "SAMPLED_EVAL_SCHEDULERS",
        "SampledEvalCase",
        "SampledEvalCaseOptions",
        "SampledEvalCheckpointSpec",
        "SampledEvalMethodSpec",
        "SampledEvalPreflightOptions",
        "SampledEvalSchedulerSpec",
        "SampledEvalTargetRequest",
        "build_sampled_eval_cases",
        "parse_sampled_eval_target_requests",
        "preflight_sampled_eval_cases",
        "resolve_sampled_eval_checkpoint_specs",
        "sampled_eval_scheduler_flags",
        "sampled_eval_scheduler_suffix",
        "sanitize_sampled_eval_label",
        "select_sampled_eval_specs_by_key",
    }
    assert [method.key for method in planning.SAMPLED_EVAL_METHODS] == ["m1", "m2", "m5"]
    assert all(
        isinstance(method.reference_assets_device_policy, ReferenceAssetsDevicePolicy)
        for method in planning.SAMPLED_EVAL_METHODS
    )
    assert [scheduler.key for scheduler in planning.SAMPLED_EVAL_SCHEDULERS] == [
        RealtimeSchedulerProfile.BLOCKING_CONTROL,
        RealtimeSchedulerProfile.FREEZE_UNTIL_CLEAN_CHUNK,
        RealtimeSchedulerProfile.ASYNC_HISTORY_FIRST,
    ]


def test_parse_and_select_target_contract_preserves_order_and_diagnostics() -> None:
    requests = planning.parse_sampled_eval_target_requests(
        ["M1:step 4:Exact step 4=/runs/step4", "m5:latest=/runs/latest"]
    )

    assert requests == [
        planning.SampledEvalTargetRequest(
            method_key="m1",
            checkpoint_key="step_4",
            label="Exact step 4",
            checkpoint="/runs/step4",
        ),
        planning.SampledEvalTargetRequest(
            method_key="m5",
            checkpoint_key="latest",
            label=None,
            checkpoint="/runs/latest",
        ),
    ]
    assert [
        method.key
        for method in planning.select_sampled_eval_specs_by_key(
            planning.SAMPLED_EVAL_METHODS,
            "m5,m1,m5",
            field_name="methods",
        )
    ] == ["m5", "m1"]
    with pytest.raises(ValueError, match=r"Duplicate --target for m1:x\."):
        planning.parse_sampled_eval_target_requests(["m1:x=/a", "m1:x=/b"])
    with pytest.raises(ValueError, match="Unknown methods key 'm9'"):
        planning.select_sampled_eval_specs_by_key(
            planning.SAMPLED_EVAL_METHODS,
            "m9",
            field_name="methods",
        )


def test_resolve_checkpoint_specs_supports_state_and_transformer_exports(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint_step_4"
    transformer_dir = checkpoint_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (checkpoint_dir / "model_state.pt").write_bytes(b"state")
    (transformer_dir / "weights.bin").write_bytes(b"weights")

    export_root = tmp_path / "export"
    export_transformer = export_root / "transformer"
    export_transformer.mkdir(parents=True)
    (export_transformer / "config.json").write_text("{}", encoding="utf-8")
    (export_transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")

    selected = [planning.SAMPLED_EVAL_METHODS[0]]
    specs = planning.resolve_sampled_eval_checkpoint_specs(
        selected_methods=selected,
        target_requests=[
            planning.SampledEvalTargetRequest("m1", "state", None, str(checkpoint_dir)),
            planning.SampledEvalTargetRequest("m1", "export", "Export", str(export_root)),
        ],
        config_override="configs/custom.yaml",
        reference_assets_device_policy_override=ReferenceAssetsDevicePolicy.CPU_OFFLOAD,
    )

    assert specs[0].checkpoint == str((checkpoint_dir / "model_state.pt").resolve())
    assert specs[0].runtime_transformer_source == "checkpoint"
    assert specs[0].extra_args == ("--merge-checkpoint-runtime-config",)
    assert specs[1].checkpoint == str(export_root)
    assert specs[1].runtime_transformer_dir == str(export_transformer.resolve())
    assert specs[1].extra_args == ()
    assert all(spec.config == "configs/custom.yaml" for spec in specs)
    assert all(
        spec.reference_assets_device_policy is ReferenceAssetsDevicePolicy.CPU_OFFLOAD
        for spec in specs
    )


def test_build_cases_owns_deterministic_matrix_without_simulator_imports(tmp_path: Path) -> None:
    episode = DatasetEpisode(
        dataset_episode_index=12,
        task_text="task",
        task_index=3,
        task_id=3,
        task_name="task_3",
        episode_idx=4,
        length=100,
        replay_status="success",
    )
    method = planning.SAMPLED_EVAL_METHODS[2]
    checkpoint = planning.SampledEvalCheckpointSpec(
        key="m5_posttrained",
        label="M5 posttrained",
        checkpoint="/models/model_state.pt",
        checkpoint_file="/models/model_state.pt",
        method_key=method.key,
        method_label=method.label,
        config=method.config,
        reference_assets_device_policy=method.reference_assets_device_policy,
    )
    scheduler = planning.SAMPLED_EVAL_SCHEDULERS[1]
    [case] = planning.build_sampled_eval_cases(
        [episode],
        checkpoint_specs=[checkpoint],
        output_root=tmp_path / "out",
        benchmark="custom_suite",
        seed=9,
        scheduler_spec=scheduler,
        options=planning.SampledEvalCaseOptions(
            python="/venv/bin/python",
            run_label="matrix run",
            eval_profile="libero_10hz_full",
            rollout_artifact_profile=RolloutArtifactProfile.DEBUG,
            max_actions=123,
            deadline_miss_policy=DeadlineMissPolicy.HOLD_STATE,
            write_fallback_timeline_video=True,
        ),
    )

    assert case.scheduler_key is RealtimeSchedulerProfile.FREEZE_UNTIL_CLEAN_CHUNK
    assert case.episode_id == 12
    assert case.init_id == 4
    assert case.suffix.endswith("freeze_until_clean_chunk_startup1")
    assert case.command_template[:2] == [
        "/venv/bin/python",
        "scripts/run_libero_realtime_sandbox.py",
    ]
    assert case.command_template[case.command_template.index("--benchmark") + 1] == "custom_suite"
    assert case.command_template[case.command_template.index("--max-actions") + 1] == "123"
    assert case.command_template[case.command_template.index("--deadline-miss-policy") + 1] == "hold_state"
    assert case.command_template[-3:] == [
        "--startup-open-loop-chunks",
        "1",
        "--write-fallback-timeline-video",
    ]
    assert "open_wam.integrations.libero_tasks" not in planning.__dict__


def test_build_cases_accepts_an_explicit_custom_method_registry(tmp_path: Path) -> None:
    method = planning.SampledEvalMethodSpec(
        key="custom",
        label="Custom method",
        config="configs/custom.yaml",
        reference_assets_device_policy="runtime",
        async_low_watermark=6,
        extra_args=("--custom-runtime-flag",),
    )
    checkpoint = planning.SampledEvalCheckpointSpec(
        key="custom_step",
        label="Custom step",
        checkpoint="/models/custom.pt",
        checkpoint_file="/models/custom.pt",
        method_key=method.key,
        method_label=method.label,
        config=method.config,
        reference_assets_device_policy=method.reference_assets_device_policy,
        extra_args=method.extra_args,
    )

    [case] = planning.build_sampled_eval_cases(
        [DatasetEpisode(0, "task", 0, 0, None, 0, 10)],
        checkpoint_specs=[checkpoint],
        output_root=tmp_path,
        benchmark="libero_10",
        seed=0,
        scheduler_spec=planning.SAMPLED_EVAL_SCHEDULERS[2],
        options=planning.SampledEvalCaseOptions(
            python="python",
            run_label="extension",
            eval_profile="custom_profile",
            rollout_artifact_profile="lean",
        ),
        methods=(method,),
    )

    assert case.method_key == "custom"
    assert case.command_template[case.command_template.index("--cfg") + 1] == method.config
    assert "--custom-runtime-flag" in case.command_template
    assert case.command_template[-2:] == ["--replan-low-watermark-actions", "6"]


def test_preflight_reports_unique_missing_inputs_in_stable_order(tmp_path: Path) -> None:
    options = planning.SampledEvalPreflightOptions(
        repo_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        python=tmp_path / "python",
        local_paths=Path("configs/local_paths.yaml"),
        libero_repo_root=tmp_path / "LIBERO",
    )
    missing_spec = planning.SampledEvalCheckpointSpec(
        key="missing",
        label="Missing",
        checkpoint=str(tmp_path / "checkpoint"),
        checkpoint_raw=str(tmp_path / "checkpoint"),
        config="configs/missing.yaml",
        preflight_problem="checkpoint is missing",
    )

    missing = planning.preflight_sampled_eval_cases(
        options=options,
        checkpoint_specs=[missing_spec, missing_spec],
        dataset_problem="metadata is missing",
    )

    assert [record["kind"] for record in missing] == [
        "dataset",
        "python",
        "local_paths",
        "libero_repo_root",
        "config",
        "checkpoint",
    ]
    assert missing[-1]["reason"] == "checkpoint is missing"
    assert asdict(missing_spec)["reference_assets_device_policy"] == "runtime"
