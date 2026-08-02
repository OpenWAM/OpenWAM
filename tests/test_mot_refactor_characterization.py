from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from open_wam.data import LatentWAMBatch
from open_wam.models.common.rollout_history import resolve_execute_action_steps
from open_wam.configs import load_experiment_config
from tests.characterization.mot_refactor_artifacts import (
    ComparisonTolerance,
    compare_characterization_reports,
    load_latent_batch_fixture,
    save_latent_batch_fixture,
    tensor_fingerprint,
)
from tests.characterization.mot_refactor_contract import (
    ALL_METHODS,
    CACHE_ROLLOVER_ASSET_IDS,
    DEFAULT_INFERENCE_CONTRACT,
    EXACT_CHECKPOINT_METHODS,
    FULL_STATE_RESUME_ASSET_ID,
    GJD_INFERENCE_CONTRACT,
    GJD_METHODS,
    METHOD_BY_ASSET_ID,
    NON_GJD_METHODS,
    GJDTrainingMode,
    MoTTrainingProfile,
    apply_gjd_ablation,
    apply_ground_truth_training_profile,
    ground_truth_training_overrides,
    inference_scenarios_for,
    load_characterization_assets,
    training_scenarios_for,
)
from tests.characterization.mot_refactor_end_to_end import (
    build_libero_rollout_command,
    build_training_cli_smoke_command,
    characterization_environment,
    collect_libero_rollout_report,
    stage_model_only_checkpoint,
)
from tests.characterization.mot_refactor_fixtures import (
    _representative_counterfactual_source_index,
)
from tests.characterization.mot_refactor_provenance import (
    apply_checkpoint_provenance_policy,
    assert_checkpoint_provenance,
    build_checkpoint_source_contract_config,
    checkpoint_provenance_report,
    expected_checkpoint_contract,
)
from tests.characterization.mot_refactor_worker import (
    CACHE_ROLLOVER_CHARACTERIZATION_CHUNKS,
    INFERENCE_CHARACTERIZATION_CHUNKS,
    RESUME_REPORT_SCHEMA_VERSION,
    _assert_cache_rollover,
    _assert_inference_progression,
    _assert_optimizer_step,
    _assert_training_scenario,
    _distributed_module_digest,
    _distributed_optimizer_digest,
    _execute_characterization_optimizer_step,
    _load_pipeline_checkpoint,
    _resume_update_differences,
    _resume_update_report_contract,
    _runtime_state_contract_differences,
)
from tests.characterization.run_mot_refactor_characterization import (
    RESUME_POST_UPDATE_METRIC_TOLERANCE,
    _comparison_projection,
    _initialize_golden_files,
    _numeric_tolerance_resolver,
    _preflight_checkpoint_provenance,
    _selected_asset_ids,
    _stage_directory,
    _stage_file,
    _validate_phase_asset_selection,
    _worker_command,
    compare_fixture_directories,
)

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def _load_method_config(config_name: str):
    return load_experiment_config(CONFIG_ROOT / f"{config_name}.yaml")


def test_characterization_matrix_has_unique_complete_asset_ids() -> None:
    assert len(ALL_METHODS) == 9
    assert len(METHOD_BY_ASSET_ID) == len(ALL_METHODS)
    assert len(NON_GJD_METHODS) == 6
    assert len(GJD_METHODS) == 3
    assert len(EXACT_CHECKPOINT_METHODS) == 6
    exact_ids = {method.asset_id for method in EXACT_CHECKPOINT_METHODS}
    assert {
        "mot_video_noisy_to_action",
        "mot_action_noisy_to_video",
        "gjd_vanilla",
    }.isdisjoint(exact_ids)
    assert "gjd_vanilla" in METHOD_BY_ASSET_ID


def test_default_exact_checkpoint_selection_uses_available_strict_assets() -> None:
    assert _selected_asset_ids(None) == tuple(
        method.asset_id for method in EXACT_CHECKPOINT_METHODS
    )
    assert _selected_asset_ids(["gjd_vanilla"]) == ("gjd_vanilla",)


def test_golden_initialization_never_replaces_an_existing_report(
    tmp_path: Path,
) -> None:
    first_report = tmp_path / "first.json"
    first_report.write_text('{"value": 1}\n', encoding="utf-8")
    second_report = tmp_path / "second.json"
    second_report.write_text('{"value": 2}\n', encoding="utf-8")
    golden_root = tmp_path / "goldens"

    _initialize_golden_files(
        ((first_report, "mot_joint.training.json"),),
        golden_root=golden_root,
    )
    golden = golden_root / "mot_joint.training.json"
    assert golden.read_bytes() == first_report.read_bytes()

    with pytest.raises(FileExistsError, match="new versioned golden root"):
        _initialize_golden_files(
            ((second_report, "mot_joint.training.json"),),
            golden_root=golden_root,
        )

    assert golden.read_bytes() == first_report.read_bytes()


def test_golden_initialization_is_all_or_nothing_for_missing_sources(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    golden_root = tmp_path / "goldens"

    with pytest.raises(FileNotFoundError, match="missing reports"):
        _initialize_golden_files(
            (
                (report, "mot_joint.training.json"),
                (tmp_path / "missing.json", "mot_joint.inference.json"),
            ),
            golden_root=golden_root,
        )

    assert not golden_root.exists()


def test_golden_initialization_rolls_back_a_partial_atomic_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = []
    for index in range(2):
        report = tmp_path / f"report-{index}.json"
        report.write_text(f'{{"value": {index}}}\n', encoding="utf-8")
        reports.append((report, f"report-{index}.json"))

    real_link = os.link
    link_calls = 0

    def fail_second_link(source, destination) -> None:
        nonlocal link_calls
        link_calls += 1
        if link_calls == 2:
            raise OSError("simulated promotion failure")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", fail_second_link)
    golden_root = tmp_path / "goldens"
    with pytest.raises(OSError, match="simulated promotion failure"):
        _initialize_golden_files(reports, golden_root=golden_root)

    assert list(golden_root.iterdir()) == []


@pytest.mark.parametrize("method", NON_GJD_METHODS, ids=lambda method: method.asset_id)
@pytest.mark.parametrize("profile", tuple(MoTTrainingProfile))
def test_non_gjd_ground_truth_training_profiles_resolve_strict_contract(
    method,
    profile: MoTTrainingProfile,
) -> None:
    config = apply_ground_truth_training_profile(
        _load_method_config(method.config_name),
        profile,
    )
    sample = config.data.sample_construction

    assert config.policy_variant.current_block_coupling.value == method.coupling
    assert (
        config.policy_variant.parallel_sequence_contract.value
        == "legacy_prefix_single_frame_perchunk_proprio"
    )
    assert config.policy_variant.noisy_video_condition_prob == pytest.approx(0.5)
    assert config.policy_variant.proprio_context_mode.value == "per_chunk_additive"
    assert (
        config.policy_variant.context_condition_latent_source.value
        == "single_frame_condition_latent"
    )
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert sample.condition_source_frame_offset == -1
    assert sample.start_padding_frames == 0
    assert sample.target_alignment.value == "legacy"
    assert sample.mode.value == "uniform_segment"
    assert sample.sample_order_mode.value == "replacement"
    assert sample.randomize_geometry is True
    assert sample.segment_locality_block_size == 1
    assert sample.require_full_segment is True
    assert sample.task_start_power == 0.0
    assert sample.demo_count_power == 0.0
    assert sample.trajectory_start_power == 0.0
    assert sample.sample_weight_mode.value == "uniform"
    assert config.training.sample_loss_weight_mode.value == "none"
    assert config.training.chunk_size == 4
    assert config.data.replay_status_policy.value == "include_all"
    assert config.data.val_replay_status_policy is None
    assert config.data.require_replay_status is False
    assert config.data.val_require_replay_status is False
    assert config.trainer.checkpoint_mode.value == "model_only"
    assert config.trainer.save_interval == 500
    assert config.trainer.max_checkpoints_to_keep == 3
    assert config.inference.frame_chunk_size == 4
    assert config.data.action_schema.action_horizon == 16

    if profile == MoTTrainingProfile.RANDOM_SEGMENT:
        assert sample.segment_min_frames == 64
        assert sample.segment_max_frames == 256
        assert sample.segment_length_stride == 4
        assert sample.randomize_segment_length is True
        assert sample.randomize_segment_start is True
        assert sample.window_size == 30
        assert config.training.window_size == 30
        assert config.training.num_steps == 5000
    else:
        assert sample.segment_min_frames == 1000
        assert sample.segment_max_frames == 1000
        assert sample.segment_length_stride == 1
        assert sample.window_size == 64
        assert sample.randomize_segment_length is False
        assert sample.randomize_segment_start is False
        assert config.training.window_size == 64
        assert config.training.num_steps == 10000


@pytest.mark.parametrize("profile", tuple(MoTTrainingProfile))
def test_non_gjd_modes_differ_only_by_name_and_coupling(
    profile: MoTTrainingProfile,
) -> None:
    normalized_configs = []
    for method in NON_GJD_METHODS:
        config = apply_ground_truth_training_profile(
            _load_method_config(method.config_name),
            profile,
        )
        normalized = asdict(config)
        normalized["name"] = "<method>"
        normalized["policy_variant"]["current_block_coupling"] = "<coupling>"
        normalized_configs.append(normalized)

    assert all(config == normalized_configs[0] for config in normalized_configs[1:])


def test_ground_truth_profile_overrides_exclude_machine_and_tracking_state() -> None:
    for profile in MoTTrainingProfile:
        keys = set(ground_truth_training_overrides(profile))
        assert {key for key in keys if key.startswith("trainer.")} == {
            "trainer.checkpoint_mode",
            "trainer.max_checkpoints_to_keep",
            "trainer.save_interval",
        }
        assert "data.local_root" not in keys
        assert "backbone.transformer_subdir" not in keys
        assert "trainer.enable_wandb" not in keys


def test_example_asset_manifest_keeps_video_base_and_transformer_distinct() -> None:
    manifest = (
        Path(__file__).resolve().parent / "characterization" / "mot_assets.example.yaml"
    ).read_text(encoding="utf-8")

    assert "base_model_root: ${OPEN_WAM_LINGBOT_VA_BASE}" in manifest
    assert "video_transformer_root: ${OPEN_WAM_LIBERO_VIDEO_TRANSFORMER}" in manifest


def test_asset_manifest_resolves_explicit_checkpoint_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "copied" / "model_state.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    resolved_config = tmp_path / "source" / "resolved_config.yaml"
    resolved_config.parent.mkdir()
    resolved_config.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "assets.yaml"
    manifest.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "paths:",
                "  dataset_root: /tmp/data",
                "  base_model_root: /tmp/base",
                "  video_transformer_root: /tmp/transformer",
                "  empty_text_embedding: /tmp/empty.pt",
                "  counterfactual_train_root: null",
                "  counterfactual_val_root: null",
                "checkpoints:",
                "  mot_joint:",
                f"    model_state: {checkpoint}",
                f"    resolved_config: {resolved_config}",
                "    accepted_origin_mismatch_fields:",
                "      - data.replay_status_policy",
            )
        ),
        encoding="utf-8",
    )

    assets = load_characterization_assets(manifest)

    assert assets.checkpoint_for("mot_joint") == checkpoint
    assert assets.checkpoint_config_for("mot_joint") == resolved_config
    assert assets.checkpoints["mot_joint"].accepted_origin_mismatch_fields == (
        "data.replay_status_policy",
    )


def test_asset_manifest_requires_checkpoint_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_state.pt"
    checkpoint.write_bytes(b"weights")
    manifest = tmp_path / "assets.yaml"
    manifest.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "paths:",
                "  dataset_root: /tmp/data",
                "  base_model_root: /tmp/base",
                "  video_transformer_root: /tmp/transformer",
                "  empty_text_embedding: /tmp/empty.pt",
                "  counterfactual_train_root: null",
                "  counterfactual_val_root: null",
                "checkpoints:",
                f"  mot_joint: {checkpoint}",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="resolved_config.yaml"):
        load_characterization_assets(manifest)


def test_asset_manifest_rejects_unresolved_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "OPEN_WAM_TEST_MISSING_CHARACTERIZATION_ROOT"
    monkeypatch.delenv(variable, raising=False)
    manifest = tmp_path / "assets.yaml"
    manifest.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "paths:",
                f"  dataset_root: ${{{variable}}}",
                "  base_model_root: /tmp/base",
                "  video_transformer_root: /tmp/transformer",
                "  empty_text_embedding: /tmp/empty.pt",
                "  counterfactual_train_root: null",
                "  counterfactual_val_root: null",
                "checkpoints: {}",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=variable):
        load_characterization_assets(manifest)


def test_counterfactual_fixture_prefers_strong_perturbed_branch() -> None:
    dataset = [
        SimpleNamespace(
            metadata={
                "counterfactual_branch": "gt",
                "counterfactual_branch_family": "demo",
                "counterfactual_branch_strength": "none",
            }
        ),
        SimpleNamespace(
            metadata={
                "counterfactual_branch": "scale_demo_0p5",
                "counterfactual_branch_family": "demo_scale",
                "counterfactual_branch_strength": "weak",
            }
        ),
        SimpleNamespace(
            metadata={
                "counterfactual_branch": "axis_pulse_x_neg",
                "counterfactual_branch_family": "axis_pulse",
                "counterfactual_branch_strength": "strong",
            }
        ),
    ]

    assert _representative_counterfactual_source_index(dataset) == 2


@pytest.mark.parametrize("method", GJD_METHODS, ids=lambda method: method.asset_id)
def test_gjd_characterization_matrix_covers_expected_training_modes(method) -> None:
    scenarios = training_scenarios_for(method)
    modes = {scenario.mode for scenario in scenarios}
    if method.gjd_ablation == "pure_joint":
        assert modes == {GJDTrainingMode.JOINT}
    else:
        assert modes == set(GJDTrainingMode)
        assert {scenario.source for scenario in scenarios} == {
            "real_demo",
            "counterfactual_dynamics",
        }

    config = apply_gjd_ablation(_load_method_config(method.config_name), method)
    assert config.policy_variant.generalist_mode_text_token is method.mode_token
    assert set(config.policy_variant.mot_generalist_training_mode_probs) == set(
        GJDTrainingMode
    )
    if method.gjd_ablation == "pure_joint":
        assert config.policy_variant.generalist_training_paradigm.value == "demo_only"
        assert (
            config.policy_variant.mot_generalist_training_mode_probs[
                GJDTrainingMode.JOINT
            ]
            == 1.0
        )
    else:
        assert (
            config.policy_variant.mot_generalist_training_mode_probs[
                GJDTrainingMode.JOINT
            ]
            == 0.6
        )


@pytest.mark.parametrize("method", GJD_METHODS, ids=lambda method: method.asset_id)
def test_gjd_characterization_matrix_covers_all_rollout_modes(method) -> None:
    assert {scenario.mode for scenario in inference_scenarios_for(method)} == set(
        GJDTrainingMode
    )


def test_gjd_standard_resolves_strict_m5_training_contract() -> None:
    method = METHOD_BY_ASSET_ID["gjd_vanilla"]
    config = apply_gjd_ablation(_load_method_config(method.config_name), method)
    sample = config.data.sample_construction
    mixture = config.data.generalist_dynamics_mixture

    assert (
        config.policy_variant.parallel_sequence_contract.value
        == "legacy_prefix_single_frame_perchunk_proprio"
    )
    assert config.policy_variant.proprio_context_mode.value == "per_chunk_additive"
    assert (
        config.policy_variant.context_condition_latent_source.value
        == "single_frame_condition_latent"
    )
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.policy_variant.joint_timestep_coupling.value == "independent"
    assert sample.condition_source_frame_offset == -1
    assert sample.target_alignment.value == "legacy"
    assert sample.mode.value == "uniform_segment"
    assert sample.sample_order_mode.value == "replacement"
    assert sample.segment_min_frames == 1000
    assert sample.segment_max_frames == 1000
    assert sample.randomize_geometry is True
    assert sample.randomize_segment_length is False
    assert sample.randomize_segment_start is False
    assert sample.require_full_segment is True
    assert sample.window_size == 64
    assert config.training.window_size == 64
    assert config.training.chunk_size == 4
    assert config.training.sample_loss_weight_mode.value == "none"
    assert config.inference.frame_chunk_size == 4
    assert config.data.action_schema.action_horizon == 16
    assert config.trainer.checkpoint_mode.value == "full_training_state"
    assert {
        "real_joint": mixture.real_joint_weight,
        "real_fdm": mixture.real_action_conditioned_video_weight,
        "real_idm": mixture.real_video_conditioned_action_weight,
        "counterfactual_fdm": mixture.counterfactual_action_conditioned_video_weight,
        "counterfactual_idm": mixture.counterfactual_video_conditioned_action_weight,
    } == {
        "real_joint": 0.6,
        "real_fdm": 0.1,
        "real_idm": 0.1,
        "counterfactual_fdm": 0.1,
        "counterfactual_idm": 0.1,
    }


def test_ground_truth_inference_contract_matches_supplied_eval_commands() -> None:
    assert asdict(DEFAULT_INFERENCE_CONTRACT) == {
        "frontend_encode_mode": "lingbot_streaming_vae",
        "inference_window_size": 30,
        "startup_model_obs_frames": 1,
        "startup_env_init_steps": 5,
        "model_frame_chunk_size": 4,
        "action_per_frame": 4,
        "action_horizon": 16,
        "execute_action_steps": 16,
        "max_timestep": 800,
        "max_chunks": 50,
    }
    assert GJD_INFERENCE_CONTRACT.max_timestep == 1500
    assert GJD_INFERENCE_CONTRACT.max_chunks == 100
    assert (
        resolve_execute_action_steps(
            None,
            action_horizon=DEFAULT_INFERENCE_CONTRACT.action_horizon,
            action_per_frame=DEFAULT_INFERENCE_CONTRACT.action_per_frame,
        )
        == 16
    )


def test_checkpoint_provenance_accepts_exact_non_gjd_contract(
    tmp_path: Path,
) -> None:
    method = METHOD_BY_ASSET_ID["mot_joint"]
    config = apply_ground_truth_training_profile(
        _load_method_config(method.config_name),
        MoTTrainingProfile.FULL_SEGMENT_W64,
    )
    config_path = tmp_path / "resolved_config.yaml"
    _write_dotted_contract(
        config_path,
        expected_checkpoint_contract(method=method, config=config),
    )

    report = checkpoint_provenance_report(
        method=method,
        expected_config=config,
        resolved_config_path=config_path,
    )

    assert report["strict_match"] is True
    assert report["mismatches"] == []
    assert len(report["resolved_config_sha256"]) == 64
    assert_checkpoint_provenance(report, asset_id=method.asset_id)


def test_checkpoint_provenance_rejects_architecture_compatible_stale_contract(
    tmp_path: Path,
) -> None:
    method = METHOD_BY_ASSET_ID["mot_action_then_video"]
    config = apply_ground_truth_training_profile(
        _load_method_config(method.config_name),
        MoTTrainingProfile.FULL_SEGMENT_W64,
    )
    contract = expected_checkpoint_contract(method=method, config=config)
    contract["policy_variant.parallel_sequence_contract"] = "default"
    config_path = tmp_path / "resolved_config.yaml"
    _write_dotted_contract(config_path, contract)

    report = checkpoint_provenance_report(
        method=method,
        expected_config=config,
        resolved_config_path=config_path,
    )

    assert report["strict_match"] is False
    assert report["mismatches"] == [
        {
            "field": "policy_variant.parallel_sequence_contract",
            "expected": "legacy_prefix_single_frame_perchunk_proprio",
            "actual": "default",
        }
    ]
    with pytest.raises(AssertionError, match="parallel_sequence_contract"):
        assert_checkpoint_provenance(report, asset_id=method.asset_id)


def test_checkpoint_provenance_requires_exact_origin_mismatch_allowlist(
    tmp_path: Path,
) -> None:
    method = METHOD_BY_ASSET_ID["mot_action_then_video"]
    config = build_checkpoint_source_contract_config(method)
    contract = expected_checkpoint_contract(method=method, config=config)
    contract["data.replay_status_policy"] = "successful_only"
    config_path = tmp_path / "resolved_config.yaml"
    _write_dotted_contract(config_path, contract)
    raw_report = checkpoint_provenance_report(
        method=method,
        expected_config=config,
        resolved_config_path=config_path,
    )

    accepted_report = apply_checkpoint_provenance_policy(
        raw_report,
        accepted_origin_mismatch_fields=("data.replay_status_policy",),
    )
    assert accepted_report["strict_match"] is False
    assert accepted_report["accepted_for_characterization"] is True
    assert_checkpoint_provenance(accepted_report, asset_id=method.asset_id)

    overbroad_report = apply_checkpoint_provenance_policy(
        raw_report,
        accepted_origin_mismatch_fields=(
            "data.replay_status_policy",
            "policy_variant.parallel_sequence_contract",
        ),
    )
    assert overbroad_report["accepted_for_characterization"] is False
    assert overbroad_report["unused_accepted_origin_mismatch_fields"] == [
        "policy_variant.parallel_sequence_contract"
    ]
    with pytest.raises(AssertionError, match="unused allowlist"):
        assert_checkpoint_provenance(overbroad_report, asset_id=method.asset_id)


def test_checkpoint_provenance_rejects_pre_cf_gjd_contract(
    tmp_path: Path,
) -> None:
    method = METHOD_BY_ASSET_ID["gjd_mode_token"]
    config = apply_gjd_ablation(_load_method_config(method.config_name), method)
    contract = expected_checkpoint_contract(method=method, config=config)
    contract["policy_variant.generalist_training_paradigm"] = "demo_only"
    contract["data.generalist_dynamics_mixture.conditional_history_frames"] = 16
    config_path = tmp_path / "resolved_config.yaml"
    _write_dotted_contract(config_path, contract)

    report = checkpoint_provenance_report(
        method=method,
        expected_config=config,
        resolved_config_path=config_path,
    )

    assert {item["field"] for item in report["mismatches"]} == {
        "policy_variant.generalist_training_paradigm",
        "data.generalist_dynamics_mixture.conditional_history_frames",
    }
    with pytest.raises(AssertionError, match="demo_only"):
        assert_checkpoint_provenance(report, asset_id=method.asset_id)


def test_parent_preflight_rejects_stale_checkpoint_before_worker_launch(
    tmp_path: Path,
) -> None:
    method = METHOD_BY_ASSET_ID["mot_action_then_video"]
    contract = expected_checkpoint_contract(
        method=method,
        config=build_checkpoint_source_contract_config(method),
    )
    contract["policy_variant.parallel_sequence_contract"] = "default"
    config_path = tmp_path / "resolved_config.yaml"
    _write_dotted_contract(config_path, contract)
    assets = SimpleNamespace(
        checkpoint_config_for=lambda _: config_path,
        checkpoints={
            method.asset_id: SimpleNamespace(
                accepted_origin_mismatch_fields=(),
            )
        },
    )

    with pytest.raises(
        AssertionError,
        match="preflight failed before staging.*",
    ):
        _preflight_checkpoint_provenance(
            assets=assets,
            selected=(method.asset_id,),
            allow_mismatch=False,
        )

    reports = _preflight_checkpoint_provenance(
        assets=assets,
        selected=(method.asset_id,),
        allow_mismatch=True,
    )
    assert reports[method.asset_id]["strict_match"] is False


def test_latent_fixture_round_trip_is_hash_verified(tmp_path: Path) -> None:
    batch = LatentWAMBatch(
        video_latents=torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2),
        actions=torch.arange(28, dtype=torch.float32).reshape(1, 4, 7),
        action_mask=torch.ones(1, 4, 7),
        state=torch.zeros(1, 1, 8),
        task_text=("task",),
        text_context=torch.ones(1, 2, 4),
        negative_text_context=torch.zeros(1, 2, 4),
        metadata=({"loss_frame_start": 1},),
    )
    metadata_path = save_latent_batch_fixture(
        tmp_path,
        fixture_id="sample",
        batch=batch,
        provenance={"sample_index": 0},
    )

    restored = load_latent_batch_fixture(metadata_path)

    assert restored.task_text == batch.task_text
    assert restored.metadata == batch.metadata
    for field_name in (
        "video_latents",
        "actions",
        "action_mask",
        "state",
        "text_context",
    ):
        torch.testing.assert_close(
            getattr(restored, field_name), getattr(batch, field_name)
        )


def test_tensor_fingerprint_records_shape_statistics_and_stable_probes() -> None:
    tensor = torch.arange(100, dtype=torch.float32)
    fingerprint = tensor_fingerprint(tensor, probe_count=5)

    assert fingerprint["shape"] == [100]
    assert fingerprint["finite_count"] == 100
    assert fingerprint["mean"] == pytest.approx(49.5)
    assert fingerprint["probe_indices"] == [0, 25, 50, 74, 99]
    assert fingerprint["probe_values"] == [0.0, 25.0, 50.0, 74.0, 99.0]
    assert len(fingerprint["content_sha256"]) == 64

    changed = tensor.clone()
    changed[37] += 1
    assert (
        tensor_fingerprint(changed, probe_count=5)["content_sha256"]
        != fingerprint["content_sha256"]
    )


def test_tensor_fingerprint_uses_json_null_statistics_for_empty_tensor() -> None:
    fingerprint = tensor_fingerprint(torch.empty(1, 0, 4))

    assert fingerprint["numel"] == 0
    assert fingerprint["mean"] is None
    assert fingerprint["l2"] is None
    assert fingerprint["probe_values"] == []


def test_tensor_fingerprint_supports_one_probe() -> None:
    fingerprint = tensor_fingerprint(
        torch.arange(10, dtype=torch.float32),
        probe_count=1,
    )

    assert fingerprint["probe_indices"] == [0]
    assert fingerprint["probe_values"] == [0.0]


def test_report_comparison_supports_explicit_float_tolerance() -> None:
    expected = {"shape": [1, 4], "mean": 1.0, "mode": "joint"}
    close = {"shape": [1, 4], "mean": 1.0001, "mode": "joint"}
    changed = {"shape": [1, 5], "mean": 1.2, "mode": "fdm"}

    assert (
        compare_characterization_reports(
            expected,
            close,
            tolerance=ComparisonTolerance(absolute=1e-3, relative=0.0),
        )
        == []
    )
    differences = compare_characterization_reports(expected, changed)
    assert any("shape[1]" in difference for difference in differences)
    assert any(".mean" in difference for difference in differences)
    assert any(".mode" in difference for difference in differences)


def test_characterization_tolerance_is_limited_to_distributed_numeric_fields() -> None:
    expected = {
        "scenarios": {
            "sample": {
                "metrics": {"loss": 1.0},
                "outputs": {"mean": 2.0},
                "gradients": {"video_backbone": {"absolute_sum": 3.0}},
                "optimizer_step": {
                    "distributed_numeric": {
                        "parameter_groups": {
                            "video_backbone": {"after": {"sum": 4.0}}
                        }
                    },
                    "scheduler": {"lr": 5.0e-6},
                },
            }
        }
    }
    actual = {
        "scenarios": {
            "sample": {
                "metrics": {"loss": 1.0001},
                "outputs": {"mean": 2.0001},
                "gradients": {"video_backbone": {"absolute_sum": 3.0001}},
                "optimizer_step": {
                    "distributed_numeric": {
                        "parameter_groups": {
                            "video_backbone": {"after": {"sum": 4.0001}}
                        }
                    },
                    "scheduler": {"lr": 5.1e-6},
                },
            }
        }
    }
    differences = compare_characterization_reports(
        expected,
        actual,
        tolerance=ComparisonTolerance(absolute=0.0, relative=0.0),
        tolerance_for_path=_numeric_tolerance_resolver(
            ComparisonTolerance(absolute=1e-3, relative=0.0)
        ),
    )

    assert len(differences) == 3
    assert any(".metrics.loss" in difference for difference in differences)
    assert any(".outputs.mean" in difference for difference in differences)
    assert any(".scheduler.lr" in difference for difference in differences)
    assert not any(".gradients." in difference for difference in differences)
    assert not any(
        ".optimizer_step.distributed_numeric." in difference
        for difference in differences
    )

    actual["scenarios"]["sample"]["optimizer_step"]["distributed_numeric"][
        "parameter_groups"
    ]["video_backbone"]["after"]["sum"] = 4.3
    differences = compare_characterization_reports(
        expected,
        actual,
        tolerance=ComparisonTolerance(absolute=0.0, relative=0.0),
        tolerance_for_path=_numeric_tolerance_resolver(
            ComparisonTolerance(absolute=1e-3, relative=0.0)
        ),
    )

    assert any(
        ".optimizer_step.distributed_numeric.parameter_groups.video_backbone.after.sum"
        in difference
        for difference in differences
    )


def test_characterization_delta_tolerance_propagates_aggregate_error() -> None:
    expected = {
        "optimizer_step": {
            "distributed_numeric": {
                "parameter_groups": {
                    "video_backbone": {"delta": {"absolute_sum": 1.0}}
                }
            }
        }
    }
    actual = json.loads(json.dumps(expected))
    delta = actual["optimizer_step"]["distributed_numeric"]["parameter_groups"][
        "video_backbone"
    ]["delta"]
    delta["absolute_sum"] = 1.5

    resolver = _numeric_tolerance_resolver(
        ComparisonTolerance(absolute=0.0, relative=0.0)
    )
    assert compare_characterization_reports(
        expected,
        actual,
        tolerance=ComparisonTolerance(absolute=0.0, relative=0.0),
        tolerance_for_path=resolver,
    ) == []

    delta["absolute_sum"] = 1.500001
    assert compare_characterization_reports(
        expected,
        actual,
        tolerance=ComparisonTolerance(absolute=0.0, relative=0.0),
        tolerance_for_path=resolver,
    )


def test_resume_post_update_metrics_have_narrow_cross_job_tolerance() -> None:
    expected = {
        "uninterrupted_update": {
            "scenario": {"metrics": {"action_mse": 0.0002}}
        },
        "resumed_update": {"scenario": {"metrics": {"action_mse": 0.0002}}},
        "first_update": {"scenario": {"metrics": {"action_mse": 0.0002}}},
    }
    actual = json.loads(json.dumps(expected))
    accepted_delta = RESUME_POST_UPDATE_METRIC_TOLERANCE.absolute
    actual["uninterrupted_update"]["scenario"]["metrics"]["action_mse"] += accepted_delta
    actual["resumed_update"]["scenario"]["metrics"]["action_mse"] += accepted_delta

    differences = compare_characterization_reports(
        expected,
        actual,
        tolerance=ComparisonTolerance(absolute=0.0, relative=0.0),
        tolerance_for_path=_numeric_tolerance_resolver(
            ComparisonTolerance(absolute=0.0, relative=0.0)
        ),
        path="gjd_mode_token.resume.json",
    )

    assert differences == []
    actual["resumed_update"]["scenario"]["metrics"]["action_mse"] += 1e-9
    differences = compare_characterization_reports(
        expected,
        actual,
        tolerance=ComparisonTolerance(absolute=0.0, relative=0.0),
        tolerance_for_path=_numeric_tolerance_resolver(
            ComparisonTolerance(absolute=0.0, relative=0.0)
        ),
        path="gjd_mode_token.resume.json",
    )
    assert any(".resumed_update.scenario.metrics.action_mse" in item for item in differences)

    actual = json.loads(json.dumps(expected))
    actual["first_update"]["scenario"]["metrics"]["action_mse"] += 1e-9
    differences = compare_characterization_reports(
        expected,
        actual,
        tolerance=ComparisonTolerance(absolute=0.0, relative=0.0),
        tolerance_for_path=_numeric_tolerance_resolver(
            ComparisonTolerance(absolute=0.0, relative=0.0)
        ),
        path="gjd_mode_token.resume.json",
    )
    assert any(".first_update.scenario.metrics.action_mse" in item for item in differences)


def test_resume_report_uses_stable_output_schema_version() -> None:
    assert RESUME_REPORT_SCHEMA_VERSION == 2


def test_fixture_directory_comparison_requires_byte_identical_files(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    (expected / "fixture.json").write_text('{"value": 1}\n', encoding="utf-8")
    (actual / "fixture.json").write_text('{"value": 1}\n', encoding="utf-8")

    assert compare_fixture_directories(expected, actual) == []

    (actual / "fixture.json").write_text('{"value": 2}\n', encoding="utf-8")
    differences = compare_fixture_directories(expected, actual)
    assert len(differences) == 1
    assert "fixture.json: expected sha256" in differences[0]


def test_checkpoint_preflight_rejects_missing_trained_mot_keys(
    tmp_path: Path,
) -> None:
    pipeline = torch.nn.Module()
    pipeline.visual_tower = torch.nn.Module()
    pipeline.visual_tower.core = torch.nn.Linear(2, 2)
    pipeline.policy_variant = torch.nn.Module()
    checkpoint_path = tmp_path / "model_state.pt"
    state_dict = pipeline.state_dict()
    del state_dict["visual_tower.core.weight"]
    torch.save(state_dict, checkpoint_path)

    with pytest.raises(
        AssertionError,
        match="missing 1 trained MoT keys",
    ):
        _load_pipeline_checkpoint(pipeline, checkpoint_path)


def test_training_assertion_requires_finite_nonzero_core_gradient_groups() -> None:
    method = METHOD_BY_ASSET_ID["gjd_mode_token"]
    scenario = training_scenarios_for(method)[0]
    report = {
        "metrics": {"loss": 1.0},
        "mode": "joint",
        "source": "real_demo",
        "input": {},
        "outputs": {
            "output": {
                "finite_count": 1,
                "numel": 1,
            }
        },
        "gradients": {
            "action_expert": {
                "parameter_elements": 2,
                "finite_elements": 2,
                "nonzero_elements": 2,
            },
            "video_backbone": {
                "parameter_elements": 2,
                "finite_elements": 1,
                "nonzero_elements": 1,
            },
        },
    }

    with pytest.raises(AssertionError, match="non-finite gradients"):
        _assert_training_scenario(report, method=method, scenario=scenario)

    report["gradients"]["video_backbone"]["finite_elements"] = 2
    with pytest.raises(AssertionError, match="generalist_mode_token"):
        _assert_training_scenario(report, method=method, scenario=scenario)


def test_training_assertion_contains_target_only_layout_to_gjd_conditionals() -> None:
    method = METHOD_BY_ASSET_ID["gjd_vanilla"]
    scenario = next(
        item
        for item in training_scenarios_for(method)
        if item.scenario_id == "counterfactual_fdm"
    )
    report = {
        "metrics": {"loss": 1.0},
        "mode": scenario.mode.value,
        "source": scenario.source,
        "input": {
            "history_frames": 1,
            "loss_frame_start": 1,
            "loss_frame_end": 32,
            "singleton_chunk_frame": 0,
            "chunk_origin_frame": 1,
            "sampled_chunk_size": 4,
            "video_latents": {"shape": [1, 48, 32, 8, 16]},
            "actions": {"shape": [1, 128, 7]},
            "action_mask": {
                "shape": [1, 128, 7],
                "numel": 896,
                "finite_count": 896,
                "min": 0.0,
                "max": 1.0,
            },
            "leading_action_mask_sum": 0.0,
            "future_action_mask_sum": 868.0,
        },
        "outputs": {
            "output": {
                "finite_count": 1,
                "numel": 1,
            }
        },
        "gradients": {
            "action_expert": {
                "parameter_elements": 2,
                "finite_elements": 2,
                "nonzero_elements": 2,
            },
            "video_backbone": {
                "parameter_elements": 2,
                "finite_elements": 2,
                "nonzero_elements": 2,
            },
        },
    }

    _assert_training_scenario(report, method=method, scenario=scenario)
    report["input"]["singleton_chunk_frame"] = None
    with pytest.raises(AssertionError, match="target-only t0 contract"):
        _assert_training_scenario(report, method=method, scenario=scenario)
    report["input"]["singleton_chunk_frame"] = 0
    report["input"]["leading_action_mask_sum"] = 1.0
    with pytest.raises(AssertionError, match="dummy four-action group"):
        _assert_training_scenario(report, method=method, scenario=scenario)

    non_gjd_method = METHOD_BY_ASSET_ID["mot_joint"]
    non_gjd_scenario = training_scenarios_for(non_gjd_method)[0]
    _assert_training_scenario(
        {
            **report,
            "mode": None,
            "source": "real_demo",
        },
        method=non_gjd_method,
        scenario=non_gjd_scenario,
    )


def test_optimizer_characterization_executes_adamw_and_scheduler_step() -> None:
    model = torch.nn.Module()
    model.action_expert = torch.nn.Linear(3, 2)
    model.visual_tower = torch.nn.Module()
    model.visual_tower.core = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: float(step + 1),
    )
    loss = (
        model.action_expert(torch.ones(1, 3)).square().mean()
        + model.visual_tower.core(torch.ones(1, 3)).square().mean()
    )
    loss.backward()

    class _Strategy:
        @staticmethod
        def unscale_(optimizer) -> None:
            del optimizer

        @staticmethod
        def clip_grad_norm_(parameters, max_norm):
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

        @staticmethod
        def optimizer_step(optimizer) -> None:
            optimizer.step()

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(
                gradient_accumulation_steps=10,
                max_grad_norm=2.0,
            )
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=_Strategy(),
    )

    report = _execute_characterization_optimizer_step(
        runtime=runtime,
        scenario_id="full_segment_w64",
    )

    assert report["configured_gradient_accumulation_steps"] == 10
    assert report["scheduler"]["last_epoch_after"] == 1
    assert report["optimizer_state_schema"]["per_rank"][0]["state_entries"] == 4
    assert {
        probe["group"] for probe in report["distributed_numeric"]["parameter_probes"]
    } == {"action_expert", "video_backbone"}
    _assert_optimizer_step(
        report,
        method=METHOD_BY_ASSET_ID["mot_joint"],
    )


def test_runtime_state_digests_cover_model_and_optimizer_bytes() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 3)).square().mean()
    loss.backward()
    optimizer.step()

    model_before = _distributed_module_digest(model)
    optimizer_before = _distributed_optimizer_digest(model, optimizer)

    with torch.no_grad():
        model.weight.view(-1)[0] += 1.0
        optimizer.state[model.weight]["exp_avg"].view(-1)[0] += 1.0

    model_after = _distributed_module_digest(model)
    optimizer_after = _distributed_optimizer_digest(model, optimizer)

    assert model_before["per_rank"][0]["sha256"] != model_after["per_rank"][0]["sha256"]
    assert (
        optimizer_before["per_rank"][0]["sha256"]
        != optimizer_after["per_rank"][0]["sha256"]
    )
    assert model_before["per_rank"][0]["total_bytes"] > 0
    assert optimizer_before["per_rank"][0]["total_bytes"] > 0


def test_resume_update_comparison_bounds_distributed_aggregate_noise() -> None:
    expected = {
        "scenario": {"outputs": {"sha256": "same"}},
        "optimizer_step": {
            "distributed_numeric": {
                "parameter_groups": {"video_backbone": {"delta": {"sum": 1.0}}},
                "parameter_probes": [{"local_flat_index": 1}],
            }
        },
    }
    actual = {
        "scenario": {"outputs": {"sha256": "same"}},
        "optimizer_step": {
            "distributed_numeric": {
                "parameter_groups": {"video_backbone": {"delta": {"sum": 1.5}}},
                "parameter_probes": [{"local_flat_index": 99}],
            }
        },
    }

    assert _resume_update_differences(expected, actual) == []

    actual["optimizer_step"]["distributed_numeric"]["parameter_groups"][
        "video_backbone"
    ]["delta"]["sum"] = 1.500001
    assert _resume_update_differences(expected, actual)


def test_final_resume_state_contract_keeps_structure_but_not_hashes() -> None:
    expected = {
        "model": {
            "per_rank": [
                {
                    "rank": 0,
                    "sha256": "before",
                    "tensor_count": 2,
                    "total_bytes": 16,
                    "components": {
                        "video_backbone": {
                            "sha256": "before",
                            "tensor_count": 2,
                            "total_bytes": 16,
                        }
                    },
                }
            ]
        },
        "optimizer": {
            "per_rank": [
                {
                    "rank": 0,
                    "sha256": "before",
                    "tensor_count": 4,
                    "total_bytes": 32,
                    "state_entries": 2,
                }
            ]
        },
        "scheduler": {"last_epoch": 2},
        "strategy": {"grad_scaler": None},
        "train_state": {"optimizer_step": 2},
    }
    actual = json.loads(json.dumps(expected))
    actual["model"]["per_rank"][0]["sha256"] = "after"
    actual["model"]["per_rank"][0]["components"]["video_backbone"]["sha256"] = "after"
    actual["optimizer"]["per_rank"][0]["sha256"] = "after"

    assert _runtime_state_contract_differences(expected, actual) == []

    actual["optimizer"]["per_rank"][0]["state_entries"] = 3
    assert _runtime_state_contract_differences(expected, actual)


def test_persisted_resume_continuation_uses_stable_output_contract() -> None:
    report = {
        "scenario": {
            "scenario_id": "after_checkpoint",
            "cuda_peak_memory_bytes": 123,
            "gradients": {
                "video_backbone": {
                    "parameter_tensors": 2,
                    "parameter_elements": 8,
                    "finite_elements": 8,
                }
            },
            "outputs": {
                "decoder.action_pred": {
                    "content_sha256": "unstable",
                    "dtype": "torch.bfloat16",
                    "finite_count": 8,
                    "l1": 4.0,
                    "local_shape": [1, 4, 2],
                    "numel": 8,
                    "shape": [1, 4, 2],
                }
            },
        },
        "optimizer_step": {
            "distributed_numeric": {"grad_norm": 1.0},
            "scheduler": {"last_epoch_after": 2},
        },
        "train_state": {"optimizer_step": 2},
    }

    stable = _resume_update_report_contract(
        report,
        preserve_output_values=False,
    )
    full = _resume_update_report_contract(
        report,
        preserve_output_values=True,
    )

    assert stable["scenario"]["outputs"]["decoder.action_pred"] == {
        "dtype": "torch.bfloat16",
        "finite_count": 8,
        "local_shape": [1, 4, 2],
        "numel": 8,
        "shape": [1, 4, 2],
    }
    assert (
        full["scenario"]["outputs"]["decoder.action_pred"]["content_sha256"]
        == "unstable"
    )


def test_inference_characterization_requires_three_advancing_states() -> None:
    method = METHOD_BY_ASSET_ID["mot_joint"]
    progression = [
        {
            "step_index": index + 1,
            "cursor_current_start_frame": 5 + index * 4,
        }
        for index in range(INFERENCE_CHARACTERIZATION_CHUNKS)
    ]

    _assert_inference_progression(
        progression,
        method=method,
        mode=None,
    )

    progression[-1]["cursor_current_start_frame"] = progression[-2][
        "cursor_current_start_frame"
    ]
    with pytest.raises(AssertionError, match="cursor monotonically"):
        _assert_inference_progression(
            progression,
            method=method,
            mode=None,
        )


@pytest.mark.parametrize(
    ("asset_id", "mode"),
    (
        ("mot_joint", None),
        ("gjd_mode_token", GJDTrainingMode.JOINT),
    ),
)
def test_cache_rollover_sentinels_cross_shared_window(
    asset_id: str,
    mode: GJDTrainingMode | None,
) -> None:
    method = METHOD_BY_ASSET_ID[asset_id]
    progression = [
        {
            "step_index": index + 1,
            "cursor_current_start_frame": 5 + index * 4,
            "past_clean_latent_frames": min(
                64,
                5 if index == 0 else (index + 1) * 4,
            ),
            "past_clean_action_steps": min(256, (index + 1) * 16),
        }
        for index in range(CACHE_ROLLOVER_CHARACTERIZATION_CHUNKS)
    ]

    _assert_inference_progression(
        progression,
        method=method,
        mode=mode,
        expected_chunk_count=CACHE_ROLLOVER_CHARACTERIZATION_CHUNKS,
    )
    _assert_cache_rollover(
        progression,
        method=method,
        mode=mode,
    )

    progression[-1]["past_clean_latent_frames"] = 68
    with pytest.raises(AssertionError, match="cache cap"):
        _assert_cache_rollover(
            progression,
            method=method,
            mode=mode,
        )


def test_shared_infrastructure_phase_selection_is_intentionally_narrow() -> None:
    _validate_phase_asset_selection(
        selected=CACHE_ROLLOVER_ASSET_IDS,
        phases=("cache_rollover",),
    )
    _validate_phase_asset_selection(
        selected=(FULL_STATE_RESUME_ASSET_ID,),
        phases=("resume",),
    )

    with pytest.raises(ValueError, match="shared-runtime sentinels"):
        _validate_phase_asset_selection(
            selected=("mot_video_then_action",),
            phases=("cache_rollover",),
        )
    with pytest.raises(ValueError, match="requires exactly"):
        _validate_phase_asset_selection(
            selected=("mot_joint",),
            phases=("resume",),
        )


def test_checkpoint_stage_does_not_reuse_same_size_different_source(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    destination = tmp_path / "stage" / "model_state.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"other")

    _stage_file(first, destination, refresh=False)
    assert destination.read_bytes() == b"first"

    _stage_file(second, destination, refresh=False)
    assert destination.read_bytes() == b"other"


def test_checkpoint_stage_detects_in_place_same_size_change(tmp_path: Path) -> None:
    source = tmp_path / "model_state.pt"
    destination = tmp_path / "stage" / "model_state.pt"
    source.write_bytes(b"first")

    _stage_file(source, destination, refresh=False)
    assert destination.read_bytes() == b"first"

    source.write_bytes(b"other")
    _stage_file(source, destination, refresh=False)
    assert destination.read_bytes() == b"other"


def test_static_stage_detects_in_place_source_file_change(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "weights.bin"
    source_file.write_bytes(b"first")
    destination = tmp_path / "stage"

    _stage_directory(source, destination, refresh=False)
    assert (destination / "weights.bin").read_bytes() == b"first"

    source_file.write_bytes(b"other")
    _stage_directory(source, destination, refresh=False)
    assert (destination / "weights.bin").read_bytes() == b"other"


def test_staging_source_to_itself_is_a_noop(tmp_path: Path) -> None:
    source_file = tmp_path / "model_state.pt"
    source_file.write_bytes(b"weights")
    source_directory = tmp_path / "model"
    source_directory.mkdir()
    (source_directory / "config.json").write_text("{}\n", encoding="utf-8")

    assert _stage_file(source_file, source_file, refresh=True) == source_file
    assert (
        _stage_directory(source_directory, source_directory, refresh=True)
        == source_directory
    )
    assert source_file.read_bytes() == b"weights"
    assert (source_directory / "config.json").is_file()


def test_worker_command_finds_torchrun_beside_active_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    interpreter = tmp_path / "bin" / "python"
    torchrun = interpreter.with_name("torchrun")
    interpreter.parent.mkdir()
    interpreter.touch()
    torchrun.touch()
    monkeypatch.setattr(
        "tests.characterization.run_mot_refactor_characterization.sys.executable",
        str(interpreter),
    )
    monkeypatch.setattr(
        "tests.characterization.run_mot_refactor_characterization.shutil.which",
        lambda _: None,
    )

    command, environment = _worker_command(
        phase="training",
        asset_id="mot_joint",
        assets_path=tmp_path / "assets.yaml",
        fixture_root=tmp_path / "fixtures",
        report_path=tmp_path / "report.json",
        checkpoint=tmp_path / "model_state.pt",
        checkpoint_config=tmp_path / "resolved_config.yaml",
        base_model_root=tmp_path / "base",
        video_transformer_root=tmp_path / "transformer",
        training_world_size=4,
        cuda_devices="0,1,2,3",
        fsdp_cpu_offload=True,
        disable_nccl_shm=True,
        allow_provenance_mismatch=False,
    )

    assert command[0] == str(torchrun)
    assert environment["OPEN_WAM_ENABLE_FIXED128_ROLLOUT_CONTEXT"] == "0"
    assert environment["OPEN_WAM_FSDP_CPU_OFFLOAD"] == "1"
    assert environment["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


def test_resume_worker_keeps_all_distributed_cuda_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tests.characterization.run_mot_refactor_characterization.shutil.which",
        lambda _: "/test/bin/torchrun",
    )

    command, environment = _worker_command(
        phase="resume",
        asset_id="gjd_mode_token",
        assets_path=tmp_path / "assets.yaml",
        fixture_root=tmp_path / "fixtures",
        report_path=tmp_path / "report.json",
        checkpoint=tmp_path / "model_state.pt",
        checkpoint_config=tmp_path / "resolved_config.yaml",
        base_model_root=tmp_path / "base",
        video_transformer_root=tmp_path / "transformer",
        training_world_size=4,
        cuda_devices="0,1,2,3",
        fsdp_cpu_offload=True,
        disable_nccl_shm=False,
        allow_provenance_mismatch=False,
    )

    assert command[:3] == [
        "/test/bin/torchrun",
        "--standalone",
        "--nproc-per-node=4",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


@pytest.mark.parametrize("method", ALL_METHODS, ids=lambda method: method.asset_id)
def test_training_cli_smoke_command_uses_real_entrypoint_and_one_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method,
) -> None:
    assets = _fake_end_to_end_assets(tmp_path)
    monkeypatch.setattr(
        "tests.characterization.mot_refactor_end_to_end._find_torchrun",
        lambda: "/test/bin/torchrun",
    )

    command = build_training_cli_smoke_command(
        method=method,
        assets=assets,
        checkpoint=tmp_path / "staged" / "model_state.pt",
        save_root=tmp_path / "output",
        world_size=4,
    )
    overrides = _command_overrides(command)

    assert command[:4] == [
        "/test/bin/torchrun",
        "--standalone",
        "--nproc-per-node=4",
        "-m",
    ]
    assert command[4:6] == [
        "open_wam.training.train",
        "--config-name",
    ]
    assert _option_value(command, "--config-name") == method.config_name
    assert overrides["training.num_steps"] == "1"
    assert overrides["training.gradient_accumulation_steps"] == "1"
    assert overrides["trainer.strategy"] == "fsdp"
    assert overrides["trainer.enable_checkpointing"] == "false"
    assert overrides["trainer.enable_jsonl_logging"] == "true"
    if not method.is_gjd:
        assert overrides["data.sample_construction.window_size"] == "64"
        assert not any(
            key.startswith("data.generalist_dynamics_mixture.") for key in overrides
        )
    elif method.gjd_ablation == "pure_joint":
        assert overrides["data.generalist_dynamics_mixture.train_latent_root"] == "null"
        assert overrides["data.generalist_dynamics_mixture.val_latent_root"] == "null"
    else:
        assert overrides["data.generalist_dynamics_mixture.train_latent_root"] == str(
            assets.counterfactual_train_root
        )
        assert overrides["data.generalist_dynamics_mixture.val_latent_root"] == str(
            assets.counterfactual_val_root
        )
    if method.is_gjd:
        assert overrides["policy_variant.generalist_mode_text_token"] == (
            "true" if method.mode_token else "false"
        )


@pytest.mark.parametrize("method", ALL_METHODS, ids=lambda method: method.asset_id)
def test_libero_rollout_command_matches_maintained_contract(
    tmp_path: Path,
    method,
) -> None:
    assets = _fake_end_to_end_assets(tmp_path)
    command = build_libero_rollout_command(
        method=method,
        assets=assets,
        output_root=tmp_path / "rollouts",
        task_id=2,
        episode_idx=3,
        seed=3,
    )

    assert command[1].endswith("scripts/run_libero_mot_visualization.py")
    assert _option_value(command, "--cfg").endswith(
        f"{method.asset_id}/resolved_config.yaml"
    )
    assert _option_value(command, "--frontend-encode-mode") == ("lingbot_streaming_vae")
    assert _option_value(command, "--mot-inference-window-size") == "30"
    assert _option_value(command, "--max-timestep") == (
        "1500" if method.is_gjd else "800"
    )
    assert _option_value(command, "--max-chunks") == ("100" if method.is_gjd else "50")
    assert _option_value(command, "--startup-model-obs-frames") == "1"
    assert _option_value(command, "--startup-env-init-steps") == "5"
    assert "--execute-action-steps" not in command
    assert ("--mot-generalist-rollout-mode" in command) is method.is_gjd
    if method.is_gjd:
        assert _option_value(command, "--mot-generalist-rollout-mode") == "joint"


def test_model_only_cli_stage_does_not_expose_sibling_training_state(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "model_state.pt"
    source.write_bytes(b"weights")
    (source_root / "train_state.json").write_text("{}\n", encoding="utf-8")
    (source_root / "full_training_state.pt").write_bytes(b"full")

    staged = stage_model_only_checkpoint(source, tmp_path / "stage")

    assert staged.is_symlink()
    assert staged.read_bytes() == b"weights"
    assert not (staged.parent / "train_state.json").exists()
    assert not (staged.parent / "full_training_state.pt").exists()


def test_end_to_end_environment_supports_nccl_shared_memory_workaround(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    environment = characterization_environment(
        cuda_devices="0,1,2,3",
        fsdp_cpu_offload=True,
        disable_nccl_shm=True,
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert environment["OPEN_WAM_FSDP_CPU_OFFLOAD"] == "1"
    assert environment["NCCL_SHM_DISABLE"] == "1"
    assert environment["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert environment["NCCL_CUMEM_HOST_ENABLE"] == "0"


def test_end_to_end_environment_preserves_compile_thread_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "4")

    environment = characterization_environment(
        cuda_devices="0",
        fsdp_cpu_offload=False,
    )

    assert environment["TORCHINDUCTOR_COMPILE_THREADS"] == "4"


def test_libero_rollout_report_requires_actions_chunks_and_video(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    video_path = artifact_root / "0_False_test.mp4"
    video_path.write_bytes(b"deterministic-video")
    action_path = artifact_root / "0_False_test_actions.jsonl"
    action_path.write_text(
        "\n".join(
            json.dumps({"action_index": index, "action": [float(index)] * 7})
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = artifact_root / "0_False_test.json"
    summary_path.write_text(
        json.dumps(
            {
                "pipeline": "open_wam_mot",
                "benchmark": "libero_10",
                "task_id": 0,
                "episode_idx": 0,
                "seed": 0,
                "success": False,
                "terminal": False,
                "chunk_count": 1,
                "env_timestep": 7,
                "action_count": 2,
                "startup_model_obs_frames": 1,
                "startup_env_init_steps": 5,
                "action_trace_path": str(action_path),
                "video_path": str(video_path),
            }
        ),
        encoding="utf-8",
    )
    (artifact_root / "0_False_test_chunks.json").write_text(
        json.dumps(
            [
                {
                    "phase": "infer",
                    "chunk_index": 0,
                    "action_shape": [16, 7],
                    "execute_action_steps": 16,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = collect_libero_rollout_report(
        method=METHOD_BY_ASSET_ID["mot_joint"],
        artifact_root=artifact_root,
    )

    assert report["action_trace"]["shape"] == [2, 7]
    assert report["action_count"] == 2
    assert len(report["action_trace_sha256"]) == 64
    assert len(report["comparison_video_sha256"]) == 64
    assert len(report["inference_chunks"]) == 1


def test_comparison_projection_quantizes_parallel_gradient_nonzero_count() -> None:
    first = {
        "parameter_elements": 2_696_904_732,
        "finite_elements": 2_696_904_732,
        "nonzero_elements": 2_696_903_696,
    }
    second = {
        "parameter_elements": 2_696_904_732,
        "finite_elements": 2_696_904_732,
        "nonzero_elements": 2_696_903_643,
    }

    assert _comparison_projection(first) == _comparison_projection(second)
    assert _comparison_projection(first)["nonzero_density"] == 1.0


def test_comparison_projection_ignores_path_sized_resume_metadata() -> None:
    first = {
        "checkpoint_artifacts": {
            "files": {
                "resolved_config.yaml": {
                    "nonempty": True,
                    "size_bytes": 100,
                },
                "model_state.pt": {
                    "nonempty": True,
                    "size_bytes": 200,
                },
            }
        }
    }
    second = json.loads(json.dumps(first))
    second["checkpoint_artifacts"]["files"]["resolved_config.yaml"]["size_bytes"] = 999

    assert _comparison_projection(first) == _comparison_projection(second)

    second["checkpoint_artifacts"]["files"]["model_state.pt"]["size_bytes"] = 201
    assert _comparison_projection(first) != _comparison_projection(second)


def _write_dotted_contract(path: Path, contract: dict[str, object]) -> None:
    import yaml

    nested: dict[str, object] = {}
    for dotted_path, value in contract.items():
        current = nested
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            child = current.setdefault(part, {})
            assert isinstance(child, dict)
            current = child
        current[parts[-1]] = value
    path.write_text(yaml.safe_dump(nested, sort_keys=True), encoding="utf-8")


def _fake_end_to_end_assets(tmp_path: Path):
    checkpoint_paths: dict[str, Path] = {}
    config_paths: dict[str, Path] = {}
    for asset_id in METHOD_BY_ASSET_ID:
        checkpoint_paths[asset_id] = tmp_path / asset_id / "model_state.pt"
        config_paths[asset_id] = tmp_path / asset_id / "resolved_config.yaml"
    return SimpleNamespace(
        dataset_root=tmp_path / "data",
        base_model_root=tmp_path / "base",
        video_transformer_root=tmp_path / "transformer",
        empty_text_embedding=tmp_path / "empty.pt",
        counterfactual_train_root=tmp_path / "cf_train",
        counterfactual_val_root=tmp_path / "cf_val",
        checkpoint_for=lambda asset_id: checkpoint_paths[asset_id],
        checkpoint_config_for=lambda asset_id: config_paths[asset_id],
    )


def _command_overrides(command: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for index, token in enumerate(command):
        if token != "--set":
            continue
        key, value = command[index + 1].split("=", maxsplit=1)
        overrides[key] = value
    return overrides


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]
