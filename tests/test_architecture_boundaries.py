from __future__ import annotations

import ast
from pathlib import Path

from open_wam.configs import (
    SharedVideoTransformerConfig,
    load_experiment_config,
    load_local_path_registry,
    read_yaml_with_local_paths,
)
from open_wam.models.video_backbone.config import (
    SharedVideoTransformerConfig as LegacySharedVideoTransformerConfig,
)
from open_wam.utils.config_loader import (
    load_experiment_config as LegacyLoadExperimentConfig,
)
from open_wam.utils.local_paths import (
    load_local_path_registry as LegacyLoadLocalPathRegistry,
    read_yaml_with_local_paths as LegacyReadYamlWithLocalPaths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "open_wam"


def _absolute_imports(package: str) -> set[str]:
    imports: set[str] = set()
    for path in (PACKAGE_ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    return imports


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_method_definitions(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_config_package_does_not_depend_on_runtime_implementations() -> None:
    forbidden_prefixes = (
        "open_wam.data",
        "open_wam.models",
        "open_wam.pipelines",
        "open_wam.training",
    )

    violations = sorted(
        imported
        for imported in _absolute_imports("configs")
        if imported.startswith(forbidden_prefixes)
    )

    assert violations == []


def test_utils_package_does_not_depend_on_model_implementations() -> None:
    violations = sorted(
        imported
        for imported in _absolute_imports("utils")
        if imported.startswith("open_wam.models")
    )

    assert violations == []


def test_core_packages_do_not_depend_on_optional_runtime_surfaces() -> None:
    forbidden_prefixes = (
        "open_wam.evals",
        "open_wam.integrations",
        "open_wam.planning",
        "open_wam.simulators",
    )
    violations: dict[str, list[str]] = {}

    for package in ("configs", "data", "models", "pipelines", "runtime", "training"):
        package_violations = sorted(
            imported
            for imported in _absolute_imports(package)
            if imported.startswith(forbidden_prefixes)
        )
        if package_violations:
            violations[package] = package_violations

    assert violations == {}


def test_lerobot_latent_repository_io_has_one_storage_owner() -> None:
    storage_owned = {
        "LocalEpisodeWindow",
        "LocalRepoBundle",
        "discover_local_lerobot_repo_bundles",
        "scan_local_latent_windows",
        "load_lerobot_v2_local_metadata",
        "resolve_latent_root",
        "reshape_latent_payload",
    }
    storage_definitions = _top_level_definitions(PACKAGE_ROOT / "data" / "lerobot_v2_latent_storage.py")
    dataset_definitions = _top_level_definitions(PACKAGE_ROOT / "data" / "lerobot_v2_latent.py")

    assert storage_owned <= storage_definitions
    assert storage_owned.isdisjoint(dataset_definitions)


def test_lerobot_latent_segment_geometry_has_one_owner() -> None:
    geometry_functions = {
        "compact_boundary_start_range",
        "resolve_compact_boundary_segment",
        "resolve_rollout_parity_boundary_segment",
        "rollout_parity_start_range",
    }
    geometry_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "latent_segment_geometry.py"
    )
    dataset_methods = _class_method_definitions(
        PACKAGE_ROOT / "data" / "lerobot_v2_latent.py",
        "UniformSegmentLocalLeRobotLatentDataset",
    )

    assert geometry_functions <= geometry_definitions
    assert {
        f"_{name}" for name in geometry_functions
    }.isdisjoint(dataset_methods)


def test_row_action_target_transform_has_one_owner() -> None:
    transform_path = PACKAGE_ROOT / "data" / "row_action_targets.py"
    transform_definitions = _top_level_definitions(transform_path)
    assert {"build_row_action_targets", "resolve_row_key"} <= transform_definitions

    adapter_classes = {
        "lerobot_v2.py": "LeRobotV2WindowDataset",
        "lerobot_v2_latent.py": "LocalLeRobotLatentWindowDataset",
        "lerobot_consortium.py": "LeRobotConsortiumWindowDataset",
    }
    for filename, class_name in adapter_classes.items():
        method = _class_method(
            PACKAGE_ROOT / "data" / filename,
            class_name,
            "_build_action_targets",
        )
        assert len(method.body) == 1
        assert isinstance(method.body[0], ast.Return)
        call = method.body[0].value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id == "build_row_action_targets"


def test_conditional_dynamics_layout_has_one_owner() -> None:
    layout_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "conditional_dynamics_layout.py"
    )
    mixture_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "generalist_dynamics.py"
    )

    assert "project_real_conditional_sample_to_target_only" in layout_definitions
    assert {
        "_project_real_conditional_sample_to_target_only",
        "_target_only_shifted_actions",
        "_target_only_prefix_state",
        "_target_only_conditional_metadata",
        "_real_conditional_target_boundary",
    }.isdisjoint(mixture_definitions)


def test_sequence_contract_semantics_have_one_config_owner() -> None:
    contract_functions = {
        "apply_parallel_sequence_contract",
        "expand_parallel_sequence_contract",
        "validate_experiment_config_runtime_contract",
        "validate_parallel_sequence_contract_override_keys",
        "validate_policy_data_sequence_contract",
    }
    config_definitions = _top_level_definitions(PACKAGE_ROOT / "configs" / "sequence_contracts.py")
    loader_definitions = _top_level_definitions(PACKAGE_ROOT / "utils" / "config_loader.py")

    assert contract_functions <= config_definitions
    assert contract_functions.isdisjoint(loader_definitions)


def test_configuration_loading_has_one_package_owner() -> None:
    canonical_definitions = _top_level_definitions(PACKAGE_ROOT / "configs" / "loader.py")
    compatibility_definitions = _top_level_definitions(PACKAGE_ROOT / "utils" / "config_loader.py")

    assert "load_experiment_config" in canonical_definitions
    assert "load_experiment_config" not in compatibility_definitions
    assert LegacyLoadExperimentConfig is load_experiment_config


def test_local_path_resolution_has_one_package_owner() -> None:
    canonical_definitions = _top_level_definitions(PACKAGE_ROOT / "configs" / "local_paths.py")
    compatibility_definitions = _top_level_definitions(PACKAGE_ROOT / "utils" / "local_paths.py")

    assert {"load_local_path_registry", "read_yaml_with_local_paths"} <= canonical_definitions
    assert {"load_local_path_registry", "read_yaml_with_local_paths"}.isdisjoint(
        compatibility_definitions
    )
    assert LegacyLoadLocalPathRegistry is load_local_path_registry
    assert LegacyReadYamlWithLocalPaths is read_yaml_with_local_paths


def test_typed_component_parsers_live_beside_their_contracts() -> None:
    parser_owners = {
        "parse_data_config": "data_parsing.py",
        "parse_shared_video_transformer_config": "backbone.py",
        "parse_training_config": "training.py",
        "parse_inference_config": "inference.py",
        "parse_trainer_config": "trainer.py",
        "parse_validation_config": "validation.py",
        "parse_visual_readout_config": "visual_readout.py",
        "parse_policy_variant_config": "policy_variant.py",
        "parse_action_decoder_config": "action_decoder.py",
    }
    loader_definitions = _top_level_definitions(PACKAGE_ROOT / "configs" / "loader.py")

    for parser_name, owner_filename in parser_owners.items():
        assert parser_name in _top_level_definitions(
            PACKAGE_ROOT / "configs" / owner_filename
        )
        assert parser_name not in loader_definitions


def test_mot_generalist_mode_semantics_have_one_owner() -> None:
    mode_functions = {
        "apply_generalist_training_mode",
        "generalist_forces_clean_video_condition",
        "generalist_rollout_enabled",
        "generalist_rollout_mode_from_value",
        "is_generalist_conditional_rollout",
        "resolve_generalist_rollout_mode",
        "resolve_generalist_training_metadata",
        "sample_generalist_training_mode",
    }
    mode_definitions = _top_level_definitions(
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "generalist_modes.py"
    )
    variant_definitions = _top_level_definitions(
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "variant.py"
    )

    assert mode_functions <= mode_definitions
    assert mode_functions.isdisjoint(variant_definitions)


def test_mot_training_layout_semantics_have_one_owner() -> None:
    layout_methods = {
        "apply_history_action_condition",
        "build_effective_action_mask",
        "build_effective_video_loss_mask",
        "resolve_action_tokens_per_frame",
        "resolve_chunk_origin_frame",
        "resolve_conditional_history_policy",
        "resolve_frame_shift",
        "resolve_history_frames",
        "resolve_loss_frame_range",
        "resolve_sampled_chunk_size",
        "resolve_sampled_window_size",
        "resolve_singleton_chunk_frame",
        "sample_full_segment_geometry",
    }
    layout_path = PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "sequence_layout.py"
    variant_path = PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "variant.py"
    retired_variant_methods = {
        "_apply_train_history_action_condition",
        "_build_action_grid_ids_for_sequence",
        "_build_effective_action_mask",
        "_build_effective_video_loss_mask",
        "_resolve_train_action_tokens_per_frame",
        "_resolve_train_chunk_origin_frame",
        "_resolve_train_conditional_history_policy",
        "_resolve_train_frame_shift",
        "_resolve_train_history_frames",
        "_resolve_train_loss_frame_range",
        "_resolve_train_sampled_chunk_size",
        "_resolve_train_sampled_window_size",
        "_resolve_train_singleton_chunk_frame",
        "_sample_full_segment_train_geometry",
    }

    assert layout_methods <= _class_method_definitions(layout_path, "MoTTrainingLayout")
    assert "build_action_grid_ids_for_sequence" in _top_level_definitions(layout_path)
    assert retired_variant_methods.isdisjoint(
        _class_method_definitions(variant_path, "MoTPolicyVariant")
    )


def test_mot_conditioning_semantics_have_one_owner() -> None:
    conditioning_methods = {
        "action_hidden_context_for_tokens",
        "append_generalist_mode_text_token",
        "build_proprio_cross_attention_mask",
        "context_condition_latent_source",
        "encode_hidden_proprio_context",
        "legacy_prefix_action_hidden_proprio_state",
        "prepend_legacy_prefix_video_latents",
        "proprio_context_token_count",
        "resolve_infer_hidden_proprio_context",
        "resolve_proprio_state",
        "resolve_text_context",
        "resolve_train_condition_latents",
        "resolve_train_hidden_proprio_context",
        "resolve_train_proprio_context",
        "select_anchor_state",
        "train_clean_video_condition_latents",
        "uses_legacy_prefix_contract",
        "uses_per_chunk_proprio_context",
        "uses_proprio_context",
        "uses_text_proprio_context",
        "video_condition_source",
        "video_hidden_context_for_tokens",
    }
    retired_variant_methods = {
        "_action_hidden_context_for_tokens",
        "_append_generalist_mode_text_token",
        "_build_proprio_cross_attention_mask",
        "_context_condition_latent_source",
        "_encode_hidden_proprio_context",
        "_legacy_prefix_action_hidden_proprio_state",
        "_prepend_legacy_prefix_video_latents",
        "_proprio_context_token_count",
        "_resolve_infer_hidden_proprio_context",
        "_resolve_proprio_state",
        "_resolve_text_context_with_proprio",
        "_resolve_train_condition_latents",
        "_resolve_train_hidden_proprio_context",
        "_resolve_train_proprio_context",
        "_select_anchor_state",
        "_train_clean_video_condition_latents",
        "_uses_per_chunk_proprio_context",
        "_uses_proprio_context",
        "_uses_text_proprio_context",
        "_video_condition_source",
        "_video_hidden_context_for_tokens",
    }
    conditioning_path = PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "conditioning.py"
    variant_path = PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "variant.py"

    assert conditioning_methods <= _class_method_definitions(
        conditioning_path,
        "MoTConditioning",
    )
    assert retired_variant_methods.isdisjoint(
        _class_method_definitions(variant_path, "MoTPolicyVariant")
    )
    assert "_uses_mot_legacy_prefix_contract" not in _top_level_definitions(variant_path)


def test_mot_runtime_controls_have_role_owners() -> None:
    routing_functions = {
        "is_mot_same_step_coupling",
        "resolve_mot_action_only_rollout",
        "resolve_mot_current_block_coupling",
        "resolve_mot_inference_window_size",
        "resolve_mot_joint_timestep_coupling",
        "resolve_mot_rollout_frame_chunk_size",
        "should_couple_mot_action_to_video_sigmas",
    }
    tensor_runtime_functions = {
        "expand_mot_scalar_timestep",
        "mot_scheduler_next_sigma",
        "rewind_mot_runtime_action_cache_to_frame",
        "step_mot_flow_with_sigmas",
    }
    retired_variant_functions = routing_functions | tensor_runtime_functions | {
        "_expand_scalar_timestep",
        "_flow_step_with_sigmas",
        "_is_mot_same_step_coupling",
        "_resolve_mot_action_only_rollout",
        "_resolve_mot_inference_window_size",
        "_resolve_mot_joint_timestep_coupling",
        "_resolve_mot_rollout_frame_chunk_size",
        "_rewind_runtime_action_cache_to_frame",
        "_scheduler_next_sigma",
        "_should_couple_mot_action_to_video_sigmas",
    }
    mot_root = PACKAGE_ROOT / "models" / "policy_variants" / "mot"

    assert routing_functions <= _top_level_definitions(mot_root / "runtime_routing.py")
    assert tensor_runtime_functions <= _top_level_definitions(mot_root / "runtime.py")
    assert retired_variant_functions.isdisjoint(
        _top_level_definitions(mot_root / "variant.py")
    )


def test_mot_packed_inference_layout_has_one_owner() -> None:
    inference_layout_path = (
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "inference_layout.py"
    )
    variant_path = PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "variant.py"

    assert {
        "MoTConditionalRolloutInputs",
        "MoTPackedHistory",
        "MoTPackedHistoryWindow",
        "MoTPackedInferenceLayout",
    } <= _top_level_definitions(inference_layout_path)
    assert {
        "from_runtime_state",
        "select_window",
    } <= _class_method_definitions(inference_layout_path, "MoTPackedHistory")
    assert {
        "compose_current_action_sequence",
        "resolve_conditional_rollout_inputs",
    } <= _class_method_definitions(
        inference_layout_path,
        "MoTPackedInferenceLayout",
    )
    assert {
        "MoTConditionalRolloutInputs",
        "MoTPackedHistory",
        "MoTPackedHistoryWindow",
        "MoTPackedInferenceLayout",
    }.isdisjoint(_top_level_definitions(variant_path))


def test_retired_ablations_namespace_is_not_packaged() -> None:
    assert not (PACKAGE_ROOT / "ablations").exists()


def test_retired_fdm_guided_planning_namespace_is_not_packaged() -> None:
    assert not (PACKAGE_ROOT / "planning").exists()


def test_retired_structured_register_runtime_is_not_packaged() -> None:
    retired_paths = (
        PACKAGE_ROOT / "models" / "common" / "joint_runtime.py",
        PACKAGE_ROOT / "models" / "common" / "register_sequence.py",
        PACKAGE_ROOT / "models" / "visual_tower" / "stream_adapters.py",
        PACKAGE_ROOT / "models" / "visual_tower" / "stream_heads.py",
        PACKAGE_ROOT / "models" / "visual_tower" / "structured_attention.py",
    )

    assert not any(path.exists() for path in retired_paths)


def test_deprecated_libero_implementations_and_configs_are_retired() -> None:
    deprecated_script_root = REPO_ROOT / "scripts" / "deprecated"
    deprecated_config_roots = (
        REPO_ROOT / "configs" / "experiments" / "deprecated",
        REPO_ROOT / "configs" / "evals" / "deprecated",
    )

    assert list(deprecated_script_root.glob("*.py")) == []
    assert list(deprecated_script_root.glob("*.sh")) == []
    assert not any(path for root in deprecated_config_roots for path in root.glob("*.yaml"))


def test_private_uva_comparison_drivers_are_retired() -> None:
    retired_paths = (
        REPO_ROOT / "scripts" / "debug_gjd_uva_mode_videos.py",
        REPO_ROOT / "scripts" / "eval_uva_openwam_aligned.py",
        REPO_ROOT / "scripts" / "eval_openwam_fdm_fvd.py",
    )

    assert not any(path.exists() for path in retired_paths)


def test_private_libero_absolute_action_experiment_harness_is_retired() -> None:
    retired_paths = (
        REPO_ROOT / "scripts" / "process_libero10_absolute_joint_dataset.py",
        REPO_ROOT / "scripts" / "calibrate_libero_integrated_delta_scale.py",
        REPO_ROOT / "scripts" / "calibrate_libero_integrated_eef_scale.py",
        REPO_ROOT / "scripts" / "check_libero_absolute_joint_adapter_sanity.py",
        REPO_ROOT / "scripts" / "materialize_libero_absolute_joint_lerobot_overlay.py",
        REPO_ROOT / "scripts" / "materialize_libero_integrated_eef6d_overlay.py",
        REPO_ROOT / "scripts" / "run_libero_abs_joint_rollout_debug.py",
        REPO_ROOT / "scripts" / "validate_libero_absolute_joint_position.py",
    )

    assert not any(path.exists() for path in retired_paths)


def test_private_local_posttraining_supervisor_is_retired() -> None:
    retired_paths = (
        REPO_ROOT / "scripts" / "run_local_libero10_posttrain_m1_m2_m5.sh",
        REPO_ROOT / "scripts" / "run_local_libero10_posttrain_m2_m5.sh",
        REPO_ROOT / "scripts" / "env_local_data_openwam.sh",
    )

    assert not any(path.exists() for path in retired_paths)


def test_orphaned_diagnostics_and_duplicate_aliases_are_retired() -> None:
    retired_paths = (
        REPO_ROOT / "scripts" / "visualize_libero_reference_pose_slurm.py",
        REPO_ROOT / "scripts" / "check_libero_proprio_state_alignment.py",
        REPO_ROOT / "scripts" / "smoke_parallel_stream_lingbot_replica.py",
        REPO_ROOT / "scripts" / "run_contract_only.sh",
        REPO_ROOT / "scripts" / "run_backbone_only.sh",
    )

    assert not any(path.exists() for path in retired_paths)


def test_public_config_enums_are_declared_once() -> None:
    path = PACKAGE_ROOT / "configs" / "enums.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert duplicates == []


def test_legacy_backbone_config_import_is_identity_preserving() -> None:
    assert LegacySharedVideoTransformerConfig is SharedVideoTransformerConfig
