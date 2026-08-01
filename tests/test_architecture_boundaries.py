from __future__ import annotations

import ast
from pathlib import Path

from open_wam.configs import (
    SharedVideoTransformerConfig,
    load_experiment_config,
    load_local_path_registry,
    read_yaml_with_local_paths,
)
from open_wam.contracts import (
    GENERALIST_TRAINING_BUCKET_METADATA_KEY,
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY,
    REPO_ROOT as ContractRepoRoot,
    GeneralistTrainingSampleMetadata,
    ResolvedSourceFps,
    ResolvedVideoClip,
    SampleConstructionMetadata,
    ViewPlacement,
    VideoFrameMapping,
    WAN_TEMPORAL_CHUNK_SIZE,
    find_repo_root,
    normalized_video_frame_count,
    resolve_repo_path,
    resolve_video_source_fps,
    single_sample_metadata_mapping,
    wan_fully_observed_latent_count,
    wan_raw_frame_count_to_latent_count,
    wan_safe_temporal_frame_count,
)
from open_wam.models.video_backbone.config import (
    SharedVideoTransformerConfig as LegacySharedVideoTransformerConfig,
)
from open_wam.configs.variant_semantics import (
    GENERALIST_TRAINING_BUCKET_METADATA_KEY
    as LegacyGeneralistTrainingBucketMetadataKey,
    GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY
    as LegacyGeneralistTrainingDropTextMetadataKey,
    GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY
    as LegacyGeneralistTrainingModeOverrideMetadataKey,
    GENERALIST_TRAINING_SOURCE_METADATA_KEY
    as LegacyGeneralistTrainingSourceMetadataKey,
)
from open_wam.data.raw_video import ViewPlacement as LegacyViewPlacement
from open_wam.data.sample_metadata import (
    GeneralistTrainingSampleMetadata as LegacyGeneralistTrainingSampleMetadata,
    SampleConstructionMetadata as LegacySampleConstructionMetadata,
    single_sample_metadata_mapping as legacy_single_sample_metadata_mapping,
)
from open_wam.runtime.paths import (
    REPO_ROOT as LegacyRepoRoot,
    find_repo_root as LegacyFindRepoRoot,
    resolve_repo_path as LegacyResolveRepoPath,
)
from open_wam.utils.config_loader import (
    load_experiment_config as LegacyLoadExperimentConfig,
)
from open_wam.utils.local_paths import (
    load_local_path_registry as LegacyLoadLocalPathRegistry,
    read_yaml_with_local_paths as LegacyReadYamlWithLocalPaths,
)
from open_wam.utils.video_timeline import (
    ResolvedSourceFps as LegacyResolvedSourceFps,
    ResolvedVideoClip as LegacyResolvedVideoClip,
    VideoFrameMapping as LegacyVideoFrameMapping,
    normalized_video_frame_count as legacy_normalized_video_frame_count,
    resolve_video_source_fps as legacy_resolve_video_source_fps,
)
from open_wam.utils.wan_geometry import (
    WAN_TEMPORAL_CHUNK_SIZE as LEGACY_WAN_TEMPORAL_CHUNK_SIZE,
    wan_fully_observed_latent_count as legacy_wan_fully_observed_latent_count,
    wan_raw_frame_count_to_latent_count as legacy_wan_raw_frame_count_to_latent_count,
    wan_safe_temporal_frame_count as legacy_wan_safe_temporal_frame_count,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "open_wam"


def _absolute_imports_for_file(path: Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _absolute_imports(package: str) -> set[str]:
    imports: set[str] = set()
    for path in (PACKAGE_ROOT / package).rglob("*.py"):
        imports.update(_absolute_imports_for_file(path))
    return imports


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _top_level_import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _compatibility_export_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id.endswith("_COMPATIBILITY_EXPORTS")
            for target in node.targets
        ):
            continue
        assert isinstance(node.value, ast.Tuple)
        assert all(isinstance(element, ast.Name) for element in node.value.elts)
        names.extend(element.id for element in node.value.elts if isinstance(element, ast.Name))
    assert len(names) == len(set(names))
    return set(names)


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


def test_config_package_does_not_depend_on_higher_or_compatibility_layers() -> None:
    forbidden_prefixes = (
        "open_wam.data",
        "open_wam.models",
        "open_wam.pipelines",
        "open_wam.runtime",
        "open_wam.training",
        "open_wam.utils",
    )

    violations = sorted(
        imported
        for imported in _absolute_imports("configs")
        if imported.startswith(forbidden_prefixes)
    )

    assert violations == []


def test_foundational_contracts_are_dependency_free() -> None:
    violations = sorted(
        imported
        for imported in _absolute_imports("contracts")
        if imported.startswith("open_wam")
    )

    assert violations == []


def test_utils_package_does_not_depend_on_model_implementations() -> None:
    violations = sorted(
        imported
        for imported in _absolute_imports("utils")
        if imported.startswith("open_wam.models")
    )

    assert violations == []


def test_visual_tower_does_not_depend_on_policy_implementations() -> None:
    violations = sorted(
        imported
        for imported in _absolute_imports("models/visual_tower")
        if imported.startswith("open_wam.models.policy_variants")
    )

    assert violations == []


def test_model_package_does_not_depend_on_data_implementations() -> None:
    violations = sorted(
        imported
        for imported in _absolute_imports("models")
        if imported.startswith("open_wam.data")
    )

    assert violations == []


def test_visual_tower_cache_lifecycle_has_one_policy_owner() -> None:
    visual_root = PACKAGE_ROOT / "models" / "visual_tower"
    lifecycle_path = visual_root / "cache_lifecycle.py"
    tower_path = visual_root / "tower.py"
    lifecycle_methods = {
        "advance_state",
        "build_update_metadata",
        "clear_state",
        "ensure_branches",
        "init_state",
        "resolve_state",
        "truncate_state",
        "_slice_attention_cache_entry",
        "_truncate_attention_cache_entry",
    }
    facade_methods = {
        "advance_runtime_cache_state",
        "build_runtime_cache_update_metadata",
        "clear_runtime_cache_state",
        "ensure_runtime_cache_branches",
        "init_runtime_cache_state",
        "resolve_runtime_cache_state",
        "truncate_runtime_cache_state",
    }

    assert lifecycle_methods <= _class_method_definitions(
        lifecycle_path,
        "RuntimeCacheLifecycle",
    )
    tower_methods = _class_method_definitions(tower_path, "VisualTower")
    assert facade_methods <= tower_methods
    assert {
        "_slice_attention_cache_entry",
        "_truncate_attention_cache_entry",
    }.isdisjoint(tower_methods)

    tower_source = tower_path.read_text(encoding="utf-8")
    assert "from .cache_lifecycle import" in tower_source
    for backend_helper in (
        "clear_cache_backend_payload",
        "init_cache_backend_payload",
        "resolve_cache_backend_spec",
    ):
        assert backend_helper not in tower_source


def test_visual_tower_runtime_backbone_policy_has_one_owner() -> None:
    visual_root = PACKAGE_ROOT / "models" / "visual_tower"
    owner_path = visual_root / "runtime_backbone.py"
    tower_path = visual_root / "tower.py"
    owner_functions = {
        "ensure_runtime_module_device",
        "initialize_runtime_backbone",
        "log_runtime_backbone_missing_keys",
        "reset_runtime_module_cache",
        "validate_runtime_backbone_request",
    }

    assert owner_functions <= _top_level_definitions(owner_path)
    tower_methods = _class_method_definitions(tower_path, "VisualTower")
    assert {
        "ensure_runtime_backbone_device",
        "get_runtime_backbone",
        "reset_runtime_backbone_cache",
    } <= tower_methods
    assert {
        "ensure_exact_runtime_transformer_device",
        "ensure_lingbot_reference_transformer_device",
        "get_exact_runtime_transformer",
        "get_lingbot_reference_transformer",
        "reset_exact_runtime_cache",
        "reset_lingbot_reference_runtime",
        "run_mot_packed_video_forward",
    }.isdisjoint(tower_methods)
    tower_source = tower_path.read_text(encoding="utf-8")
    assert "from .runtime_backbone import" in tower_source
    for loading_helper in (
        "is_allowed_runtime_missing_key",
        "is_open_wam_exported_runtime_backbone_dir",
        "load_exported_runtime_backbone_into_replica_core",
        "load_reference_weights_into_replica_core",
        "preferred_reference_dtype",
        "resolve_runtime_backbone_dir",
    ):
        assert loading_helper not in tower_source


def test_mot_policy_does_not_depend_on_parallel_stream_implementation() -> None:
    violations = sorted(
        imported
        for imported in _absolute_imports("models/policy_variants/mot")
        if imported.startswith("open_wam.models.policy_variants.parallel_stream")
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
        "LocalLatentRepository",
        "LocalRepoBundle",
        "assemble_canonical_latents",
        "condition_latent_offset_mismatches",
        "discover_local_lerobot_repo_bundles",
        "load_empty_text_embedding",
        "scan_local_latent_windows",
        "load_lerobot_v2_local_metadata",
        "resolve_latent_root",
        "reshape_latent_payload",
    }
    storage_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent_storage.py"
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent.py"
    storage_definitions = _top_level_definitions(storage_path)
    dataset_definitions = _top_level_definitions(dataset_path)

    assert storage_owned <= storage_definitions
    assert storage_owned.isdisjoint(dataset_definitions)
    assert {
        "load_episode_rows",
        "load_window_latents",
        "load_canonical_window_latents",
    } <= _class_method_definitions(storage_path, "LocalLatentRepository")

    compatibility_methods = {
        "_load_empty_text_embedding",
        "_load_window_latents",
        "_assemble_canonical_latents",
        "_condition_latent_offset_mismatches",
        "_load_canonical_window_latents",
        "_load_episode_rows",
    }
    assert compatibility_methods <= _class_method_definitions(
        dataset_path,
        "LocalLeRobotLatentWindowDataset",
    )
    for method_name in compatibility_methods:
        method = _class_method(
            dataset_path,
            "LocalLeRobotLatentWindowDataset",
            method_name,
        )
        assert len(method.body) == 1
        assert isinstance(method.body[0], ast.Return)
        assert isinstance(method.body[0].value, ast.Call)


def test_lerobot_latent_sampling_policy_has_one_owner() -> None:
    sampling_owned = {
        "HierarchicalFixedSegmentSamplingPlan",
        "HierarchicalFixedSegmentTaskSpec",
        "HierarchicalFixedSegmentTrainSampler",
        "HierarchicalFixedSegmentWindowSpec",
        "LocalLatentEpochOrderSampler",
        "LocalLatentUniformSegmentSamplingPlan",
        "LocalLatentWindowWeightPlan",
        "LocalLatentWeightedTrainSampler",
        "build_hierarchical_fixed_segment_task_specs",
    }
    sampling_path = (
        PACKAGE_ROOT / "data" / "lerobot_v2_latent_sampling.py"
    )
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent.py"

    assert sampling_owned <= _top_level_definitions(sampling_path)
    assert sampling_owned.isdisjoint(_top_level_definitions(dataset_path))
    assert {
        "draw",
        "from_task_specs",
        "iter_eligible_start_keys",
        "sample_metadata",
    } <= _class_method_definitions(
        sampling_path,
        "HierarchicalFixedSegmentSamplingPlan",
    )

    assert {
        "from_windows",
        "sample_weight_metadata",
        "task_text_for_window_index",
    } <= _class_method_definitions(
        sampling_path,
        "LocalLatentWindowWeightPlan",
    )

    base_weight_delegate = _class_method(
        dataset_path,
        "LocalLeRobotLatentWindowDataset",
        "_sample_weight_metadata",
    )
    assert len(base_weight_delegate.body) == 1
    assert isinstance(base_weight_delegate.body[0], ast.Return)
    assert isinstance(base_weight_delegate.body[0].value, ast.Call)

    task_text_delegate = _class_method(
        dataset_path,
        "LocalLeRobotLatentWindowDataset",
        "task_text_for_window_index",
    )
    assert ast.get_docstring(task_text_delegate)
    assert len(task_text_delegate.body) == 2
    assert isinstance(task_text_delegate.body[1], ast.Return)
    assert isinstance(task_text_delegate.body[1].value, ast.Call)
    assert {
        "_build_sample_weights",
        "_estimate_mean_task_demo_count",
        "_estimate_mean_valid_action_steps",
        "_estimate_task_demo_counts",
        "_estimate_window_valid_action_steps",
        "_window_task_text",
    }.isdisjoint(
        _class_method_definitions(
            dataset_path,
            "LocalLeRobotLatentWindowDataset",
        )
    )

    assert {
        "build_epoch_index_order",
        "build_sample_weights",
        "build_virtual_index",
        "eligible_segment_lengths",
        "estimate_segment_valid_action_steps",
        "estimate_virtual_valid_action_steps",
        "from_windows",
        "materialize_task_virtual_start_counts",
        "materialize_virtual_indices_by_window",
        "resolve_segment_length_candidates",
        "resolve_start_padding_frames",
        "sample_attention_geometry",
        "sample_segment_geometry",
        "sample_weight_metadata",
    } <= _class_method_definitions(
        sampling_path,
        "LocalLatentUniformSegmentSamplingPlan",
    )

    uniform_compatibility_methods = {
        "_sample_weight_metadata",
        "build_epoch_index_order",
    }
    for method_name in uniform_compatibility_methods:
        method = _class_method(
            dataset_path,
            "UniformSegmentLocalLeRobotLatentDataset",
            method_name,
        )
        assert len(method.body) == 1
        assert isinstance(method.body[0], ast.Return)

    retired_uniform_helpers = {
        "_build_virtual_index",
        "_build_virtual_indices_by_window",
        "_build_virtual_sample_weights",
        "_eligible_segment_lengths",
        "_estimate_mean_task_virtual_start_count",
        "_estimate_segment_valid_action_steps",
        "_estimate_task_virtual_start_counts",
        "_estimate_virtual_mean_valid_action_steps",
        "_estimate_virtual_valid_action_steps",
        "_resolve_segment_length_candidates",
        "_sample_segment_geometry",
        "_sample_uniform_segment_attention_geometry",
        "_window_start_padding_frames",
    }
    assert retired_uniform_helpers.isdisjoint(
        _class_method_definitions(
            dataset_path,
            "UniformSegmentLocalLeRobotLatentDataset",
        )
    )

    retained_diagnostic_methods = {
        "resolve_hierarchical_sample_key",
        "iter_hierarchical_eligible_start_keys",
    }
    for method_name in retained_diagnostic_methods:
        method = _class_method(
            dataset_path,
            "HierarchicalFixedSegmentLocalLeRobotLatentDataset",
            method_name,
        )
        executable_statements = [
            statement
            for statement in method.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        assert len(executable_statements) == 1
        assert isinstance(executable_statements[0], ast.Return)

    retired_hierarchical_helpers = {
        "_build_task_specs",
        "_build_window_start_ranges_by_chunk",
        "_draw_hierarchical_sample",
        "_hierarchical_chunk_size_candidates",
        "_hierarchical_context_prefix_frames",
        "_hierarchical_sample_metadata",
    }
    assert retired_hierarchical_helpers.isdisjoint(
        _class_method_definitions(
            dataset_path,
            "HierarchicalFixedSegmentLocalLeRobotLatentDataset",
        )
    )


def test_lerobot_latent_sample_source_has_one_owner() -> None:
    source_path = (
        PACKAGE_ROOT / "data" / "lerobot_v2_latent_source.py"
    )
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent.py"

    source_contracts = {
        "LocalLatentSampleConditioning",
        "LocalLatentSampleSource",
        "LocalLatentSampleSourceLoader",
    }
    assert source_contracts <= _top_level_definitions(source_path)
    assert source_contracts.isdisjoint(_top_level_definitions(dataset_path))
    assert {"conditioning_for_frame"} <= _class_method_definitions(
        source_path,
        "LocalLatentSampleSource",
    )
    assert {"load"} <= _class_method_definitions(
        source_path,
        "LocalLatentSampleSourceLoader",
    )

    source_delegate = _class_method(
        dataset_path,
        "LocalLeRobotLatentWindowDataset",
        "_load_sample_source",
    )
    assert len(source_delegate.body) == 1
    assert isinstance(source_delegate.body[0], ast.Return)
    assert isinstance(source_delegate.body[0].value, ast.Call)

    for class_name in (
        "LocalLeRobotLatentWindowDataset",
        "UniformSegmentLocalLeRobotLatentDataset",
        "HierarchicalFixedSegmentLocalLeRobotLatentDataset",
        "CausalPrefixSuffixLocalLeRobotLatentDataset",
    ):
        getitem = _class_method(dataset_path, class_name, "__getitem__")
        source_calls = [
            node
            for node in ast.walk(getitem)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_load_sample_source"
        ]
        assert len(source_calls) == 1
        called_methods = {
            node.func.attr
            for node in ast.walk(getitem)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        assert {
            "_load_canonical_window_latents",
            "_load_episode_rows",
            "_load_window_latents",
        }.isdisjoint(called_methods)


def test_lerobot_latent_hierarchical_segment_planning_has_one_owner() -> None:
    planner_path = PACKAGE_ROOT / "data" / "latent_hierarchical_sampling.py"
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent.py"

    assert {
        "LocalLatentHierarchicalSampleKey",
        "LocalLatentHierarchicalSegmentPlan",
    } <= _top_level_definitions(planner_path)
    assert {
        "context_prefix_frames",
        "draw",
        "from_windows",
        "iter_eligible_start_keys",
        "resolve_chunk_size_candidates",
        "resolve_context_prefix_frames",
        "resolve_sample_key",
        "sample_metadata",
    } <= _class_method_definitions(
        planner_path,
        "LocalLatentHierarchicalSegmentPlan",
    )
    assert {"as_metadata"} <= _class_method_definitions(
        planner_path,
        "LocalLatentHierarchicalSampleKey",
    )
    assert not any(
        imported == "torch" or imported.startswith("torch.")
        for imported in _absolute_imports_for_file(planner_path)
    )

    dataset_methods = _class_method_definitions(
        dataset_path,
        "HierarchicalFixedSegmentLocalLeRobotLatentDataset",
    )
    assert {
        "_build_task_specs",
        "_build_window_start_ranges_by_chunk",
        "_draw_hierarchical_sample",
        "_hierarchical_chunk_size_candidates",
        "_hierarchical_context_prefix_frames",
        "_hierarchical_sample_metadata",
    }.isdisjoint(dataset_methods)


def test_lerobot_consortium_epoch_order_planning_has_one_owner() -> None:
    planning_path = (
        PACKAGE_ROOT / "data" / "lerobot_consortium_sampling.py"
    )
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_consortium.py"

    assert {"ConsortiumEpochOrderPlan"} <= _top_level_definitions(
        planning_path
    )
    assert {
        "build_epoch_index_order",
        "from_member_indices",
    } <= _class_method_definitions(
        planning_path,
        "ConsortiumEpochOrderPlan",
    )
    assert not any(
        imported == "torch" or imported.startswith("torch.")
        for imported in _absolute_imports_for_file(planning_path)
    )
    assert {
        "_build_weighted_round_robin_schedule",
        "_cycle_take",
        "_largest_remainder_counts",
        "_resolve_per_dataset_target_counts",
        "_seeded_shuffle",
        "_stable_int_seed",
    }.isdisjoint(_top_level_definitions(dataset_path))

    compatibility_method = _class_method(
        dataset_path,
        "LeRobotConsortiumWindowDataset",
        "build_epoch_index_order",
    )
    assert len(compatibility_method.body) == 1
    assert isinstance(compatibility_method.body[0], ast.Return)
    delegated_call = compatibility_method.body[0].value
    assert isinstance(delegated_call, ast.Call)
    assert isinstance(delegated_call.func, ast.Name)
    assert delegated_call.func.id == "list"
    owner_call = delegated_call.args[0]
    assert isinstance(owner_call, ast.Call)
    assert isinstance(owner_call.func, ast.Attribute)
    assert owner_call.func.attr == "build_epoch_index_order"


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


def test_lerobot_latent_segment_materialization_has_one_owner() -> None:
    materialization_path = (
        PACKAGE_ROOT / "data" / "latent_segment_materialization.py"
    )
    segment_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent_segment.py"
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent.py"
    assert {
        "LatentSegmentMaterializationPlan",
        "plan_latent_segment_materialization",
        "slice_latent_segment_with_zero_order_hold",
    } <= _top_level_definitions(materialization_path)
    assert {
        "LocalLatentSegment",
        "LocalLatentSegmentAssembler",
    } <= _top_level_definitions(segment_path)

    builder = _class_method(
        segment_path,
        "LocalLatentSegmentAssembler",
        "build",
    )
    direct_plan_calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "plan_latent_segment_materialization"
    ]
    assert len(direct_plan_calls) == 1
    direct_slice_calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "slice_latent_segment_with_zero_order_hold"
    ]
    assert len(direct_slice_calls) == 2

    retired_dataset_helpers = {
        "_build_uniform_segment",
        "_resolve_sample_state_anchor_frame",
        "_segment_observed_frame_ids",
        "_slice_video_latents_with_zero_hold",
    }
    assert retired_dataset_helpers.isdisjoint(
        _class_method_definitions(
            dataset_path,
            "UniformSegmentLocalLeRobotLatentDataset",
        )
    )

    for dataset_class in (
        "UniformSegmentLocalLeRobotLatentDataset",
        "HierarchicalFixedSegmentLocalLeRobotLatentDataset",
    ):
        getitem = _class_method(dataset_path, dataset_class, "__getitem__")
        owner_calls = [
            node
            for node in ast.walk(getitem)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "build"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_segment_assembler"
        ]
        assert len(owner_calls) == 1


def test_lerobot_latent_causal_sampling_has_one_owner() -> None:
    planner_path = PACKAGE_ROOT / "data" / "latent_causal_sampling.py"
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent.py"

    assert {
        "LatentCausalPrefixSuffixCandidate",
        "LatentCausalPrefixSuffixWindowPlan",
        "LatentCausalPrefixSuffixWindowPlanner",
    } <= _top_level_definitions(planner_path)
    assert {
        "build_candidates",
        "from_data_config",
        "plan",
        "select_candidate",
    } <= _class_method_definitions(
        planner_path,
        "LatentCausalPrefixSuffixWindowPlanner",
    )
    assert not any(
        imported == "torch" or imported.startswith("torch.")
        for imported in _absolute_imports_for_file(planner_path)
    )

    assert "_build_raw_bucket_boundaries" not in _class_method_definitions(
        dataset_path,
        "LocalLeRobotLatentWindowDataset",
    )
    assert (
        "_sample_causal_prefix_suffix_subwindow"
        not in _class_method_definitions(
            dataset_path,
            "CausalPrefixSuffixLocalLeRobotLatentDataset",
        )
    )

    getitem = _class_method(
        dataset_path,
        "CausalPrefixSuffixLocalLeRobotLatentDataset",
        "__getitem__",
    )
    planner_calls = [
        node
        for node in ast.walk(getitem)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "plan"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_causal_sampling_planner"
    ]
    assert len(planner_calls) == 1


def test_lerobot_latent_train_val_window_planning_has_one_owner() -> None:
    planner_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent_split.py"
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent.py"

    assert {
        "LocalLatentTrainValWindowPlan",
        "LocalLatentTrainValWindowPlanner",
    } <= _top_level_definitions(planner_path)
    assert {
        "_filtered_windows_for_roots",
        "_plan_explicit_roots",
        "_plan_shared_roots",
        "plan",
    } <= _class_method_definitions(
        planner_path,
        "LocalLatentTrainValWindowPlanner",
    )
    assert not any(
        imported == "torch" or imported.startswith("torch.")
        for imported in _absolute_imports_for_file(planner_path)
    )

    builder = next(
        node
        for node in ast.parse(dataset_path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_local_lerobot_latent_train_val_datasets"
    )
    planner_calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "plan"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "LocalLatentTrainValWindowPlanner"
    ]
    assert len(planner_calls) == 1
    assert {
        "discover_local_lerobot_repo_bundles",
        "load_replay_status_records",
        "scan_local_latent_windows",
        "split_episode_indices_by_replay_status",
    }.isdisjoint(
        {
            node.id
            for node in ast.walk(builder)
            if isinstance(node, ast.Name)
        }
    )


def test_mixed_video_catalog_has_one_owner() -> None:
    from open_wam.data import MixedVideoCatalog as PublicMixedVideoCatalog
    from open_wam.data.mixed_video import (
        MixedVideoCatalog as LegacyMixedVideoCatalog,
        MixedVideoEpisodeRecord as LegacyMixedVideoEpisodeRecord,
        MixedVideoStreamRecord as LegacyMixedVideoStreamRecord,
        load_mixed_video_catalog as legacy_load_mixed_video_catalog,
        split_mixed_video_episodes as legacy_split_mixed_video_episodes,
    )
    from open_wam.data.mixed_video_catalog import (
        MixedVideoCatalog,
        MixedVideoEpisodeRecord,
        MixedVideoStreamRecord,
        load_mixed_video_catalog,
        split_mixed_video_episodes,
    )

    canonical_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "mixed_video_catalog.py"
    )
    compatibility_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "mixed_video.py"
    )
    canonical_names = {
        "MixedVideoCatalog",
        "MixedVideoEpisodeRecord",
        "MixedVideoStreamRecord",
        "load_mixed_video_catalog",
        "split_mixed_video_episodes",
    }

    assert canonical_names <= canonical_definitions
    assert canonical_names.isdisjoint(compatibility_definitions)
    assert PublicMixedVideoCatalog is MixedVideoCatalog
    assert LegacyMixedVideoCatalog is MixedVideoCatalog
    assert LegacyMixedVideoEpisodeRecord is MixedVideoEpisodeRecord
    assert LegacyMixedVideoStreamRecord is MixedVideoStreamRecord
    assert legacy_load_mixed_video_catalog is load_mixed_video_catalog
    assert legacy_split_mixed_video_episodes is split_mixed_video_episodes

    encoder_imports = _absolute_imports_for_file(
        REPO_ROOT / "scripts" / "encode_mixed_video_latents.py"
    )
    assert "open_wam.data.mixed_video_catalog" in encoder_imports


def test_mixed_video_decode_has_one_owner() -> None:
    from open_wam.data import (
        decode_video_frames as public_decode_video_frames,
        transform_frame as public_transform_frame,
    )
    from open_wam.data.mixed_video import (
        MixedVideoResolvedDecodeSize as LegacyMixedVideoResolvedDecodeSize,
        decode_mixed_video_stream_frame_chunk as legacy_decode_stream_chunk,
        decode_video_frames as legacy_decode_video_frames,
        iter_mixed_video_stream_frame_chunks as legacy_iter_stream_chunks,
        normalized_video_frame_count as legacy_normalized_frame_count,
        resample_video_frames_to_fps as legacy_resample_frames,
        resolve_mixed_video_decode_size as legacy_resolve_decode_size,
        resolve_mixed_video_observation_fps as legacy_resolve_fps,
        transform_frame as legacy_transform_frame,
    )
    from open_wam.data.mixed_video_decode import (
        MixedVideoResolvedDecodeSize,
        decode_mixed_video_stream_frame_chunk,
        decode_mixed_video_stream_frames,
        decode_video_frames,
        iter_mixed_video_stream_frame_chunks,
        normalized_video_frame_count,
        resample_video_frames_to_fps,
        resolve_mixed_video_decode_size,
        resolve_mixed_video_observation_fps,
        transform_frame,
    )

    canonical_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "mixed_video_decode.py"
    )
    compatibility_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "mixed_video.py"
    )
    canonical_names = {
        "MixedVideoResolvedDecodeSize",
        "decode_mixed_video_stream_frame_chunk",
        "decode_mixed_video_stream_frames",
        "decode_video_frames",
        "iter_mixed_video_stream_frame_chunks",
        "normalized_video_frame_count",
        "resample_video_frames_to_fps",
        "resolve_mixed_video_decode_size",
        "resolve_mixed_video_observation_fps",
        "transform_frame",
    }

    assert canonical_names <= canonical_definitions
    assert canonical_names.isdisjoint(compatibility_definitions)
    assert LegacyMixedVideoResolvedDecodeSize is MixedVideoResolvedDecodeSize
    assert legacy_decode_stream_chunk is decode_mixed_video_stream_frame_chunk
    assert legacy_decode_video_frames is decode_video_frames
    assert legacy_iter_stream_chunks is iter_mixed_video_stream_frame_chunks
    assert legacy_normalized_frame_count is normalized_video_frame_count
    assert legacy_resample_frames is resample_video_frames_to_fps
    assert legacy_resolve_decode_size is resolve_mixed_video_decode_size
    assert legacy_resolve_fps is resolve_mixed_video_observation_fps
    assert legacy_transform_frame is transform_frame
    assert public_decode_video_frames is decode_video_frames
    assert public_transform_frame is transform_frame
    assert (
        decode_mixed_video_stream_frames.__module__
        == "open_wam.data.mixed_video_decode"
    )

    encoder_imports = _absolute_imports_for_file(
        REPO_ROOT / "scripts" / "encode_mixed_video_latents.py"
    )
    assert "open_wam.data.mixed_video_decode" in encoder_imports

    dataset_loader = _class_method(
        PACKAGE_ROOT / "data" / "mixed_video.py",
        "MixedVideoWindowDataset",
        "_load_stream_frames",
    )
    full_stream_decode_calls = [
        node
        for node in ast.walk(dataset_loader)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decode_mixed_video_stream_frames"
    ]
    assert len(full_stream_decode_calls) == 1


def test_mixed_video_latent_repository_has_one_owner() -> None:
    storage_path = (
        PACKAGE_ROOT / "data" / "mixed_video_latent_storage.py"
    )
    dataset_path = PACKAGE_ROOT / "data" / "mixed_video.py"
    storage_owned = {
        "MixedVideoLatentRepository",
        "load_mixed_video_latent_tensor",
        "mixed_video_latent_cache_key",
        "resolve_mixed_video_latent_path",
    }

    assert storage_owned <= _top_level_definitions(storage_path)
    assert storage_owned.isdisjoint(_top_level_definitions(dataset_path))
    assert {"cache_capacity", "load"} <= _class_method_definitions(
        storage_path,
        "MixedVideoLatentRepository",
    )

    compatibility_loader = _class_method(
        dataset_path,
        "MixedVideoLatentWindowDataset",
        "_load_stream_latents",
    )
    assert len(compatibility_loader.body) == 1
    assert isinstance(compatibility_loader.body[0], ast.Return)
    delegated_call = compatibility_loader.body[0].value
    assert isinstance(delegated_call, ast.Call)
    assert isinstance(delegated_call.func, ast.Attribute)
    assert delegated_call.func.attr == "load"


def test_mixed_video_window_planning_has_one_owner() -> None:
    planning_path = (
        PACKAGE_ROOT / "data" / "mixed_video_planning.py"
    )
    dataset_path = PACKAGE_ROOT / "data" / "mixed_video.py"

    assert {
        "MixedVideoWindowPlanner",
        "MixedVideoWindowRecord",
    } <= _top_level_definitions(planning_path)
    assert {
        "MixedVideoWindowPlanner",
        "MixedVideoWindowRecord",
    }.isdisjoint(_top_level_definitions(dataset_path))
    assert {
        "build_episode_windows",
        "build_latent_view_windows",
        "build_source_balanced_epoch_order",
        "valid_latent_view_combinations",
    } <= _class_method_definitions(
        planning_path,
        "MixedVideoWindowPlanner",
    )
    assert "torch" not in _absolute_imports_for_file(planning_path)

    delegates = (
        (
            "MixedVideoWindowDataset",
            "_build_sample_index",
            "build_episode_windows",
        ),
        (
            "MixedVideoLatentWindowDataset",
            "_build_sample_index",
            "build_latent_view_windows",
        ),
        (
            "MixedVideoWindowDataset",
            "build_epoch_index_order",
            "build_source_balanced_epoch_order",
        ),
    )
    for class_name, method_name, owner_method in delegates:
        compatibility_method = _class_method(
            dataset_path,
            class_name,
            method_name,
        )
        assert len(compatibility_method.body) == 1
        assert isinstance(compatibility_method.body[0], ast.Return)
        delegated_call = compatibility_method.body[0].value
        assert isinstance(delegated_call, ast.Call)
        assert isinstance(delegated_call.func, ast.Attribute)
        assert delegated_call.func.attr == owner_method


def test_lerobot_latent_supervision_assembly_has_one_owner() -> None:
    supervision_path = (
        PACKAGE_ROOT / "data" / "lerobot_v2_latent_supervision.py"
    )
    dataset_path = PACKAGE_ROOT / "data" / "lerobot_v2_latent.py"
    assembler_methods = {
        "build_action_targets",
        "build_lingbot_window_action_targets",
        "build_standard_policy_window_action_targets",
        "extract_proprio_context_frames",
        "extract_proprio_context_state_sequence",
        "extract_sequence",
        "extract_state_at_frame",
        "extract_state_history_at_frame",
    }
    assert assembler_methods <= _class_method_definitions(
        supervision_path,
        "LocalLatentSupervisionAssembler",
    )

    compatibility_methods = {
        "_build_lingbot_window_action_targets",
        "_build_standard_policy_window_action_targets",
        "_extract_proprio_context_state_sequence",
        "_extract_state_history_at_frame",
    }
    assert compatibility_methods <= _class_method_definitions(
        dataset_path,
        "LocalLeRobotLatentWindowDataset",
    )
    for method_name in compatibility_methods:
        method = _class_method(
            dataset_path,
            "LocalLeRobotLatentWindowDataset",
            method_name,
        )
        assert len(method.body) == 1
        assert isinstance(method.body[0], ast.Return)
        assert isinstance(method.body[0].value, ast.Call)

    retired_private_facades = {
        "_build_action_targets",
        "_extract_proprio_context_frames",
        "_extract_sequence",
        "_extract_state_at_frame",
    }
    assert retired_private_facades.isdisjoint(
        _class_method_definitions(
            dataset_path,
            "LocalLeRobotLatentWindowDataset",
        )
    )


def test_row_action_target_transform_has_one_owner() -> None:
    transform_path = PACKAGE_ROOT / "data" / "row_action_targets.py"
    transform_definitions = _top_level_definitions(transform_path)
    assert {"build_row_action_targets", "resolve_row_key"} <= transform_definitions

    adapter_methods = (
        ("lerobot_v2.py", "LeRobotV2WindowDataset", "_build_action_targets"),
        (
            "lerobot_v2_latent_supervision.py",
            "LocalLatentSupervisionAssembler",
            "build_action_targets",
        ),
        ("lerobot_consortium.py", "LeRobotConsortiumWindowDataset", "_build_action_targets"),
    )
    for filename, class_name, method_name in adapter_methods:
        method = _class_method(
            PACKAGE_ROOT / "data" / filename,
            class_name,
            method_name,
        )
        assert len(method.body) == 1
        assert isinstance(method.body[0], ast.Return)
        call = method.body[0].value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id == "build_row_action_targets"
        packer_keywords = [
            keyword for keyword in call.keywords if keyword.arg == "pack_sequence"
        ]
        if filename == "lerobot_v2_latent_supervision.py":
            assert len(packer_keywords) == 1
            assert isinstance(packer_keywords[0].value, ast.Name)
            assert packer_keywords[0].value.id == "_TRUNCATING_SEQUENCE_PACKER"
        else:
            assert not packer_keywords


def test_temporal_sequence_packing_has_one_owner() -> None:
    packing_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "sequence_packing.py"
    )
    assert "pack_temporal_sequence" in packing_definitions

    adapter_classes = {
        "lerobot_v2.py": "LeRobotV2WindowDataset",
        "lerobot_v2_latent.py": "LocalLeRobotLatentWindowDataset",
        "lerobot_consortium.py": "LeRobotConsortiumWindowDataset",
        "libero_hdf5.py": "LiberoOfflineWindowDataset",
    }
    for filename, class_name in adapter_classes.items():
        methods = _class_method_definitions(
            PACKAGE_ROOT / "data" / filename,
            class_name,
        )
        assert "_pack_sequence" not in methods


def test_hierarchical_draw_primitives_have_one_owner() -> None:
    sampling_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "distributed_sampling.py"
    )
    assert {
        "draw_hierarchical_sample_index",
        "stable_int_seed",
        "weighted_choice_index",
    } <= sampling_definitions

    for filename in (
        "lerobot_v2_latent.py",
        "counterfactual_dynamics_dataset.py",
    ):
        adapter_definitions = _top_level_definitions(PACKAGE_ROOT / "data" / filename)
        assert {
            "_stable_int_seed",
            "_weighted_choice_index",
        }.isdisjoint(adapter_definitions)


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


def test_counterfactual_dataset_and_mixture_have_separate_owners() -> None:
    dataset_path = PACKAGE_ROOT / "data" / "counterfactual_dynamics_dataset.py"
    mixture_path = PACKAGE_ROOT / "data" / "generalist_dynamics.py"
    dataset_definitions = _top_level_definitions(dataset_path)
    mixture_definitions = _top_level_definitions(mixture_path)

    assert "EncodedCounterfactualDynamicsLatentDataset" in dataset_definitions
    assert {
        "EncodedCounterfactualDynamicsLatentDataset",
        "_build_counterfactual_fixed_segment",
        "_counterfactual_action_steps_per_frame",
        "_counterfactual_latent_state_frames",
        "_load_latent_payload",
    }.isdisjoint(mixture_definitions)
    assert {
        "GeneralistDynamicsMixtureDataset",
        "GeneralistDynamicsSourceViewDataset",
        "build_generalist_dynamics_mixture_datasets",
    } <= mixture_definitions

    from open_wam.data.counterfactual_dynamics_dataset import (
        EncodedCounterfactualDynamicsLatentDataset as CanonicalDataset,
    )
    from open_wam.data import (
        EncodedCounterfactualDynamicsLatentDataset as PublicDataset,
    )
    from open_wam.data.generalist_dynamics import (
        EncodedCounterfactualDynamicsLatentDataset as LegacyDataset,
    )

    assert PublicDataset is CanonicalDataset
    assert LegacyDataset is CanonicalDataset


def test_latent_view_assembly_has_one_owner() -> None:
    assembly_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "latent_view_assembly.py"
    )
    mixed_video_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "mixed_video.py"
    )

    assert "assemble_latent_views" in assembly_definitions
    assert {
        "assemble_mixed_video_latent_views",
        "_latent_assembly_canvas_shape",
        "_latent_assembly_placements",
    }.isdisjoint(mixed_video_definitions)


def test_sequence_contract_semantics_have_one_config_owner() -> None:
    from open_wam.configs.loader import (
        validate_experiment_config_runtime_contract as LoaderValidateRuntimeContract,
        validate_parallel_sequence_contract_override_keys as LoaderValidateOverrideKeys,
    )
    from open_wam.configs.sequence_contracts import (
        validate_experiment_config_runtime_contract as CanonicalValidateRuntimeContract,
        validate_parallel_sequence_contract_override_keys as CanonicalValidateOverrideKeys,
    )

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
    assert LoaderValidateRuntimeContract is CanonicalValidateRuntimeContract
    assert LoaderValidateOverrideKeys is CanonicalValidateOverrideKeys


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


def test_project_path_contracts_have_one_dependency_free_owner() -> None:
    canonical_definitions = _top_level_definitions(
        PACKAGE_ROOT / "contracts" / "paths.py"
    )
    compatibility_definitions = _top_level_definitions(
        PACKAGE_ROOT / "runtime" / "paths.py"
    )

    assert {"find_repo_root", "resolve_repo_path"} <= canonical_definitions
    assert {"find_repo_root", "resolve_repo_path"}.isdisjoint(
        compatibility_definitions
    )
    assert LegacyRepoRoot is ContractRepoRoot
    assert LegacyFindRepoRoot is find_repo_root
    assert LegacyResolveRepoPath is resolve_repo_path


def test_video_timeline_contracts_have_one_dependency_free_owner() -> None:
    from open_wam.models.common.video_geometry import (
        WAN_TEMPORAL_CHUNK_SIZE as MODEL_WAN_TEMPORAL_CHUNK_SIZE,
        wan_fully_observed_latent_count as model_wan_fully_observed_latent_count,
        wan_raw_frame_count_to_latent_count as model_wan_raw_frame_count_to_latent_count,
        wan_safe_temporal_frame_count as model_wan_safe_temporal_frame_count,
    )

    canonical_definitions = _top_level_definitions(
        PACKAGE_ROOT / "contracts" / "video.py"
    )
    timeline_compatibility_definitions = _top_level_definitions(
        PACKAGE_ROOT / "utils" / "video_timeline.py"
    )
    geometry_compatibility_definitions = _top_level_definitions(
        PACKAGE_ROOT / "utils" / "wan_geometry.py"
    )
    canonical_names = {
        "ResolvedSourceFps",
        "ResolvedVideoClip",
        "VideoFrameMapping",
        "normalized_video_frame_count",
        "resolve_video_source_fps",
        "wan_fully_observed_latent_count",
        "wan_raw_frame_count_to_latent_count",
        "wan_safe_temporal_frame_count",
    }

    assert canonical_names <= canonical_definitions
    assert canonical_names.isdisjoint(
        timeline_compatibility_definitions | geometry_compatibility_definitions
    )
    assert LegacyResolvedSourceFps is ResolvedSourceFps
    assert LegacyResolvedVideoClip is ResolvedVideoClip
    assert LegacyVideoFrameMapping is VideoFrameMapping
    assert legacy_normalized_video_frame_count is normalized_video_frame_count
    assert legacy_resolve_video_source_fps is resolve_video_source_fps
    assert LEGACY_WAN_TEMPORAL_CHUNK_SIZE == WAN_TEMPORAL_CHUNK_SIZE
    assert (
        legacy_wan_fully_observed_latent_count
        is wan_fully_observed_latent_count
    )
    assert (
        legacy_wan_raw_frame_count_to_latent_count
        is wan_raw_frame_count_to_latent_count
    )
    assert legacy_wan_safe_temporal_frame_count is wan_safe_temporal_frame_count
    assert MODEL_WAN_TEMPORAL_CHUNK_SIZE == WAN_TEMPORAL_CHUNK_SIZE
    assert (
        model_wan_fully_observed_latent_count
        is wan_fully_observed_latent_count
    )
    assert (
        model_wan_raw_frame_count_to_latent_count
        is wan_raw_frame_count_to_latent_count
    )
    assert model_wan_safe_temporal_frame_count is wan_safe_temporal_frame_count


def test_view_placement_has_one_dependency_free_owner() -> None:
    canonical_definitions = _top_level_definitions(
        PACKAGE_ROOT / "contracts" / "video.py"
    )
    compatibility_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "raw_video.py"
    )

    assert "ViewPlacement" in canonical_definitions
    assert "ViewPlacement" not in compatibility_definitions
    assert LegacyViewPlacement is ViewPlacement


def test_sample_metadata_has_one_dependency_free_owner() -> None:
    canonical_definitions = _top_level_definitions(
        PACKAGE_ROOT / "contracts" / "sample_metadata.py"
    )
    compatibility_definitions = _top_level_definitions(
        PACKAGE_ROOT / "data" / "sample_metadata.py"
    )
    canonical_names = {
        "GeneralistTrainingSampleMetadata",
        "SampleConstructionMetadata",
        "single_sample_metadata_mapping",
    }

    assert canonical_names <= canonical_definitions
    assert canonical_names.isdisjoint(compatibility_definitions)
    assert (
        LegacyGeneralistTrainingSampleMetadata
        is GeneralistTrainingSampleMetadata
    )
    assert LegacySampleConstructionMetadata is SampleConstructionMetadata
    assert (
        legacy_single_sample_metadata_mapping
        is single_sample_metadata_mapping
    )
    assert (
        LegacyGeneralistTrainingBucketMetadataKey
        is GENERALIST_TRAINING_BUCKET_METADATA_KEY
    )
    assert (
        LegacyGeneralistTrainingDropTextMetadataKey
        is GENERALIST_TRAINING_DROP_TEXT_METADATA_KEY
    )
    assert (
        LegacyGeneralistTrainingModeOverrideMetadataKey
        is GENERALIST_TRAINING_MODE_OVERRIDE_METADATA_KEY
    )
    assert (
        LegacyGeneralistTrainingSourceMetadataKey
        is GENERALIST_TRAINING_SOURCE_METADATA_KEY
    )


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
    flow_runtime_functions = {
        "expand_scalar_timestep",
        "explicit_sigma_euler_step",
        "zero_terminal_next_sigma",
    }
    flow_compatibility_names = {
        "expand_mot_scalar_timestep",
        "mot_scheduler_next_sigma",
        "step_mot_flow_with_sigmas",
    }
    retired_variant_functions = (
        routing_functions | flow_runtime_functions | flow_compatibility_names
    ) | {
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
    assert flow_runtime_functions <= _top_level_definitions(
        PACKAGE_ROOT / "models" / "common" / "flow_matching.py"
    )
    assert flow_runtime_functions.isdisjoint(
        _top_level_definitions(mot_root / "runtime.py")
    )
    assert flow_compatibility_names.isdisjoint(
        _top_level_definitions(mot_root / "runtime.py")
    )
    assert retired_variant_functions.isdisjoint(
        _top_level_definitions(mot_root / "variant.py")
    )


def test_mot_condition_latent_selection_has_one_owner() -> None:
    function_name = "resolve_mot_condition_latents"
    mot_root = PACKAGE_ROOT / "models" / "policy_variants" / "mot"

    assert function_name in _top_level_definitions(mot_root / "conditioning.py")
    assert function_name not in _top_level_definitions(mot_root / "runtime.py")


def test_sharded_execution_contexts_have_one_owner() -> None:
    public_contexts = {
        "checkpoint_unshard_context",
        "summon_full_parameters",
        "unshard_runtime_parameters",
    }
    retired_runtime_definitions = {
        "_checkpoint_summon_context",
        "_DummyCtx",
        "_FSDP2UnshardCtx",
        "_summon_full_params",
        "_unshard_runtime_params",
    }
    common_root = PACKAGE_ROOT / "models" / "common"
    mot_runtime = (
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "runtime.py"
    )

    assert public_contexts <= _top_level_definitions(
        common_root / "sharded_execution.py"
    )
    assert (public_contexts | retired_runtime_definitions).isdisjoint(
        _top_level_definitions(mot_runtime)
    )


def test_mot_cache_state_operations_have_one_owner() -> None:
    cache_operations = {
        "append_mot_action_cache",
        "move_mot_action_cache",
        "move_mot_video_cache",
        "rewind_mot_runtime_action_cache_to_frame",
        "trim_mot_action_cache_prefix",
        "trim_mot_action_cache_tail",
        "trim_mot_video_cache_tail",
    }
    mot_root = PACKAGE_ROOT / "models" / "policy_variants" / "mot"

    assert cache_operations <= _top_level_definitions(mot_root / "cache_state.py")
    assert cache_operations.isdisjoint(
        _top_level_definitions(mot_root / "runtime.py")
    )


def test_mot_cache_execution_has_one_owner() -> None:
    cache_execution_functions = {
        "forward_action_with_video_and_action_cache",
        "forward_action_with_video_cache",
        "prefill_video_kv_cache",
    }
    mot_root = PACKAGE_ROOT / "models" / "policy_variants" / "mot"

    assert cache_execution_functions <= _top_level_definitions(
        mot_root / "cache_execution.py"
    )
    assert cache_execution_functions.isdisjoint(
        _top_level_definitions(mot_root / "runtime.py")
    )
    for consumer_name in ("split_cache_inference.py", "unpacked_training.py"):
        assert "from .cache_execution import" in (
            mot_root / consumer_name
        ).read_text(encoding="utf-8")


def test_mot_dual_stream_execution_has_one_owner() -> None:
    execution_functions = {
        "forward_joint_video_action_denoise",
        "forward_mot_packed_coupling_denoise",
    }
    mot_root = PACKAGE_ROOT / "models" / "policy_variants" / "mot"

    assert execution_functions <= _top_level_definitions(
        mot_root / "dual_stream_execution.py"
    )
    assert execution_functions.isdisjoint(
        _top_level_definitions(mot_root / "runtime.py")
    )
    for consumer_name in (
        "joint_denoise_inference.py",
        "packed_inference.py",
        "packed_training.py",
        "unpacked_training.py",
    ):
        assert "from .dual_stream_execution import" in (
            mot_root / consumer_name
        ).read_text(encoding="utf-8")


def test_mot_unpacked_training_has_one_program_owner() -> None:
    mot_root = PACKAGE_ROOT / "models" / "policy_variants" / "mot"
    variant_path = mot_root / "variant.py"
    unpacked_path = mot_root / "unpacked_training.py"

    assert {
        "_build_video_train_rollout",
        "run_joint_denoise",
        "run_prefill_action_denoise",
    } <= _class_method_definitions(unpacked_path, "MoTUnpackedTrainingProgram")
    assert {
        "_build_video_train_rollout",
        "_forward_train_joint_denoise",
        "_forward_train_prefill_action_denoise",
    }.isdisjoint(_class_method_definitions(variant_path, "MoTPolicyVariant"))
    assert "from .unpacked_training import" in variant_path.read_text(
        encoding="utf-8"
    )


def test_mot_attention_layouts_have_one_owner() -> None:
    attention_builders = {
        "build_chunk_causal_video_mask",
        "build_mot_attention_mask",
        "build_mot_inference_action_attention_mask",
        "build_mot_packed_coupling_attention_mask",
        "build_mot_packed_coupling_attention_profile",
        "build_packed_action_attention_mask",
    }
    mot_root = PACKAGE_ROOT / "models" / "policy_variants" / "mot"

    assert attention_builders <= _top_level_definitions(mot_root / "attention.py")
    assert attention_builders.isdisjoint(
        _top_level_definitions(mot_root / "runtime.py")
    )


def test_superseded_exact_runtime_helpers_are_retired() -> None:
    mot_runtime_definitions = _top_level_definitions(
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "runtime.py"
    )
    parallel_runtime_definitions = _top_level_definitions(
        PACKAGE_ROOT
        / "models"
        / "policy_variants"
        / "parallel_stream"
        / "reference_runtime.py"
    )

    assert {
        "append_mot_video_cache",
        "build_packed_video_self_attention_mask",
        "forward_packed_action_with_video_cache",
        "forward_packed_video_denoise",
    }.isdisjoint(mot_runtime_definitions)
    assert {
        "_build_action_condition_volume",
        "_build_next_exact_cache_state",
        "_commit_joint_chunk_to_exact_cache",
        "_expand_condition_video_latents",
        "_sample_timestep_values",
        "should_couple_action_to_video_timesteps",
    }.isdisjoint(parallel_runtime_definitions)


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


def test_mot_packed_inference_program_has_one_execution_owner() -> None:
    packed_inference_path = (
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "packed_inference.py"
    )
    variant_path = PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "variant.py"

    assert "MoTPackedInferenceProgram" in _top_level_definitions(
        packed_inference_path
    )
    assert "run" in _class_method_definitions(
        packed_inference_path,
        "MoTPackedInferenceProgram",
    )

    delegate = _class_method(
        variant_path,
        "MoTPolicyVariant",
        "_forward_infer_packed_coupling",
    )
    assert len(delegate.body) == 1
    assert isinstance(delegate.body[0], ast.Return)
    run_call = delegate.body[0].value
    assert isinstance(run_call, ast.Call)
    assert isinstance(run_call.func, ast.Attribute)
    assert run_call.func.attr == "run"
    assert isinstance(run_call.func.value, ast.Call)
    assert isinstance(run_call.func.value.func, ast.Name)
    assert run_call.func.value.func.id == "MoTPackedInferenceProgram"


def test_mot_packed_training_program_has_one_execution_owner() -> None:
    packed_training_path = (
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "packed_training.py"
    )
    variant_path = PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "variant.py"

    assert "MoTPackedTrainingProgram" in _top_level_definitions(
        packed_training_path
    )
    assert "run" in _class_method_definitions(
        packed_training_path,
        "MoTPackedTrainingProgram",
    )

    delegate = _class_method(
        variant_path,
        "MoTPolicyVariant",
        "_forward_train_packed_coupling",
    )
    assert len(delegate.body) == 1
    assert isinstance(delegate.body[0], ast.Return)
    run_call = delegate.body[0].value
    assert isinstance(run_call, ast.Call)
    assert isinstance(run_call.func, ast.Attribute)
    assert run_call.func.attr == "run"
    assert isinstance(run_call.func.value, ast.Call)
    assert isinstance(run_call.func.value.func, ast.Name)
    assert run_call.func.value.func.id == "MoTPackedTrainingProgram"


def test_mot_non_packed_inference_programs_have_one_execution_owner() -> None:
    mot_root = PACKAGE_ROOT / "models" / "policy_variants" / "mot"
    variant_path = mot_root / "variant.py"
    program_owners = {
        "MoTJointDenoiseInferenceProgram": mot_root / "joint_denoise_inference.py",
        "MoTSplitCacheInferenceProgram": mot_root / "split_cache_inference.py",
    }
    for class_name, owner_path in program_owners.items():
        assert class_name in _top_level_definitions(owner_path)
        assert "run" in _class_method_definitions(owner_path, class_name)

    dispatcher = _class_method(
        variant_path,
        "MoTPolicyVariant",
        "forward_infer_step",
    )
    constructed_programs = {
        node.func.value.func.id
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
    }
    assert constructed_programs == {
        "MoTJointDenoiseInferenceProgram",
        "MoTSplitCacheInferenceProgram",
    }
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_forward_infer_packed_coupling"
        for node in ast.walk(dispatcher)
    )
    assert not any(
        isinstance(node, (ast.For, ast.While, ast.FunctionDef))
        for statement in dispatcher.body
        for node in ast.walk(statement)
    )


def test_retired_ablations_namespace_is_not_packaged() -> None:
    assert not (PACKAGE_ROOT / "ablations").exists()


def test_attention_cache_policy_has_one_implementation_owner() -> None:
    cache_policy_functions = {
        "merge_attention_cache_entries",
        "packed_slot_pool_query_sequence_ids",
        "prepend_cached_prefix_mask",
        "prepare_sdpa_mask",
        "resolve_slot_pool_prefix_visibility",
        "retained_slot_pool_indices_for_current_write",
    }
    cache_backend_path = PACKAGE_ROOT / "models" / "common" / "cache_backends.py"
    replica_core_path = PACKAGE_ROOT / "models" / "visual_tower" / "replica_core.py"

    assert cache_policy_functions <= _top_level_definitions(cache_backend_path)
    assert {
        f"_{name}" for name in cache_policy_functions
    }.isdisjoint(_top_level_definitions(replica_core_path))


def test_shared_transformer_support_has_one_implementation_owner() -> None:
    support_definitions = {
        "SharedTransformerAttention",
        "SharedTransformerBlock",
        "SharedTransformerRotaryPositionalEmbedding",
        "SharedTransformerTimeEmbedding",
        "apply_rotary_emb",
        "feed_forward_with_materialized_params",
        "layer_norm_with_materialized_params",
        "linear_with_materialized_params",
        "materialize_runtime_parameter",
        "rms_norm_with_materialized_weight",
        "select_chunk_slices",
        "select_split_segments",
    }
    support_path = PACKAGE_ROOT / "models" / "visual_tower" / "shared_transformer_support.py"
    replica_core_path = PACKAGE_ROOT / "models" / "visual_tower" / "replica_core.py"

    assert support_definitions <= _top_level_definitions(support_path)
    assert support_definitions.isdisjoint(_top_level_definitions(replica_core_path))
    assert {
        f"_{name}" for name in support_definitions if not name.startswith("Shared")
    }.isdisjoint(_top_level_definitions(replica_core_path))


def test_exact_single_stream_runtime_has_one_implementation_owner() -> None:
    exact_runtime_definitions = {
        "build_reference_mesh_id",
        "clear_exact_prediction_cache",
        "initialize_exact_runtime_cache",
        "prepare_exact_single_stream_forward_input",
        "prepare_exact_single_stream_input",
        "repeat_exact_single_stream_input_for_cfg",
        "resolve_runtime_module_dtype",
        "run_exact_single_stream_forward",
    }
    exact_runtime_path = PACKAGE_ROOT / "models" / "visual_tower" / "exact_runtime.py"
    reference_runtime_path = (
        PACKAGE_ROOT
        / "models"
        / "policy_variants"
        / "parallel_stream"
        / "reference_runtime.py"
    )

    assert exact_runtime_definitions <= _top_level_definitions(exact_runtime_path)
    assert {
        "_clear_exact_prediction_cache",
        "get_mesh_id",
        "initialize_reference_cache",
        "prepare_reference_forward_input",
        "prepare_reference_single_stream_input",
        "reference_runtime_dtype",
        "repeat_input_for_cfg",
        "run_reference_single_stream_forward",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))
    assert "unpatchify_video_sequence" in _top_level_definitions(
        PACKAGE_ROOT / "models" / "common" / "video_geometry.py"
    )
    assert "data_seq_to_patch" not in _top_level_definitions(reference_runtime_path)


def test_parallel_exact_cache_contract_has_one_implementation_owner() -> None:
    cache_contract_definitions = {
        "ExactCacheContext",
        "ExactCacheInterfaceSpec",
        "build_clean_video_action_cache_stream_ids",
        "build_dual_stream_cache_stream_ids",
        "build_exact_cache_spec",
        "count_single_stream_action_tokens",
        "ensure_exact_cache_initialized",
        "ensure_exact_text_embeddings",
        "existing_exact_cache_attention_window",
        "restore_slot_pool_layer_metadata",
        "resolve_exact_cache_context",
        "set_slot_pool_layer_metadata",
        "validate_existing_exact_cache_attention_window",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    cache_contract_path = parallel_stream_root / "exact_cache.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert cache_contract_definitions <= _top_level_definitions(cache_contract_path)
    assert {
        "ExactCacheContext",
        "ExactCacheInterfaceSpec",
        "_build_exact_cache_spec",
        "_ensure_exact_cache_initialized",
        "_existing_exact_cache_attn_window",
        "_restore_slot_pool_layer_metadata",
        "_resolve_exact_cache_context",
        "_set_slot_pool_layer_metadata",
        "_single_stream_action_token_count",
        "_stream_ids_for_clean_video_action_tokens",
        "_stream_ids_for_exact_dual_stream_split",
        "_validate_existing_exact_cache_attn_window",
        "ensure_reference_text_embeddings",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_cache_execution_has_one_implementation_owner() -> None:
    execution_definitions = {
        "build_joint_clean_cache_attention_mask",
        "build_joint_clean_cache_attention_profile",
        "summarize_slot_pool_cache_state",
        "write_exact_cache_chunk",
        "write_joint_clean_tokens_to_exact_cache",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    execution_path = parallel_stream_root / "cache_execution.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert execution_definitions <= _top_level_definitions(execution_path)
    assert {
        "_build_joint_clean_cache_attention_mask",
        "_build_joint_clean_cache_attention_profile",
        "_summarize_slot_pool_cache_state",
        "_write_exact_cache_chunk",
        "_write_joint_clean_tokens_to_exact_cache",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_cache_lifecycle_has_one_implementation_owner() -> None:
    lifecycle_definitions = {
        "commit_initial_observed_video_context",
        "run_parallel_exact_cache_warmup",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    lifecycle_path = parallel_stream_root / "cache_lifecycle.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert lifecycle_definitions <= _top_level_definitions(lifecycle_path)
    assert {
        "_maybe_commit_initial_observed_video_context",
        "run_parallel_exact_cache_warmup",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_inference_conditioning_has_one_implementation_owner() -> None:
    conditioning_definitions = {
        "append_generalist_mode_text_context",
        "repeat_parallel_exact_input_for_cfg",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    conditioning_path = parallel_stream_root / "inference_conditioning.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert conditioning_definitions <= _top_level_definitions(conditioning_path)
    assert {
        "_inject_generalist_mode_text_context",
        "_repeat_joint_input_for_cfg",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_forward_execution_has_one_implementation_owner() -> None:
    execution_definitions = {
        "build_parallel_first_frame_attention_profile",
        "run_parallel_action_conditioned_forward",
        "run_parallel_action_conditioned_train",
        "run_parallel_exact_dual_stream_forward",
        "run_parallel_exact_train",
        "run_parallel_first_frame_conditioned_forward",
        "run_parallel_first_frame_conditioned_train",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    execution_path = parallel_stream_root / "forward_execution.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert execution_definitions <= _top_level_definitions(execution_path)
    assert {
        "_build_fastwam_first_frame_attention_profile",
        "_run_parallel_action_conditioned_forward",
        "_run_parallel_exact_joint_forward_manual",
        "_run_parallel_fastwam_first_frame_forward_manual",
        "run_parallel_action_conditioned_train",
        "run_parallel_exact_train",
        "run_parallel_fastwam_first_frame_train",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_inference_artifacts_and_staged_rollout_have_one_owner() -> None:
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    artifact_path = parallel_stream_root / "inference_artifacts.py"
    staged_rollout_path = parallel_stream_root / "staged_rollout.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert "ParallelInferArtifacts" in _top_level_definitions(artifact_path)
    assert "run_parallel_staged_inference_rollout" in _top_level_definitions(
        staged_rollout_path
    )
    assert {
        "LingbotParallelInferArtifacts",
        "ParallelInferArtifacts",
        "run_parallel_exact_inference_rollout",
        "run_parallel_staged_inference_rollout",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_packed_rollout_has_one_implementation_owner() -> None:
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    packed_rollout_path = parallel_stream_root / "packed_rollout.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert {
        "_run_parallel_packed_inference_rollout_impl",
        "run_parallel_packed_action_override_rollout",
        "run_parallel_packed_inference_rollout",
    } <= _top_level_definitions(packed_rollout_path)
    assert {
        "_run_parallel_action_conditioned_inference_rollout_impl",
        "_run_parallel_packed_inference_rollout_impl",
        "run_parallel_action_conditioned_action_override_inference_rollout",
        "run_parallel_packed_action_override_rollout",
        "run_parallel_action_conditioned_inference_rollout",
        "run_parallel_packed_inference_rollout",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_anchored_action_rollout_has_one_implementation_owner() -> None:
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    anchored_rollout_path = parallel_stream_root / "anchored_action_rollout.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"
    owned_rollouts = {
        "run_parallel_current_frame_action_chunk_inference_rollout",
        "run_parallel_fastwam_first_frame_inference_rollout",
    }

    assert owned_rollouts <= _top_level_definitions(anchored_rollout_path)
    assert owned_rollouts.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_reference_runtime_is_a_compatibility_only_facade() -> None:
    reference_runtime_path = (
        PACKAGE_ROOT
        / "models"
        / "policy_variants"
        / "parallel_stream"
        / "reference_runtime.py"
    )

    assert not _top_level_definitions(reference_runtime_path)


def test_compatibility_export_anchors_only_reference_imported_symbols() -> None:
    anchored_modules = tuple(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "_COMPATIBILITY_EXPORTS" in path.read_text(encoding="utf-8")
    )

    assert anchored_modules
    for path in anchored_modules:
        export_names = _compatibility_export_names(path)
        assert export_names
        assert export_names <= _top_level_import_names(path)


def test_runtime_compatibility_facades_anchor_every_import() -> None:
    facade_paths = (
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "runtime.py",
        PACKAGE_ROOT
        / "models"
        / "policy_variants"
        / "parallel_stream"
        / "reference_runtime.py",
    )

    for path in facade_paths:
        assert _compatibility_export_names(path) == _top_level_import_names(path)


def test_parallel_training_artifacts_have_one_implementation_owner() -> None:
    artifact_definitions = {
        "LingbotParallelTrainArtifacts",
        "prepare_parallel_action_conditioned_train_artifacts",
        "prepare_parallel_current_frame_action_chunk_train_artifacts",
        "prepare_parallel_exact_train_artifacts",
        "prepare_parallel_fastwam_first_frame_train_artifacts",
        "prepare_parallel_prefix_condition_exact_train_artifacts",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    artifact_path = parallel_stream_root / "training_artifacts.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert artifact_definitions <= _top_level_definitions(artifact_path)
    assert artifact_definitions.isdisjoint(
        _top_level_definitions(reference_runtime_path)
    )


def test_parallel_runtime_semantics_have_one_implementation_owner() -> None:
    semantics_definitions = {
        "attention_profile_name_for_current_block_coupling",
        "prefix_visibility_mode_for_policy",
        "resolve_parallel_context_condition_latent_source",
        "resolve_parallel_current_block_coupling",
        "resolve_parallel_history_stream_visibility",
        "resolve_parallel_joint_timestep_coupling",
        "uses_legacy_prefix_per_chunk_proprio_contract",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    semantics_path = parallel_stream_root / "runtime_semantics.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"
    dynamics_rollout_path = PACKAGE_ROOT / "evals" / "dynamics" / "rollout.py"

    assert semantics_definitions <= _top_level_definitions(semantics_path)
    assert {
        "_attention_profile_name_for_current_block_coupling",
        "_prefix_visibility_mode_for_policy",
        "_uses_legacy_prefix_per_chunk_proprio_contract",
        "resolve_parallel_context_condition_latent_source",
        "resolve_parallel_current_block_coupling",
        "resolve_parallel_history_stream_visibility",
        "resolve_parallel_joint_timestep_coupling",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))
    assert "_resolve_parallel_current_block_coupling" not in _top_level_definitions(
        dynamics_rollout_path
    )


def test_parallel_conditional_rollout_has_one_implementation_owner() -> None:
    rollout_definitions = {
        "generalist_conditioning_chunk_size",
        "generalist_conditioning_history_stream_visibility",
        "generalist_conditioning_prefix_visibility_mode",
        "generalist_conditioning_window_size",
        "is_conditional_joint_denoise_mode",
        "resolve_action_conditioning_mode",
        "select_conditional_warmup_history_suffix",
        "slice_conditioning_chunk",
        "uses_generalist_mode_text_token",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    rollout_path = parallel_stream_root / "conditional_rollout.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert rollout_definitions <= _top_level_definitions(rollout_path)
    assert {
        "_chunk_size_for_generalist_conditioning",
        "_generalist_mode_for_action_conditioning",
        "_history_stream_visibility_for_generalist_conditioning",
        "_is_conditional_joint_denoise_mode",
        "_prefix_visibility_mode_for_generalist_conditioning",
        "_select_conditional_warmup_history_suffix",
        "_slice_conditioning_chunk",
        "_uses_generalist_mode_text_token",
        "_window_size_for_generalist_conditioning",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_training_noise_has_one_implementation_owner() -> None:
    noise_definitions = {
        "build_parallel_flow_noise_artifacts",
        "sample_coupled_parallel_timestep_values",
        "sample_index_matched_timestep_values",
        "sample_shared_video_schedule_timestep_values",
        "share_video_scheduler_grid_with_action_scheduler",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    noise_path = parallel_stream_root / "training_noise.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert noise_definitions <= _top_level_definitions(noise_path)
    assert {
        "_add_noise",
        "_sample_coupled_timestep_values",
        "_sample_index_matched_timestep_values",
        "_sample_shared_video_schedule_timestep_values",
        "_share_video_scheduler_grid_with_action_scheduler",
        "sample_timestep_id",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_latent_conditioning_has_one_implementation_owner() -> None:
    conditioning_definitions = {
        "build_repeated_first_frame_condition",
        "resolve_full_window_condition_latents",
        "select_first_frame_condition_latents",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    conditioning_path = parallel_stream_root / "latent_conditioning.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert conditioning_definitions <= _top_level_definitions(conditioning_path)
    assert {
        "_build_clean_video_condition_from_anchor",
        "_resolve_full_condition_latents",
        "_select_first_frame_condition_latents",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_generalist_training_has_one_implementation_owner() -> None:
    training_definitions = {
        "ParallelTrainArtifacts",
        "apply_generalist_joint_denoise_training_mode",
        "apply_generalist_legacy_prefix_joint_training_mode",
        "sample_generalist_joint_denoise_training_mode",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    training_path = parallel_stream_root / "generalist_training.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert training_definitions <= _top_level_definitions(training_path)
    assert {
        "_apply_generalist_joint_denoise_training_mode",
        "_apply_generalist_legacy_prefix_joint_training_mode",
        "_sample_joint_denoise_training_mode",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_parallel_proprio_conditioning_has_one_implementation_owner() -> None:
    conditioning_definitions = {
        "apply_parallel_chunk_proprio_context",
        "build_single_stream_hidden_proprio_context",
        "inject_deprecated_proprio_text_context",
    }
    parallel_stream_root = (
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream"
    )
    conditioning_path = parallel_stream_root / "proprio_conditioning.py"
    reference_runtime_path = parallel_stream_root / "reference_runtime.py"

    assert conditioning_definitions <= _top_level_definitions(conditioning_path)
    assert {
        "_apply_parallel_chunk_proprio_context",
        "_inject_proprio_text_context",
        "_single_stream_hidden_proprio_context",
    }.isdisjoint(_top_level_definitions(reference_runtime_path))


def test_visual_runtime_tensor_transport_has_one_implementation_owner() -> None:
    transport_definitions = {
        "cached_attention_profile",
        "cached_optional_tensor",
        "move_attention_profile",
        "move_optional_tensor",
        "move_slot_pool_layer_state",
    }
    transport_path = PACKAGE_ROOT / "models" / "visual_tower" / "runtime_tensor_transport.py"
    replica_core_path = PACKAGE_ROOT / "models" / "visual_tower" / "replica_core.py"

    assert transport_definitions <= _top_level_definitions(transport_path)
    assert transport_definitions.isdisjoint(_top_level_definitions(replica_core_path))


def test_visual_context_encoders_have_one_implementation_owner() -> None:
    encoder_definitions = {
        "GeneralistModeContextEncoder",
        "ProprioContextEncoder",
        "ProprioHiddenContextEncoder",
    }
    encoder_path = PACKAGE_ROOT / "models" / "visual_tower" / "context_encoders.py"
    replica_core_path = PACKAGE_ROOT / "models" / "visual_tower" / "replica_core.py"

    assert encoder_definitions <= _top_level_definitions(encoder_path)
    assert encoder_definitions.isdisjoint(_top_level_definitions(replica_core_path))


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


def test_retired_generic_parallel_layout_scaffold_is_not_packaged() -> None:
    retired_paths = (
        PACKAGE_ROOT / "models" / "policy_variants" / "common" / "caches.py",
        PACKAGE_ROOT / "models" / "policy_variants" / "common" / "masks.py",
        PACKAGE_ROOT / "models" / "policy_variants" / "common" / "positions.py",
        PACKAGE_ROOT / "models" / "policy_variants" / "common" / "timesteps.py",
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream" / "masks.py",
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream" / "packing.py",
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream" / "positions.py",
        PACKAGE_ROOT / "models" / "policy_variants" / "parallel_stream" / "timesteps.py",
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


def test_private_gjd_conditioning_study_driver_is_retired() -> None:
    retired_paths = (
        REPO_ROOT / "scripts" / "analyze_gjd_conditioning_sensitivity.py",
        REPO_ROOT / "notes" / "gjd_attention_and_rollout_parity_overnight_20260708.md",
    )
    packed_block = (
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "packed_block.py"
    ).read_text(encoding="utf-8")
    packed_runtime = (
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "runtime.py"
    ).read_text(encoding="utf-8")
    variant = (
        PACKAGE_ROOT / "models" / "policy_variants" / "mot" / "variant.py"
    ).read_text(encoding="utf-8")

    assert not any(path.exists() for path in retired_paths)
    assert "mot_collect_attention_focus" not in variant
    assert "attention_diagnostics" not in packed_block
    assert "attention_diagnostics" not in packed_runtime


def test_libero_mot_drivers_delegate_to_the_package_episode_runner() -> None:
    single_source = (
        REPO_ROOT / "scripts" / "run_libero_mot_visualization.py"
    ).read_text(encoding="utf-8")
    batch_source = (
        REPO_ROOT / "scripts" / "run_libero_mot_batch_visualization.py"
    ).read_text(encoding="utf-8")
    package_source = (
        PACKAGE_ROOT / "evals" / "libero_mot_rollout.py"
    ).read_text(encoding="utf-8")
    artifact_source = (
        PACKAGE_ROOT / "evals" / "libero_rollout_artifacts.py"
    ).read_text(encoding="utf-8")
    visualization_source = (
        PACKAGE_ROOT / "evals" / "libero_visualization.py"
    ).read_text(encoding="utf-8")
    observed_history_source = (
        PACKAGE_ROOT
        / "models"
        / "policy_variants"
        / "mot"
        / "observed_history.py"
    ).read_text(encoding="utf-8")

    assert "run_mot_libero_episode(" in single_source
    assert "run_mot_libero_episode(" in batch_source
    assert "importlib.util" not in batch_source
    assert "mot_viz._" not in batch_source
    assert "._forward_infer_with_visual_outputs(" not in package_source
    assert "runner.reconcile_observed_history(" in package_source
    assert "persist_libero_rollout_artifacts(" in package_source
    for artifact_implementation in (
        "imageio",
        "ImageDraw",
        "VideoProcessor",
        "_actions.jsonl",
        "_chunks.json",
        "_decode_latent_video",
    ):
        assert artifact_implementation not in package_source
        assert artifact_implementation in artifact_source
    assert "def decode_imagined_video" not in visualization_source
    assert "def build_comparison_video_frames" not in visualization_source
    for policy_state_field in (
        "past_clean_latents",
        "past_clean_actions",
        "past_hidden_proprio_states",
        "pending_predicted_video_frames",
    ):
        assert policy_state_field not in package_source
        assert policy_state_field in observed_history_source


def test_libero_mot_runtime_loading_has_one_owner() -> None:
    runtime_path = PACKAGE_ROOT / "evals" / "libero_mot_runtime.py"
    rollout_path = PACKAGE_ROOT / "evals" / "libero_mot_rollout.py"
    single_driver_path = REPO_ROOT / "scripts" / "run_libero_mot_visualization.py"
    batch_driver_path = (
        REPO_ROOT / "scripts" / "run_libero_mot_batch_visualization.py"
    )
    runtime_definitions = _top_level_definitions(runtime_path)
    rollout_definitions = _top_level_definitions(rollout_path)
    runtime_contract = {
        "MotLiberoLoadOptions",
        "MotLiberoRuntime",
        "_action_per_frame",
        "_build_component_report",
        "_frame_chunk_size",
        "_maybe_merge_checkpoint_runtime_config",
        "_require_current_frontend_encode_mode",
        "_resolve_mot_checkpoint_path",
        "_validate_live_sim_mot_generalist_rollout_mode",
        "_validate_mot_config",
        "load_mot_libero_runtime",
        "print_rollout_event",
    }

    assert runtime_contract <= runtime_definitions
    assert runtime_contract.isdisjoint(rollout_definitions)
    assert "open_wam.evals.libero_mot_runtime" in _absolute_imports_for_file(
        rollout_path
    )
    for driver_path in (single_driver_path, batch_driver_path):
        assert "open_wam.evals.libero_mot_runtime" in _absolute_imports_for_file(
            driver_path
        )
    runtime_source = runtime_path.read_text(encoding="utf-8")
    rollout_source = rollout_path.read_text(encoding="utf-8")
    for loading_dependency in (
        "build_variant_pipeline_from_config",
        "load_experiment_config",
        "load_pipeline_checkpoint",
        "merge_runtime_config_from_checkpoint",
        "resolve_checkpoint_file",
    ):
        assert loading_dependency in runtime_source
        assert loading_dependency not in rollout_source


def test_libero_integration_roles_have_one_owner() -> None:
    task_path = PACKAGE_ROOT / "integrations" / "libero_tasks.py"
    runtime_path = PACKAGE_ROOT / "integrations" / "libero_runtime.py"
    control_path = PACKAGE_ROOT / "integrations" / "libero_control.py"
    tracking_path = PACKAGE_ROOT / "integrations" / "libero_tracking.py"
    env_path = PACKAGE_ROOT / "integrations" / "libero_env.py"
    task_definitions = _top_level_definitions(task_path)
    runtime_definitions = _top_level_definitions(runtime_path)
    control_definitions = _top_level_definitions(control_path)
    tracking_definitions = _top_level_definitions(tracking_path)
    env_definitions = _top_level_definitions(env_path)
    task_contract = {
        "LiberoTaskSpec",
        "ensure_local_libero_config",
        "infer_task_local_episode_rank",
        "load_libero_task_init_states",
        "resolve_libero_task",
        "resolve_libero_task_by_id",
    }

    assert task_contract <= task_definitions
    assert task_contract.isdisjoint(env_definitions)
    runtime_contract = {
        "build_libero_control_env",
        "build_libero_offscreen_env",
    }
    control_contract = {
        "LiberoControlConfig",
        "absolute_joint_position_to_libero_joint_delta_action",
        "compute_osc_pose_action",
        "disable_libero_joint_position_controller_interpolator",
        "extract_gripper_positions_from_obs",
        "extract_joint_positions_from_obs",
        "extract_pose_from_obs",
        "gripper_command_for_substep",
        "gripper_qpos_tracking_command",
        "integrated_eef6d_target_to_osc_action",
        "integrated_eef6d_target_to_osc_action_from_arrays",
        "project_libero_gripper_state",
        "quaternion_angular_error_degrees",
        "quaternion_xyzw_to_rotation_matrix",
        "resolve_libero_joint_delta_limit",
        "resolve_libero_joint_limit_array",
        "resolve_libero_joint_scale_array",
        "set_libero_joint_position_controller_gain",
        "step_libero_absolute_joint_position_goal",
    }
    tracking_contract = {
        "LiberoTrackingResult",
        "track_relative_targets_in_libero_env",
    }
    assert runtime_contract <= runtime_definitions
    assert control_contract <= control_definitions
    assert tracking_contract <= tracking_definitions
    assert runtime_contract.isdisjoint(env_definitions)
    assert control_contract.isdisjoint(env_definitions)
    assert tracking_contract.isdisjoint(env_definitions)
    assert env_definitions == {
        "LiberoBenchmarkAdapter",
        "LiberoEnvConfig",
        "_source_action_from_model_action",
    }
    assert {
        "open_wam.integrations.libero_control",
        "open_wam.integrations.libero_runtime",
        "open_wam.integrations.libero_tasks",
        "open_wam.integrations.libero_tracking",
    } <= _absolute_imports_for_file(env_path)


def test_public_config_enums_are_declared_once() -> None:
    path = PACKAGE_ROOT / "configs" / "enums.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert duplicates == []


def test_legacy_backbone_config_import_is_identity_preserving() -> None:
    assert LegacySharedVideoTransformerConfig is SharedVideoTransformerConfig
