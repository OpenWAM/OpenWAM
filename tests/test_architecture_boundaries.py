from __future__ import annotations

import ast
from pathlib import Path

from open_wam.configs import SharedVideoTransformerConfig
from open_wam.models.video_backbone.config import (
    SharedVideoTransformerConfig as LegacySharedVideoTransformerConfig,
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


def test_deprecated_scripts_do_not_contain_python_runtime_implementations() -> None:
    deprecated_script_root = REPO_ROOT / "scripts" / "deprecated"

    assert list(deprecated_script_root.glob("*.py")) == []


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


def test_public_config_enums_are_declared_once() -> None:
    path = PACKAGE_ROOT / "configs" / "enums.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert duplicates == []


def test_legacy_backbone_config_import_is_identity_preserving() -> None:
    assert LegacySharedVideoTransformerConfig is SharedVideoTransformerConfig
