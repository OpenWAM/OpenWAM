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


def test_legacy_backbone_config_import_is_identity_preserving() -> None:
    assert LegacySharedVideoTransformerConfig is SharedVideoTransformerConfig
