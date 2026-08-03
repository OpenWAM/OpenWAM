from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
from typing import Any

from check_release_metadata import (
    private_sdist_path_violations,
    validate_project_metadata,
    validate_release_build_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run static, no-Torch checks for public CI.

    This script intentionally uses only the Python standard library and does
    not import ``open_wam``. The GitHub workflow runs it directly with the
    setup-python interpreter so basic PR checks do not install Torch or the
    simulator/training stack.
    """

    if os.environ.get("OPEN_WAM_CI_NO_TORCH") == "1" and importlib.util.find_spec("torch") is not None:
        raise SystemExit("Torch is importable in a no-Torch CI job. Run this check without project dependencies.")

    pyproject = _read_toml(REPO_ROOT / "pyproject.toml")
    scripts = pyproject["project"]["scripts"]
    optional_deps = pyproject["project"].get("optional-dependencies", {})

    _check_console_scripts(scripts)
    _check_release_build_config(pyproject)
    _check_optional_dependency_duplicates(pyproject["project"].get("dependencies", ()), optional_deps)
    _check_public_local_paths_sample()
    _check_baseline_templates_are_portable()
    _check_artifact_manifest()
    _check_docs_and_cards()
    _check_docs_site_source()
    experiment_paths = _check_experiment_configs()
    example_paths = _check_example_configs()
    eval_paths = _check_eval_configs(experiment_paths)
    _check_no_merge_conflict_markers()
    _check_static_source_contracts()
    _check_workflow_is_no_torch()
    _check_hardware_workflow()
    _check_pages_workflow()

    summary = {
        "artifact_manifest_entries": len(_artifact_blocks(REPO_ROOT / "configs" / "artifacts.sample.yaml")),
        "console_scripts": sorted(scripts),
        "docs_site": "staged",
        "eval_configs": len(eval_paths),
        "example_configs": len(example_paths),
        "experiment_configs": len(experiment_paths),
        "project_version": pyproject["project"]["version"],
        "torch_importable": importlib.util.find_spec("torch") is not None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _check_release_build_config(pyproject: dict[str, Any]) -> None:
    try:
        validate_project_metadata(pyproject)
        validate_release_build_config(pyproject)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    private_paths = private_sdist_path_violations(REPO_ROOT)
    if private_paths:
        raise SystemExit(f"Private paths found in the public sdist surface: {private_paths}")


def _check_console_scripts(scripts: dict[str, str]) -> None:
    expected = {
        "open-wam-train": "open_wam.cli.train:main",
        "open-wam-eval": "open_wam.cli.eval:main",
        "open-wam-inspect-config": "open_wam.cli.inspect_config:main",
        "open-wam-validate-config": "open_wam.cli.validate_config:main",
        "open-wam-sanity": "open_wam.cli.sanity:main",
        "open-wam-sim-rollout": "open_wam.cli.sim_rollout:main",
    }
    if scripts != expected:
        raise SystemExit(f"Unexpected console script declarations: {scripts!r}")
    for target in scripts.values():
        module_name, _, function_name = target.partition(":")
        module_path = REPO_ROOT / "src" / Path(*module_name.split(".")).with_suffix(".py")
        if not module_path.is_file():
            raise SystemExit(f"Console script target module is missing: {module_path.relative_to(REPO_ROOT)}")
        source = module_path.read_text(encoding="utf-8")
        if f"def {function_name}" not in source and f"import {function_name}" not in source:
            raise SystemExit(f"Console script target {target!r} does not expose {function_name!r}.")

    for script_name in (
        "train.py",
        "eval.py",
        "inspect_config.py",
        "validate_configs_static.py",
        "run_benchmark_pipeline_sanity.py",
        "run_sim_realtime_sandbox.py",
    ):
        script_path = REPO_ROOT / "scripts" / script_name
        if not script_path.is_file():
            raise SystemExit(f"Expected root script is missing: {script_path.relative_to(REPO_ROOT)}")


def _check_optional_dependency_duplicates(base_deps: list[str], optional_deps: dict[str, list[str]]) -> None:
    base_names = {_dependency_name(item) for item in base_deps}
    allowed = {
        "full": {"bddl", "cloudpickle", "easydict", "future", "gym", "hydra-core", "mujoco", "robosuite"},
        "libero": {"bddl", "cloudpickle", "easydict", "future", "gym", "hydra-core", "robosuite"},
        "sim": {"bddl", "cloudpickle", "easydict", "future", "gym", "hydra-core", "robosuite"},
    }
    duplicates: dict[str, list[str]] = {}
    for extra_name, deps in optional_deps.items():
        duplicate_names = sorted({_dependency_name(item) for item in deps}.intersection(base_names))
        duplicate_names = [name for name in duplicate_names if name not in allowed.get(extra_name, set())]
        if duplicate_names:
            duplicates[extra_name] = duplicate_names
    if duplicates:
        raise SystemExit(f"Optional extras duplicate base dependencies: {duplicates!r}")


def _dependency_name(requirement: str) -> str:
    for separator in ("[", "<", ">", "=", "!", "~", ";"):
        requirement = requirement.split(separator, 1)[0]
    return requirement.strip().lower().replace("_", "-")


def _check_public_local_paths_sample() -> None:
    sample = (REPO_ROOT / "configs" / "local_paths.sample.yaml").read_text(encoding="utf-8")
    forbidden = ("/simurgh", "/afs/", "/sailhome/", "private-user", "private-user")
    leaks = [value for value in forbidden if value in sample]
    if leaks:
        raise SystemExit(f"configs/local_paths.sample.yaml contains private path fragments: {leaks!r}")
    if "paths" not in _top_level_keys(REPO_ROOT / "configs" / "local_paths.sample.yaml"):
        raise SystemExit("configs/local_paths.sample.yaml must define top-level paths.")


def _check_baseline_templates_are_portable() -> None:
    forbidden = ("/afs/", "/hai/", "/scr/", "/simurgh2/")
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "baselines").rglob("*")):
        if path.suffix not in {".md", ".py", ".sh", ".yaml", ".yml"}:
            continue
        source = path.read_text(encoding="utf-8")
        if any(prefix in source for prefix in forbidden):
            violations.append(str(path.relative_to(REPO_ROOT)))
    if violations:
        raise SystemExit(
            "Tracked baseline templates contain private absolute paths: "
            f"{violations!r}"
        )


def _check_artifact_manifest() -> None:
    required = {
        "artifact_id",
        "method_family",
        "variant",
        "benchmark",
        "config",
        "local_path_alias",
        "expected_layout",
        "download_url",
        "checksum",
        "license",
        "source",
        "notes",
    }
    artifacts = _artifact_blocks(REPO_ROOT / "configs" / "artifacts.sample.yaml")
    if not artifacts:
        raise SystemExit("configs/artifacts.sample.yaml must contain a non-empty artifacts list.")
    for index, artifact in enumerate(artifacts):
        missing = sorted(required.difference(artifact))
        if missing:
            raise SystemExit(f"Artifact entry {index} is missing required fields: {missing!r}")


def _check_docs_and_cards() -> None:
    required_paths = (
        "CHANGELOG.md",
        "docs/release.md",
        "docs/architecture.md",
        "docs/benchmarks.md",
        "docs/method_families.md",
        "docs/running_experiments.md",
        "docs/cookbooks/new_method.md",
        "docs/cookbooks/new_action_decoder.md",
        "docs/cookbooks/new_dataset.md",
        "docs/cookbooks/new_simulator_adapter.md",
        "docs/cookbooks/reproduce_result.md",
        "docs/cards/README.md",
        "docs/cards/public_tiny_synthetic_contract.md",
        "docs/index.md",
        "mkdocs.yml",
        "scripts/build_docs_site.py",
        ".github/workflows/pages.yml",
    )
    missing = [path for path in required_paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required public docs/site files: {missing!r}")


def _check_docs_site_source() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_docs_site.py"),
                "--output",
                tmpdir,
            ],
            check=True,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
        )
        required = (
            "index.md",
            "quickstart.md",
            "architecture.md",
            "method_families.md",
            "benchmarks.md",
            "running_experiments.md",
        )
        missing = [path for path in required if not (Path(tmpdir) / path).is_file()]
        if missing:
            raise SystemExit(f"Generated docs site is missing expected pages: {missing!r}")
        if (Path(tmpdir) / "engineering-notes").exists():
            raise SystemExit("Generated docs site must not publish raw engineering notes.")


def _check_experiment_configs() -> tuple[Path, ...]:
    paths = tuple(sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml")))
    if not paths:
        raise SystemExit("No experiment configs found.")
    required_top_level = {"data", "trainer", "backbone"}
    for path in paths:
        top_level = _top_level_keys(path)
        missing = sorted(required_top_level.difference(top_level))
        if missing:
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing top-level fields: {missing!r}")
        has_policy_variant = "policy_variant" in top_level
        has_action_decoder = "action_decoder" in top_level
        has_legacy_action_head = "action_head" in top_level
        if not has_policy_variant and not has_legacy_action_head:
            raise SystemExit(
                f"{path.relative_to(REPO_ROOT)} must define policy_variant or legacy action_head."
            )
        if not _section_has_key(path, "data", "dataset_type") and not _section_has_key(path, "data", "dataset_name"):
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing data.dataset_type or data.dataset_name.")
        if has_policy_variant and not _section_has_key(path, "policy_variant", "name"):
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing policy_variant.name.")
        if has_action_decoder and not _section_has_key(path, "action_decoder", "name"):
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing action_decoder.name.")
        if has_legacy_action_head and not _section_has_key(path, "action_head", "name"):
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing action_head.name.")
    return paths


def _check_example_configs() -> tuple[Path, ...]:
    paths = tuple(sorted((REPO_ROOT / "configs" / "examples").glob("*.yaml")))
    for path in paths:
        if "data" not in _top_level_keys(path):
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing data.")
    return paths


def _check_eval_configs(experiment_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    experiment_strings = {
        str(path.relative_to(REPO_ROOT)) for path in experiment_paths
    }
    paths = tuple(sorted((REPO_ROOT / "configs" / "evals").glob("*.yaml")))
    if not paths:
        raise SystemExit("No eval configs found.")
    for path in paths:
        experiment_config = _top_level_scalar(path, "experiment_config") or str(path.relative_to(REPO_ROOT))
        if experiment_config not in experiment_strings and not (REPO_ROOT / experiment_config).is_file():
            raise SystemExit(
                f"{path.relative_to(REPO_ROOT)} references missing experiment_config {experiment_config!r}."
            )
        for int_field in ("max_batches", "max_trajectories", "max_steps_per_trajectory", "batch_size"):
            raw_value = _top_level_scalar(path, int_field)
            if raw_value not in (None, "", "null", "None") and int(raw_value) <= 0:
                raise SystemExit(f"{path.relative_to(REPO_ROOT)} has non-positive {int_field}.")
    return paths


def _check_no_merge_conflict_markers() -> None:
    roots = ("configs", "docs", "notes", "scripts", "src", "tests")
    suffixes = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
    markers = ("<<<<<<< ", ">>>>>>> ")
    violations: list[str] = []

    for root in roots:
        for path in (REPO_ROOT / root).rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if line.startswith(markers):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{line_number}")

    if violations:
        raise SystemExit(f"Unresolved merge conflict markers: {violations!r}")


def _check_static_source_contracts() -> None:
    source_checks = {
        "src/open_wam/contracts/paths.py": ("def find_repo_root", "parents[3]"),
        "src/open_wam/runtime/results.py": ("RESERVED_RESULT_KEYS", "envelope.update(extra)"),
        "src/open_wam/pipelines/registries.py": ("BuilderRegistry[ActionDecoderName", "BuilderRegistry[object"),
        "src/open_wam/__init__.py": ("version(\"open-wam\")", "__version__ = \"0.1.0\""),
    }
    for relative, (required, forbidden) in source_checks.items():
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        if required not in source:
            raise SystemExit(f"{relative} is missing expected source contract {required!r}.")
        if forbidden in source:
            raise SystemExit(f"{relative} still contains forbidden source contract {forbidden!r}.")

    legacy_bridge = REPO_ROOT / "src" / "open_wam" / "cli" / "_legacy_script.py"
    if legacy_bridge.exists():
        raise SystemExit("Installed commands must not depend on the retired legacy-script bridge.")
    for cli_path in sorted((REPO_ROOT / "src" / "open_wam" / "cli").glob("*.py")):
        source = cli_path.read_text(encoding="utf-8")
        forbidden_tokens = ("run_legacy_script", "runpy", "scripts/")
        present = [token for token in forbidden_tokens if token in source]
        if present:
            relative = cli_path.relative_to(REPO_ROOT)
            raise SystemExit(
                f"Installed command {relative} depends on checkout-only execution: {present!r}."
            )

def _check_workflow_is_no_torch() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    if "OPEN_WAM_CI_NO_TORCH" not in workflow:
        raise SystemExit("CI workflow must assert the no-Torch basic pathway environment.")
    forbidden = ("uv sync", "--extra train", "pytest -m", "--with pyyaml")
    present = [token for token in forbidden if token in workflow]
    if present:
        raise SystemExit(f"CI workflow still contains heavy install/test tokens: {present!r}")


def _check_hardware_workflow() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    marker = "  hardware-workspace-sanity:"
    if marker not in workflow:
        raise SystemExit("CI workflow is missing the hardware workspace sanity job.")
    job = workflow.split(marker, maxsplit=1)[1]
    next_job = re.search(r"(?m)^  [A-Za-z0-9_-]+:\s*$", job)
    if next_job is not None:
        job = job[: next_job.start()]
    required = (
        "python -m compileall -q deployment",
        "python -m pyflakes deployment",
        'find_spec("torch") is None',
        'find_spec("rclpy") is None',
        "pytest deployment/tests -q",
    )
    missing = [token for token in required if token not in job]
    if missing:
        raise SystemExit(
            f"Hardware workspace CI is missing expected gates: {missing!r}"
        )
    forbidden = (
        "uv sync",
        "pip install .",
        "open-wam[",
        "pip install torch",
        "--with torch",
    )
    present = [token for token in forbidden if token in job.lower()]
    if present:
        raise SystemExit(
            f"Hardware workspace CI contains heavy package installs: {present!r}"
        )


def _check_pages_workflow() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    required = (
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v4",
        "actions/deploy-pages@v4",
        "mkdocs==1.6.1",
        "python scripts/build_docs_site.py --output .docs_site",
        "mkdocs build --clean",
    )
    missing = [token for token in required if token not in workflow]
    if missing:
        raise SystemExit(f"Pages workflow is missing expected docs deployment steps: {missing!r}")
    forbidden = ("uv sync", "--extra train", "pytest", "pip install .")
    present = [token for token in forbidden if token in workflow]
    if present:
        raise SystemExit(f"Pages workflow contains heavy install/test tokens: {present!r}")


def _top_level_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "\t", "#")) or not line.strip():
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
        if match:
            keys.add(match.group(1))
    return keys


def _top_level_scalar(path: Path, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.*)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "\t", "#")):
            continue
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip()
        if not value:
            return ""
        return _strip_scalar(value)
    return None


def _section_has_key(path: Path, section: str, key: str) -> bool:
    in_section = False
    key_pattern = re.compile(rf"^\s+{re.escape(key)}\s*:")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            in_section = bool(re.match(rf"^{re.escape(section)}\s*:", line))
            continue
        if in_section and key_pattern.match(line):
            return True
    return False


def _artifact_blocks(path: Path) -> list[set[str]]:
    blocks: list[set[str]] = []
    current: set[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("  - "):
            if current is not None:
                blocks.append(current)
            current = set()
            item = line.removeprefix("  - ").strip()
            if ":" in item:
                current.add(item.split(":", 1)[0].strip())
            continue
        if current is not None and line.startswith("    ") and ":" in stripped:
            current.add(stripped.split(":", 1)[0].strip())
    if current is not None:
        blocks.append(current)
    return blocks


def _strip_scalar(value: str) -> str:
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


if __name__ == "__main__":
    main()
