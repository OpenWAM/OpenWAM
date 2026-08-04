from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/build_libero_replay_metadata.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_libero_replay_metadata_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_subset_roots_resolve_exact_directories(tmp_path: Path) -> None:
    script = _load_script()
    libero_10 = tmp_path / "libero_10"
    libero_90 = tmp_path / "custom_90_location"

    roots = script.resolve_subset_dataset_roots(
        ["libero_10", "libero_90"],
        dataset_root=None,
        subset_root_specs=[
            f"libero_10={libero_10}",
            f"libero_90={libero_90}",
        ],
    )

    assert roots == {"libero_10": libero_10, "libero_90": libero_90}


def test_compatibility_parent_root_keeps_previous_layout(tmp_path: Path) -> None:
    script = _load_script()

    roots = script.resolve_subset_dataset_roots(
        ["libero_10"],
        dataset_root=tmp_path,
        subset_root_specs=[],
    )

    assert roots == {"libero_10": tmp_path / "libero_10"}


def test_root_resolver_rejects_ambiguous_direct_calls(tmp_path: Path) -> None:
    script = _load_script()

    with pytest.raises(ValueError, match="mutually exclusive"):
        script.resolve_subset_dataset_roots(
            ["libero_10"],
            dataset_root=tmp_path,
            subset_root_specs=[f"libero_10={tmp_path / 'libero_10'}"],
        )


def test_explicit_subset_roots_require_exact_selected_set(tmp_path: Path) -> None:
    script = _load_script()

    with pytest.raises(ValueError, match="missing mappings for libero_90"):
        script.resolve_subset_dataset_roots(
            ["libero_10", "libero_90"],
            dataset_root=None,
            subset_root_specs=[f"libero_10={tmp_path / 'libero_10'}"],
        )


def test_parser_rejects_ambiguous_parent_and_subset_roots(tmp_path: Path) -> None:
    parser = _load_script().build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset-root",
                str(tmp_path),
                "--subset-root",
                f"libero_10={tmp_path / 'libero_10'}",
                "--diagnostic-root",
                str(tmp_path / "diagnostics"),
                "--from-installed-meta",
            ]
        )


def test_parser_requires_exactly_one_replay_source(tmp_path: Path) -> None:
    parser = _load_script().build_arg_parser()
    base_args = [
        "--subset-root",
        f"libero_10={tmp_path / 'libero_10'}",
        "--diagnostic-root",
        str(tmp_path / "diagnostics"),
        "--subsets",
        "libero_10",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(base_args)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *base_args,
                "--source-run",
                "libero_10=replay_run",
                "--from-installed-meta",
            ]
        )
