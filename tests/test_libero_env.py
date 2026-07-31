from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

import open_wam.integrations as integrations
from open_wam.integrations import libero_env, libero_tasks


def test_libero_task_contract_has_one_canonical_owner() -> None:
    assert integrations.LiberoTaskSpec is libero_tasks.LiberoTaskSpec
    assert libero_env.LiberoTaskSpec is libero_tasks.LiberoTaskSpec
    assert (
        integrations.resolve_libero_task_by_id
        is libero_tasks.resolve_libero_task_by_id
    )
    assert (
        libero_env.resolve_libero_task_by_id
        is libero_tasks.resolve_libero_task_by_id
    )


def test_ensure_local_libero_config_writes_deterministic_checkout_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "LIBERO"
    package_root = repo_root / "libero" / "libero"
    package_root.mkdir(parents=True)
    monkeypatch.setattr(
        libero_tasks,
        "_resolve_libero_paths",
        lambda: (repo_root, package_root),
    )
    monkeypatch.setenv("LIBERO_CONFIG_PATH", "original-config")
    project_root = tmp_path / "Open-WAM"

    config_path = libero_tasks.ensure_local_libero_config(project_root)
    first_bytes = config_path.read_bytes()
    second_path = libero_tasks.ensure_local_libero_config(project_root)

    assert second_path == config_path
    assert second_path.read_bytes() == first_bytes
    assert yaml.safe_load(first_bytes) == {
        "benchmark_root": str(package_root.resolve()),
        "bddl_files": str((package_root / "bddl_files").resolve()),
        "init_states": str((package_root / "init_files").resolve()),
        "datasets": str((repo_root / "libero" / "datasets").resolve()),
        "assets": str((package_root / "assets").resolve()),
    }


def test_infer_task_local_episode_rank_normalizes_task_identity() -> None:
    records = (
        SimpleNamespace(episode_index=3, tasks=("Pick   Cup",)),
        SimpleNamespace(episode_index=7, tasks=("pick cup",)),
        SimpleNamespace(episode_index=9, tasks=("other",)),
    )

    assert libero_tasks.infer_task_local_episode_rank(
        records,
        episode_index=7,
        task_text=" PICK CUP ",
    ) == 1


def test_load_libero_task_init_states_disables_weights_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import torch

    task_spec = libero_tasks.LiberoTaskSpec(
        benchmark_name="libero_10",
        task_id=2,
        task_name="put_cup_away",
        task_language="put cup away",
        problem_folder="suite",
        bddl_file_path="/tasks/put_cup_away.bddl",
        init_states_path=str(tmp_path / "put_cup_away.pruned_init"),
    )
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        libero_tasks,
        "ensure_local_libero_config",
        lambda project_root=None: tmp_path / "config.yaml",
    )

    def _load(path: str, *, weights_only: bool) -> str:
        calls.append((path, weights_only))
        return "states"

    monkeypatch.setattr(torch, "load", _load)

    result = libero_tasks.load_libero_task_init_states(task_spec)

    assert result == "states"
    assert calls == [(task_spec.init_states_path, False)]


def test_resolve_libero_paths_uses_fallback_checkout_without_import(monkeypatch, tmp_path: Path) -> None:
    checkout_root = tmp_path / "LIBERO"
    package_root = checkout_root / "libero" / "libero"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    def _raise_import_error(name: str):
        raise EOFError("non-interactive import prompt")

    monkeypatch.setattr(libero_tasks.importlib, "import_module", _raise_import_error)
    monkeypatch.setattr(
        libero_tasks,
        "_project_root",
        lambda _: tmp_path / "Open-WAM",
    )

    resolved_repo_root, resolved_package_root = (
        libero_tasks._resolve_libero_paths()
    )

    assert resolved_repo_root == checkout_root.resolve()
    assert resolved_package_root == package_root.resolve()
    assert str(checkout_root.resolve()) in libero_tasks.sys.path


def test_resolve_libero_paths_reraises_non_libero_module_errors(monkeypatch) -> None:
    def _raise_internal_module_error(name: str):
        raise ModuleNotFoundError("missing dependency", name="robosuite")

    monkeypatch.setattr(
        libero_tasks.importlib,
        "import_module",
        _raise_internal_module_error,
    )

    try:
        libero_tasks._resolve_libero_paths()
    except ModuleNotFoundError as exc:
        assert exc.name == "robosuite"
    else:
        raise AssertionError("Expected internal ModuleNotFoundError to be re-raised.")
