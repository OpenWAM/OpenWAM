from __future__ import annotations

from pathlib import Path

from open_wam.integrations import libero_env


def test_resolve_libero_paths_uses_fallback_checkout_without_import(monkeypatch, tmp_path: Path) -> None:
    checkout_root = tmp_path / "LIBERO"
    package_root = checkout_root / "libero" / "libero"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    def _raise_import_error(name: str):
        raise EOFError("non-interactive import prompt")

    monkeypatch.setattr(libero_env.importlib, "import_module", _raise_import_error)
    monkeypatch.setattr(libero_env, "_project_root", lambda _: tmp_path / "Open-WAM")

    resolved_repo_root, resolved_package_root = libero_env._resolve_libero_paths()

    assert resolved_repo_root == checkout_root.resolve()
    assert resolved_package_root == package_root.resolve()
    assert str(checkout_root.resolve()) in libero_env.sys.path


def test_resolve_libero_paths_reraises_non_libero_module_errors(monkeypatch) -> None:
    def _raise_internal_module_error(name: str):
        raise ModuleNotFoundError("missing dependency", name="robosuite")

    monkeypatch.setattr(libero_env.importlib, "import_module", _raise_internal_module_error)

    try:
        libero_env._resolve_libero_paths()
    except ModuleNotFoundError as exc:
        assert exc.name == "robosuite"
    else:
        raise AssertionError("Expected internal ModuleNotFoundError to be re-raised.")
