from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ("src", "scripts", "baselines", "deployment", "templates")
TRUSTED_NUMPY_PICKLE_OWNER = Path("src/open_wam/artifacts/serialization.py")
TRUSTED_NUMPY_PICKLE_FUNCTION = "load_trusted_numpy_pickle_artifact"


def _import_aliases(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    modules: dict[str, str] = {}
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"numpy", "pickle", "torch"}:
                    modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "numpy",
            "pickle",
            "torch",
        }:
            for alias in node.names:
                functions[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return modules, functions


def _qualified_call_name(
    call: ast.Call,
    *,
    modules: dict[str, str],
    functions: dict[str, str],
) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return functions.get(function.id)
    if not isinstance(function, ast.Attribute) or not isinstance(function.value, ast.Name):
        return None
    module = modules.get(function.value.id, function.value.id)
    return f"{module}.{function.attr}"


def _constant_keyword(call: ast.Call, name: str) -> object | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _inside_function(call: ast.Call, function: ast.FunctionDef) -> bool:
    return (
        function.lineno <= call.lineno
        and function.end_lineno is not None
        and call.end_lineno is not None
        and call.end_lineno <= function.end_lineno
    )


@pytest.mark.unit
def test_repository_deserialization_boundaries_are_fail_closed() -> None:
    violations: list[str] = []
    for root_name in SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root_name).rglob("*.py")):
            relative_path = path.relative_to(REPO_ROOT)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            modules, functions = _import_aliases(tree)
            trusted_numpy_functions = tuple(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and relative_path == TRUSTED_NUMPY_PICKLE_OWNER
                and node.name == TRUSTED_NUMPY_PICKLE_FUNCTION
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                call_name = _qualified_call_name(
                    node,
                    modules=modules,
                    functions=functions,
                )
                if call_name == "torch.load" and _constant_keyword(
                    node, "weights_only"
                ) is not True:
                    violations.append(f"{relative_path}:{node.lineno}: torch.load")
                if call_name in {"pickle.load", "pickle.loads"}:
                    violations.append(f"{relative_path}:{node.lineno}: {call_name}")
                if call_name == "numpy.load":
                    inside_trusted_boundary = any(
                        _inside_function(node, function)
                        for function in trusted_numpy_functions
                    )
                    expected = True if inside_trusted_boundary else False
                    if _constant_keyword(node, "allow_pickle") is not expected:
                        violations.append(
                            f"{relative_path}:{node.lineno}: numpy.load must set "
                            f"allow_pickle={expected}"
                        )

    assert violations == []
