"""Regression coverage for runtime parameter materialisation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED = (
    REPO_ROOT / "src/open_wam/models/visual_tower/shared_transformer_support.py",
    REPO_ROOT / "src/open_wam/models/visual_tower/replica_core.py",
)
GUARDED_ATTRIBUTES = ("scale_shift_table",)
MATERIALISERS = ("materialize_runtime_parameter", "_materialize_runtime_parameter")


def _bare_parameter_reads(tree: ast.AST) -> list[tuple[int, str]]:
    """`self.<guarded>[...]` not wrapped in a materialiser call."""

    wrapped: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name not in MATERIALISERS:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Attribute) and inner.attr in GUARDED_ATTRIBUTES:
                wrapped.add(id(inner))

    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not isinstance(value, ast.Attribute) or value.attr not in GUARDED_ATTRIBUTES:
            continue
        if not (isinstance(value.value, ast.Name) and value.value.id == "self"):
            continue
        if id(value) in wrapped:
            continue
        offenders.append((node.lineno, value.attr))
    return offenders


@pytest.mark.unit
def test_guarded_parameters_are_materialised_before_use() -> None:
    offenders: list[str] = []
    for path in SCANNED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, attr in _bare_parameter_reads(tree):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: self.{attr}[...]")
    assert offenders == [], (
        "these read a parameter directly inside a forward path; wrap them in "
        "materialize_runtime_parameter(device=<activation>.device, "
        "dtype=<parameter>.dtype) so an FSDP shard or a CPU-resident parameter "
        "cannot reach the arithmetic:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_the_check_can_fail() -> None:
    """A bare read is actually detected -- otherwise the test above is decorative."""

    tree = ast.parse(
        "class B:\n"
        "    def forward(self, temb):\n"
        "        return self.scale_shift_table[None] + temb\n"
    )
    assert _bare_parameter_reads(tree) == [(3, "scale_shift_table")]


@pytest.mark.unit
def test_a_materialised_read_is_accepted() -> None:
    tree = ast.parse(
        "class B:\n"
        "    def forward(self, temb):\n"
        "        return _materialize_runtime_parameter(\n"
        "            self.scale_shift_table, device=temb.device,\n"
        "            dtype=self.scale_shift_table.dtype,\n"
        "        )[None] + temb\n"
    )
    assert _bare_parameter_reads(tree) == []
