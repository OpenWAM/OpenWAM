"""Deterministic source ordering for counterfactual dynamics views."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _balanced_counterfactual_source_indices(rows: Sequence[dict[str, Any]]) -> tuple[int, ...]:
    indices_by_task_branch: dict[str, dict[str, list[int]]] = {}
    branch_order: list[str] = []
    seen_branches: set[str] = set()
    for index, row in enumerate(rows):
        task_key = _counterfactual_source_task_key(row)
        branch_key = _counterfactual_source_branch_key(row)
        if branch_key not in seen_branches:
            seen_branches.add(branch_key)
            branch_order.append(branch_key)
        task_branches = indices_by_task_branch.setdefault(task_key, {})
        task_branches.setdefault(branch_key, []).append(int(index))
    if not indices_by_task_branch:
        return tuple(range(len(rows)))

    ordered_tasks = sorted(indices_by_task_branch, key=_source_view_label_sort_key)
    ordered_branches = tuple(branch_order) if branch_order else ("unknown",)
    max_depth = max(
        len(indices)
        for task_branches in indices_by_task_branch.values()
        for indices in task_branches.values()
    )
    order: list[int] = []
    for depth in range(max_depth):
        for branch_offset in range(len(ordered_branches)):
            for task_offset, task_key in enumerate(ordered_tasks):
                branch_key = ordered_branches[(task_offset + branch_offset) % len(ordered_branches)]
                indices = indices_by_task_branch.get(task_key, {}).get(branch_key, ())
                if depth < len(indices):
                    order.append(int(indices[depth]))
    if len(order) < len(rows):
        seen = set(order)
        order.extend(index for index in range(len(rows)) if index not in seen)
    return tuple(order)


def _counterfactual_source_task_key(row: dict[str, Any]) -> str:
    for key in ("task_id", "task_key", "task_name"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _counterfactual_source_branch_key(row: dict[str, Any]) -> str:
    for key in ("branch", "counterfactual_branch", "branch_family"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _source_view_label_sort_key(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, str(label))
