"""Dependency-light verification for policy-video/action rollout goldens."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActionTraceSummary:
    count: int
    dimension: int
    sha256: str


def load_rollout_golden(path: Path) -> dict[str, Any]:
    """Load and minimally validate a rollout golden document."""

    payload = _load_json_object(path)
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported rollout golden schema in {path}: "
            f"{payload.get('schema_version')!r}."
        )
    scopes = payload.get("scopes")
    if not isinstance(scopes, Mapping) or not scopes:
        raise ValueError(f"Rollout golden {path} must define non-empty scopes.")
    return payload


def summarize_action_trace(path: Path) -> ActionTraceSummary:
    """Validate a JSONL action trace and return its exact byte fingerprint."""

    digest = hashlib.sha256()
    count = 0
    dimension: int | None = None
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid action JSONL at {path}:{count + 1}: {exc}."
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"Action row {count} in {path} is not an object.")
            if row.get("action_index") != count:
                raise ValueError(
                    f"Action row {count} in {path} has non-contiguous index "
                    f"{row.get('action_index')!r}."
                )
            action = row.get("action")
            if not isinstance(action, list) or not action:
                raise ValueError(f"Action row {count} in {path} has no action vector.")
            if dimension is None:
                dimension = len(action)
            elif len(action) != dimension:
                raise ValueError(
                    f"Action row {count} in {path} has dimension {len(action)}, "
                    f"expected {dimension}."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in action
            ):
                raise ValueError(
                    f"Action row {count} in {path} contains a non-finite scalar."
                )
            count += 1
    if count == 0 or dimension is None:
        raise ValueError(f"Action trace {path} is empty.")
    return ActionTraceSummary(
        count=count,
        dimension=dimension,
        sha256=digest.hexdigest(),
    )


def verify_rollout_artifacts(
    *,
    golden_path: Path,
    scope: str,
    report_path: Path,
    action_trace_path: Path,
    load_report_path: Path | None = None,
) -> None:
    """Verify one generated rollout against a named immutable golden scope."""

    golden = load_rollout_golden(golden_path)
    scopes = golden["scopes"]
    if scope not in scopes:
        raise ValueError(
            f"Unknown rollout golden scope {scope!r}; expected one of "
            f"{sorted(scopes)}."
        )
    expected = scopes[scope]
    if not isinstance(expected, Mapping):
        raise ValueError(f"Golden scope {scope!r} must be an object.")

    report = _load_json_object(report_path)
    _assert_expected_subset(
        actual=report,
        expected=_required_mapping(expected, "report"),
        path=f"{scope}.report",
    )
    _verify_trace_summary(
        summarize_action_trace(action_trace_path),
        _required_mapping(expected, "action_trace"),
        path=f"{scope}.action_trace",
    )

    expected_load_report = golden.get("load_report")
    if expected_load_report is not None:
        if load_report_path is None:
            raise AssertionError(
                "This golden freezes runtime-load semantics; pass load_report_path."
            )
        _assert_expected_subset(
            actual=_load_json_object(load_report_path),
            expected=_required_mapping(golden, "load_report"),
            path="load_report",
        )


def verify_checked_in_trace_fixture(golden_path: Path) -> None:
    """Verify that a checked-in exact trace still matches its golden metadata."""

    golden = load_rollout_golden(golden_path)
    for scope_name, raw_scope in golden["scopes"].items():
        scope = _required_mapping(golden["scopes"], str(scope_name))
        expected_trace = _required_mapping(scope, "action_trace")
        fixture = expected_trace.get("fixture")
        if fixture is None:
            continue
        if not isinstance(fixture, str) or not fixture:
            raise ValueError(f"{scope_name}.action_trace.fixture must be a path.")
        fixture_path = (golden_path.parent / fixture).resolve()
        if golden_path.parent.resolve() not in fixture_path.parents:
            raise ValueError(
                f"Trace fixture for {scope_name!r} escapes the golden directory."
            )
        _verify_trace_summary(
            summarize_action_trace(fixture_path),
            expected_trace,
            path=f"{scope_name}.action_trace",
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected {key!r} to be an object.")
    return value


def _verify_trace_summary(
    actual: ActionTraceSummary,
    expected: Mapping[str, Any],
    *,
    path: str,
) -> None:
    expected_summary = {
        "count": expected.get("count"),
        "dimension": expected.get("dimension"),
        "sha256": expected.get("sha256"),
    }
    actual_summary = {
        "count": actual.count,
        "dimension": actual.dimension,
        "sha256": actual.sha256,
    }
    _assert_expected_subset(
        actual=actual_summary,
        expected=expected_summary,
        path=path,
    )


def _assert_expected_subset(
    *,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    path: str,
) -> None:
    for key, expected_value in expected.items():
        child_path = f"{path}.{key}"
        if key not in actual:
            raise AssertionError(f"Missing golden field {child_path}.")
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                raise AssertionError(
                    f"Golden field {child_path} expected an object, got "
                    f"{type(actual_value).__name__}."
                )
            _assert_expected_subset(
                actual=actual_value,
                expected=expected_value,
                path=child_path,
            )
        elif actual_value != expected_value:
            raise AssertionError(
                f"Golden mismatch at {child_path}: expected "
                f"{expected_value!r}, got {actual_value!r}."
            )
