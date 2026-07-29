from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from open_wam.data import LatentWAMBatch

FIXTURE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1

_LATENT_BATCH_TENSOR_FIELDS = tuple(
    field.name
    for field in fields(LatentWAMBatch)
    if field.name not in {"task_text", "metadata"}
)


@dataclass(frozen=True)
class RolloutInputFixture:
    views: dict[str, torch.Tensor]
    state: torch.Tensor
    text_context: torch.Tensor
    negative_text_context: torch.Tensor
    action_conditioning: torch.Tensor
    video_conditioning: torch.Tensor
    task_text: tuple[str | None, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ComparisonTolerance:
    absolute: float = 5e-4
    relative: float = 5e-3


DEFAULT_COMPARISON_TOLERANCE = ComparisonTolerance()
EXACT_COMPARISON_TOLERANCE = ComparisonTolerance(absolute=0.0, relative=0.0)
ToleranceResolver = Callable[[str], ComparisonTolerance | None]


def save_latent_batch_fixture(
    root: str | Path,
    *,
    fixture_id: str,
    batch: LatentWAMBatch,
    provenance: Mapping[str, Any],
) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    tensor_path = root_path / f"{fixture_id}.safetensors"
    metadata_path = root_path / f"{fixture_id}.json"
    tensors = {
        field_name: value.detach().cpu().contiguous()
        for field_name in _LATENT_BATCH_TENSOR_FIELDS
        if isinstance((value := getattr(batch, field_name)), torch.Tensor)
    }
    save_file(tensors, tensor_path)
    payload = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_type": "latent_wam_batch",
        "fixture_id": fixture_id,
        "tensor_file": tensor_path.name,
        "tensor_sha256": sha256_file(tensor_path),
        "task_text": _jsonable(batch.task_text),
        "metadata": _jsonable(batch.metadata),
        "provenance": _jsonable(dict(provenance)),
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in sorted(tensors.items())
        },
    }
    write_json_atomic(metadata_path, payload)
    return metadata_path


def load_latent_batch_fixture(path: str | Path) -> LatentWAMBatch:
    metadata_path = Path(path)
    payload = _load_fixture_metadata(metadata_path, expected_type="latent_wam_batch")
    tensor_path = metadata_path.parent / str(payload["tensor_file"])
    _verify_file_hash(tensor_path, str(payload["tensor_sha256"]))
    tensors = load_file(tensor_path, device="cpu")
    kwargs: dict[str, Any] = {
        field_name: tensors.get(field_name)
        for field_name in _LATENT_BATCH_TENSOR_FIELDS
    }
    task_text = payload.get("task_text")
    kwargs["task_text"] = None if task_text is None else tuple(task_text)
    metadata = payload.get("metadata") or []
    kwargs["metadata"] = tuple(dict(item) for item in metadata)
    return LatentWAMBatch(**kwargs)


def save_rollout_input_fixture(
    root: str | Path,
    *,
    fixture_id: str,
    fixture: RolloutInputFixture,
    provenance: Mapping[str, Any],
) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    tensor_path = root_path / f"{fixture_id}.safetensors"
    metadata_path = root_path / f"{fixture_id}.json"
    tensors = {
        "state": fixture.state.detach().cpu().contiguous(),
        "text_context": fixture.text_context.detach().cpu().contiguous(),
        "negative_text_context": fixture.negative_text_context.detach()
        .cpu()
        .contiguous(),
        "action_conditioning": fixture.action_conditioning.detach().cpu().contiguous(),
        "video_conditioning": fixture.video_conditioning.detach().cpu().contiguous(),
    }
    view_key_map: dict[str, str] = {}
    for index, (view_name, value) in enumerate(sorted(fixture.views.items())):
        tensor_key = f"view_{index}"
        view_key_map[tensor_key] = view_name
        tensors[tensor_key] = value.detach().cpu().contiguous()
    save_file(tensors, tensor_path)
    payload = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_type": "mot_rollout_input",
        "fixture_id": fixture_id,
        "tensor_file": tensor_path.name,
        "tensor_sha256": sha256_file(tensor_path),
        "view_key_map": view_key_map,
        "task_text": _jsonable(fixture.task_text),
        "metadata": _jsonable(fixture.metadata),
        "provenance": _jsonable(dict(provenance)),
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in sorted(tensors.items())
        },
    }
    write_json_atomic(metadata_path, payload)
    return metadata_path


def load_rollout_input_fixture(path: str | Path) -> RolloutInputFixture:
    metadata_path = Path(path)
    payload = _load_fixture_metadata(metadata_path, expected_type="mot_rollout_input")
    tensor_path = metadata_path.parent / str(payload["tensor_file"])
    _verify_file_hash(tensor_path, str(payload["tensor_sha256"]))
    tensors = load_file(tensor_path, device="cpu")
    view_key_map = payload.get("view_key_map")
    if not isinstance(view_key_map, Mapping):
        raise TypeError(f"Invalid view_key_map in {metadata_path}.")
    return RolloutInputFixture(
        views={
            str(view_name): tensors[str(tensor_key)]
            for tensor_key, view_name in view_key_map.items()
        },
        state=tensors["state"],
        text_context=tensors["text_context"],
        negative_text_context=tensors["negative_text_context"],
        action_conditioning=tensors["action_conditioning"],
        video_conditioning=tensors["video_conditioning"],
        task_text=tuple(payload.get("task_text") or ()),
        metadata=dict(payload.get("metadata") or {}),
    )


def tensor_fingerprint(
    tensor: torch.Tensor, *, probe_count: int = 24
) -> dict[str, Any]:
    detached = tensor.detach()
    local = _local_tensor(detached)
    flat = local.to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    count = int(flat.numel())
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    finite = flat[finite_mask]
    if finite.numel() > 0:
        summary = {
            "mean": float(finite.mean().item()),
            "std": float(finite.std(unbiased=False).item()),
            "l1": float(finite.abs().sum().item()),
            "l2": float(torch.linalg.vector_norm(finite).item()),
            "min": float(finite.min().item()),
            "max": float(finite.max().item()),
        }
    else:
        summary = {
            "mean": None,
            "std": None,
            "l1": None,
            "l2": None,
            "min": None,
            "max": None,
        }
    probe_indices = _probe_indices(count, probe_count=probe_count)
    return {
        "shape": list(detached.shape),
        "local_shape": list(local.shape),
        "dtype": str(detached.dtype),
        "numel": count,
        "finite_count": finite_count,
        "content_sha256": _tensor_content_sha256(
            flat,
            dtype=str(detached.dtype),
            shape=tuple(detached.shape),
            local_shape=tuple(local.shape),
        ),
        **summary,
        "probe_indices": probe_indices,
        "probe_values": [float(flat[index].item()) for index in probe_indices],
    }


def collect_tensor_fingerprints(
    value: Any,
    *,
    root: str,
    max_tensors: int = 128,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def visit(current: Any, path: str) -> None:
        if len(records) >= max_tensors:
            return
        if isinstance(current, torch.Tensor):
            records[path] = tensor_fingerprint(current)
            return
        if is_dataclass(current):
            for field in fields(current):
                visit(getattr(current, field.name), f"{path}.{field.name}")
            return
        if isinstance(current, Mapping):
            for key in sorted(current, key=str):
                visit(current[key], f"{path}.{key}")
            return
        if isinstance(current, (tuple, list)):
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")

    visit(value, root)
    return records


def tensor_tree_schema(
    value: Any,
    *,
    root: str,
    max_entries: int = 256,
) -> dict[str, Any]:
    """Describe a nested runtime artifact without retaining or reducing tensors."""

    records: dict[str, Any] = {}

    def visit(current: Any, path: str) -> None:
        if len(records) >= max_entries:
            return
        if isinstance(current, torch.Tensor):
            local = _local_tensor(current)
            records[path] = {
                "kind": "tensor",
                "shape": list(current.shape),
                "local_shape": list(local.shape),
                "dtype": str(current.dtype),
            }
            return
        if is_dataclass(current):
            records[path] = {"kind": "dataclass", "type": type(current).__name__}
            for field in fields(current):
                visit(getattr(current, field.name), f"{path}.{field.name}")
            return
        if isinstance(current, Mapping):
            records[path] = {"kind": "mapping", "size": len(current)}
            for key in sorted(current, key=str):
                visit(current[key], f"{path}.{key}")
            return
        if isinstance(current, (tuple, list)):
            records[path] = {
                "kind": "sequence",
                "type": type(current).__name__,
                "size": len(current),
            }
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")
            return
        if current is None or isinstance(current, (str, int, float, bool)):
            records[path] = {"kind": type(current).__name__, "value": current}
            return
        records[path] = {"kind": "object", "type": type(current).__name__}

    visit(value, root)
    return records


def scalar_tensor_values(values: Mapping[str, Any]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for name, value in sorted(values.items()):
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            resolved[str(name)] = float(
                _local_tensor(value.detach()).float().cpu().item()
            )
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            resolved[str(name)] = float(value)
    return resolved


def compare_characterization_reports(
    expected: Any,
    actual: Any,
    *,
    tolerance: ComparisonTolerance = DEFAULT_COMPARISON_TOLERANCE,
    tolerance_for_path: ToleranceResolver | None = None,
    path: str = "report",
) -> list[str]:
    differences: list[str] = []
    _compare_values(
        expected,
        actual,
        tolerance=tolerance,
        tolerance_for_path=tolerance_for_path,
        path=path,
        differences=differences,
    )
    return differences


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _jsonable(dict(payload)),
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_fixture_metadata(path: Path, *, expected_type: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", 0)) != FIXTURE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported fixture schema in {path}.")
    if payload.get("fixture_type") != expected_type:
        raise ValueError(
            f"Expected fixture_type={expected_type!r} in {path}, "
            f"got {payload.get('fixture_type')!r}."
        )
    return payload


def _verify_file_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Fixture hash mismatch for {path}: expected {expected}, got {actual}."
        )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return tensor
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _probe_indices(count: int, *, probe_count: int) -> list[int]:
    if count <= 0 or probe_count <= 0:
        return []
    if probe_count == 1:
        return [0]
    if count <= probe_count:
        return list(range(count))
    return sorted(
        {
            round(index * (count - 1) / float(probe_count - 1))
            for index in range(probe_count)
        }
    )


def _tensor_content_sha256(
    flat: torch.Tensor,
    *,
    dtype: str,
    shape: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    digest.update(json.dumps(local_shape, separators=(",", ":")).encode("ascii"))
    digest.update(flat.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _compare_values(
    expected: Any,
    actual: Any,
    *,
    tolerance: ComparisonTolerance,
    tolerance_for_path: ToleranceResolver | None,
    path: str,
    differences: list[str],
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            differences.append(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, int) and isinstance(actual, int):
        if expected != actual:
            differences.append(f"{path}: expected {expected}, got {actual}")
        return
    if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        expected_float = float(expected)
        actual_float = float(actual)
        path_tolerance = (
            None if tolerance_for_path is None else tolerance_for_path(path)
        )
        resolved_tolerance = tolerance if path_tolerance is None else path_tolerance
        if math.isnan(expected_float) and math.isnan(actual_float):
            return
        if not math.isclose(
            expected_float,
            actual_float,
            rel_tol=resolved_tolerance.relative,
            abs_tol=resolved_tolerance.absolute,
        ):
            differences.append(
                f"{path}: expected {expected_float:.9g}, got {actual_float:.9g}"
            )
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys, key=str):
            differences.append(f"{path}: missing key {key!r}")
        for key in sorted(actual_keys - expected_keys, key=str):
            differences.append(f"{path}: unexpected key {key!r}")
        for key in sorted(expected_keys & actual_keys, key=str):
            _compare_values(
                expected[key],
                actual[key],
                tolerance=tolerance,
                tolerance_for_path=tolerance_for_path,
                path=f"{path}.{key}",
                differences=differences,
            )
        return
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            differences.append(
                f"{path}: expected sequence length {len(expected)}, got {len(actual)}"
            )
            return
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _compare_values(
                expected_item,
                actual_item,
                tolerance=tolerance,
                tolerance_for_path=tolerance_for_path,
                path=f"{path}[{index}]",
                differences=differences,
            )
        return
    if expected != actual:
        differences.append(f"{path}: expected {expected!r}, got {actual!r}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return repr(value)
