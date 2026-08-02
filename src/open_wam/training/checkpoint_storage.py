from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from open_wam.configs import ExperimentConfig
from open_wam.configs.enums import serialize_enum_values

__all__ = []


def _serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    if is_dataclass(config):
        return serialize_enum_values(asdict(config))
    raise TypeError(f"Expected dataclass config, got {type(config).__name__}.")


def _serialize_runtime_backbone_config(backbone_config: object) -> dict[str, Any]:
    if is_dataclass(backbone_config):
        return serialize_enum_values(asdict(backbone_config))
    if isinstance(backbone_config, dict):
        return serialize_enum_values(dict(backbone_config))
    return {"repr": repr(backbone_config)}


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _wait_for_file(
    path: Path, *, timeout_seconds: float = 7200.0, poll_seconds: float = 2.0
) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for checkpoint completion marker: {path}"
            )
        time.sleep(float(poll_seconds))


def _atomic_torch_save(payload: object, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_sibling_train_state(checkpoint_path: Path) -> dict[str, Any] | None:
    """Recover step metadata for lightweight model-only warm starts when available."""

    if checkpoint_path.name != "model_state.pt":
        return None
    train_state_path = checkpoint_path.parent / "train_state.json"
    if not train_state_path.is_file():
        return None
    with train_state_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected object in {train_state_path}, got {type(raw).__name__}."
        )
    # A model-only checkpoint has no optimizer/scheduler/sampler state. Preserve
    # the step counters for logging and max-step continuation, but do not skip
    # batches as if this were an exact full-training-state resume.
    raw["epoch_index"] = 0
    raw["seen_batches"] = 0
    return raw
