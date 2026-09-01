from __future__ import annotations

import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from open_wam.configs import (
    EXPERIMENT_CONFIG_SCHEMA_VERSION,
    ExperimentConfig,
    serialize_experiment_config,
)
from open_wam.configs.enums import serialize_enum_values

__all__ = []


def _serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema_version": EXPERIMENT_CONFIG_SCHEMA_VERSION,
        **serialize_experiment_config(config),
    }


def _serialize_runtime_backbone_config(backbone_config: object) -> dict[str, Any]:
    if is_dataclass(backbone_config):
        return serialize_enum_values(asdict(backbone_config))
    if isinstance(backbone_config, dict):
        return serialize_enum_values(dict(backbone_config))
    return {"repr": repr(backbone_config)}


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _wait_for_file(
    path: Path,
    *,
    timeout_seconds: float,
    poll_seconds: float = 2.0,
    error_marker: Path | None = None,
) -> None:
    """Block until ``path`` appears, a peer reports failure, or time expires."""

    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        # Failure wins if rank zero published success and then failed during a
        # later checkpoint stage before this rank observed either marker.
        if error_marker is not None and error_marker.exists():
            raise RuntimeError(
                f"Peer signalled checkpoint failure via {error_marker}; abort wait for {path}"
            )
        if path.exists():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for file: {path}")
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
