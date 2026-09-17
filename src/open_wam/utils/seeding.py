from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


def seed_everywhere(
    seed: int,
    *,
    deterministic: bool | None = None,
    warn_only: bool = False,
) -> int:
    """Seed Python, NumPy, and Torch from one place."""

    if seed < 0:
        raise ValueError(f"`seed` must be non-negative, got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic is not None:
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(
            deterministic, warn_only=warn_only if deterministic else False
        )
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = deterministic
            if deterministic:
                torch.backends.cudnn.benchmark = False

    return seed


@dataclass(frozen=True)
class RuntimeRngSnapshot:
    """Python, NumPy, and Torch RNG state captured as one restore point."""

    python_state: object
    numpy_state: tuple[str, np.ndarray, int, int, float]
    torch_cpu_state: torch.Tensor
    torch_cuda_state: list[torch.Tensor] | None


def snapshot_rng_state() -> RuntimeRngSnapshot:
    """Capture global RNG state used by a synchronous speculative branch."""

    return RuntimeRngSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_state=(
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    )


def restore_rng_state(snapshot: RuntimeRngSnapshot | None) -> None:
    """Restore a previously captured global RNG state."""

    if snapshot is None:
        return
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_cpu_state)
    if snapshot.torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_state)


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Run auxiliary work without advancing the caller's global RNG streams."""

    snapshot = snapshot_rng_state()
    try:
        yield
    finally:
        restore_rng_state(snapshot)
