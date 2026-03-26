from __future__ import annotations

import os
import random

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
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only if deterministic else False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = deterministic
            if deterministic:
                torch.backends.cudnn.benchmark = False

    return seed
