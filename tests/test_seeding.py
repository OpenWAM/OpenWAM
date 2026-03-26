from __future__ import annotations

import os
import random

import numpy as np
import torch

from open_wam.utils import seed_everywhere


def test_seed_everywhere_resets_python_numpy_and_torch_rngs() -> None:
    seed_everywhere(1234)
    first_python = random.random()
    first_numpy = float(np.random.rand())
    first_torch = torch.rand(4)

    seed_everywhere(1234)
    second_python = random.random()
    second_numpy = float(np.random.rand())
    second_torch = torch.rand(4)

    assert os.environ["PYTHONHASHSEED"] == "1234"
    assert first_python == second_python
    assert first_numpy == second_numpy
    assert torch.equal(first_torch, second_torch)


def test_seed_everywhere_can_enable_deterministic_torch_mode() -> None:
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    try:
        returned_seed = seed_everywhere(7, deterministic=True, warn_only=True)

        assert returned_seed == 7
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(original_deterministic, warn_only=original_warn_only)
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        if original_cublas_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_cublas_workspace
