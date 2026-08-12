from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from open_wam.artifacts import (
    UnsafeArtifactError,
    load_numpy_compatible_torch_artifact,
    load_tensor_artifact,
)


class _WriteSentinelOnLoad:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __reduce__(self):
        return os.system, (f"touch {self.path}",)


def test_safe_tensor_loader_preserves_tensor_containers(tmp_path: Path) -> None:
    path = tmp_path / "payload.pt"
    expected = {
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "metadata": {"frame_ids": [0, 4], "name": "fixture"},
    }
    torch.save(expected, path)

    actual = load_tensor_artifact(path)

    assert torch.equal(actual["tensor"], expected["tensor"])
    assert actual["metadata"] == expected["metadata"]


def test_safe_tensor_loader_never_retries_with_unrestricted_pickle(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "untrusted.pt"
    sentinel_path = tmp_path / "executed"
    torch.save(_WriteSentinelOnLoad(sentinel_path), artifact_path)

    with pytest.raises(UnsafeArtifactError, match="will not retry"):
        load_tensor_artifact(artifact_path)

    assert not sentinel_path.exists()


def test_numpy_compatible_loader_admits_only_restricted_numpy_payloads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "libero_init.pt"
    expected = [np.arange(6, dtype=np.float32).reshape(2, 3)]
    torch.save(expected, path)

    actual = load_numpy_compatible_torch_artifact(path)

    assert isinstance(actual, list)
    assert np.array_equal(actual[0], expected[0])


def test_numpy_compatible_loader_accepts_numpy_1_module_path(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "numpy_2.pt"
    legacy_path = tmp_path / "numpy_1.pt"
    expected = np.arange(6, dtype=np.float64).reshape(2, 3)
    torch.save(expected, current_path)

    with zipfile.ZipFile(current_path, "r") as source, zipfile.ZipFile(
        legacy_path,
        "w",
    ) as destination:
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename.endswith("data.pkl"):
                payload = payload.replace(
                    b"numpy._core.multiarray\n",
                    b"numpy.core.multiarray\n",
                )
            destination.writestr(member, payload)

    actual = load_numpy_compatible_torch_artifact(legacy_path)

    assert np.array_equal(actual, expected)
