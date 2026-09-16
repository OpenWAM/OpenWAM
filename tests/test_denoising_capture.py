import json

import pytest
import torch
from safetensors.torch import save_file

from tests.characterization.capture_denoising_execution import verify


@pytest.mark.parametrize("mutation", [None, "value", "dtype", "keys", "cursor", "file"])
def test_capture_verifier_checks_tensors_and_recurrent_metadata(tmp_path, mutation):
    reference, actual = tmp_path / "reference", tmp_path / "actual"
    for root in (reference, actual):
        root.mkdir()
        save_file({"gradient": torch.ones(2)}, root / "case.safetensors")
        (root / "case.json").write_text(json.dumps({"next_frame": 4}))
    if mutation in {"value", "dtype", "keys"}:
        value = torch.ones(2)
        if mutation == "value":
            value[0] += 1e-6
        if mutation == "dtype":
            value = value.double()
        save_file(
            {"missing" if mutation == "keys" else "gradient": value},
            actual / "case.safetensors",
        )
    elif mutation == "cursor":
        (actual / "case.json").write_text(json.dumps({"next_frame": 5}))
    elif mutation == "file":
        (actual / "case.json").unlink()
    if mutation is None:
        assert verify(reference, actual) == 1
    else:
        with pytest.raises(AssertionError):
            verify(reference, actual)
