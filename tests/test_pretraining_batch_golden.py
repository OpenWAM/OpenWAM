from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from tests.characterization.pretraining_batch import capture


@pytest.mark.parametrize("mode", ["padded", "bucket", "packed"])
@pytest.mark.parametrize("multiple", [1, 4])
def test_video_pretraining_batch_matches_preintegration_golden(mode, multiple):
    root = Path(__file__).parent / "fixtures/pretraining"
    expected = load_file(root / f"{mode}_pad{multiple}.safetensors")
    actual = capture(mode, pad_to_multiple_of=multiple)
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        # Only computed floating outputs allow cross-CPU kernel rounding.
        # Inputs, targets, masks, timesteps, and RNG remain exact.
        computed = value.is_floating_point() and (
            name.startswith(("grad.", "updated.", "output.", "metric."))
            or name in ("loss", "artifact.predicted_latents")
        )
        torch.testing.assert_close(
            value,
            expected[name],
            rtol=1e-5 if computed else 0,
            atol=2e-6 if computed else 0,
            msg=name,
        )
