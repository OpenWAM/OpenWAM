from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    (
        "run_libero_exact_realtime_sandbox.py",
        "run_libero_exact_visualization.py",
        "run_libero_mot_visualization.py",
        "run_libero_mot_batch_visualization.py",
        "run_libero_dual_expert_visualization.py",
        "run_libero_dual_expert_batch_visualization.py",
    ),
)
def test_architecture_named_inference_entrypoints_have_no_alias(name):
    assert not (Path(__file__).resolve().parents[1] / "scripts" / name).exists()
