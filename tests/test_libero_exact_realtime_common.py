from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_sandbox_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "libero_exact_realtime_common.py"
    spec = importlib.util.spec_from_file_location("libero_exact_realtime_common", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_output_stem_sanitizes_prompt_and_suffix(tmp_path: Path) -> None:
    sandbox = _load_sandbox_module()

    output_stem = sandbox._build_output_stem(
        root=tmp_path,
        benchmark_name="libero_10",
        task_id=1,
        prompt="put / both: things? in <basket>",
        episode_idx=7,
        suffix="step600/unsafe",
    )

    assert output_stem.parent.name == "1_put_both_things_in_basket"
    assert output_stem.name == "7_step600_unsafe"


def test_exact_realtime_common_rejects_frame_zero_startup_actions() -> None:
    sandbox = _load_sandbox_module()
    chunk = SimpleNamespace(
        raw_chunk_action_pred=sandbox.torch.zeros(1, 16, 1),
        debug={"generation_frame_start": 0},
        session=SimpleNamespace(policy_state=SimpleNamespace(step_index=0)),
    )

    with pytest.raises(ValueError, match="generation_frame_start < 1"):
        sandbox._chunk_to_planned_frames(
            first_chunk=chunk,
            frame_chunk_size=4,
            action_per_frame=4,
            source="startup_plan",
            ready_monotonic_s=0.0,
        )
