from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sandbox_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_libero_exact_realtime_sandbox.py"
    spec = importlib.util.spec_from_file_location("run_libero_exact_realtime_sandbox", module_path)
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
