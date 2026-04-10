from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import uuid


def _load_sandbox_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_libero_realtime_sandbox.py"
    module_name = f"run_libero_realtime_sandbox_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
    return module


def test_frame_index_to_action_start_matches_exact_realtime_convention() -> None:
    sandbox = _load_sandbox_module()

    assert sandbox._frame_index_to_action_start(1, 4) == 0
    assert sandbox._frame_index_to_action_start(2, 4) == 4
    assert sandbox._frame_index_to_action_start(3, 4) == 8


def test_merge_future_step_actions_drops_stale_steps_and_prefers_newer_future() -> None:
    sandbox = _load_sandbox_module()
    step_cls = sandbox.PlannedControlStep

    existing = {
        0: step_cls(absolute_action_index=0, generation_action_start=0, source="old"),
        1: step_cls(absolute_action_index=1, generation_action_start=0, source="old"),
        2: step_cls(absolute_action_index=2, generation_action_start=0, source="old"),
    }
    incoming = [
        step_cls(absolute_action_index=1, generation_action_start=1, source="new"),
        step_cls(absolute_action_index=3, generation_action_start=1, source="new"),
    ]

    merged = sandbox._merge_future_step_actions(existing, incoming, next_action_to_execute=1)

    assert list(merged) == [1, 2, 3]
    assert merged[1].source == "new"
    assert merged[2].source == "old"
    assert merged[3].source == "new"
