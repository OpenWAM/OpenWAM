from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.evals import libero_realtime_runtime


def test_realtime_runtime_rejects_frame_zero_startup_actions() -> None:
    chunk = SimpleNamespace(
        raw_chunk_action_pred=torch.zeros(1, 16, 1),
        debug={"generation_frame_start": 0},
        session=SimpleNamespace(policy_state=SimpleNamespace(step_index=0)),
    )

    with pytest.raises(ValueError, match="generation_frame_start < 1"):
        libero_realtime_runtime._chunk_to_planned_frames(
            first_chunk=chunk,
            frame_chunk_size=4,
            action_per_frame=4,
            source="startup_plan",
            ready_monotonic_s=0.0,
        )
