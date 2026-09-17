"""Blocking LIBERO contracts and opt-in, source-pinned bitwise episode parity."""

import json
import os
from pathlib import Path

import pytest
import torch

from open_wam.runtime.control import RolloutTerminationReason
from open_wam.runtime.rollout_engine import RolloutEngine
from tests.characterization.capture_libero_episode import CASES, capture


@pytest.mark.parametrize(
    "limit,reason",
    [
        ("time", RolloutTerminationReason.MAX_ACTIONS),
        ("terminal", RolloutTerminationReason.ENV_TERMINAL),
        ("success", RolloutTerminationReason.SUCCESS),
    ],
)
def test_libero_driver_engine_termination_receipt(monkeypatch, limit, reason):
    results = []
    run = RolloutEngine.run

    def record_run(self, *args, **kwargs):
        result = run(self, *args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(RolloutEngine, "run", record_run)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        record = capture(monkeypatch, limit=limit)
    finally:
        torch.set_num_threads(threads)
    (result,) = results
    assert result.termination.reason is reason
    assert result.termination.control_count == len(record["actions"]) == 11
    assert result.termination.transition.done is (limit != "time")
    assert result.termination.transition.success is (limit == "success")
    # All three stops finish the artifact chunk, including its partial latent frame.
    assert record["summary"]["terminal"]
    assert record["events"][-1]["phase"] == "env_rollout"
    assert record["events"][-1]["executed_actions"] == 3
    assert result.lifecycle.observed_control_end == 8


@pytest.fixture(params=CASES)
def episode(request, monkeypatch):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield request.param, capture(monkeypatch, **CASES[request.param])
    finally:
        torch.set_num_threads(threads)


def test_libero_episode_contract(episode):
    name, record = episode
    case = CASES[name]
    summary = record["summary"]
    infer = [row for row in record["events"] if row["phase"] == "infer"]
    executed = [row for row in record["events"] if row["phase"] == "env_rollout"]
    assert record["closed"]
    assert len(infer) == len(executed) == summary["chunk_count"]
    assert sum(row["executed_actions"] for row in executed) == len(record["actions"])
    assert len(record["observations"]) == case.get("startup", 1) + len(
        record["actions"]
    )
    assert summary["action_count"] == len(record["actions"])
    assert summary["success"] is (case.get("limit") == "success")
    assert len(record["videos"]) == len(infer)
    for row in infer:
        assert row["execute_action_steps"] % 4 == 0
    if name in {"empty", "startup_time"}:
        assert not record["predictions"] and not record["frontend"]
    elif name == "startup_terminal":
        assert len(infer) == 1 and not record["actions"] and not record["commits"]
    elif summary["terminal"]:
        assert executed[-1]["executed_actions"] == 3
        assert record["events"][-1]["phase"] == "env_rollout"
        assert not any(
            row["phase"].endswith("warmup") and row["chunk_index"] == len(infer) - 1
            for row in record["events"]
        )
    else:
        assert summary["chunk_count"] == 3
        assert record["events"][-1]["phase"].endswith("warmup")
    if case.get("consumer") is not None:
        assert len(record["predictions"]) == 2 * len(infer)
        assert summary["pipeline"] == "open_wam_policy_video_action_composition"


@pytest.mark.skipif(
    not os.environ.get("OPEN_WAM_LIBERO_EPISODE_REFERENCE"),
    reason="Set OPEN_WAM_LIBERO_EPISODE_REFERENCE to a pre-migration capture on the same numerical stack.",
)
def test_libero_episode_bitwise_frozen_parity(episode):
    name, actual = episode
    expected = json.loads(
        Path(os.environ["OPEN_WAM_LIBERO_EPISODE_REFERENCE"]).read_text()
    )[name]
    assert set(actual) == set(expected)
    for surface in expected:
        # Tensor and image records contain shape/dtype plus byte-level SHA256.
        # No floating tolerance, action-only comparison, or regenerated oracle.
        assert actual[surface] == expected[surface], f"{name}: {surface}"
