from __future__ import annotations

import random
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_wam.evals import realtime_speculation
from open_wam.models.visual_tower import VisualRuntimeStateSnapshot


class _RecordingVisualTower:
    def __init__(self) -> None:
        self.snapshot_cache_names: list[str | None] = []
        self.restored: list[VisualRuntimeStateSnapshot | None] = []
        self.snapshot = VisualRuntimeStateSnapshot(frontend_state=("frontend",))

    def snapshot_runtime_state(
        self,
        *,
        cache_name: str | None = None,
    ) -> VisualRuntimeStateSnapshot:
        self.snapshot_cache_names.append(cache_name)
        return self.snapshot

    def restore_runtime_state(
        self,
        snapshot: VisualRuntimeStateSnapshot | None,
    ) -> None:
        self.restored.append(snapshot)


def _runner_with_tower(tower: _RecordingVisualTower) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(visual_tower=tower),
    )


def _split_cache_config() -> SimpleNamespace:
    return SimpleNamespace(
        policy_variant=SimpleNamespace(
            name="dual_expert",
            program="video_then_action",
        )
    )


def test_visual_runtime_cache_name_prefers_session_owned_name() -> None:
    session = SimpleNamespace(
        policy_state=SimpleNamespace(cache={"cache_name": "session_cache"}),
    )

    assert realtime_speculation.visual_runtime_cache_name_for_session(
        config=SimpleNamespace(),
        session=session,
    ) == "session_cache"


def test_visual_runtime_cache_name_uses_split_cache_fallback() -> None:
    assert (
        realtime_speculation.visual_runtime_cache_name_for_session(
            config=_split_cache_config(),
            session=SimpleNamespace(policy_state=None),
        )
        == "dual_expert_split_cache"
    )


def test_visual_runtime_snapshot_delegates_through_tower_contract() -> None:
    tower = _RecordingVisualTower()
    runner = _runner_with_tower(tower)
    session = SimpleNamespace(
        policy_state=SimpleNamespace(cache={"cache_name": "planner"}),
    )

    snapshot = realtime_speculation.snapshot_visual_runtime(
        runner=runner,
        config=SimpleNamespace(),
        session=session,
    )
    realtime_speculation.restore_visual_runtime(
        runner=runner,
        snapshot=snapshot,
    )

    assert tower.snapshot_cache_names == ["planner"]
    assert tower.restored == [tower.snapshot]


def test_rejected_result_restores_visual_state_but_accepted_result_keeps_it() -> None:
    tower = _RecordingVisualTower()
    runner = _runner_with_tower(tower)

    realtime_speculation.restore_visual_runtime_if_rejected(
        {"trace": {"accepted_chunk": True}},
        runner=runner,
        snapshot=tower.snapshot,
    )
    realtime_speculation.restore_visual_runtime_if_rejected(
        {"trace": {"accepted_chunk": False}},
        runner=runner,
        snapshot=tower.snapshot,
    )

    assert tower.restored == [tower.snapshot]


def test_missing_visual_snapshot_is_a_noop_without_runner_access() -> None:
    realtime_speculation.restore_visual_runtime(
        runner=None,
        snapshot=None,
    )


def test_future_exception_restores_visual_state() -> None:
    tower = _RecordingVisualTower()
    runner = _runner_with_tower(tower)
    future: Future[dict[str, object]] = Future()
    future.set_exception(RuntimeError("planner failed"))

    with pytest.raises(RuntimeError, match="planner failed"):
        realtime_speculation.resolve_future_result(
            future,
            runner=runner,
            snapshot=tower.snapshot,
        )

    assert tower.restored == [tower.snapshot]


def test_stateful_sequence_route_does_not_snapshot_global_visual_state() -> None:
    tower = _RecordingVisualTower()

    snapshot = realtime_speculation.snapshot_sequence_visual_runtime(
        runner=_runner_with_tower(tower),
        config=_split_cache_config(),
        session=SimpleNamespace(policy_state=None),
    )

    assert snapshot is None
    assert tower.snapshot_cache_names == []


def test_session_clone_isolated_and_reference_policy_explicit() -> None:
    session = SimpleNamespace(values=[1])

    cloned = realtime_speculation.clone_session(session)
    cloned.values.append(2)

    assert session.values == [1]
    assert realtime_speculation.session_reference(session, share_session=True) is session
    assert realtime_speculation.session_reference(session, share_session=False) is not session


def test_rng_snapshot_replays_python_numpy_and_torch_draws() -> None:
    outer_snapshot = realtime_speculation.snapshot_rng_state()
    try:
        random.seed(41)
        np.random.seed(42)
        torch.manual_seed(43)
        snapshot = realtime_speculation.snapshot_rng_state()
        expected = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )

        realtime_speculation.restore_rng_state(snapshot)
        actual = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )

        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])
    finally:
        realtime_speculation.restore_rng_state(outer_snapshot)


def test_preserve_rng_state_isolates_auxiliary_draws() -> None:
    outer_snapshot = realtime_speculation.snapshot_rng_state()
    try:
        random.seed(51)
        np.random.seed(52)
        torch.manual_seed(53)
        expected_snapshot = realtime_speculation.snapshot_rng_state()

        with realtime_speculation.preserve_rng_state():
            random.random()
            np.random.rand()
            torch.rand(3)

        expected = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )
        realtime_speculation.restore_rng_state(expected_snapshot)
        actual = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
        )

        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])
    finally:
        realtime_speculation.restore_rng_state(outer_snapshot)
