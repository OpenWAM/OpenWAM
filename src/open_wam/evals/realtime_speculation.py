"""State isolation contracts for speculative realtime planning."""

from __future__ import annotations

import copy
from concurrent.futures import Future
from dataclasses import dataclass
import random
from typing import TYPE_CHECKING, Any, Mapping, Protocol, TypeVar

import numpy as np
import torch

from open_wam.models.policy_variants.mot.runtime_routing import (
    resolve_mot_runtime_route,
)
from open_wam.models.visual_tower import VisualRuntimeStateSnapshot

if TYPE_CHECKING:
    from open_wam.configs import ExperimentConfig


__all__ = [
    "RuntimeRngSnapshot",
    "clone_session",
    "resolve_future_result",
    "restore_rng_state",
    "restore_visual_runtime",
    "restore_visual_runtime_if_rejected",
    "session_reference",
    "snapshot_rng_state",
    "snapshot_sequence_visual_runtime",
    "snapshot_visual_runtime",
    "visual_runtime_cache_name_for_session",
]


class _VisualRuntimeTower(Protocol):
    def snapshot_runtime_state(
        self,
        *,
        cache_name: str | None = None,
    ) -> VisualRuntimeStateSnapshot | None:
        ...

    def restore_runtime_state(
        self,
        snapshot: VisualRuntimeStateSnapshot | None,
    ) -> None:
        ...


class _VisualRuntimePipeline(Protocol):
    visual_tower: _VisualRuntimeTower


class _VisualRuntimeRunner(Protocol):
    pipeline: _VisualRuntimePipeline


class _PlannerResultLike(Protocol):
    trace: Mapping[str, Any]


@dataclass(frozen=True)
class RuntimeRngSnapshot:
    """Python, NumPy, and Torch RNG state captured as one restore point."""

    python_state: object
    numpy_state: tuple[str, np.ndarray, int, int, float]
    torch_cpu_state: torch.Tensor
    torch_cuda_state: list[torch.Tensor] | None


SessionT = TypeVar("SessionT")
PlannerResultT = TypeVar("PlannerResultT")


def visual_runtime_cache_name_for_session(
    *,
    config: ExperimentConfig,
    session: object,
) -> str | None:
    """Resolve the named visual cache represented by one policy session."""

    policy_state = getattr(session, "policy_state", None)
    cache = getattr(policy_state, "cache", None)
    if isinstance(cache, dict):
        cache_name = cache.get("cache_name")
        if cache_name is not None:
            return str(cache_name)
    if resolve_mot_runtime_route(config).uses_split_cache_rollout:
        return "mot_non_joint_two_stream_cache"
    return None


def snapshot_visual_runtime(
    *,
    runner: _VisualRuntimeRunner,
    config: ExperimentConfig,
    session: object,
) -> VisualRuntimeStateSnapshot | None:
    """Snapshot visual runtime state without exposing tower internals."""

    cache_name = visual_runtime_cache_name_for_session(
        config=config,
        session=session,
    )
    return runner.pipeline.visual_tower.snapshot_runtime_state(
        cache_name=cache_name,
    )


def restore_visual_runtime(
    *,
    runner: _VisualRuntimeRunner,
    snapshot: VisualRuntimeStateSnapshot | None,
) -> None:
    """Restore a visual runtime snapshot through its owning tower."""

    if snapshot is None:
        return
    runner.pipeline.visual_tower.restore_runtime_state(snapshot)


def restore_visual_runtime_if_rejected(
    result: Mapping[str, Any] | _PlannerResultLike,
    *,
    runner: _VisualRuntimeRunner,
    snapshot: VisualRuntimeStateSnapshot | None,
) -> None:
    """Rollback speculative visual state unless the result was accepted."""

    trace = (
        result.get("trace", {})
        if isinstance(result, Mapping)
        else result.trace
    )
    if bool(trace.get("accepted_chunk", False)):
        return
    restore_visual_runtime(runner=runner, snapshot=snapshot)


def clone_session(session: SessionT) -> SessionT:
    """Copy a rollout session for an isolated speculative branch."""

    return copy.deepcopy(session)


def session_reference(session: SessionT, *, share_session: bool) -> SessionT:
    """Share or copy a session according to its runtime ownership model."""

    return session if share_session else clone_session(session)


def snapshot_sequence_visual_runtime(
    *,
    runner: _VisualRuntimeRunner,
    config: ExperimentConfig,
    session: object,
) -> VisualRuntimeStateSnapshot | None:
    """Snapshot global visual state only when the session does not own it."""

    if resolve_mot_runtime_route(config).uses_stateful_realtime_session:
        return None
    return snapshot_visual_runtime(
        runner=runner,
        config=config,
        session=session,
    )


def snapshot_rng_state() -> RuntimeRngSnapshot:
    """Capture global RNG state used by a synchronous speculative branch."""

    return RuntimeRngSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_state=(
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    )


def restore_rng_state(snapshot: RuntimeRngSnapshot | None) -> None:
    """Restore a previously captured global RNG state."""

    if snapshot is None:
        return
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_cpu_state)
    if snapshot.torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_state)


def resolve_future_result(
    future: Future[PlannerResultT],
    *,
    runner: _VisualRuntimeRunner,
    snapshot: VisualRuntimeStateSnapshot | None,
) -> PlannerResultT:
    """Resolve a planner future and rollback visual state on failure."""

    try:
        return future.result()
    except BaseException:
        restore_visual_runtime(runner=runner, snapshot=snapshot)
        raise
