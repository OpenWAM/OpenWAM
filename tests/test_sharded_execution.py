from __future__ import annotations

from contextlib import contextmanager

import pytest
from torch import nn

from open_wam.models.common import (
    checkpoint_unshard_context,
    summon_full_parameters,
    unshard_runtime_parameters,
)
from open_wam.models.common import sharded_execution


class _TrackedShard(nn.Module):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__()
        self.name = name
        self.events = events

    def unshard(self) -> None:
        self.events.append(f"unshard:{self.name}")

    def reshard(self) -> None:
        self.events.append(f"reshard:{self.name}")


def test_unshard_runtime_parameters_deduplicates_and_reshards_in_reverse() -> None:
    events: list[str] = []
    first = _TrackedShard("first", events)
    second = _TrackedShard("second", events)
    parent = nn.Module()
    parent.add_module("first", first)
    parent.add_module("second", second)

    with unshard_runtime_parameters(parent, first):
        events.append("execute")

    assert events == [
        "unshard:first",
        "unshard:second",
        "execute",
        "reshard:second",
        "reshard:first",
    ]


def test_unshard_runtime_parameters_reshards_after_exception() -> None:
    events: list[str] = []
    module = _TrackedShard("only", events)

    try:
        with unshard_runtime_parameters(module):
            raise RuntimeError("stop")
    except RuntimeError as exc:
        assert str(exc) == "stop"
    else:
        raise AssertionError("Expected the execution exception to propagate.")

    assert events == ["unshard:only", "reshard:only"]


def test_checkpoint_unshard_context_builds_independent_contexts() -> None:
    events: list[str] = []
    module = _TrackedShard("only", events)
    forward_context, recompute_context = checkpoint_unshard_context(module)

    assert events == ["unshard:only", "unshard:only"]

    with forward_context:
        events.append("forward")
    with recompute_context:
        events.append("recompute")

    assert events == [
        "unshard:only",
        "unshard:only",
        "forward",
        "reshard:only",
        "recompute",
        "reshard:only",
    ]


def test_summon_full_parameters_deduplicates_fsdp1_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    first = object()
    second = object()
    roots = {
        "left": (first, second),
        "right": (second,),
    }

    class _FakeFSDP:
        @staticmethod
        def fsdp_modules(module, *, root_only):
            assert root_only is False
            return roots[module]

        @staticmethod
        @contextmanager
        def summon_full_params(module, *, recurse, writeback):
            assert recurse is False
            assert writeback is False
            name = "first" if module is first else "second"
            events.append(f"enter:{name}")
            try:
                yield
            finally:
                events.append(f"exit:{name}")

    monkeypatch.setattr(sharded_execution, "FSDP", _FakeFSDP)

    with summon_full_parameters("left", "right"):
        events.append("execute")

    assert events == [
        "enter:first",
        "enter:second",
        "execute",
        "exit:second",
        "exit:first",
    ]
