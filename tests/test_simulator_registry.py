from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from open_wam.cli.sim_rollout import build_arg_parser
from open_wam.evals.sim_rollout import _build_adapter, _parse_simulator_options
from open_wam.simulators.builtins import _build_calvin
from open_wam.simulators.contracts import ObservationAdapterBackend
from open_wam.simulators.registry import (
    SimulatorFactoryContext,
    register_simulator_adapter,
    registered_simulator_adapters,
)


def test_application_simulator_factory_receives_generic_context() -> None:
    benchmark = f"fixture-{uuid4().hex}"
    captured: list[SimulatorFactoryContext] = []
    adapter = object()

    def factory(context: SimulatorFactoryContext):
        captured.append(context)
        return adapter

    register_simulator_adapter(benchmark, factory, description="Test adapter.")
    args = build_arg_parser().parse_args(
        [
            "--cfg",
            "experiment.yaml",
            "--benchmark",
            benchmark,
            "--sim-option",
            "endpoint=local",
        ]
    )

    assert _build_adapter(args) is adapter
    assert captured[0].benchmark == benchmark
    assert dict(captured[0].options) == {"endpoint": "local"}
    with pytest.raises(TypeError):
        captured[0].options["endpoint"] = "remote"  # type: ignore[index]
    assert benchmark in registered_simulator_adapters()


def test_simulator_options_reject_ambiguous_input() -> None:
    with pytest.raises(SystemExit, match="expected KEY=VALUE"):
        _parse_simulator_options(["missing-value"])
    with pytest.raises(SystemExit, match="Duplicate"):
        _parse_simulator_options(["root=one", "root=two"])


def test_simulator_factory_context_rejects_untyped_options() -> None:
    with pytest.raises(TypeError, match="string keys and values"):
        SimulatorFactoryContext(
            benchmark="fixture",
            options={"retries": 3},  # type: ignore[dict-item]
            local_paths={},
        )


def test_builtin_factory_normalizes_raw_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from open_wam.integrations import calvin_env

    from open_wam.simulators.contracts import SimulatorCapabilities

    raw_adapter = SimpleNamespace(benchmark_name="calvin", capabilities=SimulatorCapabilities("single_env_step"))
    monkeypatch.setattr(
        calvin_env,
        "CalvinBenchmarkAdapter",
        lambda config: raw_adapter,
    )

    adapter = _build_calvin(
        SimulatorFactoryContext(
            benchmark="calvin",
            options={},
            local_paths={},
        )
    )

    assert isinstance(adapter, ObservationAdapterBackend)
    assert adapter.adapter is raw_adapter


def test_unknown_simulator_reports_registered_identifiers() -> None:
    args = SimpleNamespace(
        benchmark=f"missing-{uuid4().hex}",
        sim_option=[],
    )

    with pytest.raises(SystemExit, match="Unsupported simulator adapter factory"):
        _build_adapter(args)
