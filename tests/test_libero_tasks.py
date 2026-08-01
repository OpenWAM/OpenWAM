from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import yaml

from open_wam.integrations import libero_tasks


class _FakeTask:
    def __init__(self, index: int) -> None:
        self.name = f"task_name_{index}"
        self.language = f"Do Task {index}"
        self.problem_folder = f"suite_{index % 2}"


class _FakeBenchmark:
    def __init__(self, *, expose_get_num_tasks: bool = True) -> None:
        self.tasks = [_FakeTask(index) for index in range(3)]
        self.n_tasks = len(self.tasks)
        if not expose_get_num_tasks:
            self.get_num_tasks = None  # type: ignore[assignment]

    def get_num_tasks(self) -> int:
        return len(self.tasks)

    def get_task(self, index: int) -> _FakeTask:
        return self.tasks[index]

    def get_task_bddl_file_path(self, index: int) -> str:
        return f"/bddl/task_{index}.bddl"


def _install_fake_libero(
    monkeypatch: pytest.MonkeyPatch,
    *,
    benchmark_factory,
) -> None:
    benchmark_module = types.ModuleType("libero.libero.benchmark")
    benchmark_module.get_benchmark_dict = lambda: {  # type: ignore[attr-defined]
        "libero_10": benchmark_factory
    }
    root_module = types.ModuleType("libero")
    nested_module = types.ModuleType("libero.libero")
    root_module.libero = nested_module  # type: ignore[attr-defined]
    nested_module.benchmark = benchmark_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", root_module)
    monkeypatch.setitem(sys.modules, "libero.libero", nested_module)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_module)


def _install_fake_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_root = tmp_path / "libero_config"
    config_root.mkdir()
    config_path = config_root / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"init_states": str(tmp_path / "init_states")}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config_root))
    monkeypatch.setattr(
        libero_tasks,
        "ensure_local_libero_config",
        lambda project_root=None: config_path,
    )
    return config_path


@pytest.mark.parametrize("expose_get_num_tasks", [True, False])
def test_resolve_libero_benchmark_tasks_returns_ordered_typed_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    expose_get_num_tasks: bool,
) -> None:
    _install_fake_libero(
        monkeypatch,
        benchmark_factory=lambda: _FakeBenchmark(
            expose_get_num_tasks=expose_get_num_tasks
        ),
    )
    _install_fake_config(monkeypatch, tmp_path)

    task_specs = libero_tasks.resolve_libero_benchmark_tasks(
        "libero_10",
        tmp_path / "project",
    )

    assert [task.task_id for task in task_specs] == [0, 1, 2]
    assert [task.task_language for task in task_specs] == [
        "Do Task 0",
        "Do Task 1",
        "Do Task 2",
    ]
    assert task_specs[1] == libero_tasks.LiberoTaskSpec(
        benchmark_name="libero_10",
        task_id=1,
        task_name="task_name_1",
        task_language="Do Task 1",
        problem_folder="suite_1",
        bddl_file_path="/bddl/task_1.bddl",
        init_states_path=str(
            tmp_path / "init_states" / "suite_1" / "task_name_1.pruned_init"
        ),
    )
    assert (
        libero_tasks.resolve_libero_task_by_id("libero_10", 1, tmp_path)
        == task_specs[1]
    )


def test_load_libero_benchmark_init_state_counts_preserves_requested_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_libero(monkeypatch, benchmark_factory=_FakeBenchmark)
    _install_fake_config(monkeypatch, tmp_path)
    calls: list[int] = []

    def fake_load(task_spec, project_root=None):
        del project_root
        calls.append(task_spec.task_id)
        return list(range(task_spec.task_id + 2))

    monkeypatch.setattr(libero_tasks, "load_libero_task_init_states", fake_load)

    counts = libero_tasks.load_libero_benchmark_init_state_counts(
        "libero_10",
        task_ids=[2, 0],
        project_root=tmp_path,
    )

    assert list(counts) == [2, 0]
    assert counts == {2: 4, 0: 2}
    assert calls == [2, 0]


def test_subset_count_does_not_materialize_unrequested_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class SelectiveBenchmark(_FakeBenchmark):
        def get_task(self, index: int) -> _FakeTask:
            if index != 2:
                raise AssertionError(f"unexpected task materialization: {index}")
            return super().get_task(index)

    _install_fake_libero(monkeypatch, benchmark_factory=SelectiveBenchmark)
    _install_fake_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        libero_tasks,
        "load_libero_task_init_states",
        lambda task_spec, project_root=None: [task_spec.task_id],
    )

    assert libero_tasks.load_libero_benchmark_init_state_counts(
        "libero_10",
        task_ids=[2],
        project_root=tmp_path,
    ) == {2: 1}


def test_load_libero_benchmark_init_state_counts_rejects_invalid_ids_before_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_libero(monkeypatch, benchmark_factory=_FakeBenchmark)
    _install_fake_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        libero_tasks,
        "load_libero_task_init_states",
        lambda *args, **kwargs: pytest.fail("invalid ids must fail before state I/O"),
    )

    with pytest.raises(
        ValueError,
        match=r"Requested task ids exceed benchmark 'libero_10' task count 3: \[3\]",
    ):
        libero_tasks.load_libero_benchmark_init_state_counts(
            "libero_10",
            task_ids=[3],
            project_root=tmp_path,
        )


def test_resolve_libero_benchmark_tasks_reports_available_suites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_libero(monkeypatch, benchmark_factory=_FakeBenchmark)
    _install_fake_config(monkeypatch, tmp_path)

    with pytest.raises(
        ValueError,
        match="Unknown LIBERO benchmark 'missing'; available benchmarks: libero_10",
    ):
        libero_tasks.resolve_libero_benchmark_tasks("missing", tmp_path)
