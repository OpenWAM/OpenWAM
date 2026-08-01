"""LIBERO installation discovery and benchmark task resolution."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml


__all__ = [
    "LiberoTaskSpec",
    "ensure_local_libero_config",
    "infer_task_local_episode_rank",
    "load_libero_benchmark_init_state_counts",
    "load_libero_task_init_states",
    "normalize_libero_task_text",
    "resolve_libero_benchmark_tasks",
    "resolve_libero_task",
    "resolve_libero_task_by_id",
]


@dataclass(frozen=True)
class LiberoTaskSpec:
    """Resolved LIBERO benchmark task used to construct a simulator scene."""

    benchmark_name: str
    task_id: int
    task_name: str
    task_language: str
    problem_folder: str
    bddl_file_path: str
    init_states_path: str


def ensure_local_libero_config(project_root: Path | None = None) -> Path:
    """Bootstrap LIBERO's config file without interactive prompts.

    The original LIBERO package prompts on import if `~/.libero/config.yaml`
    does not exist. Collaborative tooling should not depend on interactive
    setup, so Open-WAM writes a local config into `.cache/libero_config/` and
    points `LIBERO_CONFIG_PATH` there before importing the upstream package.
    """

    root = _project_root(project_root)
    libero_repo_root, libero_package_root = _resolve_libero_paths()
    config_dir = root / ".cache" / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "benchmark_root": str(libero_package_root.resolve()),
        "bddl_files": str((libero_package_root / "bddl_files").resolve()),
        "init_states": str((libero_package_root / "init_files").resolve()),
        "datasets": str((libero_repo_root / "libero" / "datasets").resolve()),
        "assets": str((libero_package_root / "assets").resolve()),
    }
    config_path = config_dir / "config.yaml"
    config_text = yaml.safe_dump(config, sort_keys=False)
    if (
        not config_path.is_file()
        or config_path.read_text(encoding="utf-8") != config_text
    ):
        tmp_path = config_path.with_name(
            f"{config_path.name}.{os.getpid()}.tmp"
        )
        tmp_path.write_text(config_text, encoding="utf-8")
        tmp_path.replace(config_path)

    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return config_path


def resolve_libero_task(
    task_text: str,
    project_root: Path | None = None,
    *,
    benchmark_name: str | None = None,
) -> LiberoTaskSpec:
    """Resolve dataset task text to one upstream LIBERO benchmark task."""

    ensure_local_libero_config(project_root)
    from libero.libero import benchmark  # type: ignore

    normalized_task_text = normalize_libero_task_text(task_text)
    init_states_root: Path | None = None
    matches: list[LiberoTaskSpec] = []
    benchmark_classes = benchmark.get_benchmark_dict()
    if benchmark_name is not None:
        try:
            benchmark_items = (
                (benchmark_name, benchmark_classes[benchmark_name]),
            )
        except KeyError as exc:
            available = ", ".join(sorted(benchmark_classes))
            raise ValueError(
                f"Unknown LIBERO benchmark {benchmark_name!r}; "
                f"available benchmarks: {available}"
            ) from exc
    else:
        benchmark_items = tuple(benchmark_classes.items())

    for current_benchmark_name, benchmark_class in benchmark_items:
        try:
            benchmark_instance = benchmark_class()
        except Exception:
            # Upstream registers suites such as LIBERO_100 that are not fully
            # initialized in this checkout. Task resolution should ignore
            # those and keep searching usable benchmark variants.
            continue
        for task_id in range(benchmark_instance.get_num_tasks()):
            task = benchmark_instance.get_task(task_id)
            if (
                normalize_libero_task_text(task.language)
                != normalized_task_text
            ):
                continue
            if init_states_root is None:
                init_states_root = Path(
                    _read_local_libero_config()["init_states"]
                )
            matches.append(
                _build_libero_task_spec(
                    benchmark_name=current_benchmark_name,
                    benchmark_instance=benchmark_instance,
                    task_id=task_id,
                    init_states_root=init_states_root,
                )
            )

    if not matches:
        raise ValueError(f"Could not resolve LIBERO task text: {task_text!r}")
    if len(matches) > 1:
        raise ValueError(
            f"Task text {task_text!r} matched multiple LIBERO tasks; "
            "expected exactly one. "
            f"Matches: {[match.task_name for match in matches]}"
        )

    return matches[0]


def resolve_libero_task_by_id(
    benchmark_name: str,
    task_id: int,
    project_root: Path | None = None,
) -> LiberoTaskSpec:
    """Resolve one LIBERO benchmark/task-id pair into a task spec."""

    benchmark_instance, task_count, init_states_root = _resolve_libero_benchmark(
        benchmark_name,
        project_root,
    )
    task_id = int(task_id)
    if task_id < 0 or task_id >= task_count:
        raise ValueError(
            f"task_id={task_id} is outside benchmark {benchmark_name!r} "
            f"with {task_count} tasks."
        )
    return _build_libero_task_spec(
        benchmark_name=benchmark_name,
        benchmark_instance=benchmark_instance,
        task_id=task_id,
        init_states_root=init_states_root,
    )


def resolve_libero_benchmark_tasks(
    benchmark_name: str,
    project_root: Path | None = None,
) -> tuple[LiberoTaskSpec, ...]:
    """Resolve the ordered task inventory for one upstream LIBERO suite."""

    benchmark_instance, task_count, init_states_root = _resolve_libero_benchmark(
        benchmark_name,
        project_root,
    )
    return tuple(
        _build_libero_task_spec(
            benchmark_name=benchmark_name,
            benchmark_instance=benchmark_instance,
            task_id=task_id,
            init_states_root=init_states_root,
        )
        for task_id in range(task_count)
    )


def load_libero_benchmark_init_state_counts(
    benchmark_name: str,
    *,
    task_ids: Sequence[int] | None = None,
    project_root: Path | None = None,
) -> dict[int, int]:
    """Count init states for all or selected tasks in one LIBERO suite."""

    benchmark_instance, task_count, init_states_root = _resolve_libero_benchmark(
        benchmark_name,
        project_root,
    )
    selected_task_ids = (
        list(range(task_count)) if task_ids is None else list(task_ids)
    )
    invalid_task_ids = [
        task_id
        for task_id in selected_task_ids
        if task_id < 0 or task_id >= task_count
    ]
    if invalid_task_ids:
        raise ValueError(
            f"Requested task ids exceed benchmark {benchmark_name!r} task count "
            f"{task_count}: {invalid_task_ids}"
        )
    init_counts: dict[int, int] = {}
    for task_id in selected_task_ids:
        task_spec = _build_libero_task_spec(
            benchmark_name=benchmark_name,
            benchmark_instance=benchmark_instance,
            task_id=int(task_id),
            init_states_root=init_states_root,
        )
        init_states = load_libero_task_init_states(task_spec, project_root)
        init_counts[int(task_id)] = int(len(init_states))
    return init_counts


def _resolve_libero_benchmark(
    benchmark_name: str,
    project_root: Path | None,
) -> tuple[Any, int, Path]:
    ensure_local_libero_config(project_root)
    from libero.libero import benchmark  # type: ignore

    benchmark_classes = benchmark.get_benchmark_dict()
    try:
        benchmark_class = benchmark_classes[benchmark_name]
    except KeyError as exc:
        available = ", ".join(sorted(benchmark_classes))
        raise ValueError(
            f"Unknown LIBERO benchmark {benchmark_name!r}; "
            f"available benchmarks: {available}"
        ) from exc
    benchmark_instance = benchmark_class()
    get_num_tasks = getattr(benchmark_instance, "get_num_tasks", None)
    if callable(get_num_tasks):
        task_count = int(get_num_tasks())
    else:
        raw_task_count = getattr(benchmark_instance, "n_tasks", None)
        task_count = (
            int(raw_task_count)
            if raw_task_count is not None
            else len(benchmark_instance.tasks)
        )
    init_states_root = Path(_read_local_libero_config()["init_states"])
    return benchmark_instance, task_count, init_states_root


def _build_libero_task_spec(
    *,
    benchmark_name: str,
    benchmark_instance: Any,
    task_id: int,
    init_states_root: Path,
) -> LiberoTaskSpec:
    task = benchmark_instance.get_task(task_id)
    return LiberoTaskSpec(
        benchmark_name=benchmark_name,
        task_id=int(task_id),
        task_name=str(task.name),
        task_language=str(task.language),
        problem_folder=str(task.problem_folder),
        bddl_file_path=str(benchmark_instance.get_task_bddl_file_path(task_id)),
        init_states_path=str(
            init_states_root / task.problem_folder / f"{task.name}.pruned_init"
        ),
    )


def load_libero_task_init_states(
    task_spec: LiberoTaskSpec,
    project_root: Path | None = None,
) -> Any:
    """Load benchmark init states with torch 2.6-compatible semantics."""

    import torch

    ensure_local_libero_config(project_root)
    # Upstream uses `torch.load(path)` which defaults to `weights_only=True`
    # on torch 2.6+. The init-state files are not weight checkpoints.
    return torch.load(task_spec.init_states_path, weights_only=False)


def infer_task_local_episode_rank(
    episode_records: list[Any] | tuple[Any, ...],
    *,
    episode_index: int,
    task_text: str,
) -> int:
    """Map one dataset episode to its stable per-task demo index."""

    normalized = normalize_libero_task_text(task_text)
    rank = 0
    for record in episode_records:
        record_task = record.tasks[0] if getattr(record, "tasks", None) else ""
        if int(record.episode_index) == episode_index:
            return rank
        if normalize_libero_task_text(record_task) == normalized:
            rank += 1
    raise ValueError(
        f"Episode index {episode_index} was not found in episode metadata."
    )


def normalize_libero_task_text(task_text: str) -> str:
    """Normalize task text for benchmark and dataset identity matching."""

    return " ".join(task_text.strip().lower().split())


def _read_local_libero_config() -> dict[str, Any]:
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _project_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_libero_paths() -> tuple[Path, Path]:
    """Resolve the installed LIBERO repo and nested package roots.

    Upstream LIBERO uses an unusual nested package layout:
    `<repo>/libero/libero/__init__.py`. Some local installs therefore record
    distribution metadata without exposing an importable `libero` package.
    When that happens, a configured or adjacent checkout remains usable.
    """

    env_repo_root = os.environ.get("LIBERO_REPO_ROOT")
    if env_repo_root:
        env_paths = _libero_paths_from_repo_root(
            Path(env_repo_root).expanduser()
        )
        if env_paths is not None:
            return env_paths

    import_error: Exception | None = None
    try:
        libero_pkg = importlib.import_module("libero.libero")
    except EOFError as exc:
        # Upstream can prompt before its config is bootstrapped. Fall back to a
        # checkout so `ensure_local_libero_config` can write that config first.
        import_error = exc
        libero_pkg = None
    except ModuleNotFoundError as exc:
        if exc.name not in {"libero", "libero.libero"}:
            raise
        import_error = exc
        libero_pkg = None

    if libero_pkg is not None:
        package_root = Path(libero_pkg.__file__).resolve().parent
        repo_root = package_root.parents[1]
        return repo_root, package_root

    project_root = _project_root(None)
    for repo_root in (project_root.parent / "LIBERO",):
        paths = _libero_paths_from_repo_root(repo_root)
        if paths is not None:
            return paths

    raise ImportError(
        "LIBERO could not be imported. Install `open-wam[libero]` plus an "
        "importable upstream LIBERO package, or set LIBERO_REPO_ROOT to a "
        "checkout whose structure contains `libero/libero/__init__.py`."
    ) from import_error


def _libero_paths_from_repo_root(
    repo_root: Path,
) -> tuple[Path, Path] | None:
    package_root = repo_root / "libero" / "libero"
    if not (package_root / "__init__.py").exists():
        return None
    repo_root_resolved = repo_root.resolve()
    repo_root_str = str(repo_root_resolved)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root_resolved, package_root.resolve()
