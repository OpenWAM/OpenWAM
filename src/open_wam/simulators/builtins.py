"""Lazy factories for simulator integrations maintained in this repository."""

from __future__ import annotations

from collections.abc import Mapping

from .contracts import SimulatorBackend, ensure_simulator_backend
from .registry import (
    SIMULATOR_ADAPTER_FACTORIES,
    SimulatorFactoryContext,
    register_simulator_adapter,
)


def normalize_builtin_simulator_options(
    *,
    benchmark: str,
    arguments: Mapping[str, object],
    explicit_options: Mapping[str, str],
) -> dict[str, str]:
    """Translate maintained compatibility flags into generic factory options."""

    option_values: dict[str, object | None] = {}
    if benchmark == "robotwin":
        option_values = {
            "root": arguments.get("robotwin_root"),
            "task_name": arguments.get("robotwin_task_name"),
            "task_config": arguments.get("robotwin_task_config"),
            "instruction": arguments.get("instruction"),
            "action_type": arguments.get("robotwin_action_type"),
            "expert_precheck": arguments.get("robotwin_expert_precheck"),
            "instruction_type": arguments.get("robotwin_instruction_type"),
        }
    elif benchmark == "calvin":
        option_values = {
            "root": arguments.get("calvin_root"),
            "dataset_root": arguments.get("calvin_dataset_root"),
            "task_text": arguments.get("calvin_task_text")
            or arguments.get("instruction"),
            "show_gui": arguments.get("show_gui"),
        }
    normalized = {
        option_name: _stringify_option(value)
        for option_name, value in option_values.items()
        if value is not None
    }
    normalized.update(explicit_options)
    return normalized


def register_builtin_simulator_adapters() -> None:
    """Register built-ins once without importing their optional environments."""

    for benchmark, factory, description in (
        ("robotwin", _build_robotwin, "RoboTwin official task environment."),
        ("calvin", _build_calvin, "CALVIN play-table environment."),
    ):
        if SIMULATOR_ADAPTER_FACTORIES.get(benchmark) is None:
            register_simulator_adapter(
                benchmark,
                factory,
                description=description,
            )


def _build_robotwin(context: SimulatorFactoryContext) -> SimulatorBackend:
    from open_wam.integrations.robotwin_env import (
        RobotwinBenchmarkAdapter,
        RobotwinEnvConfig,
    )

    root = _first_value(
        context.options.get("root"),
        context.local_paths.get("simulators.robotwin_root"),
    )
    if root is None:
        raise ValueError(
            "RoboTwin rollout requires --robotwin-root, --sim-option root=..., "
            "or paths.simulators.robotwin_root in the local path registry."
        )
    task_name = _first_value(
        context.options.get("task_name"),
    )
    if task_name is None:
        raise ValueError(
            "RoboTwin rollout requires --robotwin-task-name or "
            "--sim-option task_name=...."
        )
    task_config = _first_value(
        context.options.get("task_config"),
        task_name,
    )
    adapter = RobotwinBenchmarkAdapter(
        RobotwinEnvConfig(
            robotwin_root=root,
            task_name=task_name,
            task_config=task_config,
            instruction=_first_value(
                context.options.get("instruction"),
            ),
            action_type=_first_value(
                context.options.get("action_type"),
                "ee",
            ),
            expert_precheck=_option_bool(
                context.options.get("expert_precheck", "false")
            ),
            instruction_type=_first_value(
                context.options.get("instruction_type"),
                "seen",
            ),
        )
    )
    return ensure_simulator_backend(adapter)


def _build_calvin(context: SimulatorFactoryContext) -> SimulatorBackend:
    from open_wam.integrations.calvin_env import CalvinBenchmarkAdapter, CalvinEnvConfig

    adapter = CalvinBenchmarkAdapter(
        CalvinEnvConfig(
            calvin_root=_first_value(
                context.options.get("root"),
                context.local_paths.get("simulators.calvin_root"),
            ),
            dataset_root=_first_value(
                context.options.get("dataset_root"),
                context.local_paths.get("datasets.calvin_root"),
            ),
            task_text=_first_value(
                context.options.get("task_text"),
            ),
            show_gui=_option_bool(context.options.get("show_gui", "false")),
        )
    )
    return ensure_simulator_backend(adapter)


def _first_value(*values: str | None) -> str | None:
    return next((value for value in values if value is not None), None)


def _stringify_option(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _option_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean simulator option, got {value!r}.")


__all__ = [
    "normalize_builtin_simulator_options",
    "register_builtin_simulator_adapters",
]
