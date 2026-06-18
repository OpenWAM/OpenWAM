from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_wam.models.common.rollout_history import resolve_execute_action_steps


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_libero_mot_batch_visualization.py"


def _load_functions(*names: str) -> SimpleNamespace:
    module_ast = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    selected = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "SimpleNamespace": SimpleNamespace,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MODULE_PATH), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def test_parse_int_ranges_accepts_single_values_and_ranges() -> None:
    helpers = _load_functions("_parse_int_ranges")

    assert helpers._parse_int_ranges("0", label="task-ids") == [0]
    assert helpers._parse_int_ranges("0-2,5", label="episode-idxs") == [0, 1, 2, 5]


def test_parse_int_ranges_rejects_empty_and_descending_ranges() -> None:
    helpers = _load_functions("_parse_int_ranges")

    with pytest.raises(ValueError):
        helpers._parse_int_ranges("", label="task-ids")
    with pytest.raises(ValueError):
        helpers._parse_int_ranges("3-1", label="task-ids")


def test_iter_pairs_supports_task_episode_and_episode_task_order() -> None:
    helpers = _load_functions("_iter_pairs")

    assert list(helpers._iter_pairs([0, 1], [7, 8], loop_order="task_episode")) == [
        (0, 7),
        (0, 8),
        (1, 7),
        (1, 8),
    ]
    assert list(helpers._iter_pairs([0, 1], [7, 8], loop_order="episode_task")) == [
        (0, 7),
        (1, 7),
        (0, 8),
        (1, 8),
    ]


def test_iter_pairs_rejects_unknown_loop_order() -> None:
    helpers = _load_functions("_iter_pairs")

    with pytest.raises(ValueError):
        list(helpers._iter_pairs([0], [0], loop_order="bad"))


class _FakeEnv:
    def __init__(self, name: str) -> None:
        self.name = name
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_acquire_rollout_env_reuses_env_within_task_and_closes_on_task_switch() -> None:
    helpers = _load_functions("_acquire_rollout_env", "_close_reused_env")
    constructed: list[_FakeEnv] = []

    def _construct_single_env(task_spec):
        env = _FakeEnv(str(task_spec))
        constructed.append(env)
        return env

    helpers._acquire_rollout_env.__globals__["mot_viz"] = SimpleNamespace(_construct_single_env=_construct_single_env)
    args = SimpleNamespace(reuse_env_per_task=True)
    resources = SimpleNamespace(reused_env=None, reused_env_task_id=None)

    env0, close0 = helpers._acquire_rollout_env(args, resources, task_spec="task0", task_id=0)
    env0_again, close0_again = helpers._acquire_rollout_env(args, resources, task_spec="task0", task_id=0)
    env1, close1 = helpers._acquire_rollout_env(args, resources, task_spec="task1", task_id=1)

    assert env0 is env0_again
    assert env1 is not env0
    assert (close0, close0_again, close1) == (False, False, False)
    assert env0.close_calls == 1
    assert resources.reused_env is env1

    helpers._close_reused_env(resources)
    assert env1.close_calls == 1
    assert resources.reused_env is None


def test_acquire_rollout_env_does_not_reuse_when_disabled() -> None:
    helpers = _load_functions("_acquire_rollout_env", "_close_reused_env")
    constructed: list[_FakeEnv] = []

    def _construct_single_env(task_spec):
        env = _FakeEnv(str(task_spec))
        constructed.append(env)
        return env

    helpers._acquire_rollout_env.__globals__["mot_viz"] = SimpleNamespace(_construct_single_env=_construct_single_env)
    args = SimpleNamespace(reuse_env_per_task=False)
    resources = SimpleNamespace(reused_env=None, reused_env_task_id=None)

    env0, close0 = helpers._acquire_rollout_env(args, resources, task_spec="task0", task_id=0)
    env1, close1 = helpers._acquire_rollout_env(args, resources, task_spec="task0", task_id=0)

    assert env0 is not env1
    assert (close0, close1) == (True, True)
    assert resources.reused_env is None


def test_resolve_execute_action_steps_defaults_to_full_horizon() -> None:
    assert resolve_execute_action_steps(None, action_horizon=16, action_per_frame=4) == 16
    assert resolve_execute_action_steps(8, action_horizon=16, action_per_frame=4) == 8
    assert (
        resolve_execute_action_steps(
            None,
            execute_frame_chunk_size=2,
            action_horizon=16,
            action_per_frame=4,
        )
        == 8
    )


def test_resolve_execute_action_steps_rejects_unaligned_or_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_execute_action_steps(0, action_horizon=16, action_per_frame=4)
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve_execute_action_steps(20, action_horizon=16, action_per_frame=4)
    with pytest.raises(ValueError, match="aligned"):
        resolve_execute_action_steps(6, action_horizon=16, action_per_frame=4)
    with pytest.raises(ValueError, match="Pass only one"):
        resolve_execute_action_steps(8, execute_frame_chunk_size=2, action_horizon=16, action_per_frame=4)
