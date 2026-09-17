from __future__ import annotations

from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Iterator, Mapping

import numpy as np
import torch

from open_wam.runtime.policy_planner import PolicyPlanner
from open_wam.simulators.contracts import SimulatorCapabilities, SimulatorObservation
from open_wam.runtime.control import ControlCommand, ControlTransition

from open_wam.configs import DataConfig
from open_wam.pipelines import VariantRolloutRunner, VariantRolloutSession
from open_wam.integrations.simulator_configs import CalvinEnvConfig as CalvinEnvConfig

if TYPE_CHECKING:
    from open_wam.runtime.rollout_engine import RolloutControlStream, RolloutResult


class CalvinBenchmarkAdapter:
    """CALVIN simulator adapter using the official play-table env API when available."""

    benchmark_name = "calvin"
    capabilities = SimulatorCapabilities(action_step_semantics="single_env_step", action_modes=("rel_actions",))

    def __init__(self, config: CalvinEnvConfig) -> None:
        self.config = config
        self.root = None if config.calvin_root is None else Path(config.calvin_root).expanduser().resolve()
        self.dataset_root = (
            None if config.dataset_root is None else Path(config.dataset_root).expanduser().resolve()
        )
        self._env: Any | None = None
        self._task_text = config.task_text
        self._ensure_import_path()

    def reset(self, *, task_id: int | None, episode_idx: int | None, seed: int | None) -> Any:
        if task_id is not None:
            # CALVIN tasks are language/goal driven. Keep task_id accepted for
            # the shared CLI surface but do not invent a task-id mapping here.
            pass
        self.close()
        self._env = self._build_env()
        initial_state = self._load_initial_state(episode_idx=episode_idx)
        if seed is not None and hasattr(self._env, "seed"):
            with self._calvin_cwd():
                self._env.seed(int(seed))
        observation = self._reset_env(initial_state=initial_state)
        observation = _normalize_reset_output(observation)
        if observation is None and hasattr(self._env, "get_obs"):
            observation = self._env.get_obs()
        if observation is None:
            raise RuntimeError("CALVIN environment reset did not return an observation and has no get_obs().")
        return observation

    def task_text(self) -> str | None:
        return self._task_text

    def extract_views(self, observation: Any) -> dict[str, np.ndarray]:
        obs = _unwrap_observation(observation)
        return {
            "rgb_static": _extract_rgb(obs, "rgb_static"),
            "rgb_gripper": _extract_rgb(obs, "rgb_gripper"),
        }

    def extract_state(self, observation: Any) -> np.ndarray | None:
        obs = _unwrap_observation(observation)
        robot_obs = _lookup_nested(obs, ("robot_obs", "state_obs.robot_obs", "observation.robot_obs"))
        if robot_obs is None:
            return None
        return np.asarray(robot_obs, dtype=np.float32).reshape(-1)

    def materialize_control(self, source_action: np.ndarray, *, data_config: DataConfig) -> ControlCommand:
        return _materialize_calvin_control(source_action)

    def step(self, env_action: np.ndarray) -> ControlTransition:
        if self._env is None:
            raise RuntimeError("CALVIN adapter must be reset before stepping.")
        with self._calvin_cwd():
            transition = self._env.step(np.asarray(env_action, dtype=np.float32))
        observation, reward, done, info = _normalize_step_output(transition)
        return ControlTransition(
            observation=observation, reward=reward, done=done,
            success=self.success(observation, info), info=info,
        )

    def success(self, observation: Any, info: dict[str, Any]) -> bool:
        for key in ("success", "is_success", "task_success", "all_tasks_solved"):
            if key in info:
                return bool(info[key])
        solved = info.get("solved_tasks")
        if isinstance(solved, (list, tuple, set)):
            return len(solved) > 0
        return False

    def render_frame(self, observation: Any) -> np.ndarray | None:
        try:
            views = self.extract_views(observation)
        except Exception:
            return None
        static = _as_uint8(views["rgb_static"])
        gripper = _resize_nearest_to_height(_as_uint8(views["rgb_gripper"]), static.shape[0])
        return np.concatenate([static, gripper], axis=1)

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            try:
                with self._calvin_cwd():
                    self._env.close()
            except Exception:
                pass
        self._env = None

    def _ensure_import_path(self) -> None:
        if self.root is None:
            return
        if not self.root.exists():
            raise FileNotFoundError(f"CALVIN root does not exist: {self.root}")
        root_str = str(self.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def _build_env(self) -> Any:
        _patch_legacy_numpy_aliases()
        try:
            from calvin_env.envs.play_table_env import get_env  # type: ignore
            import calvin_env  # type: ignore
        except ImportError as exc:
            missing = getattr(exc, "name", None) or "calvin_env"
            raise ImportError(
                f"CALVIN runtime module {missing!r} is unavailable. Install "
                "`openwam[calvin]` plus the upstream CALVIN environment, or "
                "pass --calvin-root pointing at a compatible checkout."
            ) from exc
        if getattr(calvin_env, "__file__", None) is None and self.root is not None:
            calvin_env.__file__ = str(self.root / "calvin_env" / "calvin_env" / "__init__.py")
        dataset_path = self.dataset_root or self.root
        if dataset_path is None:
            raise ValueError("CALVIN rollout requires --calvin-dataset-root or --calvin-root.")
        obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
        with self._calvin_cwd():
            return get_env(str(dataset_path), obs_space=obs_space, show_gui=bool(self.config.show_gui))

    def _reset_env(self, *, initial_state: dict[str, np.ndarray] | None) -> Any:
        if self._env is None:
            raise RuntimeError("CALVIN environment has not been constructed.")
        if not hasattr(self._env, "reset"):
            return None
        reset = self._env.reset
        with self._calvin_cwd():
            if initial_state:
                try:
                    signature = inspect.signature(reset)
                    kwargs = {
                        key: value
                        for key, value in initial_state.items()
                        if key in signature.parameters
                    }
                    if kwargs:
                        return reset(**kwargs)
                except (TypeError, ValueError):
                    pass
            try:
                return reset()
            except TypeError:
                return reset(None)

    def _load_initial_state(self, *, episode_idx: int | None) -> dict[str, np.ndarray] | None:
        if episode_idx is None or self.dataset_root is None:
            return None
        path = self.dataset_root / f"episode_{int(episode_idx):07d}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as payload:
            state: dict[str, np.ndarray] = {}
            if "robot_obs" in payload:
                state["robot_obs"] = np.asarray(payload["robot_obs"], dtype=np.float32)
            if "scene_obs" in payload:
                state["scene_obs"] = np.asarray(payload["scene_obs"], dtype=np.float32)
            return state or None

    @contextmanager
    def _calvin_cwd(self) -> Iterator[None]:
        if self.root is None:
            yield
            return
        cwd = Path.cwd()
        try:
            os.chdir(self.root)
            yield
        finally:
            os.chdir(cwd)


class OpenWAMCalvinCustomModel:
    """Official-CALVIN-compatible driver of the shared policy control stream."""

    def __init__(
        self, *, rollout_runner: VariantRolloutRunner, data_config: DataConfig,
        device: torch.device, task_text: str | None = None,
    ) -> None:
        self.rollout_runner = rollout_runner
        self.data_config = data_config
        self.device = device
        self.task_text = task_text
        self._stream: RolloutControlStream[VariantRolloutSession, SimulatorObservation] | None = None
        self.last_rollout_result: RolloutResult[VariantRolloutSession, SimulatorObservation] | None = None
        self._active_task = None

    def reset(self) -> None:
        """Drain the previous episode; retain its result for caller diagnostics."""
        if self._stream is not None:
            self.last_rollout_result = self._stream.close()
        self._stream = None
        self._active_task = None

    def step(self, obs: Mapping[str, Any], goal: Any | None = None) -> np.ndarray:
        from open_wam.runtime.rollout_engine import RolloutEngine, RolloutOptions
        from open_wam.simulators.policy_adapter import SimulatorPolicyAdapter

        task_text = _goal_to_task_text(goal) or self.task_text
        if task_text != self._active_task:
            self.reset()
        state = _lookup_nested(obs, ("robot_obs", "state_obs.robot_obs", "observation.robot_obs"))
        observation = SimulatorObservation(
            views={name: _extract_rgb(obs, name) for name in self.data_config.camera_names},
            state=None if state is None else np.asarray(state, dtype=np.float32).reshape(-1),
            task_text=task_text, raw=obs,
        )
        if self._stream is None:
            self._active_task = task_text
            adapter = SimulatorPolicyAdapter(
                self.rollout_runner, self.data_config, self.device, _materialize_calvin_control,
            )
            density = self.rollout_runner.pipeline.policy_variant.rollout_contract.action_tokens_per_frame
            engine = RolloutEngine(PolicyPlanner(self.rollout_runner, adapter), adapter, RolloutOptions(
                max_actions=sys.maxsize, target_action_hz=None, execute_prefix_actions=density,
            ))
            self._stream = engine.control_stream(
                observation, self.rollout_runner.reset(task_text=(task_text,)),
            )
            return next(self._stream)
        return self._stream.send(ControlTransition(observation=observation))


def _materialize_calvin_control(source_action: np.ndarray) -> ControlCommand:
    action = np.asarray(source_action, dtype=np.float32).reshape(-1).copy()
    if action.shape[0] != 7:
        raise ValueError(f"CALVIN expects native 7D rel_actions, got {action.shape[0]}D.")
    _binarize_calvin_gripper(action)
    return ControlCommand(action=action, source_action=action)


def _normalize_step_output(transition: Any) -> tuple[Any, float | None, bool, dict[str, Any]]:
    if isinstance(transition, tuple):
        if len(transition) == 5:
            observation, reward, terminated, truncated, info = transition
            return observation, _float_or_none(reward), bool(terminated or truncated), _dict_or_empty(info)
        if len(transition) == 4:
            observation, reward, done, info = transition
            return observation, _float_or_none(reward), bool(done), _dict_or_empty(info)
        if len(transition) == 2:
            observation, info = transition
            return observation, None, False, _dict_or_empty(info)
    return transition, None, False, {}


def _normalize_reset_output(output: Any) -> Any:
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], Mapping):
        return output[0]
    return output


def _unwrap_observation(observation: Any) -> Mapping[str, Any]:
    if not isinstance(observation, Mapping):
        raise TypeError(f"Expected CALVIN observation mapping, got {type(observation).__name__}.")
    nested = observation.get("observation")
    return nested if isinstance(nested, Mapping) else observation


def _extract_rgb(observation: Mapping[str, Any], key: str) -> np.ndarray:
    direct = _lookup_nested(observation, (key, f"rgb_obs.{key}", f"observation.{key}", f"observation.rgb_obs.{key}"))
    if direct is None:
        raise KeyError(f"CALVIN observation does not expose RGB camera '{key}'.")
    array = np.asarray(direct)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected CALVIN camera '{key}' to have shape [H, W, 3], got {array.shape}.")
    return array[..., :3]


def _lookup_nested(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        cursor: Any = mapping
        found = True
        for part in key.split("."):
            if isinstance(cursor, Mapping) and part in cursor:
                cursor = cursor[part]
            else:
                found = False
                break
        if found:
            return cursor
    return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _goal_to_task_text(goal: Any | None) -> str | None:
    if isinstance(goal, str):
        return goal
    if isinstance(goal, Mapping):
        for key in ("language", "task", "task_text", "instruction"):
            value = goal.get(key)
            if isinstance(value, str):
                return value
    return None


def _patch_legacy_numpy_aliases() -> None:
    """Keep upstream CALVIN/TACTO importable on NumPy 2.x."""

    for alias, value in {
        "bool": np.bool_,
        "float": np.float64,
        "int": np.int_,
    }.items():
        if not hasattr(np, alias):
            setattr(np, alias, value)


def _binarize_calvin_gripper(action: np.ndarray) -> None:
    """CALVIN's relative-control API expects gripper commands in {-1, 1}."""

    action[-1] = 1.0 if float(action[-1]) >= 0.0 else -1.0


def _as_uint8(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)[..., :3]
    if array.dtype != np.uint8:
        if float(np.max(array, initial=0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _resize_nearest_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.shape[0] == target_h:
        return frame
    scale = target_h / frame.shape[0]
    target_w = max(1, int(round(frame.shape[1] * scale)))
    y_indices = np.clip((np.arange(target_h) / scale).astype(np.int64), 0, frame.shape[0] - 1)
    x_indices = np.clip((np.arange(target_w) / scale).astype(np.int64), 0, frame.shape[1] - 1)
    return frame[y_indices][:, x_indices]
