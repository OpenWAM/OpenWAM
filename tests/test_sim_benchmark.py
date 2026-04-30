from __future__ import annotations

from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from open_wam.configs import ActionMappingConfig, ActionSchemaConfig, CalvinDataConfig, RobotWinDataConfig
from open_wam.data.action_mapping import apply_action_mapping
from open_wam.integrations.calvin_env import CalvinBenchmarkAdapter, CalvinEnvConfig
from open_wam.integrations.robotwin_env import (
    RobotwinBenchmarkAdapter,
    RobotwinEnvConfig,
    _install_ee_skip_topp_planner_patch,
)
from open_wam.simulators import (
    EpisodeSpec,
    SimStepResult,
    SimulatorCapabilities,
    SimulatorObservation,
    SimulatorStepResult,
    run_closed_loop_sim_rollout,
)


def test_closed_loop_sim_rollout_uses_current_observations_only() -> None:
    data_config = CalvinDataConfig(
        num_frames=2,
        action_schema=ActionSchemaConfig(action_dim=7, action_horizon=2, state_dim=15, state_horizon=1),
    )
    adapter = _FakeCalvinAdapter()
    runner = _FakeRolloutRunner(action_dim=7, action_horizon=2)

    result = run_closed_loop_sim_rollout(
        adapter=adapter,
        rollout_runner=runner,
        data_config=data_config,
        device=torch.device("cpu"),
        task_id=None,
        episode_idx=0,
        seed=123,
        max_steps=5,
    )

    assert result.success is True
    assert result.steps == 3
    assert result.policy_action_shapes == ((1, 2, 7), (1, 2, 7), (1, 2, 7))
    assert len(result.video_frames) == 3
    assert adapter.seen_env_actions == [0.0, 1.0, 2.0]
    assert runner.seen_frame_values == [0, 1, 2]
    assert runner.seen_previous_action_shapes == [None, (1, 7), (1, 7)]


def test_closed_loop_sim_rollout_can_commit_full_policy_chunks() -> None:
    data_config = CalvinDataConfig(
        num_frames=2,
        action_schema=ActionSchemaConfig(action_dim=7, action_horizon=2, state_dim=15, state_horizon=1),
    )
    adapter = _FakeCalvinAdapter(success_after=99)
    runner = _FakeRolloutRunner(action_dim=7, action_horizon=2)

    result = run_closed_loop_sim_rollout(
        adapter=adapter,
        rollout_runner=runner,
        data_config=data_config,
        device=torch.device("cpu"),
        task_id=None,
        episode_idx=0,
        seed=123,
        max_steps=4,
        action_commit_mode="full_chunk",
    )

    assert result.success is False
    assert result.steps == 4
    assert result.policy_action_shapes == ((1, 2, 7), (1, 2, 7))
    assert adapter.seen_env_actions == [0.0, 0.0, 2.0, 2.0]
    assert runner.seen_frame_values == [0, 2]
    assert runner.seen_previous_action_shapes == [None, (1, 2, 7)]
    assert [record["chunk_action_index"] for record in result.action_records] == [0, 1, 0, 1]
    assert [record["reused_policy_output"] for record in result.action_records] == [False, True, False, True]


def test_closed_loop_sim_rollout_accepts_normalized_simulator_backend() -> None:
    data_config = CalvinDataConfig(
        num_frames=2,
        action_schema=ActionSchemaConfig(action_dim=7, action_horizon=2, state_dim=15, state_horizon=1),
    )
    adapter = _FakeNormalizedSimulator()
    runner = _FakeRolloutRunner(action_dim=7, action_horizon=2)

    result = run_closed_loop_sim_rollout(
        adapter=adapter,
        rollout_runner=runner,
        data_config=data_config,
        device=torch.device("cpu"),
        task_id=4,
        episode_idx=5,
        seed=6,
        max_steps=2,
    )

    assert result.benchmark == "fake_normalized"
    assert result.task_text == "normalized task"
    assert result.steps == 2
    assert adapter.seen_specs == [EpisodeSpec(task_id=4, episode_idx=5, seed=6)]
    assert adapter.seen_env_actions == [0.0, 1.0]
    assert runner.seen_frame_values == [0, 1]
    assert [record["sim_action_step_semantics"] for record in result.action_records] == [
        "single_env_step",
        "single_env_step",
    ]


def test_calvin_adapter_inverse_maps_sparse_30d_to_native_7d(tmp_path: Path) -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=7,
        target_dim=30,
        source_to_target_indices=(0, 1, 2, 3, 4, 5, 28),
        active_target_indices=(0, 1, 2, 3, 4, 5, 28),
    )
    config = CalvinDataConfig(
        action_schema=ActionSchemaConfig(action_dim=30, action_horizon=2, state_dim=15, state_horizon=1),
        action_mapping=mapping,
    )
    source = torch.arange(7, dtype=torch.float32).unsqueeze(0)
    mapped = apply_action_mapping(source, torch.ones_like(source), mapping, target_dim=30)
    adapter = CalvinBenchmarkAdapter(CalvinEnvConfig(dataset_root=str(tmp_path)))

    env_action = adapter.model_action_to_env_action(mapped.actions[0].numpy(), data_config=config)

    assert env_action.shape == (7,)
    np.testing.assert_allclose(env_action[:6], source[0, :6].numpy())
    assert env_action[6] == 1.0


def test_robotwin_adapter_inverse_maps_sparse_30d_and_normalizes_quaternions(tmp_path: Path) -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=16,
        target_dim=30,
        source_to_target_indices=(0, 1, 2, 3, 4, 5, 6, 28, 7, 8, 9, 10, 11, 12, 13, 29),
        active_target_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 28, 29),
    )
    config = RobotWinDataConfig(
        action_schema=ActionSchemaConfig(action_dim=30, action_horizon=2, state_dim=16, state_horizon=1),
        action_mapping=mapping,
    )
    source = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.5, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 3.0, -0.5]],
        dtype=torch.float32,
    )
    mapped = apply_action_mapping(source, torch.ones_like(source), mapping, target_dim=30)
    adapter = RobotwinBenchmarkAdapter(
        RobotwinEnvConfig(robotwin_root=str(tmp_path), task_name="dummy_task", task_config="dummy_task")
    )

    env_action = adapter.model_action_to_env_action(mapped.actions[0].numpy(), data_config=config)

    assert env_action.shape == (16,)
    np.testing.assert_allclose(env_action[0:3], source[0, 0:3].numpy())
    np.testing.assert_allclose(env_action[3:7], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(env_action[11:15], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert env_action[7] == source[0, 7].item()
    assert env_action[15] == source[0, 15].item()


def test_robotwin_qpos_adapter_drops_eef_quaternion_padding(tmp_path: Path) -> None:
    mapping = ActionMappingConfig(
        mode="sparse_canvas",
        source_dim=16,
        target_dim=30,
        source_to_target_indices=(0, 1, 2, 3, 4, 5, 6, 28, 7, 8, 9, 10, 11, 12, 13, 29),
        active_target_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 28, 29),
    )
    config = RobotWinDataConfig(
        action_schema=ActionSchemaConfig(action_dim=30, action_horizon=2, state_dim=16, state_horizon=1),
        action_mapping=mapping,
    )
    source = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    mapped = apply_action_mapping(source, torch.ones_like(source), mapping, target_dim=30)
    adapter = RobotwinBenchmarkAdapter(
        RobotwinEnvConfig(
            robotwin_root=str(tmp_path),
            task_name="dummy_task",
            task_config="dummy_task",
            action_type="qpos",
        )
    )

    env_action = adapter.model_action_to_env_action(mapped.actions[0].numpy(), data_config=config)

    assert env_action.shape == (14,)
    np.testing.assert_allclose(
        env_action,
        np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15], dtype=np.float32),
    )


def test_robotwin_adapter_extracts_state_matching_action_mode(tmp_path: Path) -> None:
    observation = {
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
        "endpose": {
            "left_endpose": list(np.arange(7, dtype=np.float32)),
            "left_gripper": 7.0,
            "right_endpose": list(np.arange(8, 15, dtype=np.float32)),
            "right_gripper": 15.0,
        },
    }
    ee_adapter = RobotwinBenchmarkAdapter(
        RobotwinEnvConfig(robotwin_root=str(tmp_path), task_name="dummy_task", task_config="dummy_task")
    )
    qpos_adapter = RobotwinBenchmarkAdapter(
        RobotwinEnvConfig(
            robotwin_root=str(tmp_path),
            task_name="dummy_task",
            task_config="dummy_task",
            action_type="qpos",
        )
    )

    ee_state = ee_adapter.extract_state(observation)
    qpos_state = qpos_adapter.extract_state(observation)

    assert ee_state is not None
    assert qpos_state is not None
    assert ee_state.shape == (16,)
    assert qpos_state.shape == (14,)
    np.testing.assert_allclose(ee_state, np.arange(16, dtype=np.float32))
    np.testing.assert_allclose(qpos_state, np.arange(14, dtype=np.float32))


def test_robotwin_ee_patch_skips_topp_planners_but_preserves_curobo_setup(monkeypatch) -> None:
    robot_module = types.ModuleType("envs.robot.robot")
    calls: list[tuple[bool, str]] = []

    class Robot:
        def __init__(self) -> None:
            self.need_topp = True
            self.left_planner = None
            self.right_planner = None

        def set_planner(self, scene=None) -> None:
            calls.append((self.need_topp, scene))
            self.left_planner = "left_curobo"
            self.right_planner = "right_curobo"
            if self.need_topp:
                raise AssertionError("EEF compatibility patch should disable TOPP during planner construction.")

    robot_module.Robot = Robot
    monkeypatch.setitem(sys.modules, "envs.robot.robot", robot_module)

    _install_ee_skip_topp_planner_patch()
    robot = Robot()
    robot.set_planner(scene="scene")

    assert calls == [(False, "scene")]
    assert robot.need_topp is True
    assert robot.left_planner == "left_curobo"
    assert robot.right_planner == "right_curobo"


class _FakeCalvinAdapter:
    benchmark_name = "fake_calvin"

    def __init__(self, *, success_after: int = 3) -> None:
        self.step_index = 0
        self.seen_env_actions: list[float] = []
        self.success_after = success_after

    def reset(self, *, task_id: int | None, episode_idx: int | None, seed: int | None) -> dict[str, Any]:
        del task_id, episode_idx, seed
        self.step_index = 0
        return self._obs()

    def task_text(self) -> str:
        return "slide the block"

    def extract_views(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        return {
            "rgb_static": observation["rgb_static"],
            "rgb_gripper": observation["rgb_gripper"],
        }

    def extract_state(self, observation: dict[str, Any]) -> np.ndarray:
        return observation["robot_obs"]

    def model_action_to_env_action(self, model_action: np.ndarray, *, data_config: CalvinDataConfig) -> np.ndarray:
        del data_config
        return np.asarray(model_action, dtype=np.float32)

    def step(self, env_action: np.ndarray) -> SimStepResult:
        self.seen_env_actions.append(float(env_action[0]))
        self.step_index += 1
        success = self.step_index >= self.success_after
        return SimStepResult(observation=self._obs(), done=success, info={"success": success})

    def success(self, observation: Any, info: dict[str, Any]) -> bool:
        del observation
        return bool(info.get("success", False))

    def render_frame(self, observation: dict[str, Any]) -> np.ndarray:
        return observation["rgb_static"]

    def close(self) -> None:
        return None

    def _obs(self) -> dict[str, np.ndarray]:
        value = np.uint8(self.step_index)
        return {
            "rgb_static": np.full((8, 8, 3), value, dtype=np.uint8),
            "rgb_gripper": np.full((4, 4, 3), value + 1, dtype=np.uint8),
            "robot_obs": np.full(15, float(self.step_index), dtype=np.float32),
        }


class _FakeNormalizedSimulator:
    benchmark_name = "fake_normalized"
    capabilities = SimulatorCapabilities(action_step_semantics="single_env_step")

    def __init__(self) -> None:
        self.step_index = 0
        self.seen_specs: list[EpisodeSpec] = []
        self.seen_env_actions: list[float] = []

    def reset(self, spec: EpisodeSpec) -> SimulatorObservation:
        self.step_index = 0
        self.seen_specs.append(spec)
        return self._obs()

    def task_text(self) -> str:
        return "normalized task"

    def action_from_model_action(self, model_action: np.ndarray, *, data_config: CalvinDataConfig) -> np.ndarray:
        del data_config
        return np.asarray(model_action, dtype=np.float32)

    def step(self, action: np.ndarray) -> SimulatorStepResult:
        self.seen_env_actions.append(float(action[0]))
        self.step_index += 1
        return SimulatorStepResult(
            observation=self._obs(),
            done=False,
            success=False,
            info={"step_index": self.step_index},
        )

    def render_frame(self, observation: SimulatorObservation) -> np.ndarray:
        return observation.views["rgb_static"]

    def close(self) -> None:
        return None

    def _obs(self) -> SimulatorObservation:
        value = np.uint8(self.step_index)
        views = {
            "rgb_static": np.full((8, 8, 3), value, dtype=np.uint8),
            "rgb_gripper": np.full((4, 4, 3), value + 1, dtype=np.uint8),
        }
        return SimulatorObservation(
            views=views,
            state=np.full(15, float(self.step_index), dtype=np.float32),
            task_text="normalized task",
            raw={"step_index": self.step_index},
        )


class _FakeRolloutRunner:
    def __init__(self, *, action_dim: int, action_horizon: int) -> None:
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.seen_frame_values: list[int] = []
        self.seen_previous_action_shapes: list[tuple[int, ...] | None] = []

    def reset(self, **_: Any) -> Any:
        return SimpleNamespace(policy_state=None)

    def infer_step(self, *, session: Any, context: Any, views: dict[str, torch.Tensor]) -> Any:
        previous_action = getattr(context, "previous_action", None)
        self.seen_previous_action_shapes.append(None if previous_action is None else tuple(previous_action.shape))
        frame_value = int(views["rgb_static"][0, -1, 0, 0, 0].item())
        self.seen_frame_values.append(frame_value)
        action_pred = torch.zeros(1, self.action_horizon, self.action_dim, dtype=torch.float32)
        action_pred[:, :, 0] = float(frame_value)
        return SimpleNamespace(
            session=session,
            infer_output=SimpleNamespace(decoder_output=SimpleNamespace(action_pred=action_pred)),
        )
