from __future__ import annotations

import builtins
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from open_wam.configs import (
    ActionMappingConfig,
    ActionNormalizationConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    CalvinDataConfig,
    LiberoAbsoluteJointExecutionMode,
    LiberoDataConfig,
    RobotWinDataConfig,
)
from open_wam.data.action_adapter import ConfiguredActionAdapter
from open_wam.data.action_mapping import apply_action_mapping
from open_wam.data.action_transforms import (
    PoseSequence,
    axis_angle_to_quaternion,
    quaternion_to_continuous_6d,
)
from open_wam.integrations.calvin_env import CalvinBenchmarkAdapter, CalvinEnvConfig
from open_wam.integrations.libero_env import (
    LiberoBenchmarkAdapter,
    LiberoEnvConfig,
    absolute_joint_position_to_libero_joint_delta_action,
    integrated_eef6d_target_to_osc_action,
    resolve_libero_joint_delta_limit,
    step_libero_absolute_joint_position_goal,
)
from open_wam.integrations.robotwin_env import (
    RobotwinBenchmarkAdapter,
    RobotwinEnvConfig,
    _install_ee_skip_topp_planner_patch,
)
import open_wam.integrations.robotwin_env as robotwin_env_module


def test_integrated_eef6d_target_recovers_osc_delta_action() -> None:
    position_scale = 0.01
    rotation_scale = 0.2
    previous = PoseSequence(
        position=torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32),
        quaternion=axis_angle_to_quaternion(torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float32))[0],
        gripper=torch.tensor([0.0], dtype=torch.float32),
    )
    source_action = torch.tensor([0.5, -0.25, 0.1, 0.0, 0.2, -0.1, 1.0], dtype=torch.float32)
    target_quaternion = axis_angle_to_quaternion((source_action[3:6] * rotation_scale).unsqueeze(0))[0]
    target_quaternion = torch.nn.functional.normalize(
        torch.tensor(
            [
                target_quaternion[3] * previous.quaternion[0]
                + target_quaternion[0] * previous.quaternion[3]
                + target_quaternion[1] * previous.quaternion[2]
                - target_quaternion[2] * previous.quaternion[1],
                target_quaternion[3] * previous.quaternion[1]
                - target_quaternion[0] * previous.quaternion[2]
                + target_quaternion[1] * previous.quaternion[3]
                + target_quaternion[2] * previous.quaternion[0],
                target_quaternion[3] * previous.quaternion[2]
                + target_quaternion[0] * previous.quaternion[1]
                - target_quaternion[1] * previous.quaternion[0]
                + target_quaternion[2] * previous.quaternion[3],
                target_quaternion[3] * previous.quaternion[3]
                - target_quaternion[0] * previous.quaternion[0]
                - target_quaternion[1] * previous.quaternion[1]
                - target_quaternion[2] * previous.quaternion[2],
            ],
            dtype=torch.float32,
        ),
        dim=0,
    )
    target = torch.cat(
        [
            previous.position + source_action[:3] * position_scale,
            quaternion_to_continuous_6d(target_quaternion.unsqueeze(0))[0],
            source_action[6:7],
        ],
        dim=0,
    )

    recovered, next_target = integrated_eef6d_target_to_osc_action(
        previous_target=previous,
        target=target.numpy(),
        position_scale=position_scale,
        rotation_scale=rotation_scale,
    )

    np.testing.assert_allclose(recovered, source_action.numpy(), atol=1e-4)
    np.testing.assert_allclose(next_target.position.numpy(), target[:3].numpy(), atol=1e-6)


def test_libero_integrated_eef_adapter_denormalizes_model_target() -> None:
    data_config = LiberoDataConfig(
        action_schema=ActionSchemaConfig(action_dim=10, action_horizon=1, state_dim=8, state_horizon=1),
        action_target=ActionTargetConfig(
            representation="raw",
            source_key="integrated_eef6d_action",
            normalization=ActionNormalizationConfig(
                mode="gaussian",
                mean=(0.1,) * 10,
                std=(2.0,) * 10,
            ),
        ),
    )
    adapter = LiberoBenchmarkAdapter(
        LiberoEnvConfig(
            action_mode="integrated_eef6d_osc",
            controller="OSC_POSE",
            integrated_eef_position_scale=0.01,
            integrated_eef_rotation_scale=0.2,
        )
    )
    state = np.asarray([0.1, -0.2, 0.3, 0.0, 0.0, 0.1, 0.04, -0.04], dtype=np.float32)
    adapter.set_integrated_eef6d_previous_target_from_state(state)
    target = np.asarray([0.105, -0.2025, 0.301, 1.0, 0.0, 0.0, 0.0, 0.9950042, 0.0998334, 1.0], dtype=np.float32)
    model_action = (target - 0.1) / 2.0

    recovered = _materialize_model_action(adapter, model_action, data_config)

    np.testing.assert_allclose(recovered[:3], np.asarray([0.5, -0.25, 0.1], dtype=np.float32), atol=1e-4)
    np.testing.assert_allclose(recovered[-1], 1.0, atol=1e-6)


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

    env_action = _materialize_model_action(adapter, mapped.actions[0].numpy(), config)

    assert env_action.shape == (7,)
    np.testing.assert_allclose(env_action[:6], source[0, :6].numpy())
    assert env_action[6] == 1.0


def test_calvin_adapter_reports_missing_benchmark_runtime(monkeypatch, tmp_path: Path) -> None:
    adapter = CalvinBenchmarkAdapter(
        CalvinEnvConfig(calvin_root=str(tmp_path), dataset_root=str(tmp_path))
    )
    real_import = builtins.__import__

    def fail_calvin_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "calvin_env" or name.startswith("calvin_env."):
            raise ModuleNotFoundError("No module named 'calvin_env'", name="calvin_env")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_calvin_import)

    with pytest.raises(ImportError, match=r"openwam\[calvin\].*--calvin-root"):
        adapter._build_env()


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

    env_action = _materialize_model_action(adapter, mapped.actions[0].numpy(), config)

    assert env_action.shape == (16,)
    np.testing.assert_allclose(env_action[0:3], source[0, 0:3].numpy())
    np.testing.assert_allclose(env_action[3:7], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(env_action[11:15], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert env_action[7] == source[0, 7].item()
    assert env_action[15] == source[0, 15].item()


def test_robotwin_adapter_reports_missing_benchmark_runtime(monkeypatch, tmp_path: Path) -> None:
    adapter = RobotwinBenchmarkAdapter(
        RobotwinEnvConfig(
            robotwin_root=str(tmp_path),
            task_name="dummy_task",
            task_config="dummy_task",
        )
    )

    def fail_task_import(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name="sapien")

    monkeypatch.setattr(robotwin_env_module.importlib, "import_module", fail_task_import)

    with pytest.raises(ImportError, match=r"openwam\[robotwin\].*--robotwin-root"):
        adapter._build_task_env("dummy_task")


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

    env_action = _materialize_model_action(adapter, mapped.actions[0].numpy(), config)

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


def test_libero_absolute_joint_position_action_normalizes_delta_and_gripper() -> None:
    action = absolute_joint_position_to_libero_joint_delta_action(
        target_joint_positions=np.asarray([0.025, -0.10, 0.20], dtype=np.float32),
        current_joint_positions=np.asarray([0.0, 0.0, 0.05], dtype=np.float32),
        gripper_command=2.0,
        joint_delta_limit_rad=np.asarray([0.05, 0.05, 0.10], dtype=np.float32),
    )

    np.testing.assert_allclose(action, np.asarray([0.5, -1.0, 1.0, 1.0], dtype=np.float32))


def test_libero_absolute_joint_position_goal_step_uses_set_qpos_hook() -> None:
    calls: list[tuple[np.ndarray, np.ndarray | None]] = []

    class Controller:
        control_dim = 3

        def set_goal(self, action: np.ndarray, *, set_qpos: np.ndarray | None = None) -> None:
            calls.append((np.asarray(action), None if set_qpos is None else np.asarray(set_qpos)))

    controller = Controller()
    robot = SimpleNamespace(controller=controller, action_dim=4)

    class Env:
        robots = [robot]

        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            controller.set_goal(action[:3])
            return {"ok": True}, 0.0, False, {"action": action.copy()}

    obs, _, done, info = step_libero_absolute_joint_position_goal(
        Env(),
        target_joint_positions=np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
        gripper_command=2.0,
    )

    assert obs == {"ok": True}
    assert done is False
    np.testing.assert_allclose(info["action"], np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(calls[0][1], np.asarray([0.1, -0.2, 0.3], dtype=np.float32))


def test_libero_adapter_denormalizes_absolute_joint_model_action() -> None:
    data_config = LiberoDataConfig(
        action_schema=ActionSchemaConfig(action_dim=3, action_horizon=1, state_dim=2, state_horizon=1),
        action_target=ActionTargetConfig(
            representation="absolute_joint_position",
            source_key="actions",
            include_gripper=True,
            gripper_representation="action_command",
            gripper_action_index=-1,
            joint_position_source_key="robot0_joint_pos",
            joint_position_normalization=ActionNormalizationConfig(
                mode="joint_limits",
                lower=(-2.0, -1.0),
                upper=(2.0, 3.0),
            ),
        ),
    )
    adapter = LiberoBenchmarkAdapter()
    adapter._last_obs = {"robot0_joint_pos": np.asarray([0.0, 1.0], dtype=np.float32)}
    adapter._joint_delta_limit = np.asarray([0.05, 0.10], dtype=np.float32)

    env_action = _materialize_model_action(adapter, np.asarray([0.5, -0.5, -0.25], dtype=np.float32), data_config)

    # Normalized qpos [0.5, -0.5] maps to physical qpos [1.0, 0.0].
    np.testing.assert_allclose(env_action, np.asarray([1.0, -1.0, -0.25], dtype=np.float32))


def test_libero_adapter_direct_goal_returns_target_and_steps_with_controller_hook() -> None:
    data_config = LiberoDataConfig(
        action_schema=ActionSchemaConfig(action_dim=3, action_horizon=1, state_dim=2, state_horizon=1),
        action_target=ActionTargetConfig(
            representation="absolute_joint_position",
            source_key="actions",
            include_gripper=True,
            gripper_representation="action_command",
            gripper_action_index=-1,
            joint_position_source_key="robot0_joint_pos",
            joint_position_normalization=ActionNormalizationConfig(
                mode="joint_limits",
                lower=(-2.0, -1.0),
                upper=(2.0, 3.0),
            ),
        ),
    )

    calls: list[tuple[np.ndarray, np.ndarray | None]] = []

    class Controller:
        control_dim = 2

        def set_goal(self, action: np.ndarray, *, set_qpos: np.ndarray | None = None) -> None:
            calls.append((np.asarray(action).copy(), None if set_qpos is None else np.asarray(set_qpos).copy()))

    controller = Controller()
    robot = SimpleNamespace(controller=controller, action_dim=3)

    class Env:
        robots = [robot]

        def __init__(self) -> None:
            self.step_count = 0

        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            self.step_count += 1
            controller.set_goal(action[:2])
            return {"robot0_joint_pos": np.asarray([1.0, 0.0], dtype=np.float32)}, 0.0, False, {}

        def check_success(self) -> bool:
            return False

    adapter = LiberoBenchmarkAdapter(
        LiberoEnvConfig(
            controller="JOINT_POSITION",
            action_mode="absolute_joint_position",
            absolute_joint_execution_mode=LiberoAbsoluteJointExecutionMode.DIRECT_GOAL,
            absolute_joint_substeps_per_target=2,
            absolute_joint_gripper_substep_policy="first_only",
        )
    )
    adapter._env = Env()
    adapter._last_obs = {"robot0_joint_pos": np.asarray([0.0, 1.0], dtype=np.float32)}

    env_action = _materialize_model_action(adapter, np.asarray([0.5, -0.5, -0.25], dtype=np.float32), data_config)
    np.testing.assert_allclose(env_action, np.asarray([1.0, 0.0, -0.25], dtype=np.float32))

    transition = adapter.step(env_action)

    assert transition.done is False
    assert transition.info["absolute_joint_execution_mode"] == "direct_goal"
    assert transition.info["absolute_joint_env_substeps"] == 2
    assert len(calls) == 2
    np.testing.assert_allclose(calls[0][1], np.asarray([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(calls[1][1], np.asarray([1.0, 0.0], dtype=np.float32))


def test_libero_adapter_normalized_delta_substeps_recompute_from_current_qpos() -> None:
    data_config = LiberoDataConfig(
        action_schema=ActionSchemaConfig(action_dim=3, action_horizon=1, state_dim=2, state_horizon=1),
        action_target=ActionTargetConfig(
            representation="absolute_joint_position",
            source_key="actions",
            include_gripper=True,
            gripper_representation="action_command",
            gripper_action_index=-1,
            joint_position_source_key="robot0_joint_pos",
            joint_position_normalization=ActionNormalizationConfig(
                mode="joint_limits",
                lower=(-2.0, -1.0),
                upper=(2.0, 3.0),
            ),
        ),
    )

    actions: list[np.ndarray] = []

    class Env:
        def __init__(self) -> None:
            self.qpos = np.asarray([0.0, 1.0], dtype=np.float32)

        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            actions.append(np.asarray(action, dtype=np.float32).copy())
            self.qpos = self.qpos + np.asarray(action[:2], dtype=np.float32) * 0.5
            return {"robot0_joint_pos": self.qpos.copy()}, 0.0, False, {}

        def check_success(self) -> bool:
            return False

    adapter = LiberoBenchmarkAdapter(
        LiberoEnvConfig(
            controller="JOINT_POSITION",
            action_mode="absolute_joint_position",
            absolute_joint_execution_mode=LiberoAbsoluteJointExecutionMode.NORMALIZED_DELTA,
            absolute_joint_substeps_per_target=2,
            absolute_joint_gripper_substep_policy="repeat",
            joint_delta_limit_rad=0.5,
        )
    )
    adapter._env = Env()
    adapter._last_obs = {"robot0_joint_pos": np.asarray([0.0, 1.0], dtype=np.float32)}
    adapter._joint_delta_limit = np.asarray([0.5, 0.5], dtype=np.float32)

    env_action = _materialize_model_action(adapter, np.asarray([0.5, -0.5, -0.25], dtype=np.float32), data_config)
    np.testing.assert_allclose(env_action, np.asarray([1.0, 0.0, -0.25], dtype=np.float32))

    transition = adapter.step(env_action)

    assert transition.info["absolute_joint_execution_mode"] == "normalized_delta"
    assert transition.info["absolute_joint_env_substeps"] == 2
    assert len(actions) == 2
    np.testing.assert_allclose(actions[0], np.asarray([1.0, -1.0, -0.25], dtype=np.float32))
    np.testing.assert_allclose(actions[1], np.asarray([1.0, -1.0, -0.25], dtype=np.float32))


def test_libero_adapter_integrated_delta_differences_previous_targets() -> None:
    data_config = LiberoDataConfig(
        action_schema=ActionSchemaConfig(action_dim=3, action_horizon=1, state_dim=2, state_horizon=1),
        action_target=ActionTargetConfig(
            representation="absolute_joint_position",
            source_key="actions",
            include_gripper=True,
            gripper_representation="action_command",
            gripper_action_index=-1,
            joint_position_source_key="robot0_joint_pos",
            joint_position_normalization=ActionNormalizationConfig(mode="none"),
        ),
    )

    actions: list[np.ndarray] = []

    class Env:
        def __init__(self) -> None:
            self.qpos = np.asarray([0.0, 1.0], dtype=np.float32)

        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            actions.append(np.asarray(action, dtype=np.float32).copy())
            # Deliberately do not track the pseudo target exactly. Integrated
            # mode should recover deltas from target differences, not sim qpos.
            self.qpos = self.qpos + np.asarray([0.01, 0.01], dtype=np.float32)
            return {"robot0_joint_pos": self.qpos.copy()}, 0.0, False, {}

        def check_success(self) -> bool:
            return False

    adapter = LiberoBenchmarkAdapter(
        LiberoEnvConfig(
            controller="JOINT_POSITION",
            action_mode="absolute_joint_position",
            absolute_joint_execution_mode=LiberoAbsoluteJointExecutionMode.INTEGRATED_DELTA,
            absolute_joint_delta_integration_scale=0.5,
        )
    )
    adapter._env = Env()
    adapter._last_obs = {"robot0_joint_pos": np.asarray([0.0, 1.0], dtype=np.float32)}
    adapter._joint_delta_limit = np.asarray([0.5, 0.5], dtype=np.float32)
    adapter._absolute_joint_previous_target_qpos = np.asarray([0.0, 1.0], dtype=np.float32)

    first = _materialize_model_action(adapter, np.asarray([0.5, 0.5, -1.0], dtype=np.float32), data_config)
    adapter.step(first)
    second = _materialize_model_action(adapter, np.asarray([0.75, 0.0, 1.0], dtype=np.float32), data_config)
    adapter.step(second)

    assert len(actions) == 2
    np.testing.assert_allclose(actions[0], np.asarray([1.0, -1.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(actions[1], np.asarray([0.5, -1.0, 1.0], dtype=np.float32))


def test_libero_adapter_integrated_delta_allows_negative_dataset_scale() -> None:
    data_config = LiberoDataConfig(
        action_schema=ActionSchemaConfig(action_dim=3, action_horizon=1, state_dim=2, state_horizon=1),
        action_target=ActionTargetConfig(
            representation="absolute_joint_position",
            source_key="actions",
            include_gripper=True,
            gripper_representation="action_command",
            joint_position_source_key="robot0_joint_pos",
            joint_position_normalization=ActionNormalizationConfig(mode="none"),
        ),
    )
    actions: list[np.ndarray] = []

    class Env:
        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
            actions.append(np.asarray(action, dtype=np.float32).copy())
            return {"robot0_joint_pos": np.asarray([0.0, 0.0], dtype=np.float32)}, 0.0, False, {}

        def check_success(self) -> bool:
            return False

    adapter = LiberoBenchmarkAdapter(
        LiberoEnvConfig(
            controller="JOINT_POSITION",
            action_mode="absolute_joint_position",
            absolute_joint_execution_mode=LiberoAbsoluteJointExecutionMode.INTEGRATED_DELTA,
            absolute_joint_delta_integration_scale=-0.5,
        )
    )
    adapter._env = Env()
    adapter._last_obs = {"robot0_joint_pos": np.asarray([0.0, 0.0], dtype=np.float32)}
    adapter._absolute_joint_previous_target_qpos = np.asarray([0.0, 0.0], dtype=np.float32)

    action = _materialize_model_action(adapter, np.asarray([-0.5, 0.25, 0.0], dtype=np.float32), data_config)
    adapter.step(action)

    np.testing.assert_allclose(actions[0], np.asarray([1.0, -0.5, 0.0], dtype=np.float32))


def test_libero_adapter_absolute_joint_can_track_measured_gripper_qpos() -> None:
    data_config = LiberoDataConfig(
        action_schema=ActionSchemaConfig(action_dim=3, action_horizon=1, state_dim=2, state_horizon=1),
        action_target=ActionTargetConfig(
            representation="absolute_joint_position",
            source_key="actions",
            include_gripper=True,
            gripper_representation="first_channel",
            joint_position_source_key="robot0_joint_pos",
            gripper_position_source_key="robot0_gripper_qpos",
            joint_position_normalization=ActionNormalizationConfig(
                mode="joint_limits",
                lower=(-2.0, -1.0),
                upper=(2.0, 3.0),
            ),
        ),
    )
    adapter = LiberoBenchmarkAdapter(
        LiberoEnvConfig(
            controller="JOINT_POSITION",
            action_mode="absolute_joint_position",
            absolute_joint_execution_mode="direct_goal",
        )
    )
    adapter._last_obs = {
        "robot0_joint_pos": np.asarray([0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.04, -0.04], dtype=np.float32),
    }

    env_action = _materialize_model_action(adapter, np.asarray([0.5, -0.5, 0.0], dtype=np.float32), data_config)

    # Direct-goal mode preserves measured gripper qpos targets so step() can
    # recompute open/close commands from the current gripper state per substep.
    np.testing.assert_allclose(env_action, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_libero_joint_delta_limit_reads_robosuite_controller_limits() -> None:
    controller = SimpleNamespace(
        output_max=np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
        output_min=np.asarray([-0.02, -0.01, -0.04, -0.03], dtype=np.float32),
    )
    env = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(controller=controller)]))

    limit = resolve_libero_joint_delta_limit(env, joint_dim=4)

    np.testing.assert_allclose(limit, np.asarray([0.02, 0.02, 0.04, 0.04], dtype=np.float32))


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

def _materialize_model_action(backend, model_action, data_config):
    mapping = ConfiguredActionAdapter(
        data_config.action_mapping, data_config.action_target.normalization,
        data_config.action_schema.action_dim,
    )
    source = mapping.to_source(torch.as_tensor(model_action)).numpy()
    return backend.materialize_control(source, data_config=data_config).action
