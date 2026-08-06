from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import yaml

import open_wam.integrations as integrations
from open_wam.data.action_transforms import PoseSequence, axis_angle_to_quaternion
from open_wam.integrations import (
    libero_control,
    libero_env,
    libero_joint_control,
    libero_observations,
    libero_osc_control,
    libero_runtime,
    simulator_configs,
    libero_tasks,
    libero_tracking,
)


def test_libero_task_contract_has_one_canonical_owner() -> None:
    assert integrations.LiberoTaskSpec is libero_tasks.LiberoTaskSpec
    assert libero_env.LiberoTaskSpec is libero_tasks.LiberoTaskSpec
    assert (
        integrations.resolve_libero_task_by_id
        is libero_tasks.resolve_libero_task_by_id
    )
    assert (
        libero_env.resolve_libero_task_by_id
        is libero_tasks.resolve_libero_task_by_id
    )


def test_libero_runtime_contracts_have_one_canonical_owner() -> None:
    control_owners = {
        "LiberoControlConfig": simulator_configs,
        "absolute_joint_position_to_libero_joint_delta_action": (
            libero_joint_control
        ),
        "compute_osc_pose_action": libero_osc_control,
        "disable_libero_joint_position_controller_interpolator": (
            libero_joint_control
        ),
        "extract_gripper_positions_from_obs": libero_observations,
        "extract_joint_positions_from_obs": libero_observations,
        "extract_pose_from_obs": libero_observations,
        "integrated_eef6d_target_to_osc_action": libero_osc_control,
        "resolve_libero_joint_delta_limit": libero_joint_control,
        "set_libero_joint_position_controller_gain": libero_joint_control,
        "step_libero_absolute_joint_position_goal": libero_joint_control,
    }
    for name, owner in control_owners.items():
        canonical = getattr(owner, name)
        assert getattr(libero_control, name) is canonical
        assert getattr(integrations, name) is canonical
        assert getattr(libero_env, name) is canonical
    assert (
        libero_env.quaternion_angular_error_degrees
        is libero_osc_control.quaternion_angular_error_degrees
    )

    runtime_exports = (
        "build_libero_control_env",
        "build_libero_offscreen_env",
    )
    for name in runtime_exports:
        canonical = getattr(libero_runtime, name)
        assert getattr(integrations, name) is canonical
        assert getattr(libero_env, name) is canonical

    tracking_exports = (
        "LiberoTrackingResult",
        "track_relative_targets_in_libero_env",
    )
    for name in tracking_exports:
        canonical = getattr(libero_tracking, name)
        assert getattr(integrations, name) is canonical
        assert getattr(libero_env, name) is canonical


def test_libero_control_math_matches_frozen_outputs() -> None:
    current_pose = PoseSequence(
        position=torch.tensor([0.0, 0.0, 0.0]),
        quaternion=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        gripper=torch.tensor([0.03, -0.03]),
    )
    desired_quaternion = axis_angle_to_quaternion(
        torch.tensor([[0.0, 0.0, 0.25]])
    )[0]
    expected_prefix = [
        0.5,
        -1.0,
        0.19999998807907104,
        0.0,
        0.0,
        0.4999999701976776,
    ]
    cases = (
        ("first_channel", [0.04], -1.0),
        ("all_channels", [0.01, -0.01], 1.0),
        ("action_command", [1.4], 1.0),
    )

    for representation, gripper, expected_command in cases:
        desired_pose = PoseSequence(
            position=torch.tensor([0.025, -0.1, 0.01]),
            quaternion=desired_quaternion,
            gripper=torch.tensor(gripper),
        )
        action = libero_control.compute_osc_pose_action(
            current_pose=current_pose,
            desired_pose=desired_pose,
            control_config=libero_control.LiberoControlConfig(),
            gripper_representation=representation,
        )
        np.testing.assert_array_equal(
            action,
            np.asarray([*expected_prefix, expected_command], dtype=np.float32),
        )

    matrix = libero_control.quaternion_xyzw_to_rotation_matrix(
        np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        matrix,
        np.asarray(
            [
                [0.7263157367706299, -0.6105263233184814, -0.31578949093818665],
                [0.5263158082962036, 0.7894736528396606, -0.31578949093818665],
                [0.4421052634716034, 0.06315788626670837, 0.8947368264198303],
            ],
            dtype=np.float32,
        ),
    )


def test_libero_control_policies_and_controller_mutations() -> None:
    assert [
        libero_control.gripper_command_for_substep(
            0.25,
            substep_index=index,
            substeps=3,
            policy="first_only",
        )
        for index in range(3)
    ] == [0.25, 0.0, 0.0]
    assert [
        libero_control.gripper_command_for_substep(
            0.25,
            substep_index=index,
            substeps=3,
            policy="last_only",
        )
        for index in range(3)
    ] == [0.0, 0.0, 0.25]
    assert libero_control.gripper_qpos_tracking_command(
        current_gripper_positions=np.asarray([0.04, -0.04]),
        target_gripper_positions=np.asarray([0.03, -0.03]),
    ) == 1.0

    controller = SimpleNamespace(control_dim=3, interpolator=object())
    env = SimpleNamespace(robots=[SimpleNamespace(controller=controller)])
    libero_control.set_libero_joint_position_controller_gain(env, kp=9.0)
    libero_control.disable_libero_joint_position_controller_interpolator(env)

    np.testing.assert_array_equal(controller.kp, np.full(3, 9.0, dtype=np.float64))
    np.testing.assert_array_equal(controller.kd, np.full(3, 6.0, dtype=np.float64))
    assert controller.interpolator is None


def test_libero_runtime_factories_forward_explicit_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    bootstrapped_roots: list[Path | None] = []
    libero_package = ModuleType("libero")
    libero_package.__path__ = []  # type: ignore[attr-defined]
    libero_subpackage = ModuleType("libero.libero")
    libero_subpackage.__path__ = []  # type: ignore[attr-defined]
    envs_module = ModuleType("libero.libero.envs")
    envs_module.__path__ = []  # type: ignore[attr-defined]
    wrapper_module = ModuleType("libero.libero.envs.env_wrapper")

    def _offscreen_factory(**kwargs):
        calls.append(("offscreen", kwargs))
        return "offscreen-env"

    def _control_factory(**kwargs):
        calls.append(("control", kwargs))
        return "control-env"

    envs_module.OffScreenRenderEnv = _offscreen_factory
    wrapper_module.ControlEnv = _control_factory
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_subpackage)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_module)
    monkeypatch.setitem(
        sys.modules,
        "libero.libero.envs.env_wrapper",
        wrapper_module,
    )
    monkeypatch.setattr(
        libero_runtime,
        "ensure_local_libero_config",
        lambda project_root=None: bootstrapped_roots.append(project_root),
    )
    task_spec = libero_tasks.LiberoTaskSpec(
        benchmark_name="libero_10",
        task_id=0,
        task_name="task",
        task_language="task text",
        problem_folder="suite",
        bddl_file_path="/tmp/task.bddl",
        init_states_path="/tmp/task.init",
    )

    offscreen = libero_runtime.build_libero_offscreen_env(
        task_spec,
        controller="JOINT_POSITION",
        camera_height=96,
        camera_width=160,
        horizon=123,
        ignore_done=False,
        control_freq=17,
        project_root=tmp_path,
    )
    control = libero_runtime.build_libero_control_env(
        task_spec,
        controller="OSC_POSE",
        camera_height=80,
        camera_width=120,
        horizon=456,
        ignore_done=True,
        control_freq=11,
        use_camera_obs=True,
        has_offscreen_renderer=True,
        project_root=tmp_path,
    )

    assert offscreen == "offscreen-env"
    assert control == "control-env"
    assert bootstrapped_roots == [tmp_path, tmp_path]
    assert calls == [
        (
            "offscreen",
            {
                "bddl_file_name": "/tmp/task.bddl",
                "controller": "JOINT_POSITION",
                "camera_heights": 96,
                "camera_widths": 160,
                "horizon": 123,
                "ignore_done": False,
                "control_freq": 17,
            },
        ),
        (
            "control",
            {
                "bddl_file_name": "/tmp/task.bddl",
                "controller": "OSC_POSE",
                "use_camera_obs": True,
                "has_offscreen_renderer": True,
                "has_renderer": False,
                "camera_heights": 80,
                "camera_widths": 120,
                "horizon": 456,
                "ignore_done": True,
                "control_freq": 11,
            },
        ),
    ]


def test_libero_tracking_preserves_target_and_substep_alignment(monkeypatch) -> None:
    task_spec = libero_tasks.LiberoTaskSpec(
        benchmark_name="libero_10",
        task_id=0,
        task_name="task",
        task_language="task text",
        problem_folder="suite",
        bddl_file_path="/tmp/task.bddl",
        init_states_path="/tmp/task.init",
    )

    class FakeEnv:
        def __init__(self) -> None:
            self.closed = False
            self.obs = {
                "robot0_eef_pos": np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
                "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "robot0_gripper_qpos": np.asarray([0.03, -0.03], dtype=np.float32),
                "agentview_image": np.full((4, 5, 3), 17, dtype=np.uint8),
                "robot0_eye_in_hand_image": np.full((4, 5, 3), 29, dtype=np.uint8),
            }

        def reset(self):
            return self.obs

        def set_init_state(self, state):
            return self.obs

        def step(self, action):
            return self.obs, 0.0, False, {}

        def close(self) -> None:
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(libero_tracking, "resolve_libero_task", lambda *args, **kwargs: task_spec)
    monkeypatch.setattr(
        libero_tracking,
        "load_libero_task_init_states",
        lambda *args, **kwargs: [torch.zeros(1), torch.ones(1)],
    )
    monkeypatch.setattr(
        libero_tracking,
        "build_libero_offscreen_env",
        lambda *args, **kwargs: env,
    )

    result = libero_tracking.track_relative_targets_in_libero_env(
        task_text="task text",
        relative_pose_targets=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.1, 0.04],
            ],
            dtype=torch.float32,
        ),
        rotation_representation="axis_angle",
        reference_position=torch.tensor([0.1, -0.2, 0.3]),
        reference_quaternion=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        gripper_representation="first_channel",
        init_state_index=3,
        control_config=libero_control.LiberoControlConfig(
            control_substeps_per_target=2
        ),
    )

    assert result.task_spec is task_spec
    assert result.init_state_index == 1
    assert result.rendered_target_indices == [0, 0, 1, 1]
    torch.testing.assert_close(
        result.desired_pose.position,
        torch.tensor([[0.1, -0.2, 0.3], [0.11, -0.2, 0.3]]),
    )
    torch.testing.assert_close(
        result.tracked_pose.position,
        torch.tensor([[0.1, -0.2, 0.3], [0.1, -0.2, 0.3]]),
    )
    torch.testing.assert_close(
        result.position_error_per_target,
        torch.tensor([0.0, 0.01]),
    )
    torch.testing.assert_close(
        result.gripper_error_per_target,
        torch.tensor([0.01, 0.01]),
    )
    assert [int(frame[0, 0, 0]) for frame in result.camera_frames["agentview_image"]] == [17] * 4
    assert env.closed is True


def test_ensure_local_libero_config_writes_deterministic_checkout_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "LIBERO"
    package_root = repo_root / "libero" / "libero"
    package_root.mkdir(parents=True)
    monkeypatch.setattr(
        libero_tasks,
        "_resolve_libero_paths",
        lambda: (repo_root, package_root),
    )
    monkeypatch.setenv("LIBERO_CONFIG_PATH", "original-config")
    project_root = tmp_path / "Open-WAM"

    config_path = libero_tasks.ensure_local_libero_config(project_root)
    first_bytes = config_path.read_bytes()
    second_path = libero_tasks.ensure_local_libero_config(project_root)

    assert second_path == config_path
    assert second_path.read_bytes() == first_bytes
    assert yaml.safe_load(first_bytes) == {
        "benchmark_root": str(package_root.resolve()),
        "bddl_files": str((package_root / "bddl_files").resolve()),
        "init_states": str((package_root / "init_files").resolve()),
        "datasets": str((repo_root / "libero" / "datasets").resolve()),
        "assets": str((package_root / "assets").resolve()),
    }


def test_infer_task_local_episode_rank_normalizes_task_identity() -> None:
    records = (
        SimpleNamespace(episode_index=3, tasks=("Pick   Cup",)),
        SimpleNamespace(episode_index=7, tasks=("pick cup",)),
        SimpleNamespace(episode_index=9, tasks=("other",)),
    )

    assert libero_tasks.infer_task_local_episode_rank(
        records,
        episode_index=7,
        task_text=" PICK CUP ",
    ) == 1


def test_load_libero_task_init_states_uses_restricted_numpy_compatibility(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task_spec = libero_tasks.LiberoTaskSpec(
        benchmark_name="libero_10",
        task_id=2,
        task_name="put_cup_away",
        task_language="put cup away",
        problem_folder="suite",
        bddl_file_path="/tasks/put_cup_away.bddl",
        init_states_path=str(tmp_path / "put_cup_away.pruned_init"),
    )
    monkeypatch.setattr(
        libero_tasks,
        "ensure_local_libero_config",
        lambda project_root=None: tmp_path / "config.yaml",
    )

    expected = [np.arange(6, dtype=np.float32).reshape(2, 3)]
    torch.save(expected, task_spec.init_states_path)

    result = libero_tasks.load_libero_task_init_states(task_spec)

    assert isinstance(result, list)
    assert np.array_equal(result[0], expected[0])


def test_resolve_libero_paths_uses_fallback_checkout_without_import(monkeypatch, tmp_path: Path) -> None:
    checkout_root = tmp_path / "LIBERO"
    package_root = checkout_root / "libero" / "libero"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    def _raise_import_error(name: str):
        raise EOFError("non-interactive import prompt")

    monkeypatch.setattr(libero_tasks.importlib, "import_module", _raise_import_error)
    monkeypatch.setattr(
        libero_tasks,
        "_project_root",
        lambda _: tmp_path / "Open-WAM",
    )

    resolved_repo_root, resolved_package_root = (
        libero_tasks._resolve_libero_paths()
    )

    assert resolved_repo_root == checkout_root.resolve()
    assert resolved_package_root == package_root.resolve()
    assert str(checkout_root.resolve()) in libero_tasks.sys.path


def test_resolve_libero_paths_reraises_non_libero_module_errors(monkeypatch) -> None:
    def _raise_internal_module_error(name: str):
        raise ModuleNotFoundError("missing dependency", name="robosuite")

    monkeypatch.setattr(
        libero_tasks.importlib,
        "import_module",
        _raise_internal_module_error,
    )

    try:
        libero_tasks._resolve_libero_paths()
    except ModuleNotFoundError as exc:
        assert exc.name == "robosuite"
    else:
        raise AssertionError("Expected internal ModuleNotFoundError to be re-raised.")
