from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
import pytest
import sys
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "process_libero10_absolute_joint_dataset.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("process_libero10_absolute_joint_dataset", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)
_joint_position_command_for_substep = SCRIPT_MODULE._joint_position_command_for_substep
_build_absolute_joint_actions = SCRIPT_MODULE._build_absolute_joint_actions
_process_one_episode = SCRIPT_MODULE._process_one_episode
AbsoluteReplayProfile = SCRIPT_MODULE.AbsoluteReplayProfile
integrate_delta_joint_positions = SCRIPT_MODULE.integrate_delta_joint_positions
fit_integrated_delta_scale = SCRIPT_MODULE.fit_integrated_delta_scale


def test_joint_substep_hold_commands_target_immediately() -> None:
    previous = np.asarray([0.0, 2.0], dtype=np.float32)
    target = np.asarray([4.0, -2.0], dtype=np.float32)

    command = _joint_position_command_for_substep(
        previous_target_positions=previous,
        target_joint_positions=target,
        substep_index=0,
        substeps=4,
        policy="hold",
    )

    np.testing.assert_allclose(command, target)


def test_joint_substep_linear_reaches_target_on_final_substep() -> None:
    previous = np.asarray([0.0, 2.0], dtype=np.float32)
    target = np.asarray([4.0, -2.0], dtype=np.float32)

    first = _joint_position_command_for_substep(
        previous_target_positions=previous,
        target_joint_positions=target,
        substep_index=0,
        substeps=4,
        policy="linear",
    )
    final = _joint_position_command_for_substep(
        previous_target_positions=previous,
        target_joint_positions=target,
        substep_index=3,
        substeps=4,
        policy="linear",
    )

    np.testing.assert_allclose(first, np.asarray([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(final, target)


def test_joint_substep_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unknown joint substep policy"):
        _joint_position_command_for_substep(
            previous_target_positions=np.zeros(2, dtype=np.float32),
            target_joint_positions=np.ones(2, dtype=np.float32),
            substep_index=0,
            substeps=1,
            policy="bad",
        )


def test_absolute_joint_sidecar_targets_can_use_raw_gripper_command() -> None:
    data_config = SimpleNamespace(
        action_target=SimpleNamespace(
            gripper_representation="action_command",
            gripper_action_index=-1,
        )
    )

    actions = _build_absolute_joint_actions(
        joint_positions_after_action=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        gripper_positions_after_action=np.asarray([[0.01, -0.01], [0.02, -0.02]], dtype=np.float32),
        raw_osc_actions=np.asarray([[1.0, 2.0, -1.0], [3.0, 4.0, 1.0]], dtype=np.float32),
        rollout_data_config=data_config,
    )

    np.testing.assert_allclose(
        actions,
        np.asarray([[0.1, 0.2, -1.0], [0.3, 0.4, 1.0]], dtype=np.float32),
    )


def test_absolute_joint_sidecar_targets_can_use_measured_gripper_qpos() -> None:
    data_config = SimpleNamespace(
        action_target=SimpleNamespace(
            gripper_representation="all_channels",
            gripper_action_index=-1,
        )
    )

    actions = _build_absolute_joint_actions(
        joint_positions_after_action=np.asarray([[0.1, 0.2]], dtype=np.float32),
        gripper_positions_after_action=np.asarray([[0.01, -0.01]], dtype=np.float32),
        raw_osc_actions=np.asarray([[1.0, 2.0, -1.0]], dtype=np.float32),
        rollout_data_config=data_config,
    )

    np.testing.assert_allclose(actions, np.asarray([[0.1, 0.2, 0.01, -0.01]], dtype=np.float32))


def test_integrated_delta_joint_positions_round_trip_with_fixed_scale() -> None:
    initial = np.asarray([0.0, 1.0], dtype=np.float32)
    deltas = np.asarray([[1.0, -1.0, 0.0], [0.5, 0.25, 0.0]], dtype=np.float32)

    targets = integrate_delta_joint_positions(
        initial_joint_positions=initial,
        delta_actions=deltas,
        scale=0.2,
        joint_dim=2,
    )

    np.testing.assert_allclose(
        targets,
        np.asarray([[0.2, 0.8], [0.3, 0.85]], dtype=np.float32),
    )


def test_integrated_delta_scale_fit_matches_measured_qpos() -> None:
    initial = np.asarray([1.0, -1.0], dtype=np.float32)
    deltas = np.asarray([[1.0, 0.5], [0.5, -1.0], [-0.25, 0.25]], dtype=np.float32)
    measured = integrate_delta_joint_positions(
        initial_joint_positions=initial,
        delta_actions=deltas,
        scale=0.07,
        joint_dim=2,
    )

    scale = fit_integrated_delta_scale(
        initial_joint_positions=initial,
        delta_actions=deltas,
        measured_joint_positions=measured,
    )

    assert scale == pytest.approx(0.07)


def test_process_integrated_delta_writes_target_positions_to_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = {
        "success": True,
        "steps": 2,
        "initial_joint_positions": np.asarray([0.0, 0.0], dtype=np.float32),
        "joint_positions": np.asarray([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
        "gripper_positions": np.asarray([[0.01, -0.01], [0.02, -0.02]], dtype=np.float32),
        "raw_actions": np.asarray([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]], dtype=np.float32),
    }
    expected_targets = integrate_delta_joint_positions(
        initial_joint_positions=raw["initial_joint_positions"],
        delta_actions=raw["raw_actions"],
        scale=0.1,
        joint_dim=2,
    )

    monkeypatch.setattr(SCRIPT_MODULE, "_load_episode_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(SCRIPT_MODULE, "_task_resources", lambda *_args, **_kwargs: (object(), [object()]))
    monkeypatch.setattr(SCRIPT_MODULE, "_resolved_reset_seed", lambda _row: 123)

    def fake_absolute_rollout(**kwargs: object) -> dict[str, object]:
        np.testing.assert_allclose(kwargs["target_joint_positions"], expected_targets)
        return {
            "absolute_control_freq": 20,
            "absolute_control_frequency_mode": "raw_control_freq",
            "absolute_tracking_mode": "joint_position",
            "rollout_adapter_execution_mode": "normalized_delta",
            "absolute_gripper_substep_policy": "hold",
            "absolute_gripper_tracking_mode": "measured_qpos",
            "absolute_gripper_target_delay_steps": 0,
            "absolute_joint_substep_policy": "hold",
            "absolute_joint_settle_l2_tolerance": None,
            "absolute_joint_kp": None,
            "absolute_disable_interpolator": True,
            "success": True,
            "steps": 2,
            "success_grace_steps_taken": 0,
            "qpos_l2_errors": np.zeros(2, dtype=np.float32),
            "qpos_linf_errors": np.zeros(2, dtype=np.float32),
        }

    monkeypatch.setattr(SCRIPT_MODULE, "_rollout_absolute_control_env", fake_absolute_rollout)
    row = {
        "dataset_episode_index": 0,
        "upstream_task_id": 0,
        "resolved_init_state_index": 0,
        "attempts": [{"init_state_index": 0, "reset_seed": 123, "success_step": 1}],
    }
    data_config = SimpleNamespace(
        action_target=SimpleNamespace(
            gripper_representation="action_command",
            gripper_action_index=-1,
        )
    )
    sidecar_path = tmp_path / "episode_000000_abs_joint.npz"
    profile = AbsoluteReplayProfile(
        name="test",
        tracking_mode="joint_position",
        control_frequency_mode="raw_control_freq",
        gripper_substep_policy="hold",
        gripper_tracking_mode="measured_qpos",
        gripper_target_delay_steps=0,
        joint_substep_policy="hold",
        joint_settle_l2_tolerance=None,
        joint_kp=None,
        disable_interpolator=True,
    )

    record = _process_one_episode(
        row=row,
        dataset_root=tmp_path,
        episode_records=[],
        sidecar_path=sidecar_path,
        task_cache={},
        substeps_sweep=[1],
        camera_height=8,
        camera_width=8,
        camera_key="agentview_image",
        raw_stop_on_success=False,
        absolute_stop_on_success=False,
        absolute_profiles=[profile],
        absolute_success_grace_steps=0,
        rollout_data_config=data_config,
        absolute_joint_target_source="integrated_delta",
        integrated_delta_scale=0.1,
        raw_replay_cache={0: raw},
        warmup_override=0,
    )

    assert record["conversion_status"] == "success"
    with np.load(sidecar_path) as sidecar:
        np.testing.assert_allclose(sidecar["joint_positions_after_action"], expected_targets)
        assert not np.allclose(sidecar["joint_positions_after_action"], raw["joint_positions"])
