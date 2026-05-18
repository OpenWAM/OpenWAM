from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_libero_integrated_eef_scale.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("calibrate_libero_integrated_eef_scale", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

build_integrated_eef_targets = SCRIPT_MODULE.build_integrated_eef_targets
recover_osc_actions_from_integrated_eef_targets = SCRIPT_MODULE.recover_osc_actions_from_integrated_eef_targets
fit_integrated_eef_scales = SCRIPT_MODULE.fit_integrated_eef_scales
integrate_action_rotation_matrices = SCRIPT_MODULE._integrate_action_rotation_matrices
rotation_matrix_to_axis_angle = SCRIPT_MODULE._rotation_matrix_to_axis_angle_np


def test_integrated_eef_targets_recover_source_osc_actions() -> None:
    initial_state = np.asarray([0.1, -0.2, 0.3, 3.1, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)
    actions = np.asarray(
        [
            [0.5, -0.25, 0.0, 0.1, 0.0, -0.2, -1.0],
            [-0.1, 0.0, 0.25, 0.0, 0.3, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    targets = build_integrated_eef_targets(
        initial_state=initial_state,
        actions=actions,
        position_scale=0.01,
        rotation_scale=0.5,
    )
    recovered = recover_osc_actions_from_integrated_eef_targets(
        initial_state=initial_state,
        targets=targets,
        position_scale=0.01,
        rotation_scale=0.5,
    )

    assert targets.shape == (2, 10)
    np.testing.assert_allclose(recovered, actions, atol=1e-5)


def test_integrated_eef_scale_fit_uses_dataset_wide_linear_least_squares() -> None:
    actions = np.asarray(
        [
            [1.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0, -0.5, 0.25, 0.0],
            [-0.25, 0.0, 0.25, 0.1, 0.0, -0.1, 0.0],
        ],
        dtype=np.float64,
    )
    initial = np.asarray([0.0, 1.0, -1.0, 3.0, 0.1, -0.2, 0.0, 0.0], dtype=np.float64)
    position_scale = 0.02
    rotation_scale = -0.1
    state = np.zeros((actions.shape[0], 8), dtype=np.float64)
    state[:, 0:3] = initial[0:3] + np.concatenate(
        [np.zeros((1, 3)), np.cumsum(actions[:-1, 0:3], axis=0)],
        axis=0,
    ) * position_scale
    state[:, 3:6] = rotation_matrix_to_axis_angle(
        np.concatenate(
            [
                SCRIPT_MODULE._axis_angle_to_rotation_matrix_np(initial[3:6])[None, :, :],
                integrate_action_rotation_matrices(
                    initial_axis_angle=initial[3:6],
                    rotational_actions=actions[:-1, 3:6],
                    rotation_scale=rotation_scale,
                ),
            ],
            axis=0,
        )
    )

    report = fit_integrated_eef_scales(
        [{"action": actions, "state": state}],
        alignment="pre_action",
        rotation_scale_candidates=(-0.2, -0.1, -0.05, 0.05, 0.1, 0.2),
    )

    assert report["selected"]["position_scale"] == pytest.approx(position_scale)
    assert report["selected"]["rotation_scale"] == pytest.approx(rotation_scale)
    assert report["selected"]["rotation_mean_geodesic_rad"] < 1e-8
    assert report["selected"]["recovered_action_max_abs_error"] < 1e-5
