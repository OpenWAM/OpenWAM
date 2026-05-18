from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from process_libero10_absolute_joint_dataset import (  # noqa: E402
    _build_control_env,
    _env_reached_success,
    _resolved_reset_seed,
    _reset_seeded_env,
    _task_resources,
)
from validate_libero_absolute_joint_position import _row_action  # noqa: E402


DEFAULT_ROTATION_SCALE_CANDIDATES = (
    -0.5,
    -0.1,
    -0.05,
    -0.01,
    -0.005,
    -0.0025,
    0.0025,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
)


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    replay_status_path = Path(args.replay_status_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rows = _load_success_rows(replay_status_path)
    rows = _filter_episode_indices(rows, args.episode_indices)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError("No metadata-success rows selected.")

    dataset_info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    action_names = ((dataset_info.get("features") or {}).get("action") or {}).get("names")
    action_motor_names = (action_names or {}).get("motors") if isinstance(action_names, dict) else action_names
    source_episodes = [_load_episode_arrays(dataset_root, dataset_info=dataset_info, episode_index=int(row["dataset_episode_index"])) for row in rows]

    calibration = fit_integrated_eef_scales(
        source_episodes,
        alignment=args.alignment,
        rotation_scale_candidates=_parse_float_list(args.rotation_scale_candidates),
    )
    selected_alignment = str(calibration["selected_alignment"])
    position_scale = float(calibration["selected"]["position_scale"])
    rotation_scale = float(calibration["selected"]["rotation_scale"])
    sanity = None
    if args.sanity_episode_indices:
        sanity_rows = _filter_episode_indices(rows, args.sanity_episode_indices)
        sanity = _run_replay_sanity(
            dataset_root=dataset_root,
            replay_rows=sanity_rows,
            position_scale=position_scale,
            rotation_scale=rotation_scale,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )

    report = {
        "dataset_root": str(dataset_root),
        "replay_status_path": str(replay_status_path),
        "episodes_fit": len(source_episodes),
        "action_motor_names": action_motor_names,
        "calibration": calibration,
        "selected_scale": {
            "alignment": selected_alignment,
            "position_scale": position_scale,
            "rotation_scale": rotation_scale,
            "target_dim": 10,
            "rotation_representation": "continuous_6d",
        },
        "sanity": sanity,
        "warning": (
            "This is valid only for datasets whose source action is OSC/EEF delta. "
            "The transform is pseudo-absolute: finite differences recover the OSC command exactly."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "eef_scale_calibrated", "output": str(output_path), "selected_scale": report["selected_scale"]}, sort_keys=True))
    if sanity is not None:
        print(json.dumps({"event": "eef_sanity_done", **sanity["summary"]}, sort_keys=True))


def fit_integrated_eef_scales(
    episodes: list[dict[str, np.ndarray]],
    *,
    alignment: str,
    rotation_scale_candidates: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    alignments = ("same_after_action", "next_after_action", "pre_action") if alignment == "auto" else (alignment,)
    results = []
    for candidate in alignments:
        position_scale, rotation_scale = _fit_scales_for_alignment(
            episodes,
            alignment=candidate,
            rotation_scale_candidates=rotation_scale_candidates,
        )
        metrics = _evaluate_scales(
            episodes,
            alignment=candidate,
            position_scale=position_scale,
            rotation_scale=rotation_scale,
        )
        results.append(
            {
                "alignment": candidate,
                "position_scale": float(position_scale),
                "rotation_scale": float(rotation_scale),
                **metrics,
            }
        )
    selected = min(
        results,
        key=lambda item: (float(item["position_mean_l2"]) + float(item["rotation_mean_geodesic_rad"])),
    )
    return {
        "selected_alignment": selected["alignment"],
        "selected": selected,
        "candidates": results,
    }


def build_integrated_eef_targets(
    *,
    initial_state: np.ndarray,
    actions: np.ndarray,
    position_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float32).reshape(-1)
    action = np.asarray(actions, dtype=np.float32)
    if state.shape[0] < 6:
        raise ValueError(f"Expected initial EEF state with at least 6 channels, got {state.shape[0]}.")
    if action.ndim != 2 or action.shape[1] < 7:
        raise ValueError(f"Expected actions with shape [T, >=7], got {action.shape}.")
    position = state[0:3][None, :] + np.cumsum(action[:, 0:3], axis=0, dtype=np.float32) * float(position_scale)
    rotation_matrices = _integrate_action_rotation_matrices(
        initial_axis_angle=state[3:6],
        rotational_actions=action[:, 3:6],
        rotation_scale=rotation_scale,
    )
    rotation_6d = _rotation_matrix_to_continuous_6d(rotation_matrices)
    return np.concatenate([position, rotation_6d, action[:, 6:7]], axis=1).astype(np.float32)


def recover_osc_actions_from_integrated_eef_targets(
    *,
    initial_state: np.ndarray,
    targets: np.ndarray,
    position_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float32).reshape(-1)
    target = np.asarray(targets, dtype=np.float32)
    if state.shape[0] < 6:
        raise ValueError(f"Expected initial EEF state with at least 6 channels, got {state.shape[0]}.")
    if target.ndim != 2 or target.shape[1] < 10:
        raise ValueError(f"Expected targets with shape [T, >=10], got {target.shape}.")
    if abs(float(position_scale)) <= 1e-12 or abs(float(rotation_scale)) <= 1e-12:
        raise ValueError("Position and rotation scales must both be nonzero for target-to-action recovery.")
    previous_position = np.concatenate([state[0:3][None, :], target[:-1, 0:3]], axis=0)
    previous_rotation = np.concatenate(
        [_axis_angle_to_rotation_matrix_np(state[3:6])[None, :, :], _rotation_6d_to_matrix_np(target[:-1, 3:9])],
        axis=0,
    )
    target_rotation = _rotation_6d_to_matrix_np(target[:, 3:9])
    actions = np.zeros((target.shape[0], 7), dtype=np.float32)
    actions[:, 0:3] = (target[:, 0:3] - previous_position) / float(position_scale)
    actions[:, 3:6] = _relative_rotation_matrix_to_axis_angle(target_rotation, previous_rotation) / float(
        rotation_scale
    )
    actions[:, 6] = target[:, 9]
    return np.clip(actions, -1.0, 1.0).astype(np.float32)


def _fit_scales_for_alignment(
    episodes: list[dict[str, np.ndarray]],
    *,
    alignment: str,
    rotation_scale_candidates: tuple[float, ...] | None,
) -> tuple[float, float]:
    position_num = 0.0
    position_den = 0.0
    for episode in episodes:
        cumulative, eef_delta = _aligned_cumulative_and_delta(episode, alignment=alignment)
        position_num += float(np.sum(cumulative[:, 0:3] * eef_delta[:, 0:3]))
        position_den += float(np.sum(cumulative[:, 0:3] * cumulative[:, 0:3]))
    if position_den <= 1e-12:
        raise ValueError("Cannot fit EEF position scale because source position action signal is degenerate.")
    if rotation_scale_candidates is None:
        rotation_scale = _fit_rotation_scale_by_so3_trajectory(episodes, alignment=alignment)
    else:
        rotation_scale = _fit_rotation_scale_for_alignment(
            episodes,
            alignment=alignment,
            rotation_scale_candidates=rotation_scale_candidates,
        )
    return position_num / position_den, rotation_scale


def _evaluate_scales(
    episodes: list[dict[str, np.ndarray]],
    *,
    alignment: str,
    position_scale: float,
    rotation_scale: float,
) -> dict[str, float]:
    position_errors: list[float] = []
    rotation_errors: list[float] = []
    reconstruction_errors: list[float] = []
    for episode in episodes:
        cumulative, eef_delta = _aligned_cumulative_and_delta(episode, alignment=alignment)
        pred_position_delta = cumulative[:, 0:3] * float(position_scale)
        position_error = pred_position_delta - eef_delta[:, 0:3]
        position_errors.extend(np.linalg.norm(position_error, axis=1).tolist())
        rotation_errors.extend(
            _rotation_geodesic_errors_for_scale(
                episode,
                alignment=alignment,
                rotation_scale=rotation_scale,
            ).tolist()
        )

        targets = build_integrated_eef_targets(
            initial_state=episode["state"][0],
            actions=episode["action"],
            position_scale=position_scale,
            rotation_scale=rotation_scale,
        )
        recovered = recover_osc_actions_from_integrated_eef_targets(
            initial_state=episode["state"][0],
            targets=targets,
            position_scale=position_scale,
            rotation_scale=rotation_scale,
        )
        reconstruction_errors.extend(np.max(np.abs(recovered - episode["action"][:, :7]), axis=1).tolist())
    return {
        "position_mean_l2": float(np.mean(position_errors)),
        "position_p95_l2": float(np.quantile(position_errors, 0.95)),
        "position_max_l2": float(np.max(position_errors)),
        "rotation_mean_geodesic_rad": float(np.mean(rotation_errors)),
        "rotation_p95_geodesic_rad": float(np.quantile(rotation_errors, 0.95)),
        "rotation_max_geodesic_rad": float(np.max(rotation_errors)),
        "recovered_action_max_abs_error": float(np.max(reconstruction_errors)),
    }


def _aligned_cumulative_and_delta(episode: dict[str, np.ndarray], *, alignment: str) -> tuple[np.ndarray, np.ndarray]:
    action = episode["action"]
    state = episode["state"]
    if alignment == "same_after_action":
        cumulative = np.cumsum(action[:, 0:6], axis=0, dtype=np.float64)
        eef_delta = state[:, 0:6] - state[0:1, 0:6]
    elif alignment == "next_after_action":
        cumulative = np.cumsum(action[:-1, 0:6], axis=0, dtype=np.float64)
        eef_delta = state[1:, 0:6] - state[0:1, 0:6]
    elif alignment == "pre_action":
        cumulative = np.concatenate(
            [np.zeros((1, 6), dtype=np.float64), np.cumsum(action[:-1, 0:6], axis=0, dtype=np.float64)],
            axis=0,
        )
        eef_delta = state[:, 0:6] - state[0:1, 0:6]
    else:
        raise ValueError(f"Unknown alignment: {alignment!r}.")
    return cumulative, eef_delta


def _fit_rotation_scale_for_alignment(
    episodes: list[dict[str, np.ndarray]],
    *,
    alignment: str,
    rotation_scale_candidates: tuple[float, ...],
) -> float:
    candidates = tuple(
        float(candidate)
        for candidate in rotation_scale_candidates
        if math.isfinite(float(candidate)) and abs(float(candidate)) > 1e-12
    )
    if not candidates:
        raise ValueError("At least one nonzero finite rotation-scale candidate is required.")
    errors = []
    for candidate in candidates:
        per_episode = [
            _rotation_geodesic_errors_for_scale(episode, alignment=alignment, rotation_scale=candidate)
            for episode in episodes
        ]
        concatenated = np.concatenate(per_episode, axis=0)
        errors.append((float(np.mean(concatenated)), abs(candidate), candidate))
    return min(errors)[2]


def _fit_rotation_scale_by_so3_trajectory(
    episodes: list[dict[str, np.ndarray]],
    *,
    alignment: str,
) -> float:
    initial = _fit_rotation_scale_from_observed_steps(episodes, alignment=alignment)
    lower, upper = _rotation_scale_search_bounds(initial)

    try:
        from scipy.optimize import minimize_scalar  # type: ignore[import-not-found]

        result = minimize_scalar(
            lambda scale: _mean_rotation_trajectory_loss(episodes, alignment=alignment, rotation_scale=float(scale)),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 48},
        )
        if bool(result.success) and math.isfinite(float(result.x)) and abs(float(result.x)) > 1e-12:
            return float(result.x)
    except Exception:
        pass

    local = np.linspace(lower, upper, num=81, dtype=np.float64)
    candidates = tuple(float(value) for value in local if abs(float(value)) > 1e-12)
    return _fit_rotation_scale_for_alignment(
        episodes,
        alignment=alignment,
        rotation_scale_candidates=candidates,
    )


def _fit_rotation_scale_from_observed_steps(
    episodes: list[dict[str, np.ndarray]],
    *,
    alignment: str,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for episode in episodes:
        action = np.asarray(episode["action"], dtype=np.float64)
        state = np.asarray(episode["state"], dtype=np.float64)
        if action.shape[0] < 2 or state.shape[0] < 2:
            continue
        if alignment == "same_after_action":
            rotational_actions = action[1:, 3:6]
        elif alignment in {"next_after_action", "pre_action"}:
            rotational_actions = action[:-1, 3:6]
        else:
            raise ValueError(f"Unknown alignment: {alignment!r}.")
        observed_rotations = _axis_angle_to_rotation_matrix_np(state[:, 3:6])
        observed_delta_axis_angle = _relative_rotation_matrix_to_axis_angle(
            observed_rotations[1:],
            observed_rotations[:-1],
        )
        length = min(rotational_actions.shape[0], observed_delta_axis_angle.shape[0])
        if length <= 0:
            continue
        action_slice = rotational_actions[:length]
        observed_slice = observed_delta_axis_angle[:length]
        numerator += float(np.sum(action_slice * observed_slice))
        denominator += float(np.sum(action_slice * action_slice))
    if denominator <= 1e-12:
        raise ValueError("Cannot fit EEF rotation scale because source rotation action signal is degenerate.")
    return numerator / denominator


def _rotation_scale_search_bounds(initial: float) -> tuple[float, float]:
    center = float(initial)
    radius = max(0.05, abs(center) * 5.0)
    lower = max(-1.0, center - radius)
    upper = min(1.0, center + radius)
    if lower >= upper:
        return -1.0, 1.0
    if lower <= 0.0 <= upper:
        return lower, upper
    return lower, upper


def _mean_rotation_trajectory_loss(
    episodes: list[dict[str, np.ndarray]],
    *,
    alignment: str,
    rotation_scale: float,
) -> float:
    errors = [
        _rotation_geodesic_errors_for_scale(episode, alignment=alignment, rotation_scale=rotation_scale)
        for episode in episodes
    ]
    concatenated = np.concatenate(errors, axis=0)
    return float(np.mean(concatenated * concatenated))


def _rotation_geodesic_errors_for_scale(
    episode: dict[str, np.ndarray],
    *,
    alignment: str,
    rotation_scale: float,
) -> np.ndarray:
    action = np.asarray(episode["action"], dtype=np.float64)
    state = np.asarray(episode["state"], dtype=np.float64)
    predicted_after = _integrate_action_rotation_matrices(
        initial_axis_angle=state[0, 3:6],
        rotational_actions=action[:, 3:6],
        rotation_scale=rotation_scale,
    )
    observed = _axis_angle_to_rotation_matrix_np(state[:, 3:6])
    if alignment == "same_after_action":
        predicted = predicted_after
        target = observed[: predicted.shape[0]]
    elif alignment == "next_after_action":
        predicted = predicted_after[:-1]
        target = observed[1 : 1 + predicted.shape[0]]
    elif alignment == "pre_action":
        predicted = np.concatenate([observed[0:1], predicted_after[:-1]], axis=0)
        target = observed[: predicted.shape[0]]
    else:
        raise ValueError(f"Unknown alignment: {alignment!r}.")
    if predicted.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    return _rotation_matrix_geodesic_distance(predicted, target)


def _integrate_action_rotation_matrices(
    *,
    initial_axis_angle: np.ndarray,
    rotational_actions: np.ndarray,
    rotation_scale: float,
) -> np.ndarray:
    actions = np.asarray(rotational_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[-1] != 3:
        raise ValueError(f"Expected rotational actions with shape [T, 3], got {actions.shape}.")
    current = _axis_angle_to_quaternion_np(np.asarray(initial_axis_angle, dtype=np.float64).reshape(3))
    rotations = []
    for action in actions:
        delta = _axis_angle_to_quaternion_np(action * float(rotation_scale))
        current = _normalize_quaternion_np(_quaternion_multiply_np(delta, current))
        rotations.append(_quaternion_to_rotation_matrix_np(current))
    if not rotations:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.stack(rotations, axis=0)


def _axis_angle_to_rotation_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    return _quaternion_to_rotation_matrix_np(_axis_angle_to_quaternion_np(axis_angle))


def _axis_angle_to_quaternion_np(axis_angle: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=np.float64)
    if vector.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle vector ending in 3 dims, got {vector.shape}.")
    angle = np.linalg.norm(vector, axis=-1, keepdims=True)
    safe_axis = vector / np.maximum(angle, 1e-12)
    half_angle = 0.5 * angle
    xyz = safe_axis * np.sin(half_angle)
    w = np.cos(half_angle)
    quaternion = np.concatenate([xyz, w], axis=-1)
    identity = np.zeros_like(quaternion)
    identity[..., 3] = 1.0
    return _normalize_quaternion_np(np.where(angle > 1e-12, quaternion, identity))


def _quaternion_to_rotation_matrix_np(quaternion: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion_np(np.asarray(quaternion, dtype=np.float64))
    x, y, z, w = np.moveaxis(quat, -1, 0)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    matrix = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[..., 0, 1] = 2.0 * (xy - zw)
    matrix[..., 0, 2] = 2.0 * (xz + yw)
    matrix[..., 1, 0] = 2.0 * (xy + zw)
    matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[..., 1, 2] = 2.0 * (yz - xw)
    matrix[..., 2, 0] = 2.0 * (xz - yw)
    matrix[..., 2, 1] = 2.0 * (yz + xw)
    matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def _normalize_quaternion_np(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    return quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)


def _quaternion_multiply_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_arr = np.asarray(lhs, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    if lhs_arr.shape[-1] != 4 or rhs_arr.shape[-1] != 4:
        raise ValueError("Quaternion multiplication expects arrays ending in 4 dims.")
    x1, y1, z1, w1 = np.moveaxis(lhs_arr, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(rhs_arr, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _rotation_matrix_to_continuous_6d(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices ending in [3, 3], got {mat.shape}.")
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def _rotation_6d_to_matrix_np(rotation_6d: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation_6d, dtype=np.float64)
    if rot.shape[-1] != 6:
        raise ValueError(f"Expected continuous-6D rotations ending in 6 dims, got {rot.shape}.")
    first = _normalize_vectors_np(rot[..., 0:3])
    second_raw = rot[..., 3:6] - np.sum(first * rot[..., 3:6], axis=-1, keepdims=True) * first
    second = _normalize_vectors_np(_replace_degenerate_second_axis(first, second_raw))
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1)


def _replace_degenerate_second_axis(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    fallback_seed = np.zeros_like(first)
    fallback_seed[..., 0] = 1.0
    near_x = np.abs(np.sum(first * fallback_seed, axis=-1, keepdims=True)) > 0.9
    fallback_seed = np.where(near_x, np.asarray([0.0, 1.0, 0.0], dtype=np.float64), fallback_seed)
    fallback = np.cross(first, fallback_seed)
    return np.where(norm > 1e-8, second, fallback)


def _normalize_vectors_np(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-12)


def _relative_rotation_matrix_to_axis_angle(target: np.ndarray, previous: np.ndarray) -> np.ndarray:
    target_matrix = np.asarray(target, dtype=np.float64)
    previous_matrix = np.asarray(previous, dtype=np.float64)
    delta = np.matmul(target_matrix, np.swapaxes(previous_matrix, -1, -2))
    return _rotation_matrix_to_axis_angle_np(delta)


def _rotation_matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices ending in [3, 3], got {mat.shape}.")
    trace = np.trace(mat, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    vee = np.stack(
        [
            mat[..., 2, 1] - mat[..., 1, 2],
            mat[..., 0, 2] - mat[..., 2, 0],
            mat[..., 1, 0] - mat[..., 0, 1],
        ],
        axis=-1,
    )
    sin_angle = np.sin(angle)
    regular = vee / np.maximum(2.0 * sin_angle[..., None], 1e-12) * angle[..., None]
    small_angle = 0.5 * vee
    return np.where(angle[..., None] > 1e-6, regular, small_angle)


def _rotation_matrix_geodesic_distance(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    relative = np.matmul(np.asarray(lhs, dtype=np.float64), np.swapaxes(np.asarray(rhs, dtype=np.float64), -1, -2))
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def _run_replay_sanity(
    *,
    dataset_root: Path,
    replay_rows: list[dict[str, Any]],
    position_scale: float,
    rotation_scale: float,
    camera_height: int,
    camera_width: int,
) -> dict[str, Any]:
    task_cache: dict[int, tuple[Any, Any]] = {}
    results = []
    for row in replay_rows:
        episode_index = int(row["dataset_episode_index"])
        episode_rows = _load_episode_rows_from_parquet(dataset_root, episode_index=episode_index)
        actions = np.asarray([_row_action(item) for item in episode_rows], dtype=np.float32)
        states = np.asarray([item["observation.state"] for item in episode_rows], dtype=np.float32)
        targets = build_integrated_eef_targets(
            initial_state=states[0],
            actions=actions,
            position_scale=position_scale,
            rotation_scale=rotation_scale,
        )
        recovered_actions = recover_osc_actions_from_integrated_eef_targets(
            initial_state=states[0],
            targets=targets,
            position_scale=position_scale,
            rotation_scale=rotation_scale,
        )
        max_reconstruction_error = float(np.max(np.abs(recovered_actions - actions[:, :7])))

        task_spec, init_states = _task_resources(
            task_cache,
            benchmark=str(row.get("upstream_benchmark", "libero_10")),
            task_id=int(row["upstream_task_id"]),
        )
        env = _build_control_env(
            task_spec,
            controller="OSC_POSE",
            camera_height=camera_height,
            camera_width=camera_width,
            control_freq=int(row.get("env_control_freq", 20) or 20),
        )
        try:
            obs = _reset_seeded_env(
                env,
                init_state=init_states[int(row["resolved_init_state_index"])],
                reset_seed=_resolved_reset_seed(row),
            )
            success = False
            for action in recovered_actions:
                obs, _, done, _ = env.step(action.astype(np.float32, copy=False))
                if _env_reached_success(env, done=bool(done)):
                    success = True
                    break
        finally:
            env.close()
        results.append(
            {
                "dataset_episode_index": episode_index,
                "success": bool(success),
                "max_reconstruction_error": max_reconstruction_error,
            }
        )
    success_count = sum(1 for item in results if item["success"])
    return {
        "summary": {
            "episodes": len(results),
            "success_count": int(success_count),
            "success_rate": None if not results else float(success_count / len(results)),
            "max_reconstruction_error": float(max(item["max_reconstruction_error"] for item in results)) if results else None,
        },
        "episodes": results,
    }


def _load_success_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("replay_status") == "success"]
    rows.sort(key=lambda row: int(row["dataset_episode_index"]))
    return rows


def _filter_episode_indices(rows: list[dict[str, Any]], raw_indices: str | None) -> list[dict[str, Any]]:
    if raw_indices is None:
        return rows
    wanted = {int(part.strip()) for part in raw_indices.split(",") if part.strip()}
    return [row for row in rows if int(row["dataset_episode_index"]) in wanted]


def _parse_float_list(raw: str | None) -> tuple[float, ...] | None:
    if raw is None or raw.strip().lower() in {"", "default"}:
        return None
    if raw.strip().lower() == "grid":
        return DEFAULT_ROTATION_SCALE_CANDIDATES
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one numeric value in a comma-separated float list.")
    return values


def _load_episode_arrays(dataset_root: Path, *, dataset_info: dict[str, Any], episode_index: int) -> dict[str, np.ndarray]:
    data_path = dataset_root / str(dataset_info["data_path"]).format(
        episode_chunk=episode_index // int(dataset_info.get("chunks_size", 1000)),
        episode_index=episode_index,
    )
    table = pq.read_table(data_path, columns=["action", "observation.state"])
    return {
        "episode_index": np.asarray([episode_index], dtype=np.int64),
        "action": np.asarray(table["action"].to_pylist(), dtype=np.float64),
        "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float64),
    }


def _load_episode_rows_from_parquet(dataset_root: Path, *, episode_index: int) -> list[dict[str, Any]]:
    dataset_info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    data_path = dataset_root / str(dataset_info["data_path"]).format(
        episode_chunk=episode_index // int(dataset_info.get("chunks_size", 1000)),
        episode_index=episode_index,
    )
    table = pq.read_table(data_path)
    return table.to_pylist()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit LIBERO-wide pseudo-absolute EEF scales from OSC delta actions to observation.state."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--replay-status-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-indices", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--alignment",
        choices=("auto", "same_after_action", "next_after_action", "pre_action"),
        default="auto",
    )
    parser.add_argument(
        "--rotation-scale-candidates",
        default=None,
        help=(
            "Optional comma-separated rotation-scale candidates for SO(3) geodesic fitting. "
            "Use 'grid' for a small signed grid. Omit to fit the scalar by cumulative SO(3) trajectory error."
        ),
    )
    parser.add_argument(
        "--sanity-episode-indices",
        default=None,
        help="Optional comma-separated episodes to replay after target->delta reconstruction.",
    )
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    main()
