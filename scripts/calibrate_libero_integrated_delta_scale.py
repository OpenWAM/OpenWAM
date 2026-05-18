from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    args = _parse_args()
    records = _load_success_records(Path(args.conversion_manifest).expanduser())
    if args.limit is not None:
        records = records[: int(args.limit)]
    if not records:
        raise ValueError("No successful records found for calibration.")

    numerator = 0.0
    denominator = 0.0
    per_episode: list[dict[str, Any]] = []
    for record in records:
        sidecar_path = Path(str(record["sidecar_path"])).expanduser()
        if not sidecar_path.is_file():
            continue
        sidecar = np.load(sidecar_path)
        measured_qpos = np.asarray(sidecar["joint_positions_after_action"], dtype=np.float64)
        source_actions = np.asarray(sidecar["raw_osc_actions"], dtype=np.float64)
        if measured_qpos.ndim != 2 or source_actions.ndim != 2:
            continue
        joint_dim = measured_qpos.shape[1]
        if source_actions.shape[1] < joint_dim:
            continue
        ep_num, ep_den = _fit_terms_with_episode_intercept(
            measured_qpos=measured_qpos,
            source_actions=source_actions[:, :joint_dim],
        )
        if ep_den <= 1e-12:
            continue
        ep_scale = ep_num / ep_den
        numerator += ep_num
        denominator += ep_den
        per_episode.append(
            {
                "dataset_episode_index": int(record["dataset_episode_index"]),
                "scale": float(ep_scale),
                "frames": int(measured_qpos.shape[0]),
            }
        )

    if denominator <= 1e-12:
        raise ValueError("No nonzero source-action signal found for calibration.")
    fitted_scale = float(numerator / denominator)
    candidate_scales = _candidate_scales(args.candidate_scales, fitted_scale=fitted_scale)
    metrics = [
        _evaluate_scale(records=records, scale=scale)
        for scale in candidate_scales
    ]
    best_by_mean_l2 = min(metrics, key=lambda item: float(item["mean_l2"]))
    positive_metrics = [item for item in metrics if float(item["scale"]) > 0.0]
    best_positive_by_mean_l2 = min(positive_metrics, key=lambda item: float(item["mean_l2"])) if positive_metrics else None
    report = {
        "source": {
            "conversion_manifest": str(Path(args.conversion_manifest).expanduser()),
            "records_considered": len(records),
            "episodes_fit": len(per_episode),
        },
        "fitted_scale_unconstrained": fitted_scale,
        "fitted_scale_positive": max(float(args.min_positive_scale), fitted_scale),
        "per_episode_scale_summary": _summary([row["scale"] for row in per_episode]),
        "candidate_metrics": metrics,
        "best_by_mean_l2": best_by_mean_l2,
        "best_positive_by_mean_l2": best_positive_by_mean_l2,
        "warning": (
            "LIBERO source action metadata names are x/y/z/roll/pitch/yaw/gripper; "
            "a near-zero or negative fitted scale means these actions are not usable as joint deltas."
        ),
    }
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "calibrated", "output": str(output_path), **report["source"]}, sort_keys=True))
    print(json.dumps({"best_by_mean_l2": best_by_mean_l2, "best_positive_by_mean_l2": best_positive_by_mean_l2}, sort_keys=True))


def _load_success_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("conversion_status") == "success":
                records.append(record)
    records.sort(key=lambda record: int(record["dataset_episode_index"]))
    return records


def _fit_terms_with_episode_intercept(*, measured_qpos: np.ndarray, source_actions: np.ndarray) -> tuple[float, float]:
    cumulative = np.cumsum(source_actions, axis=0, dtype=np.float64)
    centered_cumulative = cumulative - cumulative[0:1]
    centered_qpos = measured_qpos - measured_qpos[0:1]
    return float(np.sum(centered_cumulative * centered_qpos)), float(np.sum(centered_cumulative * centered_cumulative))


def _candidate_scales(raw: str, *, fitted_scale: float) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "fit":
            values.append(float(fitted_scale))
        else:
            values.append(float(part))
    return sorted(set(values))


def _evaluate_scale(*, records: list[dict[str, Any]], scale: float) -> dict[str, Any]:
    l2_values: list[float] = []
    linf_values: list[float] = []
    episode_count = 0
    for record in records:
        sidecar_path = Path(str(record["sidecar_path"])).expanduser()
        if not sidecar_path.is_file():
            continue
        sidecar = np.load(sidecar_path)
        measured_qpos = np.asarray(sidecar["joint_positions_after_action"], dtype=np.float64)
        source_actions = np.asarray(sidecar["raw_osc_actions"], dtype=np.float64)
        joint_dim = measured_qpos.shape[1]
        if source_actions.shape[1] < joint_dim:
            continue
        cumulative = np.cumsum(source_actions[:, :joint_dim], axis=0, dtype=np.float64)
        predicted = measured_qpos[0:1] + (cumulative - cumulative[0:1]) * float(scale)
        error = predicted - measured_qpos
        l2_values.extend(np.linalg.norm(error, axis=1).tolist())
        linf_values.extend(np.max(np.abs(error), axis=1).tolist())
        episode_count += 1
    return {
        "scale": float(scale),
        "episodes": int(episode_count),
        "mean_l2": float(np.mean(l2_values)),
        "median_l2": float(np.median(l2_values)),
        "p95_l2": float(np.quantile(l2_values, 0.95)),
        "max_l2": float(np.max(l2_values)),
        "mean_linf": float(np.mean(linf_values)),
        "p95_linf": float(np.quantile(linf_values, 0.95)),
        "max_linf": float(np.max(linf_values)),
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one deterministic scalar for pseudo-absolute joint targets built by integrating source deltas. "
            "This uses existing successful conversion sidecars and does not launch LIBERO."
        )
    )
    parser.add_argument("--conversion-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--candidate-scales",
        default="fit,0.0,0.001,0.002,0.005,0.01,0.02,0.03,0.05,0.08,0.1",
    )
    parser.add_argument("--min-positive-scale", type=float, default=1e-6)
    return parser.parse_args()


if __name__ == "__main__":
    main()
