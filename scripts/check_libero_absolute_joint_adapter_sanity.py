from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AdapterProfile:
    name: str
    execution_mode: str
    substeps: str
    control_frequency_mode: str
    gripper_substep_policy: str
    joint_kp: float | None
    target_source: str = "measured_qpos"
    integrated_delta_scale: str = "0.05"


QUICK_PROFILES: tuple[AdapterProfile, ...] = (
    AdapterProfile(
        name="public_delta_s1_raw",
        execution_mode="normalized_delta",
        substeps="1",
        control_frequency_mode="raw",
        gripper_substep_policy="repeat",
        joint_kp=None,
    ),
    AdapterProfile(
        name="normalized_delta_s4_raw_freq",
        execution_mode="normalized_delta",
        substeps="4",
        control_frequency_mode="raw",
        gripper_substep_policy="repeat",
        joint_kp=None,
    ),
    AdapterProfile(
        name="normalized_delta_s8_raw_freq",
        execution_mode="normalized_delta",
        substeps="8",
        control_frequency_mode="raw",
        gripper_substep_policy="repeat",
        joint_kp=None,
    ),
    AdapterProfile(
        name="integrated_delta_auto_s1_raw",
        execution_mode="integrated_delta",
        substeps="1",
        control_frequency_mode="raw",
        gripper_substep_policy="repeat",
        joint_kp=None,
        target_source="integrated_delta",
        integrated_delta_scale="auto",
    ),
    AdapterProfile(
        name="direct_goal_kp500_s2_scaled_raw_gripper",
        execution_mode="direct_goal",
        substeps="2",
        control_frequency_mode="scaled_by_substeps",
        gripper_substep_policy="repeat",
        joint_kp=500.0,
    ),
    AdapterProfile(
        name="direct_goal_kp500_s4_scaled_raw_gripper",
        execution_mode="direct_goal",
        substeps="4",
        control_frequency_mode="scaled_by_substeps",
        gripper_substep_policy="repeat",
        joint_kp=500.0,
    ),
)

EXTENDED_PROFILES: tuple[AdapterProfile, ...] = QUICK_PROFILES + (
    AdapterProfile(
        name="direct_goal_default_gain_s2_scaled_raw_gripper",
        execution_mode="direct_goal",
        substeps="2",
        control_frequency_mode="scaled_by_substeps",
        gripper_substep_policy="repeat",
        joint_kp=None,
    ),
    AdapterProfile(
        name="direct_goal_kp500_s8_scaled_raw_gripper",
        execution_mode="direct_goal",
        substeps="8",
        control_frequency_mode="scaled_by_substeps",
        gripper_substep_policy="repeat",
        joint_kp=500.0,
    ),
    AdapterProfile(
        name="direct_goal_kp500_s2_raw_freq_raw_gripper",
        execution_mode="direct_goal",
        substeps="2",
        control_frequency_mode="raw",
        gripper_substep_policy="repeat",
        joint_kp=500.0,
    ),
)


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    profiles = EXTENDED_PROFILES if args.profile_set == "extended" else QUICK_PROFILES
    started_at = time.time()
    results: list[dict[str, Any]] = []
    for profile in profiles:
        result = _run_profile(args, profile=profile, output_root=output_root)
        results.append(result)
        print(json.dumps({"event": "profile_done", **result}, sort_keys=True), flush=True)
    report = {
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "replay_status_path": str(Path(args.replay_status_path).expanduser().resolve()),
        "rollout_config": str(_resolve_repo_path(args.rollout_config)),
        "episode_indices": args.episode_indices,
        "profile_set": args.profile_set,
        "profiles": results,
        "best_profiles": _best_profiles(results),
        "wall_time_s": round(time.time() - started_at, 3),
    }
    report_path = output_root / "adapter_sanity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "sanity_report", "report_path": str(report_path)}, sort_keys=True), flush=True)


def _run_profile(args: argparse.Namespace, *, profile: AdapterProfile, output_root: Path) -> dict[str, Any]:
    profile_root = output_root / profile.name
    logs_dir = profile_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "process_libero10_absolute_joint_dataset.py"),
        "--dataset-root",
        str(Path(args.dataset_root).expanduser()),
        "--replay-status-path",
        str(Path(args.replay_status_path).expanduser()),
        "--output-root",
        str(profile_root),
        "--episode-indices",
        args.episode_indices,
        "--camera-height",
        str(args.camera_height),
        "--camera-width",
        str(args.camera_width),
        "--absolute-profile-sweep",
        "rollout_adapter",
        "--rollout-adapter-execution-mode",
        profile.execution_mode,
        "--substeps-sweep",
        profile.substeps,
        "--absolute-control-frequency-mode",
        profile.control_frequency_mode,
        "--absolute-joint-target-source",
        profile.target_source,
        "--integrated-delta-scale",
        profile.integrated_delta_scale,
        "--absolute-gripper-substep-policy",
        profile.gripper_substep_policy,
        "--absolute-success-grace-steps",
        "0",
        "--rollout-config",
        str(_resolve_repo_path(args.rollout_config)),
    ]
    if profile.joint_kp is not None:
        command.extend(["--absolute-joint-kp", str(profile.joint_kp)])
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    started_at = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    summary_path = profile_root / "summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    status_counts = dict(summary.get("conversion_status_counts") or {})
    success_count = int(status_counts.get("success", 0))
    total = int(summary.get("total_selected_metadata_success_rows", 0) or 0)
    failed = list(summary.get("failed_records") or [])
    return {
        "profile": profile.name,
        "returncode": int(completed.returncode),
        "output_root": str(profile_root),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "success_count": success_count,
        "total": total,
        "success_rate": None if total <= 0 else success_count / total,
        "status_counts": status_counts,
        "failed_episode_indices": [int(record["dataset_episode_index"]) for record in failed],
        "wall_time_s": round(time.time() - started_at, 3),
    }


def _best_profiles(results: list[dict[str, Any]]) -> list[str]:
    if not results:
        return []
    best = max(int(result.get("success_count") or 0) for result in results)
    return [str(result["profile"]) for result in results if int(result.get("success_count") or 0) == best]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run small LIBERO absolute-joint adapter replay probes before full dataset conversion. "
            "Every profile uses process_libero10_absolute_joint_dataset.py with rollout_adapter verification."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--replay-status-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episode-indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--profile-set",
        choices=("quick", "extended"),
        default="quick",
        help="`quick` compares public JOINT_POSITION delta substeps against direct-goal candidates.",
    )
    parser.add_argument(
        "--rollout-config",
        default="configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_abs_joint.yaml",
    )
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
