from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.utils import load_local_path_registry


SANDBOX_SCRIPT = REPO_ROOT / "scripts" / "run_libero_realtime_sandbox.py"
LOCAL_PATHS_ENV_VAR = "OPEN_WAM_LOCAL_PATHS"


@dataclass(frozen=True)
class PlannerProfile:
    name: str
    planner_mode: str
    sequence_empty_plan_policy: str
    deadline_miss_policy: str
    startup_open_loop_chunks: int = 0
    description: str = ""


@dataclass(frozen=True)
class RolloutCase:
    name: str
    config: Path
    checkpoint: str | None
    description: str = ""


@dataclass(frozen=True)
class AblationJob:
    case: RolloutCase
    profile: PlannerProfile
    target_action_hz: float
    command: list[str]
    summary_path: Path | None


DEFAULT_PROFILES: dict[str, PlannerProfile] = {
    "naive_blocking": PlannerProfile(
        name="naive_blocking",
        planner_mode="history_only",
        sequence_empty_plan_policy="wait_for_replan",
        deadline_miss_policy="hold_last",
        description="Block simulation whenever a model plan is late; closest mirror of offline/naive rollout.",
    ),
    "buffered_blocking": PlannerProfile(
        name="buffered_blocking",
        planner_mode="async_buffer",
        sequence_empty_plan_policy="wait_for_replan",
        deadline_miss_policy="hold_last",
        description="Allow async buffering, but pause simulation instead of executing fallback actions.",
    ),
    "live_history_hold": PlannerProfile(
        name="live_history_hold",
        planner_mode="history_only",
        sequence_empty_plan_policy="fallback",
        deadline_miss_policy="hold_last",
        description="Keep simulation clock running with observation-conditioned replans only; hold last action on misses.",
    ),
    "live_history_zero": PlannerProfile(
        name="live_history_zero",
        planner_mode="history_only",
        sequence_empty_plan_policy="fallback",
        deadline_miss_policy="zero",
        description="Keep simulation clock running with observation-conditioned replans only; zero action on misses.",
    ),
    "live_async_hold": PlannerProfile(
        name="live_async_hold",
        planner_mode="async_buffer",
        sequence_empty_plan_policy="fallback",
        deadline_miss_policy="hold_last",
        description="Keep simulation clock running with async buffering and hold-last fallback on misses.",
    ),
    "live_async_history_first_hold": PlannerProfile(
        name="live_async_history_first_hold",
        planner_mode="async_history_first",
        sequence_empty_plan_policy="fallback",
        deadline_miss_policy="hold_last",
        description=(
            "Keep simulation clock running; submit observation-conditioned history replans before "
            "open-loop extensions, holding last action on misses."
        ),
    ),
    "live_async_mix_hold": PlannerProfile(
        name="live_async_mix_hold",
        planner_mode="async_mix",
        sequence_empty_plan_policy="fallback",
        deadline_miss_policy="hold_last",
        description="Keep simulation clock running with a mixed history/extension async planner.",
    ),
    "live_async_zero": PlannerProfile(
        name="live_async_zero",
        planner_mode="async_buffer",
        sequence_empty_plan_policy="fallback",
        deadline_miss_policy="zero",
        description="Keep simulation clock running with async buffering and zero-action fallback on misses.",
    ),
    "live_async_prebuffer_hold": PlannerProfile(
        name="live_async_prebuffer_hold",
        planner_mode="async_buffer",
        sequence_empty_plan_policy="fallback",
        deadline_miss_policy="hold_last",
        startup_open_loop_chunks=2,
        description="Async buffering with two startup open-loop chunks before the live clock starts.",
    ),
    "live_async_prebuffer_history_first_hold": PlannerProfile(
        name="live_async_prebuffer_history_first_hold",
        planner_mode="async_history_first",
        sequence_empty_plan_policy="fallback",
        deadline_miss_policy="hold_last",
        startup_open_loop_chunks=2,
        description=(
            "Two startup open-loop chunks, then observation-conditioned history replans are preferred "
            "over further open-loop extensions."
        ),
    ),
}


DEFAULT_CASES: dict[str, RolloutCase] = {
    "method1_exact_step400": RolloutCase(
        name="method1_exact_step400",
        config=Path("configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"),
        checkpoint="${paths.checkpoints.parallel_stream_exact_libero_step_400}",
        description="Heng-compatible method-1 exact checkpoint reported to succeed on LIBERO.",
    ),
    "method1_exact_latest": RolloutCase(
        name="method1_exact_latest",
        config=Path("configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"),
        checkpoint="${paths.checkpoints.parallel_stream_exact_libero_step_1100_0402}",
        description="Newest local method-1 exact checkpoint alias.",
    ),
    "method2_joint_step600": RolloutCase(
        name="method2_joint_step600",
        config=Path("configs/experiments/parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml"),
        checkpoint="${paths.checkpoints.parallel_stream_joint_libero_step_600_0402}",
        description="Method-2/functionality via action-conditioned parallel-stream runtime.",
    ),
    "method3_vsp_step800": RolloutCase(
        name="method3_vsp_step800",
        config=Path("configs/experiments/video_sequence_policy_libero_latent_local_random_subwindow.yaml"),
        checkpoint="${paths.checkpoints.video_sequence_policy_libero_random_subwindow_step_800_0402}",
        description="Method-3 video-sequence policy checkpoint.",
    ),
    "method4_generated_step5000": RolloutCase(
        name="method4_generated_step5000",
        config=Path("configs/experiments/post_latent_libero_latent_local_generated_video_conditioned.yaml"),
        checkpoint="${paths.checkpoints.method4_generated_video_conditioned_step_5000}",
        description="Method-4 action head trained on generated video-conditioning latents.",
    ),
    "method5_mot_idm_step1900": RolloutCase(
        name="method5_mot_idm_step1900",
        config=Path("configs/experiments/mot_libero_latent_local_idm.yaml"),
        checkpoint="${paths.checkpoints.mot_libero_idm_step_1900_root}",
        description="Method-5 MoT IDM checkpoint.",
    ),
    "method5_mot_joint_step900": RolloutCase(
        name="method5_mot_joint_step900",
        config=Path("configs/experiments/mot_libero_latent_local_joint.yaml"),
        checkpoint="${paths.checkpoints.mot_libero_joint_step_900_root}",
        description="Method-5 MoT joint-denoise checkpoint.",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a LIBERO realtime-inference ablation matrix over planner profiles, "
            "target action rates, and method/checkpoint cases."
        )
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="method1_exact_step400",
        help=(
            "Comma-separated builtin case names, `all`, or custom entries of the form "
            "`name:config[:checkpoint]`."
        ),
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="naive_blocking,buffered_blocking,live_history_hold,live_history_zero,live_async_hold,live_async_zero",
        help=f"Comma-separated profile names or `all`. Builtins: {', '.join(DEFAULT_PROFILES)}",
    )
    parser.add_argument("--target-action-hz", type=str, default="1,2,5,10,15,20")
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_realtime_ablation")
    parser.add_argument("--suffix", type=str, default="ablation")
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--runtime-devices", type=str, default=None)
    parser.add_argument("--runtime-prep-device", type=str, default=None)
    parser.add_argument("--runtime-output-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument("--reference-assets-device-policy", choices=("cpu_offload", "runtime"), default="runtime")
    parser.add_argument("--sequence-buffer-threshold", type=int, default=3)
    parser.add_argument("--video-num-inference-steps", type=int, default=None)
    parser.add_argument("--action-num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--action-guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE environment variable for subprocesses.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--summary-name", type=str, default="ablation_summary.json")
    args = parser.parse_args()

    if args.max_actions <= 0:
        raise ValueError("--max-actions must be positive.")

    local_path_registry = _load_local_path_registry()
    cases = _resolve_cases(args.cases, local_path_registry=local_path_registry)
    profiles = _resolve_profiles(args.profiles)
    target_action_hz_values = _parse_float_csv(args.target_action_hz)
    jobs = _build_jobs(
        cases=cases,
        profiles=profiles,
        target_action_hz_values=target_action_hz_values,
        args=args,
        local_path_registry=local_path_registry,
    )
    if args.dry_run:
        print(json.dumps([_job_to_report(job, status="dry_run") for job in jobs], indent=2))
        return

    output_dir = _repo_path(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(_parse_env_overrides(args.env))
    reports: list[dict[str, Any]] = []
    for job in jobs:
        started_report = _job_to_report(job, status="running")
        print(json.dumps(started_report, sort_keys=True), flush=True)
        try:
            completed = subprocess.run(
                job.command,
                cwd=str(REPO_ROOT),
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            report = _job_to_report(job, status="passed")
            report["stdout_tail"] = completed.stdout[-4000:]
            report["sandbox_summary"] = _extract_last_json_object(completed.stdout)
        except subprocess.CalledProcessError as exc:
            report = _job_to_report(job, status="failed")
            report["returncode"] = int(exc.returncode)
            report["stdout_tail"] = (exc.stdout or "")[-4000:]
            reports.append(report)
            _write_summary(output_dir / args.summary_name, reports)
            print(json.dumps(report, sort_keys=True), flush=True)
            if not args.continue_on_error:
                raise
        else:
            reports.append(report)
            _write_summary(output_dir / args.summary_name, reports)
            print(json.dumps(report, sort_keys=True), flush=True)

    _write_summary(output_dir / args.summary_name, reports)
    print(json.dumps({"summary_path": str((output_dir / args.summary_name).resolve()), "jobs": len(reports)}, indent=2))


def _build_jobs(
    *,
    cases: list[RolloutCase],
    profiles: list[PlannerProfile],
    target_action_hz_values: list[float],
    args: argparse.Namespace,
    local_path_registry: Mapping[str, str],
) -> list[AblationJob]:
    jobs: list[AblationJob] = []
    for case in cases:
        config_path = _repo_path(case.config)
        checkpoint = (
            _resolve_path_token(case.checkpoint, local_path_registry=local_path_registry)
            if case.checkpoint
            else None
        )
        for profile in profiles:
            for target_hz in target_action_hz_values:
                suffix = _safe_token(f"{args.suffix}_{case.name}_{profile.name}_{target_hz:g}hz")
                command = [
                    sys.executable,
                    str(SANDBOX_SCRIPT),
                    "--cfg",
                    str(config_path),
                    "--benchmark",
                    str(args.benchmark),
                    "--task-id",
                    str(args.task_id),
                    "--episode-idx",
                    str(args.episode_idx),
                    "--max-actions",
                    str(args.max_actions),
                    "--target-action-hz",
                    f"{target_hz:g}",
                    "--video-fps",
                    f"{args.video_fps:g}",
                    "--planner-mode",
                    profile.planner_mode,
                    "--sequence-empty-plan-policy",
                    profile.sequence_empty_plan_policy,
                    "--deadline-miss-policy",
                    profile.deadline_miss_policy,
                    "--startup-open-loop-chunks",
                    str(profile.startup_open_loop_chunks),
                    "--sequence-buffer-threshold",
                    str(args.sequence_buffer_threshold),
                    "--reference-assets-device-policy",
                    args.reference_assets_device_policy,
                    "--output-dir",
                    str(_repo_path(Path(args.output_dir))),
                    "--suffix",
                    suffix,
                    "--seed",
                    str(args.seed),
                ]
                if checkpoint is not None:
                    command.extend(["--checkpoint", checkpoint])
                _append_optional_arg(command, "--runtime-device", args.runtime_device)
                _append_optional_arg(command, "--runtime-devices", args.runtime_devices)
                _append_optional_arg(command, "--runtime-prep-device", args.runtime_prep_device)
                _append_optional_arg(command, "--runtime-output-device", args.runtime_output_device)
                _append_optional_arg(command, "--frontend-device", args.frontend_device)
                _append_optional_arg(command, "--decode-device", args.decode_device)
                _append_optional_arg(command, "--video-num-inference-steps", args.video_num_inference_steps)
                _append_optional_arg(command, "--action-num-inference-steps", args.action_num_inference_steps)
                _append_optional_arg(command, "--guidance-scale", args.guidance_scale)
                _append_optional_arg(command, "--action-guidance-scale", args.action_guidance_scale)
                jobs.append(
                    AblationJob(
                        case=case,
                        profile=profile,
                        target_action_hz=float(target_hz),
                        command=command,
                        summary_path=None,
                    )
                )
    return jobs


def _resolve_cases(raw: str, *, local_path_registry: Mapping[str, str]) -> list[RolloutCase]:
    names = _parse_csv(raw)
    if names == ["all"]:
        names = list(DEFAULT_CASES)
    cases: list[RolloutCase] = []
    for item in names:
        if item in DEFAULT_CASES:
            case = DEFAULT_CASES[item]
            if case.checkpoint is not None:
                _resolve_path_token(case.checkpoint, local_path_registry=local_path_registry)
            cases.append(case)
            continue
        parts = item.split(":", 2)
        if len(parts) not in {2, 3}:
            raise ValueError(
                f"Unknown case {item!r}. Use a builtin name, `all`, or `name:config[:checkpoint]`."
            )
        cases.append(RolloutCase(name=parts[0], config=Path(parts[1]), checkpoint=parts[2] if len(parts) == 3 else None))
    return cases


def _resolve_profiles(raw: str) -> list[PlannerProfile]:
    names = _parse_csv(raw)
    if names == ["all"]:
        names = list(DEFAULT_PROFILES)
    missing = [name for name in names if name not in DEFAULT_PROFILES]
    if missing:
        raise ValueError(f"Unknown planner profile(s): {', '.join(missing)}")
    return [DEFAULT_PROFILES[name] for name in names]


def _load_local_path_registry() -> dict[str, str]:
    return load_local_path_registry()


def _resolve_path_token(value: str | None, *, local_path_registry: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    pattern = re.compile(r"\$\{([^}]+)\}")

    def replace(match: re.Match[str]) -> str:
        alias = match.group(1)
        registry_key = alias[6:] if alias.startswith("paths.") else alias
        resolved = local_path_registry.get(registry_key)
        if resolved is None:
            raise KeyError(
                f"Missing local path alias {alias!r} in the merged local path registry. "
                "Define it in `configs/local_paths.yaml`, add it to `configs/local_paths.sample.yaml`, "
                f"or point {LOCAL_PATHS_ENV_VAR} at the registry file that provides it."
            )
        return str(resolved)

    previous = value
    for _ in range(8):
        current = pattern.sub(replace, previous)
        if current == previous:
            return current
        previous = current
    raise ValueError(f"Path alias expansion did not converge for {value!r}.")


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    values = [float(item) for item in _parse_csv(raw)]
    if not values:
        raise ValueError("Expected at least one target action Hz value.")
    if any(value <= 0.0 for value in values):
        raise ValueError("Target action Hz values must be positive.")
    return values


def _parse_env_overrides(raw_values: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in raw_values:
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise ValueError(f"Expected --env entries to have form KEY=VALUE, got {raw!r}.")
        env[key] = value
    return env


def _append_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _safe_token(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or "run"


def _job_to_report(job: AblationJob, *, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "case": job.case.name,
        "case_description": job.case.description,
        "profile": job.profile.name,
        "profile_description": job.profile.description,
        "target_action_hz": float(job.target_action_hz),
        "command": job.command,
    }


def _load_summary_if_available(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    candidate: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if "summary_path" in value or "executed_actions" in value:
                candidate = value
            elif candidate is None:
                candidate = value
    return candidate


def _write_summary(path: Path, reports: list[dict[str, Any]]) -> None:
    payload = {
        "reports": reports,
        "passed": sum(1 for report in reports if report.get("status") == "passed"),
        "failed": sum(1 for report in reports if report.get("status") == "failed"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
