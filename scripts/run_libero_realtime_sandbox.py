from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _prepend_import_path(path: Path) -> None:
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


_prepend_import_path(SRC_ROOT)

from open_wam.runtime.policy_planner import PolicyPlanner
from open_wam.configs import (
    LiberoRendererProfile,
    load_experiment_config,
)
from open_wam.configs.enums import (
    DeadlineMissPolicy,
    RealtimeEmptyPlanPolicy,
    RealtimePlannerMode,
    RealtimeSchedulerProfile,
    RolloutArtifactProfile,
)
from open_wam.evals import libero_rollout_artifacts as rollout_artifacts
from open_wam.evals import libero_visualization as libero_visualization
from open_wam.integrations import (
    activate_libero_renderer,
    ensure_local_libero_config,
    libero_rollout,
    load_libero_task_init_states,
    resolve_libero_task_by_id,
)
from open_wam.integrations.realtime_control import (
    build_live_rollout_summary,
)
from open_wam.runtime.realtime_scheduling import (
    resolve_realtime_scheduler_defaults,
)
from open_wam.models.visual_tower import (
    resolve_runtime_backbone_dir,
)
from open_wam.pipelines import (
    VariantRolloutRunner,
    build_variant_pipeline_from_config,
)
from open_wam.runtime import checkpoints as runtime_checkpoints
from open_wam.runtime import rollout as rollout_runtime
from open_wam.runtime.checkpoint_artifacts import is_usable_transformer_dir
from open_wam.utils import (
    apply_config_overrides,
    merge_runtime_config_from_checkpoint,
    parse_override_assignments,
    resolve_transformer_dir_override,
    seed_everywhere,
    validate_positive_step_override,
)
from open_wam.utils.libero_paradigm import (
    require_current_libero_policy_paradigm,
)

VERBOSE = False


EVAL_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "debug_short": {},
    "libero_10hz_full": {
        "max_actions": 3000,
        "env_horizon": 5000,
        "target_action_hz": 10.0,
        "deadline_miss_policy": DeadlineMissPolicy.HOLD_STATE.value,
    },
}


def _apply_realtime_cli_profiles(args: argparse.Namespace, argv: list[str]) -> None:
    eval_defaults = EVAL_PROFILE_DEFAULTS.get(str(args.eval_profile))
    if eval_defaults is None:
        raise ValueError(f"Unsupported eval profile: {args.eval_profile!r}")
    _apply_cli_profile_defaults(
        args,
        argv,
        eval_defaults,
        flag_aliases={
            "max_actions": ("--max-actions",),
            "env_horizon": ("--env-horizon",),
            "target_action_hz": ("--target-action-hz",),
            "video_fps": ("--video-fps",),
            "deadline_miss_policy": ("--deadline-miss-policy",),
        },
    )

    try:
        scheduler_defaults = resolve_realtime_scheduler_defaults(
            args.realtime_scheduler_profile,
        )
    except ValueError as error:
        raise ValueError(
            f"Unsupported realtime scheduler profile: {args.realtime_scheduler_profile!r}"
        ) from error
    _apply_cli_profile_defaults(
        args,
        argv,
        scheduler_defaults.to_override_mapping(),
        flag_aliases={
            "planner_mode": ("--planner-mode",),
            "sequence_empty_plan_policy": ("--sequence-empty-plan-policy",),
            "startup_open_loop_chunks": ("--startup-open-loop-chunks",),
            "replan_low_watermark_actions": ("--replan-low-watermark-actions",),
        },
    )
    args.realtime_scheduler_profile = RealtimeSchedulerProfile(
        args.realtime_scheduler_profile
    )
    args.planner_mode = RealtimePlannerMode(args.planner_mode)
    args.sequence_empty_plan_policy = RealtimeEmptyPlanPolicy(
        args.sequence_empty_plan_policy
    )


def _apply_cli_profile_defaults(
    args: argparse.Namespace,
    argv: list[str],
    defaults: dict[str, object],
    *,
    flag_aliases: dict[str, tuple[str, ...]],
) -> None:
    for attr, value in defaults.items():
        if _cli_flag_present(argv, *flag_aliases.get(attr, ())):
            continue
        setattr(args, attr, value)


def _cli_flag_present(argv: list[str], *flags: str) -> bool:
    for token in argv:
        for flag in flags:
            if token == flag or token.startswith(f"{flag}="):
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one trained LIBERO policy in a fixed-rate realtime sandbox across exact/joint, "
            "feature-attached and dual-expert policy architectures."
        )
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/parallel_stream_libero_video_then_action.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file, checkpoint_step_* directory, or run directory. "
        "If omitted, infer it from the configured runtime-backbone artifact.",
    )
    parser.add_argument(
        "--transformer-dir",
        type=str,
        default=None,
        help=(
            "Exact/joint exported-transformer override. This intentionally does "
            "not merge checkpoint resolved_config.yaml."
        ),
    )
    parser.add_argument(
        "--pretrained-model-root",
        type=str,
        default=None,
        help=(
            "Optional reference asset root override for VAE/text/tokenizer assets. "
            "Use this with --transformer-dir when comparing against an external full-model export."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Apply training-style config overrides after checkpoint runtime-config merging.",
    )
    parser.add_argument(
        "--merge-checkpoint-runtime-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge the checkpoint's resolved runtime settings before explicit --set overrides.",
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--episode-idx", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument(
        "--env-horizon",
        type=int,
        default=None,
        help=(
            "Override the LIBERO/robosuite environment horizon (max env steps "
            "before the episode auto-terminates). Set this to a value >= "
            "--max-actions when running long realtime rollouts at high "
            "control rates, otherwise robosuite raises 'executing action in "
            "terminated episode' once the default horizon (typically 600) is "
            "reached. None keeps the upstream LIBERO default."
        ),
    )
    parser.add_argument("--target-action-hz", type=float, default=10.0)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument(
        "--eval-profile",
        choices=tuple(EVAL_PROFILE_DEFAULTS),
        default="debug_short",
        help=(
            "Named rollout defaults. `debug_short` preserves the historical short sandbox defaults. "
            "`libero_10hz_full` sets the long 10 Hz LIBERO evaluation protocol."
        ),
    )
    parser.add_argument(
        "--write-fallback-timeline-video",
        action="store_true",
        help=(
            "Also render a debug MP4 that includes fallback-history decisions and the full fallback timeline. "
            "Disabled by default because it duplicates frame materialization and video encoding work."
        ),
    )
    parser.add_argument(
        "--artifact-profile",
        type=RolloutArtifactProfile,
        choices=tuple(RolloutArtifactProfile),
        default=RolloutArtifactProfile.STANDARD,
        help=(
            "Output artifact set. `lean` writes only the summary JSON; "
            "`standard` preserves the historical rollout MP4, trace JSONL files, and load report; "
            "`debug` also writes the fallback-timeline MP4."
        ),
    )
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--runtime-devices", type=str, default=None)
    parser.add_argument("--runtime-prep-device", type=str, default=None)
    parser.add_argument("--runtime-output-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument(
        "--reference-assets-device-policy",
        type=str,
        choices=("cpu_offload", "runtime"),
        default="runtime",
    )
    parser.add_argument(
        "--video-num-inference-steps",
        type=int,
        default=None,
        help="Optional override for inference.video_num_inference_steps. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--action-num-inference-steps",
        type=int,
        default=None,
        help="Optional override for inference.action_num_inference_steps. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Optional override for inference.guidance_scale. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--action-guidance-scale",
        type=float,
        default=None,
        help="Optional override for inference.action_guidance_scale. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--planner-mode",
        type=RealtimePlannerMode,
        choices=tuple(RealtimePlannerMode),
        default=RealtimePlannerMode.ASYNC_BUFFER,
    )
    parser.add_argument(
        "--realtime-scheduler-profile",
        type=RealtimeSchedulerProfile,
        choices=tuple(RealtimeSchedulerProfile),
        default=RealtimeSchedulerProfile.MANUAL,
        help="Named realtime scheduler defaults; explicit low-level scheduler flags still override the profile.",
    )
    parser.add_argument("--sequence-buffer-threshold", type=int, default=3)
    parser.add_argument(
        "--execute-prefix-actions",
        type=int,
        default=None,
        help="Execute a whole-model-frame prefix of each prediction before replanning.",
    )
    parser.add_argument(
        "--sequence-empty-plan-policy",
        type=RealtimeEmptyPlanPolicy,
        choices=tuple(RealtimeEmptyPlanPolicy),
        default=RealtimeEmptyPlanPolicy.FALLBACK,
        help=(
            "All rollout-capable policies. `fallback` preserves strict fixed-rate behavior. "
            "`wait_for_replan` blocks the sim when the next action chunk is late and executes the model plan."
        ),
    )
    parser.add_argument(
        "--startup-open-loop-chunks",
        type=int,
        default=0,
        help=(
            "Open-loop startup: precompute this many model open-loop chunks before the live clock starts. "
            "This avoids fallback without using future observations, but it is not observation-conditioned replanning."
        ),
    )
    parser.add_argument(
        "--replan-low-watermark-actions",
        dest="replan_low_watermark_actions",
        type=int,
        default=0,
        help=(
            "Submit asynchronous planning when at most K actions remain in the buffer. "
            "The unit is control steps, not video frames."
        ),
    )
    parser.add_argument(
        "--deadline-miss-policy",
        type=str,
        choices=tuple(policy.value for policy in DeadlineMissPolicy),
        default=DeadlineMissPolicy.HOLD_STATE.value,
        help=(
            "`hold_state` zeroes delta-motion channels and preserves configured absolute-tail channels; "
            "for ACTION_COMMAND gripper configs with no absolute tail, it zeroes the full action. "
            "`hold_last` repeats the previous raw action. `zero` sends all zeros."
        ),
    )
    parser.add_argument("--deadline-tolerance-ms", type=float, default=2.0)
    parser.add_argument(
        "--output-dir", type=str, default="outputs/libero_realtime_validation"
    )
    parser.add_argument("--suffix", type=str, default="sandbox")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-deprecated-libero-config",
        action="store_true",
        help=(
            "Allow explicitly retired LIBERO config identities for historical debugging."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    activate_libero_renderer(LiberoRendererProfile.ONLINE_ROLLOUT)
    _apply_realtime_cli_profiles(args, sys.argv[1:])

    if args.max_actions <= 0:
        raise ValueError("--max-actions must be positive.")
    if args.env_horizon is not None and args.env_horizon <= 0:
        raise ValueError("--env-horizon must be positive when provided.")
    if args.target_action_hz <= 0:
        raise ValueError("--target-action-hz must be positive.")
    if args.sequence_buffer_threshold < 0:
        raise ValueError("--sequence-buffer-threshold must be non-negative.")
    if args.startup_open_loop_chunks < 0:
        raise ValueError("--startup-open-loop-chunks must be non-negative.")
    if args.replan_low_watermark_actions < 0:
        raise ValueError("--replan-low-watermark-actions must be non-negative.")

    global VERBOSE
    VERBOSE = bool(args.verbose)

    seed_everywhere(args.seed)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    if args.transformer_dir is not None:
        if args.checkpoint is not None:
            raise ValueError(
                "--transformer-dir initializes the backbone and cannot be combined with --checkpoint."
            )
        checkpoint_path = None
        config = replace(
            config,
            backbone=replace(
                config.backbone,
                runtime_backbone_artifact_path=str(
                    resolve_transformer_dir_override(args.transformer_dir)
                ),
            ),
        )
        checkpoint_runtime_config_path = None
    else:
        checkpoint_path = _resolve_checkpoint_path_for_config(
            config=config, checkpoint_arg=args.checkpoint
        )
        should_merge_checkpoint_runtime_config = args.merge_checkpoint_runtime_config
        if should_merge_checkpoint_runtime_config:
            config, checkpoint_runtime_config_path = (
                merge_runtime_config_from_checkpoint(config, checkpoint_path)
            )
            if checkpoint_runtime_config_path is not None and VERBOSE:
                print(
                    "[realtime_sandbox] merged checkpoint runtime config "
                    f"{checkpoint_runtime_config_path} into {config_path}",
                    file=sys.stderr,
                )
        else:
            checkpoint_runtime_config_path = None
        config = _apply_checkpoint_backbone_override(
            config, checkpoint_path=checkpoint_path
        )
    if args.set_overrides:
        config = apply_config_overrides(
            config,
            parse_override_assignments(tuple(args.set_overrides)),
        )
    if args.pretrained_model_root is not None:
        pretrained_model_root = Path(args.pretrained_model_root).expanduser().resolve()
        if not pretrained_model_root.is_dir():
            raise FileNotFoundError(
                f"--pretrained-model-root must be an existing directory: {pretrained_model_root}"
            )
        config = replace(
            config,
            backbone=replace(
                config.backbone,
                pretrained_model_name_or_path=str(pretrained_model_root),
            ),
        )
    config = replace(
        config,
        backbone=replace(
            config.backbone,
            reference_assets_device_policy=args.reference_assets_device_policy,
        ),
    )
    require_current_libero_policy_paradigm(
        config,
        config_path=config_path,
        source="run_libero_realtime_sandbox.py",
        allow_deprecated=bool(args.allow_deprecated_libero_config),
    )

    runtime_device = libero_visualization.resolve_device(args.runtime_device)
    frontend_device = libero_visualization.resolve_device(
        args.frontend_device, fallback=runtime_device
    )
    runtime_devices = rollout_runtime.resolve_runtime_devices(
        args.runtime_devices,
        fallback=runtime_device,
    )
    runtime_prep_device = libero_visualization.resolve_device(
        args.runtime_prep_device, fallback=runtime_device
    )
    runtime_output_device = libero_visualization.resolve_device(
        args.runtime_output_device, fallback=runtime_device
    )
    summary = _run_realtime_rollout(
        args=args,
        config=config,
        checkpoint_path=checkpoint_path,
        runtime_device=runtime_device,
        frontend_device=frontend_device,
        runtime_devices=runtime_devices,
        runtime_prep_device=runtime_prep_device,
        runtime_output_device=runtime_output_device,
    )
    print(json.dumps(summary, indent=2))


def _resolve_checkpoint_path_for_config(
    *, config, checkpoint_arg: str | None
) -> Path | None:
    if checkpoint_arg is not None:
        return runtime_checkpoints.resolve_checkpoint_file(Path(checkpoint_arg))
    runtime_backbone_dir = resolve_runtime_backbone_dir(config.backbone)
    if runtime_backbone_dir is None:
        return None
    try:
        return runtime_checkpoints.resolve_checkpoint_file(
            runtime_checkpoints.resolve_checkpoint_step_dir_from_transformer_dir(
                runtime_backbone_dir
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        if VERBOSE:
            print(
                "[realtime_sandbox] Failed to infer a checkpoint file from "
                f"runtime backbone {runtime_backbone_dir}: {exc}",
                file=sys.stderr,
            )
        return None


def _apply_checkpoint_backbone_override(config, *, checkpoint_path: Path | None):
    if checkpoint_path is None:
        return config
    checkpoint_step_dir = checkpoint_path.parent
    transformer_dir = checkpoint_step_dir / "transformer"
    if is_usable_transformer_dir(transformer_dir):
        return replace(
            config,
            backbone=replace(
                config.backbone,
                runtime_backbone_artifact_path=str(transformer_dir.resolve()),
            ),
        )
    return config


def _apply_common_inference_overrides(
    config,
    *,
    video_num_inference_steps: int | None,
    action_num_inference_steps: int | None,
    guidance_scale: float | None,
    action_guidance_scale: float | None,
):
    video_steps = validate_positive_step_override(
        "video_num_inference_steps",
        video_num_inference_steps,
    )
    action_steps = validate_positive_step_override(
        "action_num_inference_steps",
        action_num_inference_steps,
    )
    overrides = {
        key: value
        for key, value in {
            "video_num_inference_steps": video_steps,
            "action_num_inference_steps": action_steps,
            "guidance_scale": guidance_scale,
            "action_guidance_scale": action_guidance_scale,
        }.items()
        if value is not None
    }
    return replace(config, inference=replace(config.inference, **overrides))


def _construct_realtime_libero_env(task_spec, *, env_horizon: int | None):
    activate_libero_renderer(LiberoRendererProfile.ONLINE_ROLLOUT)
    ensure_local_libero_config(REPO_ROOT)
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    count = 0
    env = None
    while env is None and count < 5:
        try:
            kwargs: dict[str, Any] = {
                "bddl_file_name": task_spec.bddl_file_path,
                "camera_heights": 128,
                "camera_widths": 128,
            }
            if env_horizon is not None:
                kwargs["horizon"] = int(env_horizon)
            env = OffScreenRenderEnv(**kwargs)
        except Exception as exc:
            print(f"construct env failed ({count + 1}/5): {exc}")
            time.sleep(5)
            count += 1
    return env


def _run_realtime_rollout(
    *,
    args,
    config,
    checkpoint_path,
    runtime_device,
    frontend_device,
    runtime_devices,
    runtime_prep_device,
    runtime_output_device,
):
    from open_wam.integrations.libero_realtime import LiberoRolloutAdapter
    from open_wam.runtime.rollout_engine import RolloutEngine, RolloutOptions

    LiberoRolloutAdapter.validate_config(config)
    config = _apply_common_inference_overrides(
        config,
        video_num_inference_steps=args.video_num_inference_steps,
        action_num_inference_steps=args.action_num_inference_steps,
        guidance_scale=args.guidance_scale,
        action_guidance_scale=args.action_guidance_scale,
    )
    pipeline = build_variant_pipeline_from_config(config)
    checkpoint_report = None
    if checkpoint_path is not None:
        checkpoint_report = runtime_checkpoints.load_pipeline_checkpoint(
            pipeline,
            checkpoint_path,
            compatibility=runtime_checkpoints.CheckpointCompatibilityPolicy.ALLOW_CHECKPOINT_SUPERSET,
        )
    pipeline.to(runtime_device).eval()
    pipeline.visual_tower.configure_runtime_devices(
        runtime_devices,
        prep_device=runtime_prep_device,
        output_device=runtime_output_device,
    )
    runner = VariantRolloutRunner(pipeline)
    task = resolve_libero_task_by_id(args.benchmark, args.task_id, REPO_ROOT)
    states = load_libero_task_init_states(task, REPO_ROOT)
    env = _construct_realtime_libero_env(task, env_horizon=args.env_horizon)
    if env is None:
        raise RuntimeError("Failed to construct the LIBERO environment.")
    try:
        initial = libero_rollout.initialize_libero_observation_window(
            env,
            states[args.episode_idx % len(states)],
            num_frames=1,
        )[-1]
        adapter = LiberoRolloutAdapter(
            runner=runner,
            config=config,
            environment=env,
            prompt=task.task_language,
            frontend_device=frontend_device,
            runtime_device=runtime_device,
            deadline_miss_policy=DeadlineMissPolicy(args.deadline_miss_policy),
        )
        options = RolloutOptions(
            max_actions=args.max_actions,
            target_action_hz=args.target_action_hz,
            planner_mode=args.planner_mode,
            empty_plan_policy=args.sequence_empty_plan_policy,
            buffer_threshold=args.sequence_buffer_threshold,
            replan_low_watermark_actions=args.replan_low_watermark_actions,
            startup_open_loop_chunks=args.startup_open_loop_chunks,
            execute_prefix_actions=args.execute_prefix_actions,
        )
        result = RolloutEngine(PolicyPlanner(runner, adapter), adapter, options).run(
            initial, runner.reset(task_text=(task.task_language,))
        , step=adapter.step)
        action_records = [receipt.to_record() for receipt in result.actions]
        replan_records = [receipt.to_record() for receipt in result.replans]
        extension_records = [receipt.to_record() for receipt in result.extensions]
        summary = build_live_rollout_summary(
            action_records=action_records,
            replan_records=replan_records,
            target_action_hz=args.target_action_hz,
            live_wall_time_s=result.live_wall_time_s,
            startup_prepare_s=result.startup.prepare_s,
            startup_infer_s=result.startup.infer_s,
            deadline_tolerance_s=args.deadline_tolerance_ms / 1000,
        )
        summary.update(
            benchmark=args.benchmark,
            task_id=args.task_id,
            episode_idx=args.episode_idx,
            success=result.success,
            termination_reason=result.termination.reason.value,
            planner_teardown_error=result.planner_teardown.error,
            planner_drain_time_s=result.planner_teardown.drain_time_s,
            executed_actions=len(result.actions),
            seed=args.seed,
            prompt=task.task_language,
            startup_plan_trace=result.startup.to_record(),
            history_replan_count=len(result.replans),
            open_loop_extension_count=len(result.extensions),
        )
        report = {
            "pipeline": "variant_rollout_engine",
            "policy_variant": config.policy_variant.name,
            "runtime_device": str(runtime_device),
            "frontend_device": str(frontend_device),
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "missing_keys": []
            if checkpoint_report is None
            else list(checkpoint_report.missing_keys),
            "unexpected_keys": []
            if checkpoint_report is None
            else list(checkpoint_report.unexpected_keys),
        }
        return _finalize_rollout_outputs(
            summary=summary,
            action_records=action_records,
            action_video_records=[
                dict(record, obs=obs)
                for record, obs in zip(action_records, result.observations, strict=True)
            ],
            replan_records=replan_records,
            extension_records=extension_records,
            component_report=report,
            output_dir=Path(args.output_dir),
            benchmark=args.benchmark,
            task_id=args.task_id,
            prompt=task.task_language,
            episode_idx=args.episode_idx,
            suffix=args.suffix,
            video_fps=args.video_fps or args.target_action_hz,
            action_per_frame=pipeline.policy_variant.rollout_contract.action_tokens_per_frame,
            write_fallback_timeline_video=args.write_fallback_timeline_video,
            artifact_profile=args.artifact_profile,
        )
    finally:
        env.close()


def _finalize_rollout_outputs(
    *,
    summary: dict[str, Any],
    action_records: list[dict[str, Any]],
    action_video_records: list[dict[str, Any]],
    replan_records: list[dict[str, Any]],
    extension_records: list[dict[str, Any]],
    component_report: dict[str, Any],
    output_dir: Path,
    benchmark: str,
    task_id: int,
    prompt: str,
    episode_idx: int,
    suffix: str,
    video_fps: float,
    action_per_frame: int,
    write_fallback_timeline_video: bool,
    artifact_profile: RolloutArtifactProfile | str,
    startup_debug_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = rollout_artifacts.persist_libero_realtime_artifacts(
        identity=rollout_artifacts.LiberoRealtimeArtifactIdentity(
            benchmark=benchmark,
            task_id=task_id,
            prompt=prompt,
            episode_idx=episode_idx,
            suffix=suffix,
        ),
        options=rollout_artifacts.LiberoRealtimeArtifactOptions(
            output_root=output_dir,
            video_fps=video_fps,
            action_per_frame=action_per_frame,
            policy=rollout_artifacts.RolloutArtifactPolicy.from_value(
                artifact_profile,
                write_fallback_timeline_video=write_fallback_timeline_video,
            ),
        ),
        payload=rollout_artifacts.LiberoRealtimeArtifactPayload(
            action_records=action_records,
            action_video_records=action_video_records,
            replan_records=replan_records,
            extension_records=extension_records,
            component_report=component_report,
            startup_debug_report=startup_debug_report,
        ),
        summary=summary,
    )
    return output.summary


def _print_stage(name: str, **payload: object) -> None:
    if VERBOSE:
        print(json.dumps({"stage": name, **payload}), flush=True)


if __name__ == "__main__":
    main()
