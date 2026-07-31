from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.evals import libero_mot_rollout as mot_viz  # noqa: E402
from open_wam.utils import seed_everywhere  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run many LIBERO MoT rollouts while loading the config/checkpoint/pipeline once. "
            "This preserves the single-rollout semantics from run_libero_mot_visualization.py."
        )
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        required=True,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Apply a config override such as `--set policy_variant.generalist_mode_text_token=true`.",
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument(
        "--task-ids",
        "--task-id",
        dest="task_ids",
        type=str,
        required=True,
        help="Task ids to evaluate, e.g. `0-9`, `0,3,7`, `0-2,5`, or a single `0`.",
    )
    parser.add_argument(
        "--episode-idxs",
        "--episode-idx",
        dest="episode_idxs",
        type=str,
        required=True,
        help="Episode indices to evaluate, e.g. `0-49`, `0,3,7`, `0-2,5`, or a single `0`.",
    )
    parser.add_argument(
        "--loop-order",
        choices=("task_episode", "episode_task"),
        default="task_episode",
        help="Outer loop ordering. Use task_episode to run all episodes of a task before switching task.",
    )
    parser.add_argument("--merge-checkpoint-runtime-config", action="store_true")
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--raw-window-frames", type=int, default=None)
    parser.add_argument(
        "--execute-action-steps",
        type=int,
        default=None,
        help=(
            "Execute only the first N predicted actions from each MoT chunk before replanning. "
            "Defaults to the full action horizon. N must be positive, <= action_horizon, "
            "and aligned to action_per_frame."
        ),
    )
    parser.add_argument(
        "--execute-frame-chunk-size",
        type=int,
        default=None,
        help=(
            "Execute only the first N latent-frame groups from each MoT chunk before replanning. "
            "This preserves the model's configured inference.frame_chunk_size and maps to "
            "N * action_per_frame executed actions."
        ),
    )
    parser.add_argument(
        "--mot-rollout-frame-chunk-size",
        type=int,
        default=None,
        help=(
            "Override MoT's internal inference chunk to N latent frames. Unlike "
            "--execute-frame-chunk-size, this reduces the generated video frames and action horizon "
            "inside the policy while preserving the checkpoint's action_per_frame."
        ),
    )
    parser.add_argument("--mot-inference-window-size", type=int, default=None)
    parser.add_argument(
        "--mot-action-only-rollout",
        action="store_true",
        help=(
            "Skip imagined-video denoising during MoT rollout and produce actions only. "
            "Supported only for action_then_video and decoupled_same_step couplings."
        ),
    )
    parser.add_argument(
        "--mot-gjd-action-route",
        choices=sorted(mot_viz.MOT_GJD_ACTION_ROUTES),
        default="joint",
        help=(
            "Diagnostic M5 GJD live-sim action route. `joint` is the normal rollout; "
            "`joint_video_then_idm` generates video with joint denoising and executes "
            "IDM actions conditioned on that generated video."
        ),
    )
    parser.add_argument(
        "--frontend-encode-mode",
        choices=(mot_viz.DEPRECATED_FRONTEND_ENCODE_MODE, mot_viz.CURRENT_FRONTEND_ENCODE_MODE),
        default=mot_viz.CURRENT_FRONTEND_ENCODE_MODE,
        help=(
            "Current rollout contract is lingbot_streaming_vae. rolling_offline is deprecated "
            "and requires --allow-deprecated-frontend-encode-mode."
        ),
    )
    parser.add_argument("--startup-model-obs-frames", type=int, default=1)
    parser.add_argument("--startup-env-init-steps", type=int, default=5)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_mot_visualization_batch")
    parser.add_argument("--suffix", type=str, default="open_wam_mot")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma/range list of rollout seeds to evaluate with one loaded pipeline. "
            "Each seed appends `_seedN` to the per-rollout suffix."
        ),
    )
    parser.add_argument(
        "--seed-by-episode",
        action="store_true",
        help="Use episode_idx as the per-rollout seed, matching shell loops that pass `--seed ${EP}`.",
    )
    parser.add_argument(
        "--reuse-env-per-task",
        action="store_true",
        help=(
            "Reuse one LIBERO env across all episodes of the current task. "
            "Requires --loop-order task_episode and matches FastWAM-style task-level env reuse."
        ),
    )
    parser.add_argument(
        "--skip-comparison-video",
        action="store_true",
        help=(
            "Skip imagined-video decode and comparison-video writing. Summaries, actions, "
            "chunk logs, and optional --save-rollout-video are still written."
        ),
    )
    parser.add_argument("--save-rollout-video", action="store_true")
    parser.add_argument("--max-imagined-latent-frames", type=int, default=None)
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--action-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument("--reset-policy-state-each-chunk", action="store_true")
    parser.add_argument("--allow-deprecated-libero-config", action="store_true")
    parser.add_argument("--allow-deprecated-frontend-encode-mode", action="store_true")
    args = parser.parse_args()
    if args.seeds is not None and (args.seed is not None or args.seed_by_episode):
        parser.error("--seeds is mutually exclusive with --seed and --seed-by-episode.")

    resources = _load_batch_resources(args)
    task_ids = _parse_int_ranges(args.task_ids, label="task-ids")
    episode_idxs = _parse_int_ranges(args.episode_idxs, label="episode-idxs")
    if args.reuse_env_per_task and args.loop_order != "task_episode":
        raise ValueError("--reuse-env-per-task requires --loop-order task_episode.")
    pairs = list(_iter_pairs(task_ids, episode_idxs, loop_order=args.loop_order))
    rollout_seed_specs = _resolve_rollout_seed_specs(args)

    summaries: list[dict[str, object]] = []
    base_suffix = str(args.suffix)
    try:
        for suffix, explicit_seed in rollout_seed_specs:
            args.suffix = suffix
            for task_id, episode_idx in pairs:
                rollout_seed = explicit_seed
                if rollout_seed is None:
                    rollout_seed = _resolve_rollout_seed(args, episode_idx=episode_idx)
                mot_viz.print_rollout_event(
                    "batch_rollout_start",
                    {"task_id": int(task_id), "episode_idx": int(episode_idx), "seed": rollout_seed},
                )
                summary = _run_one_loaded_rollout(
                    args,
                    resources,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    seed=rollout_seed,
                )
                summaries.append(summary)
                mot_viz.print_rollout_event(
                    "batch_rollout_done",
                    {
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "success": bool(summary.get("success", False)),
                        "env_timestep": int(summary.get("env_timestep", 0)),
                    },
                )
    finally:
        args.suffix = base_suffix
        _close_reused_env(resources)

    batch_summary_path = Path(args.output_dir) / f"{args.suffix}_batch_summary.json"
    batch_summary_path.parent.mkdir(parents=True, exist_ok=True)
    batch_summary_path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"batch_summary_path": str(batch_summary_path.resolve()), "rollouts": len(summaries)}, indent=2))


def _load_batch_resources(args: argparse.Namespace) -> SimpleNamespace:
    runtime = mot_viz.load_mot_libero_runtime(
        mot_viz.MotLiberoLoadOptions(
            config=args.config,
            checkpoint=args.checkpoint,
            merge_checkpoint_runtime_config=bool(
                args.merge_checkpoint_runtime_config
            ),
            set_overrides=tuple(args.set_overrides),
            source="run_libero_mot_batch_visualization.py",
            checkpoint_error="MoT batch visualization requires --checkpoint.",
            raw_window_frames=args.raw_window_frames,
            startup_model_obs_frames=args.startup_model_obs_frames,
            startup_env_init_steps=args.startup_env_init_steps,
            mot_inference_window_size=args.mot_inference_window_size,
            mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
            mot_action_only_rollout=bool(args.mot_action_only_rollout),
            mot_generalist_rollout_mode=None,
            mot_gjd_action_route=args.mot_gjd_action_route,
            execute_action_steps=args.execute_action_steps,
            execute_frame_chunk_size=args.execute_frame_chunk_size,
            frontend_encode_mode=args.frontend_encode_mode,
            reset_policy_state_each_chunk=bool(
                args.reset_policy_state_each_chunk
            ),
            runtime_device=args.runtime_device,
            action_device=args.action_device,
            frontend_device=args.frontend_device,
            decode_device=args.decode_device,
            allow_deprecated_libero_config=bool(
                args.allow_deprecated_libero_config
            ),
            allow_deprecated_frontend_encode_mode=bool(
                args.allow_deprecated_frontend_encode_mode
            ),
            component_report_extra={
                "batch_driver": "run_libero_mot_batch_visualization.py"
            },
        )
    )
    return SimpleNamespace(
        runtime=runtime,
        task_cache={},
        reused_env=None,
        reused_env_task_id=None,
    )


def _run_one_loaded_rollout(
    args: argparse.Namespace,
    resources: SimpleNamespace,
    *,
    task_id: int,
    episode_idx: int,
    seed: int | None,
) -> dict[str, object]:
    if seed is not None:
        # Preserve the loaded-once driver's historical pre-reset seeding.
        seed_everywhere(seed)
    task_resources = _resolve_task(
        resources,
        args.benchmark,
        task_id,
    )
    env, close_env_after_rollout = _acquire_rollout_env(
        args,
        resources,
        task_spec=task_resources.task_spec,
        task_id=task_id,
    )
    episode = mot_viz.MotLiberoEpisodeOptions(
        benchmark=args.benchmark,
        task_id=task_id,
        episode_idx=episode_idx,
        max_timestep=args.max_timestep,
        max_chunks=args.max_chunks,
        execute_action_steps=args.execute_action_steps,
        execute_frame_chunk_size=args.execute_frame_chunk_size,
        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
        mot_inference_window_size=args.mot_inference_window_size,
        mot_action_only_rollout=bool(args.mot_action_only_rollout),
        mot_generalist_rollout_mode=None,
        mot_gjd_action_route=args.mot_gjd_action_route,
        reset_policy_state_each_chunk=bool(args.reset_policy_state_each_chunk),
        max_imagined_latent_frames=args.max_imagined_latent_frames,
        output_dir=args.output_dir,
        suffix=args.suffix,
        video_fps=args.video_fps,
        seed=seed,
        save_rollout_video=bool(args.save_rollout_video),
        skip_comparison_video=bool(args.skip_comparison_video),
    )
    return mot_viz.run_mot_libero_episode(
        episode,
        resources.runtime,
        task_resources,
        env,
        include_episode_coordinates=True,
        close_env_after_rollout=close_env_after_rollout,
    )


def _resolve_task(resources: SimpleNamespace, benchmark_name: str, task_id: int):
    cache_key = (benchmark_name, int(task_id))
    if cache_key not in resources.task_cache:
        resources.task_cache[cache_key] = mot_viz.resolve_mot_libero_task_resources(
            benchmark_name,
            int(task_id),
        )
    return resources.task_cache[cache_key]


def _acquire_rollout_env(
    args: argparse.Namespace,
    resources: SimpleNamespace,
    *,
    task_spec,
    task_id: int,
):
    if not args.reuse_env_per_task:
        return mot_viz.construct_mot_libero_env(task_spec), True
    if resources.reused_env is not None and resources.reused_env_task_id != int(task_id):
        _close_reused_env(resources)
    if resources.reused_env is None:
        resources.reused_env = mot_viz.construct_mot_libero_env(task_spec)
        resources.reused_env_task_id = int(task_id)
    return resources.reused_env, False


def _close_reused_env(resources: SimpleNamespace) -> None:
    env = getattr(resources, "reused_env", None)
    if env is not None:
        env.close()
    resources.reused_env = None
    resources.reused_env_task_id = None


def _parse_int_ranges(raw: str, *, label: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid {label} range {token!r}: end < start.")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise ValueError(f"Expected at least one value for --{label}.")
    return values


def _iter_pairs(task_ids: list[int], episode_idxs: list[int], *, loop_order: str):
    if loop_order == "task_episode":
        for task_id in task_ids:
            for episode_idx in episode_idxs:
                yield task_id, episode_idx
    elif loop_order == "episode_task":
        for episode_idx in episode_idxs:
            for task_id in task_ids:
                yield task_id, episode_idx
    else:  # pragma: no cover - argparse choices should prevent this
        raise ValueError(f"Unsupported loop_order={loop_order!r}.")


def _resolve_rollout_seed(args: argparse.Namespace, *, episode_idx: int) -> int | None:
    if args.seed_by_episode:
        return int(episode_idx)
    if args.seed is None:
        return None
    return int(args.seed)


def _resolve_rollout_seed_specs(args: argparse.Namespace) -> list[tuple[str, int | None]]:
    base_suffix = str(args.suffix)
    if args.seeds is None:
        return [(base_suffix, None)]
    seeds = _parse_int_ranges(args.seeds, label="seeds")
    if not seeds:
        raise ValueError("--seeds must resolve to at least one seed.")
    return [(f"{base_suffix}_seed{int(seed)}", int(seed)) for seed in seeds]


if __name__ == "__main__":
    main()
