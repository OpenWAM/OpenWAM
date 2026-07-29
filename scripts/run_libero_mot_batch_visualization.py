from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from open_wam.configs import ReferenceCoreInitMode  # noqa: E402
from open_wam.integrations import load_libero_task_init_states  # noqa: E402
from open_wam.models.policy_variants.mot.runtime_routing import ensure_mot_inference_backend  # noqa: E402
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils import (  # noqa: E402
    apply_config_overrides,
    load_experiment_config,
    parse_override_assignments,
    seed_everywhere,
)
from open_wam.utils.libero_paradigm import require_current_libero_policy_paradigm  # noqa: E402

_MOT_VIZ_PATH = REPO_ROOT / "scripts" / "run_libero_mot_visualization.py"
_spec = importlib.util.spec_from_file_location("open_wam_mot_single_visualization", _MOT_VIZ_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - importlib defensive guard
    raise RuntimeError(f"Failed to import MoT visualization helpers from {_MOT_VIZ_PATH}.")
mot_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mot_viz)


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
                mot_viz._print_log(
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
                mot_viz._print_log(
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
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    mot_viz._validate_mot_config(config)
    checkpoint_path = mot_viz._resolve_mot_checkpoint_path(
        config_path=config_path,
        checkpoint_arg=args.checkpoint,
        transformer_subdir=str(config.backbone.transformer_subdir),
    )
    if checkpoint_path is None:
        raise ValueError("MoT batch visualization requires --checkpoint.")
    config, checkpoint_runtime_config_path = mot_viz._maybe_merge_checkpoint_runtime_config(
        config,
        checkpoint_path,
        merge_enabled=bool(args.merge_checkpoint_runtime_config),
    )
    if args.set_overrides:
        config = apply_config_overrides(
            config,
            parse_override_assignments(tuple(args.set_overrides)),
        )
    mot_viz._validate_mot_config(config)
    require_current_libero_policy_paradigm(
        config,
        config_path=config_path,
        source="run_libero_mot_batch_visualization.py",
        allow_deprecated=bool(args.allow_deprecated_libero_config),
    )
    transformer_dir = checkpoint_path.parent / "transformer"
    if transformer_dir.is_dir():
        object.__setattr__(config.backbone, "transformer_subdir", str(transformer_dir.resolve()))
        object.__setattr__(config.backbone, "reference_core_init_mode", ReferenceCoreInitMode.FULL)

    runtime_device = mot_viz._resolve_device(args.runtime_device)
    action_device = mot_viz._resolve_device(args.action_device, fallback=runtime_device)
    frontend_device = mot_viz._resolve_device(args.frontend_device, fallback=runtime_device)
    decode_device = mot_viz._resolve_device(args.decode_device, fallback=frontend_device)
    raw_window_frames = (
        int(args.raw_window_frames)
        if args.raw_window_frames is not None
        else mot_viz._default_raw_window_frames(int(config.data.num_frames))
    )
    startup_model_obs_frames = int(args.startup_model_obs_frames)
    if startup_model_obs_frames <= 0:
        raise ValueError(f"Expected --startup-model-obs-frames to be positive, got {startup_model_obs_frames}.")
    if startup_model_obs_frames > raw_window_frames:
        raise ValueError(
            "--startup-model-obs-frames must be <= --raw-window-frames, "
            f"got startup_model_obs_frames={startup_model_obs_frames}, raw_window_frames={raw_window_frames}."
        )
    startup_env_init_steps = int(args.startup_env_init_steps)
    if startup_env_init_steps <= 0:
        raise ValueError(f"Expected --startup-env-init-steps to be positive, got {startup_env_init_steps}.")
    if args.mot_inference_window_size is not None and int(args.mot_inference_window_size) <= 0:
        raise ValueError(
            "Expected --mot-inference-window-size to be positive when provided, "
            f"got {args.mot_inference_window_size}."
        )
    if args.mot_rollout_frame_chunk_size is not None:
        rollout_frame_chunk_size = int(args.mot_rollout_frame_chunk_size)
        configured_frame_chunk_size = mot_viz._frame_chunk_size(config)
        if rollout_frame_chunk_size <= 0:
            raise ValueError(
                "Expected --mot-rollout-frame-chunk-size to be positive when provided, "
                f"got {args.mot_rollout_frame_chunk_size}."
            )
        if rollout_frame_chunk_size > configured_frame_chunk_size:
            raise ValueError(
                "--mot-rollout-frame-chunk-size cannot exceed configured inference.frame_chunk_size, "
                f"got override={rollout_frame_chunk_size}, configured={configured_frame_chunk_size}."
            )
    if args.execute_action_steps is not None or args.execute_frame_chunk_size is not None:
        mot_viz._resolve_execute_action_steps(
            args.execute_action_steps,
            execute_frame_chunk_size=args.execute_frame_chunk_size,
            action_horizon=int(config.data.action_schema.action_horizon),
            action_per_frame=mot_viz._action_per_frame(config),
        )
    mot_viz._require_current_frontend_encode_mode(
        args.frontend_encode_mode,
        allow_deprecated=bool(args.allow_deprecated_frontend_encode_mode),
        source="run_libero_mot_batch_visualization.py",
    )
    use_lingbot_streaming_vae = args.frontend_encode_mode == mot_viz.CURRENT_FRONTEND_ENCODE_MODE
    if use_lingbot_streaming_vae and startup_model_obs_frames != 1:
        raise ValueError(
            "`--frontend-encode-mode lingbot_streaming_vae` expects `--startup-model-obs-frames 1`."
        )
    if use_lingbot_streaming_vae and args.reset_policy_state_each_chunk:
        raise ValueError(
            "`--frontend-encode-mode lingbot_streaming_vae` requires persistent policy/frontend state; "
            "drop `--reset-policy-state-each-chunk`."
        )

    pipeline = build_variant_pipeline_from_config(config)
    mot_viz.video_viz._load_pipeline_checkpoint(pipeline, checkpoint_path)
    pipeline.to(device=runtime_device)
    if hasattr(pipeline.policy_variant, "_maybe_initialize_action_expert"):
        pipeline.policy_variant._maybe_initialize_action_expert(pipeline.visual_tower)
    mot_inference_backend = ensure_mot_inference_backend(pipeline, config)
    if mot_inference_backend["legacy_split_cache_restored_this_call"]:
        mot_viz._print_log("stage", {"name": "mot_legacy_cache_inference_blocks_restored"})
    if hasattr(pipeline.policy_variant, "action_expert"):
        pipeline.policy_variant.action_expert.to(device=action_device)
    pipeline.eval()
    runner = VariantRolloutRunner(pipeline)
    component_report = mot_viz._build_component_report(
        config,
        pipeline,
        runtime_device=runtime_device,
        action_device=action_device,
        frontend_device=frontend_device,
        decode_device=decode_device,
        raw_window_frames=raw_window_frames,
        mot_inference_window_size=args.mot_inference_window_size,
        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
        mot_action_only_rollout=bool(args.mot_action_only_rollout),
        mot_generalist_rollout_mode=None,
        mot_gjd_action_route=args.mot_gjd_action_route,
    )
    component_report["mot_inference_backend"] = mot_inference_backend
    component_report["checkpoint_file"] = str(checkpoint_path.resolve())
    component_report["checkpoint_runtime_config_path"] = (
        None if checkpoint_runtime_config_path is None else str(checkpoint_runtime_config_path)
    )
    component_report["checkpoint_runtime_config_merged"] = checkpoint_runtime_config_path is not None
    component_report["pipeline_training_mode"] = bool(pipeline.training)
    component_report["frontend_encode_mode"] = str(args.frontend_encode_mode)
    component_report["mot_rollout_frame_chunk_size"] = (
        None if args.mot_rollout_frame_chunk_size is None else int(args.mot_rollout_frame_chunk_size)
    )
    component_report["execute_action_steps"] = (
        None if args.execute_action_steps is None else int(args.execute_action_steps)
    )
    component_report["execute_frame_chunk_size"] = (
        None if args.execute_frame_chunk_size is None else int(args.execute_frame_chunk_size)
    )
    component_report["batch_driver"] = "run_libero_mot_batch_visualization.py"
    mot_viz._print_log("load_report", component_report)

    return SimpleNamespace(
        config=config,
        checkpoint_path=checkpoint_path,
        pipeline=pipeline,
        runner=runner,
        component_report=component_report,
        runtime_device=runtime_device,
        action_device=action_device,
        frontend_device=frontend_device,
        decode_device=decode_device,
        raw_window_frames=raw_window_frames,
        startup_model_obs_frames=startup_model_obs_frames,
        startup_env_init_steps=startup_env_init_steps,
        use_lingbot_streaming_vae=use_lingbot_streaming_vae,
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
        seed_everywhere(seed)
    task_spec, prompt, init_states = _resolve_task(resources, args.benchmark, task_id)
    env, close_env_after_rollout = _acquire_rollout_env(args, resources, task_spec=task_spec, task_id=task_id)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    config = resources.config
    pipeline = resources.pipeline
    runner = resources.runner
    use_lingbot_streaming_vae = bool(resources.use_lingbot_streaming_vae)

    try:
        mot_viz._print_log("stage", {"name": "init_env_rollout_start", "episode_idx": int(episode_idx)})
        initial_obs_window = mot_viz._init_single_env(
            env,
            init_states[episode_idx % len(init_states)],
            num_frames=resources.startup_model_obs_frames,
            init_steps=resources.startup_env_init_steps,
        )
        mot_viz._print_log(
            "stage",
            {
                "name": "init_env_rollout_done",
                "initial_window": len(initial_obs_window),
                "startup_model_obs_frames": int(resources.startup_model_obs_frames),
                "startup_env_init_steps": int(resources.startup_env_init_steps),
                "startup_env_steps_executed": int(max(resources.startup_env_init_steps, resources.startup_model_obs_frames)),
            },
        )
        frame_window: deque[dict[str, np.ndarray]] = deque(maxlen=resources.raw_window_frames)
        for obs in initial_obs_window:
            frame_window.append({key: np.array(value, copy=True) for key, value in obs.items()})

        predicted_latent_chunks: list[torch.Tensor] = []
        rollout_frames: list[dict[str, np.ndarray]] = [
            {key: np.array(value, copy=True) for key, value in obs.items()}
            for obs in list(frame_window)
        ]
        action_trace: list[np.ndarray] = []
        chunk_logs: list[dict[str, object]] = []
        done = False
        terminal = False
        chunk_count = 0
        session = runner.reset(task_text=(prompt,))
        streaming_next_visual_outputs = None
        streaming_next_obs_window: list[dict[str, np.ndarray]] | None = None
        if use_lingbot_streaming_vae:
            pipeline.visual_tower.reset_runtime_state()

        while env.env.timestep < args.max_timestep and not done and not terminal:
            if args.max_chunks is not None and chunk_count >= args.max_chunks:
                break
            if seed is not None:
                seed_everywhere(seed + chunk_count)

            with torch.inference_mode():
                mot_viz._print_log(
                    "stage",
                    {
                        "name": "chunk_prepare_start",
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
                    },
                )
                if use_lingbot_streaming_vae and chunk_count > 0:
                    if streaming_next_visual_outputs is None or streaming_next_obs_window is None:
                        raise RuntimeError(
                            "LingBot streaming VAE rollout expected encoded observations from the previous "
                            f"environment chunk before chunk_index={chunk_count}."
                        )
                    model_obs_window = streaming_next_obs_window
                    visual_outputs = streaming_next_visual_outputs
                    streaming_next_visual_outputs = None
                    streaming_next_obs_window = None
                    frontend_path = "lingbot_streaming_vae"
                else:
                    model_obs_window = mot_viz._select_model_obs_window(
                        list(frame_window),
                        chunk_index=chunk_count,
                        startup_model_obs_frames=resources.startup_model_obs_frames,
                    )
                    views = mot_viz._obs_list_to_views(model_obs_window, device=resources.frontend_device)
                    visual_outputs = mot_viz._prepare_mot_visual_outputs(
                        pipeline,
                        views=views,
                        task_text=(prompt,),
                        frontend_device=resources.frontend_device,
                        runtime_device=resources.runtime_device,
                        use_streaming_frontend=chunk_count == 0 or use_lingbot_streaming_vae,
                        preserve_stream_cache=False,
                        text_context=session.text_context,
                        negative_text_context=session.negative_text_context,
                    )
                    frontend_path = "lingbot_streaming_vae_init" if use_lingbot_streaming_vae else (
                        "streaming" if chunk_count == 0 else "offline"
                    )
                mot_viz._print_log(
                    "stage",
                    {
                        "name": "chunk_infer_start",
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
                        "model_obs_frames": int(len(model_obs_window)),
                        "video_latent_frames": int(visual_outputs.frontend.video_latents.shape[2]),
                        "frontend_path": frontend_path,
                    },
                )
                infer_context = mot_viz._build_infer_context(
                    prompt,
                    action_device=resources.action_device,
                    model_obs_window=model_obs_window,
                    config=config,
                    runtime_device=resources.runtime_device,
                    mot_inference_window_size=args.mot_inference_window_size,
                    mot_action_only_rollout=bool(args.mot_action_only_rollout),
                    mot_generalist_rollout_mode=None,
                )
                pre_infer_policy_state = (
                    None
                    if session.policy_state is None
                    else copy.deepcopy(session.policy_state)
                )
                infer_output = pipeline._forward_infer_with_visual_outputs(
                    visual_outputs,
                    context=infer_context,
                    infer_state=None if args.reset_policy_state_each_chunk else session.policy_state,
                )
                route_predicted_latents = mot_viz._extract_predicted_latents(infer_output)
                if args.mot_gjd_action_route == "joint_video_then_idm":
                    if not isinstance(route_predicted_latents, torch.Tensor) or int(route_predicted_latents.shape[2]) <= 0:
                        raise RuntimeError(
                            "joint_video_then_idm route requires joint rollout to produce a non-empty predicted video chunk."
                        )
                    idm_context = mot_viz._build_infer_context(
                        prompt,
                        action_device=resources.action_device,
                        model_obs_window=model_obs_window,
                        config=config,
                        runtime_device=resources.runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
                        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                        mot_action_only_rollout=False,
                        mot_generalist_rollout_mode=None,
                    )
                    idm_context.extra["action_conditioning_mode"] = "video_conditioned_action"
                    idm_context.extra["mot_generalist_rollout_mode"] = "video_conditioned_action"
                    idm_context.extra["mot_video_condition_latents"] = route_predicted_latents.detach().to(
                        device=resources.runtime_device,
                        dtype=route_predicted_latents.dtype,
                    )
                    infer_output = pipeline._forward_infer_with_visual_outputs(
                        visual_outputs,
                        context=idm_context,
                        infer_state=None if args.reset_policy_state_each_chunk else pre_infer_policy_state,
                    )
                mot_viz._print_log(
                    "stage",
                    {
                        "name": "chunk_infer_done",
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
                        "mot_gjd_action_route": str(args.mot_gjd_action_route),
                    },
                )
            session = runner.reset(
                task_text=session.task_text,
                text_context=(
                    visual_outputs.frontend.conditioning.text_context
                    if visual_outputs.frontend.conditioning.text_context is not None
                    else session.text_context
                ),
                negative_text_context=(
                    visual_outputs.frontend.conditioning.negative_text_context
                    if visual_outputs.frontend.conditioning.negative_text_context is not None
                    else session.negative_text_context
                ),
            )
            session.policy_state = infer_output.policy_output.next_state
            actions = infer_output.decoder_output.action_pred[0].detach().to(dtype=torch.float32).cpu().numpy()
            configured_frame_chunk_size = mot_viz._frame_chunk_size(config)
            configured_action_per_frame = mot_viz._action_per_frame(config)
            action_per_frame = configured_action_per_frame
            if int(actions.shape[0]) % int(action_per_frame) != 0:
                raise ValueError(
                    "MoT action output length must be divisible by configured action_per_frame, "
                    f"got action_shape={actions.shape}, action_per_frame={action_per_frame}."
                )
            frame_chunk_size = int(actions.shape[0]) // int(action_per_frame)
            execute_action_steps = mot_viz._resolve_execute_action_steps(
                args.execute_action_steps,
                execute_frame_chunk_size=args.execute_frame_chunk_size,
                action_horizon=int(actions.shape[0]),
                action_per_frame=action_per_frame,
            )
            frame_actions = actions.reshape(frame_chunk_size, action_per_frame, actions.shape[-1])
            predicted_latents = mot_viz._extract_predicted_latents(infer_output)
            if args.mot_gjd_action_route == "joint_video_then_idm" and isinstance(route_predicted_latents, torch.Tensor):
                predicted_latents = route_predicted_latents
            if not args.skip_comparison_video and isinstance(predicted_latents, torch.Tensor):
                mot_viz._append_predicted_latent_chunk(
                    predicted_latent_chunks,
                    predicted_latents,
                    max_imagined_latent_frames=args.max_imagined_latent_frames,
                )

            chunk_log = {
                "task_id": int(task_id),
                "episode_idx": int(episode_idx),
                "chunk_index": chunk_count,
                "phase": "infer",
                "env_timestep_before": int(env.env.timestep),
                "window_size": len(frame_window),
                "model_obs_frames": len(model_obs_window),
                "video_latent_frames": int(visual_outputs.frontend.video_latents.shape[2]),
                "frontend_path": frontend_path,
                "action_shape": list(actions.shape),
                "execute_action_steps": int(execute_action_steps),
                "configured_frame_chunk_size": int(configured_frame_chunk_size),
                "rollout_frame_chunk_size": int(frame_chunk_size),
                "execute_frame_chunk_size": int(execute_action_steps // action_per_frame),
                "predicted_latents_shape": None if not isinstance(predicted_latents, torch.Tensor) else list(predicted_latents.shape),
                "mot_gjd_action_route": str(args.mot_gjd_action_route),
                "first_action_preview": [float(v) for v in actions[0].tolist()],
                "policy_debug": mot_viz._summarize_policy_debug(infer_output.policy_output.aux),
            }
            mot_viz._print_log(f"task_{task_id}_episode_{episode_idx}_chunk_{chunk_count}", chunk_log)
            chunk_logs.append(chunk_log)

            real_future_frames: list[dict[str, np.ndarray]] = []
            executed_actions = 0
            executed_control_actions: list[np.ndarray] = []
            executed_obs_frames: list[dict[str, np.ndarray]] = []
            policy_debug = mot_viz._summarize_policy_debug(infer_output.policy_output.aux)
            generation_frame_start = int(
                policy_debug.get("generation_frame_start", 0)
                if "generation_frame_start" in policy_debug
                else policy_debug.get("mot_cache_debug", {}).get("current_action_frame_start", 0)
            )
            start_frame_group = 1 if chunk_count == 0 and generation_frame_start <= 0 else 0
            max_action_index = min(int(execute_action_steps), int(actions.shape[0]))
            for frame_group in range(start_frame_group, frame_actions.shape[0]):
                for action_offset, action in enumerate(frame_actions[frame_group]):
                    absolute_action_index = frame_group * action_per_frame + action_offset
                    if absolute_action_index >= max_action_index:
                        break
                    if _raw_env_done(env) or env.env.timestep >= args.max_timestep:
                        terminal = True
                        break
                    control_action = np.clip(action.astype(np.float32, copy=False), -1.0, 1.0)
                    executed_control_actions.append(np.array(control_action, copy=True))
                    action_trace.append(np.array(control_action, copy=True))
                    obs, _, step_success, _ = env.step(control_action)
                    done = bool(done or step_success)
                    executed_actions += 1
                    extracted = mot_viz._extract_obs(obs)
                    extracted_record = {key: np.array(value, copy=True) for key, value in extracted.items()}
                    rollout_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    frame_window.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    executed_obs_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    real_future_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    terminal = bool(done or _raw_env_done(env) or env.env.timestep >= args.max_timestep)
                    if terminal:
                        break
                if terminal:
                    break
                next_frame_group_first_action = (frame_group + 1) * action_per_frame
                if next_frame_group_first_action >= max_action_index:
                    break
            terminal = bool(done or _raw_env_done(env) or env.env.timestep >= args.max_timestep)

            chunk_result_log = {
                "task_id": int(task_id),
                "episode_idx": int(episode_idx),
                "chunk_index": chunk_count,
                "phase": "env_rollout",
                "env_timestep_after": int(env.env.timestep),
                "executed_actions": int(executed_actions),
                "execute_action_steps": int(execute_action_steps),
                "configured_frame_chunk_size": int(configured_frame_chunk_size),
                "rollout_frame_chunk_size": int(frame_chunk_size),
                "execute_frame_chunk_size": int(execute_action_steps // action_per_frame),
                "start_frame_group": int(start_frame_group),
                "done_after_chunk": bool(terminal),
                "success_after_chunk": bool(done),
            }
            mot_viz._print_log(f"task_{task_id}_episode_{episode_idx}_chunk_{chunk_count}", chunk_result_log)
            chunk_logs.append(chunk_result_log)

            warmup_action_history = mot_viz._build_executed_action_history_tensor(
                executed_control_actions,
                start_frame_group=start_frame_group,
                action_per_frame=action_per_frame,
                action_dim=actions.shape[-1],
            )
            if (
                use_lingbot_streaming_vae
                and executed_obs_frames
                and not terminal
                and env.env.timestep < args.max_timestep
            ):
                streaming_views = mot_viz._obs_list_to_views(executed_obs_frames, device=resources.frontend_device)
                streaming_next_visual_outputs = mot_viz._prepare_mot_visual_outputs(
                    pipeline,
                    views=streaming_views,
                    task_text=(prompt,),
                    frontend_device=resources.frontend_device,
                    runtime_device=resources.runtime_device,
                    use_streaming_frontend=True,
                    preserve_stream_cache=True,
                    text_context=session.text_context,
                    negative_text_context=session.negative_text_context,
                )
                streaming_next_obs_window = [
                    {key: np.array(value, copy=True) for key, value in obs.items()}
                    for obs in executed_obs_frames
                ]
                streaming_update_log = {
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "chunk_index": chunk_count,
                    "phase": "lingbot_streaming_vae_update",
                    "real_obs_frames": int(len(streaming_next_obs_window)),
                    "real_latent_frames": int(streaming_next_visual_outputs.frontend.video_latents.shape[2]),
                }
                mot_viz._print_log(f"task_{task_id}_episode_{episode_idx}_chunk_{chunk_count}", streaming_update_log)
                chunk_logs.append(streaming_update_log)

            if (
                (executed_obs_frames if use_lingbot_streaming_vae else real_future_frames)
                and warmup_action_history is not None
                and not terminal
                and env.env.timestep < args.max_timestep
                and "mot_packed_history_debug" in infer_output.policy_output.aux
            ):
                if use_lingbot_streaming_vae:
                    if streaming_next_visual_outputs is None or streaming_next_obs_window is None:
                        raise RuntimeError("Streaming VAE packed warmup expected pre-encoded next observations.")
                    warmup_debug = mot_viz._warmup_mot_packed_history_from_visual_outputs(
                        pipeline,
                        config=config,
                        session=session,
                        warmup_outputs=streaming_next_visual_outputs,
                        obs_frame_count=len(streaming_next_obs_window),
                        obs_list=streaming_next_obs_window,
                        action_history=warmup_action_history,
                        runtime_device=resources.runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
                        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                    )
                else:
                    warmup_debug = mot_viz._warmup_mot_packed_history_from_observations(
                        pipeline,
                        config=config,
                        session=session,
                        obs_list=real_future_frames,
                        action_history=warmup_action_history,
                        task_text=(prompt,),
                        frontend_device=resources.frontend_device,
                        runtime_device=resources.runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
                        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                    )
                warmup_log = {
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "chunk_index": chunk_count,
                    "phase": "packed_history_warmup",
                    **warmup_debug,
                }
                mot_viz._print_log(f"task_{task_id}_episode_{episode_idx}_chunk_{chunk_count}", warmup_log)
                chunk_logs.append(warmup_log)

            chunk_count += 1

        output_path = mot_viz._build_output_path(
            root=Path(args.output_dir),
            benchmark_name=args.benchmark,
            task_id=task_id,
            prompt=prompt,
            episode_idx=episode_idx,
            done=done,
            suffix=args.suffix,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_video_path, rollout_path = _write_rollout_videos(
            args,
            resources,
            pipeline=pipeline,
            predicted_latent_chunks=predicted_latent_chunks,
            rollout_frames=rollout_frames,
            output_path=output_path,
        )

        summary = {
            "benchmark": args.benchmark,
            "task_id": task_id,
            "prompt": prompt,
            "episode_idx": episode_idx,
            "success": bool(done),
            "terminal": bool(terminal),
            "chunk_count": chunk_count,
            "env_timestep": int(env.env.timestep),
            "seed": seed,
            "video_path": comparison_video_path,
            "comparison_video_path": comparison_video_path,
            "rollout_video_path": None if rollout_path is None else str(rollout_path.resolve()),
            "pipeline": "open_wam_mot",
            "runtime_mode": str(config.policy_variant.runtime_mode),
            "condition_mode": str(config.policy_variant.condition_mode),
            "startup_model_obs_frames": int(resources.startup_model_obs_frames),
            "startup_env_init_steps": int(resources.startup_env_init_steps),
            "startup_env_steps_executed": int(max(resources.startup_env_init_steps, resources.startup_model_obs_frames)),
            "execute_action_steps": None if args.execute_action_steps is None else int(args.execute_action_steps),
            "execute_frame_chunk_size": (
                None if args.execute_frame_chunk_size is None else int(args.execute_frame_chunk_size)
            ),
            "action_count": len(action_trace),
            "checkpoint_file": str(resources.checkpoint_path.resolve()),
            "mot_gjd_action_route": str(args.mot_gjd_action_route),
        }
        summary_path = output_path.with_suffix(".json")
        action_trace_path = output_path.with_name(f"{output_path.stem}_actions.jsonl")
        summary["action_trace_path"] = str(action_trace_path.resolve())
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with action_trace_path.open("w", encoding="utf-8") as handle:
            for action_index, action in enumerate(action_trace):
                handle.write(
                    json.dumps(
                        {
                            "action_index": int(action_index),
                            "action": np.asarray(action, dtype=np.float32).tolist(),
                        }
                    )
                    + "\n"
                )
        chunk_log_path = output_path.with_name(f"{output_path.stem}_chunks.json")
        chunk_log_path.write_text(json.dumps(chunk_logs, indent=2, default=str), encoding="utf-8")
        load_report_path = output_path.with_name(f"{output_path.stem}_load_report.json")
        load_report_path.write_text(json.dumps(resources.component_report, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if close_env_after_rollout:
            env.close()


def _raw_env_done(env: object) -> bool:
    """Return robosuite's terminal flag behind LIBERO's success-only wrapper."""
    raw_env = getattr(env, "env", env)
    return bool(getattr(raw_env, "done", False))


def _resolve_task(resources: SimpleNamespace, benchmark_name: str, task_id: int):
    cache_key = (benchmark_name, int(task_id))
    if cache_key not in resources.task_cache:
        task_spec, prompt = mot_viz._resolve_task_spec(benchmark_name, int(task_id))
        init_states = load_libero_task_init_states(task_spec)
        resources.task_cache[cache_key] = (task_spec, prompt, init_states)
    return resources.task_cache[cache_key]


def _acquire_rollout_env(
    args: argparse.Namespace,
    resources: SimpleNamespace,
    *,
    task_spec,
    task_id: int,
):
    if not args.reuse_env_per_task:
        return mot_viz._construct_single_env(task_spec), True
    if resources.reused_env is not None and resources.reused_env_task_id != int(task_id):
        _close_reused_env(resources)
    if resources.reused_env is None:
        resources.reused_env = mot_viz._construct_single_env(task_spec)
        resources.reused_env_task_id = int(task_id)
    return resources.reused_env, False


def _close_reused_env(resources: SimpleNamespace) -> None:
    env = getattr(resources, "reused_env", None)
    if env is not None:
        env.close()
    resources.reused_env = None
    resources.reused_env_task_id = None


def _write_rollout_videos(
    args: argparse.Namespace,
    resources: SimpleNamespace,
    *,
    pipeline,
    predicted_latent_chunks: list[torch.Tensor],
    rollout_frames: list[dict[str, np.ndarray]],
    output_path: Path,
) -> tuple[str | None, Path | None]:
    comparison_video_path = None
    if not args.skip_comparison_video:
        imagined_video = mot_viz._decode_latent_video_chunks(
            pipeline,
            predicted_latent_chunks,
            decode_device=resources.decode_device,
            restore_vae=False,
        )
        mot_viz._write_video_frames(
            output_path,
            mot_viz._iter_comparison_video_frames(
                real_obs_list=rollout_frames,
                imagined_video=imagined_video,
            ),
            fps=args.video_fps,
        )
        comparison_video_path = str(output_path.resolve())
    rollout_path = None
    if args.save_rollout_video:
        rollout_path = output_path.with_name(f"{output_path.stem}_rollout.mp4")
        mot_viz._write_video_frames(
            rollout_path,
            mot_viz._iter_rollout_video_frames(real_obs_list=rollout_frames),
            fps=args.video_fps,
        )
    return comparison_video_path, rollout_path


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
