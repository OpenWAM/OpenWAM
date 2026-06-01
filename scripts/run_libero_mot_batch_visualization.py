from __future__ import annotations

import argparse
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
from open_wam.utils import load_experiment_config, seed_everywhere  # noqa: E402
from open_wam.utils.libero_paradigm import require_current_libero_policy_paradigm  # noqa: E402

_MOT_VIZ_PATH = REPO_ROOT / "scripts" / "deprecated" / "run_libero_mot_visualization.py"
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
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument(
        "--task-ids",
        type=str,
        required=True,
        help="Task ids to evaluate, e.g. `0-9`, `0,3,7`, or `0-2,5`.",
    )
    parser.add_argument(
        "--episode-idxs",
        type=str,
        required=True,
        help="Episode indices to evaluate, e.g. `0-49`, `0,3,7`, or `0-2,5`.",
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
        "--seed-by-episode",
        action="store_true",
        help="Use episode_idx as the per-rollout seed, matching shell loops that pass `--seed ${EP}`.",
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

    resources = _load_batch_resources(args)
    task_ids = _parse_int_ranges(args.task_ids, label="task-ids")
    episode_idxs = _parse_int_ranges(args.episode_idxs, label="episode-idxs")
    pairs = _iter_pairs(task_ids, episode_idxs, loop_order=args.loop_order)

    summaries: list[dict[str, object]] = []
    for task_id, episode_idx in pairs:
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
        mot_action_only_rollout=bool(args.mot_action_only_rollout),
    )
    component_report["mot_inference_backend"] = mot_inference_backend
    component_report["checkpoint_file"] = str(checkpoint_path.resolve())
    component_report["checkpoint_runtime_config_path"] = (
        None if checkpoint_runtime_config_path is None else str(checkpoint_runtime_config_path)
    )
    component_report["checkpoint_runtime_config_merged"] = checkpoint_runtime_config_path is not None
    component_report["pipeline_training_mode"] = bool(pipeline.training)
    component_report["frontend_encode_mode"] = str(args.frontend_encode_mode)
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
    env = mot_viz._construct_single_env(task_spec)
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
        chunk_count = 0
        session = runner.reset(task_text=(prompt,))
        streaming_next_visual_outputs = None
        streaming_next_obs_window: list[dict[str, np.ndarray]] | None = None
        if use_lingbot_streaming_vae:
            pipeline.visual_tower.reset_runtime_state()

        while env.env.timestep < args.max_timestep and not done:
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
                infer_output = pipeline._forward_infer_with_visual_outputs(
                    visual_outputs,
                    context=mot_viz._build_infer_context(
                        prompt,
                        action_device=resources.action_device,
                        model_obs_window=model_obs_window,
                        config=config,
                        runtime_device=resources.runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
                        mot_action_only_rollout=bool(args.mot_action_only_rollout),
                    ),
                    infer_state=None if args.reset_policy_state_each_chunk else session.policy_state,
                )
                mot_viz._print_log(
                    "stage",
                    {
                        "name": "chunk_infer_done",
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
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
            frame_chunk_size = mot_viz._frame_chunk_size(config)
            action_per_frame = mot_viz._action_per_frame(config)
            frame_actions = actions.reshape(frame_chunk_size, action_per_frame, actions.shape[-1])
            predicted_latents = infer_output.decoder_output.aux.get("predicted_latents")
            if not isinstance(predicted_latents, torch.Tensor):
                predicted_latents = infer_output.policy_output.aux.get("predicted_latents")
            if isinstance(predicted_latents, torch.Tensor):
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
                "predicted_latents_shape": None if not isinstance(predicted_latents, torch.Tensor) else list(predicted_latents.shape),
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
            for frame_group in range(start_frame_group, frame_actions.shape[0]):
                for action_offset, action in enumerate(frame_actions[frame_group]):
                    del action_offset
                    control_action = np.clip(action.astype(np.float32, copy=False), -1.0, 1.0)
                    executed_control_actions.append(np.array(control_action, copy=True))
                    action_trace.append(np.array(control_action, copy=True))
                    obs, _, done, _ = env.step(control_action)
                    executed_actions += 1
                    extracted = mot_viz._extract_obs(obs)
                    extracted_record = {key: np.array(value, copy=True) for key, value in extracted.items()}
                    rollout_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    frame_window.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    executed_obs_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    real_future_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    if done or env.env.timestep >= args.max_timestep:
                        break
                if done or env.env.timestep >= args.max_timestep:
                    break

            chunk_result_log = {
                "task_id": int(task_id),
                "episode_idx": int(episode_idx),
                "chunk_index": chunk_count,
                "phase": "env_rollout",
                "env_timestep_after": int(env.env.timestep),
                "executed_actions": int(executed_actions),
                "start_frame_group": int(start_frame_group),
                "done_after_chunk": bool(done),
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
                and not done
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
                and not done
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
                        action_history=warmup_action_history,
                        runtime_device=resources.runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
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

        imagined_video = mot_viz._decode_latent_video_chunks(
            pipeline,
            predicted_latent_chunks,
            decode_device=resources.decode_device,
            restore_vae=False,
        )
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
        mot_viz._write_video_frames(
            output_path,
            mot_viz._iter_comparison_video_frames(
                real_obs_list=rollout_frames,
                imagined_video=imagined_video,
            ),
            fps=args.video_fps,
        )
        rollout_path = None
        if args.save_rollout_video:
            rollout_path = output_path.with_name(f"{output_path.stem}_rollout.mp4")
            mot_viz._write_video_frames(
                rollout_path,
                mot_viz._iter_rollout_video_frames(real_obs_list=rollout_frames),
                fps=args.video_fps,
            )

        summary = {
            "benchmark": args.benchmark,
            "task_id": task_id,
            "prompt": prompt,
            "episode_idx": episode_idx,
            "success": bool(done),
            "chunk_count": chunk_count,
            "env_timestep": int(env.env.timestep),
            "seed": seed,
            "video_path": str(output_path.resolve()),
            "comparison_video_path": str(output_path.resolve()),
            "rollout_video_path": None if rollout_path is None else str(rollout_path.resolve()),
            "pipeline": "open_wam_mot",
            "runtime_mode": str(config.policy_variant.runtime_mode),
            "condition_mode": str(config.policy_variant.condition_mode),
            "startup_model_obs_frames": int(resources.startup_model_obs_frames),
            "startup_env_init_steps": int(resources.startup_env_init_steps),
            "startup_env_steps_executed": int(max(resources.startup_env_init_steps, resources.startup_model_obs_frames)),
            "action_count": len(action_trace),
            "checkpoint_file": str(resources.checkpoint_path.resolve()),
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
        env.close()


def _resolve_task(resources: SimpleNamespace, benchmark_name: str, task_id: int):
    cache_key = (benchmark_name, int(task_id))
    if cache_key not in resources.task_cache:
        task_spec, prompt = mot_viz._resolve_task_spec(benchmark_name, int(task_id))
        init_states = load_libero_task_init_states(task_spec)
        resources.task_cache[cache_key] = (task_spec, prompt, init_states)
    return resources.task_cache[cache_key]


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


if __name__ == "__main__":
    main()
