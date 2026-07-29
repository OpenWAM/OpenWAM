from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers.video_processor import VideoProcessor
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
for script_path in (SCRIPT_ROOT, Path(__file__).resolve().parent):
    if str(script_path) not in sys.path:
        sys.path.insert(0, str(script_path))

from open_wam.configs import ReferenceCoreInitMode  # noqa: E402
from open_wam.data.latent_temporal import raw_window_frames_for_latents  # noqa: E402
from open_wam.integrations import (  # noqa: E402
    LiberoTaskSpec,
    build_libero_state_history,
    ensure_local_libero_config,
    load_libero_task_init_states,
)
from open_wam.models.common.rollout_history import (  # noqa: E402
    build_executed_action_history_tensor as _build_shared_executed_action_history_tensor,
    resolve_execute_action_steps as _resolve_shared_execute_action_steps,
)
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.models.policy_variants.mot.runtime_routing import (  # noqa: E402
    ensure_mot_inference_backend,
    resolve_mot_rollout_cache_window_frames,
)
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.runtime.checkpoints import (  # noqa: E402
    load_pipeline_checkpoint,
    resolve_checkpoint_file,
    resolve_checkpoint_step_dir_from_transformer_dir,
)
from open_wam.utils.local_paths import read_yaml_with_local_paths  # noqa: E402
from open_wam.utils import (  # noqa: E402
    apply_config_overrides,
    load_experiment_config,
    merge_runtime_config_from_checkpoint,
    parse_override_assignments,
    seed_everywhere,
)
from open_wam.utils.libero_paradigm import (  # noqa: E402
    require_current_libero_policy_paradigm,
)

LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)
CURRENT_FRONTEND_ENCODE_MODE = "lingbot_streaming_vae"
DEPRECATED_FRONTEND_ENCODE_MODE = "rolling_offline"
LIVE_SIM_MOT_GENERALIST_ROLLOUT_MODES = frozenset(
    {
        "joint",
        "vanilla_joint_rollout",
    }
)
MOT_GJD_ACTION_ROUTES = frozenset(
    {
        "joint",
        "joint_video_then_idm",
    }
)
OFFLINE_DIAGNOSTIC_MOT_GENERALIST_ROLLOUT_MODES = frozenset(
    {
        "clean_action_feedback",
        "forced_action_joint_fdm",
        "action_conditioned_video",
        "video_conditioned_action",
        "fdm",
        "idm",
    }
)


def _raw_env_done(env: object) -> bool:
    """Return robosuite's terminal flag behind LIBERO's success-only wrapper."""
    raw_env = getattr(env, "env", env)
    return bool(getattr(raw_env, "done", False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one LIBERO rollout with a MoT policy and save a rollout video."
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/mot_libero_latent_local_joint_heng_compatible.yaml",
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint file, checkpoint_step_* directory, or run directory. "
            "If omitted, use top-level checkpoint_path in the config, then infer from backbone.transformer_subdir."
        ),
    )
    parser.add_argument(
        "--merge-checkpoint-runtime-config",
        action="store_true",
        help=(
            "Opt into merging the checkpoint's resolved_config.yaml before rollout. "
            "By default this script treats --cfg as the rollout contract and only uses "
            "the checkpoint directory for weights/exported transformer assets, keeping "
            "old checkpoints with stale resolved_config.yaml files usable."
        ),
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Apply a config override such as `--set policy_variant.generalist_mode_text_token=true`.",
    )
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--episode-idx", type=int, default=0)
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
    parser.add_argument(
        "--mot-inference-window-size",
        type=int,
        default=None,
        help=(
            "Optional MoT rollout attention/cache window override in latent-frame block units. "
            "For legacy split-cache modes this overrides the LingBot slot-pool attn_window; "
            "for packed coupling modes it overrides the training_config.window_size used at inference."
        ),
    )
    parser.add_argument(
        "--mot-action-only-rollout",
        action="store_true",
        help=(
            "Skip imagined-video denoising during MoT rollout and produce actions only. "
            "Supported only for action_then_video and decoupled_same_step couplings."
        ),
    )
    parser.add_argument(
        "--mot-generalist-rollout-mode",
        metavar="{joint,vanilla_joint_rollout}",
        default=None,
        help=(
            "Optional M5 GJD rollout mode forwarded to PolicyInferContext. "
            "Live sim rollout supports only joint/vanilla modes; FDM/IDM and "
            "clean-action diagnostic modes require offline GT action/video tensors. "
            "`--mot-action-only-rollout` is not an IDM substitute."
        ),
    )
    parser.add_argument(
        "--mot-gjd-action-route",
        choices=sorted(MOT_GJD_ACTION_ROUTES),
        default="joint",
        help=(
            "Diagnostic M5 GJD live-sim action route. `joint` is the maintained "
            "normal rollout. `joint_video_then_idm` first generates the current "
            "video chunk with joint denoising, then reruns IDM from the same "
            "pre-step state using only that generated video as clean condition "
            "and executes the IDM action chunk."
        ),
    )
    parser.add_argument(
        "--frontend-encode-mode",
        choices=(DEPRECATED_FRONTEND_ENCODE_MODE, CURRENT_FRONTEND_ENCODE_MODE),
        default=CURRENT_FRONTEND_ENCODE_MODE,
        help=(
            "RGB-to-latent frontend mode. The current supported rollout contract is "
            "`lingbot_streaming_vae`, which keeps the Wan VAE stream cache alive and "
            "encodes only newly executed env observations between chunks. `rolling_offline` "
            "is deprecated historical compatibility and requires "
            "`--allow-deprecated-frontend-encode-mode`."
        ),
    )
    parser.add_argument(
        "--startup-model-obs-frames",
        type=int,
        default=1,
        help=(
            "Number of initial observations fed to the model on chunk 0. "
            "Defaults to 1 to match Method-1 exact startup; the rolling window "
            "used after chunk 0 still keeps `--raw-window-frames`."
        ),
    )
    parser.add_argument(
        "--startup-env-init-steps",
        type=int,
        default=5,
        help=(
            "Number of zero-action environment steps before chunk 0. "
            "Defaults to 5 to match Method-1 exact startup; if smaller than "
            "--startup-model-obs-frames, it is raised to keep enough observations."
        ),
    )
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_mot_visualization")
    parser.add_argument("--suffix", type=str, default="open_wam_mot")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-rollout-video", action="store_true")
    parser.add_argument(
        "--max-imagined-latent-frames",
        type=int,
        default=None,
        help=(
            "Optional cap on predicted latent frames retained for the imagined-video panel. "
            "Default keeps all imagined latent frames for full comparison videos. "
            "Use 0 to disable imagined-video decode for long memory-constrained rollouts."
        ),
    )
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--action-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument(
        "--reset-policy-state-each-chunk",
        action="store_true",
        help=(
            "Debug/compatibility mode matching the original standalone MoT script: "
            "rebuild the MoT policy state for each observation window instead of carrying caches across chunks."
        ),
    )
    parser.add_argument(
        "--allow-deprecated-libero-config",
        action="store_true",
        help=(
            "Allow historical LIBERO M5 configs that do not match the current strict fixed-128, "
            "one-frame, proprio-conditioned training/eval paradigm."
        ),
    )
    parser.add_argument(
        "--allow-deprecated-frontend-encode-mode",
        action="store_true",
        help=(
            "Allow non-lingbot_streaming_vae frontend encode modes only for historical debugging. "
            "Current M5 rollout comparisons should not use this."
        ),
    )
    args = parser.parse_args()
    _validate_live_sim_mot_generalist_rollout_mode(args.mot_generalist_rollout_mode)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    _validate_mot_config(config)
    checkpoint_path = _resolve_mot_checkpoint_path(
        config_path=config_path,
        checkpoint_arg=args.checkpoint,
        transformer_subdir=str(config.backbone.transformer_subdir),
    )
    if checkpoint_path is None:
        raise ValueError(
            "MoT visualization requires a trained checkpoint. Pass `--checkpoint`, set top-level "
            "`checkpoint_path` in the config, or point `backbone.transformer_subdir` at an exported checkpoint."
        )
    config, checkpoint_runtime_config_path = _maybe_merge_checkpoint_runtime_config(
        config,
        checkpoint_path,
        merge_enabled=bool(args.merge_checkpoint_runtime_config),
    )
    if args.set_overrides:
        config = apply_config_overrides(
            config,
            parse_override_assignments(tuple(args.set_overrides)),
        )
    _validate_mot_config(config)
    require_current_libero_policy_paradigm(
        config,
        config_path=config_path,
        source="run_libero_mot_visualization.py",
        allow_deprecated=bool(args.allow_deprecated_libero_config),
    )
    transformer_dir = checkpoint_path.parent / "transformer"
    if transformer_dir.is_dir():
        object.__setattr__(config.backbone, "transformer_subdir", str(transformer_dir.resolve()))
        object.__setattr__(config.backbone, "reference_core_init_mode", ReferenceCoreInitMode.FULL)

    runtime_device = _resolve_device(args.runtime_device)
    action_device = _resolve_device(args.action_device, fallback=runtime_device)
    frontend_device = _resolve_device(args.frontend_device, fallback=runtime_device)
    decode_device = _resolve_device(args.decode_device, fallback=frontend_device)
    raw_window_frames = (
        int(args.raw_window_frames)
        if args.raw_window_frames is not None
        else _default_raw_window_frames(int(config.data.num_frames))
    )
    startup_model_obs_frames = int(args.startup_model_obs_frames)
    if startup_model_obs_frames <= 0:
        raise ValueError(
            f"Expected --startup-model-obs-frames to be positive, got {startup_model_obs_frames}."
        )
    if startup_model_obs_frames > raw_window_frames:
        raise ValueError(
            "--startup-model-obs-frames must be <= --raw-window-frames, "
            f"got startup_model_obs_frames={startup_model_obs_frames}, raw_window_frames={raw_window_frames}."
        )
    startup_env_init_steps = int(args.startup_env_init_steps)
    if startup_env_init_steps <= 0:
        raise ValueError(
            f"Expected --startup-env-init-steps to be positive, got {startup_env_init_steps}."
        )
    if args.mot_inference_window_size is not None and int(args.mot_inference_window_size) <= 0:
        raise ValueError(
            "Expected --mot-inference-window-size to be positive when provided, "
            f"got {args.mot_inference_window_size}."
        )
    if args.mot_rollout_frame_chunk_size is not None:
        rollout_frame_chunk_size = int(args.mot_rollout_frame_chunk_size)
        configured_frame_chunk_size = _frame_chunk_size(config)
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
        _resolve_execute_action_steps(
            args.execute_action_steps,
            execute_frame_chunk_size=args.execute_frame_chunk_size,
            action_horizon=int(config.data.action_schema.action_horizon),
            action_per_frame=_action_per_frame(config),
        )
    _require_current_frontend_encode_mode(
        args.frontend_encode_mode,
        allow_deprecated=bool(args.allow_deprecated_frontend_encode_mode),
        source="run_libero_mot_visualization.py",
    )
    use_lingbot_streaming_vae = args.frontend_encode_mode == CURRENT_FRONTEND_ENCODE_MODE
    if use_lingbot_streaming_vae and startup_model_obs_frames != 1:
        raise ValueError(
            "`--frontend-encode-mode lingbot_streaming_vae` expects "
            "`--startup-model-obs-frames 1` to match LingBot-VA's first-frame bootstrap."
        )
    if use_lingbot_streaming_vae and args.reset_policy_state_each_chunk:
        raise ValueError(
            "`--frontend-encode-mode lingbot_streaming_vae` requires persistent policy/frontend state; "
            "drop `--reset-policy-state-each-chunk`."
        )

    pipeline = build_variant_pipeline_from_config(config)
    checkpoint_report = load_pipeline_checkpoint(pipeline, checkpoint_path)
    if checkpoint_report.missing_keys:
        print(f"viz.checkpoint_missing_keys {len(checkpoint_report.missing_keys)}")
    if checkpoint_report.unexpected_keys:
        print(f"viz.checkpoint_unexpected_keys {len(checkpoint_report.unexpected_keys)}")
    pipeline.to(device=runtime_device)
    if hasattr(pipeline.policy_variant, "_maybe_initialize_action_expert"):
        pipeline.policy_variant._maybe_initialize_action_expert(pipeline.visual_tower)
    mot_inference_backend = ensure_mot_inference_backend(pipeline, config)
    if mot_inference_backend["legacy_split_cache_restored_this_call"]:
        _print_log("stage", {"name": "mot_legacy_cache_inference_blocks_restored"})
    if hasattr(pipeline.policy_variant, "action_expert"):
        pipeline.policy_variant.action_expert.to(device=action_device)
    pipeline.eval()
    runner = VariantRolloutRunner(pipeline)
    component_report = _build_component_report(
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
        mot_generalist_rollout_mode=args.mot_generalist_rollout_mode,
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
    _print_log("load_report", component_report)

    task_spec, prompt = _resolve_task_spec(args.benchmark, args.task_id)
    init_states = load_libero_task_init_states(task_spec)
    env = _construct_single_env(task_spec)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    try:
        _print_log("stage", {"name": "init_env_rollout_start", "episode_idx": int(args.episode_idx)})
        initial_obs_window = _init_single_env(
            env,
            init_states[args.episode_idx % len(init_states)],
            num_frames=startup_model_obs_frames,
            init_steps=startup_env_init_steps,
        )
        _print_log(
            "stage",
            {
                "name": "init_env_rollout_done",
                "initial_window": len(initial_obs_window),
                "startup_model_obs_frames": int(startup_model_obs_frames),
                "startup_env_init_steps": int(startup_env_init_steps),
                "startup_env_steps_executed": int(max(startup_env_init_steps, startup_model_obs_frames)),
            },
        )
        frame_window: deque[dict[str, np.ndarray]] = deque(maxlen=raw_window_frames)
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

            if args.seed is not None:
                seed_everywhere(args.seed + chunk_count)

            with torch.inference_mode():
                _print_log(
                    "stage",
                    {
                        "name": "chunk_prepare_start",
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
                    model_obs_window = _select_model_obs_window(
                        list(frame_window),
                        chunk_index=chunk_count,
                        startup_model_obs_frames=startup_model_obs_frames,
                    )
                    views = _obs_list_to_views(model_obs_window, device=frontend_device)
                    visual_outputs = _prepare_mot_visual_outputs(
                        pipeline,
                        views=views,
                        task_text=(prompt,),
                        frontend_device=frontend_device,
                        runtime_device=runtime_device,
                        use_streaming_frontend=chunk_count == 0 or use_lingbot_streaming_vae,
                        preserve_stream_cache=False,
                        text_context=session.text_context,
                        negative_text_context=session.negative_text_context,
                    )
                    frontend_path = "lingbot_streaming_vae_init" if use_lingbot_streaming_vae else (
                        "streaming" if chunk_count == 0 else "offline"
                    )
                _print_log(
                    "stage",
                    {
                        "name": "chunk_infer_start",
                        "chunk_index": int(chunk_count),
                        "env_timestep": int(env.env.timestep),
                        "model_obs_frames": int(len(model_obs_window)),
                        "video_latent_frames": int(visual_outputs.frontend.video_latents.shape[2]),
                        "frontend_path": frontend_path,
                    },
                )
                infer_context = _build_infer_context(
                    prompt,
                    action_device=action_device,
                    model_obs_window=model_obs_window,
                    config=config,
                    runtime_device=runtime_device,
                    mot_inference_window_size=args.mot_inference_window_size,
                    mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                    mot_action_only_rollout=bool(args.mot_action_only_rollout),
                    mot_generalist_rollout_mode=args.mot_generalist_rollout_mode,
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
                route_predicted_latents = _extract_predicted_latents(infer_output)
                if args.mot_gjd_action_route == "joint_video_then_idm":
                    if args.mot_generalist_rollout_mode not in (None, "joint", "vanilla_joint_rollout"):
                        raise ValueError(
                            "`--mot-gjd-action-route joint_video_then_idm` must start from joint GJD rollout; "
                            f"got --mot-generalist-rollout-mode={args.mot_generalist_rollout_mode!r}."
                        )
                    if not isinstance(route_predicted_latents, torch.Tensor) or int(route_predicted_latents.shape[2]) <= 0:
                        raise RuntimeError(
                            "joint_video_then_idm route requires joint rollout to produce a non-empty predicted video chunk."
                        )
                    idm_context = _build_infer_context(
                        prompt,
                        action_device=action_device,
                        model_obs_window=model_obs_window,
                        config=config,
                        runtime_device=runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
                        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                        mot_action_only_rollout=False,
                        mot_generalist_rollout_mode=None,
                    )
                    idm_context.extra["action_conditioning_mode"] = "video_conditioned_action"
                    idm_context.extra["mot_generalist_rollout_mode"] = "video_conditioned_action"
                    idm_context.extra["mot_video_condition_latents"] = route_predicted_latents.detach().to(
                        device=runtime_device,
                        dtype=route_predicted_latents.dtype,
                    )
                    infer_output = pipeline._forward_infer_with_visual_outputs(
                        visual_outputs,
                        context=idm_context,
                        infer_state=None if args.reset_policy_state_each_chunk else pre_infer_policy_state,
                    )
                _print_log(
                    "stage",
                    {
                        "name": "chunk_infer_done",
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
            configured_frame_chunk_size = _frame_chunk_size(config)
            configured_action_per_frame = _action_per_frame(config)
            action_per_frame = configured_action_per_frame
            if int(actions.shape[0]) % int(action_per_frame) != 0:
                raise ValueError(
                    "MoT action output length must be divisible by configured action_per_frame, "
                    f"got action_shape={actions.shape}, action_per_frame={action_per_frame}."
                )
            frame_chunk_size = int(actions.shape[0]) // int(action_per_frame)
            execute_action_steps = _resolve_execute_action_steps(
                args.execute_action_steps,
                execute_frame_chunk_size=args.execute_frame_chunk_size,
                action_horizon=int(actions.shape[0]),
                action_per_frame=action_per_frame,
            )
            frame_actions = actions.reshape(frame_chunk_size, action_per_frame, actions.shape[-1])
            predicted_latents = _extract_predicted_latents(infer_output)
            if args.mot_gjd_action_route == "joint_video_then_idm" and isinstance(route_predicted_latents, torch.Tensor):
                predicted_latents = route_predicted_latents
            if isinstance(predicted_latents, torch.Tensor):
                _append_predicted_latent_chunk(
                    predicted_latent_chunks,
                    predicted_latents,
                    max_imagined_latent_frames=args.max_imagined_latent_frames,
                )

            chunk_log = {
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
                "policy_debug": _summarize_policy_debug(infer_output.policy_output.aux),
            }
            _print_log(f"chunk_{chunk_count}", chunk_log)
            chunk_logs.append(chunk_log)

            real_future_frames: list[dict[str, np.ndarray]] = []
            executed_actions = 0
            executed_control_actions: list[np.ndarray] = []
            executed_obs_frames: list[dict[str, np.ndarray]] = []
            policy_debug = _summarize_policy_debug(infer_output.policy_output.aux)
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
                    extracted = _extract_obs(obs)
                    extracted_record = {key: np.array(value, copy=True) for key, value in extracted.items()}
                    rollout_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    frame_window.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    executed_obs_frames.append({key: np.array(value, copy=True) for key, value in extracted_record.items()})
                    # Packed M5 history warmup must encode the dense executed
                    # segment. Sparse keyframes collapse a 4-latent chunk to a
                    # single VAE latent and shift all subsequent history ids.
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
            _print_log(f"chunk_{chunk_count}", chunk_result_log)
            chunk_logs.append(chunk_result_log)

            warmup_action_history = _build_executed_action_history_tensor(
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
                streaming_views = _obs_list_to_views(executed_obs_frames, device=frontend_device)
                streaming_next_visual_outputs = _prepare_mot_visual_outputs(
                    pipeline,
                    views=streaming_views,
                    task_text=(prompt,),
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
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
                    "chunk_index": chunk_count,
                    "phase": "lingbot_streaming_vae_update",
                    "real_obs_frames": int(len(streaming_next_obs_window)),
                    "real_latent_frames": int(streaming_next_visual_outputs.frontend.video_latents.shape[2]),
                }
                _print_log(f"chunk_{chunk_count}", streaming_update_log)
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
                    warmup_debug = _warmup_mot_packed_history_from_visual_outputs(
                        pipeline,
                        config=config,
                        session=session,
                        warmup_outputs=streaming_next_visual_outputs,
                        obs_frame_count=len(streaming_next_obs_window),
                        obs_list=streaming_next_obs_window,
                        action_history=warmup_action_history,
                        runtime_device=runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
                        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                    )
                else:
                    warmup_debug = _warmup_mot_packed_history_from_observations(
                        pipeline,
                        config=config,
                        session=session,
                        obs_list=real_future_frames,
                        action_history=warmup_action_history,
                        task_text=(prompt,),
                        frontend_device=frontend_device,
                        runtime_device=runtime_device,
                        mot_inference_window_size=args.mot_inference_window_size,
                        mot_rollout_frame_chunk_size=args.mot_rollout_frame_chunk_size,
                    )
                warmup_log = {
                    "chunk_index": chunk_count,
                    "phase": "packed_history_warmup",
                    **warmup_debug,
                }
                _print_log(f"chunk_{chunk_count}", warmup_log)
                chunk_logs.append(warmup_log)

            chunk_count += 1

        imagined_video = _decode_latent_video_chunks(
            pipeline,
            predicted_latent_chunks,
            decode_device=decode_device,
            restore_vae=False,
        )

        output_path = _build_output_path(
            root=Path(args.output_dir),
            benchmark_name=args.benchmark,
            task_id=args.task_id,
            prompt=prompt,
            episode_idx=args.episode_idx,
            done=done,
            suffix=args.suffix,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_video_frames(
            output_path,
            _iter_comparison_video_frames(
                real_obs_list=rollout_frames,
                imagined_video=imagined_video,
            ),
            fps=args.video_fps,
        )
        rollout_path = None
        if args.save_rollout_video:
            rollout_path = output_path.with_name(f"{output_path.stem}_rollout.mp4")
            _write_video_frames(
                rollout_path,
                _iter_rollout_video_frames(real_obs_list=rollout_frames),
                fps=args.video_fps,
            )

        summary = {
            "benchmark": args.benchmark,
            "task_id": args.task_id,
            "prompt": prompt,
            "episode_idx": args.episode_idx,
            "success": bool(done),
            "terminal": bool(terminal),
            "chunk_count": chunk_count,
            "env_timestep": int(env.env.timestep),
            "seed": args.seed,
            "video_path": str(output_path.resolve()),
            "comparison_video_path": str(output_path.resolve()),
            "rollout_video_path": None if rollout_path is None else str(rollout_path.resolve()),
            "pipeline": "open_wam_mot",
            "runtime_mode": str(config.policy_variant.runtime_mode),
            "condition_mode": str(config.policy_variant.condition_mode),
            "startup_model_obs_frames": int(startup_model_obs_frames),
            "startup_env_init_steps": int(startup_env_init_steps),
            "startup_env_steps_executed": int(max(startup_env_init_steps, startup_model_obs_frames)),
            "execute_action_steps": None if args.execute_action_steps is None else int(args.execute_action_steps),
            "execute_frame_chunk_size": (
                None if args.execute_frame_chunk_size is None else int(args.execute_frame_chunk_size)
            ),
            "action_count": len(action_trace),
            "checkpoint_file": str(checkpoint_path.resolve()),
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
        load_report_path.write_text(json.dumps(component_report, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
    finally:
        env.close()


def _validate_mot_config(config) -> None:
    if str(config.policy_variant.name) != "mot":
        raise ValueError(
            "run_libero_mot_visualization.py requires a `mot` policy variant, "
            f"got policy_variant.name={config.policy_variant.name!r}."
        )


def _require_current_frontend_encode_mode(
    frontend_encode_mode: str,
    *,
    allow_deprecated: bool,
    source: str,
) -> None:
    if frontend_encode_mode == CURRENT_FRONTEND_ENCODE_MODE:
        return
    if allow_deprecated:
        return
    raise ValueError(
        f"{source} frontend encode mode {frontend_encode_mode!r} is deprecated. "
        f"Use `--frontend-encode-mode {CURRENT_FRONTEND_ENCODE_MODE}`. Pass "
        "`--allow-deprecated-frontend-encode-mode` only for historical debugging."
    )


def _validate_live_sim_mot_generalist_rollout_mode(mode: str | None) -> None:
    if mode is None or mode in LIVE_SIM_MOT_GENERALIST_ROLLOUT_MODES:
        return
    if mode in OFFLINE_DIAGNOSTIC_MOT_GENERALIST_ROLLOUT_MODES:
        supported = ", ".join(sorted(LIVE_SIM_MOT_GENERALIST_ROLLOUT_MODES))
        raise ValueError(
            f"--mot-generalist-rollout-mode={mode!r} is an offline diagnostic mode, not a live sim rollout mode. "
            "It requires ground-truth clean action and/or video condition tensors that this LIBERO visualization "
            f"script does not provide. Use one of [{supported}] here, or use "
            "open_wam.ablations.joint_denoising_fdm.cli for offline FDM/IDM diagnostics."
        )
    raise ValueError(f"Unsupported --mot-generalist-rollout-mode={mode!r}.")


def _maybe_merge_checkpoint_runtime_config(
    config,
    checkpoint_path: Path,
    *,
    merge_enabled: bool,
):
    if not merge_enabled:
        return config, None
    return merge_runtime_config_from_checkpoint(config, checkpoint_path)


def _resolve_mot_checkpoint_path(
    *,
    config_path: Path,
    checkpoint_arg: str | None,
    transformer_subdir: str | None,
) -> Path | None:
    if checkpoint_arg is not None:
        return resolve_checkpoint_file(Path(checkpoint_arg))
    raw = read_yaml_with_local_paths(config_path)
    raw_checkpoint = raw.get("checkpoint_path")
    if raw_checkpoint is not None:
        return resolve_checkpoint_file(Path(str(raw_checkpoint)))
    if transformer_subdir is None:
        return None
    try:
        return resolve_checkpoint_file(
            resolve_checkpoint_step_dir_from_transformer_dir(transformer_subdir)
        )
    except (FileNotFoundError, ValueError):
        return None


def _build_infer_context(
    prompt: str,
    *,
    action_device: torch.device,
    model_obs_window: list[dict[str, np.ndarray]],
    config,
    runtime_device: torch.device,
    mot_inference_window_size: int | None,
    mot_action_only_rollout: bool,
    mot_generalist_rollout_mode: str | None,
    mot_rollout_frame_chunk_size: int | None = None,
):
    extra: dict[str, object] = {"task_text": (prompt,), "action_device": str(action_device)}
    if mot_inference_window_size is not None:
        extra["mot_inference_window_size"] = int(mot_inference_window_size)
    if mot_rollout_frame_chunk_size is not None:
        extra["mot_rollout_frame_chunk_size"] = int(mot_rollout_frame_chunk_size)
    if mot_action_only_rollout:
        extra["mot_action_only_rollout"] = True
    if mot_generalist_rollout_mode is not None:
        _validate_live_sim_mot_generalist_rollout_mode(mot_generalist_rollout_mode)
        extra["action_conditioning_mode"] = str(mot_generalist_rollout_mode)
        extra["mot_generalist_rollout_mode"] = str(mot_generalist_rollout_mode)
    return PolicyInferContext(
        state=build_libero_state_history(
            model_obs_window,
            state_horizon=int(config.data.action_schema.state_horizon),
            state_encoding=config.data.action_target.state_encoding,
        )
        .unsqueeze(0)
        .to(device=runtime_device),
        extra=extra,
    )


def _resolve_task_spec(benchmark_name: str, task_id: int) -> tuple[LiberoTaskSpec, str]:
    ensure_local_libero_config(REPO_ROOT)
    from libero.libero import benchmark  # type: ignore

    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    prompt = benchmark_instance.get_task(task_id).language
    task = benchmark_instance.get_task(task_id)
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        libero_config = yaml.safe_load(handle)
    task_spec = LiberoTaskSpec(
        benchmark_name=benchmark_name,
        task_id=task_id,
        task_name=task.name,
        task_language=task.language,
        problem_folder=task.problem_folder,
        bddl_file_path=benchmark_instance.get_task_bddl_file_path(task_id),
        init_states_path=str(Path(libero_config["init_states"]) / task.problem_folder / f"{task.name}.pruned_init"),
    )
    return task_spec, prompt


def _construct_single_env(task_spec: LiberoTaskSpec):
    ensure_local_libero_config(REPO_ROOT)
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    count = 0
    env = None
    while env is None and count < 5:
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=task_spec.bddl_file_path,
                camera_heights=128,
                camera_widths=128,
            )
        except Exception as exc:  # pragma: no cover - best-effort retry path
            print(f"construct env failed ({count + 1}/5): {exc}")
            time.sleep(5)
            count += 1
    return env


def _init_single_env(
    env,
    init_state,
    *,
    num_frames: int,
    init_steps: int = 5,
) -> list[dict[str, np.ndarray]]:
    env.reset()
    env.set_init_state(init_state)
    if num_frames <= 0:
        raise ValueError(f"Expected positive num_frames, got {num_frames}.")
    if init_steps <= 0:
        raise ValueError(f"Expected positive init_steps, got {init_steps}.")
    resolved_init_steps = max(init_steps, num_frames)
    obs_window: list[dict[str, np.ndarray]] = []
    for _ in range(resolved_init_steps):
        obs, _, _, _ = env.step([0.0] * 7)
        obs_window.append(_extract_obs(obs))
    if not obs_window:
        raise RuntimeError("LIBERO env did not return an observation during initialization.")
    return obs_window[-num_frames:]


def _extract_obs(obs) -> dict[str, np.ndarray]:
    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(obs["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy(),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy(),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).copy(),
    }


def _obs_list_to_views(
    obs_list: list[dict[str, np.ndarray]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        LIBERO_OBS_KEYS[0]: torch.from_numpy(np.stack([obs[LIBERO_OBS_KEYS[0]] for obs in obs_list], axis=0)).to(device=device),
        LIBERO_OBS_KEYS[1]: torch.from_numpy(np.stack([obs[LIBERO_OBS_KEYS[1]] for obs in obs_list], axis=0)).to(device=device),
    }


def _select_model_obs_window(
    frame_window: list[dict[str, np.ndarray]],
    *,
    chunk_index: int,
    startup_model_obs_frames: int,
) -> list[dict[str, np.ndarray]]:
    if not frame_window:
        raise ValueError("MoT visualization requires at least one observation frame.")
    if chunk_index == 0:
        if startup_model_obs_frames <= 0:
            raise ValueError(
                f"Expected positive startup_model_obs_frames, got {startup_model_obs_frames}."
            )
        if startup_model_obs_frames > len(frame_window):
            raise ValueError(
                "startup_model_obs_frames cannot exceed the available startup window, "
                f"got startup_model_obs_frames={startup_model_obs_frames}, window={len(frame_window)}."
            )
        return list(frame_window[-startup_model_obs_frames:])
    return list(frame_window)


def _prepare_mot_visual_outputs(
    pipeline,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    use_streaming_frontend: bool,
    preserve_stream_cache: bool = False,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
):
    if use_streaming_frontend:
        canonical_batch = pipeline.canonicalize(views)
        canonical_video = canonical_batch.video.to(device=frontend_device)
        frontend_output = pipeline.visual_tower.run_frontend(
            canonical_video,
            placements=canonical_batch.placements,
            task_text=task_text,
            text_context=text_context,
            negative_text_context=negative_text_context,
            preserve_stream_cache=preserve_stream_cache,
        )
        runtime_dtype = pipeline.visual_tower.core.patch_embedding_mlp.weight.dtype
        return pipeline.prepare_visual_outputs_from_latents(
            frontend_output.video_latents.to(device=runtime_device, dtype=runtime_dtype),
            task_text=task_text,
            text_context=(
                None
                if frontend_output.conditioning.text_context is None
                else frontend_output.conditioning.text_context.to(device=runtime_device, dtype=runtime_dtype)
            ),
            negative_text_context=(
                None
                if frontend_output.conditioning.negative_text_context is None
                else frontend_output.conditioning.negative_text_context.to(
                    device=runtime_device,
                    dtype=runtime_dtype,
                )
            ),
            canonical_video=canonical_video.to(device=runtime_device),
        )
    return _prepare_visual_outputs_offline(
        pipeline,
        views=views,
        task_text=task_text,
        frontend_device=frontend_device,
        runtime_device=runtime_device,
        text_context=text_context,
        negative_text_context=negative_text_context,
    )


def _prepare_visual_outputs_offline(
    pipeline,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
):
    canonical_batch = pipeline.canonicalize(views)
    canonical_video = canonical_batch.video.to(device=frontend_device)
    frontend = pipeline.visual_tower.frontend
    assets = frontend.reference_assets
    runtime_dtype = pipeline.visual_tower.core.patch_embedding_mlp.weight.dtype

    if assets.has_vae:
        video_latents = _encode_video_window_offline(
            assets,
            canonical_video=canonical_video,
            placements=canonical_batch.placements,
            device=frontend_device,
        ).to(device=runtime_device, dtype=runtime_dtype)
        resolved_text_context = text_context
        if resolved_text_context is None:
            resolved_text_context = assets.encode_text(
                task_text,
                device=frontend_device,
                dtype=canonical_video.dtype,
            )
        resolved_negative_text_context = negative_text_context
        if resolved_negative_text_context is None and resolved_text_context is not None:
            resolved_negative_text_context = assets.encode_blank_text(
                batch_size=canonical_video.shape[0],
                device=frontend_device,
                dtype=canonical_video.dtype,
            )
        return pipeline.prepare_visual_outputs_from_latents(
            video_latents,
            task_text=task_text,
            text_context=(
                None
                if resolved_text_context is None
                else resolved_text_context.to(device=runtime_device, dtype=runtime_dtype)
            ),
            negative_text_context=(
                None
                if resolved_negative_text_context is None
                else resolved_negative_text_context.to(device=runtime_device, dtype=runtime_dtype)
            ),
            canonical_video=canonical_video.to(device=runtime_device),
        )

    return pipeline.prepare_visual_outputs(
        views,
        task_text=task_text,
        text_context=text_context,
        negative_text_context=negative_text_context,
    )


def _build_output_path(
    *,
    root: Path,
    benchmark_name: str,
    task_id: int,
    prompt: str,
    episode_idx: int,
    done: bool,
    suffix: str,
) -> Path:
    safe_prompt = prompt.replace(" ", "_")
    return root / benchmark_name / f"{task_id}_{safe_prompt}" / f"{episode_idx}_{done}_{suffix}.mp4"


def _warmup_mot_packed_history_from_observations(
    pipeline,
    *,
    config,
    session,
    obs_list: list[dict[str, np.ndarray]],
    action_history: torch.Tensor | None,
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    mot_inference_window_size: int | None,
    mot_rollout_frame_chunk_size: int | None,
) -> dict[str, object]:
    if not obs_list:
        return {"warmup_skipped": True, "reason": "empty_obs_list"}
    policy_state = session.policy_state
    runtime_state = getattr(policy_state, "variant_state", None) if policy_state is not None else None
    if runtime_state is None or not hasattr(runtime_state, "past_clean_latents"):
        return {"warmup_skipped": True, "reason": "no_mot_runtime_state"}

    views = _obs_list_to_views(obs_list, device=frontend_device)
    warmup_outputs = _prepare_mot_visual_outputs(
        pipeline,
        views=views,
        task_text=task_text,
        frontend_device=frontend_device,
        runtime_device=runtime_device,
        use_streaming_frontend=False,
        text_context=session.text_context,
        negative_text_context=session.negative_text_context,
    )
    return _warmup_mot_packed_history_from_visual_outputs(
        pipeline,
        config=config,
        session=session,
        warmup_outputs=warmup_outputs,
        obs_frame_count=len(obs_list),
        obs_list=obs_list,
        action_history=action_history,
        runtime_device=runtime_device,
        mot_inference_window_size=mot_inference_window_size,
        mot_rollout_frame_chunk_size=mot_rollout_frame_chunk_size,
    )


def _warmup_mot_packed_history_from_visual_outputs(
    pipeline,
    *,
    config,
    session,
    warmup_outputs,
    obs_frame_count: int,
    obs_list: list[dict[str, np.ndarray]] | None,
    action_history: torch.Tensor | None,
    runtime_device: torch.device,
    mot_inference_window_size: int | None,
    mot_rollout_frame_chunk_size: int | None,
) -> dict[str, object]:
    policy_state = session.policy_state
    runtime_state = getattr(policy_state, "variant_state", None) if policy_state is not None else None
    if runtime_state is None or not hasattr(runtime_state, "past_clean_latents"):
        return {"warmup_skipped": True, "reason": "no_mot_runtime_state"}

    runtime_dtype = pipeline.visual_tower.core.patch_embedding_mlp.weight.dtype
    real_latents = warmup_outputs.frontend.video_latents.to(device=runtime_device, dtype=runtime_dtype)
    past_latents = runtime_state.past_clean_latents
    past_hidden_proprio = getattr(runtime_state, "past_hidden_proprio_states", None)
    frame_chunk_size = (
        int(mot_rollout_frame_chunk_size)
        if mot_rollout_frame_chunk_size is not None
        else _frame_chunk_size(config)
    )
    history_window_size = (
        int(mot_inference_window_size)
        if mot_inference_window_size is not None
        else int(getattr(pipeline.policy_variant.training_config, "window_size", real_latents.shape[2]))
    )
    if history_window_size <= 0:
        raise ValueError(f"MoT packed warmup window size must be positive, got {history_window_size}.")
    history_window_frames = max(
        int(real_latents.shape[2]),
        resolve_mot_rollout_cache_window_frames(
            window_size=history_window_size,
            frame_chunk_size=frame_chunk_size,
        ),
    )
    dropped_pred_latent_frames = 0
    pending_pred_latent_frames = int(
        getattr(runtime_state, "pending_predicted_video_frames", frame_chunk_size)
    )
    if past_latents is None:
        base_latents = None
    else:
        past_latents = past_latents.to(device=runtime_device, dtype=runtime_dtype)
        dropped_pred_latent_frames = min(
            max(0, pending_pred_latent_frames),
            int(past_latents.shape[2]),
        )
        if dropped_pred_latent_frames <= 0:
            base_latents = past_latents
        elif int(getattr(policy_state, "step_index", 0)) <= 1 and dropped_pred_latent_frames >= int(past_latents.shape[2]):
            # Chunk 0 keeps the single bootstrap observation; real_future_frames
            # only contains frames collected after executing the first action
            # chunk, mirroring M1's initial_latents + key_frame_latents warmup.
            base_latents = past_latents[:, :, :1]
            dropped_pred_latent_frames = max(0, int(past_latents.shape[2]) - 1)
        else:
            base_latents = past_latents[:, :, :-dropped_pred_latent_frames]
    if base_latents is None or int(base_latents.shape[2]) == 0:
        combined = real_latents
    else:
        combined = torch.cat([base_latents, real_latents], dim=2)
    runtime_state.past_clean_latents = combined[:, :, -history_window_frames:].detach()
    runtime_state.pending_predicted_video_frames = 0

    appended_hidden_proprio_frames = 0
    if (
        obs_list is not None
        and hasattr(runtime_state, "past_hidden_proprio_states")
        and getattr(runtime_state, "past_hidden_proprio_states", None) is not None
    ):
        raw_state = build_libero_state_history(
            obs_list,
            state_horizon=len(obs_list),
            state_encoding=config.data.action_target.state_encoding,
        ).to(device=runtime_device, dtype=runtime_dtype)
        if raw_state.ndim == 2 and int(raw_state.shape[0]) > 0 and int(real_latents.shape[2]) > 0:
            if int(raw_state.shape[0]) >= int(real_latents.shape[2]):
                latent_state = raw_state[-int(real_latents.shape[2]) :, :]
            else:
                pad_count = int(real_latents.shape[2]) - int(raw_state.shape[0])
                latent_state = torch.cat(
                    [
                        raw_state,
                        raw_state[-1:, :].expand(pad_count, -1),
                    ],
                    dim=0,
                )
            latent_state = latent_state.unsqueeze(0)
            if past_hidden_proprio is None:
                base_hidden = None
            else:
                past_hidden_proprio = past_hidden_proprio.to(device=runtime_device, dtype=runtime_dtype)
                base_hidden = (
                    past_hidden_proprio[:, :-dropped_pred_latent_frames]
                    if dropped_pred_latent_frames > 0
                    else past_hidden_proprio
                )
            if base_hidden is None or int(base_hidden.shape[1]) == 0:
                combined_hidden = latent_state
            else:
                combined_hidden = torch.cat([base_hidden, latent_state], dim=1)
            runtime_state.past_hidden_proprio_states = combined_hidden[:, -history_window_frames:].detach()
            appended_hidden_proprio_frames = int(latent_state.shape[1])

    appended_action_tokens = 0
    dropped_pred_action_tokens = 0
    if action_history is not None and hasattr(runtime_state, "past_clean_actions"):
        action_dim = int(getattr(pipeline.policy_variant, "action_dim", action_history.shape[-1]))
        if action_history.ndim != 3 or action_history.shape[-1] != action_dim:
            raise ValueError(
                "MoT packed warmup action history must be [B, T_action, D_action], "
                f"got {tuple(action_history.shape)}, action_dim={action_dim}."
            )
        action_tokens_per_frame = _action_per_frame(config)
        action_horizon = int(config.data.action_schema.action_horizon)
        warm_actions = action_history[:, :action_horizon].to(device=runtime_device, dtype=runtime_dtype)
        if warm_actions.shape[1] > 0:
            past_actions = runtime_state.past_clean_actions
            if past_actions is None:
                base_actions = None
            else:
                past_actions = past_actions.to(device=runtime_device, dtype=runtime_dtype)
                dropped_pred_action_tokens = min(int(action_horizon), int(past_actions.shape[1]))
                base_actions = past_actions[:, :-dropped_pred_action_tokens] if dropped_pred_action_tokens > 0 else past_actions
            if base_actions is None or int(base_actions.shape[1]) == 0:
                combined_actions = warm_actions
            else:
                combined_actions = torch.cat([base_actions, warm_actions], dim=1)
            max_action_history_tokens = int(history_window_frames) * int(action_tokens_per_frame)
            runtime_state.past_clean_actions = combined_actions[:, -max_action_history_tokens:].detach()
            appended_action_tokens = int(warm_actions.shape[1])
    if warmup_outputs.frontend.conditioning.text_context is not None:
        session.text_context = warmup_outputs.frontend.conditioning.text_context
    if warmup_outputs.frontend.conditioning.negative_text_context is not None:
        session.negative_text_context = warmup_outputs.frontend.conditioning.negative_text_context
    return {
        "warmup_skipped": False,
        "real_obs_frames": int(obs_frame_count),
        "real_latent_frames": int(real_latents.shape[2]),
        "past_clean_latent_frames_after": int(runtime_state.past_clean_latents.shape[2]),
        "past_clean_action_frames_after": (
            0
            if getattr(runtime_state, "past_clean_actions", None) is None
            else int(runtime_state.past_clean_actions.shape[1] // _action_per_frame(config))
        ),
        "appended_action_tokens": int(appended_action_tokens),
        "appended_hidden_proprio_frames": int(appended_hidden_proprio_frames),
        "pending_pred_latent_frames_before": int(pending_pred_latent_frames),
        "dropped_pred_latent_frames": int(dropped_pred_latent_frames),
        "dropped_pred_action_tokens": int(dropped_pred_action_tokens),
        "inference_window_size": int(history_window_size),
        "history_window_frames": int(history_window_frames),
    }


def _encode_video_window_offline(
    assets,
    *,
    canonical_video: torch.Tensor,
    placements,
    device: torch.device,
) -> torch.Tensor:
    if not assets.has_vae:
        raise RuntimeError("Wan VAE assets are not loaded for offline video encoding.")
    del device
    return assets.encode_video(canonical_video, placements=placements, reset_cache=True)



def _build_executed_action_history_tensor(
    executed_control_actions: list[np.ndarray],
    *,
    start_frame_group: int,
    action_per_frame: int,
    action_dim: int,
) -> torch.Tensor | None:
    return _build_shared_executed_action_history_tensor(
        executed_control_actions,
        start_frame_group=start_frame_group,
        action_per_frame=action_per_frame,
        action_dim=action_dim,
    )


def _frame_chunk_size(config) -> int:
    frame_chunk_size = max(1, int(config.inference.frame_chunk_size))
    action_horizon = int(config.data.action_schema.action_horizon)
    if action_horizon % frame_chunk_size != 0:
        raise ValueError(
            "MoT rollout expects action_horizon to divide by inference.frame_chunk_size, "
            f"got action_horizon={action_horizon}, frame_chunk_size={frame_chunk_size}."
        )
    return frame_chunk_size


def _action_per_frame(config) -> int:
    return max(1, int(config.data.action_schema.action_horizon) // _frame_chunk_size(config))


def _resolve_execute_action_steps(
    execute_action_steps: int | None,
    *,
    execute_frame_chunk_size: int | None = None,
    action_horizon: int,
    action_per_frame: int,
) -> int:
    return _resolve_shared_execute_action_steps(
        execute_action_steps,
        execute_frame_chunk_size=execute_frame_chunk_size,
        action_horizon=action_horizon,
        action_per_frame=action_per_frame,
    )


def _append_predicted_latent_chunk(
    predicted_latent_chunks: list[torch.Tensor],
    predicted_latents: torch.Tensor,
    *,
    max_imagined_latent_frames: int | None,
) -> None:
    if predicted_latents.ndim != 5:
        raise ValueError(
            "Predicted latent chunks must have shape [B, C, T, H, W], "
            f"got {tuple(predicted_latents.shape)}."
        )
    if max_imagined_latent_frames is not None:
        cap = int(max_imagined_latent_frames)
        if cap <= 0:
            return
        retained_frames = sum(int(chunk.shape[2]) for chunk in predicted_latent_chunks)
        if retained_frames >= cap:
            return
        remaining_frames = cap - retained_frames
        predicted_latents = predicted_latents[:, :, :remaining_frames]
    if int(predicted_latents.shape[2]) <= 0:
        return
    predicted_latent_chunks.append(predicted_latents.detach().cpu())


def _write_video_frames(
    output_path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float,
) -> None:
    wrote_frame = False
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
            wrote_frame = True
    if not wrote_frame:
        raise ValueError(f"No frames were produced for video output {output_path}.")


def _build_rollout_video_frames(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
) -> list[np.ndarray]:
    return list(_iter_rollout_video_frames(real_obs_list=real_obs_list))


def _iter_rollout_video_frames(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
) -> Iterable[np.ndarray]:
    for obs in real_obs_list:
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        row_real = np.ascontiguousarray(row_real)
        row_real = np.array(_with_title(Image.fromarray(row_real), "MoT Rollout (AgentView / Wrist)"), copy=True)
        yield np.ascontiguousarray(row_real)


def _build_comparison_video_frames(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
    imagined_video: np.ndarray | None,
) -> list[np.ndarray]:
    return list(
        _iter_comparison_video_frames(
            real_obs_list=real_obs_list,
            imagined_video=imagined_video,
        )
    )


def _iter_comparison_video_frames(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
    imagined_video: np.ndarray | None,
) -> Iterable[np.ndarray]:
    panel_height = 300
    total = len(real_obs_list)
    for frame_index in range(total):
        real_obs = real_obs_list[frame_index]
        imagined_frame = _imagined_frame_for_rollout_index(
            imagined_video=imagined_video,
            frame_index=frame_index,
            target_length=total,
        )
        agentview = np.ascontiguousarray(real_obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(real_obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        row_real = np.ascontiguousarray(row_real)
        row_real = np.array(
            _with_title(Image.fromarray(row_real), f"Real Rollout Frame {frame_index}"),
            copy=True,
        )
        target_width = row_real.shape[1]
        if imagined_frame is None:
            row_imagined = Image.new("RGB", (target_width, panel_height), color=(0, 0, 0))
            draw = ImageDraw.Draw(row_imagined)
            draw.text((10, panel_height // 2), "No imagined frame", fill=(120, 120, 120))
        else:
            image = Image.fromarray(_to_uint8(imagined_frame))
            scale = min(target_width / image.width, panel_height / image.height)
            resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
            row_imagined = Image.new("RGB", (target_width, panel_height), color=(0, 0, 0))
            row_imagined.paste(
                resized,
                ((target_width - resized.width) // 2, (panel_height - resized.height) // 2),
            )
        row_imagined = _with_title(
            row_imagined,
            f"Imagined Frame {frame_index}",
        )
        yield np.ascontiguousarray(np.vstack([row_real, np.array(row_imagined, copy=True)]))


def _imagined_frame_for_rollout_index(
    *,
    imagined_video: np.ndarray | None,
    frame_index: int,
    target_length: int,
) -> np.ndarray | None:
    if imagined_video is None or target_length <= 0:
        return None
    imagined_frame_count = len(imagined_video)
    if imagined_frame_count <= 0:
        return None
    if imagined_frame_count == 1 or target_length == 1:
        imagined_index = 0
    elif imagined_frame_count == target_length:
        imagined_index = frame_index
    else:
        imagined_index = int(round(frame_index * (imagined_frame_count - 1) / (target_length - 1)))
    imagined_index = max(0, min(imagined_frame_count - 1, imagined_index))
    return np.array(imagined_video[imagined_index], copy=True)


def _align_imagined_video_to_rollout(
    *,
    imagined_video: np.ndarray | None,
    target_length: int,
) -> list[np.ndarray]:
    if imagined_video is None or target_length <= 0:
        return []
    imagined_frames = list(imagined_video)
    if not imagined_frames:
        return []
    if len(imagined_frames) == target_length:
        return [np.array(frame, copy=True) for frame in imagined_frames]
    if len(imagined_frames) == 1:
        return [np.array(imagined_frames[0], copy=True) for _ in range(target_length)]
    indices = np.linspace(0, len(imagined_frames) - 1, num=target_length)
    return [np.array(imagined_frames[int(round(index))], copy=True) for index in indices.tolist()]


def _with_title(image: Image.Image, title: str) -> Image.Image:
    title_height = 36
    canvas = Image.new("RGB", (image.width, image.height + title_height), color=(0, 0, 0))
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    return canvas


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if float(frame.max()) <= 1.0001:
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def _decode_latent_video_chunks(
    pipeline,
    latent_chunks: list[torch.Tensor],
    *,
    decode_device: torch.device,
    restore_vae: bool = True,
) -> np.ndarray | None:
    if not latent_chunks:
        return None
    return _decode_latent_video(
        pipeline,
        torch.cat(latent_chunks, dim=2),
        decode_device=decode_device,
        restore_vae=restore_vae,
    )


def _extract_predicted_latents(infer_output) -> torch.Tensor | None:
    predicted_latents = infer_output.decoder_output.aux.get("predicted_latents")
    if not isinstance(predicted_latents, torch.Tensor):
        predicted_latents = infer_output.policy_output.aux.get("predicted_latents")
    return predicted_latents if isinstance(predicted_latents, torch.Tensor) else None


def _decode_latent_video(
    pipeline,
    latents: torch.Tensor,
    *,
    decode_device: torch.device,
    restore_vae: bool = True,
) -> np.ndarray | None:
    assets = pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None
    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype
    target_dtype = torch.bfloat16 if decode_device.type == "cuda" else torch.float32
    if original_device != decode_device or original_dtype != target_dtype:
        vae = vae.to(device=decode_device, dtype=target_dtype)
    latents = latents.to(device=decode_device, dtype=target_dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
        .view(1, vae.config.z_dim, 1, 1, 1)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
        .view(1, vae.config.z_dim, 1, 1, 1)
    )
    latents = latents / latents_std + latents_mean
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    imagined_video = video_processor.postprocess_video(decoded, output_type="np")[0]
    if (
        restore_vae
        and (
            next(assets.vae.parameters()).device != original_device
            or next(assets.vae.parameters()).dtype != original_dtype
        )
    ):
        assets.vae = assets.vae.to(device=original_device, dtype=original_dtype)
    return imagined_video


def _build_component_report(
    config,
    pipeline,
    *,
    runtime_device: torch.device,
    action_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    raw_window_frames: int,
    mot_inference_window_size: int | None,
    mot_rollout_frame_chunk_size: int | None,
    mot_action_only_rollout: bool,
    mot_generalist_rollout_mode: str | None,
    mot_gjd_action_route: str,
) -> dict[str, object]:
    backbone = config.backbone
    policy_variant = pipeline.policy_variant
    action_expert = getattr(policy_variant, "action_expert", None)
    return {
        "pipeline": "open_wam_mot",
        "runtime_device": str(runtime_device),
        "action_device": str(action_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "raw_window_frames": int(raw_window_frames),
        "mot_inference_window_size": (
            None if mot_inference_window_size is None else int(mot_inference_window_size)
        ),
        "mot_rollout_frame_chunk_size": (
            None if mot_rollout_frame_chunk_size is None else int(mot_rollout_frame_chunk_size)
        ),
        "mot_action_only_rollout": bool(mot_action_only_rollout),
        "mot_generalist_rollout_mode": mot_generalist_rollout_mode,
        "mot_gjd_action_route": str(mot_gjd_action_route),
        "config_name": config.name,
        "policy_variant_class": policy_variant.__class__.__name__,
        "runtime_mode": str(policy_variant.config.runtime_mode),
        "condition_mode": str(policy_variant.config.condition_mode),
        "video_prefix_frames": int(policy_variant.config.video_prefix_frames),
        "video_can_attend_action": bool(getattr(policy_variant.config, "video_can_attend_action", False)),
        "backbone_hidden_size": int(backbone.hidden_size),
        "backbone_num_layers": int(backbone.num_layers),
        "action_hidden_size": (
            None if action_expert is None else int(getattr(action_expert, "hidden_size", 0))
        ),
        "action_num_layers": int(policy_variant.config.num_action_layers),
        "action_horizon": int(config.data.action_schema.action_horizon),
        "action_dim": int(config.data.action_schema.action_dim),
        "trainable_parameters": _count_trainable_parameters(pipeline),
        "total_parameters": sum(parameter.numel() for parameter in pipeline.parameters()),
        "backbone_pretrained_root": str(backbone.pretrained_model_name_or_path),
        "transformer_subdir": str(backbone.transformer_subdir),
        "config_sha256": _sha256_if_exists(Path(backbone.pretrained_model_name_or_path) / "transformer" / "config.json")
        if backbone.pretrained_model_name_or_path
        else None,
    }


def _sha256_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_log(label: str, payload: dict[str, object]) -> None:
    print(f"[{label}] {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _summarize_policy_debug(aux: dict[str, object]) -> dict[str, object]:
    """Keep rollout logs readable by replacing large tensors with metadata."""

    summary: dict[str, object] = {}
    for key, value in aux.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
            continue
        if key == "mot_infer_artifacts" and hasattr(value, "action_pred"):
            action_pred = getattr(value, "action_pred", None)
            predicted_latents = getattr(value, "predicted_latents", None)
            summary[key] = {
                "action_pred_shape": (
                    list(action_pred.shape) if isinstance(action_pred, torch.Tensor) else None
                ),
                "predicted_latents_shape": (
                    list(predicted_latents.shape) if isinstance(predicted_latents, torch.Tensor) else None
                ),
                "condition_mode": str(getattr(value, "condition_mode", "")),
                "runtime_mode": str(getattr(value, "runtime_mode", "")),
            }
            continue
        if isinstance(value, dict):
            summary[key] = value
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            summary[key] = value
            continue
        summary[key] = type(value).__name__
    return summary


def _resolve_device(device_arg: str | None, *, fallback: torch.device | None = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_raw_window_frames(latent_num_frames: int) -> int:
    return raw_window_frames_for_latents(latent_num_frames)


if __name__ == "__main__":
    main()
