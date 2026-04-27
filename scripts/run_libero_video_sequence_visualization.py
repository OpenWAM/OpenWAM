from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers.video_processor import VideoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import ActionTargetRepresentation, ReferenceCoreInitMode  # noqa: E402
from open_wam.data import reconstruct_absolute_pose_targets  # noqa: E402
from open_wam.data.action_transforms import PoseSequence, normalize_quaternion, quaternion_to_axis_angle  # noqa: E402
from open_wam.integrations import (  # noqa: E402
    LiberoControlConfig,
    LiberoTaskSpec,
    compute_osc_pose_action,
    ensure_local_libero_config,
    load_libero_task_init_states,
)
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.models.policy_variants.common import derive_video_condition_sample_seed  # noqa: E402
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils import (  # noqa: E402
    load_experiment_config,
    merge_runtime_config_from_checkpoint,
    seed_everywhere,
)

LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def _build_sequence_rollout_infer_extra(
    *,
    config,
    prompt: str,
    generation_action_start: int,
    runtime_device: torch.device | None = None,
    task_id: int | None = None,
    episode_idx: int | None = None,
) -> dict[str, object]:
    extra: dict[str, object] = {
        "task_text": (prompt,),
    }
    policy_name = str(config.policy_variant.name)
    if policy_name in {"post_latent", "post_decoded"}:
        extra["video_condition_frame_start"] = int(generation_action_start)
        # Runtime rollouts only have a trailing observed window. Use the newest
        # latent as the conditioning prefix so generated future frames start
        # from the same current observation in visualization and sandbox paths.
        extra["video_condition_observed_prefix_anchor"] = "end"
        sample_seed = derive_video_condition_sample_seed(
            {
                "task_index": task_id,
                "episode_index": episode_idx,
                "anchor_frame_index": int(generation_action_start),
                "action_start_index": int(generation_action_start),
            }
        )
        if sample_seed is not None:
            extra["video_condition_sample_seed"] = int(sample_seed)
    if policy_name == "mot" and runtime_device is not None:
        extra["action_device"] = str(runtime_device)
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one method-3 LIBERO rollout and save decoded predicted-latent videos."
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/video_sequence_policy_libero_latent_local_random_subwindow.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file, checkpoint_step_* directory, or run directory. "
        "If omitted, infer from `backbone.transformer_subdir` in the config.",
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_video_sequence_visualization")
    parser.add_argument("--suffix", type=str, default="open_wam_method3")
    parser.add_argument(
        "--raw-window-frames",
        type=int,
        default=None,
        help="Number of raw env frames to encode into one latent window. Defaults to 4 * latent_num_frames - 1.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--runtime-devices", type=str, default=None)
    parser.add_argument("--runtime-prep-device", type=str, default=None)
    parser.add_argument("--runtime-output-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument("--video-steps", type=int, default=None)
    parser.add_argument("--action-steps", type=int, default=None)
    parser.add_argument(
        "--rollout-chunk-steps",
        type=int,
        default=None,
        help="Optional override for action_decoder.rollout_chunk_steps. Defaults to the experiment config.",
    )
    parser.add_argument(
        "--initial-generation-action-start",
        type=int,
        default=None,
        help=(
            "Optional initial generation action/frame start for generated-video rollouts. "
            "Defaults to the exact warmup-window length."
        ),
    )
    parser.add_argument(
        "--visualization-mode",
        type=str,
        default="compare_observed_vs_predicted",
        choices=("compare_keyframes", "compare_observed_vs_predicted", "env_only", "latent_grid"),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    checkpoint_path = _resolve_checkpoint_path_from_args_or_config(
        checkpoint_arg=args.checkpoint,
        transformer_subdir=str(config.backbone.transformer_subdir),
    )
    config, _ = merge_runtime_config_from_checkpoint(config, checkpoint_path)
    if args.video_steps is not None:
        object.__setattr__(config.inference, "video_num_inference_steps", int(args.video_steps))
    if args.action_steps is not None:
        object.__setattr__(config.inference, "action_num_inference_steps", int(args.action_steps))
    _apply_rollout_chunk_steps_override(config, args.rollout_chunk_steps)
    action_target_representation = ActionTargetRepresentation(config.data.action_target.representation)

    checkpoint_step_dir = checkpoint_path.parent
    transformer_dir = checkpoint_step_dir / "transformer"
    if transformer_dir.is_dir():
        object.__setattr__(config.backbone, "transformer_subdir", str(transformer_dir.resolve()))
        object.__setattr__(config.backbone, "reference_core_init_mode", ReferenceCoreInitMode.FULL)

    print(
        json.dumps(
            {
                "phase": "config_override",
                "checkpoint_file": str(checkpoint_path),
                "checkpoint_step_dir": str(checkpoint_step_dir),
                "effective_transformer_subdir": str(config.backbone.transformer_subdir),
                "effective_reference_core_init_mode": str(config.backbone.reference_core_init_mode),
            }
        )
    )

    runtime_device = _resolve_device(args.runtime_device)
    frontend_device = _resolve_device(args.frontend_device, fallback=runtime_device)
    decode_device = _resolve_device(args.decode_device, fallback=frontend_device)
    runtime_devices = _resolve_runtime_devices(args.runtime_devices, fallback=runtime_device)
    runtime_prep_device = _resolve_device(args.runtime_prep_device, fallback=runtime_device)
    runtime_output_device = _resolve_device(args.runtime_output_device, fallback=runtime_device)
    raw_window_frames = (
        int(args.raw_window_frames)
        if args.raw_window_frames is not None
        else _default_raw_window_frames(int(config.data.num_frames))
    )

    _print_log("stage", {"name": "build_pipeline_start"})
    pipeline = build_variant_pipeline_from_config(config)
    _print_log("stage", {"name": "build_pipeline_done"})
    _print_log("stage", {"name": "load_checkpoint_start", "checkpoint_file": str(checkpoint_path)})
    _load_pipeline_checkpoint(pipeline, checkpoint_path)
    _print_log("stage", {"name": "load_checkpoint_done"})
    _print_log("stage", {"name": "move_pipeline_start", "runtime_device": str(runtime_device)})
    pipeline = pipeline.to(runtime_device)
    _print_log("stage", {"name": "move_pipeline_done"})
    pipeline.visual_tower.configure_runtime_devices(
        runtime_devices,
        prep_device=runtime_prep_device,
        output_device=runtime_output_device,
    )
    pipeline.eval()
    runner = VariantRolloutRunner(pipeline)

    load_report = {
        "pipeline": "open_wam_video_sequence",
        "policy_variant": config.policy_variant.name,
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_step_dir": str(checkpoint_step_dir),
        "transformer_dir": str(transformer_dir.resolve()) if transformer_dir.is_dir() else None,
        "runtime_device": str(runtime_device),
        "runtime_prep_device": str(runtime_prep_device),
        "runtime_output_device": str(runtime_output_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "runtime_devices": [str(device) for device in runtime_devices],
        "video_steps": int(config.inference.video_num_inference_steps),
        "action_steps": int(config.inference.action_num_inference_steps),
        "data_num_frames": int(config.data.num_frames),
        "raw_window_frames": raw_window_frames,
        "action_horizon": int(config.data.action_schema.action_horizon),
    }
    _print_log("load_report", load_report)

    _print_log("stage", {"name": "resolve_task_start", "benchmark": args.benchmark, "task_id": args.task_id})
    task_spec, prompt = _resolve_task_spec(args.benchmark, args.task_id)
    _print_log("stage", {"name": "resolve_task_done", "task_name": task_spec.task_name, "prompt": prompt})
    _print_log("stage", {"name": "load_init_states_start"})
    init_states = load_libero_task_init_states(task_spec)
    _print_log("stage", {"name": "load_init_states_done", "num_init_states": len(init_states)})
    _print_log("stage", {"name": "construct_env_start"})
    env = _construct_single_env(task_spec)
    _print_log("stage", {"name": "construct_env_done", "env_created": env is not None})
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    try:
        with torch.inference_mode():
            _print_log("stage", {"name": "init_env_rollout_start"})
            initial_obs_window = _init_single_env(
                env,
                init_states[args.episode_idx % len(init_states)],
                num_frames=raw_window_frames,
            )
            _print_log("stage", {"name": "init_env_rollout_done", "initial_obs_window": len(initial_obs_window)})
            initial_rollout_obs = list(initial_obs_window)
            _print_log("stage", {"name": "prepare_initial_inputs_start"})
            initial_inputs = _prepare_rollout_inputs(
                pipeline,
                views=_obs_window_to_rollout_views(initial_rollout_obs, device=frontend_device),
                task_text=(prompt,),
                frontend_device=frontend_device,
                runtime_device=runtime_device,
            )
            _print_log(
                "stage",
                {
                    "name": "prepare_initial_inputs_done",
                    "initial_video_latents_shape": list(initial_inputs["video_latents"].shape),
                },
            )
            session = runner.reset(
                task_text=(prompt,),
                text_context=initial_inputs["text_context"],
                negative_text_context=initial_inputs["negative_text_context"],
            )

            predicted_latent_chunks: list[torch.Tensor] = []
            observed_latent_chunks: list[torch.Tensor] = [initial_inputs["video_latents"].detach().cpu()]
            obs_window = list(initial_obs_window)
            real_obs_list: list[dict[str, np.ndarray]] = [
                {key: np.array(value, copy=True) for key, value in obs.items()}
                for obs in initial_obs_window
            ]
            keyframe_obs_list: list[dict[str, np.ndarray]] = []
            done = False
            chunk_count = 0
            next_generation_action_start = _resolve_initial_generation_action_start(
                initial_obs_window,
                initial_generation_action_start=args.initial_generation_action_start,
                rollout_starts_at_action_zero=_uses_zero_based_generation_start(config),
            )

            while env.env.timestep < args.max_timestep and not done:
                if args.max_chunks is not None and chunk_count >= args.max_chunks:
                    break

                if args.seed is not None:
                    seed_everywhere(args.seed + chunk_count)

                rollout_obs = list(obs_window)
                rollout_inputs = _prepare_rollout_inputs(
                    pipeline,
                    views=_obs_window_to_rollout_views(rollout_obs, device=frontend_device),
                    task_text=(prompt,),
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                    text_context=session.text_context,
                    negative_text_context=session.negative_text_context,
                    preserve_stream_cache=False,
                )
                if chunk_count > 0:
                    observed_latent_chunks.append(rollout_inputs["video_latents"].detach().cpu())

                timestep_before = int(env.env.timestep)
                step_output = runner.infer_step(
                    session=session,
                    context=PolicyInferContext(
                        state=_build_state_inputs_from_obs_window(
                            obs_window,
                            state_horizon=int(config.data.action_schema.state_horizon),
                            state_encoding=str(config.data.action_target.state_encoding),
                        ).unsqueeze(0).to(device=runtime_device),
                        extra=_build_sequence_rollout_infer_extra(
                            config=config,
                            prompt=prompt,
                            generation_action_start=int(next_generation_action_start),
                            runtime_device=runtime_device,
                            task_id=int(args.task_id),
                            episode_idx=int(args.episode_idx),
                        ),
                    ),
                    video_latents=rollout_inputs["video_latents"],
                )
                session = step_output.session
                infer_output = step_output.infer_output

                action_pred, action_plan_metadata = _decoder_output_to_rollout_action_plan(
                    infer_output.decoder_output
                )
                _advance_decoder_state_to_rollout_commit(session, action_plan_metadata)
                predicted_latents = infer_output.decoder_output.aux.get("predicted_latents")
                if not isinstance(predicted_latents, torch.Tensor):
                    predicted_latents = infer_output.policy_output.aux.get("predicted_latents")
                if isinstance(predicted_latents, torch.Tensor):
                    predicted_latent_chunks.append(predicted_latents.detach().cpu())

                _print_log(
                    f"chunk_{chunk_count}",
                    {
                        "phase": "infer",
                        "env_timestep_before": timestep_before,
                        "obs_window_size": len(obs_window),
                        "action_pred_shape": list(action_pred.shape),
                        "predicted_latents_shape": (
                            list(predicted_latents.shape)
                            if isinstance(predicted_latents, torch.Tensor)
                            else None
                        ),
                        **action_plan_metadata,
                    },
                )

                desired_pose_targets = None
                if action_target_representation == ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
                    desired_pose_targets = _reconstruct_chunk_pose_targets(
                        action_pred,
                        reference_obs=obs_window[-1],
                        rotation_representation=str(config.data.action_target.rotation_representation),
                    )
                elif action_target_representation != ActionTargetRepresentation.RAW:
                    raise ValueError(
                        f"Unsupported action target representation for LIBERO rollout: {action_target_representation!r}."
                    )
                control_config = LiberoControlConfig()
                key_frame_list: list[dict[str, np.ndarray]] = []
                for action_index in range(action_pred.shape[0]):
                    current_obs_record = obs_window[-1]
                    if action_target_representation == ActionTargetRepresentation.RAW:
                        control_action = _materialize_raw_control_action(action_pred[action_index])
                    else:
                        if desired_pose_targets is None:
                            raise RuntimeError("Relative-pose rollout is missing reconstructed pose targets.")
                        control_action = compute_osc_pose_action(
                            current_pose=_pose_from_obs_record(current_obs_record),
                            desired_pose=PoseSequence(
                                position=desired_pose_targets.position[action_index],
                                quaternion=desired_pose_targets.quaternion[action_index],
                                gripper=(
                                    None
                                    if desired_pose_targets.gripper is None
                                    else desired_pose_targets.gripper[action_index]
                                ),
                            ),
                            control_config=control_config,
                            gripper_representation=str(config.data.action_target.gripper_representation),
                        )
                    obs, _, done, _ = env.step(control_action.astype(np.float32))
                    current_obs = _extract_obs(obs)
                    if not done:
                        obs_window.append(current_obs)
                        if len(obs_window) > raw_window_frames:
                            obs_window = obs_window[-raw_window_frames:]
                        real_obs_list.append({key: np.array(value, copy=True) for key, value in current_obs.items()})
                        if action_index == action_pred.shape[0] - 1:
                            key_frame_list.append(current_obs)
                            keyframe_obs_list.append({key: np.array(value, copy=True) for key, value in current_obs.items()})
                    if done:
                        break

                _print_log(
                    f"chunk_{chunk_count}",
                    {
                        "phase": "env_rollout",
                        "env_timestep_after": int(env.env.timestep),
                        "done": bool(done),
                        "key_frame_count": len(key_frame_list),
                    },
                )
                next_generation_action_start += int(action_pred.shape[0])
                chunk_count += 1

        imagined_video = _decode_latent_video(
            pipeline,
            predicted_latent_chunks,
            decode_device=decode_device,
            mode=args.visualization_mode,
        )
        observed_video = _decode_latent_video(
            pipeline,
            observed_latent_chunks,
            decode_device=decode_device,
            mode=args.visualization_mode,
        )
        video_path = _write_visualization_video(
            real_obs_list=real_obs_list,
            keyframe_obs_list=keyframe_obs_list,
            observed_video=observed_video,
            imagined_video=imagined_video,
            prompt=prompt,
            task_spec=task_spec,
            episode_idx=args.episode_idx,
            success=bool(done),
            output_dir=Path(args.output_dir),
            suffix=args.suffix,
            video_fps=args.video_fps,
            mode=args.visualization_mode,
        )

        summary = {
            "benchmark": args.benchmark,
            "task_id": args.task_id,
            "prompt": prompt,
            "episode_idx": args.episode_idx,
            "success": bool(done),
            "chunk_count": chunk_count,
            "env_timestep": int(env.env.timestep),
            "seed": args.seed,
            "video_path": str(video_path),
            "pipeline": "open_wam_video_sequence",
            "checkpoint_file": str(checkpoint_path),
        }
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


def _normalize_checkpoint_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must be a raw state_dict or a checkpoint with `state_dict`/`model_state_dict`.")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        normalized_key = key[len("pipeline.") :] if key.startswith("pipeline.") else key
        normalized[normalized_key] = value
    return normalized


def _load_pipeline_checkpoint(pipeline: torch.nn.Module, checkpoint_path: Path) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _normalize_checkpoint_state_dict(checkpoint)
    missing, unexpected = pipeline.load_state_dict(state_dict, strict=False)
    _mark_loaded_lazy_components_initialized(pipeline, state_dict, missing_keys=missing)
    if missing:
        print(f"viz.checkpoint_missing_keys {len(missing)}")
    if unexpected:
        print(f"viz.checkpoint_unexpected_keys {len(unexpected)}")


def _mark_loaded_lazy_components_initialized(
    pipeline: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    missing_keys: list[str] | tuple[str, ...] = (),
) -> None:
    policy_variant = getattr(pipeline, "policy_variant", None)
    if policy_variant is None:
        return
    has_action_expert_weights = any(key.startswith("policy_variant.action_expert.") for key in state_dict)
    missing_action_expert_weights = any(key.startswith("policy_variant.action_expert.") for key in missing_keys)
    if (
        hasattr(policy_variant, "_action_expert_initialized")
        and has_action_expert_weights
        and not missing_action_expert_weights
    ):
        policy_variant._action_expert_initialized = True


def _resolve_checkpoint_file(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_file():
        return candidate
    if (candidate / "model_state.pt").is_file():
        return candidate / "model_state.pt"
    if (candidate / "full_training_state.pt").is_file():
        return candidate / "full_training_state.pt"
    if candidate.name.startswith("checkpoint_step_"):
        model_state = candidate / "model_state.pt"
        if model_state.is_file():
            return model_state
        full_state = candidate / "full_training_state.pt"
        if full_state.is_file():
            return full_state
    checkpoint_dirs = sorted(
        [child for child in candidate.glob("checkpoint_step_*") if child.is_dir()],
        key=lambda child: int(child.name.rsplit("_", 1)[-1]),
    )
    for checkpoint_dir in reversed(checkpoint_dirs):
        model_state = checkpoint_dir / "model_state.pt"
        if model_state.is_file():
            return model_state
        full_state = checkpoint_dir / "full_training_state.pt"
        if full_state.is_file():
            return full_state
    raise FileNotFoundError(f"Could not resolve model_state.pt or full_training_state.pt from {path}.")


def _resolve_checkpoint_path_from_args_or_config(
    *,
    checkpoint_arg: str | None,
    transformer_subdir: str | None,
) -> Path:
    if checkpoint_arg is not None:
        return _resolve_checkpoint_file(Path(checkpoint_arg))
    if transformer_subdir is None:
        raise ValueError("Either `--checkpoint` must be provided or `backbone.transformer_subdir` must be set.")

    transformer_dir = Path(transformer_subdir).expanduser().resolve()
    checkpoint_step_dir = _resolve_checkpoint_step_dir_from_transformer_dir(transformer_dir)
    return _resolve_checkpoint_file(checkpoint_step_dir)


def _resolve_checkpoint_step_dir_from_transformer_dir(transformer_dir: Path) -> Path:
    candidate = transformer_dir
    if candidate.name != "transformer":
        raise FileNotFoundError(
            "Expected `backbone.transformer_subdir` to point at a `.../checkpoint_step_*/transformer` directory, "
            f"got {transformer_dir}."
        )
    checkpoint_step_dir = candidate.parent
    if not checkpoint_step_dir.name.startswith("checkpoint_step_"):
        raise FileNotFoundError(
            "Unable to infer checkpoint directory from `backbone.transformer_subdir`; expected parent directory "
            f"named `checkpoint_step_*`, got {checkpoint_step_dir}."
        )
    return checkpoint_step_dir


def _resolve_task_spec(benchmark_name: str, task_id: int) -> tuple[LiberoTaskSpec, str]:
    ensure_local_libero_config(REPO_ROOT)
    from libero.libero import benchmark  # type: ignore

    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    prompt = benchmark_instance.get_task(task_id).language
    task = benchmark_instance.get_task(task_id)
    config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        libero_config = json.load(handle) if config_path.suffix == ".json" else None
    if libero_config is None:
        import yaml

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
        except Exception as exc:
            print(f"construct env failed ({count + 1}/5): {exc}")
            time.sleep(5)
            count += 1
    return env


def _init_single_env(env, init_state, *, num_frames: int) -> list[dict[str, np.ndarray]]:
    env.reset()
    env.set_init_state(init_state)
    obs_window: list[dict[str, np.ndarray]] = []
    for _ in range(max(5, num_frames)):
        obs, _, _, _ = env.step([0.0] * 7)
        extracted = _extract_obs(obs)
        obs_window.append(extracted)
    if not obs_window:
        raise RuntimeError("LIBERO env did not return an observation during initialization.")
    return obs_window[-max(1, num_frames):]


def _extract_obs(obs) -> dict[str, np.ndarray]:
    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(obs["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy(),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy(),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).copy(),
    }


def _obs_window_to_rollout_views(
    obs_window: list[dict[str, np.ndarray]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        LIBERO_OBS_KEYS[0]: torch.from_numpy(
            np.stack([obs[LIBERO_OBS_KEYS[0]] for obs in obs_window], axis=0)
        ).to(device=device),
        LIBERO_OBS_KEYS[1]: torch.from_numpy(
            np.stack([obs[LIBERO_OBS_KEYS[1]] for obs in obs_window], axis=0)
        ).to(device=device),
    }


def _prepare_rollout_inputs(
    pipeline,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
    preserve_stream_cache: bool = False,
) -> dict[str, torch.Tensor | None]:
    canonical_batch = pipeline.canonicalize(views)
    canonical_video = canonical_batch.video.to(device=frontend_device)
    frontend = pipeline.visual_tower.frontend
    assets = frontend.reference_assets
    del preserve_stream_cache

    if assets.has_vae:
        video_latents = _encode_video_window_offline(
            assets,
            canonical_video=canonical_video,
            placements=canonical_batch.placements,
            device=frontend_device,
        )
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
    else:
        frontend_output = pipeline.visual_tower.run_frontend(
            canonical_video,
            placements=canonical_batch.placements,
            task_text=task_text,
            text_context=(None if text_context is None else text_context.to(device=frontend_device)),
            negative_text_context=(
                None if negative_text_context is None else negative_text_context.to(device=frontend_device)
            ),
            preserve_stream_cache=False,
        )
        video_latents = frontend_output.video_latents
        resolved_text_context = frontend_output.conditioning.text_context
        resolved_negative_text_context = frontend_output.conditioning.negative_text_context
    return {
        "video_latents": video_latents.to(device=runtime_device),
        "text_context": (
            None
            if resolved_text_context is None
            else resolved_text_context.to(device=runtime_device)
        ),
        "negative_text_context": (
            None
            if resolved_negative_text_context is None
            else resolved_negative_text_context.to(device=runtime_device)
        ),
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
    assets._ensure_vae_runtime_device(device)

    if assets._matches_robotwin_layout(placements, canonical_video):
        top = placements[0]
        left = placements[1]
        right = placements[2]
        high_video = canonical_video[
            :,
            :,
            :,
            top.top : top.top + top.height,
            top.left : top.left + top.width,
        ]
        high_video = assets._resize_rgb_chunk(high_video, top.height, top.width)
        left_video = canonical_video[
            :,
            :,
            :,
            left.top : left.top + left.height,
            left.left : left.left + left.width,
        ]
        left_video = assets._resize_rgb_chunk(left_video, left.height, left.width)
        right_video = canonical_video[
            :,
            :,
            :,
            right.top : right.top + right.height,
            right.left : right.left + right.width,
        ]
        right_video = assets._resize_rgb_chunk(right_video, right.height, right.width)
        high_latent = _offline_encode_chunk(assets, high_video)
        wrist_latent_left = _offline_encode_chunk(assets, left_video)
        wrist_latent_right = _offline_encode_chunk(assets, right_video)
        wrist_latent = torch.cat([wrist_latent_left, wrist_latent_right], dim=-1)
        return torch.cat([high_latent, wrist_latent], dim=-2)

    if assets._matches_libero_layout(placements, canonical_video):
        agentview = placements[0]
        wrist = placements[1]
        agentview_video = canonical_video[
            :,
            :,
            :,
            agentview.top : agentview.top + agentview.height,
            agentview.left : agentview.left + agentview.width,
        ]
        agentview_video = assets._resize_rgb_chunk(agentview_video, agentview.height, agentview.width)
        wrist_video = canonical_video[
            :,
            :,
            :,
            wrist.top : wrist.top + wrist.height,
            wrist.left : wrist.left + wrist.width,
        ]
        wrist_video = assets._resize_rgb_chunk(wrist_video, wrist.height, wrist.width)
        batch_size = canonical_video.shape[0]
        encoded = _offline_encode_chunk(assets, torch.cat([agentview_video, wrist_video], dim=0))
        agentview_latent, wrist_latent = encoded.split(batch_size, dim=0)
        return torch.cat([agentview_latent, wrist_latent], dim=-1)

    return _offline_encode_chunk(assets, canonical_video)


def _offline_encode_chunk(assets, video: torch.Tensor) -> torch.Tensor:
    vae = assets.vae
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    scaled = (video.to(device=vae_device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=vae_dtype)
    with torch.no_grad():
        posterior = vae.encode(scaled, return_dict=False)[0]
    if hasattr(posterior, "mode"):
        latents = posterior.mode()
    elif hasattr(posterior, "mean"):
        latents = posterior.mean
    else:
        raise TypeError(f"Unsupported VAE encode output type: {type(posterior)!r}")
    normalized = assets._normalize_reference_latents(latents)
    return normalized.to(device=video.device)


def _pose_from_obs_record(obs_record: dict[str, np.ndarray]) -> PoseSequence:
    return PoseSequence(
        position=torch.from_numpy(np.asarray(obs_record["robot0_eef_pos"], dtype=np.float32)),
        quaternion=normalize_quaternion(
            torch.from_numpy(np.asarray(obs_record["robot0_eef_quat"], dtype=np.float32)).unsqueeze(0)
        )[0],
        gripper=torch.from_numpy(np.asarray(obs_record["robot0_gripper_qpos"], dtype=np.float32)),
    )


def _build_state_inputs_from_obs_window(
    obs_window: list[dict[str, np.ndarray]],
    *,
    state_horizon: int,
    state_encoding: str,
) -> torch.Tensor:
    if state_horizon <= 0:
        raise ValueError(f"Expected positive state_horizon, got {state_horizon}.")
    if not obs_window:
        raise ValueError("Cannot build state inputs from an empty observation window.")
    state_records = obs_window[-state_horizon:]
    if len(state_records) < state_horizon:
        state_records = [state_records[0]] * (state_horizon - len(state_records)) + state_records
    state_tensors = [_build_state_vector_from_obs_record(record, state_encoding=state_encoding) for record in state_records]
    return torch.stack(state_tensors, dim=0)


def _build_state_vector_from_obs_record(
    obs_record: dict[str, np.ndarray],
    *,
    state_encoding: str,
) -> torch.Tensor:
    position = torch.from_numpy(np.asarray(obs_record["robot0_eef_pos"], dtype=np.float32))
    quaternion = normalize_quaternion(
        torch.from_numpy(np.asarray(obs_record["robot0_eef_quat"], dtype=np.float32)).unsqueeze(0)
    )[0]
    gripper = torch.from_numpy(np.asarray(obs_record["robot0_gripper_qpos"], dtype=np.float32))

    if state_encoding == "eef_pos_axisangle_gripper_2d":
        axis_angle = quaternion_to_axis_angle(quaternion.unsqueeze(0))[0]
        return torch.cat([position, axis_angle, gripper], dim=0)
    if state_encoding == "eef_pos_quat_gripper_1d":
        return torch.cat([position, quaternion, gripper[:1]], dim=0)
    raise ValueError(f"Unsupported state encoding for visualization rollout: {state_encoding}")


def _reconstruct_chunk_pose_targets(
    action_pred: np.ndarray,
    *,
    reference_obs: dict[str, np.ndarray],
    rotation_representation: str,
) -> PoseSequence:
    relative_pose_targets = torch.from_numpy(np.asarray(action_pred, dtype=np.float32))
    reference_position = torch.from_numpy(np.asarray(reference_obs["robot0_eef_pos"], dtype=np.float32))
    reference_quaternion = normalize_quaternion(
        torch.from_numpy(np.asarray(reference_obs["robot0_eef_quat"], dtype=np.float32)).unsqueeze(0)
    )[0]
    return reconstruct_absolute_pose_targets(
        reference_position=reference_position,
        reference_quaternion=reference_quaternion,
        relative_pose_targets=relative_pose_targets,
        rotation_representation=rotation_representation,
    )


def _resolve_device(device: str | None, *, fallback: torch.device | None = None) -> torch.device:
    if device is None:
        if fallback is not None:
            return fallback
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_runtime_devices(
    raw: str | None,
    *,
    fallback: torch.device,
) -> tuple[torch.device, ...]:
    if raw is None or not raw.strip():
        return (fallback,)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return (fallback,)
    return tuple(torch.device(part) for part in parts)


def _default_raw_window_frames(latent_num_frames: int) -> int:
    if latent_num_frames <= 0:
        raise ValueError(f"Expected positive latent_num_frames, got {latent_num_frames}.")
    # Wan/LingBot frontend uses temporal compression that maps 15 raw frames -> 4 latent frames.
    # The equivalent general form is 4 * latent_num_frames - 1.
    return 4 * latent_num_frames - 1


def _initial_generation_action_start(initial_obs_window: list[dict[str, np.ndarray]]) -> int:
    return max(0, int(len(initial_obs_window)))


def _uses_zero_based_generation_start(config) -> bool:
    policy_variant = getattr(config, "policy_variant", None)
    if policy_variant is None:
        return False
    policy_name = getattr(policy_variant, "name", None)
    if str(policy_name) == "mot":
        return True
    train_source = getattr(policy_variant, "train_video_condition_source", None)
    return str(train_source) == "generated_future"


def _resolve_initial_generation_action_start(
    initial_obs_window: list[dict[str, np.ndarray]],
    *,
    initial_generation_action_start: int | None,
    rollout_starts_at_action_zero: bool = False,
) -> int:
    if initial_generation_action_start is not None:
        return max(0, int(initial_generation_action_start))
    if rollout_starts_at_action_zero:
        return 0
    return _initial_generation_action_start(initial_obs_window)


def _materialize_raw_control_action(action: np.ndarray | torch.Tensor) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def _apply_rollout_chunk_steps_override(config, rollout_chunk_steps: int | None) -> None:
    if rollout_chunk_steps is None:
        return
    decoder = getattr(config, "action_decoder", None)
    if decoder is None or not hasattr(decoder, "rollout_chunk_steps"):
        raise ValueError("Requested rollout chunk override, but the config action decoder has no rollout chunk steps.")
    object.__setattr__(decoder, "rollout_chunk_steps", int(rollout_chunk_steps))


def _decoder_output_to_rollout_action_plan(decoder_output) -> tuple[np.ndarray, dict[str, Any]]:
    action_chunk = decoder_output.action_pred[0].detach().to(dtype=torch.float32).cpu()
    current_action = decoder_output.aux.get("current_action")
    if isinstance(current_action, torch.Tensor):
        current_action_index = _resolve_decoder_current_action_index(decoder_output)
        rollout_chunk_steps = _resolve_decoder_rollout_chunk_steps(decoder_output)
        if current_action_index < 0:
            raise ValueError(f"Decoder current action index must be non-negative, got {current_action_index}.")
        commit_end_index = min(
            int(action_chunk.shape[0]),
            max(int(current_action_index) + 1, int(rollout_chunk_steps)),
        )
        planned_chunk = action_chunk[int(current_action_index) : commit_end_index]
        source = "decoder_current_action_rollout_chunk"
        if int(planned_chunk.shape[0]) == 0:
            planned_chunk = _current_action_tensor_to_chunk(current_action)
            commit_end_index = int(current_action_index) + int(planned_chunk.shape[0])
            source = "decoder_current_action"
        return planned_chunk.numpy(), {
            "action_plan_source": source,
            "decoder_rollout_chunk_steps": int(rollout_chunk_steps),
            "decoder_rollout_commit_start_index": int(current_action_index),
            "decoder_rollout_commit_end_index": int(commit_end_index),
            "decoder_rollout_committed_actions": int(planned_chunk.shape[0]),
        }
    return (
        action_chunk.numpy(),
        {
            "action_plan_source": "decoder_action_chunk",
            "decoder_rollout_chunk_steps": None,
            "decoder_rollout_commit_start_index": None,
            "decoder_rollout_commit_end_index": None,
            "decoder_rollout_committed_actions": int(action_chunk.shape[0]),
        },
    )


def _current_action_tensor_to_chunk(current_action: torch.Tensor) -> torch.Tensor:
    current_action = current_action.detach().to(dtype=torch.float32).cpu()
    if current_action.ndim == 1:
        return current_action.unsqueeze(0)
    if current_action.ndim == 2:
        return current_action[:1]
    raise ValueError(f"Expected current_action shape [D] or [B, D], got {tuple(current_action.shape)}.")


def _resolve_decoder_current_action_index(decoder_output) -> int:
    current_action_index = decoder_output.aux.get("current_action_index")
    if current_action_index is not None:
        return int(_json_scalar_from_tensor(current_action_index))
    next_state = getattr(decoder_output, "next_state", None)
    if next_state is not None and hasattr(next_state, "step_within_chunk"):
        return max(0, int(next_state.step_within_chunk) - 1)
    return 0


def _resolve_decoder_rollout_chunk_steps(decoder_output) -> int:
    rollout_chunk_steps = decoder_output.aux.get("rollout_chunk_steps")
    if rollout_chunk_steps is None:
        next_state = getattr(decoder_output, "next_state", None)
        rollout_chunk_steps = (
            getattr(next_state, "aux", {}).get("rollout_chunk_steps")
            if next_state is not None
            else None
        )
    if rollout_chunk_steps is None:
        return 1
    return max(1, int(_json_scalar_from_tensor(rollout_chunk_steps)))


def _advance_decoder_state_to_rollout_commit(session, action_plan_metadata: dict[str, Any]) -> None:
    commit_end_index = action_plan_metadata.get("decoder_rollout_commit_end_index")
    if commit_end_index is None:
        return
    policy_state = getattr(session, "policy_state", None)
    decoder_state = getattr(policy_state, "decoder_state", None)
    if decoder_state is None or not hasattr(decoder_state, "step_within_chunk"):
        return
    decoder_state.step_within_chunk = max(int(decoder_state.step_within_chunk), int(commit_end_index))


def _json_scalar_from_tensor(value) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _decode_latent_video(
    pipeline,
    latent_chunks: list[torch.Tensor],
    *,
    decode_device: torch.device,
    mode: str,
) -> np.ndarray | None:
    if mode == "env_only" or not latent_chunks:
        return None
    assets = pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None

    if mode in {"compare_keyframes", "compare_observed_vs_predicted"}:
        latents = torch.cat([chunk[:, :, -1:, :, :] for chunk in latent_chunks], dim=2)
    else:
        latents = torch.cat(latent_chunks, dim=2)
    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype

    try:
        vae.to(device=decode_device, dtype=torch.bfloat16)
        decode_dtype = next(vae.parameters()).dtype
        latents = latents.to(device=decode_device, dtype=decode_dtype)
        latents = _denormalize_reference_latents(latents, vae)
        video = vae.decode(latents, return_dict=False)[0]
        video = video_processor.postprocess_video(video, output_type="np")[0]
        return video
    finally:
        vae.to(device=original_device, dtype=original_dtype)


def _denormalize_reference_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    latents_mean = getattr(vae.config, "latents_mean", None)
    latents_std = getattr(vae.config, "latents_std", None)
    if latents_mean is not None and latents_std is not None:
        mean = torch.tensor(latents_mean, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(latents_std, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        return (latents.float() * std + mean).to(dtype=latents.dtype)
    scaling_factor = getattr(vae.config, "scaling_factor", None)
    if scaling_factor is not None:
        return latents / float(scaling_factor)
    return latents


def _write_visualization_video(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
    keyframe_obs_list: list[dict[str, np.ndarray]],
    observed_video: np.ndarray | None,
    imagined_video: np.ndarray | None,
    prompt: str,
    task_spec: LiberoTaskSpec,
    episode_idx: int,
    success: bool,
    output_dir: Path,
    suffix: str,
    video_fps: float,
    mode: str,
) -> Path:
    output_task_dir = output_dir / task_spec.benchmark_name / f"{task_spec.task_id}_{task_spec.task_name}"
    output_task_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_task_dir / f"{episode_idx}_{success}_{suffix}.mp4"

    frames: list[np.ndarray] = []
    if mode == "env_only":
        observed_frames: list[np.ndarray] = []
        imagined_frames: list[np.ndarray] = []
        reference_obs_list = real_obs_list
    elif mode in {"compare_keyframes", "compare_observed_vs_predicted"}:
        observed_frames = list(observed_video) if observed_video is not None else []
        imagined_frames = list(imagined_video) if imagined_video is not None else []
        reference_obs_list = _expand_obs_records(
            keyframe_obs_list,
            target_length=max(len(observed_frames), len(imagined_frames)),
        )
    else:
        observed_frames = []
        imagined_frames = list(imagined_video) if imagined_video is not None else []
        reference_obs_list = real_obs_list
    total_frames = max(len(reference_obs_list), len(observed_frames), len(imagined_frames))
    for index in range(total_frames):
        if index < len(reference_obs_list):
            obs = reference_obs_list[index]
            agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
            wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
            top = np.concatenate([agentview, wrist], axis=1)
        else:
            top = np.zeros((128, 256, 3), dtype=np.uint8)

        if mode == "compare_observed_vs_predicted":
            if index < len(observed_frames):
                observed_bottom = _to_uint8(observed_frames[index])
            else:
                observed_bottom = np.zeros_like(top)
            if index < len(imagined_frames):
                predicted_bottom = _to_uint8(imagined_frames[index])
            else:
                predicted_bottom = np.zeros_like(top)
            bottom = np.concatenate([observed_bottom, predicted_bottom], axis=1)
            top = np.concatenate([top, top], axis=1)
        elif index < len(imagined_frames):
            bottom = _to_uint8(imagined_frames[index])
        else:
            bottom = np.zeros_like(top)

        combined = np.concatenate([top, bottom], axis=0)
        title = f"{prompt} | frame {index}"
        if mode == "compare_observed_vs_predicted":
            title = f"{title} | bottom-left: observed_latents | bottom-right: predicted_latents"
        framed = np.array(_add_title_bar(Image.fromarray(combined), title))
        frames.append(framed)

    imageio.mimsave(video_path, frames, fps=video_fps)
    return video_path


def _add_title_bar(image: Image.Image, title: str) -> Image.Image:
    title_height = 36
    canvas = Image.new("RGB", (image.width, image.height + title_height), color=(0, 0, 0))
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))
    return canvas


def _expand_obs_records(
    obs_records: list[dict[str, np.ndarray]],
    *,
    target_length: int,
) -> list[dict[str, np.ndarray]]:
    if target_length <= 0 or not obs_records:
        return obs_records
    if len(obs_records) >= target_length:
        return obs_records[:target_length]
    expanded: list[dict[str, np.ndarray]] = []
    source_count = len(obs_records)
    for target_index in range(target_length):
        source_index = min(source_count - 1, (target_index * source_count) // target_length)
        expanded.append(obs_records[source_index])
    return expanded


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame = np.asarray(frame)
    if float(frame.max()) <= 1.0001:
        return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def _print_log(tag: str, payload: dict[str, object]) -> None:
    print(f"[{tag}] {json.dumps(payload, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
