from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers.video_processor import VideoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_libero_video_sequence_visualization as video_viz  # noqa: E402

from open_wam.configs import ActionTargetRepresentation, ReferenceCoreInitMode  # noqa: E402
from open_wam.data import reconstruct_absolute_pose_targets  # noqa: E402
from open_wam.integrations import (  # noqa: E402
    LiberoTaskSpec,
    LiberoControlConfig,
    compute_osc_pose_action,
    ensure_local_libero_config,
    load_libero_task_init_states,
)
from open_wam.data.action_transforms import PoseSequence, normalize_quaternion, quaternion_to_axis_angle  # noqa: E402
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils import load_experiment_config, seed_everywhere  # noqa: E402

LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one register-attached LIBERO rollout with Open-WAM and save a comparison video."
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/register_attached_libero_latent_local.yaml",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint root or checkpoint_step_* directory.")
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_register_attached_visualization")
    parser.add_argument("--suffix", type=str, default="open_wam_method2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--runtime-devices", type=str, default=None)
    parser.add_argument("--runtime-prep-device", type=str, default=None)
    parser.add_argument("--runtime-output-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument("--video-steps", type=int, default=None)
    parser.add_argument("--action-steps", type=int, default=None)
    parser.add_argument("--joint-steps", type=int, default=None)
    parser.add_argument(
        "--visualization-mode",
        type=str,
        default="compare_keyframes",
        choices=("compare_keyframes", "compare_observed_vs_predicted", "env_only", "latent_grid"),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config = load_experiment_config(config_path)
    if args.video_steps is not None:
        object.__setattr__(config.inference, "video_num_inference_steps", int(args.video_steps))
    if args.action_steps is not None:
        object.__setattr__(config.inference, "action_num_inference_steps", int(args.action_steps))
    if args.joint_steps is not None:
        object.__setattr__(config.inference, "joint_num_inference_steps", int(args.joint_steps))
    action_target_representation = ActionTargetRepresentation(config.data.action_target.representation)

    checkpoint_step_dir = _resolve_checkpoint_step_dir(Path(args.checkpoint))
    checkpoint_path = video_viz._resolve_checkpoint_file(checkpoint_step_dir)
    object.__setattr__(config.backbone, "transformer_subdir", str((checkpoint_step_dir / "transformer").resolve()))
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
    pipeline = build_variant_pipeline_from_config(config)
    video_viz._load_pipeline_checkpoint(pipeline, checkpoint_path)
    pipeline = pipeline.to(runtime_device)
    pipeline.visual_tower.configure_runtime_devices(
        runtime_devices,
        prep_device=runtime_prep_device,
        output_device=runtime_output_device,
    )
    pipeline.eval()
    runner = VariantRolloutRunner(pipeline)

    load_report = {
        "pipeline": "open_wam",
        "policy_variant": config.policy_variant.name,
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_step_dir": str(checkpoint_step_dir),
        "transformer_dir": str((checkpoint_step_dir / "transformer").resolve()),
        "runtime_device": str(runtime_device),
        "runtime_prep_device": str(runtime_prep_device),
        "runtime_output_device": str(runtime_output_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "runtime_devices": [str(device) for device in runtime_devices],
        "video_steps": int(config.inference.video_num_inference_steps),
        "action_steps": int(config.inference.action_num_inference_steps),
        "joint_steps": int(config.inference.joint_num_inference_steps),
        "joint_observed_video_prefix_frames": int(config.inference.joint_observed_video_prefix_frames),
        "data_num_frames": int(config.data.num_frames),
        "action_horizon": int(config.data.action_schema.action_horizon),
    }
    _print_log("load_report", load_report)

    task_spec, prompt = _resolve_task_spec(args.benchmark, args.task_id)
    init_states = load_libero_task_init_states(task_spec)
    env = _construct_single_env(task_spec)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    try:
        with torch.inference_mode():
            initial_obs_window = _init_single_env(
                env,
                init_states[args.episode_idx % len(init_states)],
                num_frames=int(config.data.num_frames),
            )
            initial_rollout_obs = [initial_obs_window[-1]]
            initial_inputs = _prepare_register_rollout_inputs(
                pipeline,
                views=_obs_window_to_rollout_views(initial_rollout_obs, device=frontend_device),
                task_text=(prompt,),
                frontend_device=frontend_device,
                runtime_device=runtime_device,
            )
            session = runner.reset(
                task_text=(prompt,),
                text_context=initial_inputs["text_context"],
                negative_text_context=initial_inputs["negative_text_context"],
            )

            predicted_latent_chunks: list[torch.Tensor] = []
            observed_latent_chunks: list[torch.Tensor] = []
            obs_window = list(initial_obs_window)
            pending_chunk_frames: list[dict[str, np.ndarray]] = list(initial_obs_window[1:])
            real_obs_list: list[dict[str, np.ndarray]] = [
                {key: np.array(value, copy=True) for key, value in obs.items()}
                for obs in initial_obs_window
            ]
            keyframe_obs_list: list[dict[str, np.ndarray]] = []
            done = False
            chunk_count = 0
            first_chunk = True

            while env.env.timestep < args.max_timestep and not done:
                if args.max_chunks is not None and chunk_count >= args.max_chunks:
                    break

                if args.seed is not None:
                    seed_everywhere(args.seed + chunk_count)

                if (
                    session.policy_state is not None
                    and int(session.policy_state.cursor.current_start_frame) >= int(config.data.num_frames)
                ):
                    _print_log(
                        "rollout_reset",
                        {
                            "reason": "current_start_frame_reached_window_limit",
                            "current_start_frame": int(session.policy_state.cursor.current_start_frame),
                            "window_limit": int(config.data.num_frames),
                            "chunk_index": chunk_count,
                        },
                    )
                    session = runner.reset(
                        task_text=(prompt,),
                        text_context=session.text_context,
                        negative_text_context=session.negative_text_context,
                    )

                timestep_before = int(env.env.timestep)
                if first_chunk:
                    rollout_obs = [obs_window[-1]]
                else:
                    rollout_obs = _take_or_pad_chunk_frames(
                        pending_chunk_frames,
                        chunk_size=int(config.data.num_frames),
                    )
                rollout_inputs = _prepare_register_rollout_inputs(
                    pipeline,
                    views=_obs_window_to_rollout_views(rollout_obs, device=frontend_device),
                    task_text=(prompt,),
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                    text_context=session.text_context,
                    negative_text_context=session.negative_text_context,
                    preserve_stream_cache=not first_chunk,
                )
                observed_latent_chunks.append(rollout_inputs["video_latents"].detach().cpu())
                step_output = runner.infer_step(
                    session=session,
                    context=PolicyInferContext(
                        state=_build_state_inputs_from_obs_window(
                            obs_window,
                            state_horizon=int(config.data.action_schema.state_horizon),
                            state_encoding=str(config.data.action_target.state_encoding),
                        ).unsqueeze(0),
                        extra={"task_text": (prompt,)},
                    ),
                    video_latents=rollout_inputs["video_latents"],
                )
                session = step_output.session
                infer_output = step_output.infer_output

                action_pred = infer_output.decoder_output.action_pred[0].detach().to(dtype=torch.float32).cpu().numpy()
                predicted_latents = infer_output.policy_output.aux.get("predicted_latents")
                if isinstance(predicted_latents, torch.Tensor):
                    predicted_latent_chunks.append(predicted_latents.detach().cpu())

                _print_log(
                    f"chunk_{chunk_count}",
                    {
                        "phase": "infer",
                        "env_timestep_before": timestep_before,
                        "obs_window_size": len(obs_window),
                        "rollout_chunk_frames": len(rollout_obs),
                        "first_chunk": first_chunk,
                        "action_pred_shape": list(infer_output.decoder_output.action_pred.shape),
                        "action_preview": action_pred[: min(3, action_pred.shape[0]), : min(7, action_pred.shape[1])].tolist(),
                        "predicted_latents_shape": (
                            list(predicted_latents.shape)
                            if isinstance(predicted_latents, torch.Tensor)
                            else None
                        ),
                        "session_step_index_after": int(session.policy_state.step_index) if session.policy_state is not None else None,
                    },
                )

                desired_pose_targets = None
                if action_target_representation == ActionTargetRepresentation.EEF_POSE_RELATIVE_TO_REFERENCE:
                    desired_pose_targets = _reconstruct_chunk_pose_targets(
                        action_pred,
                        reference_obs=rollout_obs[0],
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
                        control_action = np.asarray(action_pred[action_index], dtype=np.float32)
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
                    if done:
                        break
                    current_obs = _extract_obs(obs)
                    obs_window.append(current_obs)
                    if len(obs_window) > int(config.data.num_frames):
                        obs_window = obs_window[-int(config.data.num_frames):]
                    pending_chunk_frames.append(current_obs)
                    real_obs_list.append({key: np.array(value, copy=True) for key, value in current_obs.items()})
                    if action_index == action_pred.shape[0] - 1:
                        key_frame_list.append(current_obs)
                        keyframe_obs_list.append({key: np.array(value, copy=True) for key, value in current_obs.items()})

                _print_log(
                    f"chunk_{chunk_count}",
                    {
                        "phase": "env_rollout",
                        "env_timestep_after": int(env.env.timestep),
                        "done": bool(done),
                        "key_frame_count": len(key_frame_list),
                    },
                )
                chunk_count += 1
                first_chunk = False

        imagined_video = _decode_latent_video(
            pipeline,
            predicted_latent_chunks,
            decode_device=decode_device,
            observed_prefix_frames=int(config.inference.joint_observed_video_prefix_frames),
            mode=args.visualization_mode,
        )
        observed_video = _decode_latent_video(
            pipeline,
            observed_latent_chunks,
            decode_device=decode_device,
            observed_prefix_frames=int(config.inference.joint_observed_video_prefix_frames),
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
            "pipeline": "open_wam_register_attached",
            "checkpoint_step_dir": str(checkpoint_step_dir),
        }
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


def _resolve_checkpoint_step_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    if candidate.name.startswith("checkpoint_step_"):
        return candidate
    checkpoint_dirs = sorted(
        [child for child in candidate.glob("checkpoint_step_*") if child.is_dir()],
        key=lambda child: int(child.name.rsplit("_", 1)[-1]),
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1]
    raise FileNotFoundError(f"Could not resolve checkpoint_step_* directory from {path}.")


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
    for _ in range(5):
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


def _prepare_register_rollout_inputs(
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
    frontend_output = pipeline.visual_tower.run_frontend(
        canonical_video,
        placements=canonical_batch.placements,
        task_text=task_text,
        text_context=(
            None
            if text_context is None
            else text_context.to(device=frontend_device)
        ),
        negative_text_context=(
            None
            if negative_text_context is None
            else negative_text_context.to(device=frontend_device)
        ),
        preserve_stream_cache=preserve_stream_cache,
    )
    return {
        "video_latents": frontend_output.video_latents.to(device=runtime_device),
        "text_context": (
            None
            if frontend_output.conditioning.text_context is None
            else frontend_output.conditioning.text_context.to(device=runtime_device)
        ),
        "negative_text_context": (
            None
            if frontend_output.conditioning.negative_text_context is None
            else frontend_output.conditioning.negative_text_context.to(device=runtime_device)
        ),
    }


def _take_or_pad_chunk_frames(
    frame_buffer: list[dict[str, np.ndarray]],
    *,
    chunk_size: int,
) -> list[dict[str, np.ndarray]]:
    if chunk_size <= 0:
        raise ValueError(f"Expected positive chunk_size, got {chunk_size}.")
    if not frame_buffer:
        raise ValueError("Cannot build DreamZero-style rollout chunk from an empty frame buffer.")
    available = frame_buffer[:chunk_size]
    del frame_buffer[: min(len(frame_buffer), chunk_size)]
    if len(available) >= chunk_size:
        return available
    pad_frame = available[0]
    return [pad_frame] * (chunk_size - len(available)) + available


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


def _decode_latent_video(
    pipeline,
    latent_chunks: list[torch.Tensor],
    *,
    decode_device: torch.device,
    observed_prefix_frames: int,
    mode: str,
) -> np.ndarray | None:
    if mode == "env_only" or not latent_chunks:
        return None
    assets = pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None

    if mode in {"compare_keyframes", "compare_observed_vs_predicted"}:
        selected_chunks: list[torch.Tensor] = []
        for chunk in latent_chunks:
            if mode == "compare_keyframes":
                if chunk.shape[2] <= observed_prefix_frames:
                    continue
                selected_chunks.append(chunk[:, :, -1:, :, :])
                continue
            selected_chunks.append(chunk[:, :, -1:, :, :])
        if not selected_chunks:
            return None
        latents = torch.cat(selected_chunks, dim=2)
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
        reference_obs_list = keyframe_obs_list
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
