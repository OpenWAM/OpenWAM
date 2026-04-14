from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_libero_video_sequence_visualization as video_viz  # noqa: E402

from open_wam.configs import ReferenceCoreInitMode  # noqa: E402
from open_wam.integrations import (  # noqa: E402
    LiberoTaskSpec,
    ensure_local_libero_config,
    load_libero_task_init_states,
)
from open_wam.models.policy_variants import PolicyInferContext  # noqa: E402
from open_wam.pipelines import VariantRolloutRunner, build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils.local_paths import read_yaml_with_local_paths  # noqa: E402
from open_wam.utils import load_experiment_config, seed_everywhere  # noqa: E402

LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one LIBERO rollout with a MoT policy and save a rollout video."
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/mot_libero_latent_local_idm.yaml",
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
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--raw-window-frames", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_mot_visualization")
    parser.add_argument("--suffix", type=str, default="open_wam_mot")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-rollout-video", action="store_true")
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--action-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    args = parser.parse_args()

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

    pipeline = build_variant_pipeline_from_config(config)
    video_viz._load_pipeline_checkpoint(pipeline, checkpoint_path)
    pipeline.to(device=runtime_device)
    if hasattr(pipeline.policy_variant, "_maybe_initialize_action_expert"):
        pipeline.policy_variant._maybe_initialize_action_expert(pipeline.visual_tower)
    if hasattr(pipeline.policy_variant, "action_expert"):
        pipeline.policy_variant.action_expert.to(device=action_device)
    runner = VariantRolloutRunner(pipeline)
    component_report = _build_component_report(
        config,
        pipeline,
        runtime_device=runtime_device,
        action_device=action_device,
        frontend_device=frontend_device,
        decode_device=decode_device,
        raw_window_frames=raw_window_frames,
    )
    component_report["checkpoint_file"] = str(checkpoint_path.resolve())
    _print_log("load_report", component_report)

    task_spec, prompt = _resolve_task_spec(args.benchmark, args.task_id)
    init_states = load_libero_task_init_states(task_spec)
    env = _construct_single_env(task_spec)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    try:
        initial_obs_window = _init_single_env(
            env,
            init_states[args.episode_idx % len(init_states)],
            num_frames=raw_window_frames,
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
        chunk_count = 0
        session = runner.reset(task_text=(prompt,))

        while env.env.timestep < args.max_timestep and not done:
            if args.max_chunks is not None and chunk_count >= args.max_chunks:
                break

            if args.seed is not None:
                seed_everywhere(args.seed + chunk_count)

            step_session = runner.reset(
                task_text=session.task_text,
                text_context=session.text_context,
                negative_text_context=session.negative_text_context,
            )
            with torch.inference_mode():
                views = _obs_list_to_views(list(frame_window), device=frontend_device)
                visual_outputs = _prepare_visual_outputs_offline(
                    pipeline,
                    views=views,
                    task_text=(prompt,),
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                )
                infer_output = pipeline._forward_infer_with_visual_outputs(
                    visual_outputs,
                    context=_build_infer_context(prompt, action_device=action_device),
                    infer_state=step_session.policy_state,
                )
            session = runner.reset(
                task_text=step_session.task_text,
                text_context=(
                    visual_outputs.frontend.conditioning.text_context
                    if visual_outputs.frontend.conditioning.text_context is not None
                    else step_session.text_context
                ),
                negative_text_context=(
                    visual_outputs.frontend.conditioning.negative_text_context
                    if visual_outputs.frontend.conditioning.negative_text_context is not None
                    else step_session.negative_text_context
                ),
            )
            session.policy_state = infer_output.policy_output.next_state
            actions = infer_output.decoder_output.action_pred[0].detach().to(dtype=torch.float32).cpu().numpy()
            action_trace.extend(actions)
            predicted_latents = infer_output.decoder_output.aux.get("predicted_latents")
            if not isinstance(predicted_latents, torch.Tensor):
                predicted_latents = infer_output.policy_output.aux.get("predicted_latents")
            if isinstance(predicted_latents, torch.Tensor):
                predicted_latent_chunks.append(predicted_latents.detach().cpu())

            chunk_log = {
                "chunk_index": chunk_count,
                "phase": "infer",
                "env_timestep_before": int(env.env.timestep),
                "window_size": len(frame_window),
                "action_shape": list(actions.shape),
                "predicted_latents_shape": None if not isinstance(predicted_latents, torch.Tensor) else list(predicted_latents.shape),
                "first_action_preview": [float(v) for v in actions[0].tolist()],
                "policy_debug": infer_output.policy_output.aux,
            }
            _print_log(f"chunk_{chunk_count}", chunk_log)
            chunk_logs.append(chunk_log)

            real_future_frames: list[dict[str, np.ndarray]] = []
            future_frame_count = _future_frame_count(config)
            sample_indices = _future_sample_indices(
                action_count=actions.shape[0],
                future_frame_count=future_frame_count,
            )
            executed_actions = 0
            for action_index, action in enumerate(actions):
                obs, _, done, _ = env.step(action.astype(np.float32))
                executed_actions += 1
                extracted = _extract_obs(obs)
                rollout_frames.append({key: np.array(value, copy=True) for key, value in extracted.items()})
                frame_window.append({key: np.array(value, copy=True) for key, value in extracted.items()})
                if action_index in sample_indices:
                    real_future_frames.append({key: np.array(value, copy=True) for key, value in extracted.items()})
                if done or env.env.timestep >= args.max_timestep:
                    break

            chunk_result_log = {
                "chunk_index": chunk_count,
                "phase": "env_rollout",
                "env_timestep_after": int(env.env.timestep),
                "executed_actions": int(executed_actions),
                "done_after_chunk": bool(done),
                "success_after_chunk": bool(done),
            }
            _print_log(f"chunk_{chunk_count}", chunk_result_log)
            chunk_logs.append(chunk_result_log)

            chunk_count += 1

        imagined_video = _decode_latent_video_chunks(
            pipeline,
            predicted_latent_chunks,
            decode_device=decode_device,
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
        comparison_frames = _build_comparison_video_frames(
            real_obs_list=rollout_frames,
            imagined_video=imagined_video,
        )
        imageio.mimsave(output_path, comparison_frames, fps=args.video_fps)
        rollout_path = None
        if args.save_rollout_video:
            rollout_path = output_path.with_name(f"{output_path.stem}_rollout.mp4")
            rollout_video_frames = _build_rollout_video_frames(real_obs_list=rollout_frames)
            imageio.mimsave(rollout_path, rollout_video_frames, fps=args.video_fps)

        summary = {
            "benchmark": args.benchmark,
            "task_id": args.task_id,
            "prompt": prompt,
            "episode_idx": args.episode_idx,
            "success": bool(done),
            "chunk_count": chunk_count,
            "env_timestep": int(env.env.timestep),
            "seed": args.seed,
            "video_path": str(output_path.resolve()),
            "comparison_video_path": str(output_path.resolve()),
            "rollout_video_path": None if rollout_path is None else str(rollout_path.resolve()),
            "pipeline": "open_wam_mot",
            "runtime_mode": str(config.policy_variant.runtime_mode),
            "condition_mode": str(config.policy_variant.condition_mode),
            "action_count": len(action_trace),
            "checkpoint_file": str(checkpoint_path.resolve()),
        }
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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


def _resolve_mot_checkpoint_path(
    *,
    config_path: Path,
    checkpoint_arg: str | None,
    transformer_subdir: str | None,
) -> Path | None:
    if checkpoint_arg is not None:
        return video_viz._resolve_checkpoint_file(Path(checkpoint_arg))
    raw = read_yaml_with_local_paths(config_path)
    raw_checkpoint = raw.get("checkpoint_path")
    if raw_checkpoint is not None:
        return video_viz._resolve_checkpoint_file(Path(str(raw_checkpoint)))
    if transformer_subdir is None:
        return None
    try:
        return video_viz._resolve_checkpoint_path_from_args_or_config(
            checkpoint_arg=None,
            transformer_subdir=transformer_subdir,
        )
    except (FileNotFoundError, ValueError):
        return None


def _build_infer_context(prompt: str, *, action_device: torch.device):
    return PolicyInferContext(extra={"task_text": (prompt,), "action_device": str(action_device)})


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


def _init_single_env(env, init_state, *, num_frames: int) -> list[dict[str, np.ndarray]]:
    env.reset()
    env.set_init_state(init_state)
    if num_frames <= 0:
        raise ValueError(f"Expected positive num_frames, got {num_frames}.")
    obs_window: list[dict[str, np.ndarray]] = []
    for _ in range(max(5, num_frames)):
        obs, _, _, _ = env.step([0.0] * 7)
        obs_window.append(_extract_obs(obs))
    if not obs_window:
        raise RuntimeError("LIBERO env did not return an observation during initialization.")
    return obs_window[-num_frames:]


def _extract_obs(obs) -> dict[str, np.ndarray]:
    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(obs["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
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


def _future_frame_count(config) -> int:
    return max(1, int(config.data.num_frames) - int(config.policy_variant.video_prefix_frames))


def _future_sample_indices(*, action_count: int, future_frame_count: int) -> list[int]:
    if action_count <= 0:
        return []
    if future_frame_count <= 1:
        return [action_count - 1]
    raw = np.linspace(0, action_count - 1, num=future_frame_count)
    indices = [int(round(value)) for value in raw.tolist()]
    deduped: list[int] = []
    for index in indices:
        clamped = max(0, min(action_count - 1, index))
        if clamped not in deduped:
            deduped.append(clamped)
    return deduped


def _build_rollout_video_frames(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
) -> list[np.ndarray]:
    final_frames: list[np.ndarray] = []
    for obs in real_obs_list:
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        row_real = np.ascontiguousarray(row_real)
        row_real = np.array(_with_title(Image.fromarray(row_real), "MoT Rollout (AgentView / Wrist)"), copy=True)
        final_frames.append(np.ascontiguousarray(row_real))
    return final_frames


def _build_comparison_video_frames(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
    imagined_video: np.ndarray | None,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    panel_height = 300
    aligned_imagined_frames = _align_imagined_video_to_rollout(
        imagined_video=imagined_video,
        target_length=len(real_obs_list),
    )
    total = max(len(real_obs_list), len(aligned_imagined_frames))
    for frame_index in range(total):
        real_obs = None if frame_index >= len(real_obs_list) else real_obs_list[frame_index]
        imagined_frame = None if frame_index >= len(aligned_imagined_frames) else aligned_imagined_frames[frame_index]
        if real_obs is None:
            target_width = 256
            row_real = np.array(
                _with_title(Image.new("RGB", (target_width, panel_height), color=(0, 0, 0)), "Missing Rollout Frame"),
                copy=True,
            )
        else:
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
        frames.append(np.ascontiguousarray(np.vstack([row_real, np.array(row_imagined, copy=True)])))
    return frames


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
) -> np.ndarray | None:
    if not latent_chunks:
        return None
    return _decode_latent_video(
        pipeline,
        torch.cat(latent_chunks, dim=2),
        decode_device=decode_device,
    )


def _decode_latent_video(
    pipeline,
    latents: torch.Tensor,
    *,
    decode_device: torch.device,
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
    if next(assets.vae.parameters()).device != original_device or next(assets.vae.parameters()).dtype != original_dtype:
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
    print(f"[{label}] {json.dumps(payload, sort_keys=True, default=str)}")


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _resolve_device(device_arg: str | None, *, fallback: torch.device | None = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_raw_window_frames(latent_num_frames: int) -> int:
    if latent_num_frames <= 0:
        raise ValueError(f"Expected positive latent_num_frames, got {latent_num_frames}.")
    return 4 * latent_num_frames - 1


if __name__ == "__main__":
    main()
