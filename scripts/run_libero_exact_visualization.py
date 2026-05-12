from __future__ import annotations

import argparse
import hashlib
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
from einops import rearrange
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.integrations import (  # noqa: E402
    LiberoTaskSpec,
    ensure_local_libero_config,
    load_libero_task_init_states,
)
from open_wam.evals.evaluate import EvaluationRequest, resolve_evaluation_request  # noqa: E402
from open_wam.models.visual_tower.reference_loader import resolve_pretrained_component_dir  # noqa: E402
from open_wam.pipelines import build_exact_runtime_runner_from_config  # noqa: E402
from open_wam.utils import (  # noqa: E402
    load_experiment_config,
    resolve_transformer_dir_override,
    seed_everywhere,
)

LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Heng-style LIBERO exact rollout with Open-WAM and save a comparison video."
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/parallel_stream_libero_lingbot_exact_local.yaml",
    )
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=800)
    parser.add_argument(
        "--env-horizon",
        type=int,
        default=None,
        help=(
            "Optional LIBERO/robosuite internal episode horizon. Use this with large "
            "--max-timestep values so failed rollouts can write summaries instead of "
            "terminating inside the simulator."
        ),
    )
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default="outputs/libero_exact_visualization")
    parser.add_argument("--suffix", type=str, default="open_wam")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runtime-device", type=str, default=None)
    parser.add_argument("--frontend-device", type=str, default=None)
    parser.add_argument("--decode-device", type=str, default=None)
    parser.add_argument(
        "--transformer-dir",
        type=str,
        default=None,
        help=(
            "Optional exported transformer override. If omitted, exact visualization keeps "
            "`backbone.transformer_subdir` from the experiment config even when `--cfg` points "
            "at an eval wrapper."
        ),
    )
    parser.add_argument(
        "--exact-startup-bootstrap-padding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Exact-runtime startup parity mode. The default is the legacy single-frame first-chunk startup. "
            "Pass --exact-startup-bootstrap-padding to encode the first observation once, repeat that latent "
            "to a full startup chunk, warm negative frames, and execute from generated frame 1."
        ),
    )
    args = parser.parse_args()

    request = _resolve_visualization_request(args.config)
    config = load_experiment_config(request.experiment_config_path)
    effective_transformer_subdir = _resolve_visualization_transformer_subdir(
        config=config,
        transformer_dir_arg=args.transformer_dir,
    )
    if effective_transformer_subdir != str(config.backbone.transformer_subdir):
        object.__setattr__(
            config.backbone,
            "transformer_subdir",
            effective_transformer_subdir,
        )
    startup_checkpoint_path = _resolve_visualization_startup_checkpoint_path(
        request=request,
        effective_transformer_subdir=effective_transformer_subdir,
    )
    exact_startup_bootstrap_padding = _resolve_exact_startup_bootstrap_padding(
        config,
        cli_value=args.exact_startup_bootstrap_padding,
        checkpoint_path=startup_checkpoint_path,
    )
    runner = build_exact_runtime_runner_from_config(config)
    runtime_device = _resolve_device(args.runtime_device)
    frontend_device = _resolve_device(args.frontend_device, fallback=runtime_device)
    decode_device = _resolve_device(args.decode_device, fallback=frontend_device)
    component_report = _build_open_wam_component_report(
        config,
        runner,
        runtime_device=runtime_device,
        frontend_device=frontend_device,
        decode_device=decode_device,
        requested_eval_checkpoint=request.checkpoint_path,
    )
    _print_log("load_report", component_report)

    task_spec, prompt = _resolve_task_spec(args.benchmark, args.task_id)
    init_states = load_libero_task_init_states(task_spec)
    env = _construct_single_env(task_spec, env_horizon=args.env_horizon)
    if env is None:
        raise RuntimeError("Failed to construct LIBERO OffScreenRenderEnv after 5 retries.")

    try:
        first_obs = _init_single_env(env, init_states[args.episode_idx % len(init_states)])
        session = runner.reset(task_text=(prompt,))

        predicted_latent_chunks: list[torch.Tensor] = []
        real_obs_list: list[dict[str, np.ndarray]] = [{key: np.array(value, copy=True) for key, value in first_obs.items()}]
        done = False
        first_chunk = True
        chunk_count = 0

        while env.env.timestep < args.max_timestep and not done:
            if args.max_chunks is not None and chunk_count >= args.max_chunks:
                break

            if args.seed is not None:
                seed_everywhere(args.seed + chunk_count)
            timestep_before = int(env.env.timestep)
            if first_chunk:
                first_chunk_inputs = _prepare_exact_runtime_inputs(
                    runner,
                    views=_obs_list_to_views([first_obs], config=config, device=frontend_device),
                    task_text=(prompt,),
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                )
                if exact_startup_bootstrap_padding:
                    first_chunk_inputs = _repeat_exact_startup_bootstrap_latents(
                        first_chunk_inputs,
                        frame_chunk_size=int(config.inference.frame_chunk_size),
                    )
                    startup_action_history = _exact_startup_bootstrap_action_history(
                        frame_chunk_size=int(config.inference.frame_chunk_size),
                        action_per_frame=int(config.policy_variant.action_per_frame),
                        action_dim=_exact_startup_bootstrap_raw_action_dim(config),
                        device=runtime_device,
                    )
                    warmup = runner.warmup_cache(
                        session=session,
                        video_latents=first_chunk_inputs["video_latents"],
                        text_context=first_chunk_inputs["text_context"],
                        negative_text_context=first_chunk_inputs["negative_text_context"],
                        action_history=startup_action_history,
                        action_space="raw",
                        frame_start_override=_exact_startup_bootstrap_frame_start(
                            int(config.inference.frame_chunk_size)
                        ),
                    )
                    chunk = runner.infer_chunk(session=warmup.session)
                else:
                    chunk = runner.infer_chunk(
                        session=session,
                        video_latents=first_chunk_inputs["video_latents"],
                        text_context=first_chunk_inputs["text_context"],
                        negative_text_context=first_chunk_inputs["negative_text_context"],
                    )
            else:
                chunk = runner.infer_chunk(session=session)

            action_adapter = runner.policy_variant.exact_action_adapter
            adapter_spec = getattr(action_adapter, "spec", None)
            raw_chunk_action_pred = chunk.raw_chunk_action_pred
            if raw_chunk_action_pred is None:
                if chunk.chunk_action_pred.shape[-1] != 7:
                    raise RuntimeError(
                        "Exact runner did not produce raw 7D LIBERO actions and model action dim is not 7: "
                        f"chunk_action_pred_shape={tuple(chunk.chunk_action_pred.shape)}."
                    )
                raw_chunk_action_pred = chunk.chunk_action_pred

            predicted_latent_chunks.append(chunk.predicted_latents.detach().cpu())
            session = chunk.session

            raw_actions = rearrange(
                raw_chunk_action_pred[0],
                "(f a) c -> f a c",
                f=config.inference.frame_chunk_size,
                a=config.policy_variant.action_per_frame,
            )
            model_actions = rearrange(
                chunk.chunk_action_pred[0],
                "(f a) c -> f a c",
                f=config.inference.frame_chunk_size,
                a=config.policy_variant.action_per_frame,
            )
            raw_actions_batched = raw_actions.unsqueeze(0)
            if adapter_spec is None:
                model_actions_from_raw = raw_chunk_action_pred.to(
                    device=chunk.chunk_action_pred.device,
                    dtype=chunk.chunk_action_pred.dtype,
                )
            else:
                model_actions_from_raw = action_adapter.to_model_action_sequence(
                    raw_actions_batched,
                    action_space="raw",
                    device=chunk.chunk_action_pred.device,
                    dtype=chunk.chunk_action_pred.dtype,
                )
            obs_stride = max(1, raw_actions.shape[1] // max(1, config.inference.frame_chunk_size))
            _print_log(
                f"chunk_{chunk_count}",
                {
                    "phase": "infer",
                    "first_chunk": first_chunk,
                    "env_timestep_before": timestep_before,
                    "session_step_index_before": int(session.policy_state.step_index),
                    "session_frame_start_before": int(session.policy_state.cache.get("frame_start", -1)),
                    "condition_latents_shape": (
                        list(chunk.visual_outputs.frontend.video_latents.shape) if chunk.visual_outputs is not None else None
                    ),
                    "text_context_shape": (
                        list(chunk.visual_outputs.frontend.conditioning.text_context.shape)
                        if chunk.visual_outputs is not None and chunk.visual_outputs.frontend.conditioning.text_context is not None
                        else None
                    ),
                    "negative_text_context_shape": (
                        list(chunk.visual_outputs.frontend.conditioning.negative_text_context.shape)
                        if chunk.visual_outputs is not None
                        and chunk.visual_outputs.frontend.conditioning.negative_text_context is not None
                        else None
                    ),
                    "predicted_latents_shape": list(chunk.predicted_latents.shape),
                    "predicted_latents_mean": float(chunk.predicted_latents.float().mean().item()),
                    "predicted_latents_std": float(chunk.predicted_latents.float().std().item()),
                    "raw_actions_shape": list(raw_actions.shape),
                    "model_actions_shape": list(model_actions.shape),
                    "model_actions_from_raw_shape": list(model_actions_from_raw.shape),
                    "model_from_raw_max_abs_diff": float(
                        (chunk.chunk_action_pred - model_actions_from_raw).abs().max().item()
                    ),
                    "model_from_raw_mean_abs_diff": float(
                        (chunk.chunk_action_pred - model_actions_from_raw).abs().mean().item()
                    ),
                    "raw_action_preview": _preview_tensor(raw_actions[0, 0]),
                    "model_action_preview": _preview_tensor(model_actions[0, 0]),
                    "obs_stride": int(obs_stride),
                    "policy_debug": chunk.debug,
                },
            )

            key_frame_list: list[dict[str, np.ndarray]] = []
            start_frame_group = 0 if (first_chunk and exact_startup_bootstrap_padding) else (1 if first_chunk else 0)
            for frame_group in range(start_frame_group, raw_actions.shape[0]):
                for action_index in range(raw_actions.shape[1]):
                    action_step = raw_actions[frame_group, action_index].detach().to(dtype=torch.float32).cpu().numpy()
                    obs, _, done, _ = env.step(action_step.astype(np.float32))
                    if done:
                        break
                    if (action_index + 1) % obs_stride == 0:
                        extracted = _extract_obs(obs)
                        real_obs_list.append({key: np.array(value, copy=True) for key, value in extracted.items()})
                        key_frame_list.append(extracted)
                if done:
                    break

            chunk_count += 1
            _print_log(
                f"chunk_{chunk_count - 1}",
                {
                    "phase": "env_rollout",
                    "env_timestep_after": int(env.env.timestep),
                    "done": bool(done),
                    "key_frame_count": len(key_frame_list),
                    "start_frame_group": start_frame_group,
                },
            )
            if done:
                break
            if args.max_chunks is not None and chunk_count >= args.max_chunks:
                break
            if not key_frame_list:
                break

            if first_chunk:
                new_visual_outputs = _prepare_exact_runtime_inputs(
                    runner,
                    views=_obs_list_to_views(key_frame_list, config=config, device=frontend_device),
                    task_text=(prompt,),
                    text_context=session.text_context,
                    negative_text_context=session.negative_text_context,
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                    preserve_stream_cache=True,
                )
                if exact_startup_bootstrap_padding:
                    initial_latents = first_chunk_inputs["video_latents"][:, :, -1:]
                    combined_latents = new_visual_outputs["video_latents"]
                else:
                    initial_latents = chunk.visual_outputs.frontend.video_latents
                    combined_latents = torch.cat([initial_latents, new_visual_outputs["video_latents"]], dim=2)
                _print_log(
                    f"chunk_{chunk_count - 1}",
                    {
                        "phase": "warmup_prepare",
                        "initial_latents_shape": list(initial_latents.shape),
                        "new_latents_shape": list(new_visual_outputs["video_latents"].shape),
                        "combined_latents_shape": list(combined_latents.shape),
                        "raw_actions_shape": list(raw_actions_batched.shape),
                    },
                )
                warmup = runner.warmup_cache(
                    session=session,
                    video_latents=combined_latents,
                    text_context=new_visual_outputs["text_context"],
                    negative_text_context=new_visual_outputs["negative_text_context"],
                    action_history=raw_actions_batched,
                    action_space="raw",
                )
            else:
                warmup_inputs = _prepare_exact_runtime_inputs(
                    runner,
                    views=_obs_list_to_views(key_frame_list, config=config, device=frontend_device),
                    task_text=(prompt,),
                    text_context=session.text_context,
                    negative_text_context=session.negative_text_context,
                    frontend_device=frontend_device,
                    runtime_device=runtime_device,
                    preserve_stream_cache=True,
                )
                warmup = runner.warmup_cache(
                    session=session,
                    video_latents=warmup_inputs["video_latents"],
                    text_context=warmup_inputs["text_context"],
                    negative_text_context=warmup_inputs["negative_text_context"],
                    action_history=raw_actions_batched,
                    action_space="raw",
                )
            _print_log(
                f"chunk_{chunk_count - 1}",
                {
                    "phase": "warmup_done",
                    "warmup_debug": warmup.debug,
                    "session_step_index": int(warmup.session.policy_state.step_index),
                    "session_frame_start": int(warmup.session.policy_state.cache.get("frame_start", -1)),
                },
            )
            session = warmup.session
            first_chunk = False

        imagined_video = _decode_imagined_video(
            runner,
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
        video_frames = _build_comparison_video_frames(
            real_obs_list=real_obs_list,
            imagined_video=imagined_video,
        )
        imageio.mimsave(output_path, video_frames, fps=args.video_fps)

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
            "pipeline": "open_wam",
            "exact_startup_bootstrap_padding": bool(exact_startup_bootstrap_padding),
        }
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        load_report_path = output_path.with_name(f"{output_path.stem}_load_report.json")
        load_report_path.write_text(json.dumps(component_report, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))
    finally:
        env.close()


def _resolve_visualization_request(config_arg: str) -> EvaluationRequest:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    return resolve_evaluation_request(config_path)


def _resolve_visualization_transformer_subdir(*, config, transformer_dir_arg: str | None) -> str:
    if transformer_dir_arg is None:
        return str(config.backbone.transformer_subdir)
    return str(resolve_transformer_dir_override(transformer_dir_arg))


def _resolve_visualization_startup_checkpoint_path(
    *,
    request: EvaluationRequest,
    effective_transformer_subdir: str,
) -> Path | None:
    if request.checkpoint_path is not None:
        return Path(request.checkpoint_path)
    transformer_dir = Path(effective_transformer_subdir)
    if transformer_dir.name == "transformer":
        return transformer_dir.parent
    return None


def _resolve_exact_startup_bootstrap_padding(
    config,
    *,
    cli_value: bool | None,
    checkpoint_path: Path | None,
) -> bool:
    del config, checkpoint_path
    if cli_value is not None:
        return bool(cli_value)
    return False


def _exact_startup_bootstrap_raw_action_dim(config) -> int:
    action_schema = getattr(getattr(config, "data", None), "action_schema", None)
    action_dim = int(getattr(action_schema, "action_dim", 0) or 0)
    if action_dim <= 0:
        raise ValueError("Exact startup bootstrap requires positive data.action_schema.action_dim for raw actions.")
    return action_dim


def _repeat_exact_startup_bootstrap_latents(
    initial_inputs: dict[str, torch.Tensor | None],
    *,
    frame_chunk_size: int,
) -> dict[str, torch.Tensor | None]:
    frame_chunk_size = int(frame_chunk_size)
    if frame_chunk_size <= 0:
        raise ValueError(f"Expected positive frame_chunk_size, got {frame_chunk_size}.")
    video_latents = initial_inputs.get("video_latents")
    if not isinstance(video_latents, torch.Tensor):
        raise TypeError("Exact startup bootstrap requires tensor `video_latents` in prepared inputs.")
    if video_latents.ndim != 5:
        raise ValueError(
            "Expected exact startup video latents with shape [B, C, T, H, W], "
            f"got {tuple(video_latents.shape)}."
        )
    if video_latents.shape[2] == frame_chunk_size:
        return initial_inputs
    if video_latents.shape[2] != 1:
        raise ValueError(
            "Expected exact startup bootstrap to encode exactly one real observation before latent padding, "
            f"got latent length {video_latents.shape[2]} for frame_chunk_size={frame_chunk_size}."
        )

    updated_inputs = dict(initial_inputs)
    updated_inputs["video_latents"] = (
        video_latents[:, :, :1].expand(-1, -1, frame_chunk_size, -1, -1).contiguous()
    )
    return updated_inputs


def _exact_startup_bootstrap_frame_start(frame_chunk_size: int) -> int:
    frame_chunk_size = int(frame_chunk_size)
    if frame_chunk_size <= 0:
        raise ValueError(f"Expected positive frame_chunk_size, got {frame_chunk_size}.")
    return 1 - frame_chunk_size


def _exact_startup_bootstrap_action_history(
    *,
    frame_chunk_size: int,
    action_per_frame: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    frame_chunk_size = int(frame_chunk_size)
    action_per_frame = int(action_per_frame)
    action_dim = int(action_dim)
    if frame_chunk_size <= 0 or action_per_frame <= 0 or action_dim <= 0:
        raise ValueError(
            "Expected positive startup bootstrap action dimensions, "
            f"got frame_chunk_size={frame_chunk_size}, action_per_frame={action_per_frame}, action_dim={action_dim}."
        )
    return torch.zeros(
        1,
        frame_chunk_size * action_per_frame,
        action_dim,
        device=device,
        dtype=torch.float32,
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


def _construct_single_env(task_spec: LiberoTaskSpec, *, env_horizon: int | None):
    ensure_local_libero_config(REPO_ROOT)
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    count = 0
    env = None
    while env is None and count < 5:
        try:
            kwargs = {
                "bddl_file_name": task_spec.bddl_file_path,
                "camera_heights": 128,
                "camera_widths": 128,
            }
            if env_horizon is not None:
                kwargs["horizon"] = int(env_horizon)
            env = OffScreenRenderEnv(**kwargs)
        except Exception as exc:  # pragma: no cover - best-effort retry path
            print(f"construct env failed ({count + 1}/5): {exc}")
            time.sleep(5)
            count += 1
    return env


def _init_single_env(env, init_state) -> dict[str, np.ndarray]:
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)
    if obs is None:
        raise RuntimeError("LIBERO env did not return an observation during initialization.")
    return _extract_obs(obs)


def _extract_obs(obs) -> dict[str, np.ndarray]:
    return {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(obs["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }


def _obs_list_to_views(
    obs_list: list[dict[str, np.ndarray]],
    *,
    config,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    del config
    return {
        LIBERO_OBS_KEYS[0]: torch.from_numpy(np.stack([obs[LIBERO_OBS_KEYS[0]] for obs in obs_list], axis=0)).to(device=device),
        LIBERO_OBS_KEYS[1]: torch.from_numpy(np.stack([obs[LIBERO_OBS_KEYS[1]] for obs in obs_list], axis=0)).to(device=device),
    }


def _prepare_exact_runtime_inputs(
    runner,
    *,
    views: dict[str, torch.Tensor],
    task_text: tuple[str | None, ...] | None,
    frontend_device: torch.device,
    runtime_device: torch.device,
    text_context: torch.Tensor | None = None,
    negative_text_context: torch.Tensor | None = None,
    preserve_stream_cache: bool = False,
) -> dict[str, torch.Tensor | None]:
    canonical_batch = runner.pipeline.canonicalize(views)
    canonical_video = canonical_batch.video.to(device=frontend_device)
    frontend_output = runner.pipeline.visual_tower.run_frontend(
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


def _decode_imagined_video(
    runner,
    predicted_latent_chunks: list[torch.Tensor],
    *,
    decode_device: torch.device,
) -> np.ndarray | None:
    if not predicted_latent_chunks:
        return None
    assets = runner.pipeline.visual_tower.frontend.reference_assets
    if not assets.has_vae:
        return None

    latents = torch.cat(predicted_latent_chunks, dim=2)
    vae = assets.vae
    video_processor = VideoProcessor(vae_scale_factor=1)
    vae_param = next(vae.parameters())
    original_device = vae_param.device
    original_dtype = vae_param.dtype

    target_device = decode_device
    target_dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    if original_device != target_device or original_dtype != target_dtype:
        vae = vae.to(device=target_device, dtype=target_dtype)
    latents = latents.to(device=target_device, dtype=target_dtype)

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


def _build_comparison_video_frames(
    *,
    real_obs_list: list[dict[str, np.ndarray]],
    imagined_video: np.ndarray | None,
) -> list[np.ndarray]:
    final_frames: list[np.ndarray] = []
    imagined_frames = [] if imagined_video is None else list(imagined_video)
    panel_height = 300

    for index, obs in enumerate(real_obs_list):
        agentview = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[0]])
        wrist = np.ascontiguousarray(obs[LIBERO_OBS_KEYS[1]])
        row_real = np.hstack([agentview, wrist])
        row_real = np.ascontiguousarray(row_real)
        row_real = np.array(_with_title(Image.fromarray(row_real), "Real (AgentView / Wrist)"), copy=True)
        target_width = row_real.shape[1]

        if index < len(imagined_frames):
            img_frame = _to_uint8(imagined_frames[index])
            imagined = Image.fromarray(img_frame)
            scale = min(target_width / imagined.width, panel_height / imagined.height)
            resized_w = max(1, int(imagined.width * scale))
            resized_h = max(1, int(imagined.height * scale))
            resized = imagined.resize((resized_w, resized_h))
            row_imagined = Image.new("RGB", (target_width, panel_height), color=(0, 0, 0))
            offset_x = (target_width - resized.width) // 2
            offset_y = (panel_height - resized.height) // 2
            row_imagined.paste(resized, (offset_x, offset_y))
        else:
            row_imagined = Image.new("RGB", (target_width, panel_height), color=(0, 0, 0))
            draw = ImageDraw.Draw(row_imagined)
            draw.text((max(10, target_width // 2 - 140), 150), "No imagined video", fill=(120, 120, 120))
        row_imagined = _with_title(row_imagined, "Imagined (Open-WAM Exact)")
        full_frame = np.vstack([row_real, np.array(row_imagined, copy=True)])
        final_frames.append(np.ascontiguousarray(full_frame))
    return final_frames


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


def _build_open_wam_component_report(
    config,
    runner,
    *,
    runtime_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    requested_eval_checkpoint: Path | None,
) -> dict[str, object]:
    backbone = config.backbone
    action_decoder = runner.pipeline.action_decoder
    policy_variant = runner.pipeline.policy_variant
    adapter_spec = getattr(policy_variant.exact_action_adapter, "spec", None)
    transformer_dir = resolve_pretrained_component_dir(
        backbone.pretrained_model_name_or_path,
        backbone.transformer_subdir,
    )
    vae_dir = resolve_pretrained_component_dir(
        backbone.pretrained_model_name_or_path,
        backbone.vae_subdir,
    )
    text_encoder_dir = resolve_pretrained_component_dir(
        backbone.pretrained_model_name_or_path,
        backbone.text_encoder_subdir,
    )
    tokenizer_dir = resolve_pretrained_component_dir(
        backbone.pretrained_model_name_or_path,
        backbone.tokenizer_subdir,
    )
    transformer = runner.pipeline.visual_tower.get_runtime_backbone(
        action_dim=config.action_decoder.action_dim
    )
    transformer_config = getattr(transformer, "config", None)
    return {
        "pipeline": "open_wam",
        "runtime_device": str(runtime_device),
        "frontend_device": str(frontend_device),
        "decode_device": str(decode_device),
        "requested_eval_checkpoint_file": (
            None if requested_eval_checkpoint is None else str(requested_eval_checkpoint.resolve())
        ),
        "backbone_pretrained_root": str(backbone.pretrained_model_name_or_path),
        "transformer_dir": str(transformer_dir.resolve()) if transformer_dir is not None else None,
        "transformer_config_sha256": _sha256_if_exists(transformer_dir / "config.json" if transformer_dir is not None else None),
        "transformer_weights_sha256": _sha256_if_exists(
            transformer_dir / "diffusion_pytorch_model.safetensors" if transformer_dir is not None else None
        ),
        "vae_dir": str(vae_dir.resolve()) if vae_dir is not None else None,
        "vae_config_sha256": _sha256_if_exists(vae_dir / "config.json" if vae_dir is not None else None),
        "vae_weights_sha256": _sha256_if_exists(vae_dir / "diffusion_pytorch_model.safetensors" if vae_dir is not None else None),
        "text_encoder_dir": str(text_encoder_dir.resolve()) if text_encoder_dir is not None else None,
        "text_encoder_index_sha256": _sha256_if_exists(
            text_encoder_dir / "model.safetensors.index.json" if text_encoder_dir is not None else None
        ),
        "tokenizer_dir": str(tokenizer_dir.resolve()) if tokenizer_dir is not None else None,
        "tokenizer_json_sha256": _sha256_if_exists(tokenizer_dir / "tokenizer.json" if tokenizer_dir is not None else None),
        "spiece_sha256": _sha256_if_exists(tokenizer_dir / "spiece.model" if tokenizer_dir is not None else None),
        "transformer_class": transformer.__class__.__name__,
        "transformer_num_layers": getattr(transformer_config, "num_layers", None),
        "transformer_action_dim": getattr(transformer_config, "action_dim", None),
        "transformer_attn_mode": getattr(transformer_config, "attn_mode", None),
        "transformer_patch_size": list(getattr(transformer, "patch_size", ()) or ()),
        "max_text_tokens": int(backbone.max_text_tokens),
        "frame_chunk_size": int(config.inference.frame_chunk_size),
        "action_per_frame": int(config.policy_variant.action_per_frame),
        "action_decoder_class": action_decoder.__class__.__name__,
        "action_decoder_trainable_params": _count_trainable_parameters(action_decoder),
        "policy_variant_class": policy_variant.__class__.__name__,
        "runtime_mode": getattr(policy_variant.config, "runtime_mode", None),
        "exact_inference_uses_reference_transformer_only": False,
        "exact_inference_uses_shared_transformer_backbone": True,
        "visual_tower_decoder_bypassed_in_exact_mode": True,
        "exact_action_adapter_enabled": adapter_spec is not None,
        "exact_action_adapter_profile": getattr(config.policy_variant, "reference_profile", None),
        "exact_action_adapter_norm_method": getattr(adapter_spec, "action_norm_method", None),
        "exact_action_adapter_raw_action_dim": getattr(adapter_spec, "raw_action_dim", None),
        "exact_action_adapter_model_action_dim": getattr(adapter_spec, "model_action_dim", None),
        "exact_action_adapter_used_action_channel_ids": list(getattr(adapter_spec, "used_action_channel_ids", ()) or ()),
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


def _preview_tensor(tensor: torch.Tensor, *, limit: int = 8) -> list[float]:
    flat = tensor.detach().reshape(-1).to(dtype=torch.float32).cpu().tolist()
    return [float(value) for value in flat[:limit]]


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _resolve_device(device_arg: str | None, *, fallback: torch.device | None = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
