from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch

from open_wam.configs import ActionSpace, ProprioContextMode
from open_wam.data.action_pose import quaternion_to_axis_angle
from open_wam.data.counterfactual_actions import (
    BRANCH_PRESETS,
    apply_action_branch,
    branch_metadata,
    branch_seed_offset,
    expand_branch_names,
)
from open_wam.data.latent_temporal import raw_window_frames_for_latents
from open_wam.integrations import (
    build_libero_offscreen_env,
    ensure_local_libero_config,
    load_libero_task_init_states,
    resolve_libero_task,
)
from open_wam.utils import (
    load_experiment_config,
    merge_runtime_config_from_checkpoint,
    resolve_checkpoint_file,
    resolve_transformer_dir_override,
    seed_everywhere,
)
from open_wam.utils.config_overrides import apply_config_overrides, parse_override_assignments

from .cli import _repair_runtime_config_for_local_eval
from .metrics import rgb_mse_per_frame, simple_ssim_per_frame
from .rollout import JointDenoisingFdmRollout, make_rollout_cursor, should_drop_task_text_for_fdm_mode
from .sampling import require_chunk_aligned_horizon
from .types import FdmAblationMode
from .visualization import decode_latent_video, write_prediction_video


DEFAULT_BRANCHES = BRANCH_PRESETS["diagnostic"]
DEFAULT_COUNTERFACTUAL_MODES = (
    FdmAblationMode.FORCED_ACTION_JOINT_FDM,
    FdmAblationMode.VANILLA_JOINT_ROLLOUT,
    FdmAblationMode.CLEAN_ACTION_FEEDBACK,
)
LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)


@dataclass(frozen=True)
class CounterfactualCase:
    case_index: int
    episode_index: int
    task_text: str
    task_id: int
    init_state_index: int
    parquet_path: Path
    t0_frame: int
    context_start_frame: int
    action_length: int


@dataclass(frozen=True)
class BranchRender:
    case: CounterfactualCase
    branch_name: str
    future_actions: np.ndarray
    context_obs: list[dict[str, np.ndarray]]
    future_obs: list[dict[str, np.ndarray]]
    target_rgb: np.ndarray
    target_video_path: Path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    seed_everywhere(args.seed)
    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    if args.pyopengl_platform:
        os.environ.setdefault("PYOPENGL_PLATFORM", args.pyopengl_platform)

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_file = resolve_checkpoint_file(args.checkpoint)
    checkpoint_dir = checkpoint_file.parent
    transformer_dir = resolve_transformer_dir_override(checkpoint_dir)
    base_config = load_experiment_config(config_path)
    config, resolved_checkpoint_config = merge_runtime_config_from_checkpoint(base_config, checkpoint_file)
    config = _repair_runtime_config_for_local_eval(
        config=config,
        base_config=base_config,
        transformer_dir=transformer_dir,
        dataset_root=args.dataset_root,
        empty_text_embedding_path=args.empty_text_embedding_path,
        reference_assets_device_policy=args.reference_assets_device_policy,
        video_num_inference_steps=args.video_num_inference_steps,
        action_num_inference_steps=args.action_num_inference_steps,
    )
    if args.set_overrides:
        config = apply_config_overrides(
            config,
            parse_override_assignments(tuple(args.set_overrides)),
        )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir).expanduser().resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    action_per_frame = int(config.policy_variant.action_per_frame)
    frame_chunk_size = int(config.inference.frame_chunk_size)
    generated_frames = require_chunk_aligned_horizon(
        horizon_frames=args.horizon_frames,
        frame_chunk_size=frame_chunk_size,
    )
    replay_rows = _load_replay_status(Path(args.replay_status_path).expanduser())
    cases = _select_cases(
        replay_rows,
        episode_indices=_parse_episode_indices(args.episode_indices),
        action_per_frame=action_per_frame,
        horizon_frames=args.horizon_frames,
        context_window_frames=args.context_window_frames,
        t0_frame_arg=args.t0_frame,
        max_cases=args.max_cases,
    )
    modes = tuple(FdmAblationMode(value) for value in args.mode)
    branches = tuple(_parse_branch_names(args.branch))

    manifest = {
        "run_id": run_id,
        "config_path": str(config_path),
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_resolved_config": None if resolved_checkpoint_config is None else str(resolved_checkpoint_config),
        "transformer_dir": str(transformer_dir),
        "output_root": str(output_root),
        "replay_status_path": str(Path(args.replay_status_path).expanduser().resolve()),
        "horizon_frames": int(args.horizon_frames),
        "generated_frames": int(generated_frames),
        "frame_chunk_size": frame_chunk_size,
        "action_per_frame": action_per_frame,
        "context_raw_frames": _raw_window_frames_for_latents(
            int(args.context_window_frames),
            action_per_frame=action_per_frame,
        ),
        "target_raw_frames": _decoded_raw_frames_for_latents(
            int(args.horizon_frames),
            action_per_frame=action_per_frame,
        ),
        "context_window_frames": int(args.context_window_frames),
        "branches": list(branches),
        "branch_metadata": {branch: branch_metadata(branch) for branch in branches},
        "modes": [mode.value for mode in modes],
        "config_overrides": list(args.set_overrides),
        "fdm_drop_text_conditioning": bool(args.fdm_drop_text_conditioning),
        "model_seed_policy": "branch_independent_per_case_mode",
        "seed": int(args.seed),
        "cases": [_case_to_row(case) for case in cases],
    }
    _write_json(output_root / "manifest.json", manifest)
    if args.plan_only:
        print(json.dumps({"status": "plan_only", "manifest": str(output_root / "manifest.json")}, indent=2))
        return

    ensure_local_libero_config(Path.cwd())
    from open_wam.pipelines import build_exact_runtime_runner_from_config

    runner = build_exact_runtime_runner_from_config(config)
    fdm_rollout = JointDenoisingFdmRollout(runner)
    runtime_device = torch.device(args.runtime_device)
    frontend_device = torch.device(args.frontend_device or args.runtime_device)
    decode_device = torch.device(args.decode_device or args.runtime_device)

    metric_rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    rendered_cache: dict[tuple[int, str], BranchRender] = {}
    for case in cases:
        actions = _read_actions(case.parquet_path)
        task_spec = resolve_libero_task(case.task_text, project_root=Path.cwd(), benchmark_name=args.benchmark)
        init_states = load_libero_task_init_states(task_spec, project_root=Path.cwd())
        init_state = init_states[int(case.init_state_index) % len(init_states)]
        env = build_libero_offscreen_env(
            task_spec,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            horizon=max(args.env_horizon, int((case.t0_frame + generated_frames + 2) * action_per_frame + 32)),
            ignore_done=True,
            project_root=Path.cwd(),
        )
        try:
            for branch_name in branches:
                branch = _render_counterfactual_branch(
                    env=env,
                    init_state=init_state,
                    case=case,
                    actions=actions,
                    branch_name=branch_name,
                    horizon_frames=args.horizon_frames,
                    generated_frames=generated_frames,
                    action_per_frame=action_per_frame,
                    seed=args.seed + case.case_index * 100 + branch_seed_offset(branch_name),
                    output_root=output_root,
                    video_fps=args.video_fps,
                )
                rendered_cache[(case.case_index, branch_name)] = branch
                for mode_index, mode in enumerate(modes):
                    result = _run_model_on_branch(
                        fdm_rollout=fdm_rollout,
                        branch=branch,
                        case=case,
                        mode=mode,
                        prompt=case.task_text,
                        action_per_frame=action_per_frame,
                        frame_chunk_size=frame_chunk_size,
                        generated_frames=generated_frames,
                        runtime_device=runtime_device,
                        frontend_device=frontend_device,
                        decode_device=decode_device,
                        seed=args.seed + case.case_index * 1000 + mode_index * 10000,
                        output_root=output_root,
                        video_fps=args.video_fps,
                        fdm_drop_text_conditioning=bool(args.fdm_drop_text_conditioning),
                        config=config,
                    )
                    metric_rows.extend(result["metric_rows"])
                    sample_summaries.append(result["summary"])
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                _write_json(
                    output_root / "latest_progress.json",
                    {
                        "completed_case_branches": len(rendered_cache),
                        "total_case_branches": len(cases) * len(branches),
                    },
                )
        finally:
            env.close()

    summary_rows = _summarize_counterfactual_metric_rows(metric_rows)
    _write_csv(output_root / "metrics_per_step.csv", metric_rows)
    _write_csv(output_root / "metrics_summary.csv", summary_rows)
    _write_jsonl(output_root / "sample_results.jsonl", sample_summaries)
    summary = {
        "run_id": run_id,
        "output_root": str(output_root),
        "metrics_per_step": str(output_root / "metrics_per_step.csv"),
        "metrics_summary": str(output_root / "metrics_summary.csv"),
        "sample_results": str(output_root / "sample_results.jsonl"),
        "target_video_count": len(list((output_root / "videos").glob("target_*.mp4"))),
        "prediction_video_count": len(list((output_root / "videos").glob("pred_*.mp4"))),
        "summary_rows": summary_rows,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _select_cases(
    replay_rows: list[dict[str, Any]],
    *,
    episode_indices: tuple[int, ...],
    action_per_frame: int,
    horizon_frames: int,
    context_window_frames: int,
    t0_frame_arg: int | None,
    max_cases: int | None,
) -> list[CounterfactualCase]:
    by_episode = {int(row["dataset_episode_index"]): row for row in replay_rows}
    selected: list[CounterfactualCase] = []
    for episode_index in episode_indices:
        row = by_episode.get(int(episode_index))
        if row is None:
            raise ValueError(f"Episode {episode_index} was not found in replay metadata.")
        if row.get("failure"):
            raise ValueError(f"Episode {episode_index} is marked replay failure; choose a successful replay row.")
        parquet_path = Path(str(row["parquet_path"]))
        actions = _read_actions(parquet_path)
        total_video_frames = actions.shape[0] // int(action_per_frame)
        min_t0 = max(1, int(context_window_frames))
        max_t0 = total_video_frames - int(horizon_frames) - 1
        if max_t0 < min_t0:
            raise ValueError(
                f"Episode {episode_index} is too short for horizon={horizon_frames}: "
                f"total_video_frames={total_video_frames}, min_t0={min_t0}."
            )
        t0_frame = int(t0_frame_arg) if t0_frame_arg is not None else int(round(total_video_frames * 0.35))
        t0_frame = min(max(min_t0, t0_frame), max_t0)
        init_state_index = row.get("resolved_init_state_index")
        if init_state_index is None:
            init_state_index = row.get("primary_init_state_index")
        if init_state_index is None:
            raise ValueError(f"Episode {episode_index} has no resolved or primary init state index.")
        selected.append(
            CounterfactualCase(
                case_index=len(selected),
                episode_index=int(episode_index),
                task_text=str(row["task_text"]),
                task_id=int(row.get("metadata_task_index") or -1),
                init_state_index=int(init_state_index),
                parquet_path=parquet_path,
                t0_frame=t0_frame,
                context_start_frame=max(0, t0_frame - int(context_window_frames)),
                action_length=int(actions.shape[0]),
            )
        )
        if max_cases is not None and len(selected) >= int(max_cases):
            break
    return selected


def _render_counterfactual_branch(
    *,
    env: Any,
    init_state: Any,
    case: CounterfactualCase,
    actions: np.ndarray,
    branch_name: str,
    horizon_frames: int,
    generated_frames: int,
    action_per_frame: int,
    seed: int,
    output_root: Path,
    video_fps: float,
) -> BranchRender:
    t0_action_index = int(case.t0_frame * action_per_frame)
    future_action_count = int(horizon_frames * action_per_frame)
    context_latent_frames = int(case.t0_frame - case.context_start_frame)
    context_raw_frame_count = _raw_window_frames_for_latents(context_latent_frames, action_per_frame=action_per_frame)
    future_actions = actions[t0_action_index : t0_action_index + future_action_count].copy()
    future_actions = apply_action_branch(future_actions, branch_name=branch_name, seed=seed)

    obs = env.reset()
    obs = env.set_init_state(init_state)
    context_action_start = int(case.context_start_frame * action_per_frame)
    for action_index in range(context_action_start):
        obs, _, _, _ = env.step(actions[action_index].astype(np.float32, copy=False))

    context_obs: list[dict[str, np.ndarray]] = []
    for raw_offset in range(context_raw_frame_count):
        action_index = context_action_start + raw_offset
        obs, _, _, _ = env.step(actions[action_index].astype(np.float32, copy=False))
        context_obs.append(_extract_obs(obs))
    if len(context_obs) != context_raw_frame_count:
        raise RuntimeError(
            f"Failed to render full context for episode={case.episode_index}: "
            f"expected={context_raw_frame_count}, got={len(context_obs)}."
        )

    for action_index in range(context_action_start + context_raw_frame_count, t0_action_index):
        obs, _, _, _ = env.step(actions[action_index].astype(np.float32, copy=False))

    target_frames: list[np.ndarray] = []
    future_obs: list[dict[str, np.ndarray]] = []
    del generated_frames
    target_raw_frame_count = _decoded_raw_frames_for_latents(horizon_frames, action_per_frame=action_per_frame)
    t0_obs = _extract_obs(obs)
    future_obs.append(t0_obs)
    target_frames.append(_compose_obs_rgb(t0_obs))
    for action in future_actions[: max(0, target_raw_frame_count - 1)]:
        obs, _, _, _ = env.step(action.astype(np.float32, copy=False))
        extracted_obs = _extract_obs(obs)
        future_obs.append(extracted_obs)
        target_frames.append(_compose_obs_rgb(extracted_obs))
    if len(target_frames) != target_raw_frame_count:
        raise RuntimeError(
            "Failed to render full counterfactual target including t0 observation, "
            f"expected={target_raw_frame_count}, got={len(target_frames)}."
        )
    target_rgb = np.stack(target_frames, axis=0).astype(np.float32) / 255.0

    target_video_path = (
        output_root
        / "videos"
        / f"target_case{case.case_index:02d}_ep{case.episode_index:06d}_{branch_name}.mp4"
    )
    _write_rgb_video(target_video_path, target_rgb, fps=video_fps)
    return BranchRender(
        case=case,
        branch_name=branch_name,
        future_actions=future_actions,
        context_obs=context_obs,
        future_obs=future_obs,
        target_rgb=target_rgb,
        target_video_path=target_video_path,
    )


def _run_model_on_branch(
    *,
    fdm_rollout: JointDenoisingFdmRollout,
    branch: BranchRender,
    case: CounterfactualCase,
    mode: FdmAblationMode,
    prompt: str,
    action_per_frame: int,
    frame_chunk_size: int,
    generated_frames: int,
    runtime_device: torch.device,
    frontend_device: torch.device,
    decode_device: torch.device,
    seed: int,
    output_root: Path,
    video_fps: float,
    fdm_drop_text_conditioning: bool,
    config,
) -> dict[str, Any]:
    seed_everywhere(seed)
    runner = fdm_rollout.runner
    session = runner.reset(task_text=(prompt,))
    if case.context_start_frame:
        session.policy_state.cache["frame_start"] = int(case.context_start_frame)
        session.policy_state.cursor = make_rollout_cursor(
            session.policy_state.cursor,
            current_start_frame=int(case.context_start_frame),
            block_index=session.policy_state.cursor.block_index,
            chunk_size=session.policy_state.cursor.chunk_size,
        )
    video_latents, text_context, negative_text_context = _prepare_obs_context_with_frontend(
        runner=runner,
        obs_list=branch.context_obs,
        prompt=prompt,
        frontend_device=frontend_device,
        runtime_device=runtime_device,
    )
    expected_context_frames = int(case.t0_frame - case.context_start_frame)
    if int(video_latents.shape[2]) != expected_context_frames:
        raise RuntimeError(
            "Counterfactual context VAE encoding returned an unexpected number of latent frames: "
            f"got={int(video_latents.shape[2])}, expected={expected_context_frames}."
        )
    action_start = case.context_start_frame * action_per_frame
    action_end = case.t0_frame * action_per_frame
    all_actions = _read_actions(case.parquet_path)
    action_context = torch.from_numpy(all_actions[action_start:action_end]).unsqueeze(0).to(
        device=runtime_device,
        dtype=torch.float32,
    )
    drop_text_for_mode = _should_drop_text_conditioning(mode, fdm_drop_text_conditioning=fdm_drop_text_conditioning)
    if drop_text_for_mode:
        if text_context is None:
            visual_config = runner.pipeline.visual_tower.config
            warmup_text_context = torch.zeros(
                int(video_latents.shape[0]),
                int(visual_config.max_text_tokens),
                int(visual_config.text_dim),
                device=video_latents.device,
                dtype=video_latents.dtype,
            )
        else:
            warmup_text_context = torch.zeros_like(text_context)
        warmup_negative_text_context = (
            torch.zeros_like(negative_text_context)
            if negative_text_context is not None
            else None
        )
    else:
        warmup_text_context = text_context
        warmup_negative_text_context = negative_text_context
    warmup = runner.warmup_cache(
        session=session,
        video_latents=video_latents,
        text_context=warmup_text_context,
        negative_text_context=warmup_negative_text_context,
        action_history=action_context,
        action_space=ActionSpace.RAW,
        action_conditioning_mode=mode.value,
        proprio_state=_proprio_state_from_obs(
            branch.context_obs[-1],
            config=config,
            device=runtime_device,
        ),
    )
    session = warmup.session

    future_actions = torch.from_numpy(branch.future_actions).unsqueeze(0).to(device=runtime_device, dtype=torch.float32)
    predicted_chunks: list[torch.Tensor] = []
    chunk_debug: list[dict[str, Any]] = []
    for frame_offset in range(0, generated_frames, frame_chunk_size):
        raw_start = frame_offset * action_per_frame
        raw_end = (frame_offset + frame_chunk_size) * action_per_frame
        raw_action_chunk = future_actions[:, raw_start:raw_end]
        expected_action_count = frame_chunk_size * action_per_frame
        if raw_action_chunk.shape[1] != expected_action_count:
            raise RuntimeError(
                "Counterfactual FDM action chunk is not horizon-aligned: "
                f"got={raw_action_chunk.shape[1]}, expected={expected_action_count}. "
                "Use a horizon that is divisible by the model frame chunk size."
            )
        chunk = fdm_rollout.infer_chunk(
            session=session,
            mode=mode,
            raw_action_chunk=raw_action_chunk,
            seed=seed + frame_offset,
            drop_text_conditioning=drop_text_for_mode,
            proprio_state=_proprio_state_from_obs(
                _proprio_obs_for_future_chunk(
                    branch=branch,
                    frame_offset=frame_offset,
                    action_per_frame=action_per_frame,
                ),
                config=config,
                device=runtime_device,
            ),
        )
        predicted_chunks.append(chunk.predicted_latents.detach().cpu())
        chunk_debug.append(dict(chunk.debug))
        session = chunk.session

    target_latent_frames = int(branch.future_actions.shape[0] // action_per_frame)
    predicted_latents = torch.cat(predicted_chunks, dim=2)[:, :, :target_latent_frames].cpu()
    predicted_rgb = decode_latent_video(runner.pipeline, predicted_latents, decode_device=decode_device)
    if predicted_rgb is None:
        raise RuntimeError("Counterfactual FDM eval requires a VAE to decode predicted latents.")
    target_rgb = _resize_target_to_prediction(branch.target_rgb, predicted_rgb)
    rgb_mse = rgb_mse_per_frame(predicted_rgb, target_rgb)
    rgb_ssim = simple_ssim_per_frame(predicted_rgb, target_rgb)

    video_path = output_root / "videos" / (
        f"pred_case{case.case_index:02d}_ep{case.episode_index:06d}_{branch.branch_name}_{mode.value}.mp4"
    )
    write_prediction_video(
        output_path=video_path,
        target_rgb=target_rgb,
        predicted_rgb=predicted_rgb,
        title=f"case={case.case_index} ep={case.episode_index} {branch.branch_name} {mode.value}",
        fps=video_fps,
    )
    metric_rows = []
    for horizon_index, (mse_value, ssim_value) in enumerate(zip(rgb_mse, rgb_ssim, strict=True)):
        metric_rows.append(
            {
                **_case_to_row(case),
                "branch": branch.branch_name,
                "mode": mode.value,
                "fdm_drop_text_conditioning": bool(drop_text_for_mode),
                "horizon_index": int(horizon_index),
                "rgb_mse": float(mse_value),
                "rgb_ssim": float(ssim_value),
                "target_video_path": str(branch.target_video_path),
                "prediction_video_path": str(video_path),
            }
        )
    summary = {
        **_case_to_row(case),
        "branch": branch.branch_name,
        "mode": mode.value,
        "fdm_drop_text_conditioning": bool(drop_text_for_mode),
        "rgb_mse_mean": float(sum(rgb_mse) / len(rgb_mse)),
        "rgb_ssim_mean": float(sum(rgb_ssim) / len(rgb_ssim)),
        "target_video_path": str(branch.target_video_path),
        "prediction_video_path": str(video_path),
        "chunk_debug": _small_chunk_debug(chunk_debug),
    }
    return {"metric_rows": metric_rows, "summary": summary}


def _prepare_obs_context_with_frontend(
    *,
    runner: Any,
    obs_list: list[dict[str, np.ndarray]],
    prompt: str,
    frontend_device: torch.device,
    runtime_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Prepare simulator observations through the same LingBot frontend path used by rollout."""

    if not obs_list:
        raise ValueError("Counterfactual FDM warmup requires at least one context observation.")
    views = _obs_list_to_views(obs_list, device=frontend_device)
    visual_outputs = runner.pipeline.prepare_visual_outputs(
        views,
        task_text=(prompt,),
        preserve_stream_cache=False,
    )
    video_latents = visual_outputs.frontend.video_latents
    text_context = visual_outputs.frontend.conditioning.text_context
    negative_text_context = visual_outputs.frontend.conditioning.negative_text_context
    return (
        video_latents.to(device=runtime_device),
        None if text_context is None else text_context.to(device=runtime_device),
        None if negative_text_context is None else negative_text_context.to(device=runtime_device),
    )


def _extract_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    extracted = {
        LIBERO_OBS_KEYS[0]: np.ascontiguousarray(obs["agentview_image"][::-1]),
        LIBERO_OBS_KEYS[1]: np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if key in obs:
            extracted[key] = np.asarray(obs[key], dtype=np.float32).copy()
    return extracted


def _proprio_obs_for_future_chunk(
    *,
    branch: BranchRender,
    frame_offset: int,
    action_per_frame: int,
) -> dict[str, np.ndarray]:
    if frame_offset <= 0 or not branch.future_obs:
        return branch.context_obs[-1]
    previous_latent_frame = int(frame_offset) - 1
    raw_index = max(0, min(int(previous_latent_frame) * int(action_per_frame), len(branch.future_obs) - 1))
    return branch.future_obs[raw_index]


def _proprio_state_from_obs(
    obs: dict[str, np.ndarray],
    *,
    config,
    device: torch.device,
) -> torch.Tensor | None:
    policy_config = getattr(config, "policy_variant", None)
    mode = getattr(policy_config, "proprio_context_mode", ProprioContextMode.NONE)
    if ProprioContextMode(mode) not in {
        ProprioContextMode.PER_CHUNK_ADDITIVE,
        # Deprecated compatibility for older text-space proprio token checkpoints.
        ProprioContextMode.TEXT_CONTEXT_TOKEN,
    }:
        return None
    state_encoding = getattr(getattr(config.data, "action_target", None), "state_encoding", None)
    if state_encoding != "eef_pos_axisangle_gripper_2d":
        raise ValueError(
            "Counterfactual proprio context currently supports only "
            f"state_encoding='eef_pos_axisangle_gripper_2d', got {state_encoding!r}."
        )
    state = _extract_libero_eef_axisangle_gripper_state(obs)
    expected_dim = int(getattr(getattr(config.data, "action_schema", None), "state_dim", 0) or 0)
    if expected_dim > 0 and state.shape[0] != expected_dim:
        raise ValueError(
            "Counterfactual proprio context dim does not match data.action_schema.state_dim, "
            f"got {state.shape[0]} and expected {expected_dim}."
        )
    return torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)


def _extract_libero_eef_axisangle_gripper_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if eef_pos.shape[0] != 3:
        raise ValueError(f"Expected LIBERO robot0_eef_pos to have dim 3, got {eef_pos.shape[0]}.")
    if eef_quat.shape[0] != 4:
        raise ValueError(f"Expected LIBERO robot0_eef_quat to have dim 4, got {eef_quat.shape[0]}.")
    if gripper_qpos.shape[0] != 2:
        raise ValueError(f"Expected LIBERO robot0_gripper_qpos to have dim 2, got {gripper_qpos.shape[0]}.")
    axisangle = (
        quaternion_to_axis_angle(torch.from_numpy(eef_quat).to(dtype=torch.float32).unsqueeze(0))[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    if axisangle.shape[0] != 3:
        raise ValueError(f"Expected axis-angle proprio dim 3, got {axisangle.shape[0]}.")
    return np.concatenate([eef_pos, axisangle, gripper_qpos], axis=0).astype(np.float32, copy=False)


def _compose_obs_rgb(obs: dict[str, np.ndarray]) -> np.ndarray:
    left = _as_uint8(obs[LIBERO_OBS_KEYS[0]])
    right = _as_uint8(obs[LIBERO_OBS_KEYS[1]])
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"Expected matching camera shapes, got {left.shape} and {right.shape}.")
    return np.concatenate([left, right], axis=1)


def _obs_list_to_views(obs_list: list[dict[str, np.ndarray]], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        LIBERO_OBS_KEYS[0]: torch.from_numpy(np.stack([obs[LIBERO_OBS_KEYS[0]] for obs in obs_list], axis=0)).to(device=device),
        LIBERO_OBS_KEYS[1]: torch.from_numpy(np.stack([obs[LIBERO_OBS_KEYS[1]] for obs in obs_list], axis=0)).to(device=device),
    }


def _resize_target_to_prediction(target_rgb: np.ndarray, predicted_rgb: np.ndarray) -> np.ndarray:
    if target_rgb.shape == predicted_rgb.shape:
        return target_rgb
    if int(target_rgb.shape[0]) != int(predicted_rgb.shape[0]):
        raise ValueError(
            "Target and decoded prediction must have the same temporal length before spatial resize, "
            f"got target={target_rgb.shape} and predicted={predicted_rgb.shape}."
        )
    from PIL import Image

    frames = []
    target_u8 = _as_uint8(target_rgb)
    for frame in target_u8:
        resized = Image.fromarray(frame).resize((predicted_rgb.shape[2], predicted_rgb.shape[1]))
        frames.append(np.asarray(resized, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)


def _write_rgb_video(path: Path, rgb: np.ndarray, *, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [_as_uint8(frame) for frame in rgb], fps=float(fps), macro_block_size=1)


def _read_actions(parquet_path: Path) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=["action"])
    return np.asarray(table.column("action").to_pylist(), dtype=np.float32)


def _load_replay_status(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _case_to_row(case: CounterfactualCase) -> dict[str, Any]:
    return {
        "case_index": int(case.case_index),
        "episode_index": int(case.episode_index),
        "task_text": case.task_text,
        "task_id": int(case.task_id),
        "init_state_index": int(case.init_state_index),
        "parquet_path": str(case.parquet_path),
        "t0_frame": int(case.t0_frame),
        "context_start_frame": int(case.context_start_frame),
        "action_length": int(case.action_length),
    }


def _small_chunk_debug(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "action_conditioning_mode",
        "generalist_mode_text_token",
        "generalist_mode_text_token_count",
        "drop_text_conditioning",
        "forced_action_denoise",
        "forced_clean_action_conditioning",
        "forced_video_conditioning",
        "commit_action_override",
        "rollout_window_size",
        "generalist_conditional_history_chunks",
        "generation_frame_start",
    }
    return [{key: value for key, value in chunk.items() if key in keep} for chunk in chunks]


def _should_drop_text_conditioning(
    mode: FdmAblationMode,
    *,
    fdm_drop_text_conditioning: bool,
) -> bool:
    return bool(
        should_drop_task_text_for_fdm_mode(mode)
        or (fdm_drop_text_conditioning and mode != FdmAblationMode.VANILLA_JOINT_ROLLOUT)
    )


def _as_uint8(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array
    if array.size and float(np.nanmax(array)) <= 1.0001:
        return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _raw_window_frames_for_latents(latent_frames: int, *, action_per_frame: int) -> int:
    return raw_window_frames_for_latents(latent_frames, action_per_frame=action_per_frame)


def _decoded_raw_frames_for_latents(latent_frames: int, *, action_per_frame: int) -> int:
    if latent_frames <= 0:
        raise ValueError(f"Expected positive latent frame count, got {latent_frames}.")
    return int(action_per_frame) * int(latent_frames) + 1


def _parse_episode_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices:
        raise ValueError("--episode-indices must contain at least one integer.")
    return indices


def _parse_branch_names(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_BRANCHES)
    return list(expand_branch_names(",".join(values)))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_counterfactual_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["branch"]), str(row["mode"]), int(row["horizon_index"])),
            [],
        ).append(row)

    summaries: list[dict[str, Any]] = []
    for (branch, mode, horizon_index), group_rows in sorted(groups.items()):
        rgb_mse = np.asarray([float(row["rgb_mse"]) for row in group_rows], dtype=np.float64)
        rgb_ssim = np.asarray([float(row["rgb_ssim"]) for row in group_rows], dtype=np.float64)
        summaries.append(
            {
                "branch": branch,
                "mode": mode,
                "horizon_index": int(horizon_index),
                "count": len(group_rows),
                "rgb_mse_mean": float(rgb_mse.mean()),
                "rgb_mse_std": float(rgb_mse.std(ddof=0)),
                "rgb_ssim_mean": float(rgb_ssim.mean()),
                "rgb_ssim_std": float(rgb_ssim.std(ddof=0)),
            }
        )
    return summaries


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulator-counterfactual FDM evaluation on LIBERO.")
    parser.add_argument("--config", "--cfg", default="configs/experiments/parallel_stream_libero_lingbot_joint_denoise.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--replay-status-path", required=True)
    parser.add_argument("--episode-indices", required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/joint_denoising_fdm_counterfactual")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--horizon-frames", type=int, default=16)
    parser.add_argument("--context-window-frames", type=int, default=16)
    parser.add_argument("--t0-frame", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime-device", default="cuda:0")
    parser.add_argument("--frontend-device", default=None)
    parser.add_argument("--decode-device", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--empty-text-embedding-path", default=None)
    parser.add_argument("--reference-assets-device-policy", default=None)
    parser.add_argument("--video-num-inference-steps", type=int, default=None)
    parser.add_argument("--action-num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Repeatable `section.field=value` config override, applied after checkpoint/runtime repair.",
    )
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--env-horizon", type=int, default=5000)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--mujoco-gl", default="osmesa")
    parser.add_argument("--pyopengl-platform", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--fdm-drop-text-conditioning",
        action="store_true",
        help=(
            "Also omit task-text embeddings for non-vanilla diagnostic modes. "
            "`forced_action_joint_fdm` always drops task text to match GJD FDM training semantics."
        ),
    )
    parser.add_argument(
        "--branch",
        action="append",
        default=None,
        help=(
            "Counterfactual action branch. Repeat or comma-separate. Defaults to "
            + ",".join(DEFAULT_BRANCHES)
            + "."
        ),
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in FdmAblationMode],
        default=None,
    )
    args = parser.parse_args(argv)
    try:
        _parse_episode_indices(args.episode_indices)
    except ValueError as error:
        parser.error(str(error))
    if args.mode is None:
        args.mode = [mode.value for mode in DEFAULT_COUNTERFACTUAL_MODES]
    if FdmAblationMode.VIDEO_CONDITIONED_ACTION.value in args.mode:
        parser.error(
            "`video_conditioned_action` is an IDM/action-prediction diagnostic. "
            "Use the standard joint-denoising FDM ablation CLI for IDM rollout coverage; "
            "the counterfactual renderer only evaluates future-video modes."
        )
    if args.context_window_frames <= 0:
        parser.error("--context-window-frames must be positive for counterfactual warmup.")
    if args.horizon_frames <= 0:
        parser.error("--horizon-frames must be positive.")
    return args


if __name__ == "__main__":
    main()
