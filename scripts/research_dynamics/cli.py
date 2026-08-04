from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from open_wam.configs import ReplayStatusPolicy
from open_wam.utils import (
    load_experiment_config,
    merge_runtime_config_from_checkpoint,
    resolve_checkpoint_file,
    resolve_transformer_dir_override,
    seed_everywhere,
)
from open_wam.utils.config_overrides import (
    apply_config_overrides,
    parse_override_assignments,
)

from .metrics import (
    action_mse_per_frame,
    build_metric_rows,
    latent_mse_per_frame,
    rgb_mse_per_frame,
    simple_ssim_per_frame,
    summarize_metric_rows,
)
from .rollout import (
    DualExpertGeneralistDenoisingFdmRollout,
    JointDenoisingFdmRollout,
    should_drop_task_text_for_fdm_mode,
)
from .sampling import (
    build_counterfactual_fdm_eval_dataset,
    build_fdm_eval_dataset,
    require_chunk_aligned_horizon,
    select_counterfactual_target_only_windows,
    select_early_middle_windows,
    selection_to_manifest_row,
)
from .types import FdmAblationMode, FdmStartPolicy
from .visualization import decode_latent_video, write_prediction_video


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    seed_everywhere(args.seed)

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

    modes = tuple(FdmAblationMode(value) for value in args.mode)
    action_per_frame = _resolve_action_per_frame(config)
    selection_fit_target_start_offset_frames = max(
        (_target_start_offset_for_config_mode(config, mode) for mode in modes),
        default=0,
    )
    dataset_source: dict[str, Any]
    if args.counterfactual_latent_root is not None:
        _validate_counterfactual_eval_modes(modes)
        dataset = build_counterfactual_fdm_eval_dataset(
            config.data,
            encoded_root=args.counterfactual_latent_root,
            split=args.counterfactual_split,
        )
        selections = select_counterfactual_target_only_windows(
            dataset,
            horizon_frames=args.horizon_frames,
            frame_chunk_size=config.inference.frame_chunk_size,
            target_start_offset_frames=selection_fit_target_start_offset_frames,
        )
        dataset_source = {
            "type": "encoded_counterfactual_dynamics",
            "counterfactual_latent_root": str(Path(args.counterfactual_latent_root).expanduser().resolve()),
            "counterfactual_split": str(args.counterfactual_split),
        }
    else:
        dataset = build_fdm_eval_dataset(
            config.data,
            replay_status_policy=ReplayStatusPolicy(args.replay_status_policy),
        )
        selections = select_early_middle_windows(
            dataset,
            horizon_frames=args.horizon_frames,
            frame_chunk_size=config.inference.frame_chunk_size,
            action_per_frame=action_per_frame,
            trajectories_per_task=args.trajectories_per_task,
            seed=args.seed,
            min_context_frames=args.min_context_frames,
            context_window_frames=args.context_window_frames,
            start_policy=FdmStartPolicy(args.start_policy),
            fit_target_start_offset_frames=selection_fit_target_start_offset_frames,
        )
        dataset_source = {
            "type": "lerobot_latent_windows",
            "dataset_root": args.dataset_root,
            "replay_status_policy": args.replay_status_policy,
        }
    if args.sample_start:
        selections = selections[int(args.sample_start) :]
    if args.max_samples is not None:
        selections = selections[: int(args.max_samples)]
    generated_frames = require_chunk_aligned_horizon(
        horizon_frames=args.horizon_frames,
        frame_chunk_size=config.inference.frame_chunk_size,
    )
    manifest = {
        "run_id": run_id,
        "config_path": str(config_path),
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_resolved_config": None if resolved_checkpoint_config is None else str(resolved_checkpoint_config),
        "transformer_dir": str(transformer_dir),
        "output_root": str(output_root),
        "horizon_frames": int(args.horizon_frames),
        "generated_frames": int(generated_frames),
        "trajectories_per_task": int(args.trajectories_per_task),
        "context_window_frames": args.context_window_frames,
        "start_policy": args.start_policy,
        "seed": int(args.seed),
        "modes": [mode.value for mode in modes],
        "config_overrides": list(args.set_overrides),
        "dataset_source": dataset_source,
        "frame_chunk_size": int(config.inference.frame_chunk_size),
        "action_per_frame": int(action_per_frame),
        "selection_fit_target_start_offset_frames": int(selection_fit_target_start_offset_frames),
        "dataset_length": len(dataset),
        "selection_count": len(selections),
        "sample_start": int(args.sample_start),
        "max_samples": args.max_samples,
        "selections": [selection_to_manifest_row(selection) for selection in selections],
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "samples.jsonl", manifest["selections"])

    if args.plan_only:
        print(json.dumps({"status": "plan_only", "manifest": str(output_root / "manifest.json")}, indent=2))
        return

    runtime_device = torch.device(args.runtime_device)
    decode_device = torch.device(args.decode_device or args.runtime_device)
    runtime_dtype = _resolve_runtime_dtype(args.runtime_dtype)
    fdm_rollout = _build_fdm_rollout_for_config(
        config=config,
        checkpoint_file=checkpoint_file,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
    )

    metric_rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    saved_video_keys: set[tuple[int, str]] = set()
    for selection in selections:
        sample = dataset[selection.dataset_index]
        sample_summary = {
            **selection_to_manifest_row(selection),
            "task_text": sample.task_text,
            "sample_metadata": _sample_report_metadata(sample),
            "modes": {},
        }
        for mode_index, mode in enumerate(modes):
            result = _run_one_selection_mode(
                fdm_rollout=fdm_rollout,
                selection=selection,
                sample=sample,
                mode=mode,
                runtime_device=runtime_device,
                decode_device=decode_device,
                seed=args.seed + selection.sample_index * 1000 + mode_index * 100,
                write_video=(selection.task_rank, mode.value) not in saved_video_keys,
                video_dir=output_root / "videos",
                video_fps=args.video_fps,
            )
            metric_rows.extend(result["metric_rows"])
            sample_summary["modes"][mode.value] = result["summary"]
            if result["video_path"] is not None:
                saved_video_keys.add((selection.task_rank, mode.value))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        sample_summaries.append(sample_summary)
        _write_json(output_root / "latest_progress.json", {"completed_samples": len(sample_summaries), "total": len(selections)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = summarize_metric_rows(metric_rows)
    _write_csv(output_root / "metrics_per_step.csv", metric_rows)
    _write_csv(output_root / "metrics_summary.csv", summary_rows)
    _write_jsonl(output_root / "sample_results.jsonl", sample_summaries)
    summary = {
        "run_id": run_id,
        "output_root": str(output_root),
        "metrics_per_step": str(output_root / "metrics_per_step.csv"),
        "metrics_summary": str(output_root / "metrics_summary.csv"),
        "sample_results": str(output_root / "sample_results.jsonl"),
        "video_count": len(list((output_root / "videos").glob("*.mp4"))) if (output_root / "videos").exists() else 0,
        "summary_rows": summary_rows,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


@torch.inference_mode()
def _run_one_selection_mode(
    *,
    fdm_rollout: JointDenoisingFdmRollout | DualExpertGeneralistDenoisingFdmRollout,
    selection,
    sample,
    mode: FdmAblationMode,
    runtime_device: torch.device,
    decode_device: torch.device,
    seed: int,
    write_video: bool,
    video_dir: Path,
    video_fps: float,
) -> dict[str, Any]:
    seed_everywhere(seed)
    action_per_frame = fdm_rollout.action_per_frame
    frame_chunk_size = fdm_rollout.frame_chunk_size
    video_latents = sample.video_latents.unsqueeze(0).to(device=runtime_device, dtype=torch.float32)
    actions = sample.actions.unsqueeze(0).to(device=runtime_device, dtype=torch.float32)
    text_context = _optional_batch_tensor(sample.text_context, runtime_device)
    negative_text_context = _optional_batch_tensor(sample.negative_text_context, runtime_device)
    drop_text_conditioning = should_drop_task_text_for_fdm_mode(mode)

    mode_selection = replace(
        selection,
        target_start_offset_frames=_target_start_offset_for_rollout_mode(fdm_rollout, mode),
    )
    t0 = int(mode_selection.t0_frame)
    target_start_offset = int(mode_selection.target_start_offset_frames)
    target_start_frame = int(mode_selection.target_start_frame)
    action_source_start_frame = target_start_frame - target_start_offset
    generated_frames = int(mode_selection.generated_frames)
    target_horizon = int(mode_selection.horizon_frames)
    aligned_horizon = require_chunk_aligned_horizon(
        horizon_frames=target_horizon,
        frame_chunk_size=frame_chunk_size,
    )
    if generated_frames != aligned_horizon:
        raise ValueError(
            "FDM rollout selections must use chunk-aligned generated frames. "
            f"Got horizon_frames={target_horizon}, generated_frames={generated_frames}, "
            f"frame_chunk_size={frame_chunk_size}."
        )
    if target_start_frame + target_horizon > int(video_latents.shape[2]):
        raise ValueError(
            "FDM rollout target horizon exceeds available video latents: "
            f"target_start_frame={target_start_frame}, horizon_frames={target_horizon}, "
            f"total_video_frames={video_latents.shape[2]}."
        )
    if action_source_start_frame < 0:
        raise ValueError(
            "FDM rollout action source starts before frame zero: "
            f"target_start_frame={target_start_frame}, target_start_offset_frames={target_start_offset}."
        )
    video_context = video_latents[:, :, mode_selection.context_start_frame:target_start_frame]
    action_context = actions[
        :, mode_selection.context_start_frame * action_per_frame : target_start_frame * action_per_frame
    ]
    if target_start_offset > 0:
        action_context = torch.zeros_like(action_context)
    warmup_proprio_state = _optional_proprio_state_at_frame(sample, action_source_start_frame, runtime_device)
    warmup_hidden_proprio_history = _optional_proprio_state_sequence(
        sample,
        start_frame=int(mode_selection.context_start_frame),
        end_frame=target_start_frame,
        device=runtime_device,
    )
    session = fdm_rollout.reset_and_warmup(
        task_text=(sample.task_text,),
        video_context=video_context,
        action_context=action_context,
        text_context=text_context,
        negative_text_context=negative_text_context,
        context_start_frame=int(mode_selection.context_start_frame),
        action_space="raw",
        action_conditioning_mode=mode.value,
        drop_text_conditioning=drop_text_conditioning,
        proprio_state=warmup_proprio_state,
        hidden_proprio_history=warmup_hidden_proprio_history,
    )

    predicted_chunks: list[torch.Tensor] = []
    predicted_action_chunks: list[torch.Tensor] = []
    chunk_debug: list[dict[str, Any]] = []
    for frame_offset in range(0, generated_frames, frame_chunk_size):
        chunk_frame_start = target_start_frame + frame_offset
        chunk_action_source_start = chunk_frame_start - target_start_offset
        action_start = chunk_action_source_start * action_per_frame
        action_end = (chunk_action_source_start + frame_chunk_size) * action_per_frame
        if action_start < 0 or action_end > int(actions.shape[1]):
            raise ValueError(
                "FDM rollout action chunk exceeds available action sequence: "
                f"action_start={action_start}, action_end={action_end}, total_actions={actions.shape[1]}."
            )
        raw_action_chunk = actions[:, action_start:action_end]
        video_condition_chunk = None
        if mode == FdmAblationMode.VIDEO_CONDITIONED_ACTION:
            video_condition_chunk = video_latents[:, :, chunk_frame_start : chunk_frame_start + frame_chunk_size]
        chunk = fdm_rollout.infer_chunk(
            session=session,
            mode=mode,
            raw_action_chunk=raw_action_chunk,
            video_condition_latents=video_condition_chunk,
            seed=seed + frame_offset,
            drop_text_conditioning=drop_text_conditioning,
            proprio_state=_optional_proprio_state_at_frame(sample, chunk_action_source_start, runtime_device),
        )
        predicted_chunks.append(chunk.predicted_latents.detach().cpu())
        if chunk.raw_action_sequence is not None:
            predicted_action_chunks.append(chunk.raw_action_sequence.detach().cpu())
        chunk_debug.append(_small_debug(chunk.debug))
        session = chunk.session

    is_idm_mode = mode == FdmAblationMode.VIDEO_CONDITIONED_ACTION
    predicted_latents = torch.cat(predicted_chunks, dim=2)[:, :, :target_horizon].cpu()
    target_latents = video_latents[:, :, target_start_frame : target_start_frame + target_horizon].detach().cpu()
    latent_mse = None if is_idm_mode else latent_mse_per_frame(predicted_latents, target_latents)
    action_mse = None
    if is_idm_mode:
        if not predicted_action_chunks:
            raise RuntimeError("IDM rollout did not return action predictions.")
        predicted_actions = torch.cat(predicted_action_chunks, dim=1)[
            :, : target_horizon * action_per_frame
        ].cpu()
        target_actions = actions[
            :,
            action_source_start_frame * action_per_frame : (action_source_start_frame + target_horizon)
            * action_per_frame,
        ].detach().cpu()
        action_mse = action_mse_per_frame(
            predicted_actions,
            target_actions,
            action_per_frame=action_per_frame,
        )

    predicted_rgb = None
    target_rgb = None
    if not is_idm_mode:
        predicted_rgb = decode_latent_video(fdm_rollout.runner.pipeline, predicted_latents, decode_device=decode_device)
        target_rgb = decode_latent_video(fdm_rollout.runner.pipeline, target_latents, decode_device=decode_device)
    rgb_mse = None
    rgb_ssim = None
    video_path = None
    if predicted_rgb is not None and target_rgb is not None:
        rgb_mse = rgb_mse_per_frame(predicted_rgb, target_rgb)
        rgb_ssim = simple_ssim_per_frame(predicted_rgb, target_rgb)
        if write_video:
            video_path = video_dir / f"task_{selection.task_rank:02d}_{mode.value}_sample_{selection.sample_index:03d}.mp4"
            write_prediction_video(
                output_path=video_path,
                target_rgb=target_rgb[:target_horizon],
                predicted_rgb=predicted_rgb[:target_horizon],
                title=(
                    f"task={selection.task_rank} mode={mode.value} ep={selection.episode_index} "
                    f"t0={t0} target_start={target_start_frame}"
                ),
                fps=video_fps,
            )

    metric_rows = build_metric_rows(
        selection=mode_selection,
        mode=mode,
        latent_mse=_latent_mse_for_metric_rows(latent_mse=latent_mse, rgb_mse=rgb_mse, action_mse=action_mse),
        rgb_mse=rgb_mse,
        rgb_ssim=rgb_ssim,
        action_mse=action_mse,
    )
    summary = {
        "metric_target": "action" if is_idm_mode else "video",
        "selection_target_start_offset_frames": int(getattr(selection, "target_start_offset_frames", 0)),
        "latent_mse_mean": None if latent_mse is None else float(sum(latent_mse) / len(latent_mse)),
        "rgb_mse_mean": None if rgb_mse is None else float(sum(rgb_mse) / len(rgb_mse)),
        "action_mse_mean": None if action_mse is None else float(sum(action_mse) / len(action_mse)),
        "video_path": None if video_path is None else str(video_path),
        "chunk_debug": chunk_debug,
        "target_start_frame": target_start_frame,
        "target_start_offset_frames": target_start_offset,
        "action_source_start_frame": action_source_start_frame,
    }
    return {"metric_rows": metric_rows, "summary": summary, "video_path": None if video_path is None else str(video_path)}


def _build_fdm_rollout_for_config(
    *,
    config,
    checkpoint_file: Path,
    runtime_device: torch.device,
    runtime_dtype: torch.dtype | None,
) -> JointDenoisingFdmRollout | DualExpertGeneralistDenoisingFdmRollout:
    if _is_dual_expert_policy_config(config):
        from open_wam.pipelines import (
            VariantRolloutRunner,
            build_variant_pipeline_from_config,
        )

        pipeline = build_variant_pipeline_from_config(config)
        _load_pipeline_checkpoint_for_fdm_rollout(
            pipeline,
            checkpoint_file,
            map_location=torch.device("cpu"),
        )
        if runtime_dtype is None:
            pipeline.to(runtime_device)
        else:
            pipeline.to(device=runtime_device, dtype=runtime_dtype)
        pipeline.eval()
        return DualExpertGeneralistDenoisingFdmRollout(VariantRolloutRunner(pipeline))

    if runtime_dtype is not None:
        raise ValueError(
            "`--runtime-dtype` is currently supported only for dual-expert "
            "FDM/IDM evaluation."
        )
    from open_wam.pipelines import build_exact_runtime_runner_from_config

    return JointDenoisingFdmRollout(build_exact_runtime_runner_from_config(config))


def _is_dual_expert_policy_config(config) -> bool:
    raw_name = getattr(config.policy_variant, "name", None)
    return str(getattr(raw_name, "value", raw_name)) == "dual_expert"


def _resolve_action_per_frame(config) -> int:
    raw_action_per_frame = getattr(config.policy_variant, "action_per_frame", None)
    if raw_action_per_frame is not None:
        action_per_frame = int(raw_action_per_frame)
        if action_per_frame <= 0:
            raise ValueError(f"policy_variant.action_per_frame must be positive, got {raw_action_per_frame!r}.")
        return action_per_frame

    action_horizon = int(config.action_decoder.action_horizon)
    frame_chunk_size = int(config.inference.frame_chunk_size)
    if frame_chunk_size <= 0:
        raise ValueError(f"inference.frame_chunk_size must be positive, got {frame_chunk_size}.")
    if action_horizon <= 0:
        raise ValueError(f"action_decoder.action_horizon must be positive, got {action_horizon}.")
    if action_horizon % frame_chunk_size != 0:
        raise ValueError(
            "Cannot infer action_per_frame: action_decoder.action_horizon must be divisible by "
            f"inference.frame_chunk_size, got action_horizon={action_horizon}, "
            f"frame_chunk_size={frame_chunk_size}."
        )
    return action_horizon // frame_chunk_size


def _resolve_runtime_dtype(value: str | None) -> torch.dtype | None:
    if value is None or value == "float32":
        return None
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    raise ValueError(f"Unsupported runtime dtype {value!r}.")


def _target_start_offset_for_config_mode(config, mode: FdmAblationMode) -> int:
    if not _is_dual_expert_policy_config(config):
        return 0
    return 1 if _is_target_only_m5_gjd_mode(mode) else 0


def _target_start_offset_for_rollout_mode(
    fdm_rollout: JointDenoisingFdmRollout | DualExpertGeneralistDenoisingFdmRollout,
    mode: FdmAblationMode,
) -> int:
    if not isinstance(fdm_rollout, DualExpertGeneralistDenoisingFdmRollout):
        return 0
    return 1 if _is_target_only_m5_gjd_mode(mode) else 0


def _is_target_only_m5_gjd_mode(mode: FdmAblationMode) -> bool:
    return FdmAblationMode(mode) in {
        FdmAblationMode.FORCED_ACTION_JOINT_FDM,
        FdmAblationMode.VIDEO_CONDITIONED_ACTION,
    }


def _validate_counterfactual_eval_modes(modes: tuple[FdmAblationMode, ...]) -> None:
    unsupported = [mode.value for mode in modes if not _is_target_only_m5_gjd_mode(mode)]
    if unsupported:
        raise ValueError(
            "Encoded counterfactual dynamics evaluation is target-only and supports only "
            "`forced_action_joint_fdm` and `video_conditioned_action`; got unsupported modes "
            f"{unsupported}."
        )


def _load_pipeline_checkpoint_for_fdm_rollout(
    pipeline: torch.nn.Module,
    checkpoint_path: Path,
    *,
    map_location: torch.device,
) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must be a raw state_dict or a checkpoint with `state_dict`/`model_state_dict`.")
    normalized = {
        (key.removeprefix("pipeline.")): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }
    missing, unexpected = pipeline.load_state_dict(normalized, strict=False)
    if missing:
        print(f"fdm_eval.checkpoint_missing_keys {len(missing)}")
    if unexpected:
        print(f"fdm_eval.checkpoint_unexpected_keys {len(unexpected)}")


def _repair_runtime_config_for_local_eval(
    *,
    config,
    base_config,
    transformer_dir: Path,
    dataset_root: str | None,
    empty_text_embedding_path: str | None,
    reference_assets_device_policy: str | None,
    video_num_inference_steps: int | None,
    action_num_inference_steps: int | None,
):
    data_updates: dict[str, Any] = {}
    resolved_dataset_root = _resolve_existing_path_override(
        explicit=dataset_root,
    )
    if resolved_dataset_root is not None:
        data_updates["local_root"] = str(resolved_dataset_root)
    resolved_empty_embedding = _resolve_existing_path_override(
        explicit=empty_text_embedding_path,
    )
    if resolved_empty_embedding is not None:
        data_updates["empty_text_embedding_path"] = str(resolved_empty_embedding)
    if data_updates:
        config = replace(config, data=replace(config.data, **data_updates))

    backbone_updates: dict[str, Any] = {"transformer_subdir": str(transformer_dir)}
    base_pretrained = Path(str(base_config.backbone.pretrained_model_name_or_path)).expanduser()
    current_pretrained = Path(str(config.backbone.pretrained_model_name_or_path)).expanduser()
    if base_pretrained.exists() and not current_pretrained.exists():
        backbone_updates["pretrained_model_name_or_path"] = str(base_pretrained)
    if reference_assets_device_policy is not None:
        backbone_updates["reference_assets_device_policy"] = reference_assets_device_policy
    config = replace(config, backbone=replace(config.backbone, **backbone_updates))
    inference_updates: dict[str, Any] = {}
    if video_num_inference_steps is not None:
        inference_updates["video_num_inference_steps"] = int(video_num_inference_steps)
    if action_num_inference_steps is not None:
        inference_updates["action_num_inference_steps"] = int(action_num_inference_steps)
    if inference_updates:
        config = replace(config, inference=replace(config.inference, **inference_updates))
    return config


def _resolve_existing_path_override(
    *,
    explicit: str | None,
) -> Path | None:
    if explicit is None:
        return None
    path = Path(explicit).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured override path does not exist: {path}")
    return path


def _optional_batch_tensor(value: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return value.to(device=device, dtype=torch.float32)


def _optional_proprio_state_sequence(
    sample,
    *,
    start_frame: int,
    end_frame: int,
    device: torch.device,
) -> torch.Tensor | None:
    value, mask, source_name, mask_name = _optional_proprio_frame_source(sample)
    if not isinstance(value, torch.Tensor):
        return None
    if value.ndim != 2:
        raise ValueError(
            f"FDM eval {source_name} must have shape [latent_frames, state_dim], "
            f"got {tuple(value.shape)}."
        )
    if value.shape[0] == 0:
        return None
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame <= start_frame:
        return value.new_empty((1, 0, int(value.shape[-1])), dtype=torch.float32).to(device=device)
    indices = torch.arange(start_frame, end_frame, device=value.device, dtype=torch.long)
    indices = indices.clamp(0, int(value.shape[0]) - 1)
    state = value.index_select(0, indices).to(device=device, dtype=torch.float32)
    if isinstance(mask, torch.Tensor):
        if tuple(mask.shape) != tuple(value.shape):
            raise ValueError(
                f"FDM eval {mask_name} must match {source_name} shape, "
                f"got mask={tuple(mask.shape)} and state={tuple(value.shape)}."
            )
        state = state * mask.index_select(0, indices).to(device=device, dtype=torch.float32)
    return state.unsqueeze(0)


def _optional_proprio_state_at_frame(sample, frame_index: int, device: torch.device) -> torch.Tensor | None:
    sequence = _optional_proprio_state_sequence(
        sample,
        start_frame=int(frame_index),
        end_frame=int(frame_index) + 1,
        device=device,
    )
    if sequence is None or int(sequence.shape[1]) <= 0:
        return None
    return sequence[:, -1, :]


def _optional_proprio_frame_source(sample) -> tuple[torch.Tensor | None, torch.Tensor | None, str, str]:
    for source_name, mask_name in (
        ("proprio_context_frames", "proprio_context_frames_mask"),
        ("proprio_context_state", "proprio_context_state_mask"),
    ):
        value = getattr(sample, source_name, None)
        if isinstance(value, torch.Tensor):
            mask = getattr(sample, mask_name, None)
            return value, mask, source_name, mask_name
    return None, None, "proprio_context_frames", "proprio_context_frames_mask"


def _small_debug(debug: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "runtime_mode",
        "generation_frame_start",
        "advance_frame_start",
        "joint_denoise",
        "action_conditioning_mode",
        "generalist_mode_text_token",
        "generalist_mode_text_token_count",
        "forced_action_denoise",
        "forced_clean_action_conditioning",
        "forced_video_conditioning",
        "commit_action_override",
        "returned_action_source",
        "cache_action_source",
        "rollout_window_size",
        "generalist_conditional_history_chunks",
        "proprio_context_token_count",
        "cached_tokens",
        "prediction_tokens",
    }
    return {key: value for key, value in debug.items() if key in keep}


def _sample_report_metadata(sample) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    keys = (
        "dataset_kind",
        "counterfactual_branch",
        "counterfactual_branch_family",
        "counterfactual_branch_strength",
        "counterfactual_branch_is_ood",
        "counterfactual_sample_id",
        "counterfactual_context_id",
        "task_index",
        "episode_index",
        "loss_frame_start",
        "loss_frame_end",
        "target_frame_start",
        "target_frame_end",
        "segment_length_frames",
        "condition_latents_source",
        "proprio_context_frame_count",
    )
    return {key: metadata.get(key) for key in keys if key in metadata}


def _latent_mse_for_metric_rows(
    *,
    latent_mse: list[float] | None,
    rgb_mse: list[float] | None,
    action_mse: list[float] | None,
) -> list[float] | None:
    if latent_mse is None:
        return None
    for values in (rgb_mse, action_mse):
        if values is not None and len(values) != len(latent_mse):
            return None
    return latent_mse


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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline FDM/IDM diagnostics for a joint-denoising policy."
    )
    parser.add_argument("--config", "--cfg", default="configs/experiments/parallel_stream_libero_joint_denoise.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/joint_denoising_fdm")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--horizon-frames", type=int, default=16)
    parser.add_argument(
        "--start-policy",
        choices=[policy.value for policy in FdmStartPolicy],
        default=FdmStartPolicy.EARLY_MIDDLE.value,
    )
    parser.add_argument("--trajectories-per-task", type=int, default=2)
    parser.add_argument("--min-context-frames", type=int, default=4)
    parser.add_argument(
        "--context-window-frames",
        type=int,
        default=16,
        help="Trailing pre-t0 video frames to warm into cache. Use 0 to keep the full prefix.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=8.0)
    parser.add_argument("--runtime-device", default="cuda:0")
    parser.add_argument("--decode-device", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--counterfactual-latent-root",
        default=None,
        help="Optional encoded counterfactual latent root. When set, runs target-only CF FDM/IDM eval.",
    )
    parser.add_argument("--counterfactual-split", default="val")
    parser.add_argument("--empty-text-embedding-path", default=None)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in FdmAblationMode],
        default=None,
        help="Ablation mode to run. Repeat to run multiple modes. Defaults to all modes.",
    )
    parser.add_argument(
        "--replay-status-policy",
        choices=[policy.value for policy in ReplayStatusPolicy],
        default=ReplayStatusPolicy.INCLUDE_ALL.value,
    )
    parser.add_argument("--reference-assets-device-policy", default=None)
    parser.add_argument("--video-num-inference-steps", type=int, default=None)
    parser.add_argument("--action-num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--runtime-dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="Optional inference dtype for dual-expert offline FDM/IDM evaluation.",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Repeatable `section.field=value` config override, applied after checkpoint/runtime repair.",
    )
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = [mode.value for mode in FdmAblationMode]
    if args.context_window_frames is not None and args.context_window_frames <= 0:
        args.context_window_frames = None
    if args.sample_start < 0:
        parser.error("--sample-start must be non-negative.")
    return args
