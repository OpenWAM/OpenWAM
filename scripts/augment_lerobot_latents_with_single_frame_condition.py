#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch

from open_wam.data.latent_temporal import (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
    latent_raw_boundaries,
)

CONDITION_SOURCE_FRAME_POLICY = CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET
from open_wam.data.raw_video import ViewPlacement
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets
from open_wam.utils.latent_filenames import match_latent_window_filename


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.data_root).expanduser().resolve()
    reference_assets_root = Path(args.reference_assets_root).expanduser().resolve()
    latent_root = Path(args.latent_root).expanduser().resolve() if args.latent_root else dataset_root / "latents"
    video_root = Path(args.video_root).expanduser().resolve() if args.video_root else dataset_root / "videos"
    info = _read_json(dataset_root / "meta" / "info.json")
    chunk_size = int(info.get("chunks_size", 1000))

    payload_paths = sorted(latent_root.glob("chunk-*/*/*.pth"))
    if not payload_paths:
        raise FileNotFoundError(f"No latent payloads found under {latent_root}.")
    tasks = _build_payload_tasks(payload_paths)
    if args.max_files is not None:
        tasks = tasks[: int(args.max_files)]
    payload_paths = [path for task in tasks for path in task]

    assets = _load_assets(reference_assets_root, device=torch.device(args.device))
    if args.sanity_check:
        _run_encoding_sanity_checks(
            task=tasks[0],
            video_root=video_root,
            chunk_size=chunk_size,
            assets=assets,
            device=torch.device(args.device),
            batch_size=int(args.batch_size),
            atol=float(args.sanity_atol),
            source_frame_offset=int(args.source_frame_offset),
        )

    updated = 0
    skipped = 0
    for index, task in enumerate(tasks):
        payloads = {
            latent_path.parent.name: _load_payload(latent_path)
            for latent_path in task
        }
        if not args.overwrite and all(
            "condition_latent" in payload
            and int(payload.get("condition_source_frame_offset", 0)) == int(args.source_frame_offset)
            and payload.get("condition_source_frame_policy") == CONDITION_SOURCE_FRAME_POLICY
            for payload in payloads.values()
        ):
            skipped += len(task)
            continue

        if _is_libero_task(task):
            encoded = _encode_libero_condition_latents_for_task(
                task=task,
                payloads=payloads,
                video_root=video_root,
                chunk_size=chunk_size,
                assets=assets,
                device=torch.device(args.device),
                output_dtype_name=args.output_dtype,
                batch_size=int(args.batch_size),
                source_frame_offset=int(args.source_frame_offset),
            )
        else:
            latent_path = task[0]
            camera_name = latent_path.parent.name
            payload = payloads[camera_name]
            video_path = _source_video_path(
                video_root=video_root,
                episode_index=_episode_index_from_latent_path(latent_path),
                camera_name=camera_name,
                chunk_size=chunk_size,
            )
            encoded = {
                camera_name: _encode_condition_latents(
                    payload=payload,
                    video_path=video_path,
                    assets=assets,
                    device=torch.device(args.device),
                    output_dtype=_resolve_output_dtype(args.output_dtype, payload),
                    batch_size=int(args.batch_size),
                    source_frame_offset=int(args.source_frame_offset),
                )
            }

        task_reports = []
        for latent_path in task:
            camera_name = latent_path.parent.name
            payload = payloads[camera_name]
            condition = encoded[camera_name]
            task_reports.append(
                {
                    "path": str(latent_path),
                    "condition_latent_shape": list(condition.shape),
                    "condition_latent_dtype": str(condition.dtype).replace("torch.", ""),
                }
            )
        if args.dry_run:
            print(json.dumps({"task": task_reports}))
        else:
            for latent_path in task:
                camera_name = latent_path.parent.name
                payload = payloads[camera_name]
                payload["condition_latent"] = encoded[camera_name].contiguous()
                payload["condition_source_frame_offset"] = int(args.source_frame_offset)
                payload["condition_source_frame_policy"] = CONDITION_SOURCE_FRAME_POLICY
                _save_payload_atomic(payload, latent_path)
        updated += len(task)
        if args.log_every > 0 and (index + 1) % int(args.log_every) == 0:
            print(f"[progress] tasks={index + 1} updated={updated} skipped={skipped}", flush=True)

    print(json.dumps({"updated": updated, "skipped": skipped, "total_seen": len(payload_paths), "tasks": len(tasks)}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment LeRobot local latent payloads with `condition_latent`: "
            "single-frame Wan VAE latents encoded from each materialized context slot's "
            "rollout-parity source frame."
        )
    )
    parser.add_argument("--data-root", required=True, help="LeRobot local dataset root containing meta/data/videos/latents.")
    parser.add_argument("--reference-assets-root", required=True, help="LingBot/Wan asset root containing the VAE.")
    parser.add_argument("--latent-root", default=None, help="Optional latent root override. Defaults to DATA_ROOT/latents.")
    parser.add_argument("--video-root", default=None, help="Optional video root override. Defaults to DATA_ROOT/videos.")
    parser.add_argument("--device", default="cuda:0", help="Device used for VAE encoding.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Raw frame buckets encoded per VAE call. Default 1 matches exact rollout most closely; "
            "larger values should be gated by --sanity-check."
        ),
    )
    parser.add_argument("--output-dtype", default="match", choices=("match", "float32", "bfloat16", "float16"))
    parser.add_argument(
        "--source-frame-offset",
        type=int,
        default=0,
        help=(
            "Raw-frame offset applied to each latent bucket's source-span start before single-frame VAE encoding. "
            "Use -1 for previous-frame conditioning."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute condition_latent if already present.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned writes without modifying payloads.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional task-group limit for smoke testing. A paired LIBERO agentview/wrist item counts as one task.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help=(
            "Before processing, verify single-camera condition encoding matches the LIBERO canonical placement path "
            "and that batched single-frame encoding is independent across batch elements."
        ),
    )
    parser.add_argument(
        "--sanity-atol",
        type=float,
        default=1e-4,
        help="Maximum allowed absolute difference for --sanity-check comparisons.",
    )
    return parser.parse_args()


def _load_assets(reference_assets_root: Path, *, device: torch.device) -> LingbotReferenceAssets:
    config = LingbotCompatibleVideoBackboneConfig(
        pretrained_model_name_or_path=str(reference_assets_root),
        load_wan_vae_frontend=True,
        load_text_conditioning=False,
    )
    assets = LingbotReferenceAssets.maybe_load(config)
    if not assets.has_vae:
        raise RuntimeError(f"No Wan VAE could be loaded from {reference_assets_root}.")
    assets._ensure_vae_runtime_device(device)
    return assets


def _encode_condition_latents(
    *,
    payload: dict[str, Any],
    video_path: Path,
    assets: LingbotReferenceAssets,
    device: torch.device,
    output_dtype: torch.dtype,
    batch_size: int,
    source_frame_offset: int = 0,
) -> torch.Tensor:
    latent_num_frames = int(payload["latent_num_frames"])
    latent_height = int(payload["latent_height"])
    latent_width = int(payload["latent_width"])
    frame_ids = [int(value) for value in payload.get("frame_ids", [])]
    if not frame_ids:
        video_num_frames = int(payload.get("video_num_frames", 0))
        if video_num_frames <= 0:
            raise ValueError(f"Payload for {video_path} has neither frame_ids nor positive video_num_frames.")
        frame_ids = list(range(video_num_frames))
    source_indices = _condition_source_frame_indices(
        frame_ids=frame_ids,
        latent_num_frames=latent_num_frames,
        source_frame_offset=source_frame_offset,
    )

    reader = imageio.get_reader(str(video_path))
    encoded_chunks: list[torch.Tensor] = []
    try:
        for start in range(0, len(source_indices), max(1, int(batch_size))):
            batch_indices = source_indices[start : start + max(1, int(batch_size))]
            frames = [_read_video_frame(reader, frame_index) for frame_index in batch_indices]
            video = _frames_to_video_tensor(frames, device=device)
            latents = assets.encode_video(video, placements=None, reset_cache=True)
            if tuple(latents.shape[-2:]) != (latent_height, latent_width):
                raise ValueError(
                    "Encoded condition latent geometry does not match payload metadata: "
                    f"encoded={tuple(latents.shape)}, expected latent H/W=({latent_height}, {latent_width})."
                )
            encoded_chunks.append(latents[:, :, 0].to(device="cpu", dtype=output_dtype))
    finally:
        reader.close()

    per_frame = torch.cat(encoded_chunks, dim=0)
    return per_frame.permute(0, 2, 3, 1).reshape(latent_num_frames * latent_height * latent_width, -1)


def _encode_libero_condition_latents_for_task(
    *,
    task: tuple[Path, ...],
    payloads: dict[str, dict[str, Any]],
    video_root: Path,
    chunk_size: int,
    assets: LingbotReferenceAssets,
    device: torch.device,
    output_dtype_name: str,
    batch_size: int,
    source_frame_offset: int = 0,
) -> dict[str, torch.Tensor]:
    if len(task) != 2:
        raise ValueError(f"Expected paired LIBERO task, got {task}.")
    camera_paths = {path.parent.name: path for path in task}
    agent_camera = _resolve_libero_camera_name(camera_paths, slot=0)
    wrist_camera = _resolve_libero_camera_name(camera_paths, slot=1)
    agent_payload = payloads[agent_camera]
    wrist_payload = payloads[wrist_camera]
    _validate_paired_payloads(agent_payload, wrist_payload, task=task)
    latent_num_frames = int(agent_payload["latent_num_frames"])
    latent_height = int(agent_payload["latent_height"])
    latent_width = int(agent_payload["latent_width"])
    source_indices = _source_indices_from_payload(
        agent_payload,
        source_frame_offset=source_frame_offset,
    )
    episode_index = _episode_index_from_latent_path(camera_paths[agent_camera])
    agent_video_path = _source_video_path(
        video_root=video_root,
        episode_index=episode_index,
        camera_name=agent_camera,
        chunk_size=chunk_size,
    )
    wrist_video_path = _source_video_path(
        video_root=video_root,
        episode_index=episode_index,
        camera_name=wrist_camera,
        chunk_size=chunk_size,
    )
    agent_reader = imageio.get_reader(str(agent_video_path))
    wrist_reader = imageio.get_reader(str(wrist_video_path))
    encoded_chunks: list[torch.Tensor] = []
    try:
        step = max(1, int(batch_size))
        for start in range(0, len(source_indices), step):
            batch_indices = source_indices[start : start + step]
            agent_frames = [_read_video_frame(agent_reader, frame_index) for frame_index in batch_indices]
            wrist_frames = [_read_video_frame(wrist_reader, frame_index) for frame_index in batch_indices]
            agent_video = _frames_to_video_tensor(agent_frames, device=device)
            wrist_video = _frames_to_video_tensor(wrist_frames, device=device)
            canonical = torch.cat([agent_video, wrist_video], dim=-1)
            latents = assets.encode_video(canonical, placements=_libero_placements(), reset_cache=True)
            if tuple(latents.shape[-2:]) != (latent_height, latent_width * 2):
                raise ValueError(
                    "Encoded LIBERO condition latent geometry does not match paired payload metadata: "
                    f"encoded={tuple(latents.shape)}, expected latent H/W=({latent_height}, {latent_width * 2})."
                )
            encoded_chunks.append(latents[:, :, 0].to(device="cpu"))
    finally:
        agent_reader.close()
        wrist_reader.close()

    per_frame = torch.cat(encoded_chunks, dim=0)
    agent = per_frame[..., :latent_width]
    wrist = per_frame[..., latent_width : latent_width * 2]
    return {
        agent_camera: _flatten_condition_latents(
            agent,
            output_dtype=_resolve_output_dtype(output_dtype_name, agent_payload),
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        ),
        wrist_camera: _flatten_condition_latents(
            wrist,
            output_dtype=_resolve_output_dtype(output_dtype_name, wrist_payload),
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        ),
    }


def _flatten_condition_latents(
    latents: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    if tuple(latents.shape) != (latent_num_frames, latents.shape[1], latent_height, latent_width):
        raise ValueError(
            "Condition latent shape does not match payload metadata: "
            f"shape={tuple(latents.shape)}, expected frames/H/W=({latent_num_frames}, {latent_height}, {latent_width})."
        )
    return latents.to(dtype=output_dtype).permute(0, 2, 3, 1).reshape(latent_num_frames * latent_height * latent_width, -1)


def _run_encoding_sanity_checks(
    *,
    task: tuple[Path, ...],
    video_root: Path,
    chunk_size: int,
    assets: LingbotReferenceAssets,
    device: torch.device,
    batch_size: int,
    atol: float,
    source_frame_offset: int,
) -> None:
    if not _is_libero_task(task):
        _run_single_camera_encoding_sanity_check(
            latent_path=task[0],
            video_root=video_root,
            chunk_size=chunk_size,
            assets=assets,
            device=device,
            batch_size=batch_size,
            atol=atol,
            source_frame_offset=source_frame_offset,
        )
        return
    _run_libero_pair_encoding_sanity_check(
        task=task,
        video_root=video_root,
        chunk_size=chunk_size,
        assets=assets,
        device=device,
        batch_size=batch_size,
        atol=atol,
        source_frame_offset=source_frame_offset,
    )


def _run_single_camera_encoding_sanity_check(
    *,
    latent_path: Path,
    video_root: Path,
    chunk_size: int,
    assets: LingbotReferenceAssets,
    device: torch.device,
    batch_size: int,
    atol: float,
    source_frame_offset: int,
) -> None:
    payload = _load_payload(latent_path)
    source_frame = _source_indices_from_payload(payload, source_frame_offset=source_frame_offset)[0]
    episode_index = _episode_index_from_latent_path(latent_path)
    camera_name = latent_path.parent.name
    video_path = _source_video_path(
        video_root=video_root,
        episode_index=episode_index,
        camera_name=camera_name,
        chunk_size=chunk_size,
    )
    frame = _read_single_video_frame(video_path, source_frame)
    single_view = _frames_to_video_tensor([frame], device=device)

    direct = assets.encode_video(single_view, placements=None, reset_cache=True)
    batch_max_diff = _batch_encoding_max_diff(
        single_view,
        placements=None,
        assets=assets,
        batch_size=batch_size,
    )

    report = {
        "path": str(latent_path),
        "video_path": str(video_path),
        "camera_name": camera_name,
        "mode": "single_camera",
        "source_frame": source_frame,
        "batch_max_abs_diff": batch_max_diff,
        "atol": atol,
        "direct_shape": list(direct.shape),
        "batch_size": int(batch_size),
    }
    print(f"[sanity] {json.dumps(report, sort_keys=True)}", flush=True)
    failures = {
        name: value
        for name, value in {"batch_max_abs_diff": batch_max_diff}.items()
        if value > atol
    }
    if failures:
        raise RuntimeError(f"Condition latent encoding sanity check failed: {failures}")


def _run_libero_pair_encoding_sanity_check(
    *,
    task: tuple[Path, ...],
    video_root: Path,
    chunk_size: int,
    assets: LingbotReferenceAssets,
    device: torch.device,
    batch_size: int,
    atol: float,
    source_frame_offset: int,
) -> None:
    payloads = {path.parent.name: _load_payload(path) for path in task}
    camera_paths = {path.parent.name: path for path in task}
    agent_camera = _resolve_libero_camera_name(camera_paths, slot=0)
    wrist_camera = _resolve_libero_camera_name(camera_paths, slot=1)
    _validate_paired_payloads(payloads[agent_camera], payloads[wrist_camera], task=task)
    source_indices = _source_indices_from_payload(payloads[agent_camera], source_frame_offset=source_frame_offset)
    zero_offset_indices = _source_indices_from_payload(payloads[agent_camera], source_frame_offset=0)
    source_frame = source_indices[0]
    episode_index = _episode_index_from_latent_path(camera_paths[agent_camera])
    agent_frame = _read_single_video_frame(
        _source_video_path(
            video_root=video_root,
            episode_index=episode_index,
            camera_name=agent_camera,
            chunk_size=chunk_size,
        ),
        source_frame,
    )
    wrist_frame = _read_single_video_frame(
        _source_video_path(
            video_root=video_root,
            episode_index=episode_index,
            camera_name=wrist_camera,
            chunk_size=chunk_size,
        ),
        source_frame,
    )
    agent_video = _frames_to_video_tensor([agent_frame], device=device)
    wrist_video = _frames_to_video_tensor([wrist_frame], device=device)
    canonical = torch.cat([agent_video, wrist_video], dim=-1)

    single = assets.encode_video(canonical, placements=_libero_placements(), reset_cache=True)
    batch_max_diff = _batch_encoding_max_diff(
        canonical,
        placements=_libero_placements(),
        assets=assets,
        batch_size=batch_size,
    )
    report = {
        "task": [str(path) for path in task],
        "mode": "libero_pair",
        "source_frame": source_frame,
        "batch_max_abs_diff": batch_max_diff,
        "atol": atol,
        "single_shape": list(single.shape),
        "batch_size": int(batch_size),
        "source_indices_preview": source_indices[:8],
        "zero_offset_indices_preview": zero_offset_indices[:8],
    }
    print(f"[sanity] {json.dumps(report, sort_keys=True)}", flush=True)
    if batch_max_diff > atol:
        raise RuntimeError(f"Condition latent encoding sanity check failed: {{'batch_max_abs_diff': {batch_max_diff}}}")


def _batch_encoding_max_diff(
    video: torch.Tensor,
    *,
    placements: tuple[ViewPlacement, ...] | None,
    assets: LingbotReferenceAssets,
    batch_size: int,
) -> float:
    if int(batch_size) <= 1:
        return 0.0
    repeat_count = min(int(batch_size), 8)
    single = assets.encode_video(video, placements=placements, reset_cache=True)
    batched_video = video.expand(repeat_count, -1, -1, -1, -1).contiguous()
    batched = assets.encode_video(batched_video, placements=placements, reset_cache=True)
    return _max_abs_diff(single, batched[:1])


def _build_payload_tasks(payload_paths: list[Path]) -> list[tuple[Path, ...]]:
    grouped: dict[tuple[Path, str], list[Path]] = {}
    for path in payload_paths:
        match = match_latent_window_filename(path.name)
        if match is None:
            raise ValueError(f"Could not parse latent filename: {path}")
        group_key = (path.parent.parent, path.name)
        grouped.setdefault(group_key, []).append(path)
    tasks: list[tuple[Path, ...]] = []
    for _, paths in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        paths = sorted(paths, key=lambda item: item.parent.name)
        if len(paths) >= 2 and _paths_are_libero_pair(paths):
            tasks.append(tuple(paths))
        else:
            tasks.extend((path,) for path in paths)
    return tasks


def _paths_are_libero_pair(paths: list[Path]) -> bool:
    camera_names = {path.parent.name for path in paths}
    return any(_is_agentview_camera(name) for name in camera_names) and any(_is_wrist_camera(name) for name in camera_names)


def _is_libero_task(task: tuple[Path, ...]) -> bool:
    return len(task) == 2 and _paths_are_libero_pair(list(task))


def _resolve_libero_camera_name(camera_paths: dict[str, Path], *, slot: int) -> str:
    predicate = _is_agentview_camera if slot == 0 else _is_wrist_camera
    matches = [name for name in camera_paths if predicate(name)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one LIBERO camera for slot {slot}, got {matches}.")
    return matches[0]


def _is_agentview_camera(camera_name: str) -> bool:
    return "agentview" in camera_name or camera_name.endswith(".image") or camera_name == "image"


def _is_wrist_camera(camera_name: str) -> bool:
    return "eye_in_hand" in camera_name or "wrist" in camera_name


def _load_payload(latent_path: Path) -> dict[str, Any]:
    payload = torch.load(latent_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected latent payload dict at {latent_path}, got {type(payload).__name__}.")
    return payload


def _source_indices_from_payload(payload: dict[str, Any], *, source_frame_offset: int = 0) -> list[int]:
    frame_ids = [int(value) for value in payload.get("frame_ids", [])]
    if not frame_ids:
        video_num_frames = int(payload.get("video_num_frames", 0))
        if video_num_frames <= 0:
            raise ValueError("Payload has neither frame_ids nor positive video_num_frames.")
        frame_ids = list(range(video_num_frames))
    return _condition_source_frame_indices(
        frame_ids=frame_ids,
        latent_num_frames=int(payload["latent_num_frames"]),
        source_frame_offset=source_frame_offset,
    )


def _validate_paired_payloads(lhs: dict[str, Any], rhs: dict[str, Any], *, task: tuple[Path, ...]) -> None:
    keys = ("latent_num_frames", "latent_height", "latent_width", "video_num_frames")
    mismatches = {
        key: (lhs.get(key), rhs.get(key))
        for key in keys
        if lhs.get(key) != rhs.get(key)
    }
    if mismatches:
        raise ValueError(f"Paired LIBERO payload metadata mismatch for {task}: {mismatches}")
    if list(lhs.get("frame_ids", [])) != list(rhs.get("frame_ids", [])):
        raise ValueError(f"Paired LIBERO payload frame_ids mismatch for {task}.")


def _source_video_path(*, video_root: Path, episode_index: int, camera_name: str, chunk_size: int) -> Path:
    video_path = (
        video_root
        / f"chunk-{episode_index // chunk_size:03d}"
        / camera_name
        / f"episode_{episode_index:06d}.mp4"
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Missing source video: {video_path}")
    return video_path


def _read_single_video_frame(video_path: Path, frame_index: int) -> Any:
    reader = imageio.get_reader(str(video_path))
    try:
        return _read_video_frame(reader, frame_index)
    finally:
        reader.close()


def _libero_canonical_video_from_single_view(single_view: torch.Tensor) -> torch.Tensor:
    if single_view.ndim != 5:
        raise ValueError(f"Expected [B, C, T, H, W] single-view video, got {tuple(single_view.shape)}.")
    if tuple(single_view.shape[-2:]) != (128, 128):
        raise ValueError(f"LIBERO sanity check expects 128x128 views, got {tuple(single_view.shape[-2:])}.")
    return torch.cat([single_view, single_view], dim=-1)


def _libero_placements() -> tuple[ViewPlacement, ...]:
    return (
        ViewPlacement(
            source_name="observation.images.agentview_rgb",
            canonical_name="image",
            top=0,
            left=0,
            height=128,
            width=128,
        ),
        ViewPlacement(
            source_name="observation.images.eye_in_hand_rgb",
            canonical_name="wrist_image",
            top=0,
            left=128,
            height=128,
            width=128,
        ),
    )


def _libero_camera_slot(camera_name: str) -> int:
    if "agentview" in camera_name or camera_name.endswith(".image") or camera_name == "image":
        return 0
    if "eye_in_hand" in camera_name or "wrist" in camera_name:
        return 1
    raise ValueError(f"Could not map LIBERO camera name to canonical slot: {camera_name}")


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs.float() - rhs.float()).abs().max().item())


def _save_payload_atomic(payload: dict[str, Any], latent_path: Path) -> None:
    tmp_path = latent_path.with_name(f"{latent_path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(latent_path)


def _condition_source_frame_indices(
    *,
    frame_ids: list[int],
    latent_num_frames: int,
    source_frame_offset: int = 0,
) -> list[int]:
    """Return rollout-parity condition frames for each materialized context slot.

    In strict fixed-128 training, materialized condition slot ``j`` is used as
    the one-frame context immediately before target latent slot ``j + 1``.
    Therefore the source frame for condition slot ``j`` is computed from the
    *next* latent raw-span boundary. With Wan stride-4 and
    ``source_frame_offset=-1``, this yields the previous raw frame before the
    next target span, e.g. ``[0, 4, 8, 12]`` for anchors
    ``[0, 4, 8, 12]``.
    """

    if latent_num_frames <= 0:
        raise ValueError(f"Expected positive latent_num_frames, got {latent_num_frames}.")
    if not frame_ids:
        raise ValueError("Expected non-empty frame_ids.")
    raw_count = len(frame_ids)
    boundaries = latent_raw_boundaries(
        raw_frame_count=raw_count,
        latent_num_frames=latent_num_frames,
        layout="wan_causal_stride4",
    )
    indices: list[int] = []
    for latent_index in range(latent_num_frames):
        boundary_index = min(int(latent_index) + 1, len(boundaries) - 1)
        raw_position = min(int(boundaries[boundary_index]), raw_count - 1)
        raw_position = max(0, min(raw_position + int(source_frame_offset), raw_count - 1))
        indices.append(int(frame_ids[raw_position]))
    return indices


def _read_video_frame(reader: Any, frame_index: int) -> Any:
    try:
        return reader.get_data(int(frame_index))
    except IndexError:
        metadata = reader.get_meta_data()
        frame_count = int(metadata.get("nframes") or frame_index + 1)
        return reader.get_data(max(0, frame_count - 1))


def _frames_to_video_tensor(frames: list[Any], *, device: torch.device) -> torch.Tensor:
    tensors = []
    for frame in frames:
        tensor = torch.as_tensor(frame)
        if tensor.ndim != 3 or tensor.shape[-1] < 3:
            raise ValueError(f"Expected RGB frame with shape [H, W, C], got {tuple(tensor.shape)}.")
        tensors.append(tensor[..., :3].permute(2, 0, 1).float() / 255.0)
    return torch.stack(tensors, dim=0).unsqueeze(2).to(device=device)


def _resolve_output_dtype(name: str, payload: dict[str, Any]) -> torch.dtype:
    if name == "match":
        latent = payload["latent"]
        if isinstance(latent, torch.Tensor):
            return latent.dtype
        return torch.float32
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _episode_index_from_latent_path(path: Path) -> int:
    match = match_latent_window_filename(path.name)
    if match is None:
        raise ValueError(f"Could not parse latent filename: {path}")
    return int(match.group("episode"))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
