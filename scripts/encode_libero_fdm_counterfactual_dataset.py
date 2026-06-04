#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from open_wam.ablations.joint_denoising_fdm.cli import (
    DEFAULT_CHECKPOINT,
    _repair_runtime_config_for_local_eval,
)
from open_wam.data.latent_temporal import (
    CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET,
    latent_raw_boundaries,
)
from open_wam.data.raw_video import ViewPlacement
from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets
from open_wam.utils import (
    load_experiment_config,
    merge_runtime_config_from_checkpoint,
    resolve_checkpoint_file,
    resolve_transformer_dir_override,
)


DEFAULT_DATASET_ROOT = (
    "/path/to/private-resource"
    "libero10_fdm_counterfactual_coverage_10000_h32_ctx16_random_t0_seed0_sharded"
)
LIBERO_OBS_KEYS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)
CONDITION_SOURCE_FRAME_POLICY = CONDITION_SOURCE_FRAME_POLICY_NEXT_LATENT_SOURCE_OFFSET


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else dataset_root / "encoded_latents"
    )
    shard_roots = _resolve_shards(dataset_root, args.shards)
    if not args.overwrite:
        _validate_no_existing_encoded_outputs(
            output_root=output_root,
            shard_roots=shard_roots,
            max_contexts=args.max_contexts,
            max_samples=args.max_samples,
        )
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_encoder_config(args)
    assets = LingbotReferenceAssets.maybe_load(config.backbone)
    if not assets.has_vae:
        raise RuntimeError("Counterfactual dataset encoding requires Wan/LingBot VAE assets.")

    device = torch.device(args.device)
    output_dtype = _parse_output_dtype(args.output_dtype)
    shard_roots = _resolve_shards(dataset_root, args.shards)
    manifest = {
        "dataset_kind": "libero10_counterfactual_fdm_encoded_latents",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(resolve_checkpoint_file(args.checkpoint)),
        "device": str(device),
        "output_dtype": str(output_dtype).replace("torch.", ""),
        "batch_size": int(args.batch_size),
        "condition_latents": not bool(args.skip_condition_latents),
        "condition_source_frame_offset": int(args.condition_source_frame_offset),
        "condition_source_frame_policy": CONDITION_SOURCE_FRAME_POLICY,
        "condition_batch_size": int(args.condition_batch_size),
        "shards": [shard.name for shard in shard_roots],
        "source_summary": _read_optional_json(dataset_root / "aggregate_summary.json")
        or _read_optional_json(dataset_root / "summary.json"),
    }
    _write_json(output_root / "manifest.json", manifest)

    context_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for shard_root in shard_roots:
        shard_output = output_root / shard_root.name
        shard_output.mkdir(parents=True, exist_ok=True)
        shard_context_rows = _encode_shard_contexts(
            shard_root=shard_root,
            shard_output=shard_output,
            assets=assets,
            device=device,
            output_dtype=output_dtype,
            batch_size=args.batch_size,
            condition_source_frame_offset=None if args.skip_condition_latents else int(args.condition_source_frame_offset),
            condition_batch_size=int(args.condition_batch_size),
            overwrite=bool(args.overwrite),
            max_contexts=args.max_contexts,
        )
        context_rows.extend(shard_context_rows)

        shard_transition_rows = _encode_shard_samples(
            shard_root=shard_root,
            shard_output=shard_output,
            assets=assets,
            device=device,
            output_dtype=output_dtype,
            batch_size=args.batch_size,
            condition_source_frame_offset=None if args.skip_condition_latents else int(args.condition_source_frame_offset),
            condition_batch_size=int(args.condition_batch_size),
            overwrite=bool(args.overwrite),
            max_samples=args.max_samples,
        )
        transition_rows.extend(shard_transition_rows)

        _write_jsonl(shard_output / "metadata" / "encoded_contexts.jsonl", shard_context_rows)
        _write_jsonl(shard_output / "metadata" / "encoded_transitions.jsonl", shard_transition_rows)
        _write_json(
            shard_output / "summary.json",
            {
                "shard": shard_root.name,
                "context_count": len(shard_context_rows),
                "transition_count": len(shard_transition_rows),
                "output_root": str(shard_output),
            },
        )
        _write_json(
            output_root / "latest_progress.json",
            {
                "completed_shards": len({row["shard"] for row in context_rows + transition_rows}),
                "context_count": len(context_rows),
                "transition_count": len(transition_rows),
            },
        )

    summary = {
        "dataset_kind": manifest["dataset_kind"],
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "context_count": len(context_rows),
        "transition_count": len(transition_rows),
        "shard_count": len(shard_roots),
        "encoded_contexts_jsonl": str(output_root / "metadata" / "encoded_contexts.jsonl"),
        "encoded_transitions_jsonl": str(output_root / "metadata" / "encoded_transitions.jsonl"),
        "manifest": str(output_root / "manifest.json"),
    }
    _write_jsonl(output_root / "metadata" / "encoded_contexts.jsonl", context_rows)
    _write_jsonl(output_root / "metadata" / "encoded_transitions.jsonl", transition_rows)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _load_encoder_config(args: argparse.Namespace):
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_file = resolve_checkpoint_file(args.checkpoint)
    checkpoint_dir = checkpoint_file.parent
    transformer_dir = resolve_transformer_dir_override(checkpoint_dir)
    base_config = load_experiment_config(config_path)
    config, _ = merge_runtime_config_from_checkpoint(base_config, checkpoint_file)
    config = _repair_runtime_config_for_local_eval(
        config=config,
        base_config=base_config,
        transformer_dir=transformer_dir,
        dataset_root=None,
        empty_text_embedding_path=None,
        reference_assets_device_policy=args.reference_assets_device_policy,
        video_num_inference_steps=None,
        action_num_inference_steps=None,
    )
    # Encoding only needs the VAE. Avoid loading the text encoder for a 10k
    # transition preprocessing job.
    return replace(config, backbone=replace(config.backbone, load_text_conditioning=False))


def _validate_no_existing_encoded_outputs(
    *,
    output_root: Path,
    shard_roots: list[Path],
    max_contexts: int | None,
    max_samples: int | None,
) -> None:
    existing: list[Path] = []
    for shard_root in shard_roots:
        shard_output = output_root / shard_root.name
        context_rows = _read_jsonl(shard_root / "metadata" / "contexts.jsonl")
        if max_contexts is not None:
            context_rows = context_rows[: int(max_contexts)]
        for row in context_rows:
            path = shard_output / "contexts" / f"context_{int(row['context_id']):06d}_latents.pt"
            if path.exists():
                existing.append(path)

        transition_rows = _read_jsonl(shard_root / "metadata" / "transitions.jsonl")
        if max_samples is not None:
            transition_rows = transition_rows[: int(max_samples)]
        for row in transition_rows:
            path = shard_output / "samples" / f"sample_{int(row['sample_id']):06d}_latents.pt"
            if path.exists():
                existing.append(path)

    if existing:
        preview = ", ".join(str(path) for path in existing[:5])
        suffix = "" if len(existing) <= 5 else f", ... ({len(existing)} total)"
        raise FileExistsError(
            "Encoded counterfactual outputs already exist and --overwrite was not set. "
            "Refusing to recompute metadata against stale latent files: "
            f"{preview}{suffix}"
        )


def _encode_shard_contexts(
    *,
    shard_root: Path,
    shard_output: Path,
    assets: LingbotReferenceAssets,
    device: torch.device,
    output_dtype: torch.dtype,
    batch_size: int,
    condition_source_frame_offset: int | None,
    condition_batch_size: int,
    overwrite: bool,
    max_contexts: int | None,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(shard_root / "metadata" / "contexts.jsonl")
    if max_contexts is not None:
        rows = rows[: int(max_contexts)]
    output_dir = shard_output / "contexts"
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_rows: list[dict[str, Any]] = []
    for batch in _batched(rows, batch_size):
        rgb_batch = []
        out_paths = []
        for row in batch:
            context_path = shard_root / str(row["context_path"])
            payload = np.load(context_path)
            rgb_batch.append(_load_counterfactual_rgb(payload, legacy_key="context_rgb"))
            out_paths.append(output_dir / f"context_{int(row['context_id']):06d}_latents.pt")
        latents = _encode_rgb_batch(
            assets,
            rgb_batch,
            device=device,
            output_dtype=output_dtype,
        )
        for batch_index, (row, out_path, latent) in enumerate(zip(batch, out_paths, latents, strict=True)):
            rgb = rgb_batch[batch_index]
            condition_latents = (
                _encode_condition_latents_for_rgb(
                    assets,
                    rgb,
                    latent_frames=int(latent.shape[1]),
                    source_frame_offset=int(condition_source_frame_offset),
                    condition_batch_size=int(condition_batch_size),
                    device=device,
                    output_dtype=output_dtype,
                )
                if condition_source_frame_offset is not None
                else None
            )
            if overwrite or not out_path.exists():
                payload = {
                    "context_id": int(row["context_id"]),
                    "video_latents": latent.contiguous(),
                    "source_context_path": str(shard_root / str(row["context_path"])),
                }
                if condition_latents is not None:
                    payload.update(
                        {
                            "condition_video_latents": condition_latents.contiguous(),
                            "condition_source_frame_offset": int(condition_source_frame_offset),
                            "condition_source_frame_policy": CONDITION_SOURCE_FRAME_POLICY,
                        }
                    )
                torch.save(payload, out_path)
            condition_shape = None if condition_latents is None else list(condition_latents.shape)
            encoded_rows.append(
                {
                    **row,
                    "shard": shard_root.name,
                    "context_latent_path": str(out_path.relative_to(shard_output)),
                    "context_video_latent_shape": list(latent.shape),
                    "context_video_latent_dtype": str(latent.dtype).replace("torch.", ""),
                    "condition_video_latent_shape": condition_shape,
                }
            )
    return encoded_rows


def _encode_shard_samples(
    *,
    shard_root: Path,
    shard_output: Path,
    assets: LingbotReferenceAssets,
    device: torch.device,
    output_dtype: torch.dtype,
    batch_size: int,
    condition_source_frame_offset: int | None,
    condition_batch_size: int,
    overwrite: bool,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(shard_root / "metadata" / "transitions.jsonl")
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    output_dir = shard_output / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_rows: list[dict[str, Any]] = []
    for batch in _batched(rows, batch_size):
        rgb_batch = []
        out_paths = []
        for row in batch:
            sample_path = shard_root / str(row["sample_path"])
            payload = np.load(sample_path)
            rgb_batch.append(_load_counterfactual_rgb(payload, legacy_key="target_rgb"))
            out_paths.append(output_dir / f"sample_{int(row['sample_id']):06d}_latents.pt")
        latents = _encode_rgb_batch(
            assets,
            rgb_batch,
            device=device,
            output_dtype=output_dtype,
        )
        for batch_index, (row, out_path, latent) in enumerate(zip(batch, out_paths, latents, strict=True)):
            rgb = rgb_batch[batch_index]
            condition_latents = (
                _encode_condition_latents_for_rgb(
                    assets,
                    rgb,
                    latent_frames=int(latent.shape[1]),
                    source_frame_offset=int(condition_source_frame_offset),
                    condition_batch_size=int(condition_batch_size),
                    device=device,
                    output_dtype=output_dtype,
                )
                if condition_source_frame_offset is not None
                else None
            )
            if overwrite or not out_path.exists():
                payload = {
                    "sample_id": int(row["sample_id"]),
                    "context_id": int(row["context_id"]),
                    "target_video_latents": latent.contiguous(),
                    "source_sample_path": str(shard_root / str(row["sample_path"])),
                }
                if condition_latents is not None:
                    payload.update(
                        {
                            "target_condition_video_latents": condition_latents.contiguous(),
                            "condition_source_frame_offset": int(condition_source_frame_offset),
                            "condition_source_frame_policy": CONDITION_SOURCE_FRAME_POLICY,
                        }
                    )
                torch.save(payload, out_path)
            condition_shape = None if condition_latents is None else list(condition_latents.shape)
            encoded_rows.append(
                {
                    **row,
                    "shard": shard_root.name,
                    "target_latent_path": str(out_path.relative_to(shard_output)),
                    "target_video_latent_shape": list(latent.shape),
                    "target_video_latent_dtype": str(latent.dtype).replace("torch.", ""),
                    "target_condition_video_latent_shape": condition_shape,
                }
            )
    return encoded_rows


def _encode_rgb_batch(
    assets: LingbotReferenceAssets,
    rgb_batch: list[np.ndarray],
    *,
    device: torch.device,
    output_dtype: torch.dtype,
) -> list[torch.Tensor]:
    if not rgb_batch:
        return []
    video = _stack_libero_side_by_side_rgb(rgb_batch, device=device)
    latents = _encode_libero_side_by_side_video(assets, video, device=device)
    latents = latents.detach().to(device="cpu", dtype=output_dtype)
    return [latents[index] for index in range(latents.shape[0])]


def _encode_condition_latents_for_rgb(
    assets: LingbotReferenceAssets,
    rgb: np.ndarray,
    *,
    latent_frames: int,
    source_frame_offset: int,
    condition_batch_size: int,
    device: torch.device,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    array = _as_uint8(np.asarray(rgb))
    if array.ndim != 4 or array.shape[1] != 128 or array.shape[2] != 256 or array.shape[-1] != 3:
        raise ValueError(f"Expected LIBERO side-by-side RGB shape [T,128,256,3], got {array.shape}.")
    source_indices = _condition_source_frame_indices(
        raw_frame_count=int(array.shape[0]),
        latent_frames=int(latent_frames),
        source_frame_offset=int(source_frame_offset),
    )
    encoded_chunks: list[torch.Tensor] = []
    step = max(1, int(condition_batch_size))
    for start in range(0, len(source_indices), step):
        frames = array[np.asarray(source_indices[start : start + step], dtype=np.int64)]
        video = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)
            .unsqueeze(2)
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
        latents = _encode_libero_side_by_side_video(assets, video, device=device)
        encoded_chunks.append(latents[:, :, 0].detach().to(device="cpu", dtype=output_dtype))
    return torch.cat(encoded_chunks, dim=0).permute(1, 0, 2, 3).contiguous()


def _condition_source_frame_indices(
    *,
    raw_frame_count: int,
    latent_frames: int,
    source_frame_offset: int,
) -> list[int]:
    raw_frame_count = int(raw_frame_count)
    latent_frames = int(latent_frames)
    if raw_frame_count <= 0 or latent_frames <= 0:
        raise ValueError(
            "Expected positive raw and latent frame counts, "
            f"got raw_frame_count={raw_frame_count}, latent_frames={latent_frames}."
        )
    boundaries = latent_raw_boundaries(
        raw_frame_count=raw_frame_count,
        latent_num_frames=latent_frames,
        layout="wan_causal_stride4",
    )
    indices: list[int] = []
    for latent_index in range(latent_frames):
        boundary_index = min(int(latent_index) + 1, len(boundaries) - 1)
        raw_position = min(int(boundaries[boundary_index]), raw_frame_count - 1)
        raw_position = max(0, min(raw_position + int(source_frame_offset), raw_frame_count - 1))
        indices.append(raw_position)
    return indices


def _stack_libero_side_by_side_rgb(rgb_batch: list[np.ndarray], *, device: torch.device) -> torch.Tensor:
    arrays = []
    for rgb in rgb_batch:
        array = np.asarray(rgb)
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"Expected RGB video [T,H,W,3], got {array.shape}.")
        if array.shape[1] != 128 or array.shape[2] != 256:
            raise ValueError(f"Expected LIBERO side-by-side RGB shape [T,128,256,3], got {array.shape}.")
        if array.dtype != np.uint8:
            array = _as_uint8(array)
        arrays.append(array)
    stacked = np.stack(arrays, axis=0)
    return torch.from_numpy(stacked).permute(0, 4, 1, 2, 3).to(device=device, dtype=torch.float32) / 255.0


def _load_counterfactual_rgb(payload: np.lib.npyio.NpzFile, *, legacy_key: str) -> np.ndarray:
    if legacy_key in payload.files:
        return np.asarray(payload[legacy_key])
    missing = [key for key in LIBERO_OBS_KEYS if key not in payload.files]
    if missing:
        raise KeyError(
            f"Counterfactual RGB payload has neither legacy key {legacy_key!r} nor canonical view keys; "
            f"missing={missing}."
        )
    left = _as_uint8(np.asarray(payload[LIBERO_OBS_KEYS[0]]))
    right = _as_uint8(np.asarray(payload[LIBERO_OBS_KEYS[1]]))
    if left.ndim != 4 or right.ndim != 4:
        raise ValueError(f"Expected canonical camera videos [T,H,W,3], got {left.shape} and {right.shape}.")
    if left.shape[0] != right.shape[0] or left.shape[1:3] != right.shape[1:3]:
        raise ValueError(f"Expected matching canonical camera videos, got {left.shape} and {right.shape}.")
    return np.concatenate([left, right], axis=2)


def _encode_libero_side_by_side_video(
    assets: LingbotReferenceAssets,
    video: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected video [B,3,T,H,W], got {tuple(video.shape)}.")
    return assets.encode_video(
        video.to(device=device),
        placements=_libero_side_by_side_placements(),
        reset_cache=True,
    )


def _libero_side_by_side_placements() -> tuple[ViewPlacement, ...]:
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


def _resolve_shards(dataset_root: Path, shards_arg: str | None) -> list[Path]:
    if shards_arg is None:
        shards = sorted(path for path in dataset_root.glob("shard_tasks_*") if path.is_dir())
        if not shards and _is_single_root_dataset(dataset_root):
            shards = [dataset_root]
    else:
        shard_names = [part.strip() for part in shards_arg.split(",") if part.strip()]
        shards = [dataset_root / shard_name for shard_name in shard_names]
    if not shards:
        raise FileNotFoundError(
            f"No shard_tasks_* directories or single-root metadata dataset found under {dataset_root}."
        )
    missing = [str(path) for path in shards if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Requested shard directories do not exist: {missing}")
    return shards


def _is_single_root_dataset(dataset_root: Path) -> bool:
    return (
        (dataset_root / "metadata" / "contexts.jsonl").is_file()
        and (dataset_root / "metadata" / "transitions.jsonl").is_file()
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _batched(rows: list[dict[str, Any]], batch_size: int):
    size = max(1, int(batch_size))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _parse_output_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported output dtype: {value!r}")


def _as_uint8(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array
    if array.size and float(np.nanmax(array)) <= 1.0001:
        return (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode LIBERO FDM counterfactual RGB dataset into Wan latents.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", "--cfg", default="configs/experiments/parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-assets-device-policy", default="runtime")
    parser.add_argument("--output-dtype", default="float16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--condition-source-frame-offset",
        type=int,
        default=-1,
        help="Source-frame offset used for explicit single-frame condition latents.",
    )
    parser.add_argument(
        "--condition-batch-size",
        type=int,
        default=16,
        help="Single-frame condition videos per VAE encode call.",
    )
    parser.add_argument(
        "--skip-condition-latents",
        action="store_true",
        help=(
            "Only encode context/target video latents. This is not valid for current "
            "M5 legacy-prefix GJD configs that use condition_source_frame_offset=-1."
        ),
    )
    parser.add_argument("--shards", default=None, help="Comma-separated shard directory names. Defaults to all shards.")
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.condition_batch_size <= 0:
        parser.error("--condition-batch-size must be positive.")
    return args


if __name__ == "__main__":
    main()
