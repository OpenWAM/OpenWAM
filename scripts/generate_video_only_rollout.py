#!/usr/bin/env python
"""Evaluate causal video prediction on an adapter-produced latent sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from open_wam.configs import (
    ExperimentConfig,
    ReferenceCoreInitMode,
    load_experiment_config,
    resolve_experiment_config_reference,
    serialize_experiment_config,
    validate_experiment_config_runtime_contract,
)
from open_wam.data import (
    LatentWAMBatch,
    build_train_val_latent_datasets,
    collate_latent_wam_samples,
    move_latent_wam_batch_to_device,
)
from open_wam.evals.video_artifacts import to_uint8, write_video_frames
from open_wam.evals.video_prediction import (
    decode_canonical_latent_views,
    rollout_causal_video_prediction,
)
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.runtime.provenance import (
    ProvenanceMode,
    collect_artifact_identity,
    collect_runtime_provenance,
)
from open_wam.runtime.publication import (
    ensure_output_path_available,
    staged_output_directory,
)
from open_wam.runtime.results import build_result_envelope, write_result_json
from open_wam.utils import resolve_transformer_dir_override, seed_everywhere


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate future latents with the causal-video program and sample "
            "layout returned by a configured latent dataset adapter."
        )
    )
    parser.add_argument("--cfg", "--config", dest="config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint step directory or exported transformer directory.",
    )
    parser.add_argument(
        "--reference-assets-root",
        required=True,
        help="Model root supplying the Wan VAE used only for artifact decoding.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional override for data.local_root.",
    )
    parser.add_argument(
        "--empty-text-embedding",
        default=None,
        help="Optional override for data.empty_text_embedding_path.",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help=(
            "Number of chunks to generate; the prefix/suffix program may extend "
            "past its exact target while the chunked program uses available targets."
        ),
    )
    parser.add_argument(
        "--video-steps",
        type=int,
        default=None,
        help="Explicitly override inference.video_num_inference_steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Explicitly override inference.guidance_scale.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--decode-device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--preview-fps", type=float, default=8.0)
    parser.add_argument("--output-dir", default="outputs/video_only_rollout")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.sample_index < 0:
        raise ValueError(
            f"--sample-index must be non-negative, got {args.sample_index}."
        )
    if args.num_chunks <= 0:
        raise ValueError(f"--num-chunks must be positive, got {args.num_chunks}.")
    if args.preview_fps <= 0:
        raise ValueError(f"--preview-fps must be positive, got {args.preview_fps}.")

    config_path = resolve_experiment_config_reference(args.config)
    transformer_dir = resolve_transformer_dir_override(
        args.checkpoint,
        option_name="--checkpoint",
    )
    reference_assets_root = _existing_path(
        args.reference_assets_root, label="reference assets root"
    )
    config = _resolve_runtime_config(
        load_experiment_config(config_path),
        transformer_dir=transformer_dir,
        reference_assets_root=reference_assets_root,
        data_root=args.data_root,
        empty_text_embedding=args.empty_text_embedding,
        video_steps=args.video_steps,
        guidance_scale=args.guidance_scale,
    )

    seed_everywhere(args.seed)
    train_dataset, val_dataset = build_train_val_latent_datasets(config.data)
    dataset = train_dataset if args.split == "train" else val_dataset
    if args.sample_index >= len(dataset):
        raise IndexError(
            f"--sample-index {args.sample_index} is outside {args.split} split "
            f"length {len(dataset)}."
        )
    batch = collate_latent_wam_samples([dataset[args.sample_index]])

    resolved_config = serialize_experiment_config(config)
    artifact_identity = _rollout_artifact_identity(
        config=config,
        resolved_config=resolved_config,
        config_path=config_path,
        transformer_dir=transformer_dir,
        reference_assets_root=reference_assets_root,
        batch=batch,
        split=args.split,
        sample_index=args.sample_index,
        num_chunks=args.num_chunks,
        seed=args.seed,
        preview_fps=args.preview_fps,
    )
    output_dir = _output_directory(
        root=Path(args.output_dir).expanduser(),
        transformer_dir=transformer_dir,
        split=args.split,
        sample_index=args.sample_index,
        num_chunks=args.num_chunks,
        seed=args.seed,
        identity=artifact_identity,
    )
    ensure_output_path_available(output_dir)

    device = _resolve_device(args.device)
    decode_device = _resolve_device(args.decode_device, fallback=device)
    batch = move_latent_wam_batch_to_device(batch, device)
    pipeline = build_variant_pipeline_from_config(config).to(device)
    pipeline.eval()

    rollout = rollout_causal_video_prediction(
        pipeline,
        batch,
        num_chunks=args.num_chunks,
    )
    latent_layout = batch.metadata[0].get("latent_layout")
    target_video = decode_canonical_latent_views(
        pipeline,
        rollout.target_latents,
        latent_layout=latent_layout,
        decode_device=decode_device,
    )
    predicted_video = decode_canonical_latent_views(
        pipeline,
        rollout.predicted_latents,
        latent_layout=latent_layout,
        decode_device=decode_device,
    )

    summary = build_result_envelope(
        command="generate_video_only_rollout",
        config=str(config_path),
        checkpoint=str(transformer_dir),
        benchmark=config.data.dataset_name,
        device=str(device),
        seed=args.seed,
        metrics={
            "first_chunk_future_latent_mse": rollout.first_chunk_future_mse,
        },
        artifacts={"output_directory": str(output_dir)},
        provenance=collect_runtime_provenance(
            config_path=config_path,
            resolved_config=resolved_config,
            checkpoint_path=transformer_dir,
            dataset_root=config.data.local_root,
        ),
        extra={
            "artifact_identity": artifact_identity,
            "reference_assets_root": str(reference_assets_root),
            "split": args.split,
            "sample_index": args.sample_index,
            "sample_metadata": _jsonable(batch.metadata[0]),
            "observed_latent_frames": rollout.observed_latent_frames,
            "future_latent_frames": rollout.future_latent_frames,
            "context_latent_frames": list(rollout.context_latent_frames),
            "num_chunks": args.num_chunks,
            "video_num_inference_steps": config.inference.video_num_inference_steps,
            "guidance_scale": config.inference.guidance_scale,
            "preview_fps": args.preview_fps,
        },
    )
    summary = _publish_rollout_artifacts(
        output_dir,
        target=target_video,
        predicted=predicted_video,
        fps=args.preview_fps,
        summary=summary,
    )
    summary_path = output_dir / "summary.json"
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2))


def _resolve_runtime_config(
    config,
    *,
    transformer_dir: Path,
    reference_assets_root: Path,
    data_root: str | None,
    empty_text_embedding: str | None,
    video_steps: int | None,
    guidance_scale: float | None,
):
    data_updates: dict[str, Any] = {}
    if data_root is not None:
        data_updates["local_root"] = str(_existing_path(data_root, label="data root"))
    if empty_text_embedding is not None:
        data_updates["empty_text_embedding_path"] = str(
            _existing_path(empty_text_embedding, label="empty text embedding")
        )
    if data_updates:
        config = replace(config, data=replace(config.data, **data_updates))

    config = replace(
        config,
        backbone=replace(
            config.backbone,
            pretrained_model_name_or_path=str(reference_assets_root),
            runtime_backbone_artifact_path=str(transformer_dir),
            load_reference_core_weights=True,
            load_wan_vae_frontend=True,
            load_text_conditioning=False,
            reference_core_init_mode=ReferenceCoreInitMode.VIDEO_ONLY,
        ),
    )
    inference_updates: dict[str, Any] = {}
    if video_steps is not None:
        if int(video_steps) <= 0:
            raise ValueError(f"--video-steps must be positive, got {video_steps}.")
        inference_updates["video_num_inference_steps"] = int(video_steps)
    if guidance_scale is not None:
        if float(guidance_scale) <= 0:
            raise ValueError(
                f"--guidance-scale must be positive, got {guidance_scale}."
            )
        inference_updates["guidance_scale"] = float(guidance_scale)
    if inference_updates:
        config = replace(
            config,
            inference=replace(config.inference, **inference_updates),
        )
    return validate_experiment_config_runtime_contract(config)


def _existing_path(value: str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured {label} does not exist: {path}")
    return path


def _resolve_device(
    value: str | None,
    *,
    fallback: torch.device | None = None,
) -> torch.device:
    if value is not None:
        return torch.device(value)
    if fallback is not None:
        return fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _output_directory(
    *,
    root: Path,
    transformer_dir: Path,
    split: str,
    sample_index: int,
    num_chunks: int,
    seed: int,
    identity: dict[str, Any],
) -> Path:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return root / (
        f"{transformer_dir.parent.name}_{split}_{sample_index:06d}_"
        f"chunks{num_chunks}_seed{seed}_{digest}"
    )


def _publish_rollout_artifacts(
    output_dir: Path,
    *,
    target: np.ndarray | None,
    predicted: np.ndarray | None,
    fps: float,
    summary: dict[str, Any],
) -> dict[str, Any]:
    with staged_output_directory(output_dir) as temporary_dir:
        video_names = _write_videos(
            temporary_dir,
            target=target,
            predicted=predicted,
            fps=fps,
        )
        videos = [str(output_dir / name) for name in video_names]
        published_summary = {
            **summary,
            "artifacts": {
                **dict(summary.get("artifacts", {})),
                "videos": videos,
            },
        }
        write_result_json(
            temporary_dir / "summary.json",
            published_summary,
        )
    return published_summary


def _write_videos(
    output_dir: Path,
    *,
    target: np.ndarray | None,
    predicted: np.ndarray | None,
    fps: float,
) -> list[str]:
    videos: list[str] = []
    for name, video in (("target", target), ("predicted", predicted)):
        if video is None:
            continue
        path = output_dir / f"{name}.mp4"
        write_video_frames(path, (to_uint8(frame) for frame in video), fps=fps)
        videos.append(path.name)
    if target is not None and predicted is not None:
        overlap = min(len(target), len(predicted))
        comparison = np.concatenate([target[:overlap], predicted[:overlap]], axis=2)
        path = output_dir / "comparison_target_left_predicted_right.mp4"
        write_video_frames(
            path,
            (to_uint8(frame) for frame in comparison),
            fps=fps,
        )
        videos.append(path.name)
    return videos


def _rollout_artifact_identity(
    *,
    config: ExperimentConfig,
    resolved_config: dict[str, Any],
    config_path: Path,
    transformer_dir: Path,
    reference_assets_root: Path,
    batch: LatentWAMBatch,
    split: str,
    sample_index: int,
    num_chunks: int,
    seed: int,
    preview_fps: float,
) -> dict[str, Any]:
    return {
        "schema_version": "open_wam.causal_video_rollout.v1",
        "resolved_config": resolved_config,
        "config_file": collect_artifact_identity(
            config_path,
            mode=ProvenanceMode.FULL,
        ),
        "checkpoint_transformer": collect_artifact_identity(
            transformer_dir,
        ),
        "reference_assets": collect_artifact_identity(
            reference_assets_root,
            relative_paths=("vae/config.json", "text_encoder/config.json"),
            file_patterns=("vae/*.safetensors*", "vae/*.bin"),
        ),
        "empty_text_embedding": collect_artifact_identity(
            config.data.empty_text_embedding_path,
            mode=ProvenanceMode.FULL,
        ),
        "sample": {
            "split": split,
            "index": sample_index,
            "metadata": _jsonable(batch.metadata[0]),
            "task_text": _jsonable(batch.task_text),
            "video_latents_sha256": _tensor_sha256(batch.video_latents),
            "text_context_sha256": _tensor_sha256(batch.text_context),
            "negative_text_context_sha256": _tensor_sha256(batch.negative_text_context),
        },
        "inference": {
            "num_chunks": num_chunks,
            "seed": seed,
            "preview_fps": preview_fps,
        },
    }


def _tensor_sha256(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


if __name__ == "__main__":
    main()
