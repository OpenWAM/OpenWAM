#!/usr/bin/env python
"""Decode and summarize sharded causal-video latent evaluation artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from open_wam.artifacts.serialization import load_tensor_artifact
from open_wam.configs import load_experiment_config
from open_wam.contracts import CanonicalViewLayout, ViewPlacement
from open_wam.evals import decode_canonical_latent_views
from open_wam.models.common.video_geometry import WAN_TEMPORAL_CHUNK_SIZE
from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode saved video-marginal latent artifacts without loading the "
            "transformer, then aggregate dense RGB and optional FVD metrics."
        )
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--reference-assets-root", required=True)
    parser.add_argument("--decode-device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--preview-fps", type=float, default=8.0)
    parser.add_argument(
        "--fvd-horizons",
        default="1",
        help="Comma-separated chunk horizons to score with continuous-frame FVD.",
    )
    parser.add_argument(
        "--uva-root",
        default=None,
        help="Optional unified_video_action checkout providing its FVD implementation.",
    )
    parser.add_argument("--i3d-checkpoint", default=None)
    parser.add_argument("--fvd-device", default=None)
    parser.add_argument("--fvd-batch-size", type=int, default=2)
    return parser


def _positive_csv_ints(value: str, *, option: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise ValueError(f"{option} must contain comma-separated integers.") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{option} must contain positive integers, got {value!r}.")
    return parsed


def _load_sharded_records(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reports = [json.loads(path.read_text()) for path in paths]
    if not reports:
        raise ValueError("At least one report is required.")
    contract_keys = (
        "schema_version",
        "config",
        "transformer_dir",
        "data_root",
        "split",
        "segment_frames",
        "rollout_frame_chunk_size",
        "rollout_chunk_horizons",
        "geometry_mode",
        "train_window_size",
        "max_train_chunk_size",
        "skip_train_metrics",
        "text_source",
        "guidance_scale",
        "seed",
        "num_shards",
        "global_sample_indices",
    )
    reference = reports[0]
    for report in reports[1:]:
        for key in contract_keys:
            if report.get(key) != reference.get(key):
                raise ValueError(
                    f"Sharded reports disagree on {key!r}: "
                    f"{reference.get(key)!r} != {report.get(key)!r}."
                )
        # Commands differ by shard/output path; scientific identities must not.
        for key in ("config", "checkpoint", "dataset"):
            if report.get("provenance", {}).get(key) != reference.get("provenance", {}).get(key):
                raise ValueError(f"Sharded reports disagree on provenance {key!r}.")
        for key in ("commit", "dirty"):
            source = report.get("provenance", {}).get("source", {})
            expected_source = reference.get("provenance", {}).get("source", {})
            if source.get(key) != expected_source.get(key):
                raise ValueError(f"Sharded reports disagree on source {key!r}.")
    records = sorted(
        (record for report in reports for record in report["records"]),
        key=lambda record: int(record["ordinal"]),
    )
    ordinals = [int(record["ordinal"]) for record in records]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("Sharded reports contain duplicate global sample ordinals.")
    expected_count = len(reference["global_sample_indices"])
    if expected_count == 0:
        raise ValueError("Reports must contain a nonempty global sample population.")
    if ordinals != list(range(expected_count)):
        raise ValueError(
            "Sharded reports do not cover the full global sample population: "
            f"expected={list(range(expected_count))}, actual={ordinals}."
        )
    return records, reference


def _derive_latent_layout(
    *,
    config: Any,
    latent_height: int,
    latent_width: int,
) -> dict[str, Any]:
    canvas_height = int(config.data.canonical_height)
    canvas_width = int(config.data.canonical_width)
    if canvas_height % latent_height != 0 or canvas_width % latent_width != 0:
        raise ValueError(
            "Canonical RGB and latent canvases require integral spatial strides: "
            f"rgb={(canvas_height, canvas_width)}, "
            f"latent={(latent_height, latent_width)}."
        )
    stride_h = canvas_height // latent_height
    stride_w = canvas_width // latent_width
    placements = []
    for placement in config.data.view_layout:
        values = (
            int(placement.top),
            int(placement.left),
            int(placement.height),
            int(placement.width),
        )
        divisors = (stride_h, stride_w, stride_h, stride_w)
        if any(value % divisor != 0 for value, divisor in zip(values, divisors, strict=True)):
            raise ValueError(
                "Canonical view placement is not aligned to the latent stride: "
                f"placement={placement}, stride={(stride_h, stride_w)}."
            )
        placements.append(
            {
                "source_name": placement.source_name,
                "canonical_name": placement.canonical_name,
                "top": int(placement.top) // stride_h,
                "left": int(placement.left) // stride_w,
                "height": int(placement.height) // stride_h,
                "width": int(placement.width) // stride_w,
            }
        )
    return CanonicalViewLayout(
        canvas_height=latent_height,
        canvas_width=latent_width,
        placements=tuple(
            ViewPlacement(**placement)
            for placement in placements
        ),
    ).to_metadata()


def _decode_pipeline(config: Any, *, reference_assets_root: Path) -> Any:
    backbone = replace(
        config.backbone,
        pretrained_model_name_or_path=str(reference_assets_root),
        load_wan_vae_frontend=True,
        load_text_conditioning=False,
        load_reference_core_weights=False,
        runtime_backbone_artifact_path=None,
    )
    assets = LingbotReferenceAssets.maybe_load(backbone)
    if not assets.has_vae:
        raise RuntimeError("The requested reference assets did not load a WAN VAE.")
    return SimpleNamespace(
        visual_tower=SimpleNamespace(
            frontend=SimpleNamespace(reference_assets=assets)
        )
    )


def _unit_float(video: np.ndarray) -> np.ndarray:
    value = np.asarray(video)
    if value.dtype == np.uint8:
        return value.astype(np.float32) / 255.0
    return np.clip(value.astype(np.float32), 0.0, 1.0)


def _uint8(video: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(_unit_float(video) * 255.0), 0, 255).astype(np.uint8)


def _ssim_per_frame(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = _unit_float(predicted)
    tgt = _unit_float(target)
    if pred.shape != tgt.shape:
        raise ValueError(f"RGB shape mismatch: {pred.shape} != {tgt.shape}.")
    pred = pred.reshape(pred.shape[0], -1, pred.shape[-1])
    tgt = tgt.reshape(tgt.shape[0], -1, tgt.shape[-1])
    mu_pred = pred.mean(axis=1)
    mu_tgt = tgt.mean(axis=1)
    var_pred = pred.var(axis=1)
    var_tgt = tgt.var(axis=1)
    covariance = ((pred - mu_pred[:, None]) * (tgt - mu_tgt[:, None])).mean(
        axis=1
    )
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_pred * mu_tgt + c1) * (2 * covariance + c2)
    denominator = (np.square(mu_pred) + np.square(mu_tgt) + c1) * (
        var_pred + var_tgt + c2
    )
    return (numerator / np.maximum(denominator, 1e-12)).mean(axis=1)


def _rgb_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = _unit_float(predicted)
    tgt = _unit_float(target)
    if pred.shape != tgt.shape:
        raise ValueError(f"RGB shape mismatch: {pred.shape} != {tgt.shape}.")
    per_frame_mse = np.square(pred - tgt).mean(axis=(1, 2, 3))
    mse = float(per_frame_mse.mean())
    return {
        "rgb_mse": mse,
        "rgb_psnr_db": float("inf") if mse <= 0 else -10.0 * math.log10(mse),
        "rgb_ssim": float(_ssim_per_frame(pred, tgt).mean()),
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values or any(math.isnan(value) for value in values):
        raise ValueError("Metrics must be nonempty and contain no NaN values.")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if all(map(math.isfinite, values)) else None,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _json_metrics(value: Any) -> Any:
    """Keep perfect-prediction PSNR exact without emitting nonstandard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("Cannot serialize a NaN metric.")
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_metrics(item) for item in value]
    return value


def _requested_fvd_horizons(args: argparse.Namespace, horizons: tuple[int, ...]) -> tuple[int, ...]:
    if (args.uva_root is None) != (args.i3d_checkpoint is None):
        raise ValueError("FVD requires both --uva-root and --i3d-checkpoint.")
    if args.uva_root is None:
        return ()
    requested = _positive_csv_ints(args.fvd_horizons, option="--fvd-horizons")
    if any(horizon not in horizons for horizon in requested):
        raise ValueError(f"FVD horizons {requested} are not present in report horizons {horizons}.")
    return requested


def _write_preview(
    path: Path,
    *,
    target: np.ndarray,
    predicted: np.ndarray,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.concatenate([_uint8(target), _uint8(predicted)], axis=2)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))


def _fvd_logits(
    clips: np.ndarray,
    *,
    i3d: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    get_fvd_logits: Any,
) -> torch.Tensor:
    batches = []
    for start in range(0, len(clips), batch_size):
        batches.append(
            get_fvd_logits(
                np.ascontiguousarray(clips[start : start + batch_size]),
                i3d=i3d,
                device=device,
            ).detach().cpu()
        )
    return torch.cat(batches, dim=0).to(device)


def _compute_fvd(
    *,
    target: np.ndarray,
    predicted: np.ndarray,
    uva_root: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> float:
    sys.path.insert(0, str(uva_root))
    from unified_video_action.fvd.fvd import frechet_distance, get_fvd_logits
    from unified_video_action.fvd.pytorch_i3d import InceptionI3d

    i3d = InceptionI3d(400, in_channels=3).to(device)
    i3d.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    i3d.eval()
    target_logits = _fvd_logits(
        target,
        i3d=i3d,
        device=device,
        batch_size=batch_size,
        get_fvd_logits=get_fvd_logits,
    )
    predicted_logits = _fvd_logits(
        predicted,
        i3d=i3d,
        device=device,
        batch_size=batch_size,
        get_fvd_logits=get_fvd_logits,
    )
    return float(
        frechet_distance(predicted_logits, target_logits).detach().cpu().item()
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.preview_count < 0:
        raise ValueError("--preview-count cannot be negative.")
    if not math.isfinite(args.preview_fps) or args.preview_fps <= 0:
        raise ValueError("--preview-fps must be finite and positive.")
    if args.fvd_batch_size <= 0:
        raise ValueError("--fvd-batch-size must be positive.")
    report_paths = [Path(path).expanduser().resolve() for path in args.reports]
    records, report = _load_sharded_records(report_paths)
    horizons = tuple(int(value) for value in report["rollout_chunk_horizons"])
    fvd_horizons = _requested_fvd_horizons(args, horizons)

    config = load_experiment_config(
        Path(report["config"]), checkpoint_runtime_compat=True
    )
    pipeline = _decode_pipeline(
        config,
        reference_assets_root=Path(args.reference_assets_root).expanduser().resolve(),
    )
    decode_device = torch.device(args.decode_device)
    assets = pipeline.visual_tower.frontend.reference_assets
    decode_dtype = torch.bfloat16 if decode_device.type == "cuda" else torch.float32
    assets.vae = assets.vae.to(device=decode_device, dtype=decode_dtype)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    fvd_clips: dict[int, dict[str, list[np.ndarray]]] = {
        horizon: {"target": [], "predicted": []} for horizon in fvd_horizons
    }

    for record_index, record in enumerate(records):
        artifact = load_tensor_artifact(
            Path(record["latent_artifact"]), map_location="cpu"
        )
        observed = artifact["observed_latents"]
        predicted_future = artifact["predicted_future_latents"]
        target_future = artifact["target_future_latents"]
        layout_metadata = artifact.get("metadata", {}).get("latent_layout")
        if layout_metadata is None:
            layout_metadata = _derive_latent_layout(
                config=config,
                latent_height=int(observed.shape[-2]),
                latent_width=int(observed.shape[-1]),
            )
        target_latents = torch.cat([observed, target_future], dim=2)
        predicted_latents = torch.cat([observed, predicted_future], dim=2)
        target_video = decode_canonical_latent_views(
            pipeline,
            target_latents,
            latent_layout=layout_metadata,
            decode_device=decode_device,
        )
        predicted_video = decode_canonical_latent_views(
            pipeline,
            predicted_latents,
            latent_layout=layout_metadata,
            decode_device=decode_device,
        )
        if target_video is None or predicted_video is None:
            raise RuntimeError("WAN VAE decode unexpectedly returned no video.")
        dense_future_frames = (
            int(predicted_future.shape[2]) * WAN_TEMPORAL_CHUNK_SIZE
        )
        expected_total_frames = dense_future_frames + 1
        if (
            len(target_video) != expected_total_frames
            or len(predicted_video) != expected_total_frames
        ):
            raise ValueError(
                "WAN decode did not preserve the expected one-context plus dense-future "
                f"contract: expected={expected_total_frames}, "
                f"target={len(target_video)}, predicted={len(predicted_video)}."
            )
        target_video = target_video[1:]
        predicted_video = predicted_video[1:]
        layout = CanonicalViewLayout.from_metadata(layout_metadata)
        spatial_scale_h = target_video.shape[1] // layout.canvas_height
        spatial_scale_w = target_video.shape[2] // layout.canvas_width

        for horizon in horizons:
            frame_count = (
                int(horizon)
                * int(report["rollout_frame_chunk_size"])
                * WAN_TEMPORAL_CHUNK_SIZE
            )
            if frame_count > dense_future_frames:
                raise ValueError(
                    f"Artifact has {dense_future_frames} future frames, "
                    f"but horizon {horizon} requires {frame_count}."
                )
            target_prefix = target_video[:frame_count]
            predicted_prefix = predicted_video[:frame_count]
            row: dict[str, Any] = {
                "label": args.label,
                "ordinal": int(record["ordinal"]),
                "dataset_index": int(record["dataset_index"]),
                "task_index": int(record["task_index"]),
                "episode_index": int(record["episode_index"]),
                "horizon_chunks": int(horizon),
                "dense_future_frames": int(frame_count),
            }
            row.update(
                {
                    f"canonical_{key}": value
                    for key, value in _rgb_metrics(
                        predicted_prefix, target_prefix
                    ).items()
                }
            )
            for placement in layout.placements:
                top = placement.top * spatial_scale_h
                left = placement.left * spatial_scale_w
                height = placement.height * spatial_scale_h
                width = placement.width * spatial_scale_w
                view_target = target_prefix[
                    :, top : top + height, left : left + width
                ]
                view_predicted = predicted_prefix[
                    :, top : top + height, left : left + width
                ]
                row.update(
                    {
                        f"{placement.canonical_name}_{key}": value
                        for key, value in _rgb_metrics(
                            view_predicted, view_target
                        ).items()
                    }
                )
                if (
                    horizon in fvd_clips
                    and placement.canonical_name == "image"
                ):
                    fvd_clips[horizon]["target"].append(_uint8(view_target))
                    fvd_clips[horizon]["predicted"].append(_uint8(view_predicted))
            metric_rows.append(row)

        if record_index < int(args.preview_count):
            _write_preview(
                output_dir / "previews" / f"sample_{int(record['ordinal']):04d}.mp4",
                target=target_video,
                predicted=predicted_video,
                fps=float(args.preview_fps),
            )
        print(
            f"[{args.label}] decoded {record_index + 1}/{len(records)}",
            flush=True,
        )

    summaries: dict[str, Any] = {}
    metric_names = sorted(
        key
        for key in metric_rows[0]
        if key.endswith(("_rgb_mse", "_rgb_psnr_db", "_rgb_ssim"))
    )
    for horizon in horizons:
        horizon_rows = [
            row for row in metric_rows if row["horizon_chunks"] == horizon
        ]
        summaries[f"h{horizon}"] = {
            name: _summary([float(row[name]) for row in horizon_rows])
            for name in metric_names
        }

    fvd: dict[str, Any] = {}
    if fvd_horizons:
        fvd_device = torch.device(args.fvd_device or args.decode_device)
        for horizon, clips in fvd_clips.items():
            target_array = np.stack(clips["target"])
            predicted_array = np.stack(clips["predicted"])
            np.save(output_dir / f"h{horizon}_target_agentview_uint8.npy", target_array)
            np.save(
                output_dir / f"h{horizon}_predicted_agentview_uint8.npy",
                predicted_array,
            )
            fvd[f"h{horizon}_agentview_continuous_fvd"] = _compute_fvd(
                target=target_array,
                predicted=predicted_array,
                uva_root=Path(args.uva_root).expanduser().resolve(),
                checkpoint=Path(args.i3d_checkpoint).expanduser().resolve(),
                device=fvd_device,
                batch_size=int(args.fvd_batch_size),
            )

    result = {
        "schema_version": "open_wam.video_marginal_rgb_eval.v1",
        "label": args.label,
        "reports": [str(path) for path in report_paths],
        "reference_assets_root": str(
            Path(args.reference_assets_root).expanduser().resolve()
        ),
        "sample_count": len(records),
        "sample_ordinals": [int(record["ordinal"]) for record in records],
        "horizons": list(horizons),
        "dense_frame_contract": "one WAN context frame removed; four RGB frames per future latent",
        "ssim_contract": "dependency-free global RGB SSIM averaged over dense frames",
        "psnr_contract": "unit-range RGB; perfect MSE gives string Infinity; std is null for nonfinite samples",
        "fvd_contract": "UVA I3D; continuous agentview frames; no temporal repetition",
        "metrics": summaries,
        "fvd": fvd,
        "records": metric_rows,
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(_json_metrics(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output_json": str(output_path), **fvd}, indent=2))


if __name__ == "__main__":
    main()
