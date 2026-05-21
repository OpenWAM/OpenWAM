from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import threading
import queue

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import (  # noqa: E402
    CausalPrefixSuffixBucketConfig,
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoFrameFitMode,
    MixedVideoResizeBinConfig,
    MixedVideoSourceFormat,
)
from open_wam.configs.enums import serialize_enum_values  # noqa: E402
from open_wam.data.mixed_video import (  # noqa: E402
    MixedVideoEpisodeRecord,
    MixedVideoStreamRecord,
    iter_mixed_video_stream_frame_chunks,
    load_mixed_video_catalog,
    split_mixed_video_episodes,
)
from open_wam.data.raw_video import build_canonical_video_preprocessor  # noqa: E402
from open_wam.models.common.video_geometry import wan_raw_frame_count_to_latent_count  # noqa: E402
from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets  # noqa: E402
from open_wam.utils.config_loader import load_experiment_config  # noqa: E402
from open_wam.utils.video_timeline import VideoFrameMapping  # noqa: E402


LATENT_KEY = "video_latents"
MAX_PENDING_LATENT_SAVE_FUTURES = 1


def _mixed_video_transform_signature(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord | None = None,
) -> dict[str, Any]:
    resize_bins = serialize_enum_values(asdict(data_config))["decode_resize_bins"]
    view_layout = [
        {
            "source_name": layout.source_name,
            "canonical_name": layout.canonical_name,
            "top": int(layout.top),
            "left": int(layout.left),
            "height": int(layout.height),
            "width": int(layout.width),
        }
        for layout in data_config.view_layout
    ]
    signature: dict[str, Any] = {
        "canonical_height": int(data_config.canonical_height),
        "canonical_width": int(data_config.canonical_width),
        "camera_names": list(data_config.camera_names),
        "latent_camera_names": list(data_config.latent_camera_names),
        "view_layout": view_layout,
        "decode_size_mode": data_config.decode_size_mode.value,
        "decode_fit_mode": data_config.decode_fit_mode.value,
        "decode_allow_upscale": bool(data_config.decode_allow_upscale),
        "decode_height": int(data_config.decode_height),
        "decode_width": int(data_config.decode_width),
        "decode_resize_bins": resize_bins,
        "target_observation_fps": data_config.target_observation_fps,
        "missing_observation_fps": float(data_config.missing_observation_fps),
    }
    if episode is not None:
        signature["streams"] = [
            {
                "target_slot": stream.target_slot,
                "stream_key": stream.stream_key,
                "clip_id": stream.clip_id,
                "length_frames": int(stream.length_frames),
                "latent_length_frames": stream.latent_length_frames,
                "observation_fps": stream.observation_fps,
                "resolved_observation_fps": stream.clip.source_fps,
                "source_fps_source": stream.clip.source_fps_source,
                "from_timestamp": stream.from_timestamp,
                "to_timestamp": stream.to_timestamp,
                "width": stream.width,
                "height": stream.height,
                "source_format": stream.source_format.value,
            }
            for stream in sorted(episode.streams, key=lambda item: (item.target_slot, item.stream_index, item.stream_key))
            if stream.target_slot in data_config.camera_names
        ]
    return signature


def _mixed_video_transform_signature_hash(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord | None = None,
) -> str:
    payload = json.dumps(_mixed_video_transform_signature(data_config, episode), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EncodedEpisode:
    source_id: str
    dataset_id: str
    episode_index: int
    clip_id: str
    latent_path: Path
    latent_shape: tuple[int, int, int, int]
    raw_length_frames: int
    latent_length_frames: int
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class EncoderSelection:
    split: str
    source_ids: tuple[str, ...] = ()
    episode_indices: tuple[int, ...] = ()
    max_episodes: int | None = None
    shard_count: int = 1
    shard_index: int = 0


def _wait_for_latent_save(latent_path: Path, save_future: Future) -> None:
    try:
        save_future.result()
    except Exception as exc:
        raise RuntimeError(f"Failed to write mixed-video latent sidecar: {latent_path}") from exc


def _submit_latent_save(
    *,
    save_executor: ThreadPoolExecutor,
    save_futures: list[tuple[Path, Future]],
    latent_path: Path,
    payload: dict[str, Any],
    max_pending: int = MAX_PENDING_LATENT_SAVE_FUTURES,
) -> None:
    pending_limit = max(1, int(max_pending))
    while len(save_futures) >= pending_limit:
        _wait_for_latent_save(*save_futures.pop(0))
    save_futures.append((latent_path, save_executor.submit(torch.save, payload, latent_path)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode mixed-video RGB manifests into local WAN/VAE latent sidecars and "
            "write latent-first manifests for later training."
        )
    )
    parser.add_argument("--cfg", "--config", dest="config", required=True, help="Mixed-video experiment YAML.")
    parser.add_argument("--output-root", required=True, help="Directory for latent sidecars, manifests, and report.")
    parser.add_argument("--device", default="cuda:0", help="Runtime device for VAE encoding.")
    parser.add_argument(
        "--split",
        choices=("all", "train", "val"),
        default="all",
        help="Episode split to encode. Default encodes every manifest episode.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=None,
        help="Restrict to one source_id. May be passed multiple times.",
    )
    parser.add_argument(
        "--episode-index",
        action="append",
        type=int,
        default=None,
        help="Restrict to one episode index. May be passed multiple times.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional global episode limit after filtering.")
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Manual distributed encoding: total number of deterministic episode shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Manual distributed encoding: zero-based shard index for this process.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated devices for local multi-process encoding, e.g. cuda:0,cuda:1. "
            "The parent launches one shard per device, then merges manifests/reports."
        ),
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=65,
        help=(
            "Maximum raw frames per VAE call. Values >=5 are rounded down to WAN-safe "
            "1+4k/4k streaming chunks. Use 0 to encode each episode in one call."
        ),
    )
    parser.add_argument(
        "--decode-size-mode",
        choices=("aspect_ratio_bins", "fixed", "config"),
        default="aspect_ratio_bins",
        help="Decode override for RGB inputs. Default uses VAE-friendly aspect-ratio bins.",
    )
    parser.add_argument(
        "--decode-fit-mode",
        choices=("letterbox_pad", "center_crop", "config"),
        default="config",
        help=(
            "Frame fit override for RGB inputs. Default keeps the YAML setting; use letterbox_pad or center_crop "
            "to override explicitly."
        ),
    )
    parser.add_argument("--decode-height", type=int, default=None, help="Override fixed decode height.")
    parser.add_argument("--decode-width", type=int, default=None, help="Override fixed decode width.")
    parser.add_argument(
        "--decode-resize-bins",
        default=None,
        help="Optional JSON/YAML file containing a list of mixed-video resize-bin objects.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing latent sidecars and manifests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Resume a previous run: reuse existing latent sidecars and regenerate manifests/reports. "
            "Missing selected sidecars are encoded normally."
        ),
    )
    parser.add_argument(
        "--shard-output-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    if not isinstance(config.data, MixedVideoDataConfig):
        raise ValueError("encode_mixed_video_latents.py requires `data.dataset_type: mixed_video`.")
    data_config = resolve_encoder_data_config(
        config.data,
        decode_size_mode=args.decode_size_mode,
        decode_fit_mode=args.decode_fit_mode,
        decode_height=args.decode_height,
        decode_width=args.decode_width,
        decode_resize_bins_path=args.decode_resize_bins,
    )
    selection = EncoderSelection(
        split=args.split,
        source_ids=tuple(args.source_id or ()),
        episode_indices=tuple(args.episode_index or ()),
        max_episodes=args.max_episodes,
        shard_count=int(args.shard_count),
        shard_index=int(args.shard_index),
    )
    if args.devices:
        report = launch_parallel_mixed_video_encoding(
            args=args,
            experiment_config=config,
            data_config=data_config,
            selection=selection,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    output_root = Path(args.output_root).expanduser().resolve()
    if args.shard_output_only:
        _preflight_shard_report_path(
            output_root,
            shard_index=int(args.shard_index),
            overwrite=bool(args.overwrite or args.skip_existing),
        )
    selected_episodes = _select_encoder_episodes(data_config, selection=selection, apply_shard=True)
    all_selected_sidecars_exist = all(
        _latent_path_for_episode(output_root / "latents", episode).exists()
        for episode in selected_episodes
    )
    if args.skip_existing and all_selected_sidecars_exist:
        assets = None
    else:
        assets = LingbotReferenceAssets.maybe_load(config.backbone)
        if not assets.has_vae:
            raise RuntimeError("The selected config must load WAN VAE assets (`backbone.load_wan_vae_frontend: true`).")

    report = encode_mixed_video_latent_sources(
        data_config=data_config,
        assets=assets,
        output_root=output_root,
        device=torch.device(args.device),
        selection=selection,
        experiment_config=config,
        chunk_frames=args.chunk_frames,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        write_manifests=not args.shard_output_only,
    )
    if args.shard_output_only:
        shard_report_path = _write_shard_report(
            report,
            output_root=Path(args.output_root),
            shard_index=int(args.shard_index),
            overwrite=bool(args.overwrite or args.skip_existing),
        )
        report = {**report, "shard_report_path": str(shard_report_path)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def resolve_encoder_data_config(
    data_config: MixedVideoDataConfig,
    *,
    decode_size_mode: str,
    decode_fit_mode: str,
    decode_height: int | None = None,
    decode_width: int | None = None,
    decode_resize_bins_path: str | None = None,
) -> MixedVideoDataConfig:
    updates: dict[str, Any] = {}
    if decode_size_mode != "config":
        updates["decode_size_mode"] = MixedVideoDecodeSizeMode(decode_size_mode)
    if decode_fit_mode != "config":
        updates["decode_fit_mode"] = MixedVideoFrameFitMode(decode_fit_mode)
    if decode_height is not None:
        updates["decode_height"] = int(decode_height)
    if decode_width is not None:
        updates["decode_width"] = int(decode_width)
    if decode_resize_bins_path is not None:
        updates["decode_resize_bins"] = _load_resize_bins(Path(decode_resize_bins_path))
    if updates.get("decode_size_mode", data_config.decode_size_mode) == MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS:
        # The encoder itself is single-episode, but the data config validator
        # also protects training-time collation. Normalize these fields so a
        # fixed-size training config can still be reused for offline encoding.
        updates.setdefault("train_batch_size", 1)
        updates.setdefault("val_batch_size", 1)
    return replace(data_config, **updates)


def launch_parallel_mixed_video_encoding(
    *,
    args: argparse.Namespace,
    experiment_config,
    data_config: MixedVideoDataConfig,
    selection: EncoderSelection,
) -> dict[str, Any]:
    devices = tuple(device.strip() for device in str(args.devices).split(",") if device.strip())
    if not devices:
        raise ValueError("--devices must list at least one device when provided.")
    output_root = Path(args.output_root).expanduser().resolve()
    latents_root = output_root / "latents"
    manifests_root = output_root / "manifests"
    latents_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    all_episodes = _select_encoder_episodes(data_config, selection=selection, apply_shard=False)
    _preflight_output_paths(
        all_episodes,
        output_root=output_root,
        latents_root=latents_root,
        manifests_root=manifests_root,
        overwrite=bool(args.overwrite),
        skip_existing=bool(args.skip_existing),
        write_manifests=True,
    )
    _preflight_shard_report_paths(
        output_root,
        shard_indices=range(len(devices)),
        overwrite=bool(args.overwrite or args.skip_existing),
    )

    worker_count = len(devices)
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for shard_index, device in enumerate(devices):
        worker_selection = replace(selection, shard_count=worker_count, shard_index=shard_index)
        command = _parallel_worker_command(args, device=device, selection=worker_selection)
        processes.append((device, subprocess.Popen(command, text=True)))

    failures: list[str] = []
    for device, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append(f"{device}: exit {return_code}")
    if failures:
        formatted = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"Mixed-video latent encoding worker failure:\n{formatted}")

    # Merge from sidecars without loading the VAE in the parent process.
    return encode_mixed_video_latent_sources(
        data_config=data_config,
        assets=None,
        output_root=output_root,
        device=torch.device("cpu"),
        selection=replace(selection, shard_count=1, shard_index=0),
        experiment_config=experiment_config,
        chunk_frames=int(args.chunk_frames),
        overwrite=True,
        skip_existing=True,
        write_manifests=True,
    )


def _parallel_worker_command(
    args: argparse.Namespace,
    *,
    device: str,
    selection: EncoderSelection,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cfg",
        str(args.config),
        "--output-root",
        str(args.output_root),
        "--device",
        device,
        "--split",
        selection.split,
        "--chunk-frames",
        str(args.chunk_frames),
        "--decode-size-mode",
        str(args.decode_size_mode),
        "--decode-fit-mode",
        str(args.decode_fit_mode),
        "--shard-count",
        str(selection.shard_count),
        "--shard-index",
        str(selection.shard_index),
        "--shard-output-only",
    ]
    for source_id in selection.source_ids:
        command.extend(["--source-id", str(source_id)])
    for episode_index in selection.episode_indices:
        command.extend(["--episode-index", str(episode_index)])
    if selection.max_episodes is not None:
        command.extend(["--max-episodes", str(selection.max_episodes)])
    if args.decode_height is not None:
        command.extend(["--decode-height", str(args.decode_height)])
    if args.decode_width is not None:
        command.extend(["--decode-width", str(args.decode_width)])
    if args.decode_resize_bins is not None:
        command.extend(["--decode-resize-bins", str(args.decode_resize_bins)])
    if args.overwrite:
        command.append("--overwrite")
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def _write_shard_report(
    report: dict[str, Any],
    *,
    output_root: Path,
    shard_index: int,
    overwrite: bool,
) -> Path:
    report_dir = output_root.expanduser().resolve() / "shard_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"shard_{int(shard_index):04d}.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"Shard report already exists: {report_path}. Pass --overwrite to replace it.")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _preflight_shard_report_paths(
    output_root: Path,
    *,
    shard_indices,
    overwrite: bool,
) -> None:
    for shard_index in shard_indices:
        _preflight_shard_report_path(output_root, shard_index=int(shard_index), overwrite=overwrite)


def _preflight_shard_report_path(
    output_root: Path,
    *,
    shard_index: int,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    report_path = output_root.expanduser().resolve() / "shard_reports" / f"shard_{int(shard_index):04d}.json"
    if report_path.exists():
        raise FileExistsError(
            "Preflight failed: shard report already exists. Pass --overwrite to replace it "
            f"or --skip-existing to resume sidecars:\n- {report_path}"
        )


def encode_mixed_video_latent_sources(
    *,
    data_config: MixedVideoDataConfig,
    assets,
    output_root: Path,
    device: torch.device,
    split: str = "all",
    source_ids: tuple[str, ...] = (),
    selection: EncoderSelection | None = None,
    experiment_config=None,
    max_episodes: int | None = None,
    chunk_frames: int = 65,
    overwrite: bool = False,
    skip_existing: bool = False,
    write_manifests: bool = True,
) -> dict[str, Any]:
    if selection is None:
        selection = EncoderSelection(
            split=split,
            source_ids=tuple(source_ids),
            max_episodes=max_episodes,
        )
    if chunk_frames != 0 and chunk_frames < 5:
        raise ValueError("--chunk-frames must be 0 or at least 5.")
    output_root = output_root.expanduser().resolve()
    latents_root = output_root / "latents"
    manifests_root = output_root / "manifests"
    latents_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    all_selected_episodes = _select_encoder_episodes(data_config, selection=selection, apply_shard=False)
    episodes = _select_encoder_episodes(data_config, selection=selection, apply_shard=True)
    _preflight_output_paths(
        episodes,
        output_root=output_root,
        latents_root=latents_root,
        manifests_root=manifests_root,
        overwrite=overwrite,
        skip_existing=skip_existing,
        write_manifests=write_manifests,
    )

    canonicalizer = build_canonical_video_preprocessor(data_config) if assets is not None else None
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    encoded: list[EncodedEpisode] = []
    reused_episodes = 0
    save_executor: ThreadPoolExecutor | None = None
    save_futures: list[tuple[Path, Future]] = []
    try:
        for episode in episodes:
            latent_path = _latent_path_for_episode(latents_root, episode)
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            if skip_existing and latent_path.exists():
                record = _encoded_episode_from_existing_sidecar(latent_path, episode, data_config=data_config)
                reused_episodes += 1
            else:
                if assets is None or canonicalizer is None:
                    raise FileNotFoundError(
                        f"Missing encoded sidecar for source={episode.source_id!r}, "
                        f"episode={episode.episode_index}: {latent_path}"
                    )
                latents, encoding_metadata = _encode_episode_latents_streaming(
                    data_config,
                    episode,
                    canonicalizer=canonicalizer,
                    assets=assets,
                    device=device,
                    chunk_frames=chunk_frames,
                )
                latents = latents.detach().cpu().contiguous()[0]
                payload = {
                    LATENT_KEY: latents,
                    "metadata": {
                        "source_id": episode.source_id,
                        "dataset_id": episode.dataset_id,
                        "episode_index": episode.episode_index,
                        "clip_id": episode.clip_id,
                        "raw_length_frames": int(episode.length_frames),
                        "native_length_frames": int(episode.native_length_frames),
                        "target_observation_fps": data_config.target_observation_fps,
                        "missing_observation_fps": float(data_config.missing_observation_fps),
                        "latent_length_frames": int(latents.shape[1]),
                        "latent_shape": list(latents.shape),
                        "decode_size_mode": data_config.decode_size_mode.value,
                        "decode_fit_mode": data_config.decode_fit_mode.value,
                        "decode_allow_upscale": bool(data_config.decode_allow_upscale),
                        "decode_height": int(data_config.decode_height),
                        "decode_width": int(data_config.decode_width),
                        "decode_resize_bins": _mixed_video_transform_signature(data_config)["decode_resize_bins"],
                        "stream_transform_signature": _mixed_video_transform_signature(data_config, episode).get(
                            "streams",
                            [],
                        ),
                        "transform_signature_hash": _mixed_video_transform_signature_hash(data_config, episode),
                        **encoding_metadata,
                        "tasks": list(episode.tasks),
                    },
                }
                # WHY async save: torch.save serializes to disk synchronously which
                # blocks the next episode's decode. Keep only one pending sidecar so
                # large latent tensors cannot accumulate across the whole encode run.
                if save_executor is None:
                    save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mixed-video-save")
                _submit_latent_save(
                    save_executor=save_executor,
                    save_futures=save_futures,
                    latent_path=latent_path,
                    payload=payload,
                )
                record = EncodedEpisode(
                    source_id=episode.source_id,
                    dataset_id=episode.dataset_id,
                    episode_index=episode.episode_index,
                    clip_id=episode.clip_id,
                    latent_path=latent_path,
                    latent_shape=tuple(int(value) for value in latents.shape),
                    raw_length_frames=int(episode.length_frames),
                    latent_length_frames=int(latents.shape[1]),
                    tasks=episode.tasks,
                )
            encoded.append(record)
            manifest_path = manifests_root / f"{_safe_path_part(episode.source_id)}.csv"
            rows_by_source.setdefault(episode.source_id, []).append(
                _manifest_row_for_encoded_episode(
                    record,
                    manifest_path=manifest_path,
                    target_slot=_encoded_latent_target_slot(data_config),
                )
            )
    finally:
        if save_executor is not None:
            save_executor.shutdown(wait=True)
            save_executor = None

    # WHY drain before manifests: a manifest row must never point at a sidecar
    # whose background torch.save failed.
    for latent_path, save_future in save_futures:
        _wait_for_latent_save(latent_path, save_future)
    manifest_paths: dict[str, Path] = {}
    config_patch_path: Path | None = None
    latent_training_config_path: Path | None = None
    if write_manifests:
        if experiment_config is not None:
            _validate_encoded_records_for_backbone(encoded, experiment_config=experiment_config)
        manifest_paths = _write_source_manifests(rows_by_source, manifests_root=manifests_root, overwrite=True)
        config_patch_path = _write_latent_source_config_patch(
            data_config,
            manifest_paths=manifest_paths,
            output_root=output_root,
        )
        if experiment_config is not None:
            latent_training_config_path = _write_latent_training_config(
                experiment_config,
                data_config=data_config,
                manifest_paths=manifest_paths,
                output_root=output_root,
            )
    report = {
        "output_root": str(output_root),
        "encoded_episodes": len(encoded),
        "newly_encoded_episodes": len(encoded) - reused_episodes,
        "reused_episodes": reused_episodes,
        "selected_episodes": len(all_selected_episodes),
        "shard_episodes": len(episodes),
        "split": selection.split,
        "shard_count": int(selection.shard_count),
        "shard_index": int(selection.shard_index),
        "source_ids": sorted(rows_by_source),
        "decode_size_mode": data_config.decode_size_mode.value,
        "decode_fit_mode": data_config.decode_fit_mode.value,
        "manifest_paths": {source_id: str(path) for source_id, path in manifest_paths.items()},
        "config_patch_path": None if config_patch_path is None else str(config_patch_path),
        "latent_training_config_path": (
            None if latent_training_config_path is None else str(latent_training_config_path)
        ),
        "latent_shapes": {
            f"{record.source_id}:{record.dataset_id}:{record.episode_index}:{record.clip_id}": list(
                record.latent_shape
            )
            for record in encoded
        },
    }
    if write_manifests:
        report_path = output_root / "encode_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _selected_episode_keys(data_config: MixedVideoDataConfig, catalog, *, split: str) -> set[str]:
    if split == "all":
        return {episode.key for episode in catalog.episodes}
    train_keys, val_keys = split_mixed_video_episodes(data_config, catalog)
    return set(train_keys if split == "train" else val_keys)


def _select_encoder_episodes(
    data_config: MixedVideoDataConfig,
    *,
    selection: EncoderSelection,
    apply_shard: bool,
) -> list[MixedVideoEpisodeRecord]:
    if selection.shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {selection.shard_count}.")
    if selection.shard_index < 0 or selection.shard_index >= selection.shard_count:
        raise ValueError(
            f"shard_index must be in [0, {selection.shard_count}), got {selection.shard_index}."
        )
    catalog = load_mixed_video_catalog(data_config)
    selected_keys = _selected_episode_keys(data_config, catalog, split=selection.split)
    source_filter = set(selection.source_ids)
    episode_filter = {int(index) for index in selection.episode_indices}
    episodes = [
        episode
        for episode in catalog.episodes
        if episode.key in selected_keys
        and (not source_filter or episode.source_id in source_filter)
        and (not episode_filter or int(episode.episode_index) in episode_filter)
        and _episode_has_rgb_streams(episode)
    ]
    if selection.max_episodes is not None:
        episodes = episodes[: int(selection.max_episodes)]
    if apply_shard and selection.shard_count > 1:
        episodes = episodes[int(selection.shard_index) :: int(selection.shard_count)]
    return episodes


def _episode_has_rgb_streams(episode: MixedVideoEpisodeRecord) -> bool:
    return any(
        stream.source_format in {MixedVideoSourceFormat.RGB, MixedVideoSourceFormat.RGB_AND_LATENT}
        for stream in episode.streams
    )


def _preflight_output_paths(
    episodes: list[MixedVideoEpisodeRecord],
    *,
    output_root: Path,
    latents_root: Path,
    manifests_root: Path,
    overwrite: bool,
    skip_existing: bool = False,
    write_manifests: bool = True,
) -> None:
    latent_paths = [_latent_path_for_episode(latents_root, episode) for episode in episodes]
    source_manifest_paths = [
        manifests_root / f"{_safe_path_part(source_id)}.csv"
        for source_id in sorted({episode.source_id for episode in episodes})
    ]
    duplicate_paths = sorted(
        {
            path
            for paths in (latent_paths, source_manifest_paths)
            for path, count in Counter(paths).items()
            if count > 1
        }
    )
    if duplicate_paths:
        formatted = "\n".join(f"- {path}" for path in duplicate_paths)
        raise FileExistsError(f"Preflight failed: multiple selected episodes would write the same path:\n{formatted}")
    if overwrite:
        return
    checked_paths = [] if skip_existing else list(latent_paths)
    if write_manifests and not skip_existing:
        checked_paths.extend(
            [
                output_root / "encode_report.json",
                output_root / "latent_training_sources.yaml",
                output_root / "latent_training_config.yaml",
            ]
        )
        checked_paths.extend(source_manifest_paths)
    existing_paths = [path for path in checked_paths if path.exists()]
    if existing_paths:
        formatted = "\n".join(f"- {path}" for path in existing_paths)
        raise FileExistsError(
            "Preflight failed: output paths already exist. Pass --overwrite to replace them "
            f"or --skip-existing to resume sidecars:\n{formatted}"
        )


def _encoded_episode_from_existing_sidecar(
    latent_path: Path,
    episode: MixedVideoEpisodeRecord,
    *,
    data_config: MixedVideoDataConfig,
) -> EncodedEpisode:
    payload = torch.load(latent_path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        latents = payload
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict) and LATENT_KEY in payload:
        latents = payload[LATENT_KEY]
        raw_metadata = payload.get("metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    else:
        raise KeyError(f"Encoded sidecar must contain {LATENT_KEY!r}: {latent_path}")
    if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
        raise ValueError(f"Expected sidecar {latent_path} to contain [C, T, H, W] latents.")
    latent_shape = tuple(int(value) for value in latents.shape)
    _validate_existing_sidecar_metadata(
        latent_path,
        metadata=metadata,
        episode=episode,
        data_config=data_config,
        latent_shape=latent_shape,
    )
    return EncodedEpisode(
        source_id=episode.source_id,
        dataset_id=episode.dataset_id,
        episode_index=int(episode.episode_index),
        clip_id=episode.clip_id,
        latent_path=latent_path,
        latent_shape=latent_shape,
        raw_length_frames=int(episode.length_frames),
        latent_length_frames=int(latents.shape[1]),
        tasks=episode.tasks,
    )


def _validate_existing_sidecar_metadata(
    latent_path: Path,
    *,
    metadata: dict[str, Any],
    episode: MixedVideoEpisodeRecord,
    data_config: MixedVideoDataConfig,
    latent_shape: tuple[int, int, int, int],
) -> None:
    expected_fields = {
        "source_id": str(episode.source_id),
        "dataset_id": str(episode.dataset_id),
        "clip_id": str(episode.clip_id),
        "decode_size_mode": data_config.decode_size_mode.value,
        "decode_fit_mode": data_config.decode_fit_mode.value,
        "decode_allow_upscale": str(bool(data_config.decode_allow_upscale)),
        "target_observation_fps": str(data_config.target_observation_fps),
        "missing_observation_fps": str(float(data_config.missing_observation_fps)),
        "transform_signature_hash": _mixed_video_transform_signature_hash(data_config, episode),
    }
    for field_name, expected in expected_fields.items():
        if field_name not in metadata:
            raise ValueError(
                f"Existing sidecar metadata missing required field {field_name!r} in {latent_path}. "
                "Re-run without --skip-existing or pass --overwrite to regenerate it."
            )
        if str(metadata[field_name]) != expected:
            raise ValueError(
                f"Existing sidecar metadata mismatch for {latent_path}: "
                f"{field_name}={metadata[field_name]!r}, expected {expected!r}."
            )
    int_fields = {
        "episode_index": int(episode.episode_index),
        "raw_length_frames": int(episode.length_frames),
        "native_length_frames": int(episode.native_length_frames),
        "latent_length_frames": int(latent_shape[1]),
    }
    for field_name, expected in {
        "decode_height": int(data_config.decode_height),
        "decode_width": int(data_config.decode_width),
    }.items():
        if field_name in metadata:
            int_fields[field_name] = expected
    for field_name, expected in int_fields.items():
        if field_name not in metadata:
            continue
        try:
            actual = int(metadata[field_name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Existing sidecar metadata field {field_name!r} is not an integer in {latent_path}: "
                f"{metadata[field_name]!r}."
            ) from error
        if actual != expected:
            raise ValueError(
                f"Existing sidecar metadata mismatch for {latent_path}: "
                f"{field_name}={actual}, expected {expected}."
            )
    if "latent_shape" in metadata:
        try:
            metadata_shape = tuple(int(value) for value in metadata["latent_shape"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Existing sidecar metadata field 'latent_shape' is invalid in {latent_path}: "
                f"{metadata['latent_shape']!r}."
            ) from error
        if metadata_shape != latent_shape:
            raise ValueError(
                f"Existing sidecar latent_shape metadata mismatch for {latent_path}: "
                f"{metadata_shape}, actual {latent_shape}."
            )
    if "decode_resize_bins" in metadata:
        expected_bins = _mixed_video_transform_signature(data_config)["decode_resize_bins"]
        if metadata["decode_resize_bins"] != expected_bins:
            raise ValueError(
                f"Existing sidecar decode_resize_bins metadata mismatch for {latent_path}; "
                "re-run without --skip-existing or pass --overwrite to regenerate it."
            )


def _encode_episode_latents_streaming(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    canonicalizer,
    assets,
    device: torch.device,
    chunk_frames: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode one episode with CPU/GPU overlap double buffering.

    WHY double buffer: CPU video decode and GPU VAE encode are independent
    workloads. By decoding chunk N+1 on CPU while chunk N encodes on GPU,
    we overlap ~60% of decode time with encode time, yielding ~1.5-2x speedup.
    """
    latent_chunks: list[torch.Tensor] = []
    placements_metadata: list[dict[str, Any]] | None = None
    canonical_shape: list[int] | None = None
    raw_chunk_ranges = _streaming_chunk_ranges(int(episode.length_frames), max_chunk_frames=chunk_frames)
    chunk_iterator = enumerate(
        _iter_episode_view_chunks(data_config, episode, raw_chunk_ranges=raw_chunk_ranges)
    )

    # WHY queue maxsize=2: one slot for the chunk being GPU-encoded, one for
    # the prefetched next chunk. Larger queues waste CPU memory on decoded
    # frames without additional GPU overlap benefit.
    prefetch_queue: queue.Queue = queue.Queue(maxsize=2)
    prefetch_error: list[Exception] = []

    def _prefetch_worker():
        """Background thread: decode + canonicalize chunks on CPU, push to queue."""
        try:
            for chunk_index, views in chunk_iterator:
                canonical = canonicalizer(views)
                # WHY keep on CPU here: CUDA init from a background thread fails
                # on nodes with older drivers. H2D transfer happens in main thread.
                prefetch_queue.put((chunk_index, canonical.video, canonical.placements))
        except Exception as exc:
            prefetch_error.append(exc)
        finally:
            prefetch_queue.put(None)  # sentinel

    # WHY daemon=True: if main thread crashes, prefetch thread dies immediately
    worker = threading.Thread(target=_prefetch_worker, daemon=True)
    worker.start()

    while True:
        item = prefetch_queue.get()
        if item is None:
            break
        chunk_index, video_cpu, placements = item
        # WHY .to(device) in main thread: avoids CUDA init in background thread
        # which fails on nodes with old drivers (CUDA 12030)
        video_on_device = video_cpu.to(device=device)
        if placements_metadata is None:
            placements_metadata = [asdict(placement) for placement in placements]
            canonical_shape = list(video_on_device.shape[1:])
        latent_chunks.append(
            assets.encode_video(
                video_on_device,
                placements=placements,
                reset_cache=chunk_index == 0,
            ).detach()
        )

    worker.join()
    if prefetch_error:
        raise prefetch_error[0]

    if not latent_chunks:
        raise ValueError(f"No latent chunks were produced for mixed-video episode {episode.key!r}.")
    metadata = {
        "canonical_shape": canonical_shape,
        "placements": placements_metadata or [],
        "raw_chunk_ranges": [list(item) for item in raw_chunk_ranges],
        "native_length_frames": int(episode.native_length_frames),
        "normalized_length_frames": int(episode.length_frames),
        "target_observation_fps": data_config.target_observation_fps,
        "chunk_frames": int(chunk_frames),
    }
    return torch.cat(latent_chunks, dim=2), metadata


def _iter_episode_view_chunks(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    raw_chunk_ranges: tuple[tuple[int, int], ...],
) -> Iterator[dict[str, torch.Tensor]]:
    streams_by_slot: dict[str, MixedVideoStreamRecord] = {}
    for stream in sorted(episode.streams, key=lambda item: item.stream_index):
        streams_by_slot.setdefault(stream.target_slot, stream)

    stream_iterators: dict[str, Iterator[torch.Tensor]] = {}
    for camera_name in data_config.camera_names:
        stream = streams_by_slot.get(camera_name)
        if stream is None:
            raise KeyError(
                f"Cannot encode mixed-video episode {episode.key!r}: missing RGB stream for slot {camera_name!r}."
            )
        if stream.source_format not in {MixedVideoSourceFormat.RGB, MixedVideoSourceFormat.RGB_AND_LATENT}:
            raise ValueError(
                f"Cannot encode source={stream.source_id!r}, episode={stream.episode_index}, "
                f"stream={stream.stream_key}: source_format={stream.source_format.value!r} has no RGB input."
            )
        stream_iterators[camera_name] = iter_mixed_video_stream_frame_chunks(
            data_config,
            stream,
            raw_chunk_ranges=raw_chunk_ranges,
        )

    for _ in raw_chunk_ranges:
        yield {
            camera_name: next(stream_iterator)
            for camera_name, stream_iterator in stream_iterators.items()
        }


def _decode_episode_view_chunk(
    data_config: MixedVideoDataConfig,
    episode: MixedVideoEpisodeRecord,
    *,
    start_frame: int,
    end_frame: int,
) -> dict[str, torch.Tensor]:
    iterator = _iter_episode_view_chunks(
        data_config,
        episode,
        raw_chunk_ranges=((int(start_frame), int(end_frame)),),
    )
    return next(iterator)


def _streaming_chunk_ranges(num_frames: int, *, max_chunk_frames: int) -> tuple[tuple[int, int], ...]:
    if num_frames <= 0:
        raise ValueError(f"Cannot encode an empty video: num_frames={num_frames}.")
    latent_frames = wan_raw_frame_count_to_latent_count(num_frames)
    encoded_raw_frames = 1 + 4 * (latent_frames - 1)
    if max_chunk_frames == 0 or encoded_raw_frames <= max_chunk_frames:
        return ((0, encoded_raw_frames),)
    if max_chunk_frames < 5:
        raise ValueError("max_chunk_frames must be 0 or at least 5.")
    first_capacity = 1 + 4 * ((max_chunk_frames - 1) // 4)
    stream_capacity = 4 * (max_chunk_frames // 4)
    ranges: list[tuple[int, int]] = []
    first_end = min(encoded_raw_frames, first_capacity)
    ranges.append((0, first_end))
    start = first_end
    while start < encoded_raw_frames:
        end = min(encoded_raw_frames, start + stream_capacity)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _latent_path_for_episode(latents_root: Path, episode: MixedVideoEpisodeRecord) -> Path:
    source = _safe_path_part(episode.source_id)
    dataset = _safe_path_part(episode.dataset_id)
    suffix = "" if episode.clip_id == "default" else f"_{_safe_path_part(episode.clip_id)}"
    return latents_root / source / dataset / f"episode_{int(episode.episode_index):06d}{suffix}.pt"


def _manifest_row_for_encoded_episode(
    record: EncodedEpisode,
    *,
    manifest_path: Path,
    target_slot: str,
) -> dict[str, Any]:
    return {
        "source_id": record.source_id,
        "dataset_id": record.dataset_id,
        "episode_index": int(record.episode_index),
        "clip_id": record.clip_id,
        "stream_index": 0,
        "stream_key": "encoded_video_latents",
        "target_slot_key": target_slot,
        "latent_path": os.path.relpath(record.latent_path, manifest_path.parent),
        "latent_length_frames": int(record.latent_length_frames),
        "length_frames": int(record.latent_length_frames),
        "raw_length_frames": int(record.raw_length_frames),
        "latent_key": LATENT_KEY,
        "width": int(record.latent_shape[-1]),
        "height": int(record.latent_shape[-2]),
        "channels": int(record.latent_shape[0]),
        "tasks": "|".join(record.tasks),
    }


def _encoded_latent_target_slot(data_config: MixedVideoDataConfig) -> str:
    if not data_config.camera_names:
        raise ValueError("Mixed-video latent encoding requires at least one configured camera slot.")
    return data_config.camera_names[0]


def _encoded_latent_view_layout(data_config: MixedVideoDataConfig) -> dict[str, Any]:
    target_slot = _encoded_latent_target_slot(data_config)
    return {
        "source_name": target_slot,
        "canonical_name": target_slot,
        "top": 0,
        "left": 0,
        "height": int(data_config.canonical_height),
        "width": int(data_config.canonical_width),
    }


def _write_source_manifests(
    rows_by_source: dict[str, list[dict[str, Any]]],
    *,
    manifests_root: Path,
    overwrite: bool,
) -> dict[str, Path]:
    manifest_paths: dict[str, Path] = {}
    fieldnames = (
        "source_id",
        "dataset_id",
        "episode_index",
        "clip_id",
        "stream_index",
        "stream_key",
        "target_slot_key",
        "latent_path",
        "latent_length_frames",
        "length_frames",
        "raw_length_frames",
        "latent_key",
        "width",
        "height",
        "channels",
        "tasks",
    )
    for source_id, rows in sorted(rows_by_source.items()):
        manifest_path = manifests_root / f"{_safe_path_part(source_id)}.csv"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"Manifest already exists: {manifest_path}. Pass --overwrite to replace it.")
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        manifest_paths[source_id] = manifest_path
    return manifest_paths


def _write_latent_source_config_patch(
    data_config: MixedVideoDataConfig,
    *,
    manifest_paths: dict[str, Path],
    output_root: Path,
) -> Path:
    source_by_id = {source.source_id: source for source in data_config.video_sources}
    latent_num_frames = wan_raw_frame_count_to_latent_count(int(data_config.num_frames))
    latent_buckets = _latent_causal_bucket_specs(data_config)
    target_slot = _encoded_latent_target_slot(data_config)
    latent_view_layout = _encoded_latent_view_layout(data_config)
    lines = [
        "# Include this block in a mixed-video latent-first training config.",
        "# The generated manifests use latent-frame units for length_frames.",
        "# These num_frames/bucket values are converted from the RGB/WAN raw-frame config.",
        "data:",
        f"  camera_names: [{json.dumps(target_slot)}]",
        f"  latent_camera_names: [{json.dumps(target_slot)}]",
        f"  canonical_height: {int(data_config.canonical_height)}",
        f"  canonical_width: {int(data_config.canonical_width)}",
        "  view_layout:",
        f"    - source_name: {json.dumps(latent_view_layout['source_name'])}",
        f"      canonical_name: {json.dumps(latent_view_layout['canonical_name'])}",
        f"      top: {latent_view_layout['top']}",
        f"      left: {latent_view_layout['left']}",
        f"      height: {latent_view_layout['height']}",
        f"      width: {latent_view_layout['width']}",
        f"  num_frames: {latent_num_frames}",
        "  frame_stride: 1",
        "  sample_stride: 1",
        "  video_sources:",
    ]
    for source_id, manifest_path in sorted(manifest_paths.items()):
        source = source_by_id.get(source_id)
        sampling_weight = None if source is None else source.sampling_weight
        lines.extend(
            [
                f"    - source_id: {source_id}",
                f"      manifest_csv: {manifest_path}",
                "      source_format: latent",
                f"      latent_key: {LATENT_KEY}",
            ]
        )
        if sampling_weight is not None:
            lines.append(f"      sampling_weight: {float(sampling_weight)}")
    lines.extend(
        [
            "  sample_construction:",
            "    mode: causal_prefix_suffix",
            f"    num_frames: {latent_num_frames}",
            "    action_horizon: 0",
            "    state_horizon: 0",
            "    frame_stride: 1",
            "    causal_prefix_suffix_buckets:",
        ]
    )
    for bucket in latent_buckets:
        lines.extend(
            [
                f"      # raw {bucket['raw_observed_frames']} + {bucket['raw_future_frames']} "
                f"frames -> latent {bucket['observed_frames']} + {bucket['future_frames']} frames",
                f"      - observed_frames: {bucket['observed_frames']}",
                f"        future_frames: {bucket['future_frames']}",
            ]
        )
    lines.extend(
        [
            "trainer:",
            "  batch_adapter: latents",
            "",
        ]
    )
    patch_path = output_root / "latent_training_sources.yaml"
    patch_path.write_text("\n".join(lines), encoding="utf-8")
    return patch_path


def _write_latent_training_config(
    experiment_config,
    *,
    data_config: MixedVideoDataConfig,
    manifest_paths: dict[str, Path],
    output_root: Path,
) -> Path:
    import yaml

    payload = serialize_enum_values(asdict(experiment_config))
    payload["name"] = f"{payload.get('name', 'mixed_video')}_latent_encoded"

    data_payload = dict(payload.get("data", {}))
    latent_num_frames = wan_raw_frame_count_to_latent_count(int(data_config.num_frames))
    target_slot = _encoded_latent_target_slot(data_config)
    data_payload.update(
        {
            "dataset_name": data_config.dataset_name,
            "dataset_type": data_config.dataset_type,
            "camera_names": [target_slot],
            "latent_camera_names": [target_slot],
            "canonical_height": int(data_config.canonical_height),
            "canonical_width": int(data_config.canonical_width),
            "view_layout": [_encoded_latent_view_layout(data_config)],
            "num_frames": latent_num_frames,
            "frame_stride": 1,
            "sample_stride": 1,
            "train_batch_size": 1,
            "val_batch_size": 1,
            "video_sources": _latent_video_source_config_entries(
                data_config,
                manifest_paths=manifest_paths,
            ),
        }
    )
    sample_construction = dict(data_payload.get("sample_construction", {}))
    sample_construction.update(
        {
            "mode": "causal_prefix_suffix",
            "num_frames": latent_num_frames,
            "action_horizon": 0,
            "state_horizon": 0,
            "frame_stride": 1,
            "causal_prefix_suffix_buckets": [
                {
                    "observed_frames": int(bucket["observed_frames"]),
                    "future_frames": int(bucket["future_frames"]),
                }
                for bucket in _latent_causal_bucket_specs(data_config)
            ],
        }
    )
    data_payload["sample_construction"] = sample_construction
    payload["data"] = data_payload

    trainer_payload = dict(payload.get("trainer", {}))
    trainer_payload["batch_adapter"] = "latents"
    payload["trainer"] = trainer_payload

    backbone_payload = dict(payload.get("backbone", {}))
    # Latent-first training enters after VAE encoding, so loading the VAE again
    # is unnecessary and can fail on machines that only have precomputed sidecars.
    backbone_payload["load_wan_vae_frontend"] = False
    payload["backbone"] = backbone_payload

    config_path = output_root / "latent_training_config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _latent_video_source_config_entries(
    data_config: MixedVideoDataConfig,
    *,
    manifest_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    source_by_id = {source.source_id: source for source in data_config.video_sources}
    entries: list[dict[str, Any]] = []
    for source_id, manifest_path in sorted(manifest_paths.items()):
        source = source_by_id.get(source_id)
        entry: dict[str, Any] = {
            "source_id": source_id,
            "manifest_csv": str(manifest_path),
            "source_format": "latent",
            "latent_key": LATENT_KEY,
        }
        if source is not None and source.sampling_weight is not None:
            entry["sampling_weight"] = float(source.sampling_weight)
        entries.append(entry)
    return entries


def _validate_encoded_records_for_backbone(
    encoded: list[EncodedEpisode],
    *,
    experiment_config,
) -> None:
    patch_t = int(getattr(experiment_config.backbone, "patch_size_t", 1))
    patch_h = int(getattr(experiment_config.backbone, "patch_size_h", 1))
    patch_w = int(getattr(experiment_config.backbone, "patch_size_w", 1))
    errors: list[str] = []
    for record in encoded:
        _, latent_frames, latent_height, latent_width = record.latent_shape
        if (
            int(latent_frames) % patch_t != 0
            or int(latent_height) % patch_h != 0
            or int(latent_width) % patch_w != 0
        ):
            errors.append(
                f"{record.source_id}:{record.dataset_id}:{record.episode_index}:{record.clip_id} "
                f"shape={record.latent_shape} patch={(patch_t, patch_h, patch_w)}"
            )
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(
            "Encoded mixed-video latents are not compatible with the configured shared-transformer patch size. "
            "Re-encode with patch-compatible resize bins, or change backbone.patch_size_* before training:\n"
            f"{formatted}"
        )


def _latent_causal_bucket_specs(data_config: MixedVideoDataConfig) -> list[dict[str, int]]:
    raw_buckets = data_config.sample_construction.causal_prefix_suffix_buckets
    if not raw_buckets:
        raw_observed = max(1, int(data_config.num_frames) // 2)
        raw_buckets = (
            CausalPrefixSuffixBucketConfig(
                observed_frames=raw_observed,
                future_frames=int(data_config.num_frames) - raw_observed,
            ),
        )
    specs: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for bucket in raw_buckets:
        raw_observed = int(bucket.observed_frames)
        raw_future = int(bucket.future_frames)
        try:
            mapping = VideoFrameMapping.wan_causal_prefix_suffix(
                raw_observed_frames=raw_observed,
                raw_future_frames=raw_future,
            )
        except ValueError:
            continue
        observed = mapping.observed_frames
        future = mapping.future_frames
        key = (observed, future)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "raw_observed_frames": raw_observed,
                "raw_future_frames": raw_future,
                "observed_frames": observed,
                "future_frames": future,
            }
        )
    if not specs:
        raise ValueError("No causal prefix/suffix buckets remain valid after raw-to-WAN-latent conversion.")
    return specs


def _load_resize_bins(path: Path) -> tuple[MixedVideoResizeBinConfig, ...]:
    raw_text = path.expanduser().read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(raw_text)
    else:
        raw = json.loads(raw_text)
    if not isinstance(raw, list):
        raise ValueError(f"Expected resize-bin file to contain a list, got {type(raw)!r}.")
    return tuple(
        item if isinstance(item, MixedVideoResizeBinConfig) else MixedVideoResizeBinConfig(**item)
        for item in raw
    )


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
