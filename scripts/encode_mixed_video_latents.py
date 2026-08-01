from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import (  # noqa: E402
    ExperimentConfig,
    MixedVideoDataConfig,
    MixedVideoResizeBinConfig,
    load_experiment_config,
)
import open_wam.data.mixed_video_encoding as _encoding  # noqa: E402
from open_wam.models.visual_tower.reference_assets import LingbotReferenceAssets  # noqa: E402


# Compatibility aliases for callers that historically imported this script.
EncodedEpisode = _encoding.EncodedEpisode
EncodingTarget = _encoding.EncodingTarget
EncoderSelection = _encoding.EncoderSelection
encode_mixed_video_latent_sources = _encoding.encode_mixed_video_latent_sources


def __getattr__(name: str) -> Any:
    return getattr(_encoding, name)


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


def resolve_encoder_data_config(
    data_config: MixedVideoDataConfig,
    *,
    decode_size_mode: str,
    decode_fit_mode: str,
    decode_height: int | None = None,
    decode_width: int | None = None,
    decode_resize_bins_path: str | None = None,
) -> MixedVideoDataConfig:
    resize_bins = (
        None
        if decode_resize_bins_path is None
        else _load_resize_bins(Path(decode_resize_bins_path))
    )
    return _encoding.resolve_mixed_video_encoding_config(
        data_config,
        decode_size_mode=None if decode_size_mode == "config" else decode_size_mode,
        decode_fit_mode=None if decode_fit_mode == "config" else decode_fit_mode,
        decode_height=decode_height,
        decode_width=decode_width,
        decode_resize_bins=resize_bins,
    )


def _write_shard_report(
    report: Mapping[str, Any],
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
    shard_indices: Iterable[int],
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


def launch_parallel_mixed_video_encoding(
    *,
    args: argparse.Namespace,
    experiment_config: ExperimentConfig,
    data_config: MixedVideoDataConfig,
    selection: EncoderSelection,
) -> _encoding.MixedVideoEncodingReport:
    devices = tuple(device.strip() for device in str(args.devices).split(",") if device.strip())
    if not devices:
        raise ValueError("--devices must list at least one device when provided.")
    output_root = Path(args.output_root).expanduser().resolve()
    latents_root = output_root / "latents"
    manifests_root = output_root / "manifests"
    latents_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    all_episodes = _encoding.select_mixed_video_encoding_episodes(
        data_config,
        selection=selection,
        apply_shard=False,
    )
    _encoding.preflight_mixed_video_encoding_outputs(
        all_episodes,
        data_config=data_config,
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
    selected_episodes = _encoding.select_mixed_video_encoding_episodes(
        data_config,
        selection=selection,
        apply_shard=True,
    )
    all_selected_sidecars_exist = all(
        _encoding.resolve_existing_mixed_video_encoding_target(target) is not None
        for episode in selected_episodes
        for target in _encoding.plan_mixed_video_episode_encoding_targets(
            output_root / "latents",
            episode,
            data_config,
        )
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


if __name__ == "__main__":
    raise SystemExit(main())
