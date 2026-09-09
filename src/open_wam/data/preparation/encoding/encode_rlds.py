#!/usr/bin/env python3
"""Encode RLDS/TFRecord episode videos into Wan 2.2 VAE latents, one camera at a time.

The companion script reads LeRobot repos, which store one mp4 per episode. RLDS
stores whole episodes as TFRecord examples with the frames inline as an image
sequence, so it needs a different reader — but everything downstream of "here are
this episode's frames" is shared, and imported from `encode_latents`.

Output layout matches the LeRobot path exactly:

    <out-root>/<dataset>/latents/chunk-{chunk:03d}/<camera>/episode_{index:06d}_{start}_{end}.pth

A caveat worth knowing before using the result: most RLDS robot datasets were
recorded at low control rates (3-10 Hz), not at video frame rates. Resampling
those up to 15 fps duplicates frames rather than inventing motion, so the
"video" a model sees is mostly held frames. Pass --source-fps to state the real
rate; without it the rate has to come from --fps-table or the run refuses to
guess.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from open_wam.configs.data_mixed_video import MixedVideoResizeBinConfig
from open_wam.data.preparation.encoding.payload import DTYPES
from open_wam.data.mixed_video_decode_frames import select_mixed_video_resize_bin
from open_wam.data.preparation.frames import (
    DEFAULT_BINS,
    FIT_MODES,
    fit_frames,
    resample_indices,
    truncate_to_temporal_stride,
)
from open_wam.models.visual_tower.vae_encoding import encode_batch, load_vae

# Control rates for the Open X-Embodiment mixtures, which RLDS metadata does not
# record. Values follow the dataset papers; anything absent must be passed with
# --source-fps rather than assumed.
DEFAULT_FPS_TABLE = {
    "austin_buds_dataset_converted_externally_to_rlds": 20.0,
    "austin_sailor_dataset_converted_externally_to_rlds": 20.0,
    "austin_sirius_dataset_converted_externally_to_rlds": 20.0,
    "bc_z": 10.0,
    "berkeley_autolab_ur5": 5.0,
    "berkeley_cable_routing": 10.0,
    "berkeley_fanuc_manipulation": 10.0,
    "bridge": 5.0,
    "cmu_stretch": 10.0,
    "dlr_edan_shared_control_converted_externally_to_rlds": 5.0,
    "dobbe": 30.0,
    "droid": 15.0,
    "fmb": 10.0,
    "fractal20220817_data": 3.0,
    "furniture_bench_dataset_converted_externally_to_rlds": 10.0,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 20.0,
    "jaco_play": 10.0,
    "kuka": 10.0,
    "language_table": 10.0,
    "nyu_franka_play_dataset_converted_externally_to_rlds": 3.0,
    "roboturk": 10.0,
    "stanford_hydra_dataset_converted_externally_to_rlds": 10.0,
    "taco_play": 15.0,
    "toto": 30.0,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": 2.0,
    "utaustin_mutex": 20.0,
    "viola": 20.0,
}


def discover_datasets(root: Path) -> list[Path]:
    """An RLDS dataset directory holds a version subdirectory with features.json."""

    if list(root.glob("*/features.json")):
        return [root]
    found = sorted({p.parent.parent for p in root.glob("*/*/features.json")})
    if not found:
        raise FileNotFoundError(f"No RLDS dataset (*/features.json) under {root}")
    return found


def version_dir(dataset: Path) -> Path:
    versions = sorted(p.parent for p in dataset.glob("*/features.json"))
    if not versions:
        raise FileNotFoundError(f"{dataset}: no version directory with features.json")
    return versions[-1]


def image_features(dataset: Path) -> list[str]:
    """Names of the per-step image fields, read from the dataset's own schema.

    Field names vary a lot across RLDS datasets — `image`, `image_0`,
    `exterior_image_1_left` — so they are discovered rather than assumed.
    """

    schema = json.loads((version_dir(dataset) / "features.json").read_text())
    names: list[str] = []

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if "Image" in str(node.get("pythonClassName", "")):
            # .../observation/featuresDict/features/<name>
            parts = [
                p
                for p in path.split("/")
                if p not in {"featuresDict", "features", "sequence", "feature"}
            ]
            if parts:
                names.append(parts[-1])
            return
        for key, value in node.items():
            walk(value, f"{path}/{key}")

    walk(schema, "")
    return sorted(dict.fromkeys(names))


def episode_frames(episode, camera: str) -> np.ndarray:
    """Stack one camera's frames for one RLDS episode into uint8 [N, H, W, 3]."""

    frames = [step["observation"][camera].numpy() for step in episode["steps"]]
    if not frames:
        raise ValueError("episode has no steps")
    return np.stack(frames)


def resolve_fps(name: str, args: argparse.Namespace, table: dict[str, float]) -> float:
    if args.source_fps is not None:
        return float(args.source_fps)
    if name in table:
        return float(table[name])
    raise KeyError(
        f"{name}: no control rate known. RLDS does not record one, and guessing it "
        f"would silently mis-time every clip. Pass --source-fps, or add an entry via --fps-table."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="An RLDS dataset dir, or a parent of several",
    )
    parser.add_argument("--vae", help="Wan 2.2 VAE directory")
    parser.add_argument("--out-root", help="Where latent trees go")
    parser.add_argument(
        "--fps", type=float, default=15.0, help="Target frame rate (default: 15)"
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Override the source control rate",
    )
    parser.add_argument(
        "--fps-table",
        type=Path,
        default=None,
        help="JSON mapping dataset name to control rate",
    )
    parser.add_argument(
        "--split", default="train", help="RLDS split to read (default: train)"
    )
    parser.add_argument(
        "--size-mode", choices=("aspect_bins", "fixed"), default="aspect_bins"
    )
    parser.add_argument("--fit-mode", choices=FIT_MODES, default="letterbox_pad")
    parser.add_argument(
        "--resolution", type=int, default=256, help="Square size for --size-mode fixed"
    )
    parser.add_argument(
        "--cameras", nargs="*", default=None, help="Subset of image fields"
    )
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Episodes per chunk directory"
    )
    parser.add_argument("--latent-subdir", default="latents")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--store-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plan", action="store_true", help="Report what would be encoded and exit"
    )
    args = parser.parse_args()

    table = dict(DEFAULT_FPS_TABLE)
    if args.fps_table:
        table.update(json.loads(args.fps_table.read_text()))

    datasets = discover_datasets(args.dataset)

    if args.plan:
        print("%-56s %10s %8s %28s" % ("dataset", "src fps", "cameras", "image fields"))
        print("-" * 110)
        for dataset in datasets:
            cameras = image_features(dataset)
            try:
                fps = resolve_fps(dataset.name, args, table)
                rate = f"{fps:g}"
            except KeyError:
                rate = "UNKNOWN"
            print(
                "%-56s %10s %8d %28s"
                % (dataset.name[:56], rate, len(cameras), ",".join(cameras)[:28])
            )
        print("-" * 110)
        print(f"{len(datasets)} datasets")
        return 0

    for required in ("vae", "out_root"):
        if getattr(args, required) is None:
            parser.error(
                f"--{required.replace('_', '-')} is required unless --plan is given"
            )

    import tensorflow_datasets as tfds  # heavy; only needed for a real run

    device = torch.device(args.device)
    vae = load_vae(args.vae, device, DTYPES[args.dtype])
    vae.disable_slicing()
    temporal_stride = int(vae.config.scale_factor_temporal)

    print(f"vae      : {args.vae}")
    print(
        f"target   : {args.fps} fps, {args.fit_mode}, per-camera, {args.store_dtype} on disk"
    )
    print(f"in       : {args.dataset}  ({len(datasets)} datasets)")
    print(f"out      : {args.out_root}")

    written = skipped = failed = 0
    started = time.monotonic()

    for dataset in datasets:
        source_fps = resolve_fps(dataset.name, args, table)
        cameras = [
            c
            for c in image_features(dataset)
            if args.cameras is None or c in args.cameras
        ]
        if not cameras:
            print(f"  {dataset.name}: no matching image fields", file=sys.stderr)
            continue

        builder = tfds.builder_from_directory(str(version_dir(dataset)))
        total = builder.info.splits[args.split].num_examples
        first = max(0, args.start_episode)
        last = total if args.episodes is None else min(first + args.episodes, total)
        print(
            f"\n  {dataset.name}: episodes [{first}, {last}) of {total} "
            f"x {len(cameras)} cameras @ {source_fps} fps -> {cameras}"
        )
        if first >= last:
            continue

        read = builder.as_dataset(split=f"{args.split}[{first}:{last}]")
        pending: dict[tuple[int, int], list] = {}

        def flush(key):
            nonlocal written, failed
            group = pending.pop(key, None)
            if not group:
                return
            try:
                latents = encode_batch(
                    vae, [g["video"] for g in group], normalize=not args.no_normalize
                )
            except Exception as failure:
                print(
                    f"    ! batch of {len(group)} at {key}: {failure}", file=sys.stderr
                )
                failed += len(group)
                return
            for prepared, latent in zip(group, latents):
                tensor = latent.to(DTYPES[args.store_dtype])
                payload = {
                    "latent": tensor,
                    "latent_num_frames": int(tensor.shape[0]),
                    "latent_height": int(tensor.shape[1]),
                    "latent_width": int(tensor.shape[2]),
                    "frame_ids": prepared["indices"],
                    "start_frame": 0,
                    "end_frame": prepared["source_frames"],
                    "video_num_frames": len(prepared["indices"]),
                    "fps": float(args.fps),
                    "ori_fps": float(source_fps),
                    "video_height": prepared["bin"].target_height,
                    "video_width": prepared["bin"].target_width,
                    "resize_bin": prepared["bin"].name,
                    "camera": prepared["camera"],
                    "episode_index": prepared["episode"],
                    "fit_mode": args.fit_mode,
                    "latents_normalized": not args.no_normalize,
                    "vae_path": str(args.vae),
                    "source_format": "rlds",
                    "split": args.split,
                }
                out_path = (
                    Path(args.out_root)
                    / dataset.name
                    / args.latent_subdir
                    / f"chunk-{prepared['episode'] // args.chunk_size:03d}"
                    / prepared["camera"]
                    / f"episode_{prepared['episode']:06d}_0_{prepared['source_frames']}.pth"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, out_path)
                written += 1
                if written == 1 or written % 50 == 0:
                    rate = written / max(1e-9, time.monotonic() - started)
                    print(
                        f"    [{written}] ep{prepared['episode']} {prepared['camera']} "
                        f"-> latent {tuple(tensor.shape)} ({rate:.1f}/s)"
                    )

        for offset, episode in enumerate(read):
            index = first + offset
            for camera in cameras:
                out_dir = (
                    Path(args.out_root)
                    / dataset.name
                    / args.latent_subdir
                    / f"chunk-{index // args.chunk_size:03d}"
                    / camera
                )
                if not args.overwrite and list(
                    out_dir.glob(f"episode_{index:06d}_*.pth")
                ):
                    skipped += 1
                    continue
                try:
                    frames = episode_frames(episode, camera)
                    indices = resample_indices(len(frames), source_fps, args.fps)
                    keep = truncate_to_temporal_stride(len(indices), temporal_stride)
                    if keep < 1:
                        raise ValueError(
                            f"{len(frames)} frames @ {source_fps} fps is too short"
                        )
                    indices = indices[:keep]
                    if args.size_mode == "aspect_bins":
                        bin_config = select_mixed_video_resize_bin(
                            DEFAULT_BINS, source_height=frames.shape[1], source_width=frames.shape[2]
                        )
                    else:
                        bin_config = MixedVideoResizeBinConfig(
                            "fixed", 1, 1, args.resolution, args.resolution
                        )
                    video = fit_frames(
                        frames[indices],
                        bin_config.target_height,
                        bin_config.target_width,
                        args.fit_mode,
                    )
                except Exception as failure:
                    print(f"    ! ep{index} {camera}: {failure}", file=sys.stderr)
                    failed += 1
                    continue

                key = (bin_config.target_height, bin_config.target_width)
                pending.setdefault(key, []).append(
                    {
                        "video": video,
                        "indices": [int(i) for i in indices],
                        "source_frames": int(len(frames)),
                        "bin": bin_config,
                        "camera": camera,
                        "episode": index,
                    }
                )
                if len(pending[key]) >= args.batch_size:
                    flush(key)

        for key in list(pending):
            flush(key)

    elapsed = time.monotonic() - started
    print(
        f"\nwrote {written}, skipped {skipped}, failed {failed} in {elapsed / 60:.1f} min"
    )
    return 1 if failed and not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
