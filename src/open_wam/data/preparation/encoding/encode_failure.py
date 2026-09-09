"""Encode RoboMind failure HDF5 JPEG RGB separately from official embodiment color rules."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from open_wam.data.preparation.encoding.robomind_metadata import read_native_text
from open_wam.data.preparation.encoding.payload import build_payload
from open_wam.data.mixed_video_decode_frames import select_mixed_video_resize_bin
from open_wam.data.preparation.frames import (
    DEFAULT_BINS,
    fit_frames,
    resample_indices,
    truncate_to_temporal_stride,
)
from open_wam.data.preparation.prepare_dataset import make_row
from open_wam.data.preparation.storage import LocalStore
from open_wam.models.visual_tower.vae_encoding import encode_clip, load_vae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--episodes-out", required=True, type=Path)
    parser.add_argument(
        "--source-fps",
        required=True,
        type=float,
        help="FPS from native metadata, never guessed",
    )
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    paths = sorted(set(args.dataset.rglob("*.hdf5")) | set(args.dataset.rglob("*.h5")))
    if not paths or args.source_fps <= 0:
        raise ValueError("Expected HDF5 input and explicit positive native FPS")
    if args.plan:
        print(
            json.dumps(
                dict(episodes=len(paths), policy="robomind_failure_standard_jpeg_v1")
            )
        )
        return
    if args.episodes_out.exists():
        raise FileExistsError("Use a new episode index output path")
    vae = None
    rows = []
    for path in paths:
        relative = path.relative_to(args.dataset).as_posix()
        repo = "failure__" + relative.replace("/", "__").rsplit(".", 1)[0]
        with h5py.File(path, "r") as handle:
            text = read_native_text(handle)
            group = handle["observations/rgb_images"]
            cameras = sorted(c for c in group if len(group[c]))
            lengths = {c: len(group[c]) for c in cameras}
            for camera in cameras:
                frames = []
                for frame in group[camera]:
                    raw = bytes(frame)
                    if not raw.startswith(b"\xff\xd8"):
                        raise ValueError(
                            "Failure adapter admits only standard JPEG bytes"
                        )
                    frames.append(np.array(Image.open(io.BytesIO(raw)).convert("RGB")))
                rgb = np.stack(frames)
                indices = resample_indices(len(rgb), args.source_fps, 15.0)
                indices = indices[: truncate_to_temporal_stride(len(indices), 4)]
                if len(indices) < 5:
                    raise ValueError("Failure video is too short for future prediction")
                bin_config = select_mixed_video_resize_bin(
                    DEFAULT_BINS, source_height=rgb.shape[1], source_width=rgb.shape[2]
                )
                target = (
                    args.out_root
                    / repo
                    / "latents/chunk-000"
                    / camera
                    / f"episode_000000_0_{len(rgb)}.pth"
                )
                if target.exists():
                    raise FileExistsError(
                        "Use an empty output root or verify previous outputs before resuming"
                    )
                if vae is None:
                    vae = load_vae(
                        args.vae,
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        torch.bfloat16,
                    )
                video = fit_frames(
                    rgb[indices],
                    bin_config.target_height,
                    bin_config.target_width,
                    "letterbox_pad",
                )
                latent = encode_clip(vae, video, normalize=True)
                payload = build_payload(
                    latent=latent.cpu(),
                    camera=camera,
                    episode_index=0,
                    indices=indices,
                    source_frames=len(rgb),
                    source_fps=args.source_fps,
                    bin_config=bin_config,
                    fps=15.0,
                    store_dtype="fp16",
                    fit_mode="letterbox_pad",
                    normalize_latents=True,
                    vae_path=args.vae,
                )
                payload.update(
                    color_policy="robomind_failure_standard_jpeg_v1",
                    task=text["task"],
                    source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                )
                memory = io.BytesIO()
                torch.save(payload, memory)
                LocalStore().write_bytes(str(target), memory.getvalue())
        row = make_row(
            "VPT-06",
            repo,
            0,
            cameras,
            lengths,
            dict(
                type="robomind_failure_hdf5",
                uri=str(path.resolve()),
                fps=args.source_fps,
                color_policy="robomind_failure_standard_jpeg_v1",
            ),
            text["task"],
            str(path.resolve()) + "#" + ",".join(text["text_fields"]),
        )
        rows.append(row)
    args.episodes_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.episodes_out.with_suffix(".pending.jsonl")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    temporary.replace(args.episodes_out)
    print(
        json.dumps(dict(episodes=len(rows), policy="robomind_failure_standard_jpeg_v1"))
    )


if __name__ == "__main__":
    main()
