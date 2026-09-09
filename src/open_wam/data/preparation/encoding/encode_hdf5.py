"""Encode RoboMind-style HDF5 episodes to latents, reusing the video encoder's parts.

RoboMind stores each episode as one trajectory.hdf5 holding per-frame JPEG bytes in
variable-length datasets:

    observations/rgb_images/camera_front     (N,) object -> JPEG bytes per frame
    observations/rgb_images/camera_left
    observations/rgb_images/camera_right
    observations/rgb_images/camera_wrist
    observations/depth_images/...            depth, not wanted here
    language_instruction                     bytes

There is no video container anywhere, so the existing encoder -- which discovers
repos by meta/info.json and decodes videos through a path template -- cannot see
this corpus at all.

WHY THIS IS A SEPARATE FILE RATHER THAN A BRANCH INSIDE encode_latents.py:
two encode jobs are reading that file right now, and editing a module under a
running interpreter is how a 31-second race once wrote 5,007 latents with the wrong
source rate. Everything below the frame-reading step is IMPORTED from it, so the
resampling, the bin selection, the letterbox fit, the VAE call, the payload fields
and the atomic write are literally the same code the mp4 path uses. Only the step
that produces RGB frames differs.

WHAT IS NOT REUSED, AND WHY: decode_video. It takes container bytes and returns a
Decoded with the source frame count and rate. Here the frames arrive already
decoded, one JPEG at a time, so this module builds the equivalent itself and keeps
the same contract -- `fps` is the rate the returned frames are AT, `source_fps` is
the rate of the recording, and both are needed because end_frame counts source
frames. Confusing those two is what once wrote a corpus at half the requested rate.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

from open_wam.configs.data_mixed_video import MixedVideoResizeBinConfig
from open_wam.data.preparation.encoding.payload import build_payload
from open_wam.data.preparation.encoding.robomind_metadata import RGB_GROUP
from open_wam.data.preparation.encoding.robomind_rgb import raw_shape
from open_wam.data.mixed_video_decode_frames import select_mixed_video_resize_bin
from open_wam.data.preparation.frames import (
    fit_frames,
    resample_indices,
    truncate_to_temporal_stride,
)
from open_wam.data.preparation.storage import LocalStore, join
from open_wam.models.visual_tower.vae_encoding import encode_clip, load_vae

DEFAULT_CAMERAS = ("camera_front", "camera_left", "camera_right", "camera_wrist")


@dataclass
class H5Episode:
    path: str
    repo: str
    index: int
    cameras: tuple[str, ...]
    frames: int


def find_episodes(root: str, limit: int | None = None) -> list[H5Episode]:
    """Walk a RoboMind tree for trajectory.hdf5 files.

    The repo name carries the path relative to the scan root, separators flattened.
    Taking only the last segment would be wrong for the same reason it was wrong for
    the nested LeRobot corpus: 'data' is the parent directory of every single
    trajectory.hdf5 here, so every episode would collapse onto one name.
    """
    import h5py

    out: list[H5Episode] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn != "trajectory.hdf5":
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(os.path.dirname(path), root)
            # .../<task>/<variant>/<timestamp>/data/trajectory.hdf5 -> drop the "data"
            parts = [p for p in rel.split(os.sep) if p not in (".", "data")]
            repo = "__".join(parts).replace(" ", "_")
            try:
                with h5py.File(path, "r") as fh:
                    if RGB_GROUP not in fh:
                        continue
                    # A CAMERA KEY IS NOT A CAMERA. RoboMind's franka_1rgb episodes
                    # carry camera_left and camera_right as placeholders: the datasets
                    # exist, have the right length, and every frame is zero bytes. The
                    # decoder raised on the first of them and took the whole episode
                    # down with it -- nine archives failed with "frame 0 is empty" and
                    # wrote nothing, although camera_top was fine in all of them.
                    # Probing one frame per camera here costs a single h5py read and
                    # keeps the empties out of every count downstream.
                    live = []
                    n = 0
                    for key in fh[RGB_GROUP].keys():
                        ds = fh[f"{RGB_GROUP}/{key}"]
                        if len(ds) == 0:
                            continue
                        try:
                            if len(bytes(ds[0])) == 0:
                                continue
                        except Exception:
                            continue
                        live.append(key)
                        n = max(n, len(ds))
                    cams = tuple(live)
            except Exception as exc:
                print(f"  ! {path}: {exc}", file=sys.stderr)
                continue
            if not cams or n == 0:
                continue
            out.append(H5Episode(path=path, repo=repo, index=0, cameras=cams, frames=n))
            if limit and len(out) >= limit:
                return out
    return out


def read_frames(path: str, camera: str) -> np.ndarray:
    """Decode one camera's frames, however this archive happens to store them.

    Most of RoboMind stores JPEG bytes. Part of it -- the 2024_09_20 Franka
    batch among others -- stores uncompressed RGB instead, in the same object
    dtype, with no header to tell them apart. Those archives failed every
    episode with "cannot identify image file" and produced nothing, which is at
    least loud; the danger is the other direction, so the raw path resolves its
    shape by evidence and refuses when the evidence is thin.
    """
    import h5py
    from PIL import Image

    with h5py.File(path, "r") as fh:
        ds = fh[f"{RGB_GROUP}/{camera}"]
        frames = []
        shape = None
        for i in range(len(ds)):
            raw = bytes(ds[i])
            if not raw:
                raise ValueError(f"{camera} frame {i} is empty")
            try:
                frames.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")))
                continue
            except Exception:
                pass
            # Resolve the geometry once per camera, then hold every later frame
            # to it: a buffer that changes size mid-episode is corruption, not a
            # second resolution, and silently reshaping it would hide that.
            if shape is None:
                shape = raw_shape(raw)
            h, w = shape
            if len(raw) != h * w * 3:
                raise ValueError(
                    f"{camera} frame {i}: {len(raw)} bytes against the "
                    f"{h}x{w}x3 established by frame 0"
                )
            frames.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
    if not frames:
        raise ValueError("no frames")
    return np.stack(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="root of a RoboMind tree")
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dataset-id", default="VPT-06")
    ap.add_argument("--fps", type=float, default=15.0, help="target rate")
    ap.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="rate of the recording; RoboMind hdf5 does not state it",
    )
    # AUTO IS THE DEFAULT BECAUSE A FIXED LIST DOES NOT DESCRIBE THIS CORPUS.
    # A survey of the archives found four different camera sets across platforms --
    # (left,right,top) on franka, (front_external,handeye,left_external,
    # right_external) in simulation, (handeye,left_external,right_external) in
    # another simulation split, and (top,) on tienkung -- so the old default
    # matched nothing at all on several of them and the encoder refused whole
    # archives: 4,225 episodes in five archives, one of which held 1,532.
    # "auto" means each episode encodes the cameras it actually has.
    ap.add_argument(
        "--cameras",
        default="auto",
        help='comma-separated camera names, or "auto" for whatever each episode has',
    )
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="stop after N episodes")
    ap.add_argument("--store-dtype", default="fp16")
    # build_payload reads this; the mp4 path normalises by default and the
    # loader assumes latents are stored as (mu-mean)/std, so the default here
    # must match or the two corpora would be in different spaces.
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--size-mode", default="aspect_bins")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--fit-mode", default="letterbox_pad")
    ap.add_argument("--plan", action="store_true", help="list work and exit")
    args = ap.parse_args()

    auto_cameras = args.cameras.strip().lower() == "auto"
    wanted = (
        ()
        if auto_cameras
        else tuple(c.strip() for c in args.cameras.split(",") if c.strip())
    )

    def episode_cameras(episode):
        """Cameras to encode for one episode: its own under auto, else the intersection."""
        return (
            episode.cameras
            if auto_cameras
            else tuple(c for c in episode.cameras if c in wanted)
        )

    eps = find_episodes(args.dataset, limit=args.limit)
    print(f"  {len(eps)} episodes discovered under {args.dataset}")
    if not eps:
        return 1
    cams_seen = sorted({c for e in eps for c in e.cameras})
    print(f"  cameras present: {cams_seen}")
    print(
        f"  encoding: {cams_seen if auto_cameras else [c for c in wanted if c in cams_seen]}"
        f"{'  (auto)' if auto_cameras else ''}"
    )
    total = sum(len(episode_cameras(e)) for e in eps)
    print(f"  clips to encode: {total}")
    # An episode whose cameras do not intersect --cameras used to vanish between
    # the discovery line and the summary, and a whole batch of them printed the
    # same "wrote 0, skipped 0, failed 0" as a batch with nothing left to do.
    unmatched = [e for e in eps if not episode_cameras(e)]
    if unmatched:
        print(
            f"  !! {len(unmatched)} of {len(eps)} episodes have no usable camera"
            f"{'' if auto_cameras else f' in {sorted(wanted)}'}; "
            f"e.g. {unmatched[0].repo[:60]} has {sorted(unmatched[0].cameras)}"
        )
    if total == 0:
        print(
            f"\nwrote 0, skipped 0, failed 0 -- {len(eps)} episodes discovered but none "
            f"carries a requested camera; refusing to report success"
        )
        return 2
    if args.plan:
        for e in eps[:10]:
            print(f"    {e.repo[:70]:72} {e.frames:5} frames  {len(e.cameras)} cams")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(args.vae, device, torch.bfloat16)
    out_store = LocalStore()
    # the same four bins the mp4 path uses
    bins = (
        MixedVideoResizeBinConfig("square_128", 1, 1, 128, 128),
        MixedVideoResizeBinConfig("square_256", 1, 1, 256, 256),
        MixedVideoResizeBinConfig("four_three_352x256", 4, 3, 256, 352),
        MixedVideoResizeBinConfig("sixteen_nine_352x192", 16, 9, 192, 352),
    )
    temporal_stride = 4

    wrote = failed = skipped = 0
    t0 = time.time()
    for e in eps:
        for cam in e.cameras:
            if not auto_cameras and cam not in wanted:
                continue
            stem = f"episode_{e.index:06d}_0_{e.frames}"
            key = join(
                args.out_root, e.repo, "latents", "chunk-000", cam, stem + ".pth"
            )
            if os.path.exists(key):
                skipped += 1
                continue
            try:
                frames = read_frames(e.path, cam)
                # Same contract as decode_video: `indices` select rows of the array
                # we hold, `frame_ids` name positions on the SOURCE timeline.
                idx = resample_indices(len(frames), args.source_fps, args.fps)
                keep = truncate_to_temporal_stride(len(idx), temporal_stride)
                if keep < 1:
                    raise ValueError(
                        f"{len(frames)} frames @ {args.source_fps} is too short for {args.fps}"
                    )
                idx = idx[:keep]
                if args.size_mode == "aspect_bins":
                    bin_config = select_mixed_video_resize_bin(
                        bins, source_height=frames.shape[1], source_width=frames.shape[2]
                    )
                else:
                    bin_config = MixedVideoResizeBinConfig(
                        "fixed", 1, 1, args.resolution, args.resolution
                    )
                video = fit_frames(
                    frames[idx],
                    bin_config.target_height,
                    bin_config.target_width,
                    args.fit_mode,
                )
                # encode_clip takes [T,3,H,W] and handles batching and device
                # placement itself; see visual_tower.vae_encoding.
                latent = encode_clip(vae, video, normalize=not args.no_normalize)
                payload = build_payload(
                    latent=latent.cpu(),
                    camera=cam,
                    episode_index=e.index,
                    indices=idx,
                    source_frames=len(frames),
                    source_fps=args.source_fps,
                    bin_config=bin_config,
                    fps=args.fps,
                    store_dtype=args.store_dtype,
                    fit_mode=args.fit_mode,
                    normalize_latents=not args.no_normalize,
                    vae_path=args.vae,
                )
                buf = io.BytesIO()
                torch.save(payload, buf)
                out_store.write_bytes(key, buf.getvalue())
                wrote += 1
                if wrote % 25 == 0:
                    rate = wrote / max(time.time() - t0, 1e-9) * 60
                    print(
                        f"    [{wrote}/{total}] {e.repo[:44]} {cam} -> {tuple(latent.shape)} ({rate:.0f}/min)",
                        flush=True,
                    )
            except Exception as exc:
                print(f"  ! {e.repo} {cam}: {exc}", file=sys.stderr)
                failed += 1
    print(
        f"\nwrote {wrote}, skipped {skipped}, failed {failed} in {(time.time() - t0) / 60:.1f} min"
    )
    # "some were already there" is not a licence to call a run in which everything
    # else failed a success: exit non-zero whenever a clip was lost.
    if failed:
        return 1
    return 0 if wrote or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
