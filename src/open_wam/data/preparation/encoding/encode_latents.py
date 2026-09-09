"""Encode latents."""

from __future__ import annotations

import argparse
import binascii
import collections
import io
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import torch

from open_wam.configs.data_mixed_video import MixedVideoResizeBinConfig
from open_wam.data.mixed_video_decode_frames import select_mixed_video_resize_bin
from open_wam.data.preparation.encoding.payload import DTYPES, build_payload
from open_wam.data.preparation.frames import (
    DEFAULT_BINS,
    FIT_MODES,
    decode_video,
    fit_frames,
    resample_indices,
    truncate_to_temporal_stride,
)
from open_wam.data.preparation.storage import join, make_store
from open_wam.data.preparation.video_sources import (
    Clip,
    Repo,
    discover_repos,
    discover_video_tree_repos,
)
from open_wam.models.visual_tower.vae_encoding import encode_batch, load_vae

@dataclass(frozen=True)
class Job:
    repo: Repo
    episode: int
    camera: str
    out_key: str  # relative to <out-root>/<repo>/<latent-subdir>
    clip: Clip


class VideoCache:
    """Hold the few most recent video files in memory.

    Under v3.0 one file serves many episodes, so fetching per episode would move
    the same hundreds of megabytes over and over. Under v2.1 every path is
    distinct and the cache simply never hits, costing one slot.
    """

    def __init__(self, store, capacity: int = 4) -> None:
        self.store = store
        self.capacity = max(1, capacity)
        self._entries: "collections.OrderedDict[str, bytes]" = collections.OrderedDict()
        self._lock = threading.Lock()
        self._loading: dict[str, threading.Event] = {}

    def get(self, path: str) -> bytes:
        while True:
            with self._lock:
                if path in self._entries:
                    self._entries.move_to_end(path)
                    return self._entries[path]
                waiter = self._loading.get(path)
                if waiter is None:
                    self._loading[path] = threading.Event()
                    break
            # Another thread is already fetching this file; wait rather than
            # duplicate a large download.
            waiter.wait()

        try:
            data = self.store.read_bytes(path)
        finally:
            with self._lock:
                event = self._loading.pop(path, None)
            if event is not None:
                event.set()

        with self._lock:
            self._entries[path] = data
            self._entries.move_to_end(path)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return data


def prefetch(
    store,
    jobs: list[Job],
    depth: int,
    workers: int,
    *,
    target_fps: float | None = None,
) -> queue.Queue:
    """Fetch and decode ahead of the GPU. Yields (job, frames_or_None, error_or_None)."""

    pending: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue(maxsize=max(1, depth))
    cache = VideoCache(store, capacity=max(2, workers // 2))

    # Consecutive jobs then share a file, so the cache hits and the seek is short.
    for job in sorted(jobs, key=lambda j: (j.camera, j.clip.path, j.episode)):
        pending.put(job)

    def worker() -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                results.put(
                    (
                        job,
                        decode_video(
                            cache.get(job.clip.path),
                            job.clip,
                            target_fps=target_fps,
                            declared_fps=job.repo.fps,
                        ),
                        None,
                    )
                )
            except Exception as error:
                results.put((job, None, error))

    threads = [
        threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))
    ]
    for thread in threads:
        thread.start()

    def sentinel() -> None:
        for thread in threads:
            thread.join()
        results.put(None)

    threading.Thread(target=sentinel, daemon=True).start()
    return results


def _completed_latent(
    existing: set[str], stem: str, expected_frames: int | None = None
) -> bool:
    """Check filename inventory against the native length when available.

    This avoids reopening finished videos; it is not tensor certification.
    Manifest admission still validates decoded coverage, shape, and metadata.
    """

    matches = [k for k in existing if k.startswith(stem)]
    if not matches:
        return False
    if not expected_frames:
        return True
    for key in matches:
        tail = key[len(stem) :].rsplit(".", 1)[0]
        parts = tail.split("_")
        if len(parts) < 2:
            continue
        try:
            end = int(parts[1])
        except ValueError:
            continue
        if end >= expected_frames:
            return True
    return False


def drop_completed(
    repos,
    *,
    out_store,
    out_root,
    latent_subdir,
    cameras_filter,
    start_episode,
    episodes,
    overwrite,
):
    """Filter completed repos before clip-level stable-hash assignment."""

    if overwrite:
        return repos
    remaining = []
    for repo in repos:
        cams = [
            c for c in repo.cameras if cameras_filter is None or c in cameras_filter
        ]
        if not cams:
            continue
        first = max(0, start_episode)
        last = (
            repo.total_episodes
            if episodes is None
            else min(first + episodes, repo.total_episodes)
        )
        existing = out_store.list_keys(join(out_root, repo.name, latent_subdir))
        for episode in range(first, last):
            stop = False
            for camera in cams:
                stem = f"chunk-{repo.chunk_of(episode):03d}/{camera}/episode_{episode:06d}_"
                if not _completed_latent(existing, stem, repo.length_of(episode)):
                    remaining.append(repo)
                    stop = True
                    break
            if stop:
                break
    return remaining


def print_plan(repos: list[Repo]) -> int:
    """Report the work a run would do, so a launcher can size its shards.

    The last line is machine-readable on purpose: sharding splits each repo's
    episode range, so a launcher needs the largest range, not the total.
    """

    print(f"{'repo':<44} {'episodes':>9} {'cameras':>8} {'fps':>6} {'clips':>8}")
    print("-" * 80)
    total_clips = 0
    largest = 0
    for repo in repos:
        clips = repo.total_episodes * len(repo.cameras)
        total_clips += clips
        largest = max(largest, repo.total_episodes)
        print(
            f"{repo.name[:44]:<44} {repo.total_episodes:>9} "
            f"{len(repo.cameras):>8} {repo.fps or 0:>6.0f} {clips:>8}"
        )
    print("-" * 80)
    print(f"{len(repos)} repos, {total_clips} clips")
    print(f"MAX_EPISODES_PER_REPO={largest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="LeRobot repo root or parent; local path or s3:// URI",
    )
    parser.add_argument("--vae", help="Wan 2.2 VAE directory (config.json + weights)")
    parser.add_argument(
        "--out-root", help="Where latent trees go; local path or s3:// URI"
    )
    parser.add_argument(
        "--latent-subdir",
        default="latents",
        help="Subdirectory under each repo, matching where the loader looks (default: latents)",
    )
    parser.add_argument(
        "--fps", type=float, default=15.0, help="Target frame rate (default: 15)"
    )
    parser.add_argument(
        "--size-mode",
        choices=("aspect_bins", "fixed"),
        default="aspect_bins",
        help="Route each source into an aspect-ratio bin, or force one square size",
    )
    parser.add_argument("--fit-mode", choices=FIT_MODES, default="letterbox_pad")
    parser.add_argument(
        "--layout",
        choices=("lerobot", "video_tree"),
        default="lerobot",
        help="How to find episodes: LeRobot metadata, or folders of synchronised videos",
    )
    parser.add_argument(
        "--video-glob",
        default="*/*.mp4",
        help="For --layout video_tree: glob relative to --dataset selecting the videos",
    )
    parser.add_argument(
        "--bins",
        default=None,
        help="JSON file overriding the aspect-ratio bins (same fields as ResizeBin)",
    )
    parser.add_argument(
        "--resolution", type=int, default=256, help="Square size for --size-mode fixed"
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Subset of camera keys (default: all)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="Take only shard i of N clips, as i/N. Ownership is a stable hash of "
        "(repo, episode, camera), so shards partition the work no matter what "
        "each one happened to discover -- which is what lets them run against a "
        "corpus that is still growing",
    )
    parser.add_argument(
        "--repos-file",
        default=None,
        help="File of repo names, one per line; discovery is restricted to these. "
        "Lets a caller decide which repos are ready without the encoder listing "
        "every video tree once per shard",
    )
    parser.add_argument(
        "--length-slack",
        type=int,
        default=1,
        help="Frames a decode may differ from the repo's declared episode length "
        "before it is refused (default: 1)",
    )
    parser.add_argument(
        "--start-episode", type=int, default=0, help="First episode index, for sharding"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="How many episodes from --start-episode",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", choices=tuple(DTYPES), default="bf16", help="VAE runtime dtype"
    )
    parser.add_argument(
        "--store-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="On-disk latent dtype",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Store raw posterior means"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=8, help="Video download/decode fetchers"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Clips encoded per VAE pass; 1 disables batching (default: 8)",
    )
    parser.add_argument(
        "--s3-endpoint",
        default=os.environ.get("AWS_ENDPOINT_URL"),
        help="S3 endpoint; defaults to $AWS_ENDPOINT_URL, else the botocore profile",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Report what would be encoded and exit, without loading the VAE",
    )
    args = parser.parse_args()

    in_store = make_store(args.dataset, args.s3_endpoint)

    wanted = None
    if args.repos_file:
        with open(args.repos_file) as handle:
            wanted = {line.strip() for line in handle if line.strip()}
        if not wanted:
            raise SystemExit(f"--repos-file {args.repos_file} is empty")

    def apply_filter(found):
        return found if wanted is None else [r for r in found if r.name in wanted]

    shard_index = shard_total = None
    if args.shard:
        shard_index, shard_total = (int(v) for v in args.shard.split("/"))
        if not 0 <= shard_index < shard_total:
            raise SystemExit(f"--shard {args.shard} is out of range")

    def owned(repo_name: str, episode: int, camera: str) -> bool:
        """Whose clip is this?

        Sharding by position in the discovered list only partitions the work if
        every shard discovers the same list. Against a prefix that is still
        being uploaded they do not: a repo that completes between two shards'
        startup scans shifts every later index, and the same clip gets encoded
        twice while another is missed. A hash of the clip's own identity does
        not move, so ownership holds however much the corpus has grown.
        """

        if shard_total is None:
            return True
        key = f"{repo_name}\x00{episode}\x00{camera}".encode()
        return binascii.crc32(key) % shard_total == shard_index

    if args.plan:
        found = (
            discover_video_tree_repos(in_store, args.dataset, pattern=args.video_glob)
            if args.layout == "video_tree"
            else discover_repos(in_store, args.dataset, keep=wanted)
        )
        return print_plan(apply_filter(found))

    for required in ("vae", "out_root"):
        if getattr(args, required) is None:
            parser.error(
                f"--{required.replace('_', '-')} is required unless --plan is given"
            )

    out_store = make_store(args.out_root, args.s3_endpoint)

    bins = DEFAULT_BINS
    if args.bins:
        with open(args.bins) as handle:
            bins = tuple(
                MixedVideoResizeBinConfig(**entry) for entry in json.load(handle)
            )

    if args.layout == "video_tree":
        repos = discover_video_tree_repos(
            in_store, args.dataset, pattern=args.video_glob
        )
    else:
        repos = discover_repos(in_store, args.dataset, keep=wanted)
    repos = apply_filter(repos)
    before_filter = len(repos)
    repos = drop_completed(
        repos,
        out_store=out_store,
        out_root=args.out_root,
        latent_subdir=args.latent_subdir,
        cameras_filter=args.cameras,
        start_episode=args.start_episode,
        episodes=args.episodes,
        overwrite=args.overwrite,
    )
    print(
        f"{before_filter} repos discovered, {len(repos)} with work left",
        file=sys.stderr,
    )
    device = torch.device(args.device)
    vae = load_vae(args.vae, device, DTYPES[args.dtype])
    vae.disable_slicing()  # slicing would undo batching by splitting it back apart
    temporal_stride = int(vae.config.scale_factor_temporal)

    print(f"vae      : {args.vae}")
    print(
        f"           z_dim={vae.config.z_dim} temporal={temporal_stride}x spatial={vae.config.scale_factor_spatial}x"
    )
    if args.size_mode == "aspect_bins":
        sizes = ", ".join(f"{b.name} {b.target_height}x{b.target_width}" for b in bins)
    else:
        sizes = f"fixed {args.resolution}x{args.resolution}"
    print(
        f"target   : {args.fps} fps, {args.fit_mode}, per-camera, {args.store_dtype} on disk"
    )
    print(f"bins     : {sizes}")
    print(f"in       : {args.dataset}  ({len(repos)} repos)")
    print(f"out      : {args.out_root}")

    written = skipped = failed = 0
    started = time.monotonic()

    for repo in repos:
        cameras = [c for c in repo.cameras if args.cameras is None or c in args.cameras]
        if not cameras:
            print(
                f"  {repo.name}: no matching cameras in {repo.cameras}", file=sys.stderr
            )
            continue

        first = max(0, args.start_episode)
        last = (
            repo.total_episodes
            if args.episodes is None
            else min(first + args.episodes, repo.total_episodes)
        )
        out_prefix = join(args.out_root, repo.name, args.latent_subdir)
        existing = set() if args.overwrite else out_store.list_keys(out_prefix)

        jobs = []
        for episode in range(first, last):
            for camera in cameras:
                if not owned(repo.name, episode, camera):
                    continue
                stem = f"chunk-{repo.chunk_of(episode):03d}/{camera}/episode_{episode:06d}_"
                if _completed_latent(existing, stem, repo.length_of(episode)):
                    skipped += 1
                    continue
                try:
                    clip = repo.clip_of(episode, camera)
                except KeyError as missing:
                    print(f"    ! {missing}", file=sys.stderr)
                    failed += 1
                    continue
                jobs.append(
                    Job(
                        repo=repo,
                        episode=episode,
                        camera=camera,
                        out_key=stem,
                        clip=clip,
                    )
                )

        print(
            f"\n  {repo.name}: episodes [{first}, {last}) x {len(cameras)} cameras "
            f"@ {repo.fps} fps -> {len(jobs)} to encode, {skipped} already present"
        )
        if not jobs:
            continue

        results = prefetch(
            in_store,
            jobs,
            depth=2 * args.workers,
            workers=args.workers,
            target_fps=args.fps,
        )

        # Clips wait here until enough of the same canvas size have arrived to
        # fill a batch. Different bins cannot share a pass, so each keeps its own
        # queue rather than forcing a flush whenever the shape changes.
        batches: dict[tuple[int, int], list] = {}

        def flush(key: tuple[int, int]) -> None:
            nonlocal written, failed
            pending = batches.pop(key, None)
            if not pending:
                return
            try:
                latents = encode_batch(
                    vae, [p["video"] for p in pending], normalize=not args.no_normalize
                )
            except Exception as failure:
                print(
                    f"    ! batch of {len(pending)} at {key}: {failure}",
                    file=sys.stderr,
                )
                failed += len(pending)
                return
            for prepared, latent in zip(pending, latents):
                job = prepared["job"]
                try:
                    payload = build_payload(
                        latent=latent,
                        camera=job.camera,
                        episode_index=job.episode,
                        indices=prepared["indices"],
                        source_frames=prepared["source_frames"],
                        source_fps=prepared["source_fps"],
                        bin_config=prepared["bin"],
                        fps=args.fps,
                        store_dtype=args.store_dtype,
                        fit_mode=args.fit_mode,
                        normalize_latents=not args.no_normalize,
                        vae_path=args.vae,
                    )
                    buffer = io.BytesIO()
                    torch.save(payload, buffer)
                    out_store.write_bytes(
                        join(
                            out_prefix,
                            f"{job.out_key}0_{prepared['source_frames']}.pth",
                        ),
                        buffer.getvalue(),
                    )
                except Exception as failure:
                    print(
                        f"    ! ep{job.episode} {job.camera}: {failure}",
                        file=sys.stderr,
                    )
                    failed += 1
                    continue
                written += 1
                if written == 1 or written % 50 == 0:
                    rate = written / max(1e-9, time.monotonic() - started)
                    print(
                        f"    [{written}/{len(jobs)}] ep{job.episode} {job.camera} "
                        f"-> latent {tuple(latent.shape)} ({rate:.1f}/s)"
                    )

        while True:
            item = results.get()
            if item is None:
                break
            job, decoded, error = item
            if error is not None:
                print(f"    ! ep{job.episode} {job.camera}: {error}", file=sys.stderr)
                failed += 1
                continue
            try:
                frames = decoded.frames
                # `decoded.fps` is the rate `frames` is ALREADY at. The repo's
                # declared rate is the rate of the source, and resampling
                # against it here would decimate the array a second time.
                source_fps = decoded.fps
                declared = repo.length_of(job.episode)
                if (
                    declared is not None
                    and abs(decoded.source_frames - declared) > args.length_slack
                ):
                    raise ValueError(
                        f"decoded {decoded.source_frames} source frames but the repo declares "
                        f"{declared}; refusing to write a latent that cannot be aligned"
                    )
                indices = resample_indices(len(frames), source_fps, args.fps)
                keep = truncate_to_temporal_stride(len(indices), temporal_stride)
                if keep < 1:
                    raise ValueError(
                        f"{len(frames)} frames @ {source_fps} fps is too short for {args.fps} fps"
                    )
                indices = indices[:keep]
                # Two index spaces, and they must not be confused: `indices`
                # select rows of the decoded array, while `frame_ids` name
                # positions on the source timeline the action rows share.
                frame_ids = [decoded.source_ids[i] for i in indices]

                if args.size_mode == "aspect_bins":
                    bin_config = select_mixed_video_resize_bin(
                        bins, source_height=frames.shape[1], source_width=frames.shape[2]
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
            except Exception as failure:  # one bad video should not stop a corpus
                print(f"    ! ep{job.episode} {job.camera}: {failure}", file=sys.stderr)
                failed += 1
                continue

            key = (bin_config.target_height, bin_config.target_width)
            batches.setdefault(key, []).append(
                {
                    "job": job,
                    "video": video,
                    "indices": frame_ids,
                    "source_frames": decoded.source_frames,
                    # The payload records the footage's own rate, not the rate
                    # the frames were decimated to: `end_frame` is a count of
                    # source frames, and the two have to share a unit.
                    "source_fps": decoded.source_fps,
                    "bin": bin_config,
                }
            )
            if len(batches[key]) >= args.batch_size:
                flush(key)

        for key in list(batches):
            flush(key)

    elapsed = time.monotonic() - started
    print(
        f"\nwrote {written}, skipped {skipped}, failed {failed} in {elapsed / 60:.1f} min"
    )
    return 1 if failed and not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
