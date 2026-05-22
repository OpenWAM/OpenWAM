#!/usr/bin/env python3
"""Generate a PR#108-compatible manifest CSV for LIBERO-10.

LIBERO-10 is distributed as a HuggingFace dataset with one MP4 per episode,
one camera (observation.images.image at 256x256, 10fps). No packed-bundle
slicing needed — each episode is a standalone file.

Source: https://huggingface.co/datasets/lerobot/libero_10
Download: huggingface-cli download lerobot/libero_10 --repo-type dataset --local-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# WHY these columns: must match src/open_wam/data/mixed_video.py _load_source_streams
# which parses manifest CSVs for the mixed-video training reader.
FIELDNAMES = (
    "source_id", "dataset_id", "episode_index", "stream_index",
    "stream_key", "target_slot_key", "video_path", "length_frames",
    "observation_fps", "tasks", "width", "height", "channels",
    "from_timestamp", "to_timestamp",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root of the downloaded lerobot/libero_10 dataset.")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output manifest CSV path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max episodes to include (for smoke testing).")
    parser.add_argument("--source-id", type=str, default="libero10",
                        help="Source identifier in the manifest.")
    parser.add_argument("--video-dir", type=Path, default=None,
                        help="Directory containing per-episode MP4s (for datasets where "
                             "info.json video_path is null). Files should be named "
                             "episode_NNNNNN.mp4 or follow LeRobot chunk layout.")
    args = parser.parse_args()

    root = args.source_root.resolve()
    meta_path = root / "meta" / "info.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Is --source-root a valid LeRobot dataset?", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    fps = info["fps"]

    # WHY read tasks.parquet: LIBERO task descriptions are stored in LeRobot v3
    # tasks.parquet, not in per-episode metadata files.
    tasks_map: dict[int, str] = {}
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            import pyarrow.parquet as pq
            import pyarrow.compute as pc
            tasks_table = pq.read_table(tasks_path)
            for i in range(len(tasks_table)):
                task_idx = tasks_table.column("task_index")[i].as_py()
                task_text = tasks_table.column("task")[i].as_py()
                tasks_map[task_idx] = task_text
        except Exception:
            pass

    # WHY read episode parquet: need per-episode frame count + task mapping.
    # LeRobot v3 stores frame-level data in data/chunk-NNN/file-NNN.parquet.
    episode_info: dict[int, dict] = {}
    data_dir = root / "data"
    if data_dir.exists():
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        for parquet_path in sorted(data_dir.rglob("*.parquet")):
            table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "task_index"])
            ep_col = table.column("episode_index")
            unique_eps = pc.unique(ep_col).to_pylist()
            for ep_idx in unique_eps:
                if ep_idx in episode_info:
                    continue
                mask = pc.equal(ep_col, ep_idx)
                filtered = table.filter(mask)
                frames = filtered.column("frame_index").to_pylist()
                frame_count = max(frames) - min(frames) + 1
                task_indices = pc.unique(filtered.column("task_index")).to_pylist()
                task_text = "; ".join(tasks_map.get(t, "") for t in task_indices if tasks_map.get(t))
                episode_info[ep_idx] = {"length_frames": frame_count, "tasks": task_text}

    # WHY video_path from info.json can be null: LIBERO-10 was originally an
    # image-based dataset. LeRobot v3.0 sets video_path=null when the dataset
    # stores frames as images, not videos. Users must pre-convert to per-episode
    # MP4s and pass --video-dir pointing to that directory.
    video_path_template = info.get("video_path")
    chunks_size = info.get("chunks_size", 1000)

    camera_key = "observation.images.image"

    if video_path_template is None and args.video_dir is None:
        # WHY try standard LeRobot layout: some downloads have videos/ even when
        # info.json says null (e.g. after lerobot convert).
        fallback = root / "videos" / camera_key.replace(".", "/")
        if fallback.exists():
            print(f"info.json video_path is null; found videos at {fallback}")
        else:
            print(
                f"ERROR: info.json video_path is null and --video-dir not provided.\n"
                f"LIBERO-10 stores frames as images, not videos. Convert episodes to\n"
                f"per-episode MP4s first, then pass --video-dir <directory>.\n"
                f"Expected files: episode_000000.mp4, episode_000001.mp4, ...",
                file=sys.stderr,
            )
            sys.exit(1)

    video_dir = args.video_dir.resolve() if args.video_dir else None

    rows = []
    limit = args.limit or total_episodes
    for ep_idx in range(min(total_episodes, limit)):
        chunk_index = ep_idx // chunks_size
        file_index = ep_idx % chunks_size

        if video_dir is not None:
            # WHY episode_NNNNNN.mp4: convention used by our image-to-video
            # conversion scripts for LIBERO and similar image-based datasets.
            video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
        elif video_path_template is not None:
            rel_video = video_path_template.format(
                video_key=camera_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            video_path = root / rel_video
        else:
            # WHY fallback to standard LeRobot chunk layout: when video_path is null
            # but videos/ directory exists (post-conversion).
            video_path = root / "videos" / camera_key.replace(".", "/") / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"

        ep = episode_info.get(ep_idx, {})
        length_frames = ep.get("length_frames", 0)
        if length_frames <= 0 and video_path.exists():
            # WHY fallback probe: if parquet not available, probe MP4 directly
            length_frames = _probe_frame_count(video_path)

        rows.append({
            "source_id": args.source_id,
            "dataset_id": args.source_id,
            "episode_index": ep_idx,
            "stream_index": 0,
            "stream_key": camera_key,
            "target_slot_key": "observation.images.slot0",
            "video_path": str(video_path),
            "length_frames": length_frames,
            "observation_fps": fps,
            "tasks": ep.get("tasks", ""),
            "width": 256,
            "height": 256,
            "channels": 3,
            # WHY empty from/to_timestamp: LIBERO uses per-episode MP4 (not packed-bundle),
            # so the entire file is one episode.
            "from_timestamp": "",
            "to_timestamp": "",
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


def _probe_frame_count(video_path: Path) -> int:
    """Probe MP4 frame count via ffprobe if available."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


if __name__ == "__main__":
    main()
