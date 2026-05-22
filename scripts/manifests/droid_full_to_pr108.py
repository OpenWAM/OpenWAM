#!/usr/bin/env python3
"""Generate a PR#108-compatible manifest CSV for Droid (full).

Droid is a LeRobot v3.0 packed-bundle dataset (~95K episodes): multiple episodes
are packed into shared MP4 files. Each episode's boundaries come from the parquet
timestamp data. The default camera is exterior_image_1_left at 320x180, 15fps.

Source: https://huggingface.co/datasets/lerobot/droid
Download: huggingface-cli download lerobot/droid --repo-type dataset --local-dir <dir>

Note: Droid is very large (~2TB video). Consider downloading only the camera
you need: --include "videos/observation.images.exterior_image_1_left/**"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FIELDNAMES = (
    "source_id", "dataset_id", "episode_index", "stream_index",
    "stream_key", "target_slot_key", "video_path", "length_frames",
    "observation_fps", "tasks", "width", "height", "channels",
    "from_timestamp", "to_timestamp",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root of the downloaded Droid dataset.")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output manifest CSV path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max episodes to include (for smoke testing).")
    parser.add_argument("--source-id", type=str, default="droid_full")
    parser.add_argument("--camera-key", type=str,
                        default="observation.images.exterior_image_1_left",
                        help="Video stream key. Droid has exterior_image_1_left (default), "
                             "exterior_image_2_left, wrist_image_left.")
    args = parser.parse_args()

    root = args.source_root.resolve()
    meta_path = root / "meta" / "info.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        info = json.load(f)

    fps = info["fps"]
    chunks_size = info.get("chunks_size", 1000)
    video_path_template = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    # WHY probe resolution: Droid exterior_1_left is 320x180 but may vary by camera.
    sample_video = root / video_path_template.format(
        video_key=args.camera_key, chunk_index=0, file_index=0,
    )
    width, height = _probe_resolution(sample_video) if sample_video.exists() else (320, 180)
    if width <= 0:
        width, height = 320, 180

    tasks_map = _load_tasks(root)

    # WHY stream processing: Droid has ~95K episodes across ~96 parquet files.
    # Loading all into memory at once would use ~10GB. Process chunk by chunk.
    print(f"Scanning parquet files for episode boundaries (fps={fps}, limit={args.limit}) ...")
    episodes = _load_packed_bundle_episodes(root, fps, tasks_map, limit=args.limit)
    print(f"Found {len(episodes)} episodes")

    rows = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // chunks_size

        video_rel = video_path_template.format(
            video_key=args.camera_key, chunk_index=chunk_idx, file_index=0,
        )
        video_path = root / video_rel

        rows.append({
            "source_id": args.source_id,
            "dataset_id": args.source_id,
            "episode_index": ep_idx,
            "stream_index": 0,
            "stream_key": args.camera_key,
            "target_slot_key": "observation.images.slot0",
            "video_path": str(video_path),
            "length_frames": ep["length_frames"],
            "observation_fps": fps,
            "tasks": ep.get("tasks", ""),
            "width": width,
            "height": height,
            "channels": 3,
            "from_timestamp": f"{ep['from_timestamp']:.10f}",
            "to_timestamp": f"{ep['to_timestamp']:.10f}",
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


def _load_tasks(root: Path) -> dict[int, str]:
    tasks_map: dict[int, str] = {}
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(tasks_path)
            for i in range(len(table)):
                tasks_map[table.column("task_index")[i].as_py()] = table.column("task")[i].as_py()
        except Exception:
            pass
    return tasks_map


def _load_packed_bundle_episodes(
    root: Path, fps: float, tasks_map: dict[int, str], *, limit: int | None = None,
) -> list[dict]:
    """Extract per-episode boundaries from packed-bundle parquet.

    WHY process parquet file-by-file: Droid has ~95K episodes across ~96 parquet
    files (1000 episodes each). Reading all at once would OOM on the login node.
    We process one parquet file at a time and aggregate episode boundaries.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    episodes: dict[int, dict] = {}
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    for file_num, parquet_path in enumerate(parquet_files):
        if limit is not None and len(episodes) >= limit:
            break
        if (file_num + 1) % 10 == 0:
            print(f"  ...processed {file_num + 1}/{len(parquet_files)} parquet files ({len(episodes)} episodes)")

        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "timestamp", "task_index"])
        ep_col = table.column("episode_index")
        for ep_idx in pc.unique(ep_col).to_pylist():
            if ep_idx in episodes:
                continue
            if limit is not None and len(episodes) >= limit:
                break
            mask = pc.equal(ep_col, ep_idx)
            filtered = table.filter(mask)
            frames = filtered.column("frame_index").to_pylist()
            timestamps = filtered.column("timestamp").to_pylist()
            task_indices = pc.unique(filtered.column("task_index")).to_pylist()
            task_text = "; ".join(tasks_map.get(t, "") for t in task_indices if tasks_map.get(t))

            episodes[ep_idx] = {
                "episode_index": ep_idx,
                "length_frames": max(frames) - min(frames) + 1,
                "from_timestamp": min(timestamps),
                "to_timestamp": max(timestamps) + 1.0 / fps,
                "tasks": task_text,
            }

    return [episodes[k] for k in sorted(episodes)]


def _probe_resolution(video_path: Path) -> tuple[int, int]:
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        parts = result.stdout.strip().split("x")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


if __name__ == "__main__":
    main()
