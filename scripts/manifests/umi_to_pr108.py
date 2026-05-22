#!/usr/bin/env python3
"""Generate a PR#108-compatible manifest CSV for UMI.

UMI is a LeRobot v3.0 packed-bundle dataset: multiple episodes are packed into
shared MP4 files (file-NNN.mp4). Each episode's time range is determined by
reading the parquet data which stores per-frame timestamps.

Source: https://huggingface.co/datasets/lerobot/umi_cup_in_the_wild
Download: huggingface-cli download lerobot/umi_cup_in_the_wild --repo-type dataset --local-dir <dir>
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
                        help="Root of the downloaded UMI dataset.")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output manifest CSV path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max episodes to include (for smoke testing).")
    parser.add_argument("--source-id", type=str, default="umi")
    parser.add_argument("--camera-key", type=str, default="observation.images.camera0_rgb",
                        help="Video stream key. UMI default: camera0_rgb.")
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

    # WHY probe resolution from first video: UMI is 224x224 but we don't hardcode it.
    sample_video = root / video_path_template.format(
        video_key=args.camera_key, chunk_index=0, file_index=0,
    )
    width, height = _probe_resolution(sample_video) if sample_video.exists() else (224, 224)
    if width <= 0:
        width, height = 224, 224

    tasks_map = _load_tasks(root)
    episodes = _load_packed_bundle_episodes(root, fps, tasks_map, limit=args.limit)

    rows = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // chunks_size

        # WHY shared video path: packed-bundle means many episodes are concatenated
        # into one file-NNN.mp4. The mixed_video reader uses from/to_timestamp to
        # seek to the correct segment.
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

    WHY timestamps from parquet: packed-bundle MP4 files contain multiple episodes
    sequentially. The parquet stores per-frame (episode_index, timestamp) which we
    aggregate to find each episode's time range in the shared video file.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    episodes: dict[int, dict] = {}
    data_dir = root / "data"
    for parquet_path in sorted(data_dir.rglob("*.parquet")):
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

            # WHY +1/fps for to_timestamp: the last frame's timestamp is its start time;
            # we need to extend past it by one frame so the reader's `>= to_timestamp`
            # condition correctly includes the last frame.
            episodes[ep_idx] = {
                "episode_index": ep_idx,
                "length_frames": max(frames) - min(frames) + 1,
                "from_timestamp": min(timestamps),
                "to_timestamp": max(timestamps) + 1.0 / fps,
                "tasks": task_text,
            }
        if limit is not None and len(episodes) >= limit:
            break

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
