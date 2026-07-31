#!/usr/bin/env python3
"""Generate a PR#108-compatible manifest CSV for RobotTwin.

RobotTwin is organized as a collection of per-task sub-datasets under a variant
directory. Each sub-dataset has its own meta/info.json, data parquet, and per-
episode MP4 videos.

Two variants exist:
  - aug  (lerobot_robotwin_eef_aug_500):   50 tasks × ~500 episodes = ~25K
  - clean (lerobot_robotwin_eef_clean_50): 50 tasks × ~50 episodes  = ~2.5K

Cameras: cam_high, cam_left_wrist, cam_right_wrist (default: cam_high).

Source: https://huggingface.co/datasets/OpenRobotLab/robotwin
Download: huggingface-cli download OpenRobotLab/robotwin --repo-type dataset --local-dir <dir>
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

VARIANT_DIRS = {
    "aug": "lerobot_robotwin_eef_aug_500",
    "clean": "lerobot_robotwin_eef_clean_50",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root of the downloaded RobotTwin dataset "
                             "(contains lerobot_robotwin_eef_aug_500/ etc).")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output manifest CSV path.")
    parser.add_argument("--variant", type=str, required=True, choices=["aug", "clean"],
                        help="Which variant to generate: 'aug' or 'clean'.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max episodes to include across all tasks (for smoke testing).")
    parser.add_argument("--camera-key", type=str, default="observation.images.cam_high",
                        help="Camera stream to use. Default: cam_high (top-down view).")
    args = parser.parse_args()

    root = args.source_root.resolve()
    source_id = f"robotwin_{args.variant}"

    variant_dir = root / VARIANT_DIRS[args.variant]
    if not variant_dir.exists():
        print(f"ERROR: {variant_dir} not found.", file=sys.stderr)
        sys.exit(1)

    # WHY iterate task dirs: RobotTwin stores each task as its own LeRobot
    # sub-dataset with independent meta/info.json and episode numbering.
    task_dirs = sorted(
        d for d in variant_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    print(f"Found {len(task_dirs)} task directories in {variant_dir.name}")

    rows = []
    total_count = 0
    for task_dir in task_dirs:
        if args.limit is not None and total_count >= args.limit:
            break

        meta_path = task_dir / "meta" / "info.json"
        if not meta_path.exists():
            print(f"  SKIP {task_dir.name}: no meta/info.json")
            continue

        with open(meta_path) as f:
            info = json.load(f)

        fps = info.get("fps", 50)
        chunks_size = info.get("chunks_size", 1000)
        # WHY video_path uses episode_chunk/episode_index: RobotTwin v2.1 format
        # differs from standard v3.0 packed-bundle. Per-episode MP4 files.
        video_path_template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )

        tasks_map = _load_tasks(task_dir)
        dataset_id = task_dir.name

        remaining = None
        if args.limit is not None:
            remaining = args.limit - total_count
            if remaining <= 0:
                break

        task_episodes = _load_episodes(
            task_dir, fps, tasks_map,
            camera_key=args.camera_key,
            video_path_template=video_path_template,
            chunks_size=chunks_size,
            limit=remaining,
        )

        for ep in task_episodes:
            rows.append({
                "source_id": source_id,
                "dataset_id": dataset_id,
                "episode_index": ep["episode_index"],
                "stream_index": 0,
                "stream_key": args.camera_key,
                "target_slot_key": "observation.images.slot0",
                "video_path": ep["video_path"],
                "length_frames": ep["length_frames"],
                "observation_fps": fps,
                "tasks": ep.get("tasks", ""),
                "width": ep.get("width", 640),
                "height": ep.get("height", 480),
                "channels": 3,
                "from_timestamp": "",
                "to_timestamp": "",
            })
            total_count += 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


def _load_tasks(task_dir: Path) -> dict[int, str]:
    tasks_map: dict[int, str] = {}
    tasks_path = task_dir / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(tasks_path)
            for i in range(len(table)):
                tasks_map[table.column("task_index")[i].as_py()] = table.column("task")[i].as_py()
        except Exception:
            pass
    return tasks_map


def _load_episodes(
    task_dir: Path, fps: float, tasks_map: dict[int, str], *,
    camera_key: str, video_path_template: str, chunks_size: int,
    limit: int | None = None,
) -> list[dict]:
    """Read per-episode parquet files to get frame counts and task text."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    episodes: dict[int, dict] = {}
    data_dir = task_dir / "data"
    if not data_dir.exists():
        return []

    for parquet_path in sorted(data_dir.rglob("*.parquet")):
        cols = ["episode_index", "frame_index", "task_index"]
        table = pq.read_table(parquet_path, columns=cols)
        ep_col = table.column("episode_index")
        for ep_idx in pc.unique(ep_col).to_pylist():
            if ep_idx in episodes:
                continue
            if limit is not None and len(episodes) >= limit:
                break
            mask = pc.equal(ep_col, ep_idx)
            filtered = table.filter(mask)
            frames = filtered.column("frame_index").to_pylist()
            task_indices = pc.unique(filtered.column("task_index")).to_pylist()
            task_text = "; ".join(tasks_map.get(t, "") for t in task_indices if tasks_map.get(t))

            chunk_idx = ep_idx // chunks_size
            # WHY episode_chunk + episode_index format: RobotTwin video_path
            # template uses these keys, not chunk_index/file_index.
            video_rel = video_path_template.format(
                episode_chunk=chunk_idx,
                episode_index=ep_idx,
                video_key=camera_key,
            )
            video_path = task_dir / video_rel

            w, h = 640, 480
            if video_path.exists() and not episodes:
                pw, ph = _probe_resolution(video_path)
                if pw > 0:
                    w, h = pw, ph

            episodes[ep_idx] = {
                "episode_index": ep_idx,
                "length_frames": max(frames) - min(frames) + 1,
                "video_path": str(video_path),
                "tasks": task_text,
                "width": w,
                "height": h,
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
