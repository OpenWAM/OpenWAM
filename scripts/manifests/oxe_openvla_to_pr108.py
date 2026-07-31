#!/usr/bin/env python3
"""Generate a PR#108-compatible manifest CSV for OXE-OpenVLA.

OXE-OpenVLA is a LeRobot v3.0 packed-bundle dataset: multiple episodes share
a single MP4 file (file-NNN.mp4), and each episode's frames are identified by
from_timestamp / to_timestamp boundaries in the parquet metadata.

Source: https://huggingface.co/datasets/lerobot/oxe_openvla
Download: huggingface-cli download lerobot/oxe_openvla --repo-type dataset --local-dir <dir>
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
                        help="Root of the downloaded lerobot/oxe_openvla dataset.")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output manifest CSV path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max episodes to include (for smoke testing).")
    parser.add_argument("--source-id", type=str, default="oxe_openvla")
    parser.add_argument("--camera-key", type=str, default="observation.images.image",
                        help="Video stream key to use.")
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

    tasks_map = _load_tasks(root)
    episodes = _load_episode_boundaries(root, fps, tasks_map, limit=args.limit)

    # WHY probe first video for resolution: avoids hardcoding pixel dimensions
    # that differ across OXE sub-datasets (128x128, 256x256, etc.).
    sample_width, sample_height = 128, 128
    if episodes:
        first_ep = episodes[0]
        chunk_idx = first_ep["episode_index"] // chunks_size
        file_idx = first_ep["episode_index"] % chunks_size
        sample_video = root / video_path_template.format(
            video_key=args.camera_key, chunk_index=chunk_idx, file_index=file_idx,
        )
        w, h = _probe_resolution(sample_video)
        if w > 0:
            sample_width, sample_height = w, h

    rows = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // chunks_size
        file_idx = ep_idx // chunks_size  # WHY same as chunk: packed-bundle packs by chunk

        # WHY resolve to the shared mp4: in packed-bundle, many episodes share one file.
        # The file_index in video_path_template maps to the chunk's shared container.
        video_rel = video_path_template.format(
            video_key=args.camera_key, chunk_index=chunk_idx, file_index=0,
        )
        video_path = root / video_rel

        # WHY from_timestamp/to_timestamp: packed-bundle slicing — the mixed_video reader
        # uses these to seek into the shared MP4 and decode only this episode's frames.
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
            "width": sample_width,
            "height": sample_height,
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


def _load_episode_boundaries(
    root: Path, fps: float, tasks_map: dict[int, str], *, limit: int | None = None,
) -> list[dict]:
    """Read parquet to compute per-episode frame count and timestamp boundaries.

    WHY read parquet: packed-bundle datasets don't have per-episode MP4 files,
    so frame counts and timestamp ranges must be extracted from the data parquet
    which stores (episode_index, frame_index, timestamp, task_index) per frame.
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

            # WHY min/max timestamp: these become from_timestamp/to_timestamp in the manifest,
            # telling the mixed_video reader where this episode starts/ends in the shared MP4.
            # WHY +1/fps for to_timestamp: the last frame's timestamp marks its START, but
            # the reader uses `timestamp >= to_timestamp` as the stop condition, so we need
            # to extend past the last frame by one frame duration.
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
