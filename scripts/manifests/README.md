# Manifest Generation Scripts

These scripts generate PR#108-compatible manifest CSVs for the `mixed_video` training reader. Each script reads a downloaded LeRobot v3.0 HuggingFace dataset and produces a CSV with the columns expected by `src/open_wam/data/mixed_video.py`.

## CSV Columns

| Column | Required | Description |
|--------|----------|-------------|
| `source_id` | yes | Groups episodes by source for federated sampling |
| `dataset_id` | yes | Dataset identifier within source |
| `episode_index` | yes | Episode number within the dataset |
| `stream_index` | yes | Stream ordinal (0 for single-camera) |
| `stream_key` | yes | Original camera key in the dataset |
| `target_slot_key` | yes | Canonical slot the reader maps this stream to |
| `video_path` | yes | Absolute path to the MP4 file |
| `length_frames` | yes | Frame count for sampling window bounds |
| `observation_fps` | yes | Source FPS for temporal alignment |
| `tasks` | yes | Semicolon-separated task descriptions |
| `width` | yes | Pixel width for `decode_size_mode: aspect_ratio_bins` |
| `height` | yes | Pixel height |
| `channels` | yes | Channel count (3 for RGB) |
| `from_timestamp` | packed-bundle only | Start timestamp in shared MP4 (empty = full file) |
| `to_timestamp` | packed-bundle only | End timestamp in shared MP4 (empty = full file) |

## Sources

### LIBERO-10

Per-episode MP4 files (not packed-bundle). Single agentview camera at 256x256, 10fps.

**Note:** LIBERO-10's `info.json` has `video_path: null` because the original
dataset stores frames as images, not videos. You must first convert episodes to
per-episode MP4s (e.g. via `ffmpeg`), then point `--video-dir` at that directory.
Files should be named `episode_000000.mp4`, `episode_000001.mp4`, etc.

```bash
# Download
huggingface-cli download lerobot/libero_10 --repo-type dataset --local-dir /data/libero_10

# Generate manifest (--video-dir points to pre-converted MP4 directory)
python scripts/manifests/libero10_to_pr108.py \
    --source-root /data/libero_10 \
    --video-dir /data/libero_10_videos/observation_images_image \
    --output-csv manifests/libero10.csv

# Smoke test (first 10 episodes)
python scripts/manifests/libero10_to_pr108.py \
    --source-root /data/libero_10 \
    --video-dir /data/libero_10_videos/observation_images_image \
    --output-csv /tmp/libero10_smoke.csv \
    --limit 10
```

### OXE-OpenVLA

Packed-bundle format: episodes share MP4 files with `from_timestamp`/`to_timestamp` boundaries. 128x128, 20fps.

```bash
huggingface-cli download lerobot/oxe_openvla --repo-type dataset --local-dir /data/oxe_openvla

python scripts/manifests/oxe_openvla_to_pr108.py \
    --source-root /data/oxe_openvla \
    --output-csv manifests/oxe_openvla.csv
```

### RobotTwin

Per-episode MP4 files organized by task subdirectory. Two variants:
- `aug` (`lerobot_robotwin_eef_aug_500`): 50 tasks × ~500 episodes = ~25K
- `clean` (`lerobot_robotwin_eef_clean_50`): 50 tasks × ~50 episodes = ~2.5K

640x480, 50fps, 3 cameras (only `cam_high` used by default).

```bash
huggingface-cli download OpenRobotLab/robotwin --repo-type dataset --local-dir /data/robotwin

# Augmented variant
python scripts/manifests/robotwin_to_pr108.py \
    --source-root /data/robotwin \
    --output-csv manifests/robotwin_aug.csv \
    --variant aug

# Clean variant
python scripts/manifests/robotwin_to_pr108.py \
    --source-root /data/robotwin \
    --output-csv manifests/robotwin_clean.csv \
    --variant clean
```

### UMI

Packed-bundle format. 224x224, 30fps, single camera (`camera0_rgb`). 305 episodes.

```bash
huggingface-cli download lerobot/umi_cup_in_the_wild --repo-type dataset --local-dir /data/umi

python scripts/manifests/umi_to_pr108.py \
    --source-root /data/umi \
    --output-csv manifests/umi.csv
```

### Droid (Full)

Packed-bundle format. 320x180, 15fps. ~95K episodes. Very large (~2TB video).

```bash
# Download only the camera you need to save space
huggingface-cli download lerobot/droid --repo-type dataset --local-dir /data/droid \
    --include "data/**" "meta/**" "videos/observation.images.exterior_image_1_left/**"

python scripts/manifests/droid_full_to_pr108.py \
    --source-root /data/droid \
    --output-csv manifests/droid_full.csv
```

## Dependencies

- `pyarrow` (for reading LeRobot v3.0 parquet metadata)
- `ffprobe` (optional, for probing video resolution; falls back to defaults)

## Packed-Bundle vs Per-Episode

Some datasets (LIBERO, RobotTwin) store one MP4 per episode. Others (OXE, UMI, Droid) use LeRobot v3.0 "packed-bundle" format where multiple episodes are concatenated into shared MP4 files. For packed-bundle sources, the manifest includes `from_timestamp` and `to_timestamp` so the `mixed_video` reader can seek to the correct segment within the shared file.
