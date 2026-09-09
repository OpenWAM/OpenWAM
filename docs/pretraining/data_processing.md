# Pretraining Dataset Processing, Composition, Text, and Publication

This guide builds the video pretraining datasets described in [datasets.md](datasets.md), where each source has official download and format instructions. It covers pretraining inputs and their internal train/validation split, not downstream policy fine-tuning or benchmark evaluation data. Run all commands from the repository root with the environment variables defined in the [README](index.md).

## 1. Pin downloaded content

Do not record only a moving reference such as `main`. First inspect the files selected by the download tool, then download them. Before transferring files, the tool resolves the requested branch to a commit, uses that commit for the entire download, and writes `OPENWAM_SOURCE.json`.

```bash
python scripts/pretraining/download_hf.py \
  --repo lerobot/libero --revision main --include 'meta/**' 'videos/**' 'data/**' \
  --out "$DATA_ROOT/raw/VPT-01/libero" --plan

# After checking the version and file selection, remove --plan to download.
# For subsequent builds, use resolved_commit from OPENWAM_SOURCE.json.
```

`--include` is required so that the selected cameras and data partitions are explicit. For LeRobot inputs, downloading MP4 files alone is insufficient: `meta/info.json`, episode/task metadata, and v3 file time ranges are needed to recover durations, text, and synchronization. Download the corresponding `data/` files too when `task_index` appears only in Parquet rows. For gated datasets, obtain access through the official process before using your configured Hugging Face login. The code does not store tokens.

## 2. Prepare an episode index

`prepare_dataset.py --format lerobot` discovers native LeRobot repositories under either a local directory or an object-storage URI. Its `--raw-root` must match the scan root passed to `encode_latents.py --dataset`, especially when the root contains multiple nested repositories. Nested repository names incorporate their relative paths to prevent same-named tasks in different directories from overwriting one another.

```bash
python scripts/pretraining/prepare_dataset.py \
  --source VPT-07 --format lerobot \
  --raw-root "$DATA_ROOT/raw/VPT-07" \
  --out "$WORK_ROOT/episodes/VPT-07.jsonl"
```

Each JSONL row describes one original physical episode:

```json
{
  "source_id": "VPT-01",
  "repo_id": "libero",
  "episode_index": 0,
  "physical_episode_key": "SHA256 determined by source_id, repo_id, and episode_index",
  "cameras": ["observation.images.image", "observation.images.wrist_image"],
  "native_end_frames": {"observation.images.image": 300, "observation.images.wrist_image": 300},
  "raw_source": {"type": "lerobot", "root": "/path/to/raw/libero", "fps": 20},
  "task": "Native task text",
  "text_status": "native",
  "text_provenance": "Native metadata location"
}
```

The numbers illustrate the fields; they do not prescribe LIBERO's source FPS. Read actual FPS and episode lengths from the downloaded revision. The generator validates episode/task metadata and rejects conflicting v3 time ranges. When metadata needs correction, write a new derived directory and document the evidence for the change instead of overwriting the downloaded originals.

`raw_source.type` is supplied by the selected preparation adapter, not inferred
from a dataset name. The supported declarations are `lerobot`, `egoexo_aligned`,
`robomind_official_archive`, and `robomind_failure_hdf5`. Single-view manifest
admission rejects missing or unknown declarations and applies the declared
format's RGB checks regardless of `source_id`. A renamed RoboMind source still
needs its official-archive or standard-JPEG evidence. This field describes raw
preparation inputs; training continues to select its adapter via `data.dataset_type`.

If multiple native tasks cannot be resolved unambiguously, prepare an evidence-based `--text-overrides reviewed.jsonl` first. Each row must contain `repo_id`, `episode_index`, `task`, and `text_provenance`. The script rejects duplicate overrides and overrides without a matching episode. Missing text remains an empty string with an explicit missing status; directory names or visual models are not used to invent labels. Preparation and encoding tools can retain these records, but `make_config.py` rejects task-prompt pretraining snapshots containing unlabeled samples. Recover their native labels first or create a separate, explicit unconditional training recipe.

`physical_episode_key` excludes the camera, crop, segment range, latent output path, and installation location, so single-view and multiview representations share a physical identity. EgoExo4D RGB and SLAM streams also share the physical identity of their take and must remain on the same side of the pretraining train/validation split. Renaming an original repository changes this identity; preserve stable repository names during a version migration or explicitly rebuild the index.

## 3. Encode single-view VAE latents

```bash
python scripts/pretraining/encoding/encode_latents.py \
  --dataset "$DATA_ROOT/raw/VPT-07" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/latents/single/VPT-07" \
  --fps 15 --size-mode aspect_bins --fit-mode letterbox_pad \
  --dtype bf16 --store-dtype fp16 --workers 2 --batch-size 1
```

Start with a small batch and inspect memory use and the output for a complete episode before increasing the VAE `--batch-size` or CPU `--workers`. This is the **encoding batch size**, a separate parameter from the training setting `data.train_batch_size`. For long source videos, the single-view entry point retains the sampled episode's RGB frames in memory. Object-storage input avoids staging a complete MP4 on disk, but it reads the current MP4 into memory; memory consumption is therefore not constant.

Use `--plan` to inspect discovered repositories and cameras, or `--repos-file names.txt` to restrict the repository set. `--start-episode` and `--episodes` select a known range for a pilot. Output layout:

```text
<out-root>/<repo>/latents/chunk-000/<camera>/episode_000000_0_300.pth
```

The filename's start/end indices refer to the **source-frame timeline**. The payload also records `ori_fps`, target `fps`, `frame_ids`, `video_num_frames`, and `latent_num_frames`. A range of 300 source frames does not mean 300 latent frames.

Processing order: decode to uint8 RGB → sample at 15 FPS using the actual source FPS → retain `1+4k` frames → fit an aspect-ratio geometry bin → convert to float32 `[0,1]` → map to float32 `[-1,1]` → run the bf16 VAE → normalize the posterior mean with the VAE mean/std → save fp16 THWC tensors.

The latent reader converts the payload's declared `latent_layout: THWC` to
the training contract `[C,T,H,W]` before selecting temporal windows. It does
not rescale or renormalize values. Existing CTHW sidecars, including bare
tensors without a declaration, keep their existing interpretation. Unsupported
explicit layouts are rejected rather than inferred from tensor dimensions.

| Default bin | Width×height | Intended input geometry |
| --- | --- | --- |
| square_128 | 128×128 | Small native square inputs |
| square_256 | 256×256 | Standard square inputs |
| four_three_352x256 | 352×256 | Approximately 4:3 |
| sixteen_nine_352x192 | 352×192 | Approximately 16:9 |

Single-view inputs do not all contain exactly 65,536 pixels. The 256×256-scale area constraint with ±20% tolerance primarily applies to additional composed views. Single-view processing retains the specified aspect bins.

RoboMind has distinct input and color rules. Follow the [RoboMind instructions](datasets.md) and use `encode_hdf5_text.py` or the failure-data entry point. Do not rebuild these inputs through the generic legacy HDF5 CLI.

## 4. Validate single-view latents and create pretraining CSVs

```bash
python scripts/pretraining/encoding/verify_latents.py \
  "$WORK_ROOT/latents/single/VPT-07"

python scripts/pretraining/build_manifest.py \
  --source VPT-07 --episodes "$WORK_ROOT/episodes/VPT-07.jsonl" \
  --latent-root "$WORK_ROOT/latents/single/VPT-07" \
  --out "$WORK_ROOT/manifests/single/VPT-07.csv"
```

The manifest builder reads the tensors and checks THWC layout, 48 channels, fp16 storage, finite values, normalization markers, 16× spatial compression, 4× temporal compression, and the complete source-episode range. It also computes each file's SHA256. A plausible filename is not enough to accept a corrupt or truncated file as complete. Older tensors missing required metadata need an audit or migration first; adding a metadata field alone does not prove that their encoding was revalidated.

Both `length_frames` and `latent_length_frames` in the CSV are latent-frame counts. The generated training configuration sets `target_observation_fps` to null so that the latent timeline is not subjected to another RGB FPS conversion. `observation_fps=15` describes the sampled RGB timeline used during encoding.

Every single camera contributes a separate clip. `clip_id` includes the repository, episode, camera, and source-frame range so that different cameras or segments do not occupy the same slot. Their common `physical_episode_key` remains available for pretraining train/validation grouping.

## 5. Compose multiview RGB inputs

```bash
python scripts/pretraining/multiview/prepare.py \
  --episodes "$WORK_ROOT"/episodes/VPT-*.jsonl \
  --out "$WORK_ROOT/multiview/plans-v1"

# Smoke-test at most one plan per source in a separate output directory.
python scripts/pretraining/multiview/worker.py \
  --index "$WORK_ROOT/multiview/plans-v1/plan_index.json" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/multiview/smoke-v1" --smoke

# Encode the full plan set; --max-plans can limit an initial batch.
python scripts/pretraining/multiview/worker.py \
  --index "$WORK_ROOT/multiview/plans-v1/plan_index.json" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/multiview/encoded-v1"

# Process official RoboMind gzip archives here, reading each archive once.
python scripts/pretraining/multiview/archive_worker.py \
  --index "$WORK_ROOT/multiview/plans-v1/plan_index.json" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/multiview/encoded-v1"
```

These commands do not submit scheduler jobs. Run the two workers sequentially in the same GPU environment. For multiple GPUs within one resource allocation, assign separate `CUDA_VISIBLE_DEVICES` values and `--shard 0/4`, `1/4`, `2/4`, and `3/4`, with one process per GPU. Do not run different code or VAE versions concurrently against the same output directory.

Camera layout rules:

- Two landscape views are stacked vertically by default; two portrait or square views are arranged side by side. LIBERO and FastUMI explicitly use side-by-side layouts.
- Three views place the head view above the left/right views. Prefer `camera_top` when it exists in RoboMind and record its 180° rotation. Read the available original views from HDF5; do not assume every episode has six cameras.
- For more than three views, select one representative set with explicit camera roles. Ambiguous names do not trigger an arbitrary three-camera selection. Specify `multiview_cameras` and `camera_rotations_degrees` in the episode index when needed.
- Skip multiview processing for all EgoExo4D RGB and SLAM inputs; retain their separate streams.

Read each camera using its native timestamps, restrict the views to their common time interval, and sample at 15 FPS using nearest neighbors. A timestamp mismatch beyond the allowed tolerance raises an error. Each image uses one scale factor for both axes, preserving its aspect ratio. Canvas width and height are multiples of 32, with an area of approximately 52,429–78,643 pixels and a preference for minimizing padding beyond the layout itself. Some necessary black padding may remain.

The current FastUMI cropping rule applies only to calibrated 1280×720 originals. It combines brightness masks from five timestamps using threshold 32 and more than eight valid pixels per column, retains a 24-pixel guard region, crops at most 192 pixels on each side, and preserves all 720 rows. The crop rectangle stays fixed throughout a video; reopening the input verifies that its source object has not changed. Circular fisheye vignetting remains. This rule does not use an inscribed-rectangle crop that removes large parts of the upper and lower field of view.

**Compose RGB first, then encode the entire canvas once.** Concatenating previously encoded per-camera latents is not an equivalent replacement. This differs from the repository's separate general-purpose per-view latent-combination interface; additional multiview pretraining data must pass through this directory's `multiview/worker.py`.

Each output segment contains at most 257 RGB frames, or 65 latent frames. Its receipt records the actual encoded frame count, source time range, cameras, crop/rotation, layout, object identity, VAE and implementation fingerprints, and file SHA256. Tail frames that do not form a complete temporal group are not fabricated into additional samples.

## 6. Inspect multiview previews and completion manifests

Smoke outputs include `previews/<source>/*-rgb.mp4` and `*-vae.mp4`: the original RGB composition and its VAE reconstruction, respectively. Use them to inspect cameras, colors, orientation, cropping, and the VAE. They are not VPM future predictions.

```bash
python scripts/pretraining/build_manifest.py \
  --source VPT-09 \
  --receipts "$WORK_ROOT/multiview/encoded-v1/receipts" \
  --out "$WORK_ROOT/manifests/multiview/VPT-09.csv"
```

Only receipts with `status=complete` enter the manifest. Official RoboMind compressed archives also require a complete archive unit with the same contract, containing the plan ID, and verified gzip EOF/CRC. A written clip latent alone is insufficient for early admission. `smoke_complete`, failed, in-progress, and too-short samples are excluded. The builder rechecks the stored file size and SHA256 when reading each receipt. If a source has no completed multiview data yet, retain only its single-view CSV instead of creating an empty manifest.

## 7. Merge and progressively add pretraining data

```bash
python scripts/pretraining/publish_snapshot.py \
  --manifests "$WORK_ROOT"/manifests/single/VPT-*.csv \
              "$WORK_ROOT"/manifests/multiview/VPT-*.csv \
  --out "$WORK_ROOT/snapshots/phase-001"
```

When more encoded data becomes available, generate new complete per-source CSVs first, then create the next snapshot. `--previous` requires every sample from the preceding snapshot to remain present and unchanged:

```bash
python scripts/pretraining/publish_snapshot.py \
  --manifests "$WORK_ROOT"/manifests/next/VPT-*.csv \
  --previous "$WORK_ROOT/snapshots/phase-001" \
  --out "$WORK_ROOT/snapshots/phase-002"
```

Each snapshot records CSV SHA256 values, source clip counts, physical episode counts, text coverage, single-view/multiview counts, and encoded video durations. These durations count input camera-view time: two cameras recording the same physical demonstration for 10 seconds each contribute 20 seconds of single-view data. An additional 10-second composed video is another training representation, not 10 more seconds of physical data collection.

An active DataLoader does not read a directory that is still growing. Switch to the next immutable snapshot at a complete-checkpoint boundary. Corrections to existing samples require an explicitly documented replacement-data phase; `--previous` cannot present replacements as purely additive growth.
