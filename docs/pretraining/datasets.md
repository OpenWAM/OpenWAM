# Video Pretraining Datasets

This guide describes the nine-source video pretraining recipe: where to obtain inputs, how to encode cameras, and how to validate the resulting manifests. It does not define downstream policy training or evaluation splits. Pin each upstream revision; dataset sizes, camera keys, and episode identities can change between releases.

Published model weights: [OpenWAM-Stanford/OpenWAM-Pretraining](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining). This Hugging Face repository hosts the OpenWAM pretraining checkpoint weights; download the pretraining datasets from the upstream sources documented below.

The machine-readable registry is [datasets.yaml](../assets/pretraining/datasets.yaml). Its paths are templates whose environment variables must be expanded by the caller; it is not a model training configuration. Bind each training run to the CSV manifests, data contracts, text cache, and object catalog of its published snapshot.

## 1. Supported Sources

| ID | Pretraining data source | Raw input format |
| --- | --- | --- |
| VPT-01 | LIBERO | Converted LeRobot with workspace and wrist videos |
| VPT-04 | UMI / MV-UMI | LeRobot v3 videos and metadata |
| VPT-05 | AgiBot World Alpha / Beta | Video tar archives and task_info organized by task / episode |
| VPT-06 | RoboMIND official / failure | Split archives or individual HDF5 files |
| VPT-07 | InternData-A1 | LeRobot organized by robot and task |
| VPT-08 | RoboCOIN | Separate LeRobot repository for each task |
| VPT-09 | FastUMI | LeRobot v2.1 single-arm / dual-arm tasks |
| VPT-10R | Ego-Exo4D color streams | frame_aligned_videos and takes.json |
| VPT-10S | Ego-Exo4D monochrome SLAM streams | Monochrome streams from the same takes |

One CSV row represents a latent clip or camera stream, not an independent physical episode. Track planned, encoded, validated, and admitted clips separately in your own build.

Deduplicate by physical episode across views and source aliases, and keep those representations in the same train/validation split. Repeating an episode under another source changes its sampling weight. AgiBot Alpha / Beta releases may also overlap; release names alone do not establish independent episodes.

The source IDs are recipe identifiers, not a complete dataset catalog. OXE / DROID are not enabled by this template even though an optional RLDS encoder is available.

## 2. Portable Paths, Pinned Downloads, and the Single-View Workflow

Run the following commands from the repository root. Set storage paths appropriate for your available disk capacity:

```bash
export DATA_ROOT=/path/to/openwam-data
export WORK_ROOT=/path/to/openwam-work
export MODEL_ROOT=/path/to/openwam-models
export VAE_ROOT="$MODEL_ROOT/wan22/vae"
export MODEL_ASSETS="$MODEL_ROOT/video-init"
mkdir -p "$DATA_ROOT/raw" "$WORK_ROOT/latents/single" "$WORK_ROOT/episodes" \
  "$WORK_ROOT/manifests/single" "$WORK_ROOT/manifests/multiview"
```

`$VAE_ROOT` must contain Wan VAE configuration and weights matching the training contract. Text encoding uses `$MODEL_ASSETS/text_encoder` and `$MODEL_ASSETS/tokenizer`. See the pretraining README for dependency installation and model asset preparation. Download examples use the Hugging Face Hub CLI. For gated repositories, obtain access on the official dataset page and authenticate with your own account; keep credentials out of documentation.

For each download, first choose **one repository, one revision, and a bounded subset**. This command resolves the revision without downloading video:

```bash
export HF_REPO='IPEC-COMMUNITY/FastUMI_100k_lerobot'
export HF_REV="$(python -c 'import os; from huggingface_hub import HfApi; print(HfApi().dataset_info(os.environ["HF_REPO"]).sha)')"
printf '%s %s\n' "$HF_REPO" "$HF_REV"
```

Save the revision, downloaded file inventory, and checksums in the build record. Each `$HF_REV` below must belong to **that specific repository**. Resolve a new revision whenever `HF_REPO` changes; do not reuse another repository's commit. The examples are explicit commands, and reading this guide does not trigger an automatic full-dataset download.

### 2.1 Required LeRobot Input Files

```text
raw/<source>/<repo>/
  meta/info.json
  meta/episodes.jsonl                  # v2, or v3 meta/episodes/**/*.parquet
  meta/tasks.jsonl                     # or meta/tasks.parquet
  data/.../*.parquet                   # Actions, states, original timestamps
  videos/.../*.mp4                     # Resolve camera paths using info.json templates
```

Keep the metadata alongside MP4 files: it defines episode boundaries, camera names, FPS, task text, and start / end times within videos. A LeRobot v3 MP4 may contain multiple episodes, so filenames do not establish one file per episode. Input preparation must fail on duplicate, conflicting episodes or timestamps. Write reviewed repairs into a new directory and retain both the original version and the evidence supporting the repair.

### 2.2 Complete Common Command Sequence

After downloading or converting a source, set `SOURCE` and `RAW_ROOT`, then run:

```bash
export SOURCE='VPT-09'
export RAW_ROOT="$DATA_ROOT/raw/$SOURCE/dual_arm/Fold_the_Suit"

python scripts/pretraining/prepare_dataset.py \
  --source "$SOURCE" --format lerobot --raw-root "$RAW_ROOT" \
  --out "$WORK_ROOT/episodes/$SOURCE.jsonl"

python scripts/pretraining/encoding/encode_latents.py \
  --dataset "$RAW_ROOT" --vae "$VAE_ROOT" \
  --out-root "$WORK_ROOT/latents/single/$SOURCE" \
  --fps 15 --size-mode aspect_bins --fit-mode letterbox_pad \
  --dtype bf16 --store-dtype fp16 --workers 2 --batch-size 2

python scripts/pretraining/build_manifest.py \
  --source "$SOURCE" --episodes "$WORK_ROOT/episodes/$SOURCE.jsonl" \
  --latent-root "$WORK_ROOT/latents/single/$SOURCE" \
  --out "$WORK_ROOT/manifests/single/$SOURCE.csv"
```

Start with one small repository or task. `--batch-size 2` is an encoding example, not the global training batch size; increase it after checking GPU memory. For larger builds, use the encoder's `--shard i/N` option. All shards in one build must share a fixed input inventory, revision, and encoding parameters. Use new episode JSONL files, manifests, and snapshot directories for additional data, preserving the frozen inputs of active runs.

`prepare_dataset.py` exports episode identity, cameras, native frame counts, source provenance, and task text. The encoder writes `.pth` files; `build_manifest.py` validates the actual tensors and joins them to original episodes. An existing, unverified tensor does not establish completion. A process exit code, one `.pth` in a directory, or a legacy `.tar_complete` marker is insufficient evidence that an entire archive is complete.

### 2.3 Single-View Pixel and Latent Contracts

Decode each camera independently. Ordinary MP4 videos enter processing as RGB24; see Section 6 for RoboMIND. Sample at 15 Hz by selecting the nearest frame on the source timeline without blending frames. Lower-FPS sources may repeat frames. Do not resample an already resampled video using its original FPS. Truncate RGB length to `1 + 4k` and record the actual source frame indices.

The default is `aspect_bins + letterbox_pad`: choose the canvas with the nearest logarithmic aspect ratio, resize with isotropic bilinear interpolation, center the image, and fill unused canvas pixels with black. Center cropping and square stretching are not defaults. All canvas dimensions below are **width × height**:

| Source aspect-ratio class | Output canvas |
| --- | --- |
| Near 1:1, source area at most 128² | 128 × 128 |
| Near 1:1, larger source image | 256 × 256 |
| Near 4:3 | 352 × 256 |
| Near 16:9 | 352 × 192 |

Single-view inputs can therefore contain both padding introduced by encoding and dark corners already present in the original camera image. Distinguish them during visual inspection. Explicitly selecting `center_crop` or `stretch` changes the data contract.

Use the VAE posterior mean. Map input pixels from `[0,1]` to `[-1,1]` in float32, then compute using the model precision. Normalize latents with the VAE configuration's mean / standard deviation. The current model has spatial compression 16, temporal compression 4, and 48 latent channels; output tensors use fp16 `THWC`. A typical path is:

```text
$WORK_ROOT/latents/single/<source>/<repo>/latents/chunk-000/<camera>/episode_000000_0_<native_end>.pth
```

Each payload must retain source FPS, sampling FPS, frame_ids, original dimensions, resize mode, latent layout, and normalization metadata. A complete RGB input with N frames must have latent temporal length `1 + (N-1)/4`. The native frame count in the filename is not the latent frame count.

## 3. VPT-01: LIBERO

**Original source.** The [official LIBERO dataset documentation](https://libero-project.github.io/datasets) describes workspace cameras, wrist cameras, states, and language tasks. Original download and simulation tools are in the [official LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO). The five suites are Spatial, Object, Goal, 90, and 10; LIBERO-100 consists of 90 / 10.

**Public input supported by this encoder.** The [LeRobot LIBERO guide](https://huggingface.co/docs/lerobot/libero) lists preprocessed releases including [lerobot/libero](https://huggingface.co/datasets/lerobot/libero). Select the suites and episodes needed for your build, and verify FPS at the chosen revision.

```bash
export HF_REPO='lerobot/libero'
# Resolve this repository's HF_REV as in Section 2, then download the selected dataset.
hf download "$HF_REPO" --repo-type dataset --revision "$HF_REV" \
  --local-dir "$DATA_ROOT/raw/VPT-01/libero" \
  --include 'meta/*' 'data/*' 'videos/*' 'README.md'
export SOURCE='VPT-01'
export RAW_ROOT="$DATA_ROOT/raw/$SOURCE/libero"
# Run prepare_dataset -> encode_latents -> build_manifest from Section 2.2.
```

To convert official HDF5 files again, use the upstream LIBERO / LeRobot conversion tools corresponding to that release. Preserve the suite, task, demo ID, image-flip convention, and timestamps. This repository does not vendor the entire upstream simulation collection and HDF5-to-LeRobot conversion project. The RoboMIND-specific encoder is not a LIBERO HDF5 adapter.

**Text.** Join the episode's native `tasks` or explicit `task_index` to its task table. Workspace and wrist cameras inherit the same native task instruction.

**Multiview.** Select the agent / workspace camera and wrist camera from the same episode and place them side by side. Confirm camera roles from metadata; if a new release changes camera keys, inspect the planned order first.

## 4. VPT-04: Six UMI / MV-UMI Subsets

The upstream research projects are [UMI](https://umi-gripper.github.io/) and [MV-UMI](https://mv-umi.github.io/); original sources can also be traced through the [UMI data community](https://umi-data.github.io/). This workflow consumes the following public **LeRobot conversions**, rather than reading raw Zarr directly:

| Task | Exact Hugging Face repository ID |
| --- | --- |
| Bottles rack, segmented third-person view | [DaivdYuan/mv-umi-bottles-rack-seg-lerobot](https://huggingface.co/datasets/DaivdYuan/mv-umi-bottles-rack-seg-lerobot) |
| Markers placement, raw third-person view | [DaivdYuan/mv-umi-markers-placement-raw-lerobot](https://huggingface.co/datasets/DaivdYuan/mv-umi-markers-placement-raw-lerobot) |
| Markers placement, segmented third-person view | [DaivdYuan/mv-umi-markers-placement-seg-lerobot](https://huggingface.co/datasets/DaivdYuan/mv-umi-markers-placement-seg-lerobot) |
| Bimanual cloth folding | [DaivdYuan/umi-bimanual-cloth-folding-lerobot](https://huggingface.co/datasets/DaivdYuan/umi-bimanual-cloth-folding-lerobot) |
| Bimanual dish washing | [DaivdYuan/umi-bimanual-dish-washing-lerobot](https://huggingface.co/datasets/DaivdYuan/umi-bimanual-dish-washing-lerobot) |
| Dynamic tossing | [DaivdYuan/umi-dynamic-tossing-lerobot](https://huggingface.co/datasets/DaivdYuan/umi-dynamic-tossing-lerobot) |

Each converted repository's dataset card identifies its original source. For example, cloth folding originates from `https://real.stanford.edu/umi/data/bimanual_cloth_folding/bimanual_cloth_folding.zarr.zip`, and tossing from `https://real.stanford.edu/umi/data/dynamic_tossing/dynamic_tossing.zarr.zip`. Raw Zarr acquisition, camera preprocessing, and LeRobot conversion belong to the upstream tooling. The reproducible entry point here is a public converted artifact at a pinned revision.

Download one subset first:

```bash
export HF_REPO='DaivdYuan/umi-bimanual-cloth-folding-lerobot'
# Set HF_REV for the current HF_REPO.
hf download "$HF_REPO" --repo-type dataset --revision "$HF_REV" \
  --local-dir "$DATA_ROOT/raw/VPT-04/umi-bimanual-cloth-folding-lerobot" \
  --include 'meta/*' 'data/*' 'videos/*' 'README.md'
export SOURCE='VPT-04'
export RAW_ROOT="$DATA_ROOT/raw/$SOURCE"
# Apply Section 2.2 to downloaded subsets; create new manifests when adding subsets.
```

**Text.** Task assignments come from v3 episode metadata and the native task table. A readable repository name does not replace missing task text. Converted action or state fields may use upstream fallbacks; their presence does not establish additional action supervision in this video pretraining recipe.

**Metadata validation.** Check for duplicate episode records and conflicting time intervals before encoding. Resolve conflicts against the actual video duration and canonical episode metadata; do not select the first record arbitrarily. Keep originals unchanged and write any reviewed repair under `$DATA_ROOT/converted/VPT-04/<revision>/`.

**Multiview.** Apply the common rule to two-camera episodes: stack vertically when both images are landscape; otherwise place square, portrait, or mixed-aspect images side by side. Single-camera episodes, including dynamic tossing, remain single-view; duplicating a frame does not create another camera. The raw / segmented marker datasets are separate releases, but independence requires checking their original episode identities.

## 5. VPT-05: AgiBot World Alpha / Beta

Public releases are [agibot-world/AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) and [agibot-world/AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta). Their dataset cards describe raw layouts and official LeRobot conversion entry points. Alpha has undergone release updates and removal of anomalous samples with dropped frames; record the revision instead of relying on an old trajectory total.

**Download.** Select one task, such as task 327 from the official example, and retrieve that task with its required metadata:

```bash
export HF_REPO='agibot-world/AgiBotWorld-Alpha'
# Set HF_REV for this HF_REPO; Beta requires its own revision.
hf download "$HF_REPO" --repo-type dataset --revision "$HF_REV" \
  --local-dir "$DATA_ROOT/raw/VPT-05/alpha" \
  --include 'observations/327/*' 'task_info/task_327.json' 'README.md'
```

The raw layout is `observations/<task_id>/<episode_id>/videos/<camera>.mp4`. Archived releases use `observations/<task_id>/<range>.tar`, retaining original episode IDs inside the archive. Actions / states live separately under `proprio_stats/<task>/<episode>/proprio_stats.h5`, and camera parameters under `parameters/`; download those separately if required by a later action-training workflow.

This video pretraining workflow selects the three RGB streams `head_color`, `hand_left_color`, and `hand_right_color`. Other fisheye streams are unselected; that does not mean they are grayscale. Convert one explicitly selected tar archive at a time:

```bash
export AGIBOT_ARCHIVE="$DATA_ROOT/raw/VPT-05/alpha/observations/327/YOUR_SELECTED_RANGE.tar"
python scripts/pretraining/convert_agibot.py \
  --archive "$AGIBOT_ARCHIVE" \
  --task-info "$DATA_ROOT/raw/VPT-05/alpha/task_info/task_327.json" \
  --out "$DATA_ROOT/converted/VPT-05/alpha_task327_selected"
export SOURCE='VPT-05'
export RAW_ROOT="$DATA_ROOT/converted/$SOURCE/alpha_task327_selected"
# Run the three processing commands from Section 2.2.
```

This converter creates a bounded LeRobot input for video pretraining. The complete official data tools and action conversion remain upstream. Conversion preserves the original episode ID mapping, probes actual FPS / length, checks that all three cameras are complete, and writes `meta/info.json`, episode metadata, and task metadata. Consecutive output indices `0,1,2…` are not the original `episode_id`.

**Text.** The top level of `task_info/task_<id>.json` is a list whose entries contain `episode_id`, `task_id`, and `task_name`. Join the exact original episode ID to `task_name`. The `start_frame/end_frame/action_text` entries under `lable_info.action_config` are separate temporal annotations; they are not automatically included in the current task-level text. Archive ranges, task directory names, and adjacent episodes must not fill missing labels.

**Multiview.** Place the head image above the left / right hand images. Synchronize the streams, compose their RGB pixels, and encode the full multi view video with the VAE.

## 6. VPT-06: RoboMIND Official and Failure Data

Official source: [x-humanoid-robomind/RoboMIND](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND). This dataset includes multiple embodiments, task language, and failure data. Retain the official directory hierarchy and archive names when downloading: they contribute to pixel decoding policy and source provenance.

### 6.1 Download One Archive and All of Its Parts

Select an archive on the Files page at a pinned revision. To list candidates, the following command reads filenames only:

```bash
export HF_REPO='x-humanoid-robomind/RoboMIND'
# Set HF_REV for the current HF_REPO.
python - <<'PY'
import os
from huggingface_hub import HfApi
names = HfApi().list_repo_files(os.environ['HF_REPO'], repo_type='dataset', revision=os.environ['HF_REV'])
print('\n'.join(p for p in names if '.tar' in p or p.endswith('trajectory.hdf5')))
PY
export RM_PART='YOUR_SELECTED_OFFICIAL_ARCHIVE_OR_PART'
hf download "$HF_REPO" "$RM_PART" --repo-type dataset --revision "$HF_REV" \
  --local-dir "$DATA_ROOT/raw/VPT-06"
```

Download every part according to the official ordered inventory, verify sizes and checksums, concatenate the parts in order into a new archive, then verify the complete gzip / tar EOF before extraction. Do not encode an incomplete archive stream. `$RM_ARCHIVE` points to the verified complete local archive; `$RM_EXTRACTED` is its extracted directory, preserving the `h5_<embodiment>/...` hierarchy; `$RM_SOURCE_URI` records the official source URL at the pinned revision. HDF5 video typically lives under `observations/rgb_images/<camera>`; read text from the same file.

### 6.2 Official Encoding, Text, and RGB Correction

```bash
export RM_ARCHIVE="$DATA_ROOT/raw/VPT-06/YOUR_COMPLETE_ARCHIVE.tar.gz"
export RM_EXTRACTED="$WORK_ROOT/robomind/selected_archive"
export RM_SOURCE_URI="https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND/resolve/$HF_REV/YOUR_OFFICIAL_ARCHIVE_KEY"
python scripts/pretraining/encoding/encode_hdf5_text.py \
  --dataset "$RM_EXTRACTED" --vae "$VAE_ROOT" \
  --out-root "$WORK_ROOT/latents/single/VPT-06/official_rgbv1" \
  --archive-id 'selected_archive' --archive-source "$RM_SOURCE_URI" \
  --report "$WORK_ROOT/robomind/encode_report.json" \
  --fps 15 --size-mode aspect_bins --fit-mode letterbox_pad

python scripts/pretraining/prepare_dataset.py \
  --source VPT-06 --format robomind_report \
  --raw-root "$RM_EXTRACTED" --report "$WORK_ROOT/robomind/encode_report.json" \
  --archive-path "$RM_ARCHIVE" --archive-source "$RM_SOURCE_URI" \
  --out "$WORK_ROOT/episodes/VPT-06.official.jsonl"

python scripts/pretraining/build_manifest.py \
  --source VPT-06 --episodes "$WORK_ROOT/episodes/VPT-06.official.jsonl" \
  --latent-root "$WORK_ROOT/latents/single/VPT-06/official_rgbv1" \
  --out "$WORK_ROOT/manifests/single/VPT-06.official.csv"
```

Verify source FPS against the actual files and metadata; do not assume one rate for every embodiment. The official RGB policy uses two groups identified by **exact embodiment path components**:

| Official path class | Channel handling after decoding |
| --- | --- |
| `h5_franka_3rgb`, `h5_franka_1rgb`, `h5_ur_1rgb`, `h5_franka_fr3_dual` | Convert BGR to RGB according to the official convention |
| `h5_agilex_3rgb`, `h5_simulation`, `h5_sim_franka_3rgb`, `h5_sim_tienkung_1rgb`, `h5_tienkung_gello_1rgb`, `h5_tienkung_xsens_1rgb`, `h5_tienkung_prod1_gello_1rgb` | Retain decoded channel order according to the verified official convention |

For typed `uint8 HWC` arrays, identify the actual dimensions before processing; do not flatten the array and guess its image dimensions. Decode JPEG bytes under the relevant source convention. A global R / B swap is incorrect, and ordinary JPEG interpretation cannot be assumed for every official embodiment. Include the color policy, decoder version, native geometry, and model fingerprint in the new artifact contract.

Read task text from native HDF5 instruction fields such as `language_instruction` / `language_raw`; retain the field actually used and its exact original text. Report unknown field schemas instead of inventing an English sentence from the task directory name.

### 6.3 Failure Data Workflow and Coverage Limits

Failure data is often published as individual `.../data/trajectory.hdf5` files. The **`robomind_failure_standard_jpeg_v1`** policy decodes standard JPEG as RGB, recording the text schema and failure-source provenance separately. This differs from the official embodiment rules. The standalone `encode_failure.py` entry point accepts only standard JPEG bytes and does not apply the official embodiment channel policy:

```bash
export FAILURE_RAW="$DATA_ROOT/raw/VPT-06/failure/selected_subset"
# Verify FPS from this batch's native metadata first; 30 is only a verified example.
export FAILURE_NATIVE_FPS=30
python scripts/pretraining/encoding/encode_failure.py \
  --dataset "$FAILURE_RAW" --vae "$VAE_ROOT" \
  --source-fps "$FAILURE_NATIVE_FPS" \
  --out-root "$WORK_ROOT/latents/single/VPT-06/failure_rgbv1" \
  --episodes-out "$WORK_ROOT/episodes/VPT-06.failure.jsonl" --plan

# After reviewing the plan, encode; this also exports episode JSONL, so skip robomind_report.
python scripts/pretraining/encoding/encode_failure.py \
  --dataset "$FAILURE_RAW" --vae "$VAE_ROOT" \
  --source-fps "$FAILURE_NATIVE_FPS" \
  --out-root "$WORK_ROOT/latents/single/VPT-06/failure_rgbv1" \
  --episodes-out "$WORK_ROOT/episodes/VPT-06.failure.jsonl"
python scripts/pretraining/build_manifest.py \
  --source VPT-06 --episodes "$WORK_ROOT/episodes/VPT-06.failure.jsonl" \
  --latent-root "$WORK_ROOT/latents/single/VPT-06/failure_rgbv1" \
  --out "$WORK_ROOT/manifests/single/VPT-06.failure.csv"
```

Process different native FPS values in separate batches; do not apply one guessed rate to the entire source. `--plan` previews HDF5 counts without validating all JPEG frames or tensors. Run a small encoding and reconstruction check with the actual model before bulk use. Adding data requires validated manifests, complete text caches, and a new training snapshot.

Keep the following local locations separate:

```text
$DATA_ROOT/raw/VPT-06/official/...
$DATA_ROOT/raw/VPT-06/failure/...
$WORK_ROOT/latents/single/VPT-06/official_rgbv1/...
$WORK_ROOT/latents/single/VPT-06/failure_rgbv1/...
$WORK_ROOT/robomind/publications/<content_hash>/index.json
$WORK_ROOT/robomind/publications/<content_hash>/VPT-06.csv
```

Keep latents with different color or encoding contracts separate. Publishing a new snapshot requires validated outputs and compatible RGB, text, and sampling contracts; concatenating two CSVs alone is insufficient.

**Multiview.** Select two or three streams using their actual camera roles. `camera_left/right` can denote external cameras rather than left / right wrists; front / top and wrist roles are not interchangeable. Inspect `camera_top` orientation and record any required 180-degree correction per episode, not as a dataset-wide rotation. Keep samples with more than three cameras and ambiguous roles pending review.

## 7. VPT-07: InternData-A1

Official source: [InternRobotics/InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1), with the [project overview](https://internrobotics.github.io/interndata-a1.github.io/). The release contains task directories for multiple embodiments. The dataset card distinguishes `sim_updated` (LeRobot v2.1) and `sim_updated_lerobotv30` (v3). Obtain repository access first.

Download the README and one selected task first. `INTERN_SUBSET` must be a directory that exists in the pinned revision's file tree; avoid implicitly downloading every robot / task:

```bash
export HF_REPO='InternRobotics/InternData-A1'
# Set HF_REV for the current HF_REPO.
export INTERN_SUBSET='sim_updated/YOUR_CATEGORY/YOUR_ROBOT/YOUR_TASK'
hf download "$HF_REPO" --repo-type dataset --revision "$HF_REV" \
  --local-dir "$DATA_ROOT/raw/VPT-07" \
  --include "$INTERN_SUBSET/*" 'README.md'
export SOURCE='VPT-07'
export RAW_ROOT="$DATA_ROOT/raw/$SOURCE/$INTERN_SUBSET"
# Run the three processing commands from Section 2.2.
```

Retain complete `meta/`, `data/`, and `videos/` directories within each task. Common camera keys include `images.rgb.head`, `images.rgb.hand_left`, `images.rgb.hand_right`, or a single-arm hand camera. Treat `meta/info.json` as authoritative rather than assuming one field naming scheme across embodiments. Validate FPS using both metadata and actual video timelines.

**Text.** Prefer episode task mappings. A repository-level task may label every episode only when metadata establishes that the repository has exactly one task. Leave missing or truncated labels explicitly empty instead of reconstructing them. Count empty text and valid native labels separately in manifests.

**Version boundary.** New downloads require manifests tied to their exact source revision and episode identities. Official inputs are already LeRobot, so another format conversion is generally unnecessary.

**Multiview.** Use the common two-image rule for two cameras. For three cameras, prefer head above left / right hand. Select one explicit representative view from stereo head cameras; two head cameras do not represent two hands. Review combinations with ambiguous roles.

## 8. VPT-08: RoboCOIN

The official [RoboCOIN project](https://flagopen.github.io/RoboCOIN/) and [RoboCOIN dataset repository directory](https://huggingface.co/RoboCOIN/datasets) publish separate task repositories; tools are maintained in [FlagOpen/RoboCOIN](https://github.com/FlagOpen/RoboCOIN). Not every repository in the organization contains task data: website assets, for example, are outside the pretraining input scope.

Select an existing task repository, such as [RoboCOIN/alpha_bot_2_move_the_table](https://huggingface.co/datasets/RoboCOIN/alpha_bot_2_move_the_table). This example requires accepting its access conditions on the repository page before downloading with your authenticated account:

```bash
export HF_REPO='RoboCOIN/alpha_bot_2_move_the_table'
# Set HF_REV for the current HF_REPO.
hf download "$HF_REPO" --repo-type dataset --revision "$HF_REV" \
  --local-dir "$DATA_ROOT/raw/VPT-08/alpha_bot_2_move_the_table" \
  --include 'meta/*' 'data/*' 'videos/*' 'README.md'
export SOURCE='VPT-08'
export RAW_ROOT="$DATA_ROOT/raw/$SOURCE/alpha_bot_2_move_the_table"
# Run the three processing commands from Section 2.2.
```

Inputs consist of LeRobot episodes, task metadata, and camera videos. Camera fields vary by robot, including head, front-chest, and wrist. Record the exact repository IDs and revisions used in your build.

**Text.** Prefer episode tasks / task_index. A repository-level fallback requires metadata proving a unique task. Titles, directory names, and robot names do not replace task instructions.

**Multiview.** Select two or three cameras. When head, left-hand, and right-hand roles are known, place the head above the other two. Other three-view combinations require recorded roles or an explicitly reviewed order. Do not select the first three arbitrary cameras from a larger set.

## 9. VPT-09: FastUMI

The official public repository is [IPEC-COMMUNITY/FastUMI_100k_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot). Native inputs already use LeRobot v2.1, with tasks under `single_arm/<task>/` and `dual_arm/<task>/`. Dual-arm videos contain left and right wrist fisheye cameras. Keep the original two streams as inputs rather than treating a precomputed third-party multi view MP4 as two native cameras.

```bash
export HF_REPO='IPEC-COMMUNITY/FastUMI_100k_lerobot'
# Set HF_REV for the current HF_REPO.
hf download "$HF_REPO" --repo-type dataset --revision "$HF_REV" \
  --local-dir "$DATA_ROOT/raw/VPT-09" \
  --include 'dual_arm/Fold_the_Suit/*' 'README.md'
export SOURCE='VPT-09'
export RAW_ROOT="$DATA_ROOT/raw/$SOURCE/dual_arm/Fold_the_Suit"
# Run the three processing commands from Section 2.2.
```

A single-arm example is `single_arm/take_items_out_of_drawer/` in the official tree; confirm capitalization at the pinned revision. Keep the task's `meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/info.json`, and `data/`. Join each episode to its exact native task.

**Single-view.** Resize and pad each complete fisheye image isotropically as described in Section 2.3. Original circular dark corners remain; do not crop the image to an interior rectangle.

**Current multiview crop rule.** Place the two views side by side. For validated 1280×720 inputs, sample each view at 0%, 25%, 50%, 75%, and 100% of its timeline to choose one fixed horizontal crop for the entire video; retain all 720 original rows. A visible pixel has an RGB maximum channel value greater than 32. A column contributes to boundary detection only when it has more than 8 visible pixels. Leave a 24-pixel guard on both sides, trim at most 192 pixels per side, align boundaries to even coordinates, and retain at least 70% of the original width. Other input geometries must fail for review instead of inheriting these constants automatically.

The crop reduces outer black areas at the sides; fisheye dark corners may remain. The crop is fixed per video rather than changing between frames, and each camera uses one isotropic resize factor. The canvas is approximately 65k pixels with dimensions aligned to 32, chosen primarily to reduce padding. Dimensions vary by task; **512×128 is not a fixed output size**.

## 10. VPT-10R: Ego-Exo4D Color Streams

Follow the official [Ego-Exo4D access instructions](https://docs.ego-exo4d-data.org/getting-started/) and obtain approval first, then download selected takes using the [official CLI](https://docs.ego-exo4d-data.org/download/). This workflow does not use an unverified third-party Hugging Face mirror as its download source.

```bash
python -m pip install ego4d
egoexo -o "$DATA_ROOT/raw/EgoExo4D" --parts metadata
export TAKE_UID='YOUR_APPROVED_TAKE_UID'
egoexo -o "$DATA_ROOT/raw/EgoExo4D" --parts takes --uids "$TAKE_UID" --views ego exo
```

`takes` contains frame-aligned video, distinct from previews, feature files, and downscaled_takes. Use the official `takes.json` and `takes/<take>/frame_aligned_videos/<camera>.mp4` outputs. Record each take's official metadata, camera types, and exact filenames before building a type-filtered RGB input inventory.

RGB inputs include Aria / external camera streams explicitly marked as RGB; SLAM streams belong to 10S. `--views ego` alone cannot distinguish them because one Aria take may contain both. Place reviewed RGB files into a derived input directory that preserves the take / frame_aligned_videos structure; links to downloaded files can avoid copying videos. Do not process every MP4 twice under both source IDs.

```bash
export SOURCE='VPT-10R'
export RAW_ROOT="$DATA_ROOT/converted/EgoExo4D/rgb/takes"
python scripts/pretraining/prepare_dataset.py \
  --source "$SOURCE" --format video_tree --raw-root "$RAW_ROOT" \
  --takes "$DATA_ROOT/raw/EgoExo4D/takes.json" \
  --pattern '*/frame_aligned_videos/*.mp4' \
  --out "$WORK_ROOT/episodes/$SOURCE.jsonl"
python scripts/pretraining/encoding/encode_latents.py \
  --dataset "$RAW_ROOT" --layout video_tree \
  --video-glob '*/frame_aligned_videos/*.mp4' \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/latents/single/$SOURCE" \
  --fps 15 --size-mode aspect_bins --fit-mode letterbox_pad
python scripts/pretraining/build_manifest.py \
  --source "$SOURCE" --episodes "$WORK_ROOT/episodes/$SOURCE.jsonl" \
  --latent-root "$WORK_ROOT/latents/single/$SOURCE" \
  --out "$WORK_ROOT/manifests/single/$SOURCE.csv"
```

**Text.** Join the exact `take_name` to `task_name` in the official `takes.json`. This is take-level task text, not second-by-second narration, commentary, or generated captions. A missing exact take match cannot be repaired by approximating the directory name.

**Multiview.** This recipe retains the original individual camera streams and does not generate Ego-Exo multi view videos.

## 11. VPT-10S: Ego-Exo4D Monochrome SLAM Streams

The source, license, downloads, and original take metadata are the same as 10R. Source 10S is a **separate sampling source** because it uses monochrome SLAM images. Replicate grayscale into three equal channels for the RGB VAE. The monochrome appearance is native to the data, not an R / B channel error, and should not be artificially colorized.

Select SLAM files using official camera / stream metadata and place them under `$DATA_ROOT/converted/EgoExo4D/slam/takes`, preserving `<take>/frame_aligned_videos/<camera>.mp4`. Downloading one take can supply both 10R and 10S source files without duplicate downloads.

```bash
export SOURCE='VPT-10S'
export RAW_ROOT="$DATA_ROOT/converted/EgoExo4D/slam/takes"
python scripts/pretraining/prepare_dataset.py \
  --source "$SOURCE" --format video_tree --raw-root "$RAW_ROOT" \
  --takes "$DATA_ROOT/raw/EgoExo4D/takes.json" \
  --pattern '*/frame_aligned_videos/*.mp4' \
  --out "$WORK_ROOT/episodes/$SOURCE.jsonl"
python scripts/pretraining/encoding/encode_latents.py \
  --dataset "$RAW_ROOT" --layout video_tree \
  --video-glob '*/frame_aligned_videos/*.mp4' \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/latents/single/$SOURCE" \
  --fps 15 --size-mode aspect_bins --fit-mode letterbox_pad
python scripts/pretraining/build_manifest.py \
  --source "$SOURCE" --episodes "$WORK_ROOT/episodes/$SOURCE.jsonl" \
  --latent-root "$WORK_ROOT/latents/single/$SOURCE" \
  --out "$WORK_ROOT/manifests/single/$SOURCE.csv"
```

Text remains the take's official `task_name`. Share a physical-take split identity with 10R so that the same action does not cross training / validation sets merely because its source ID differs. Multiview augmentation is disabled.

## 12. Multiview: Synchronize Native RGB and Encode a New Latent

### 12.1 Common Generation Workflow

```bash
python scripts/pretraining/multiview/prepare.py \
  --episodes "$WORK_ROOT/episodes/VPT-01.jsonl" "$WORK_ROOT/episodes/VPT-09.jsonl" \
  --out "$WORK_ROOT/multiview/plans-v1"
python scripts/pretraining/multiview/worker.py \
  --index "$WORK_ROOT/multiview/plans-v1/plan_index.json" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/multiview/smoke-v1" \
  --smoke --max-plans 1
```

Compare original cameras, composed RGB, and VAE reconstructions for orientation, color, cropping, and synchronization. Then use a new output directory for full encoding:

```bash
python scripts/pretraining/multiview/worker.py \
  --index "$WORK_ROOT/multiview/plans-v1/plan_index.json" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/multiview/encoded-v1" \
  --shard 0/1
python scripts/pretraining/build_manifest.py \
  --source VPT-09 --receipts "$WORK_ROOT/multiview/encoded-v1/receipts" \
  --out "$WORK_ROOT/manifests/multiview/VPT-09.csv"
```

For archived HDF5 inputs, use `scripts/pretraining/multiview/archive_worker.py`; its arguments and inputs follow the archive source recorded in the plan. Temporary HDF5 processing requires complete-archive validation before admission. The ordinary MP4 worker is not a universal adapter. Never publish `--smoke` receipts as production pretraining rows.

Inputs are native camera streams from the same physical episode. **Compose RGB first, then use the VAE to create a new latent**. Do not concatenate existing latents or combine images from unrelated episodes as multiview examples. Each plan retains camera selection, source objects, timeline, color / crop policies, role / orientation corrections, and text provenance.

### 12.2 Layout and Timeline

For two cameras, stack vertically when both frames are landscape; otherwise place them side by side. Sources 01 / 09 explicitly use horizontal placement. Three-camera layouts place head above the other two views, which occupy the lower left / right positions. Review combinations whose roles are unclear. Each view uses one resize factor for both axes. Canvas dimensions are multiples of 32, and area lies within `0.8…1.2 × 65,536`; padding occupies pixels outside the image placements.

Resample cameras within their shared time window to 15 Hz. For video, select nearest frames using actual PTS and check declared FPS, container metadata, and time gaps. For HDF5, select frames using verified FPS. Split long videos into non-overlapping segments of at most 257 RGB frames, each of length `1 + 4k`, corresponding to at most 65 latent frames. Do not encode tails shorter than 5 frames; separately record the remaining 0–3 frames removed for temporal alignment.

**frame_ids use different time bases.** Single-view payloads retain original source frame indices. Multiview payloads use the synchronized 15 Hz grid and declare `frame_id_timebase_fps=15`; original FPS / timeline metadata is stored for each camera separately. Matching keys alone does not prove identical frame-by-frame sampling between versions.

### 12.3 Admission Requirements

Validate each clip for finite values, THWC layout, 48 channels, temporal / spatial compression shape, FPS, source range, normalization, and task text. Read outputs back, verify tensor equality, then atomically write a complete receipt. Missing source objects, conflicting text, incorrect color policies, incomplete archives, and smoke samples prevent admission to formal manifests.

Group splits by stable original repository / episode identity so that all cameras, single views, multi view videos, and segments remain in one set. Map cross-source aliases and RGB / SLAM streams from the same take to that same identity. A validation split built only from new multi view videos does not prove that an older checkpoint never saw their original camera streams.

## 13. Text Caches, Frozen Snapshots, and Object Storage

All nine sources use task text with native provenance. Record empty text explicitly; do not fill it from filenames, robot names, or model-generated captions. Conflicting tasks require a reviewed episode mapping. Text keys are the **SHA-256 of exact UTF-8 content**, including whitespace. Changing a label invalidates its previous embedding.

```bash
python scripts/pretraining/publish_snapshot.py \
  --manifests "$WORK_ROOT/manifests/single/VPT-01.csv" "$WORK_ROOT/manifests/single/VPT-09.csv" \
  --out "$WORK_ROOT/snapshots/pilot-v1"
```

This is an explicitly limited two-source pilot. A full nine-source snapshot must list verified CSVs for all nine sources and the selected multiview CSVs. Add `--previous "$WORK_ROOT/snapshots/previous-version"` for subsequent versions to check preservation of earlier samples. Establish identity and compatibility across aliases, RGB, text, and sampling contracts before claiming a superset; increasing the total row count is insufficient.

Generate the configuration for the same snapshot before encoding text. See the [training guide](training.md) for model asset and configuration details:

```bash
python scripts/pretraining/make_config.py \
  --text-cache "$WORK_ROOT/text/phase-001" \
  --snapshot "$WORK_ROOT/snapshots/pilot-v1" --model-assets "$MODEL_ASSETS" \
  --run-root "$WORK_ROOT/runs" --out "$WORK_ROOT/configs/pilot-v1.yaml" \
  --allow-subset --batch-size 1 --num-steps 10
python scripts/pretraining/text/encode_prompt_cache.py \
  --cfg "$WORK_ROOT/configs/pilot-v1.yaml" --assets "$MODEL_ASSETS" \
  --manifests "$WORK_ROOT/snapshots/pilot-v1" \
  --out "$WORK_ROOT/text/pilot-v1" --allow-subset
```

Omit `--allow-subset` for the complete nine-source build. The training configuration, manifest task inventory, and cache index must agree. Encode new task embeddings before admitting their data to training.

Optional object storage should use your own namespace. Obtain the root URI from your storage console and provide it as `OBJECT_URI`; the following are path templates, with variables expanded by the caller:

```text
${OBJECT_URI}/openwam/raw/<source>/<revision>/...
${OBJECT_URI}/openwam/processed/latents/objects/<first_two_sha256_characters>/<sha256>
${OBJECT_URI}/openwam/processed/latents/snapshots/<snapshot>/index.json
$WORK_ROOT/catalogs/<snapshot>.sqlite
$WORK_ROOT/text/<snapshot>/index.json
```

Upload, catalog construction, and restore entry points are under `scripts/pretraining/storage/`. A validated catalog maps logical manifest paths to stored objects, so an absent original logical `.pth` path does not automatically indicate missing training data. Configure separate capacity limits for the runtime cache and raw-data staging.

Before reclaiming raw video, establish each file's completed latent / action / text dependencies and verify that your remote backup matches its full local content. Matching names, sizes, or multipart ETags alone do not prove equivalence. Keep derived multi view videos, raw files lacking encoding mappings, and temporary files still needed by processing outside any deletion set for verified backed-up raw video. This guide contains no automatic deletion commands.

## 14. Build Records

Retain source URLs and revisions, input inventories and checksums, original episode IDs, camera / FPS / RGB contracts, VAE fingerprints, completion receipts, text provenance, and frozen CSV / split / text-cache indices. These records identify exactly what entered training.

LIBERO collection and HDF5-to-LeRobot conversion, UMI raw Zarr conversion, and Ego-Exo4D access / downloads use their upstream tools. OpenWAM handles preparation and encoding of the supported inputs described above. See [Validation](validation.md) before scaling up a build.
