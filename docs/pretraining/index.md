# OpenWAM Video Pretraining

This guide covers the **datasets used for OpenWAM video pretraining**, preprocessing, single-view and RGB multi view latents, text, snapshots, and training. It uses the existing `causal_video_prediction` policy and `video_only_decoder`, without action supervision or a separate training runtime.

## Package Boundaries

| Responsibility | Owner |
| --- | --- |
| Source metadata, RGB decoding, resize bins, manifests, snapshots | `open_wam.data.preparation` |
| Dataset-specific camera selection | `data.preparation.camera_recipes` |
| Explicit camera-order layout and rasterization | `data.preparation.multiview` |
| Wan VAE execution and normalization | `models.visual_tower.vae_encoding` |
| Offline text lookup | `models.visual_tower.prompt_cache` |
| Verified object cache, publication, and restore | `open_wam.artifacts` |
| Launch/config commands | `open_wam.cli` |
| Training semantics and loss | Existing policy, shared `TrainingRuntime`, and video decoder |

Scripts under `scripts/pretraining/` only call their importable package entry
points. Installed users can run the corresponding module with `python -m`.
The template is `configs/examples/video_pretraining.yaml`, included in the wheel.

For a new source, provide its native episode metadata and explicit camera order
to the preparation functions; the compositor does not infer source names or
learned camera roles. The supplied recipe can translate familiar source camera
conventions, but those conventions are not part of the model. Existing per-view
latent assembly and the new RGB-before-VAE multi view composition are distinct transformations;
this workflow does not substitute one for the other.

**Published pretraining model weights:** [OpenWAM-Stanford/OpenWAM-Pretraining on Hugging Face](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining). This repository hosts the OpenWAM pretraining model weights, not the pretraining datasets. Dataset download sources are listed individually below and in the [pretraining dataset guide](datasets.md).

## Single-view + multiview pretraining

**Start with the [single-view + multiview training guide](multiview_training.md)** to train on the union of original camera clips and newly encoded RGB multi views. It provides the seven supported source layouts and the complete command chain from publishing a mixed snapshot to predicting a multiview video. EgoExo4D RGB and SLAM remain single-view inputs.

Use `make_config.py --require-multiview` and `train.py --require-multiview` for this setting. Both commands verify that the selected snapshot contains original single-view clips and completed RGB multi view clips, and report their counts. The generated configuration is named `openwam_single_and_multiview_pretraining`; its per-source manifests jointly describe both representations. Each multi view composition is already one complete latent stream, so `camera_names: [observation.images.slot0]` does not indicate a single-camera-only dataset.

The [single-source pilot below](#complete-pilot-build-for-one-pretraining-source) remains a single-view setup for checking the basic pipeline. Use the linked mixed-data guide when preparing a single-view + multiview run.

This is a reproducible build workflow with configurable paths. Dataset size depends on the downloaded and pinned revision, the files that pass validation, and the published snapshot. A current public conversion repository does not automatically reproduce an earlier training manifest. The dataset guide describes this limitation for each source.

## Reading order

1. **[Pretraining dataset sources, downloads, layouts, and processing commands](datasets.md)**: detailed instructions for all nine source IDs.
2. **[Single-view + multiview pretraining](multiview_training.md)**: explicit mixed-data settings, camera layouts, admission checks, training, and inference.
3. **[Data processing and validation](data_processing.md)**: single-view encoding, RGB multiview composition, text, manifests, and incremental data publication.
4. **[Object-storage backup, on-demand reads, and caching](storage.md)**: upload verification, recovery metadata, and local storage limits.
5. **[Training, resuming, and video inference](training.md)**: model assets, batches, checkpoints, and outputs.
6. [Variable-length batch contract](batching.md): the scope of `strict / padded / bucket / packed`.
7. [Validation and reproducibility limits](validation.md): numerical gates, deliberate corrections, and full-model acceptance checks.

See the [dataset registry](../assets/pretraining/datasets.yaml) for source descriptions and processing rules. This guide is the maintained entry point for OpenWAM video pretraining.

## Pretraining datasets

| ID | Pretraining dataset family | Single-view inputs | Additional multiview inputs | Text source |
| --- | --- | --- | --- | --- |
| VPT-01 | LIBERO | Encode each camera separately | Agent and wrist views, side by side | Native episode task |
| VPT-04 | UMI | Encode each valid camera separately | Compose only synchronized pairs | Native episode/task metadata |
| VPT-05 | AgiBot | Head, left hand, and right hand | Head above both hand views | Exact original episode_id join to task_info.task_name |
| VPT-06 | RoboMind | Encode valid RGB cameras separately | Select documented pairs or triples; prefer camera_top | Native HDF5 instructions; separate color handling for failure data |
| VPT-07 | InternData | Encode each valid camera separately | Select layout by camera role and aspect ratio | Native LeRobot task metadata |
| VPT-08 | RoboCOIN | Encode each valid camera separately | Select layout by camera role and aspect ratio | Native LeRobot task metadata |
| VPT-09 | FastUMI | Retain single-view inputs | Bimanual videos side by side, with conservative outer black-column cropping | episode.tasks; never invent descriptions from directory names |
| VPT-10R | EgoExo4D RGB | Encode each RGB camera separately | No multiview processing | Native task_name from takes.json |
| VPT-10S | EgoExo4D SLAM | Encode each SLAM stream separately | No multiview processing | Native task_name for the same take |

These nine IDs identify registered pretraining sources, not nine mutually exclusive upstream formats. Use the adapter matching the downloaded LeRobot, HDF5, tar, or frame-aligned video format. Their inclusion here describes video pretraining inputs; it does not designate downstream policy training or benchmark evaluation splits.

## Environment and directories

Run the commands below from the repository root. Use Python 3.11 or 3.12. Actual VAE encoding and large-model training require a suitable CUDA/PyTorch environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,pretrain]'

export DATA_ROOT=/path/to/openwam-data
export WORK_ROOT=/path/to/openwam-work
export MODEL_ROOT=/path/to/openwam-models
export VAE_ROOT="$MODEL_ROOT/wan22/vae"
export MODEL_ASSETS="$MODEL_ROOT/video-init"
mkdir -p "$DATA_ROOT" "$WORK_ROOT" "$MODEL_ROOT"
```

The `train` extra supplies PyTorch, Diffusers, HDF5, Arrow, text-model dependencies, and related packages. The `pretrain` extra adds PyAV, the object-storage SDK, and OpenCV. If an environment already provides OpenCV, use the existing `cv2` installation and avoid installing overlapping OpenCV wheels. The optional RLDS adapter also needs TensorFlow and `tensorflow-datasets`; the default nine-source workflow does not require them.

Suggested layout:

```text
DATA_ROOT/
  raw/VPT-01/...                  # Downloaded originals, including metadata
  converted/VPT-05/<repo>/...      # Reproducible format conversions where needed
WORK_ROOT/
  episodes/VPT-01.jsonl           # Original episode identities, cameras, durations, text
  latents/single/VPT-01/<repo>/...
  manifests/single/VPT-01.csv
  manifests/multiview/VPT-01.csv
  multiview/plans-v1/plan_index.json
  multiview/encoded-v1/{latents,receipts,previews}/...
  snapshots/phase-001/{VPT-*.csv,snapshot.json}
  text/phase-001/{index.json,prompts.jsonl,embeddings/}
  storage/{inventory.jsonl,receipts/,sealed/}
  $WORK_ROOT/configs/phase-001.yaml
  runs/...
```

Keep only code, templates, and documentation in the repository. Store original videos, large tensors, model files, and caches in the configurable directories above.

## Complete pilot build for one pretraining source

First download a pinned LeRobot repository using the [LIBERO download instructions](datasets.md). The example assumes that its root contains `meta/info.json` and that `RAW_REPO` is exactly the same scan root used by the encoder.

```bash
export RAW_REPO="$DATA_ROOT/raw/VPT-01/libero"

python scripts/pretraining/prepare_dataset.py \
  --source VPT-01 --format lerobot --raw-root "$RAW_REPO" \
  --out "$WORK_ROOT/episodes/VPT-01.jsonl"

python scripts/pretraining/encoding/encode_latents.py \
  --dataset "$RAW_REPO" --vae "$VAE_ROOT" \
  --out-root "$WORK_ROOT/latents/single/VPT-01" \
  --fps 15 --size-mode aspect_bins --fit-mode letterbox_pad \
  --store-dtype fp16 --dtype bf16 --workers 2 --batch-size 1

python scripts/pretraining/build_manifest.py \
  --source VPT-01 --episodes "$WORK_ROOT/episodes/VPT-01.jsonl" \
  --latent-root "$WORK_ROOT/latents/single/VPT-01" \
  --out "$WORK_ROOT/manifests/single/VPT-01.csv"

python scripts/pretraining/publish_snapshot.py \
  --manifests "$WORK_ROOT/manifests/single/VPT-01.csv" \
  --out "$WORK_ROOT/snapshots/pilot"

python scripts/pretraining/make_config.py \
  --text-cache "$WORK_ROOT/text/phase-001" \
  --snapshot "$WORK_ROOT/snapshots/pilot" --model-assets "$MODEL_ASSETS" \
  --run-root "$WORK_ROOT/runs" --out "$WORK_ROOT/configs/pilot.yaml" \
  --allow-subset --batch-size 1 --num-steps 10
```

This pilot contains single-view data and deliberately omits `--require-multiview`. To include completed multi view compositions, follow the [mixed pretraining guide](multiview_training.md). These commands generate the configuration; they do not start training. Next, prepare the text cache and launch the run using the [training guide](training.md). A small pilot must contain physical episodes on both sides of the fixed hash-based train/validation split. A single demonstration cannot provide an independent validation set.

## Encoding and pretraining rules

- Sample inputs at **15 FPS**. Record source FPS, source frame indices, and sampled frame counts separately to prevent accidental second-pass downsampling.
- Single-view inputs use aspect-preserving resizing and letterboxing by default; they are not all stretched to 256×256. Default bins include 128×128, 256×256, 352×256, and 352×192, all stated as width×height.
- Compose multiview inputs in RGB space, then encode the whole canvas with Wan2.2 VAE once. Canvas dimensions are aligned to 32 pixels, with area within ±20% of 256×256 and a preference for less additional padding.
- Resize every component view with a single scale factor for both axes. FastUMI retains the full native height and crops only outer black columns. Record the `camera_top` 180° correction in the plan; do not apply it to every camera.
- Use the normalized VAE posterior mean. Store tensors as **fp16 / THWC / 48 channels**, with spatial stride 16 and temporal stride 4. RGB frame counts satisfy `1 + 4k`.
- Obtain text from native metadata. Mark missing text explicitly and resolve conflicting labels explicitly. The current task-prompt training configuration requires complete native labels; recover missing labels first or place those samples in a separate unconditional recipe. The UMT5 cache includes the empty string for classifier-free dropout.
- Train on the union of original single-view and additional multiview inputs. A shared `physical_episode_key` prevents the same demonstration from crossing the pretraining train/validation split. Adding episodes does not change existing episodes' assignments.
- The default recipe uses **bucket batching, batch size 36 per rank, and gradient accumulation 1**. Eight ranks give a global batch of 288. GPU memory capacity still needs to be measured: begin with a small batch before increasing it.
- Save every **1000 optimizer steps** by default and retain the latest **2** checkpoints. Publish a new snapshot and rebuild the DataLoader only at a phase boundary with a complete checkpoint.

## Script index

| Script | Purpose |
| --- | --- |
| `scripts/pretraining/download_hf.py` | Select repository, revision, and file patterns explicitly; record the resolved commit; preview a download plan |
| `scripts/pretraining/convert_agibot.py` | Extract RGB from one archive, join original episode_id to task_name, and generate LeRobot metadata |
| `scripts/pretraining/prepare_dataset.py` | Export original episodes, cameras, tasks, and physical identities |
| `scripts/pretraining/encoding/encode_latents.py` | Single-camera VAE encoding for LeRobot v2/v3 and video_tree inputs |
| `scripts/pretraining/encoding/encode_hdf5_text.py` | Official RoboMind HDF5 with explicit HWC geometry, embodiment-specific colors, and native text |
| `scripts/pretraining/encoding/encode_failure.py` | Dedicated standard-JPEG encoder for RoboMind failure data |
| `scripts/pretraining/encoding/verify_latents.py` | Validate latent shape, temporal/spatial geometry, and finite values |
| `scripts/pretraining/multiview/prepare.py` | Freeze camera roles and geometry rules into immutable plans |
| `scripts/pretraining/multiview/worker.py` | Compose synchronized RGB, encode with the VAE, and write receipts and smoke previews |
| `scripts/pretraining/multiview/archive_worker.py` | Read each compressed archive once, staging only one HDF5 member at a time |
| `scripts/pretraining/build_manifest.py` | Generate single-view or multiview CSVs after integrity validation |
| `scripts/pretraining/publish_snapshot.py` | Merge and deduplicate inputs, check additive invariants, and publish pretraining snapshots |
| `scripts/pretraining/text/encode_prompt_cache.py` | Generate fingerprint-bound UMT5 text embeddings |
| `scripts/pretraining/storage/{inventory,upload,seal,restore}.py` | Verified object-storage backups and metadata-only restoration |
| `scripts/pretraining/make_config.py` | Generate source sampling weights and configuration; `--require-multiview` verifies a single-view + RGB multi view snapshot |
| `scripts/pretraining/train.py` | Pin the snapshot, validate text and the requested view mixture, and invoke the shared training runtime |
| `scripts/pretraining/infer.py` | Generate target/prediction videos from complete model weights |

`encoding/encode_hdf5.py` remains a shared geometry dependency and legacy-format entry point. New RoboMind builds must use the two color-policy-aware encoders in the table. `encode_rlds.py` is an optional adapter; its presence does not mean all nine pretraining sources use RLDS.

## Checks and reproducibility limits

```bash
python -m pip install pytest
python -m pytest -q \
  tests/test_pretraining_workflow.py tests/test_pretraining_rgb.py \
  tests/test_pretraining_video_adapters.py \
  tests/test_pretraining_object_cache.py tests/test_pretraining_storage_roundtrip.py \
  tests/test_offline_prompt_cache.py tests/test_causal_video_batching.py \
  tests/test_mixed_video_physical_split.py
```

CPU checks cover geometry, colors, source metadata→manifest→configuration, snapshot invariants, cache budgets/read leases/corruption recovery, text, and batch isolation. CUDA encoding, training, and rollout with the full 30-layer model still require acceptance checks in the actual execution environment using the pilot commands above. Passing CPU checks does not establish that a complete dataset re-encoding or model-training run has finished.

See the [validation record](validation.md) for the complete check commands, dependency versions, and known baseline issues.

The [published OpenWAM pretraining weights](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining) contain model weights only. They do not restore AdamW moments, the scheduler, or the DataLoader cursor. See the [training guide](training.md) for downloading weights and the distinction between initialization and full-state resumption.
