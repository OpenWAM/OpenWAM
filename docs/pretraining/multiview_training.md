# Single-View + Multiview Video Pretraining

OpenWAM pretraining combines the original single-camera clips with additional **multi view RGB compositions, each encoded as one latent stream**. Both representations enter the same immutable snapshot and the same `causal_video_prediction` training runtime. This guide makes the mixed setting explicit, including camera layouts, data admission, configuration, text conditioning, training, and prediction videos.

Use `--require-multiview` with both `make_config.py` and `train.py` for this setting. The check requires original single-view clips **and** at least one completed RGB multi view clip, prints their counts, and rejects a purely single-view or purely multi view snapshot. Keep the original single-view pool with the additive publication check described below. The presence check alone does not establish that an entire historical pool has been retained.

Published model weights are available at [OpenWAM-Stanford/OpenWAM-Pretraining on Hugging Face](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining). That repository hosts pretraining weights; the [dataset guide](datasets.md) provides the download sources for the input datasets.

## 1. Supported Sources and Camera Layouts

Seven pretraining sources support additional multi view videos when an episode has synchronized cameras with identifiable roles:

| Source | Dataset | Camera selection | Multi view RGB layout |
| --- | --- | --- | --- |
| VPT-01 | LIBERO | Agent/external view and wrist view | **Horizontal:** agent on the left, wrist on the right |
| VPT-04 | UMI | Valid synchronized pair, or a documented head/left/right triple | Landscape pairs stacked vertically; portrait/square pairs side by side; triples use head above left/right |
| VPT-05 | AgiBot | Head color, left-hand color, right-hand color | Head above left hand and right hand |
| VPT-06 | RoboMind | Prefer `camera_top` with left/right wrist cameras when available | **Rotate `camera_top` by 180°**, place it above the left/right wrist views; record the rotation in the plan |
| VPT-07 | InternData | Synchronized cameras selected by native roles | Landscape pairs stacked vertically; portrait/square pairs side by side; triples use head above left/right |
| VPT-08 | RoboCOIN | Synchronized cameras selected by native roles | Landscape pairs stacked vertically; portrait/square pairs side by side; triples use head above left/right |
| VPT-09 | FastUMI | Left/right bimanual views | **Horizontal:** left view beside right view; conservative outer black-column cropping |

VPT-10R (EgoExo4D RGB) and VPT-10S (EgoExo4D SLAM) are excluded from multi view generation. Their independent camera streams remain in the single-view pretraining pool.

A source supporting multi view videos does not mean every episode contains all required cameras. The plan records the original camera count and the selected subset. For RoboMind, inspect the episode's actual camera keys instead of assuming that every episode has six views. Where a preferred triple is unavailable, only a valid, documented alternative can be selected. Two-camera inputs use sorted camera-key order unless a reviewed `multiview_cameras` override specifies the order. For a three-camera episode with a recognized head but no anatomical left/right labels, the fallback places the head above the remaining cameras in recorded name order; it does not establish anatomical left/right roles. Use a reviewed `multiview_cameras` mapping when the intended role order is not encoded by the native names. `camera_rotations_degrees` records any per-camera orientation correction. Do not rotate every camera because `camera_top` needs a correction.

Every selected camera preserves its aspect ratio. Two axes use one scale factor, and the composed canvas has dimensions divisible by 32 with area within ±20% of 256×256. The layout favors less extra padding. FastUMI retains the complete native height and crops only outer black columns with a fixed per-video rectangle; fisheye vignetting can remain. See [data processing](data_processing.md) for the crop thresholds, timestamp checks, and geometry rules.

## 2. Encode Completed RGB Multi View Videos

First prepare the native episode indexes and validated single-view manifests using the [dataset guide](datasets.md) and [processing guide](data_processing.md). Run the following commands from the repository root with the environment variables defined in the [pretraining README](index.md). Use new output names for each build because plans, snapshots, and completed caches are immutable.

```bash
python scripts/pretraining/multiview/prepare.py \
  --episodes "$WORK_ROOT"/episodes/VPT-*.jsonl \
  --out "$WORK_ROOT/multiview/plans-v1"

# Inspect RGB compositions and VAE reconstructions before the full encoding run.
python scripts/pretraining/multiview/worker.py \
  --index "$WORK_ROOT/multiview/plans-v1/plan_index.json" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/multiview/smoke-v1" --smoke

python scripts/pretraining/multiview/worker.py \
  --index "$WORK_ROOT/multiview/plans-v1/plan_index.json" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/multiview/encoded-v1"

# Official compressed RoboMind archives use this entry point.
python scripts/pretraining/multiview/archive_worker.py \
  --index "$WORK_ROOT/multiview/plans-v1/plan_index.json" \
  --vae "$VAE_ROOT" --out-root "$WORK_ROOT/multiview/encoded-v1"
```

The workers synchronize native RGB frames at 15 FPS, apply reviewed cropping and rotations, compose a canvas, and **run the whole canvas through the VAE once**. They do not concatenate separately encoded camera latents. Every completed multi view video becomes one latent stream in fp16 THWC format with 48 channels.

Generate a multiview CSV only for sources with completed output. For example, after FastUMI has completed clips:

```bash
python scripts/pretraining/build_manifest.py \
  --source VPT-09 \
  --receipts "$WORK_ROOT/multiview/encoded-v1/receipts" \
  --out "$WORK_ROOT/manifests/multiview/VPT-09.csv"
```

Repeat this command with VPT-01, VPT-04, VPT-05, VPT-06, VPT-07, or VPT-08 when their completed clips are available. A partial multiview build can enter pretraining while other encodings continue. Do not create empty CSVs for unfinished sources. Only validated `status=complete` receipts are admitted; official RoboMind archives also require a completed archive-unit record with verified gzip EOF/CRC. Smoke previews and in-progress outputs are excluded.

## 3. Publish a Snapshot Containing Both Representations

The following example assumes all nine original single-view manifests exist and at least one validated multiview manifest is available. The FastUMI CSV above provides one such completed source. Keep the original single-view snapshot as an explicit preservation baseline:

```bash
export SINGLE_SNAPSHOT="$WORK_ROOT/snapshots/single-base"
export MIXED_SNAPSHOT="$WORK_ROOT/snapshots/mixed-001"
export MIXED_CONFIG="$WORK_ROOT/configs/mixed-001.yaml"
export MIXED_TEXT_CACHE="$WORK_ROOT/text/mixed-001"

# If a complete single-view baseline already exists, use its path and skip this command.
python scripts/pretraining/publish_snapshot.py \
  --manifests "$WORK_ROOT"/manifests/single/VPT-*.csv \
  --out "$SINGLE_SNAPSHOT"

python scripts/pretraining/publish_snapshot.py \
  --manifests "$WORK_ROOT"/manifests/single/VPT-*.csv \
              "$WORK_ROOT"/manifests/multiview/VPT-*.csv \
  --previous "$SINGLE_SNAPSHOT" \
  --out "$MIXED_SNAPSHOT"
```

`--previous` requires every baseline row to remain present and unchanged. The merged per-source CSVs distinguish representations with `augmentation=single_view` and `augmentation=multi_view`. `snapshot.json` reports `single_clips` and `multi_view_clips` for each source, along with physical episode counts, text coverage, duration, and manifest hashes.

**Naming contract.** Multi view artifacts generated by earlier revisions of this PR use a previous naming contract. Rebuild the multi view plans, encoded latents, and completion receipts in new output directories, then regenerate their manifests, mixed snapshots, training configurations, and object catalogs together. Renaming receipt fields alone does not update the immutable encoding contract stored in the latent payload. Retain the original single-view data when publishing the rebuilt mixed snapshot. Do not mix artifacts from different naming contracts or alter an active training snapshot in place. The current representation is `augmentation=multi_view`; generated clip IDs retain the `multi_view/` prefix for readability.

Inference's `--multi-view-only` filter reads the manifest's `augmentation` field,
not the clip ID or dataset name. Rows without that field remain usable for
training and unfiltered inspection but are excluded by this explicit filter.
Existing manifests already carrying `augmentation=multi_view` need no migration
for this filter. Encoder implementation fingerprints remain strict: use a new
output directory when the encoding implementation changes, even for a code move.

For later additions, publish a new snapshot with `--previous` pointing to the current mixed snapshot. This preserves both the original singles and previously admitted multi view videos. Do not change a CSV while a training phase is using it.

## 4. Generate the Mixed Pretraining Configuration

```bash
python scripts/pretraining/make_config.py \
  --text-cache "$MIXED_TEXT_CACHE" \
  --snapshot "$MIXED_SNAPSHOT" \
  --model-assets "$MODEL_ASSETS" --run-root "$WORK_ROOT/runs" \
  --out "$MIXED_CONFIG" --require-multiview \
  --batch-size 36 --batching bucket --bucket-pool-size 1152 \
  --weights hours --num-steps 1000
```

`--require-multiview` makes the requested data mixture an explicit acceptance condition. It checks the snapshot and reports the single-view and multi view clip counts. Purely single-view and purely multi view snapshots fail this setting. The default configuration generator also requires all nine source IDs and native task labels for every clip. A source-subset pilot needs `--allow-subset`; it still needs both representations when `--require-multiview` is specified. The single-view-only pilot in the README intentionally omits this flag.

The generated `data.video_sources` entries point to the merged snapshot CSVs. Each source can contain both single-view and multi view rows. There is no `multiview: true` model switch: the camera composition is already encoded in each multi view tensor. A generated `camera_names: [observation.images.slot0]` represents one input latent stream, which may contain either a single camera or a complete RGB multi view video. It does not mean that the snapshot contains only single-camera data.

Both representations use the same shared `TrainingRuntime`, video-prediction objective, native-text conditioning, and valid-future-frame loss. The default bucket mode groups compatible spatial shapes and similar sequence lengths; it does not merge camera tensors inside a batch.

Snapshot clip counts describe the available data pool, not a guaranteed per-batch single-view/multiview ratio. `--weights hours` sets source sampling weights from encoded-view hours; `--weights balanced` gives equal source weights. Neither option imposes a separate fixed multi view quota. Actual exposure also depends on source weights, eligible windows, sampling, and the run length. Measure consumed samples to report a realized mixture.

## 5. Encode Text for the Merged Snapshot

```bash
python scripts/pretraining/text/encode_prompt_cache.py \
  --cfg "$MIXED_CONFIG" --assets "$MODEL_ASSETS" \
  --manifests "$MIXED_SNAPSHOT" --out "$MIXED_TEXT_CACHE" \
  --device cuda:0 --batch-size 8

# The generated configuration already selects $MIXED_TEXT_CACHE.
```

Add `--allow-subset` to the text-cache command if this is a source-subset pilot. Both single-view and multi view rows carry native task text. The cache includes their prompts and the empty-string embedding used for classifier-free dropout. If an expected encoder fingerprint is configured, update it to the matching completed cache as described in [training.md](training.md). For object-storage-backed inputs, also select the catalog that covers this snapshot's tensors and text embeddings using [storage.md](storage.md).

A completed cache is specific to its prompt inventory and encoder fingerprint. Prepare a new complete cache when a new snapshot introduces task text; keep an active phase's cache unchanged.

## 6. Check and Launch the Mixed Run

```bash
python scripts/pretraining/train.py \
  --snapshot "$MIXED_SNAPSHOT" --cfg "$MIXED_CONFIG" \
  --require-multiview --check-only

# Start with a smaller batch if GPU capacity has not yet been measured.
torchrun --standalone --nproc-per-node=8 scripts/pretraining/train.py \
  --snapshot "$MIXED_SNAPSHOT" --cfg "$MIXED_CONFIG" \
  --require-multiview --expected-world-size 8
```

The launch repeats the mixture check, so selecting a different single-view-only snapshot accidentally does not silently change the intended setting. `--check-only` validates the configuration and immutable snapshot and reports its input composition; it does not run optimizer updates or prove that every remote object can be read. Confirm representative latent/text reads and a short training run before a full run.

With batch size 36 per rank, eight ranks, and gradient accumulation 1, the global batch is 288. This is a configuration example, not a memory-capacity guarantee. No command here submits scheduler jobs.

For continuation from an existing complete checkpoint, add `--resume-from /path/to/checkpoint/full_training_state.pt`. When switching to a new expanded snapshot, also pass `--new-data-phase`, generate its configuration with the desired absolute final optimizer step, and prepare its matching text cache/catalog. Keep `--require-multiview` on both configuration generation and launch. See [checkpoint resumption](training.md#6-checkpoints-and-new-pretraining-data-phases) for full-state continuation and the separate weights-only initialization option.

A shared `physical_episode_key` keeps every camera, multi view video, and segment of the same physical recording on one side of the pretraining train/validation split. Adding a multi view video therefore does not put a second representation of a training episode into validation. Newly published episodes also retain stable split assignment across later additions.

## 7. Inspect a Multiview Prediction

Use weights from a completed checkpoint to evaluate the run above. Set `INFERENCE_WEIGHTS` to its model weights file:

```bash
export INFERENCE_WEIGHTS=/path/to/completed-checkpoint/model_state.pt

python scripts/pretraining/infer.py \
  --cfg "$MIXED_CONFIG" --weights "$INFERENCE_WEIGHTS" \
  --assets "$MODEL_ASSETS" --source VPT-09 --multi-view-only \
  --split val --sample-index 0 --device cuda:0 \
  --steps 25 --seed 1234 \
  --out "$WORK_ROOT/inference/mixed-001-fastumi"
```

Keep the matching text-cache and object-catalog settings in the resolved configuration. Choose a source that has a completed multi view video in the selected split. If a small pilot has no validation multi view video, inspect an explicitly selected training sample with `--split train`; it is then a training-sample visualization. VPT-10R and VPT-10S have no multi view videos and must not be requested with `--multi-view-only`.

The output contains `target.mp4`, `prediction.mp4`, and `summary.json`. The model predicts the complete multi view latent, and the VAE decodes it as one video. Predictions for individual cameras are not composed afterward. RGB/VAE smoke videos from the encoding step check data processing; `prediction.mp4` checks future prediction by the selected model weights.

The [published pretraining weights](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining) can also be used after downloading them through the [training guide](training.md#7-download-the-released-pretraining-weights). Such an inference uses the released model, not the newly trained checkpoint. Compare model versions with the same sample, seed, observed/future horizon, and inference settings. Neither a successful mixture check nor a completed data encoding establishes an improvement in model quality.
