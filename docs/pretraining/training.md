# Video Pretraining, Resumption, and Prediction Videos

This guide covers video pretraining on the nine [pretraining dataset sources](datasets.md), including single-view and RGB multi view latent inputs. It describes video prediction objectives and text conditioning; downstream action-policy training has its own recipes. The released model weights are available at [OpenWAM-Stanford/OpenWAM-Pretraining on Hugging Face](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining).

**For the explicit single-view + multiview setting, start with [multiview_training.md](multiview_training.md).** It shows the camera layouts, preserves the original singles while publishing completed RGB multi views, and provides the complete mixed-data command chain. The examples below use a mixed snapshot and enforce its composition with `--require-multiview`.

The workflow follows `ExperimentConfig → VariantPipeline → VisualTower → PolicyVariant → ActionDecoder`. Pretraining data preparation is separate from the training runtime. The entrypoint uses the existing shared `TrainingRuntime` for both single-view and multi view composition inputs.

## 1. Model Assets

Use mutually compatible video transformer, Wan2.2 VAE, UMT5 text encoder, and tokenizer assets. The public [Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) provides a [VAE configuration](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/blob/main/vae/config.json) with 48 latent channels, 16× spatial compression, and 4× temporal compression. [LingBot-VA base](https://huggingface.co/robbyant/lingbot-va-base) provides the reference transformer format and text assets used by this repository.

Download assets with a recorded revision:

```bash
# An initial download may use main; the tool records the resolved commit.
# Use that exact commit when repeating the experiment.
python scripts/pretraining/download_hf.py \
  --repo Wan-AI/Wan2.2-TI2V-5B-Diffusers --repo-type model --revision main \
  --include 'vae/**' 'transformer/**' --out "$MODEL_ROOT/wan22"

python scripts/pretraining/download_hf.py \
  --repo robbyant/lingbot-va-base --repo-type model --revision main \
  --include 'transformer/**' 'vae/**' 'text_encoder/**' 'tokenizer/**' \
  --out "$MODEL_ROOT/lingbot-va-base"
```

Convert the Wan video weights into the initialization format used here:

```bash
python scripts/convert_wan22_diffusers_to_lingbot_init.py \
  --wan-diffusers-root "$MODEL_ROOT/wan22/transformer" \
  --lingbot-template-root "$MODEL_ROOT/lingbot-va-base/transformer" \
  --output-root "$MODEL_ROOT/video-init/transformer"

# Link the encoding/decoding assets into the new asset root without copying weights.
ln -s "$MODEL_ROOT/wan22/vae" "$MODEL_ROOT/video-init/vae"
ln -s "$MODEL_ROOT/lingbot-va-base/text_encoder" "$MODEL_ROOT/video-init/text_encoder"
ln -s "$MODEL_ROOT/lingbot-va-base/tokenizer" "$MODEL_ROOT/video-init/tokenizer"
export MODEL_ASSETS="$MODEL_ROOT/video-init"
export VAE_ROOT="$MODEL_ASSETS/vae"
```

The converter's output directory must not already exist. An independently released compatible transformer can be used directly without conversion. Record the chosen initialization with your experiment. See [video_only_training.md](../video_only_training.md) for details of the Wan-to-LingBot weight conversion.

All inputs in one pretraining pool must use the same VAE weights and normalization. A shared “Wan2.2” label does not establish file equivalence. Multiview contracts and RoboMind encoding evidence record weight-file fingerprints.

## 2. Generate a Configuration from a Mixed Snapshot

```bash
python scripts/pretraining/make_config.py \
  --text-cache "$WORK_ROOT/text/phase-001" \
  --snapshot "$WORK_ROOT/snapshots/phase-001" \
  --model-assets "$MODEL_ASSETS" --run-root "$WORK_ROOT/runs" \
  --out "$WORK_ROOT/configs/phase-001.yaml" --require-multiview \
  --batch-size 36 --batching bucket --bucket-pool-size 1152 --num-steps 1000
```

This example assumes `phase-001` was published from the original single-view manifests together with completed RGB multi view manifests. `--require-multiview` checks both representations and reports their clip counts; a purely single-view or purely multi view composition snapshot fails. The generated configuration is named `openwam_single_and_multiview_pretraining`.

By default, all nine pretraining sources must be present and every clip must have a nonempty native task label. Add `--allow-subset` explicitly for a small source-subset trial. A mixed pilot still needs both representations; omit `--require-multiview` only when intentionally following the README's single-view pilot. Missing labels are rejected before configuration generation and cannot enter the current task-prompt recipe. Data preparation may retain a missing-label status so native labels can be recovered later, or the data can be used in a separately configured unconditional experiment. `--num-steps` specifies the **final optimizer step**, not a number of additional updates for this invocation.

Each multi view composition was composed in RGB space and encoded by the VAE as one complete latent stream. The merged CSVs carry `augmentation=single_view` or `augmentation=multi_view`; they select the input representation. There is no `multiview: true` model switch. The single `observation.images.slot0` in the latent configuration can therefore represent an entire multi view composition. Original single-camera and multi view composition rows share the same trainer and video-prediction objective.

The base model has 30 layers, hidden size 3072, 24 attention heads, 48 latent channels, text dimension 4096, and a maximum text length of 512 tokens. The default objective is latent flow only. AdamW uses learning rate `1e-5`, betas `(0.9, 0.95)`, weight decay `.01`, gradient accumulation 1, and task-text dropout `.1`.

Each sample contains at most 64 latent frames, with observed-prefix/future-suffix geometry sampled from the configured buckets. Supervision covers only valid future regions. CSV lengths are already in latent-frame units; no second RGB-FPS normalization is applied.

The default `--weights hours` assigns manual sampling weights using each source's published encoded-view hours. `--weights balanced` gives every source equal weight. With hour weighting, additional multiview data can change source sampling proportions; configuration generation prints the selected weights. Record the resolved configuration. Clip counts and upstream duration do not establish how often data was sampled during training. Neither weighting option fixes a single-view/multiview ratio in each batch; use consumed-sample records to measure actual exposure.

## 3. Native Task Text and the UMT5 Cache

```bash
python scripts/pretraining/text/encode_prompt_cache.py \
  --cfg "$WORK_ROOT/configs/phase-001.yaml" --assets "$MODEL_ASSETS" \
  --manifests "$WORK_ROOT/snapshots/phase-001" \
  --out "$WORK_ROOT/text/phase-001" --device cuda:0 --batch-size 8

# The configuration above already selects this directory with --text-cache.
```

Add `--allow-subset` to this command as well for a source-subset trial. The cache collects the first instruction from each actual task-text record, plus the empty string. It uses the original tokenizer's truncation and attention mask, saving unpadded bf16 tensors of shape `[tokens, 4096]`. At load time, tensors are padded to the model's configured capacity of 512 tokens.

The cache's encoded `max_text_tokens` must equal `backbone.max_text_tokens`
(512 in this recipe). Changing the limit requires re-encoding: UMT5 is
bidirectional, so slicing embeddings encoded with a longer prompt is not
equivalent to encoding the truncated prompt. A mismatch fails at cache loading;
embeddings are never silently truncated.

Outputs include `index.json`, `prompts.jsonl`, and `embeddings/<prompt_SHA256>.pt`. The cache records SHA256 fingerprints of the encoder/tokenizer files, manifest SHA256 values, the prompt-inventory SHA256, dimensions, dtype, and completion status. Training accepts only a complete index. To pin an independently reviewed encoder fingerprint, use `make_config.py --text-encoder-fingerprint <SHA256>`, or set it explicitly:

```yaml
backbone:
  load_text_conditioning: false
  prompt_cache:
    root: /path/to/text/phase-001
    encoder_fingerprint: <reviewed SHA256>
```

In this recipe, `backbone.load_text_conditioning: false` avoids loading online UMT5 weights into every training process. With the cache configured, native task text still reaches the visual model's cross-attention. Unknown prompts, wrong dimensions, a corrupt inventory, or a mismatched encoder fingerprint raise errors rather than becoming empty text.

When new data introduces new task text, prepare a complete cache for the new snapshot. Embeddings from an older cache with the same encoder fingerprint and token limit can be copied or linked before encoding missing prompts. Do not modify a complete cache that an active training phase is reading.

For online encoding, select `make_config.py --online-text` instead of `--text-cache`. Online UMT5 then occupies resources in the training process. Text source selection is recorded in the resolved config, never read from process-global pretraining environment variables. Cache generation explicitly loads the online encoder even when the input config selects a cache.

## 4. Validate and Launch Video Pretraining

```bash
python scripts/pretraining/train.py \
  --snapshot "$WORK_ROOT/snapshots/phase-001" \
  --cfg "$WORK_ROOT/configs/phase-001.yaml" --require-multiview --check-only

# Launch eight GPU processes within one allocation after checking memory capacity.
torchrun --standalone --nproc-per-node=8 scripts/pretraining/train.py \
  --snapshot "$WORK_ROOT/snapshots/phase-001" \
  --cfg "$WORK_ROOT/configs/phase-001.yaml" --require-multiview --expected-world-size 8
```

For a single-GPU trial, generate a smaller-batch configuration and pass `--set trainer.strategy=single_device --expected-world-size 1`. These entrypoints do not submit scheduler jobs.

`--check-only` validates configuration and snapshot hashes, checks that a text-conditioning source is selected, and reports the single-view/multi view composition. With `--require-multiview`, both representations must be present. The actual launch repeats this check and logs a `pretraining_view_mixture` event. The check-only path does not run GPU training or establish readability of every remote object. For object-storage-backed pretraining data, use the explicit catalog/cache configuration in [storage.md](storage.md). Complete representative reads and a short training run in the target environment first.

Global batch size equals per-rank batch × number of ranks × gradient accumulation. Thus, `36 × 8 × 1 = 288`. Increasing accumulation increases the effective batch without placing more samples in a single forward pass.

## 5. Padded, Bucket, and Packed Modes

| Mode | Behavior | Current recipe |
| --- | --- | --- |
| `strict` | Original direct stacking; requires compatible shapes | Compatibility baseline |
| `padded` | Dynamic padding with masks preserving each sample's real extent | Supported |
| `bucket` | Length reordering within a bounded sampled pool, followed by padding | Default; pool size 1152 |
| `packed` | Removes batch-added padding before the transformer and isolates each sample's tokens | Supported; validate and compare on the intended GPU environment |

Select a mode with `make_config.py --batching packed` or `--set data.batching.mode=packed`. Bucketing groups similar lengths without requiring identical lengths. It retains sampled indices and repeated draws, changing only their order within a bounded pool. Variable spatial canvases still follow the configured shape grouping; camera images are not arbitrarily stretched to fit a batch.

Padding is excluded from valid future loss. Packed self-attention and cross-attention must not expose another sample's video or text. Batching preserves each sample's observed/future causal boundaries. See [batching.md](batching.md) for the full constraints and CPU/CUDA verification commands.

When increasing batch size, record steady-state steps/s, valid samples/s, peak GPU memory, input wait, cache misses, and out-of-memory failures. A short two-layer benchmark does not establish the throughput of a 30-layer model. Start with a small batch and increase it after stable execution.

## 6. Checkpoints and New Pretraining Data Phases

The default saves every **1000 optimizer steps** and retains the latest **2** checkpoints. Each directory contains the full model, optimizer, scheduler, training state, and a `.checkpoint_complete` marker. Only a completed checkpoint is eligible for full resumption. Two full checkpoints can still be large; a retention count is not a fixed byte budget.

To recover within the same snapshot while preserving the DataLoader cursor:

```bash
torchrun --standalone --nproc-per-node=8 scripts/pretraining/train.py \
  --snapshot "$WORK_ROOT/snapshots/phase-001" --cfg "$WORK_ROOT/configs/phase-001.yaml" \
  --resume-from /path/to/checkpoint/full_training_state.pt \
  --require-multiview --expected-world-size 8
```

To switch to a snapshot containing additional verified multiview pretraining data:

```bash
python scripts/pretraining/make_config.py \
  --text-cache "$WORK_ROOT/text/phase-002" \
  --snapshot "$WORK_ROOT/snapshots/phase-002" --model-assets "$MODEL_ASSETS" \
  --run-root "$WORK_ROOT/runs" --out "$WORK_ROOT/configs/phase-002.yaml" \
  --batch-size 36 --num-steps 2000 --require-multiview

python scripts/pretraining/text/encode_prompt_cache.py \
  --cfg "$WORK_ROOT/configs/phase-002.yaml" --assets "$MODEL_ASSETS" \
  --manifests "$WORK_ROOT/snapshots/phase-002" \
  --out "$WORK_ROOT/text/phase-002" --device cuda:0 --batch-size 8

# For remote tensors, also configure the catalog covering the new snapshot.
torchrun --standalone --nproc-per-node=8 scripts/pretraining/train.py \
  --snapshot "$WORK_ROOT/snapshots/phase-002" --cfg "$WORK_ROOT/configs/phase-002.yaml" \
  --resume-from /path/to/checkpoint/full_training_state.pt \
  --new-data-phase --require-multiview --expected-world-size 8
```

After restoring the model, optimizer, and scheduler, `--new-data-phase` explicitly resets the dataset cursor so the new phase samples from the expanded data. The optimizer step is preserved. Without this flag, the command performs ordinary in-place resumption. Changing the sampled data does not reproduce the previous data order exactly.

New data enters training at a completed-checkpoint boundary. Tensors are fetched on the fly, while each active phase reads an immutable CSV inventory. Train/validation assignment uses a fixed hash of physical episode identity and does not reshuffle existing recordings when new data arrives. This split does not make recordings already seen by earlier models into an independent benchmark.

Snapshot publication with `--previous`, configuration generation, and training
admission all verify the inventory and manifest hashes declared in
`snapshot.json`. Additive checks cover every declared source, including your own
source names; they do not discover manifests by a dataset-specific filename glob.

## 7. Download the Released Pretraining Weights

The weights-only OpenWAM release is hosted at **[OpenWAM-Stanford/OpenWAM-Pretraining](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining)**.

```bash
python scripts/pretraining/download_hf.py \
  --repo OpenWAM-Stanford/OpenWAM-Pretraining --repo-type model --revision main \
  --include 'model_state.pt' 'resolved_config.yaml' 'SHA256SUMS' 'README.md' 'release_metadata.json' \
  --out "$MODEL_ROOT/openwam-pretraining"
```

`model_state.pt` supports inference and model initialization. It does not contain the complete AdamW/scheduler history and is not a substitute for `full_training_state.pt`. For a warm start with the same architecture and matching assets, use `--initialize-weights-from`. This loads model tensors while keeping the optimizer, scheduler, and training counters fresh. Reserve `--resume-from` for a full training-state checkpoint.

```bash
torchrun --standalone --nproc-per-node=8 scripts/pretraining/train.py \
  --snapshot "$WORK_ROOT/snapshots/phase-001" \
  --cfg "$WORK_ROOT/configs/phase-001.yaml" \
  --initialize-weights-from "$MODEL_ROOT/openwam-pretraining/model_state.pt" \
  --require-multiview --expected-world-size 8
```

Generate a training configuration for your own pretraining snapshot and assets with `make_config.py`; do not reuse paths embedded in the release package. Strict loading reports architecture mismatches. Do not use `strict=False` to ignore a large set of missing parameters.

## 8. VPM Prediction Videos

```bash
python scripts/pretraining/infer.py \
  --cfg "$WORK_ROOT/configs/phase-001.yaml" \
  --weights "$MODEL_ROOT/openwam-pretraining/model_state.pt" \
  --assets "$MODEL_ASSETS" --source VPT-09 --multi-view-only \
  --split val --sample-index 0 --device cuda:0 --steps 25 --seed 1234 \
  --out "$WORK_ROOT/inference/fastumi-example-001"
```

Keep the matching text-cache and object-catalog settings in the resolved configuration. Outputs are `target.mp4`, `prediction.mp4`, and `summary.json`, which records the seed, source sample, prefix/future lengths, and the first future segment's latent MSE. A multiview dataset sample already contains the latent of a complete RGB multi view. Model prediction and VAE decoding therefore operate on the whole multi view composition; independent single-view predictions are not assembled afterward.

Change `--source` to inspect another source with completed multiview data. Do not request `--multi-view-only` for 10R/10S. Compare checkpoints using the same sample, seed, prefix/future lengths, steps, and guidance. A video from a different task does not isolate an effect of batch size.

The geometry/color smoke output named `*-vae.mp4` demonstrates VAE reconstruction. The `prediction.mp4` produced here is the pretrained model's future prediction. Full-model GPU inference is a separate acceptance step; CPU unit tests do not establish its numerical quality.
