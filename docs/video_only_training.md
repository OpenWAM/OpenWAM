# Video-Only Training

Open-WAM trains causal video prediction through the same public composition
boundary as its video/action policies:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

`causal_video_prediction` owns observed-prefix/future-suffix semantics. The
shared `VisualTower` executes the video transformer, and `video_only_decoder`
computes masked latent-flow supervision. No training or evaluation path calls
a concrete backbone directly.

This guide covers both the installed package and optional source-checkout
utilities. Training uses the package-owned CLI. The Wan conversion commands and
decoded side-by-side rollout command are source-checkout integrations under
`scripts/`; they are not included in the wheel or source distribution.

## Prepare An Initialization (Source Checkout)

Wan2.2 is published in two transformer layouts:

| Source | Converter |
| --- | --- |
| Raw upstream keys such as `blocks.*.self_attn.*` | `scripts/convert_wan22_to_lingbot_init.py` |
| Hugging Face Diffusers keys already aligned with LingBot | `scripts/convert_wan22_diffusers_to_lingbot_init.py` |

For `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, pin a source revision and use a
LingBot-format transformer as the target schema:

```bash
uv run python scripts/convert_wan22_diffusers_to_lingbot_init.py \
  --wan-diffusers-root /models/Wan2.2-TI2V-5B-Diffusers/transformer \
  --lingbot-template-root /models/lingbot-va-base/transformer \
  --output-root /models/wan22-diffusers-lingbot-init
```

Both converters publish into a new output directory only after every file has
been written successfully. They reject existing destinations, so stale shards
cannot survive a conversion. The Diffusers converter also validates the
template against its `config.json` model schema on the meta device and checks
every shard/index entry and tensor shape. Matching video tensors come from Wan;
the flattened patch MLP is derived from Wan's Conv3d patch embedding; any
remaining supported action tensors come from the structural template. Video
training does not optimize or export those action tensors.

## Data Contract

The canonical LIBERO preset is
`configs/experiments/causal_video_prediction_libero_latent_local.yaml`. Its
adapter discovers one or more local LeRobot repositories below
`data.local_root`, reads pre-encoded per-camera latents, and emits:

- one canonical latent canvas with validated, non-overlapping view placements;
- identical frame counts, channels, and dtypes across assembled views;
- identical, integral, nonempty, nondecreasing frame IDs when present;
- one exact integral latent stride shared by every view placement;
- text tensors matching the selected task-prompt or blank-text conditioning mode;
- an observed prefix followed by a supervised future suffix;
- explicit latent-frame counts in sample metadata.

The five maintained buckets are `1+3`, `2+6`, `3+9`, `4+12`, and `5+15`
observed/future latent frames. These are latent counts, not raw RGB frame
counts. Evaluation selects through the same adapter, so nonzero sample starts
retain the episode's causal VAE history and exact camera assembly.
Frame-ID count is intentionally independent of latent count because IDs may
describe raw encoder inputs; repeated IDs remain valid for boundary padding.
Generic policy fields such as `training.chunk_size` and `training.window_size`
do not define this objective and are rejected from causal video configs.
Serialized view layouts use the strict
`open_wam.canonical_view_layout.v1` schema; consumers should reject missing or
unknown versions instead of guessing camera placement semantics.

## Text Conditioning

Video-only training exposes one policy-level choice:

```yaml
policy_variant:
  text_conditioning_mode: task_prompt  # task_prompt | disabled
```

`task_prompt` is the canonical default. It requires one non-empty instruction
and one finite, nonzero `[B, tokens, dim]` positive embedding per sample. The
maintained LIBERO preset applies blank-text conditioning to `0.1` of training
samples and uses video guidance `5.0` at inference, matching the maintained
action-enabled LIBERO policy recipe. Classifier-free training replaces the
selected positive embeddings with the standard blank-text embedding; its
probability must be in `[0, 1)`. Any positive dropout probability requires a
blank-text embedding from the dataset adapter, normally configured through
`data.empty_text_embedding_path`; zeros are not a valid substitute.

`disabled` is unconditional video training. The shared frontend ignores task
strings and positive embeddings, then uses the standard blank-text embedding
as both text branches in training and inference when one is available. It
requires `training.text_condition_dropout_prob: 0.0` and
`inference.guidance_scale: 1.0`. Real pretraining should provide the encoded
blank prompt through `data.empty_text_embedding_path` (or the repository-local
`empty_emb.pt`) or a configured frontend text encoder. When neither is
available, the shared core uses its fixed zero no-context tensor; this keeps
asset-free synthetic smoke configs runnable. Existing latent datasets do not
need to be rewritten; their positive text payload is simply ignored in this
mode.

## Train

Configure paths in `configs/local_paths.yaml`, then launch the canonical preset:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
uv run --extra train python -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29614 \
  -m open_wam.cli.train \
  --config-name causal_video_prediction_libero_latent_local \
  --devices 4 \
  --save-root /runs/wan22_video_only \
  --set data.local_root=/datasets/libero_subsets \
  --set data.empty_text_embedding_path=/datasets/empty_emb.pt \
  --set backbone.pretrained_model_name_or_path=/models/wan22-diffusers-lingbot-init
```

A source checkout also provides
`scripts/run_causal_video_prediction_posttrain_libero.sh` as a convenience
launcher for the same package training entrypoint. It is not a separate
training implementation or an installed API.

The preset trains only `visual_tower.shared_video_backbone` and freezes the
action runtime and runtime adapters. Full training state remains the default,
so optimizer, scheduler, strategy/scaler, and counters support stateful
continuation. Process and dataloader RNG streams are not checkpointed, so a
restarted run is not bitwise identical.

Each checkpoint also contains a scoped `transformer/` export. Its
`runtime_backbone_manifest.json` records the semantic component set and exact
tensor inventory. Loading recomputes that inventory from the active visual
component topology and rejects tensors outside the declared components. A
later policy loader therefore initializes omitted action components from its
model defaults without relying on a remembered consumer override. Complete
policy exports continue to declare
`visual_tower.runtime_backbone` and retain their existing load behavior.

## Evaluate (Source Checkout)

Generate an exact adapter-selected target and open-loop prediction:

```bash
uv run python scripts/generate_video_only_rollout.py \
  --config causal_video_prediction_libero_latent_local \
  --checkpoint /runs/wan22_video_only/checkpoints/checkpoint_step_3000 \
  --reference-assets-root /models/lingbot-va-base \
  --data-root /datasets/libero_subsets \
  --empty-text-embedding /datasets/empty_emb.pt \
  --split val \
  --sample-index 0 \
  --seed 1234 \
  --output-dir outputs/video_only_rollout
```

`--num-chunks 1` evaluates the adapter-produced target. Larger values append
equal-width open-loop predictions by re-feeding the complete generated latent
history; only the first chunk has a dataset target. Multi-view latents are
decoded independently and reassembled from the canonical layout contract.
The required `--checkpoint` selects transformer weights even when the source
experiment config disabled initialization-time reference loading. `--config`
remains the sole runtime and data contract after explicit CLI overrides.
Adjacent checkpoint metadata is not merged implicitly.
To reproduce a checkpoint's recorded behavior, pass its `resolved_config.yaml`
explicitly with `--config`. Detached transformer exports therefore follow the
same rule and do not require an adjacent config.
When passing a checkpoint step, that step must contain a usable local
`transformer/` export. A state file by itself is not sufficient for this
decoded-rollout utility. `backbone.transformer_subdir` normally names a
component beneath `pretrained_model_name_or_path`; absolute values remain valid
for historical or deliberately authored configs. Prefer
`runtime_backbone_artifact_path` for new detached-artifact configs.

The canonical preset uses guidance `5.0` with task-prompt dropout `0.1`.
Guidance above `1.0` is rejected unless the resolved training config records
dropout strictly between `0` and `1` and runtime positive and negative text
embeddings are finite and shape-matched. Positive text dropout requires
`trainer.batch_adapter=latents`, where dropout operates on an explicit encoded
text context. The executor retains the positive source tensor for validation
while supplying the selected positive or blank context through the same
visual-stack input used by every policy. View batches are rejected instead of
silently training without the requested unconditional branch. Guidance is
intentionally unavailable in `disabled` mode because its two text branches are
identical.

Outputs are first written to a temporary sibling and then atomically published.
The output identity includes the resolved config, checkpoint/reference file
inventory, sample metadata, latent/text tensor digests, and inference controls.
It uses standard-cost artifact provenance: files larger than 1 MiB contribute
path, size, and modification time rather than a content digest. Source and
environment provenance are recorded in `summary.json`, but are not part of the
output-directory suffix. Therefore, that suffix is deterministic bookkeeping,
not a content-addressed model or result ID. Use a new output root when code or
weights may have changed in place, and record independent full artifact digests
for publication results. Output directories are create-only and are published
atomically after all files are complete.

## Extend The Workflow

New datasets should implement the uniform latent data contract and produce the
same canonical layout metadata; camera-name parsing stays in the data adapter.
New video objectives should remain a `PolicyVariant` plus compatible
`ActionDecoder`, while shared transformer execution stays in `VisualTower`.
Use topology component groups for trainability and exports rather than matching
parameter-name prefixes.

The permanent strict-gradient and multi-chunk gates are documented in
[Testing](testing.md#video-only-parity).
