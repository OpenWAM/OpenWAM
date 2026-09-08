# Variable-Length Training Batches

Length bucketing, dynamic padding, and sequence packing allow supported latent
training recipes to use multiple trajectories per microbatch without requiring
every trajectory to have the same length. They are opt-in: `data.batching.mode`
defaults to `strict`, which preserves the existing collation and execution path.

This guide describes semantic guarantees and configuration. It does not establish
a throughput or memory improvement for a particular GPU, model, or dataset.

## Choose A Mode

| Mode | Sample grouping | Transformer execution |
| --- | --- | --- |
| `strict` | Existing sampler and direct tensor stacking | Existing path; incompatible tensor shapes still fail |
| `padded` | Existing sampler order, with dynamic padding within each microbatch | Keeps maximum-length video and action slots for every sample; masks padding |
| `bucket` | Reorders bounded pools of sampled indices by a length hint, then pads each microbatch | Same execution as `padded`, typically with fewer unused slots |
| `packed` | Existing sampler order, with padded tensors for transport | Removes batch-added padding before the transformer and combines real tokens |

`bucket` groups similar lengths; it does not require exact-length matches. The
length hint comes from dataset index metadata, without loading full samples to
sort them. For randomized segment lengths it is an estimate, not a promise about
the materialized length. Bucketing preserves the retained sampled indices and
their multiplicities, including repeated replacement draws.

`bucket_pool_size` defaults to 128. The effective pool size is rounded down to a
multiple of the microbatch size and is at least one microbatch. If training drops
an incomplete tail, that tail is removed from the original sampler stream before
sorting; samples are not discarded separately from each bucket.

## What Remains Independent

All three new modes retain each sample's true tensor lengths, metadata,
supervision boundaries, and action masks. Predictions and losses are computed
only over the original sample extent. Padding introduced by collation cannot
contribute a supervised target.

The combined transformer execution isolates samples in both self-attention and
cross-attention. A sample cannot attend to another sample's video, action, text,
or proprioceptive context. Temporal positions and the policy's within-sample
visibility rules retain their original meaning. Packing is therefore not the
same as concatenating trajectories into one longer episode.

Each sample uses its existing decoder and loss normalization. The microbatch
loss is the arithmetic mean of those per-sample losses, not a mean across all
tokens in the combined batch. A longer trajectory does not receive extra weight
merely because it contains more tokens. Existing masks still determine which
action dimensions and timesteps are supervised within each sample.

Every sample retains its own chunk/window geometry, frame positions, prefix,
and visibility rule. The combined mask applies that sample's prepared attention
predicate in its original local token coordinates. Nothing is inferred from the
first sample, and batching does not synchronize or resample geometry.

Batching does not change sequence contracts, action coordinates, conditioning
rules, or loss boundaries. In particular, `history_stream_visibility: full`
controls which historical streams are visible; it does not mean unbounded
history or automatically exclude history from supervision. Preserve the intended
policy recipe rather than changing its sequence contract to enable batching.

## Supported Configurations

Non-`strict` modes currently require:

- `trainer.batch_adapter: latents`. Any registered dataset adapter that produces
  `LatentWAMSample` can use the transport, including routed real/CF sources.
- An architecture implementing `PolicyVariant.forward_train_batch`. DualExpert
  implements this once for all six policy programs, GJD (with or without mode
  tokens), and standalone forward/inverse dynamics.
- `training.sample_loss_weight_mode: none`.

The typed experiment validator rejects unsupported execution modes, adapters,
and loss weighting before model allocation. Other configurations do not silently
fall back to independent full-model forwards.

Within a microbatch, video samples have shape `[C, T, H, W]` with `T >= 2`.
Channels, spatial dimensions, and action dimensions must agree. Each sample must
satisfy its policy's action/frame alignment contract. Optional tensor fields
can be absent in individual samples: transport records absence separately from
an empty tensor and restores it before policy preparation. If negative text context is
provided, positive text context must also be provided with the same shape within
that sample; text lengths may differ between samples.

The sample count per training microbatch remains fixed on every rank. This
implementation does not use a variable-size, token-budget batch sampler.
`drop_last_train: true` is the default for new modes. If it is disabled, the
training sampler length must be divisible by the microbatch size. Validation
retains an incomplete final batch and aggregates its metrics by original sample
count, so a one-sample tail does not receive the weight of a full microbatch.
The `strict` validation path is unchanged.

## Component Boundaries

The data layer owns grouping and transport: it records original tensor extents
and pads without changing sampled geometry or supervision. The shared latent
transport also accepts empty action tensors for action-free consumers; the
supported DualExpert recipe validates its positive, frame-aligned actions in
its own sequence preparation.

`VariantPipeline` restores each sample, runs its existing preparation and decoder,
and averages per-sample losses. `PolicyVariant.forward_train_batch` is the explicit
execution hook: return one `PolicyTrainOutput` per prepared sample, in order.
The default raises an unsupported-operation error. There is no serial full-model
fallback disguised as batching.

To add another consumer:

1. Implement that hook and declare the supported execution modes through the
   policy config's `supported_batching_modes` property. `BatchingMode.execution_mode`
   separates execution from grouping: `bucket` resolves to padded execution.
2. Keep model semantics in the consumer. Reuse prepared `PackedTokenLayout`s with
   `models.common.sequence_batch_attention` for stream-major layout composition
   and sample isolation. The prepared profile supplies `self_attention_visibility`
   as a tensor predicate over local query/key indices. Reuse the existing rule; do not
   introduce another implementation of the method's attention semantics.
3. Keep transformer topology and parameter ownership in the architecture. The
   DualExpert implementation shares visual preparation and final projection
   with its single-sequence executor, but still owns its two-expert block stack.
4. Test against independent execution using identical prepared sample contracts,
   including gradients, padding, text/proprio isolation, and distributed execution.

Adding a new policy program to the same architecture does not require a batching
allowlist entry. Its existing sample preparation, attention rule, and decoder
remain authoritative. Another transformer topology (for example, video-only or
Parallel Stream) needs its own heavy-layer execution hook and parity gate, but
not a different data transport or batching semantics. This is a training feature;
inference programs and experiment defaults are unchanged.

## Conditional And Mixed Objectives

IDM/FDM samples still use their target-only t0 singleton, limited video history,
text removal, and original loss masks. Joint samples retain their planning
history, language, and supervision. The batch layer interprets neither objective:
it combines the already-prepared sample contracts. Optional condition latents can
therefore be present for a joint sample and absent for a conditional peer.

Use `padded` or `packed` with dynamics routing. The routed sampler coordinates
objectives at each sample position across ranks; batching preserves that order,
including mixed objectives within a microbatch. `bucket` is rejected for samplers
that declare ordering constraints, since length sorting would break this
coordination. Source ratios and replacement sampling remain owned by routing.

## Configure An Experiment

Apply these fields to an existing, validated local latent-data experiment. This
is a partial override, not a standalone dataset or model configuration:

```yaml
data:
  train_batch_size: 2
  val_batch_size: 1
  batching:
    mode: bucket          # Alternatives: padded, packed; default: strict
    bucket_pool_size: 128
    pad_to_multiple_of: 1 # Latent-frame alignment for transport tensors
    drop_last_train: true
training:
  gradient_accumulation_steps: 3
  sample_loss_weight_mode: none
```

`pad_to_multiple_of` controls transport tensor alignment; it is not a guarantee
of kernel acceleration. Even `packed` uses padded transport tensors. It removes
batch-added padding at the transformer boundary, while retaining any conditioning
tokens required by the original sequence contract. Preprocessing and decoding
remain per sample; the heavy transformer layers execute the combined token batch
once per layer.

The effective batch size remains:

```text
train_batch_size × gradient_accumulation_steps × WORLD_SIZE
```

For example, on four workers, changing microbatch/accumulation from `1 × 6` to
`2 × 3` preserves an effective batch of 24. This preserves the number of sampled
trajectories per optimizer update, not the exact sampling order, random draws,
or computational cost. A larger microbatch may still exceed device memory.

Use the standard training entrypoint and an independent output directory:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m open_wam.cli.train \
  --cfg /path/to/local_latent_experiment.yaml \
  --save-root runs/latent-bucket-b2 \
  --expected-world-size 4 \
  --set data.batching.mode=bucket \
  --set data.batching.bucket_pool_size=128 \
  --set data.batching.pad_to_multiple_of=1 \
  --set data.batching.drop_last_train=true \
  --set data.train_batch_size=2 \
  --set data.val_batch_size=1 \
  --set training.gradient_accumulation_steps=3 \
  --set training.sample_loss_weight_mode=none
```

Keep the dataset, resolution, sequence semantics, optimizer settings, and
checkpoint frequency fixed for an initial comparison. Record the resolved
configuration, peak device memory, optimizer-step time, effective token
throughput, and losses. Reduced padding alone is not a measured speedup.

## Resume And Validation Comparisons

Bucketing forwards `set_epoch` to its underlying sampler and adds no random
draws of its own. Deterministic replay assumes the same sampler configuration,
mode, pool size, microbatch size, and dataset index. Changing any of these can
change grouping or draw order, so a changed batching recipe is a new experiment,
not a step-for-step continuation claim. Prefer switching recipes at an epoch
boundary and record the parent checkpoint and overrides.

Use `--resume-from` to restore training state, including optimizer and scheduler
state, or `--initialize-weights-from` to initialize only model weights. These
options are mutually exclusive. See [Training and Inference](running_experiments.md)
for the complete checkpoint lifecycle.

Bucketing also reorders validation batches. If validation is limited to a prefix
of batches, changing the mode or pool size may change which samples are evaluated.
Use the complete validation set or an explicitly fixed sample subset when
comparing modes. Validation noise and timestep sampling still follow the selected
policy's existing behavior; batching does not make validation deterministic.

## Verification And Resource Limits

After installing the development and runtime dependencies described in
[Testing](testing.md), run the focused CPU checks:

```bash
uv run pytest -q tests/test_latent_batching.py \
  tests/test_variable_batch_pipeline.py tests/test_batching_invariants.py \
  tests/test_sequence_batch_contract.py tests/test_sequence_batch_dynamics.py -m "not gpu"
uv run pytest -q tests/test_dual_expert_sequence_batches.py -m "not gpu"
```

These small, randomly initialized fixtures require neither model downloads nor
real datasets. Coverage includes per-sample reference comparisons for predictions,
losses, and gradients; immunity to transport padding; cross-sample isolation;
activation checkpointing; and one combined call per heavy transformer layer.

CPU execution uses small dense masks. CUDA execution requires FlexAttention and
uses sequence-aware block masks. On an intentionally allocated CUDA device,
enable the separate kernel check:

```bash
OPEN_WAM_RUN_GPU_SANITY=1 uv run pytest -q \
  tests/test_dual_expert_sequence_batches.py tests/test_sequence_batch_dynamics.py -m gpu
```

Confirm that CUDA tests ran rather than skipped. The kernel gate checks FP32 for
all six policy programs, retains the original VTA/Joint BF16 thresholds, and
checks self/cross-attention, backward gradients, and sample isolation. The
full-pipeline gate covers all programs in FP32 and BF16. It
does not promise bitwise agreement between differently shaped BF16 GEMMs. The
end-to-end gate additionally compares against FP32: batched BF16
maximum and RMS errors must stay within twice the independently measured B1 BF16
error plus the FP32 comparison tolerance, for outputs and gradients. This does
not relax the existing kernel parity thresholds or strict-mode baseline. It
does not replace a representative multi-GPU FSDP run with the intended precision,
activation checkpointing, and real sequence lengths.

A separate two-rank smoke uses a small transformer with unequal 9-17-frame
sequences at latent resolution 8x16, four actions per frame, and independently
varying chunks/windows. It covers six policy programs, mixed GJD, and strict
IDM/FDM, performing two BF16 FSDP optimizer updates with activation checkpointing:

```bash
OPEN_WAM_RUN_GPU_SANITY=1 torchrun --standalone --nproc-per-node=2 \
  -m pytest -q tests/test_sequence_batch_fsdp.py -m "not slow"
```

This checks the distributed training path, not pretrained-model quality or
full-model memory headroom. Allocate the CUDA devices explicitly before running.
The same tests retain 33-69-frame cases marked `slow`; omit `-m "not slow"` to
include those. Large sparse-index layouts can have substantial cold compilation
cost, so the shorter semantic gate is not a substitute for this scaling check.

The token-count comparison measures `B × max(video_tokens) + B ×
max(action_tokens)` for padded execution versus `sum(video_tokens) +
sum(action_tokens)` for packed execution. It is not a GPU benchmark, and its
ratio is not an estimate of total memory savings or wall-clock acceleration.
Validate compilation cost, memory headroom, distributed synchronization, and
throughput before adopting a new batching mode for a long run.
