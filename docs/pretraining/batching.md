# Batching Video Pretraining

This recipe uses the package's [shared latent batching](../variable_length_training_batches.md).
It does not introduce a second collator, trainer, or policy-mode allowlist.

## Transport And Execution

- `strict`: unchanged direct stacking and single-sequence execution.
- `padded`: pad temporal transport tensors and mask non-existent frames.
- `bucket`: reorder bounded pools by shape and length, then use padded execution.
- `packed`: transport padded tensors, gather valid tokens into isolated sequences
  for the heavy layers, then restore the batch layout.

For variable H/W, set `data.shape_bucketed_batching: true`. The mixed-video
planner forms spatially compatible rank-local microbatches from the weighted
draws. The shared length sorter preserves these draws and their multiplicities;
it does not open videos to obtain shape hints. Completing distributed spatial
batches can repeat tail samples. Measure actual exposure when comparing mixtures.

With shape bucketing disabled, `bucket` groups by length only and does not
require manifest height/width. Samples within a batch must still share spatial
dimensions; batch size one also works without spatial metadata.

The causal-video `prefix_suffix` program supports these modes. Its
`chunked_conditioned` program remains on strict batching. Temporal patch size
must be one for packed execution; optional tensors must be consistently present
throughout a microbatch.

Each sample retains its own prefix/future lengths, text, temporal positions,
masks, and geometry. No sample can attend to another sample's video or text.
The shared batching transport does not replace chunk/window geometry with the
first sample's geometry. Policy training, including GJD and strict dynamics,
continues to use its own declared batch-execution capability and semantics.

## Loss And Randomness

Video pretraining retains its original reduction: total squared error over valid
future elements divided by their count across the microbatch. It does not switch
to an equally weighted mean of per-video losses. The video decoder owns this
reduction. Policy decoders keep their existing per-sample reductions.

Batched noise draws retain the collated temporal capacity, including explicit
`pad_to_multiple_of` padding. This preserves the frozen seeded loss, gradient,
and optimizer-step reference. Padding never contributes supervision. Comparing
a batched run to independent B=1 runs still requires matching random tensors;
setting the same seed alone does not guarantee the same RNG consumption.

## Configuration

The generator defaults to 36 samples per rank, shape grouping, length bucket
pools of 1152, and dropped incomplete training tails. These are recipe choices,
not a hardware capacity promise. Start with a small batch on a new GPU.

```yaml
data:
  train_batch_size: 2
  val_batch_size: 1
  shape_bucketed_batching: true
  batching:
    mode: bucket
    bucket_pool_size: 128
    pad_to_multiple_of: 1
    drop_last_train: true
```

The latent adapter, variant's `forward_train_batch`, shared transformer, and
video decoder remain separate owners. Packed execution calls each heavy layer
once and uses shared sample-isolation predicates for both video and text.
No learned parameters or checkpoint keys are added.

For validation, dataset identity/splits and future masks remain unchanged.
Validation batch size stays one in this recipe. See [validation](validation.md)
for exact numerical gates and the limits of small-model checks.
