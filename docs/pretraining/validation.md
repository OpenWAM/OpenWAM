# Validate A Pretraining Run

Start with a small pilot before encoding a full corpus or launching distributed
training. Validate the data and the model separately: a readable tensor is not
evidence that its camera order, color, text, or timeline is correct.

## Data Checks

1. Pin the upstream revision and retain original episode IDs and checksums.
2. Inspect native task text, camera roles, RGB orientation, frame counts, and FPS.
3. Encode representative single-view and multiview clips with the intended VAE.
   Compare the source RGB and decoded reconstructions.
4. Admit only complete, validated clips. Group all views and segments from one
   physical episode into the same train/validation split.
5. Build a frozen snapshot and a text cache covering its prompts. For remote
   storage, verify that the catalog can restore metadata and read actual objects.

Keep encoding receipts and outputs immutable. If an encoding contract or
implementation fingerprint changes, write to a new output directory rather
than relabeling existing tensors.

## Model Checks

1. Run short training with the intended precision, strategy, batch size, and
   source mixture. Check finite losses and gradients.
2. Save and resume a full training-state checkpoint.
3. Generate prediction videos with fixed samples, seeds, and context/future
   lengths. Inspect each source and camera layout, not only aggregate loss.
4. Measure memory and throughput before increasing corpus size or GPU count.

Synthetic tests do not establish full-size model quality or GPU memory capacity.
See [Training](training.md) for commands and [Batching](batching.md) for the
supported execution modes.

## Automated Tests

From a source checkout with the training and preparation dependencies installed:

```bash
pytest -q tests/test_pretraining_*.py tests/test_offline_prompt_cache.py \
  tests/test_causal_video_batching.py tests/test_mixed_video_physical_split.py \
  tests/test_mixed_video_latent_storage.py
```

These tests cover RGB processing, latent layouts, text lookup, episode splits,
storage integrity, batched predictions, losses, and gradients. For changes to
the implementation, also follow the [numerical regression guide](../testing.md#numerical-regression).
