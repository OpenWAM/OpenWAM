# Validation And Acceptance

The preparation workflow and causal-video batching path are tested separately
from the existing action-policy modes. Small synthetic fixtures make these
checks runnable without downloading the full corpus.

## Automated Gates

Run the preparation and frozen numerical tests with the training and preparation
dependencies installed:

```bash
pytest -q tests/test_pretraining_*.py tests/test_offline_prompt_cache.py \
  tests/test_causal_video_batching.py tests/test_mixed_video_physical_split.py \
  tests/test_mixed_video_latent_storage.py
```

The frozen pre-refactor reference covers padded, bucket, and packed batches,
unequal sequence lengths, and padding alignment of one and four frames. It
records exact losses, predictions, targets, masks, timesteps, every trainable
parameter gradient, an AdamW update, and RNG state. RGB goldens separately cover
resampling, resize/crop, two/three-view multi view compositions, VAE input scaling and output
normalization. These goldens are not regenerated from the implementation under
test. Portable CI compares computed floating outputs using the repository's
`rtol=1e-5, atol=2e-6` convention for cross-CPU kernel rounding. Tensor inventories,
shapes, dtypes, inputs, targets, masks, timesteps, and RNG state remain exact.
RGB preprocessing remains bit-exact.

For a refactor, also compare both revisions in the **same environment**, with
zero tolerance for every recorded tensor. Use the candidate's capture script
with each revision's package and test helpers:

```bash
# REFERENCE_ROOT and CANDIDATE_ROOT are checkouts of the two revisions.
ARTIFACTS=$(mktemp -d)
CAPTURE="$CANDIDATE_ROOT/tests/characterization/pretraining_batch.py"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  PYTHONPATH="$REFERENCE_ROOT/src:$REFERENCE_ROOT" \
  python "$CAPTURE" --out "$ARTIFACTS/reference"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  PYTHONPATH="$CANDIDATE_ROOT/src:$CANDIDATE_ROOT" \
  python "$CAPTURE" --compare "$ARTIFACTS/reference"
```

Use the same interpreter, dependencies, and CPU backend for both commands.
The comparison never rewrites references; recording requires a new directory.

Contract tests cover native task overrides, complete-episode admission,
float32 timestamp boundaries, RGB interpretation, physical-episode splits,
additive snapshots with arbitrary source names, offline prompt lookup and dropout,
exact encoder/runtime token-limit agreement, object checksums,
publication/restore, read leases, and independent configured catalogs.

The workflow regression also loads a published THWC snapshot through the real
dataset, collator, prompt cache, and small transformer, then runs backward and
latent prediction in all four batching modes. This closes the boundary that
prepared tensor goldens alone cannot check. Separate reader tests preserve CTHW values
exactly and reject unknown layout declarations; bucketing tests cover disabled
spatial hints and partial validation batches.

The standard repository semantic gate also protects policy training and
inference. CUDA checks compare dense/packed execution, gradients, and sample
isolation for the policy programs. These tests are not evaluations of a
full-size pretrained model's closed-loop success.

## Deliberate Corrections

The initial proposal's broken prompt-encoder import, late reviewed-text
overrides, rounded boundary-frame omission, and truncated-episode admission
are rejected by regression tests, not preserved as goldens.

The refactor preserves the RGB kernels and VAE normalization. Implementation
fingerprints do change when their source modules move. Keep old receipts and
outputs immutable; start a new encoding output directory when continuing with a
different implementation fingerprint. Do not relabel old artifacts to bypass
the contract check.

## Acceptance On A Training Host

1. Pin upstream revisions and inspect representative native text, camera order,
   RGB orientation, frame counts, and FPS.
2. Encode singles and multi view compositions with the actual VAE; inspect RGB and decoded
   reconstructions. The lightweight VAE probe checks transport, not model weights.
3. Publish a pilot snapshot and matching text cache. If using object storage,
   seal, restore, and read real objects from the target endpoint.
4. Run short training and full-state resume with the intended precision,
   distributed strategy, microbatch size, and source mixture.
5. Generate prediction videos with fixed samples, seeds, and prefix/future
   lengths. Only then scale up the corpus and device count.

No throughput, memory-capacity, or model-quality improvement is asserted by the
structural reorganization.
