# Testing

Run tests from a source checkout. Pytest markers identify checks that need a
GPU, dataset, or simulator.

## CPU Tests

```bash
uv sync --frozen --group dev --extra full
CUDA_VISIBLE_DEVICES="" uv run pytest --strict-markers -q \
  -m "not (gpu or sim or data or slow)"
```

This is the standard semantic suite, including unmarked tests. It checks
configuration, data contracts, training, inference, gradients, and checkpoint
behavior using small fixtures. It does not measure full-model benchmark quality.

For a shorter end-to-end check, follow the
[CPU first run](quickstart.md#complete-cpu-first-run).

## Resource Requirements

| Marker | Requirement |
| --- | --- |
| `unit` | CPU; no external data or simulator |
| `smoke` | Short integration check; inspect additional resource markers |
| `integration` | Cross-component check |
| `gpu` | Allocated CUDA device and the test's explicit opt-in |
| `sim` | Configured external simulator |
| `data` | Non-fixture local dataset |
| `slow` | Longer-running training or rollout |

Configure local assets through `configs/local_paths.yaml` or
`OPEN_WAM_LOCAL_PATHS`. Resource-gated tests explain missing prerequisites
rather than silently substituting synthetic inputs.

```bash
OPEN_WAM_RUN_GPU_SANITY=1 uv run pytest -m gpu
uv run pytest -m sim
```

Individual GPU suites can require additional opt-ins and artifact manifests.
Run only the suites appropriate to your allocated hardware.

## Numerical Regression

For changes to attention, conditioning, caches, training, or inference, compare
against immutable references made with the same configs, checkpoint, frozen
inputs, and hardware/software stack. Keep the reference from before the change;
do not regenerate a golden to make a failing comparison pass.

The real-checkpoint runner accepts a local manifest based on
`tests/characterization/dual_expert_assets.example.yaml`. It can capture input
fixtures, training steps, recurrent inference, cache rollover, full-state
restore, and simulator rollouts. Inspect the available phases with:

```bash
uv run python -m tests.characterization.run_dual_expert_refactor_characterization --help
```

Given an existing fixture set and reviewed reference reports, record and compare
one checkpoint:

```bash
uv run python -m tests.characterization.run_dual_expert_refactor_characterization \
  record \
  --assets /path/to/assets.yaml \
  --fixture-root /path/to/frozen_fixtures \
  --output-root /path/to/new_reports \
  --stage-root /path/to/local_scratch \
  --asset-id dual_expert_joint \
  --cuda-devices 0,1,2,3

uv run python -m tests.characterization.run_dual_expert_refactor_characterization \
  verify \
  --actual-root /path/to/new_reports \
  --golden-root /path/to/reference_reports \
  --asset-id dual_expert_joint
```

The reports check input identity, shapes, dtypes, losses, predictions,
gradients, optimizer updates, and recurrent state. Comparisons use exact tensor
hashes where deterministic and explicit tolerances for supported
floating-point reductions. Equal seeds alone do not establish cross-device
bitwise parity. See [Compatibility](compatibility.md).

Checkpoint-backed coverage depends on the supplied assets. Passing synthetic
or CPU tests does not establish parity for an unavailable model or simulator.

## Video-Only Parity

Video-only training has frozen CPU tests for losses, predictions, every named
gradient, and parameter updates. Companion inference tests check clean-prefix
preservation and generated-history progression. Pretraining tests also cover
RGB processing, latent layouts, text conditioning, mixed datasets, and batching.

See [Pretraining Validation](pretraining/validation.md) for data checks before
scaling up a run.

## Documentation And Static Checks

```bash
uv run python -m pyflakes src scripts tests baselines
OPEN_WAM_CI_NO_TORCH=1 uv run python scripts/ci_basic_sanity.py
uv run --extra docs python scripts/build_docs_site.py --output .docs_site
uv run --extra docs mkdocs build --strict
```

To preview documentation locally, use `uv run --extra docs mkdocs serve` after
building `.docs_site`.
