# Testing

Open-WAM uses pytest markers to make resource requirements explicit.

## Markers

- `unit`: CPU-only unit tests with no local data, simulator, or GPU requirement.
- `smoke`: short CPU-safe integration tests for public command/config surfaces.
- `gpu`: requires CUDA and an explicit local resource gate.
- `sim`: requires an external simulator such as LIBERO, RoboTwin, or CALVIN.
- `data`: requires non-fixture local datasets.
- `slow`: long-running train/eval/rollout checks.
- `integration`: cross-component tests that are larger than unit tests.

## Public CI Tier 0

```bash
OPEN_WAM_CI_NO_TORCH=1 python scripts/ci_basic_sanity.py
```

The default GitHub PR tier is static and intentionally cheap. It must not run
`uv sync`, install the project, run pytest, import `open_wam`, install Torch, or
touch private checkpoints, local datasets, GPUs, or external simulator
checkouts.

Tier 0 checks package metadata, entrypoint declarations, public config
references, artifact manifest shape, local path sample hygiene, duplicate
optional dependencies, and source contracts that should remain import-safe.

The default PR workflow also includes two dependency-light companion jobs:

- `minimal-package`: installs only the minimal package and verifies import-safe
  package surfaces plus CLI parser construction without Torch.
- `docs-site`: installs only MkDocs, stages curated public docs, asserts Torch
  is unavailable, and builds the static GitHub Pages site.

## Local CPU Pytest Tier

After installing the development environment, run the CPU-safe pytest marker
set locally or in a future gated CI tier:

```bash
uv run --extra train pytest -m "unit or smoke or integration"
```

This tier must still not require CUDA, private checkpoints, local datasets, or
external simulator checkouts.

## Manual CPU Smoke Workflow

`.github/workflows/cpu-smoke.yml` is manual-only. It installs the Torch-backed
train/eval stack and runs the public tiny synthetic eval path. Keep it out of
default pull-request CI unless runtime and dependency cost are intentionally
accepted.

## Local Full Checks

Run GPU tests only when a GPU is intentionally allocated:

```bash
OPEN_WAM_RUN_GPU_SANITY=1 uv run pytest -m gpu
```

Run simulator tests only after configuring `configs/local_paths.yaml` or
`OPEN_WAM_LOCAL_PATHS`:

```bash
uv run pytest -m sim
```

Tests that require real datasets or simulator roots should skip with an
actionable message when the resource is missing.
