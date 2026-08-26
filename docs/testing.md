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

## Public CI Static Tier

```bash
OPEN_WAM_CI_NO_TORCH=1 python scripts/ci_basic_sanity.py
```

The dependency-light jobs remain intentionally cheap. They do not import
Torch or touch private checkpoints, local datasets, GPUs, or external
simulator checkouts.

Tier 0 checks package metadata, entrypoint declarations, public config
references, artifact manifest shape, local path sample hygiene, duplicate
optional dependencies, source contracts that should remain import-safe, and
production-core Pyflakes over `src`, `scripts`, `tests`, and `baselines`.
Pyflakes runs in an isolated `uv --no-project` environment; it
does not install Open-WAM or its runtime dependencies. The separate
`deployment` hardware/ROS workspace is not part of the installed package or
this model-library gate. Its own dependency-light CI job compiles the
workspace, lints the supported hardware runtime, and runs no-hardware tests.

Run that static lint locally with the pinned development dependency:

```bash
uv run python -m pyflakes src scripts tests baselines
```

The default PR workflow also includes three dependency-light companion jobs:

- `minimal-package`: installs only the minimal package and verifies import-safe
  package surfaces plus CLI parser construction without Torch.
- `docs-site`: installs only MkDocs, stages curated public docs, asserts Torch
  is unavailable, and builds the static GitHub Pages site.
- `hardware-workspace-sanity`: installs NumPy, PyYAML, Pytest, and Pyflakes;
  compiles `deployment/`; and verifies the supported launcher/library contract
  without Torch, ROS2, cameras, or an FR3.

## Required CPU Semantic Gate

Every pull request runs the complete CPU-safe suite on Python 3.11 and 3.12:

```bash
uv sync --frozen --group dev --extra full
uv run pytest --strict-markers -q -m "not (gpu or sim or data or slow)"
```

This expression deliberately includes unmarked tests. A missing marker cannot
silently remove a test from the standard gate. Tests marked `gpu`, `sim`,
`data`, or `slow` must have a separate documented gate and skip with an
actionable resource message when run without that resource.

## Manual CPU Smoke Workflow

`.github/workflows/cpu-smoke.yml` remains a manually runnable end-to-end CLI
check for the public tiny fixture. Its underlying contracts are also covered
by the required CPU semantic suite.

## Local Full Checks

Run GPU tests only when a GPU is intentionally allocated:

```bash
OPEN_WAM_RUN_GPU_SANITY=1 uv run pytest -m gpu
```

Core DualExpert/GJD refactors have a stricter, separately gated real-checkpoint
workflow. See
[Dual-Expert Refactor Characterization](dual_expert_refactor_characterization.md)
for its six non-GJD programs, five GJD ablations, six available exact-checkpoint
slots, frozen real-data replay, FSDP update checks, stateful inference, and
record-versus-verify commands.

Run simulator tests only after configuring `configs/local_paths.yaml` or
`OPEN_WAM_LOCAL_PATHS`:

```bash
uv run pytest -m sim
```

Tests that require real datasets or simulator roots should skip with an
actionable message when the resource is missing.

## Video-Only Parity

The CPU semantic suite permanently fixes the causal-video training contract at
the gradient level. A fixed tiny shared transformer, latent/text batch,
diffusion seed, and SGD update must preserve the exact trainable-name set and
parameter inventory. It compares the loss, predicted latents, every named
gradient, and every named updated parameter elementwise against an immutable
`safetensors` golden. Names, shapes, and dtypes must match exactly; numerical
values use a small tolerance because supported CPU kernels are not
byte-identical across hosts. The companion inference test executes the real
causal policy over multiple chunks while also verifying clean-prefix
preservation and complete generated-history progression.

Additional focused tests require exact converter tensor provenance, strict
template model-schema coverage, topology-scoped export ownership, complete
prefix timestep-zero conditioning, exact cross-view latent metadata and
geometry, explicit text-dropout source preservation, typed policy-to-decoder video-flow
artifacts, strict CFG requirements, and create-only atomic publication.
Real-checkpoint GPU runs remain the resource-gated confirmation that
dtype/device integration matches these CPU contracts.
