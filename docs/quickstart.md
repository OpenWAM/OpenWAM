# Open-WAM Quickstart

This guide is the public first-run path. It does not require private datasets,
private checkpoints, CUDA, or external simulators.

## Install

```bash
uv sync --group dev
```

The base install is intentionally minimal. It supports imports, config/static
validation, artifact metadata, and dependency-light CLI parser surfaces without
Torch or simulator packages.

For Torch-backed local train/eval smoke paths, install the relevant extra:

```bash
uv sync --group dev --extra train
uv sync --group dev --extra eval
```

For optional simulator work, install only the extras you need:

```bash
uv sync --extra libero
uv sync --extra calvin
uv sync --extra robotwin
uv sync --extra sim
```

## CPU Smoke

Run one no-Torch static validation path:

```bash
uv run open-wam-validate-config \
  configs/examples/public_tiny_synthetic_contract.yaml \
  configs/evals/public_tiny_synthetic_contract.yaml
```

After installing `--extra eval`, run one CPU-safe eval path:

```bash
uv run --extra eval open-wam-eval \
  --cfg configs/experiments/contract_only_robotwin.yaml \
  --max-batches 1 \
  --device cpu
```

Inspect a config without launching training:

```bash
uv run open-wam-inspect-config \
  --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml
```

Run a benchmark pipeline sanity check:

```bash
uv run --extra train open-wam-sanity \
  --cfg configs/examples/robotwin_lerobot_video_sparse30_sanity.yaml \
  --device cpu \
  --max-batches 1 \
  --rollout-steps 1
```

## Local Paths

Real datasets, checkpoints, simulator checkouts, and run roots are machine
local. Do not edit public experiment YAMLs to hard-code those paths.

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
```

Then replace every `/path/to/...` placeholder in `configs/local_paths.yaml`.
The file is gitignored. You can also use:

```bash
OPEN_WAM_LOCAL_PATHS=/absolute/path/to/local_paths.yaml uv run open-wam-eval ...
```

## Stable Commands

Preferred package commands:

- `open-wam-train`
- `open-wam-eval`
- `open-wam-inspect-config`
- `open-wam-validate-config`
- `open-wam-sanity`
- `open-wam-sim-rollout`

Legacy `python scripts/...` commands remain supported as compatibility
entrypoints while the runtime is migrated into package modules.

## Resource Matrix

| Command family | CPU | GPU | Local data | Simulator | Private checkpoint |
| --- | --- | --- | --- | --- | --- |
| config inspect | required | no | no | no | no |
| synthetic eval smoke | required | no | no | no | no |
| real dataset train/eval | required | optional | yes | no | optional |
| LIBERO/RoboTwin/CALVIN rollout | required | optional | optional | yes | optional |
| full realtime method comparison | required | usually | yes | yes | yes |

Use pytest markers and local path aliases to make those requirements explicit.
