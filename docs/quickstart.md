# OpenWAM Quickstart

This guide is the public first-run path. It does not require private datasets,
private checkpoints, CUDA, or external simulators.

OpenWAM supports Linux with Python 3.11 or 3.12.

## Install

### Installed SDK

Install from [PyPI](https://pypi.org/project/openwam/) in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'openwam[train,eval]'
openwam-sanity \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --device cpu --max-batches 1 --rollout-steps 1
```

The distribution is named `openwam`; Python code uses `import open_wam`.
The optional `openwam-sdk` installation alias provides the same implementation
and extras. To pin a version, use `openwam[train,eval]==0.1.1`; see
[Releases](release.md) for reproducibility guidance.

The base `pip install openwam` needs only PyYAML and supports config/metadata
APIs and CLI help. Add `[train,pretrain]` for video data preparation and
pretraining, or `[eval]` for offline model evaluation. Weights, datasets, and
external simulator source trees remain separately provisioned.

All `openwam-*` commands below also work in an installed environment: omit
their `uv run` / `uv run --extra ...` prefix after installing the relevant
extras. Packaged config references do not require cloning the repository.
Use the frozen checkout path for exact dependency reproduction; an ordinary
PyPI install resolves the declared dependency ranges, not `uv.lock`.

### Source Checkout

Install `uv` and run from a clone of the repository:

```bash
git clone https://github.com/OpenWAM/OpenWAM.git
cd OpenWAM
uv sync --frozen --group dev --extra train --extra eval
```

This installs the CPU-capable development, training, and evaluation stack used
by the complete first run below. The smaller base install is available with
`uv sync --group dev`; it supports imports, config validation, artifact
metadata, and CLI parser surfaces without Torch or simulator packages.

## Complete CPU First Run

Validate the public synthetic train and evaluation configs:

```bash
uv run openwam-validate-config \
  configs/examples/public_tiny_synthetic_contract.yaml \
  configs/evals/public_tiny_synthetic_contract.yaml
```

Train one step and write a full-state checkpoint:

```bash
RUN_ROOT="runs/public-tiny-$(date +%Y%m%d-%H%M%S)"
uv run --extra train openwam-train \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --save-root "$RUN_ROOT" \
  --expected-world-size 1 \
  --disable-wandb
```

Resume full training state from step 1 and train through step 2:

```bash
uv run --extra train openwam-train \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --save-root "$RUN_ROOT" \
  --resume-from "$RUN_ROOT/checkpoints/checkpoint_step_1" \
  --num-steps 2 \
  --expected-world-size 1 \
  --disable-wandb
```

Evaluate the resulting model checkpoint:

```bash
uv run --extra eval openwam-eval \
  --cfg configs/evals/public_tiny_synthetic_contract.yaml \
  --checkpoint "$RUN_ROOT/checkpoints/checkpoint_step_2/model_state.pt" \
  --device cpu \
  --max-batches 1
```

The run directory now contains the resolved config, logs, model state, and full
optimizer/scheduler/strategy state. `full_training_state.pt` provides stateful
continuation; process and dataloader RNG streams are not checkpointed, so it is
not a bitwise replay. `model_state.pt` is the inference and warm-start surface.

For a single-command numerical contract check, run:

```bash
uv run --extra train openwam-sanity \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --device cpu --max-batches 1 --rollout-steps 1
```

## Choose The Next Workflow

| Goal | Start here |
| --- | --- |
| Understand architecture and program choices | [Policy Architectures And Programs](policy_architectures.md) |
| Train, resume, or evaluate a maintained model | [Training And Inference](running_experiments.md) |
| Prepare benchmark data or simulator dependencies | [Benchmarks And Data](benchmarks.md) |
| Add a dataset, policy, decoder, or simulator | [Extension SDK](extension_sdk.md) |
| Record or reproduce an experiment | [Experiment Cards](experiment_cards.md) and [Artifacts](artifacts.md) |

Inspect a resolved typed config without launching training:

```bash
uv run openwam-inspect-config \
  --cfg configs/experiments/dual_expert_libero_joint.yaml
```

Run the packaged extension scaffold before customizing it:

```bash
uv run --extra train openwam-train \
  --cfg templates/extension_method/config.yaml \
  --extension open_wam.templates.extension_method \
  --save-root runs/extension-method-smoke \
  --disable-wandb
```

## Optional Runtimes

Install only the simulator extras needed by the next task:

```bash
uv sync --extra libero
uv sync --extra calvin
uv sync --extra robotwin
uv sync --extra sim
```

The benchmark-named extras add benchmark-side dependency overlays; they do not
include the model stack or upstream source trees. Use `[sim]` for model-driven
closed-loop rollouts, then install the selected benchmark source separately.

## Local Paths

Real datasets, checkpoints, simulator checkouts, and run roots are machine
local. Do not edit public experiment YAMLs to hard-code those paths.

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
```

Populate only the keys referenced by the config you intend to run; unrelated
placeholders may remain unchanged. The file is gitignored. You can also use:

```bash
OPEN_WAM_LOCAL_PATHS=/absolute/path/to/local_paths.yaml uv run openwam-eval ...
```

## LIBERO Local Rollout Setup

To run a LIBERO realtime rollout through
`scripts/run_libero_realtime_sandbox.py` on your own machine, the
`[libero]` extra is necessary but not sufficient: it pins the LIBERO-side
runtime deps (`gym==0.25.2`, `robosuite==1.4.0`, `bddl==1.0.1`, etc.) but
not the model stack (Torch, diffusers, transformers, ...). Three additional
steps are required.

### 1. Install model + simulator deps together

Use `[sim]` (or `[full]`): `[sim]` is the smallest extra that combines the
LIBERO-side deps with the model runtime stack:

```bash
uv sync --extra sim
```

### 2. Install upstream LIBERO from source

LIBERO is **not** distributed on PyPI; the `[libero]` extra only pulls its
runtime deps. Clone the upstream source and pip-install it in editable mode:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO ../LIBERO

# Upstream ships the inner `libero/` directory without an __init__.py and
# relies on namespace-package import. PEP-660 editable installs from
# setuptools generate an empty finder for that case (MAPPING == {}), so
# `import libero` fails outside the repo dir. Touching an empty
# __init__.py makes it a real package and the editable install resolves
# correctly.
touch ../LIBERO/libero/__init__.py

uv pip install -e ../LIBERO
```

### 3. Tell LIBERO where its assets live

LIBERO reads `~/.libero/config.yaml` on import. Create it before the first
run, otherwise it falls into an interactive `input()` prompt:

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<'EOF'
benchmark_root: /absolute/path/to/LIBERO/libero/libero
bddl_files:    /absolute/path/to/LIBERO/libero/libero/bddl_files
init_states:   /absolute/path/to/LIBERO/libero/libero/init_files
datasets:      /absolute/path/to/LIBERO/libero/datasets
assets:        /absolute/path/to/LIBERO/libero/libero/assets
EOF
```

The five keys must match LIBERO's loader exactly — `bddl_files` (not
`bddl_files_folder`), `init_states` (not `init_states_folder`), etc.
Otherwise `libero.libero.get_libero_path` raises `AssertionError: Key ...
not found in config file`.

### 4. Verify the stack imports

```bash
uv run --extra sim python -c "
import torch, libero, open_wam, mujoco, robosuite, diffusers, transformers
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('libero:', libero.__file__)
print('mujoco:', mujoco.__version__, 'robosuite:', robosuite.__version__)
"
```

### 5. (Optional) Silence robosuite's macro warning

The first import emits `[robosuite WARNING] No private macro file found`.
It is harmless, but can be dismissed with:

```bash
uv run --extra sim python -c "import robosuite, os; os.system(f'python {os.path.dirname(robosuite.__file__)}/scripts/setup_macros.py')"
```

## Stable Commands

Preferred package commands:

- `openwam-train`
- `openwam-eval`
- `openwam-inspect-config`
- `openwam-validate-config`
- `openwam-sanity`
- `openwam-sim-rollout`

These commands work from both a source checkout and an installed package.

## Resource Matrix

| Command family | CPU | GPU | Local data | Simulator | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| config inspect | required | no | no | no | no |
| synthetic eval smoke | required | no | no | no | no |
| real dataset train/eval | required | optional | yes | no | optional |
| LIBERO/RoboTwin/CALVIN rollout | required | optional | optional | yes | optional |
| full realtime method comparison | required | usually | yes | yes | yes |

Use pytest markers and local path aliases to make those requirements explicit.
