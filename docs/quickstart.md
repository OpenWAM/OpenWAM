# Open-WAM Quickstart

This guide is the public first-run path. It does not require private datasets,
private checkpoints, CUDA, or external simulators.

Open-WAM supports Linux with Python 3.11 or 3.12. Install `uv` before using the
commands below.

## Install

```bash
uv sync --group dev --extra train --extra eval
```

This installs the CPU-capable development, training, and evaluation stack used
by the complete first run below. The smaller base install is available with
`uv sync --group dev`; it supports imports, config validation, artifact
metadata, and CLI parser surfaces without Torch or simulator packages.

## Complete CPU First Run

Validate the public synthetic train and evaluation configs:

```bash
uv run open-wam-validate-config \
  configs/examples/public_tiny_synthetic_contract.yaml \
  configs/evals/public_tiny_synthetic_contract.yaml
```

Train one step and write a full-state checkpoint:

```bash
RUN_ROOT="runs/public-tiny-$(date +%Y%m%d-%H%M%S)"
uv run --extra train open-wam-train \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --save-root "$RUN_ROOT" \
  --expected-world-size 1 \
  --disable-wandb
```

Resume exactly from step 1 and train step 2:

```bash
uv run --extra train open-wam-train \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --save-root "$RUN_ROOT" \
  --checkpoint-root "$RUN_ROOT/checkpoints/checkpoint_step_1" \
  --num-steps 2 \
  --expected-world-size 1 \
  --disable-wandb
```

Evaluate the resulting model checkpoint:

```bash
uv run --extra eval open-wam-eval \
  --cfg configs/evals/public_tiny_synthetic_contract.yaml \
  --checkpoint "$RUN_ROOT/checkpoints/checkpoint_step_2/model_state.pt" \
  --device cpu \
  --max-batches 1
```

The run directory now contains the resolved config, logs, model state, and full
optimizer/scheduler/RNG state. `full_training_state.pt` provides exact resume;
`model_state.pt` is the inference artifact and warm-start surface.

For a single-command numerical contract check, run:

```bash
uv run --extra train open-wam-sanity \
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
| Reproduce a published result | [Experiment Cards](experiment_cards.md) and [Artifacts](artifacts.md) |

Inspect a resolved typed config without launching training:

```bash
uv run open-wam-inspect-config \
  --cfg configs/experiments/dual_expert_libero_joint.yaml
```

Run the packaged extension scaffold before customizing it:

```bash
uv run --extra train open-wam-train \
  --cfg templates/extension_method/config.yaml \
  --extension open_wam.templates.extension_method \
  --save-root runs/extension-method-smoke \
  --disable-wandb
```

## Optional Runtimes

Install only the simulator or documentation extras needed by the next task:

```bash
uv sync --extra libero
uv sync --extra calvin
uv sync --extra robotwin
uv sync --extra sim
uv sync --extra docs
```

The benchmark-named extras add benchmark-side dependency overlays; they do not
include the model stack or upstream source trees. Use `[sim]` for model-driven
closed-loop rollouts, then install the selected benchmark source separately.

For local documentation preview:

```bash
uv run --extra docs python scripts/build_docs_site.py --output .docs_site
uv run --extra docs mkdocs serve
```

## Local Paths

Real datasets, checkpoints, simulator checkouts, and run roots are machine
local. Do not edit public experiment YAMLs to hard-code those paths.

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
```

Populate only the keys referenced by the config you intend to run; unrelated
placeholders may remain unchanged. The file is gitignored. You can also use:

```bash
OPEN_WAM_LOCAL_PATHS=/absolute/path/to/local_paths.yaml uv run open-wam-eval ...
```

## LIBERO Local Rollout Setup

To actually run a LIBERO realtime rollout (for example through
`scripts/run_libero_realtime_sandbox.py` or
`scripts/run_libero_sampled_eval.py`) on your own machine, the
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

- `open-wam-train`
- `open-wam-eval`
- `open-wam-inspect-config`
- `open-wam-validate-config`
- `open-wam-sanity`
- `open-wam-sim-rollout`

Legacy `python scripts/...` commands remain supported as compatibility
entrypoints while the runtime is migrated into package modules.

## Resource Matrix

| Command family | CPU | GPU | Local data | Simulator | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| config inspect | required | no | no | no | no |
| synthetic eval smoke | required | no | no | no | no |
| real dataset train/eval | required | optional | yes | no | optional |
| LIBERO/RoboTwin/CALVIN rollout | required | optional | optional | yes | optional |
| full realtime method comparison | required | usually | yes | yes | yes |

Use pytest markers and local path aliases to make those requirements explicit.
