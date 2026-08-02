# Open-WAM

Open-WAM is a research codebase for studying **where and how to attach action
policy logic** in a world action model while keeping the **video backbone
fixed**.

The current implementation is organized around one constraint:

- the shared visual path should remain LingBot-compatible
- policy attachment structure and placement are the main research variable

## Current Status

The repo currently includes:

- a stage-aware `VisualTower + PolicyVariant + ActionDecoder` stack
- runnable `parallel_stream`, `post_latent`, `post_decoded`, `mot`, and
  `causal_video_prediction` variants
- a LingBot replica backbone as the default shared-core family for real
  multimodal variants
- a shared runtime backbone knob under `backbone.implementation`:
  - `shared_transformer` (default)
  - `dummy` (smoke/legacy only)
- an optional `backbone.load_reference_core_weights` path that loads LingBot
  backbone weights into the shared replica core
- an exact LingBot parallel-stream runtime path that executes on the same
  shared backbone object used by the other real variants
- a uniform data contract for all sources and policy variants
- a config-driven canonical RGB layout builder
- a dataset registry keyed by `data.dataset_type`
- a real LeRobot-v2 adapter for `physical-intelligence/libero`
- an explicit `contract_only` compatibility profile composed from
  `post_latent` plus the MLP action decoder
- one composable train runtime for single-device, DDP, and FSDP execution

The first real dataset path is:

- `physical-intelligence/libero`

## Repo Layout

```text
configs/         runnable experiment and eval YAMLs
docs/            public quickstart, CLI, testing, artifact, and deployment docs
notes/           research and engineering notes
deployment/      separate FR3/SO-101 hardware operations workspace
                 (checkout-only, separately tested; see deployment/README.md)
AGENTS.md        repo-level contributor and agent style guide
src/open_wam/third_party/  vendored external modules kept inside the repo
scripts/         thin wrappers, smoke tests, and inspection scripts
src/open_wam/    all source code
```

Important source packages:

- `src/open_wam/configs`: typed config contracts, loading, and local path resolution
- `src/open_wam/data`: dataset adapters, collation, and canonical RGB preprocessing
- `src/open_wam/models/visual_tower`: shared visual frontend, core, decode
  boundary, and exact LingBot reference loader
- `src/open_wam/models/policy_variants`: method-specific train/infer behavior
  for `parallel_stream`, `post_latent`, `post_decoded`, `mot`, and
  `causal_video_prediction`
- `src/open_wam/models/action_decoders`: action decoders and losses
- `src/open_wam/models/video_backbone`: backbone config and compatibility contracts
- `src/open_wam/pipelines`: variant pipeline, exact LingBot runner, and rollout helpers
- `src/open_wam/training`: data loading, train steps, strategies, logging, and checkpoints
- `src/open_wam/evals`: eval entrypoint

## Public Docs

- [Quickstart](docs/quickstart.md): fresh clone to CPU smoke, local path setup,
  and resource matrix
- [Architecture](docs/architecture.md): stable runtime boundary and extension
  contracts
- [Method families](docs/method_families.md): current policy-attachment
  families and how they share runtime infrastructure
- [Benchmarks and data](docs/benchmarks.md): public fixture, LIBERO,
  RoboTwin, CALVIN, action dimensions, and visual layout contracts
- [Running experiments](docs/running_experiments.md): train, eval, sanity, and
  realtime rollout workflow
- [CLI reference](docs/cli.md): package-owned commands and legacy script policy
- [Testing](docs/testing.md): pytest markers and CI tiers
- [Artifacts](docs/artifacts.md): local path registry, checkpoint manifests, and
  layout conventions
- [Deployment namespace](docs/deployment_namespace.md): `open_wam` research
  package vs deployment `openwam`
- [Reproducibility](docs/reproducibility.md): result schemas, experiment cards,
  and WandB naming
- [Extension SDK](docs/extension_sdk.md): dataset, policy-variant, and decoder
  registry extension points
- [Experiment cards](docs/experiment_cards.md): method-family result card
  template and current public-card status
- [GitHub Pages](docs/github_pages.md): generated MkDocs site and required
  repository settings

## Design Rules

- Raw-video ingestion lives in the data layer, not in the backbone.
- The shared visual tower should stay stable across policy-attachment experiments.
- Policy variants interact with the backbone through explicit stage contracts, not ad hoc internals.
- Camera names, camera count, layout, action dimension, action horizon, and state dimension should be configurable from YAML.
- Dataset-specific parsing should stay inside dataset adapters registered by `data.dataset_type`.
- Dataset adapters may expose transformed action supervision, not just raw controller deltas.
- Offline latent encoding composes typed contracts, deterministic planning,
  sidecar validation, and tensor execution behind the stable data facade.
- All method families should continue to share the same top-level `VariantPipeline -> VisualTower` boundary even when their within-core runtimes differ.
- For the canonical multimodal methods, differences should come from runtime
  programs, sequence semantics, cache policy, and decoders rather than from
  swapping out the transformer object underneath them.

## Trainer and Variant Flow

Training uses one generic composable runtime:

- [src/open_wam/training/train.py](src/open_wam/training/train.py) loads a root
  experiment config and constructs `TrainingRuntime`
- [src/open_wam/training/runtime.py](src/open_wam/training/runtime.py) owns
  lifecycle composition, train/validation loops, distributed metric reduction,
  and checkpoint scheduling
- [src/open_wam/training/data_loading.py](src/open_wam/training/data_loading.py)
  owns raw/latent dataset selection, samplers, collation, and loaders
- [src/open_wam/training/auxiliary_validation.py](src/open_wam/training/auxiliary_validation.py)
  owns validation source views and validation-only metadata overrides
- [src/open_wam/training/logging.py](src/open_wam/training/logging.py) and
  [src/open_wam/training/optim.py](src/open_wam/training/optim.py) own reusable
  sinks and optimizer/scheduler state handling
- [src/open_wam/training/step_executor.py](src/open_wam/training/step_executor.py)
  converts the public data batch into `PolicyTrainBatch` and calls
  `pipeline.forward_train(...)`
- [src/open_wam/pipelines/variant_pipeline.py](src/open_wam/pipelines/variant_pipeline.py)
  is where the variant actually changes behavior:
  - prepare visual stages
  - let the policy variant prepare train-time artifacts
  - run the variant forward
  - let the action decoder compute the final loss

That means the trainer itself is not variant-specific. Variants change training
semantics by implementing:

- `required_visual_stages()`
- `prepare_train_inputs()`
- `forward_train()`
- `prepare_infer_state()`
- `forward_infer_step()`

inside `src/open_wam/models/policy_variants/`.

The current method split is:

- `parallel_stream` / method 1: exact LingBot train/infer semantics through
  shared-backbone exact runtime programs
- `mot` / method 5: VTA, ATV, joint, decoupled, VNA, ANV, and GJD semantics
  expressed through the same exact shared-backbone runtime
- `post_latent` / `post_decoded`: simple feature-attached baselines over the
  same stage-aware pipeline

## Current Diffusion Granularity

Current diffusion behavior is split into three buckets.

Method 1, LingBot:

- `parallel_stream`
- separate video and action schedulers
- one sampled diffusion timestep per **frame**
- video sigma is broadcast across latent channels and spatial positions of that
  frame
- action sigma is broadcast across action channels and `action_per_frame`
  positions of that frame
- loss is reduced and normalized per frame
- the shared backbone executes method 1 through exact runtime programs rather
  than a sidecar transformer module

Traditional Method 2 `register_attached` has been removed from the public
configuration and runtime surface. Use action-conditioned `parallel_stream`
for the maintained Method 2 semantics; Git history preserves obsolete inputs.

Action-only diffusion variants:

- `post_latent`
- `post_decoded`

These still use LingBot-style action flow matching:

- action tensor is `[B, H_action, D_action]`
- one sampled diffusion timestep per **action horizon slot**
- that sigma is broadcast across all `D_action` channels at that slot
- diffusion loss is reduced per slot across action dims, then averaged over
  slots and batch

## Quick Start

Set up the minimal development environment. This installs the core package
surface only; it does not install Torch, simulator packages, or video codecs:

```bash
uv sync --group dev
uv run python -c "import open_wam; print(open_wam.__version__)"
```

Run static config validation without launching model code:

```bash
uv run open-wam-validate-config configs/examples/public_tiny_synthetic_contract.yaml
```

Inspect a config through the stable package CLI:

```bash
uv run open-wam-inspect-config --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml
```

For real datasets/checkpoints, create a local path registry:

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
```

Replace the `/path/to/...` placeholders. `configs/local_paths.yaml` is
gitignored. See [docs/quickstart.md](docs/quickstart.md) and
[docs/artifacts.md](docs/artifacts.md) for the public path workflow.

Install optional extras only when needed:

```bash
uv sync --extra torch
uv sync --extra train
uv sync --extra eval
uv sync --extra tracking
uv sync --extra viz
uv sync --extra libero
uv sync --extra robotwin
uv sync --extra calvin
uv sync --extra sim
uv sync --extra deployment
uv sync --extra docs
uv sync --extra full
```

Simulator config records, realtime planner records, scheduling and plan-queue
policy, and the generic backend protocol are available from the base install;
importing them does not load NumPy, Torch, or a benchmark. NumPy-backed plan
materialization and rollout reporting remain in the optional `[sim]` runtime.
The `[libero]`, `[robotwin]`, and `[calvin]` extras are benchmark-side
dependency overlays and do not include the model stack or upstream source
checkouts. For a model-driven simulator rollout, use `[sim]` (or `[full]`)
and install the selected benchmark source separately.

For a local LIBERO rollout, add the upstream source plus a one-line config so
LIBERO can locate its bddl / init / asset folders:

```bash
# 1. Install model + simulator deps in one shot
uv sync --extra sim

# 2. Clone upstream LIBERO; it is not on PyPI
git clone https://github.com/Lifelong-Robot-Learning/LIBERO ../LIBERO
# Empty __init__.py so editable installs see `libero` as a real package
# instead of an empty PEP-660 namespace finder.
touch ../LIBERO/libero/__init__.py
uv pip install -e ../LIBERO

# 3. Tell LIBERO where its asset/bddl/init directories live
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<'EOF'
benchmark_root: /absolute/path/to/LIBERO/libero/libero
bddl_files:    /absolute/path/to/LIBERO/libero/libero/bddl_files
init_states:   /absolute/path/to/LIBERO/libero/libero/init_files
datasets:      /absolute/path/to/LIBERO/libero/datasets
assets:        /absolute/path/to/LIBERO/libero/libero/assets
EOF

# 4. Point the local checkpoint registry at the trained Method 1 ckpt
cp configs/local_paths.sample.yaml configs/local_paths.yaml
# Replace the parallel_stream_exact_libero_step_400 placeholder with the
# absolute path to your local checkpoint_step_400 directory.
```

The keys in `~/.libero/config.yaml` must be exactly `benchmark_root`,
`bddl_files`, `init_states`, `datasets`, `assets` — without the `_folder`
suffix LIBERO's loader rejects them.

Use the lightweight smoke config after installing Torch/runtime extras when
you want a fast CPU check without instantiating the full LingBot-scale
backbone:

```bash
uv run --extra eval open-wam-eval --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml --device cpu
```

Visualize the default LIBERO reference-relative EEF target reconstructed from
the public action representation:

```bash
uv run mjpython scripts/visualize_libero_pose_compare.py \
  --cfg configs/experiments/contract_only_libero.yaml \
  --mode reconstructed
```

Compare the original absolute LIBERO state rollout and the rollout reconstructed
from our public action representation:

```bash
uv run mjpython scripts/visualize_libero_pose_compare.py --cfg configs/experiments/contract_only_libero.yaml --mode compare
```

Compare the entire episode trajectory instead of only one sampled horizon:

```bash
uv run python scripts/visualize_libero_pose_compare.py \
  --cfg configs/experiments/contract_only_libero.yaml \
  --trajectory episode \
  --episode-index 0 \
  --dry-run
```

Replay the same public trajectory inside the real LIBERO environment and save a
side-by-side GIF of original dataset frames vs env replay:

```bash
./scripts/run_eval_libero_env_tracking.sh \
  --cfg configs/experiments/contract_only_libero.yaml \
  --trajectory episode \
  --episode-index 0 \
  --control-substeps-per-target 8 \
  --output outputs/libero_tracking_ep0.gif
```

Run package-owned sanity checks against representative typed configs:

```bash
uv run --extra train open-wam-sanity \
  --cfg configs/experiments/contract_only_robotwin.yaml \
  --device cpu --batch-size 1 --rollout-steps 1
uv run --extra train open-wam-sanity \
  --cfg configs/experiments/post_latent_robotwin.yaml \
  --device cpu --batch-size 1 --rollout-steps 1
uv run --extra train open-wam-sanity \
  --cfg configs/experiments/post_decoded_robotwin.yaml \
  --device cpu --batch-size 1 --rollout-steps 1
uv run --extra train open-wam-sanity \
  --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml \
  --device cpu --batch-size 1 --rollout-steps 1
```

Train the current contract-only path:

```bash
uv run python -m open_wam.training.train --cfg configs/experiments/contract_only_libero.yaml
```

Train from the local offline LIBERO HDF5 dataset tree instead of the LeRobot
metadata path:

```bash
uv run python -m open_wam.training.train --cfg configs/experiments/contract_only_libero_local.yaml
```

Run a current Heng-compatible exact LIBERO realtime rollout with a fixed seed:

```bash
uv run python scripts/run_libero_realtime_sandbox.py \
  --cfg configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml \
  --checkpoint <CURRENT_METHOD1_CHECKPOINT>/model_state.pt \
  --merge-checkpoint-runtime-config \
  --benchmark libero_10 \
  --task-id 8 \
  --episode-idx 0 \
  --max-actions 96 \
  --seed 1234 \
  --artifact-profile standard \
  --output-dir outputs/libero_exact_visualization_chunks6_seeded

uv run python scripts/run_heng_libero_exact_visualization.py \
  --heng-repo-root <LINGBOT_VA_SOURCE_CHECKOUT> \
  --benchmark libero_10 \
  --task-id 8 \
  --episode-idx 0 \
  --max-chunks 6 \
  --seed 1234 \
  --pretrained-root <LINGBOT_VA_BASE_ROOT> \
  --transformer-dir <METHOD1_STEP400_CHECKPOINT>/transformer \
  --output-dir outputs/libero_exact_visualization_chunks6_seeded
```

Replace `<LINGBOT_VA_BASE_ROOT>` with the local LingBot/Wan base model root and
`<CURRENT_METHOD1_CHECKPOINT>` / `<METHOD1_STEP400_CHECKPOINT>` with local
checkpoint-step directories. Retired exact visualization, realtime, and
ablation command paths fail closed and point to the maintained realtime or
sampled-evaluation tools. Historical implementations remain available in Git
history and the frozen reference checkout.

Run eval:

```bash
uv run python -m open_wam.evals.evaluate --cfg configs/experiments/contract_only_libero.yaml
```

Or use an eval-wrapper YAML under `configs/evals/`:

```bash
uv run python -m open_wam.evals.evaluate --cfg configs/evals/contract_only_robotwin.yaml
```

Trajectory eval modes:

- `trajectory`: teacher-forced visual rollout over episode windows
- `trajectory_open_loop`: reuses predicted video latents across later windows
  by aligning overlapping frame indices and seeding newly entered frames from
  the current clean observation window

The current generic evaluator now:

- loads either an experiment YAML or an eval-wrapper YAML
- builds the current `VariantPipeline`
- supports three modes:
  - `batch`: independent one-window inference on each sampled batch
  - `trajectory`: stateful rollout over episode-ordered windows, carrying
    `PolicyInferState` and previous predictions across the trajectory
  - `trajectory_open_loop`: same stateful rollout, but the next step may
    consume predicted video latents instead of rereading GT RGB
- runs the standard inference path in both modes, so each evaluation step still
  includes the variant's full denoising loop
- reports action-prediction shape and mean masked action MSE
- reports video latent MSE whenever the active variant exposes predicted
  latents
- reports mean per-trajectory action MSE and video latent MSE for trajectory
  modes when available
- optionally loads a checkpoint passed with `--checkpoint`

Trajectory mode requires an episode-aware dataset adapter, i.e. one that can
group windows by `episode_index` and `observation_start`. The current LIBERO
adapters support this; the synthetic RobotWin smoke dataset does not.

Run trajectory eval on LIBERO:

```bash
uv run python -m open_wam.evals.evaluate \
  --cfg configs/evals/contract_only_libero_trajectory.yaml
```

## Backbone Sharing Clarification

All canonical multimodal methods now run through the same top-level owner:

- `VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder`

With `backbone.implementation = shared_transformer`, retained multimodal
methods run through the same shared `VisualTower` frontend and transformer-core
object.

What differs between the methods is the runtime program:

- method 1 uses exact LingBot-compatible runtime programs, chunk/window
  attention, and slot-pool cache semantics
- method 5 uses exact MoT runtime programs with config-selected sequence,
  attention, scheduler, and cache semantics
- method 4 uses the same shared core with a lightweight decoded-feature policy
  head

The historical register-attached Method 2 runtime is not a supported public
surface. Its implementation and configs were removed; historical design notes
remain under `notes/finished_roadmaps/`.

`post_latent` is the intentional exception: when configured with
`attach_site=post_frontend_latents`, it may stop at the shared frontend and
bypass the transformer core by design.

LingBot-compatible weights can initialize the shared backbone by setting:

- `backbone.implementation: shared_transformer`
- `backbone.load_reference_core_weights: true`
- `backbone.pretrained_model_name_or_path: /path/to/checkpoint-root`

Exact method-1 execution uses that same shared backbone object, but drives it
through the LingBot-compatible exact runtime programs exposed by the shared
runtime executor rather than a sidecar transformer module.

For exact loading and execution details, including the local LIBERO 30D path
and maintained realtime rollout workflow, see:

- [notes/lingbot_reference_usage.md](notes/lingbot_reference_usage.md)
- [notes/libero_realtime_sandbox.md](notes/libero_realtime_sandbox.md)

## Current Dataset Contract

All dataset adapters should return the same artifact shape after collation:

- `views`: `dict[str, Tensor]`, each view `[B, T, H, W, 3]`
- `actions`: `[B, H_action, D_action]`
- `action_mask`: optional mask aligned to `actions`
- `state`: optional `[B, H_state, D_state]`
- `state_mask`: optional mask aligned to `state`
- `task_text`: optional tuple of task strings
- `metadata`: tuple of per-sample metadata dicts

The shared visual path canonicalizes `views` into one RGB canvas and emits
stageful `VisualStageOutputs`.

For LIBERO specifically, `actions` default to a transformed 7D
reference-relative EEF target `[rel_xyz, rel_axis_angle, gripper_1d_command]`.
The pose part comes from dataset state, while the last scalar is copied from
the raw LIBERO action command rather than from finger-joint state. That public
target is now supported by:

- exact original-vs-reconstructed trajectory comparison over either a sampled
  horizon or a full episode
- closed-loop conversion back into LIBERO `OSC_POSE` actions for simulator
  replay / evaluation

## Notes

Start here for collaborator-facing context:

- [docs/running_experiments.md](docs/running_experiments.md)
- [notes/README.md](notes/README.md)
- [notes/collaboration_guide.md](notes/collaboration_guide.md)
- [notes/architecture.md](notes/architecture.md)
- [notes/current_all_variant_execution_status.md](notes/current_all_variant_execution_status.md)
- [notes/current_method_architecture.md](notes/current_method_architecture.md)
- [notes/libero_realtime_sandbox.md](notes/libero_realtime_sandbox.md)
- [notes/lingbot_reference_usage.md](notes/lingbot_reference_usage.md)
- [notes/libero_lerobot.md](notes/libero_lerobot.md)

## Current Caveat

`physical-intelligence/libero` is structurally a LeRobot-format dataset, but the
installed `lerobot` package in this environment does not safely load the repo
revision currently on Hugging Face. The current adapter therefore reads the
repo's metadata and episode parquet files directly.
