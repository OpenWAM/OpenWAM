# Open-WAM

Open-WAM is a research codebase for studying **where to attach the action head**
in a world action model while keeping the **video backbone fixed**.

The current implementation is organized around one constraint:

- the shared visual path should remain LingBot-compatible
- policy attachment structure and placement are the main research variable

## Current Status

The repo currently includes:

- a stage-aware `VisualTower + PolicyVariant + ActionDecoder` stack
- runnable `parallel_stream`, `register_attached`, `post_latent`, and
  `post_decoded` policy variants
- a LingBot replica backbone as the default shared-core family for real
  multimodal variants
- a shared runtime backbone knob under `backbone.implementation`:
  - `shared_transformer` (default)
  - `dummy` (smoke/legacy only)
- an optional `backbone.load_reference_core_weights` path that loads LingBot
  backbone weights into the shared replica core for
  `register_attached`, `post_latent`, and `post_decoded`
- an exact LingBot parallel-stream runtime path that executes on the same
  shared backbone object used by the other real variants
- a uniform data contract for all sources and policy variants
- a config-driven canonical RGB layout builder
- a dataset registry keyed by `data.dataset_type`
- a real LeRobot-v2 adapter for `physical-intelligence/libero`
- legacy `contract_only` compatibility via config migration into the new stack
- Lightning train/eval wrappers and root experiment YAMLs

The first real dataset path is:

- `physical-intelligence/libero`

## Repo Layout

```text
configs/         runnable experiment and eval YAMLs
notes/           research and engineering notes
AGENTS.md        repo-level contributor and agent style guide
src/open_wam/third_party/  vendored external modules kept inside the repo
scripts/         thin wrappers, smoke tests, and inspection scripts
src/open_wam/    all source code
```

Important source packages:

- `src/open_wam/configs`: typed config contracts
- `src/open_wam/data`: dataset adapters, collation, and canonical RGB preprocessing
- `src/open_wam/models/visual_tower`: shared visual frontend, core, decode
  boundary, and exact LingBot reference loader
- `src/open_wam/models/policy_variants`: `parallel_stream`,
  `register_attached`, `post_latent`, and `post_decoded` attachment paths
- `src/open_wam/models/action_decoders`: action decoders and losses
- `src/open_wam/models/video_backbone`: backbone config and compatibility contracts
- `src/open_wam/pipelines`: variant pipeline, exact LingBot runner, and rollout helpers
- `src/open_wam/lightning`: Lightning module and datamodule
- `src/open_wam/training`: train entrypoint
- `src/open_wam/evals`: eval entrypoint

## Design Rules

- Raw-video ingestion lives in the data layer, not in the backbone.
- The shared visual tower should stay stable across policy-attachment experiments.
- Policy variants interact with the backbone through explicit stage contracts, not ad hoc internals.
- Camera names, camera count, layout, action dimension, action horizon, and state dimension should be configurable from YAML.
- Dataset-specific parsing should stay inside dataset adapters registered by `data.dataset_type`.
- Dataset adapters may expose transformed action supervision, not just raw controller deltas.
- All four methodologies should continue to share the same top-level `VariantPipeline -> VisualTower` boundary even when their within-core runtimes differ.
- For the canonical multimodal methods, differences should come from runtime
  programs, sequence semantics, cache policy, and decoders rather than from
  swapping out the transformer object underneath them.

## Trainer and Variant Flow

Training uses one generic Lightning stack:

- [src/open_wam/training/train.py](src/open_wam/training/train.py) loads a root
  experiment config and instantiates one `OpenWAMLightningModule` and one
  `OpenWAMDataModule`
- [src/open_wam/lightning/module.py](src/open_wam/lightning/module.py) converts
  `WAMBatch` into `PolicyTrainBatch` and always calls
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
- `register_attached` / method 2: shared runtime-program executor with
  structured sequence adapters, structured attention kernels, shared stream
  adapters, and shared stream output heads
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

Method 2, DreamZero-style register-attached on LingBot backbone:

- video latents get their own noise scheduler, targets, and weighted loss
- actions get their own noise scheduler, targets, and weighted loss
- the shared core sees noisy video and noisy action tokens together
- training loss is `video_diffusion_loss + action_diffusion_loss`
- `register_attached`
  - full clean-video teacher-forcing prefix during training
  - one sampled timestep per video frame
  - action timesteps are coupled to future video blocks by default
  - joint inference rollout: update video and action in the same denoising loop
  - `inference.joint_sampler: unipc` by default for DreamZero-style multistep sampling
  - optional shared denoising count via `inference.joint_num_inference_steps`
  - per-stream CFG stays generic:
    - `inference.video_cfg_mode: guided`
    - `inference.action_cfg_mode: conditioned`
  - cache warmup stays generic:
    - `inference.joint_cache_warmup_source`
    - `inference.joint_cache_initial_warmup_anchor`
    - `inference.joint_cache_rollout_warmup_anchor`
  - `inference.joint_observed_video_prefix_frames: 1` keeps the observed
    first frame fixed during inference-time denoising
  - stream tokenizers and flow heads are now backbone-owned shared runtime
    components rather than variant-local modules

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

Set up the `uv` environment used for current CUDA runs:

```bash
uv sync --group dev
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Install MuJoCo-backed visualization extras only when you need the viewer scripts:

```bash
uv sync --extra viz
```

Inspect the current LIBERO adapter:

```bash
python scripts/inspect_libero_adapter.py --cfg configs/experiments/contract_only_libero.yaml
```

Use the lightweight smoke configs when you want fast CPU checks of method 1 and
method 2 without instantiating the full LingBot-scale backbone:

```bash
python -m open_wam.evals.evaluate --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml --device cpu
python -m open_wam.evals.evaluate --cfg configs/experiments/register_attached_robotwin_smoke.yaml --device cpu
```

Visualize the default LIBERO reference-relative EEF target in MuJoCo:

```bash
uv run mjpython scripts/visualize_libero_reference_pose.py --cfg configs/experiments/contract_only_libero.yaml
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

Run smoke tests:

```bash
python scripts/smoke_backbone_only.py
python scripts/smoke_phase_two.py
python scripts/smoke_parallel_stream.py
python scripts/smoke_lingbot_exact_runner.py
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

Run the Heng-compatible exact LIBERO side-by-side render with a fixed seed:

```bash
uv run python scripts/run_libero_exact_visualization.py \
  --cfg configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml \
  --benchmark libero_10 \
  --task-id 8 \
  --episode-idx 0 \
  --max-chunks 6 \
  --seed 1234 \
  --output-dir outputs/libero_exact_visualization_chunks6_seeded

uv run python scripts/run_heng_libero_exact_visualization.py \
  --benchmark libero_10 \
  --task-id 8 \
  --episode-idx 0 \
  --max-chunks 6 \
  --seed 1234 \
  --output-dir outputs/libero_exact_visualization_chunks6_seeded
```

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

With `backbone.implementation = shared_transformer`, methods 1, 2, and 4 run
through the same shared `VisualTower` frontend and shared transformer-core
object.

What differs between the methods is the runtime program:

- method 1 uses exact LingBot-compatible runtime programs, chunk/window
  attention, and slot-pool cache semantics
- method 2 uses structured register-sequence runtime programs, structured
  branchwise attention, and structured rollout-cache semantics
- method 4 uses the same shared core with a lightweight decoded-feature policy
  head

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
and Heng comparison workflow, see:

- [notes/lingbot_reference_usage.md](notes/lingbot_reference_usage.md)
- [notes/libero_exact_rendering.md](notes/libero_exact_rendering.md)

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

- [notes/README.md](notes/README.md)
- [notes/collaboration_guide.md](notes/collaboration_guide.md)
- [notes/architecture.md](notes/architecture.md)
- [notes/current_all_variant_execution_status.md](notes/current_all_variant_execution_status.md)
- [notes/current_four_method_architecture.md](notes/current_four_method_architecture.md)
- [notes/libero_exact_rendering.md](notes/libero_exact_rendering.md)
- [notes/lingbot_reference_usage.md](notes/lingbot_reference_usage.md)
- [notes/libero_lerobot.md](notes/libero_lerobot.md)

## Current Caveat

`physical-intelligence/libero` is structurally a LeRobot-format dataset, but the
installed `lerobot` package in this environment does not safely load the repo
revision currently on Hugging Face. The current adapter therefore reads the
repo's metadata and episode parquet files directly.
