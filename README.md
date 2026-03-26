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
- a LingBot-compatible backbone knob under `backbone.implementation`:
  - `dummy`
  - `lingbot_replica`
- an optional `backbone.load_reference_core_weights` path that loads LingBot
  backbone weights into the shared replica core for
  `register_attached`, `post_latent`, and `post_decoded`
- an exact LingBot parallel-stream loading path owned by `VisualTower`
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
- `src/open_wam/pipelines`: variant pipeline, exact LingBot runner, and compatibility builders
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
python scripts/smoke_parallel_stream_lingbot_replica.py
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

## Backbone Sharing Clarification

All four methods now run through the same top-level owner:

- `VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder`

With `backbone.implementation = lingbot_replica`, methods 2, 3, and 4 use the
shared `LingbotVisualFrontend` plus `LingbotReplicaVisualCore`.

The parallel-stream variant has two paths:

- the standard parallel-stream path can also use `lingbot_replica`
- the exact LingBot-compatible path uses a reference transformer loaded as-is
  and owned by `VisualTower`

The exact LingBot path now defaults to the vendored implementation under
`src/open_wam/third_party/lingbot`. You only need
`backbone.reference_model_path` when you deliberately want to override that
with another `model.py`.

So the owner and frontend boundary are shared across all methods, but exact
parallel-stream does not yet share the same physical core module or weights as
`register_attached`, `post_latent`, and `post_decoded`.

The other three variants can use LingBot backbone weights by setting:

- `backbone.implementation: lingbot_replica`
- `backbone.load_reference_core_weights: true`
- `backbone.pretrained_model_name_or_path: /path/to/checkpoint-root`

That path initializes the shared replica core from LingBot-compatible weights
while keeping the stage-aware `VisualTower` contracts intact. An external
`reference_model_path` is optional there too.

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

- [notes/collaboration_guide.md](notes/collaboration_guide.md)
- [notes/architecture.md](notes/architecture.md)
- [notes/current_all_variant_execution_status.md](notes/current_all_variant_execution_status.md)
- [notes/current_four_method_architecture.md](notes/current_four_method_architecture.md)
- [notes/libero_exact_rendering.md](notes/libero_exact_rendering.md)
- [notes/lingbot_reference_usage.md](notes/lingbot_reference_usage.md)
- [notes/libero_lerobot.md](notes/libero_lerobot.md)
- [notes/new_work_roadmap.md](notes/new_work_roadmap.md)

## Current Caveat

`physical-intelligence/libero` is structurally a LeRobot-format dataset, but the
installed `lerobot` package in this environment does not safely load the repo
revision currently on Hugging Face. The current adapter therefore reads the
repo's metadata and episode parquet files directly.
