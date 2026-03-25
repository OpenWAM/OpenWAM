# Open-WAM

Open-WAM is a research codebase for studying **where to attach the action head**
in a world action model while keeping the **video backbone fixed**.

The current implementation is organized around one constraint:

- the shared video path should remain LingBot-compatible
- action-head structure and placement are the main research variable

## Current Status

The repo currently includes:

- a protected LingBot-compatible video backbone path under `src/open_wam/models/video_backbone`
- a uniform data contract for all sources and future action heads
- a config-driven canonical RGB layout builder
- a dataset registry keyed by `data.dataset_type`
- a real LeRobot-v2 adapter for `physical-intelligence/libero`
- a placeholder `contract_only` action head used to validate train and infer boundaries
- Lightning train/eval wrappers and root experiment YAMLs

The first real dataset path is:

- `physical-intelligence/libero`

## Repo Layout

```text
configs/         runnable experiment and eval YAMLs
notes/           research and engineering notes
previous_works/  linked prior projects and references
scripts/         thin wrappers, smoke tests, and inspection scripts
src/open_wam/    all source code
```

Important source packages:

- `src/open_wam/configs`: typed config contracts
- `src/open_wam/data`: dataset adapters, collation, and canonical RGB preprocessing
- `src/open_wam/models/video_backbone`: protected shared video backbone
- `src/open_wam/models/action_heads`: action-head interface and variants
- `src/open_wam/pipelines`: backbone-only and unified WAM orchestration
- `src/open_wam/lightning`: Lightning module and datamodule
- `src/open_wam/training`: train entrypoint
- `src/open_wam/evals`: eval entrypoint

## Design Rules

- Raw-video ingestion lives in the data layer, not in the backbone.
- The shared video backbone should stay stable across action-head experiments.
- Action heads interact with the backbone through explicit contracts, not ad hoc internals.
- Camera names, camera count, layout, action dimension, action horizon, and state dimension should be configurable from YAML.
- Dataset-specific parsing should stay inside dataset adapters registered by `data.dataset_type`.
- Dataset adapters may expose transformed action supervision, not just raw controller deltas.

## Quick Start

Install dependencies with `uv`:

```bash
uv sync
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
```

Train the current contract-only path:

```bash
uv run python -m open_wam.training.train --cfg configs/experiments/contract_only_libero.yaml
```

Run eval:

```bash
uv run python -m open_wam.evals.evaluate --cfg configs/experiments/contract_only_libero.yaml
```

## Current Dataset Contract

All dataset adapters should return the same artifact shape after collation:

- `views`: `dict[str, Tensor]`, each view `[B, T, H, W, 3]`
- `actions`: `[B, H_action, D_action]`
- `action_mask`: optional mask aligned to `actions`
- `state`: optional `[B, H_state, D_state]`
- `state_mask`: optional mask aligned to `state`
- `task_text`: optional tuple of task strings
- `metadata`: tuple of per-sample metadata dicts

The shared backbone canonicalizes `views` into one RGB canvas and emits
`BackboneOutput`.

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
- [notes/libero_lerobot.md](notes/libero_lerobot.md)
- [notes/new_work_roadmap.md](notes/new_work_roadmap.md)

## Current Caveat

`physical-intelligence/libero` is structurally a LeRobot-format dataset, but the
installed `lerobot` package in this environment does not safely load the repo
revision currently on Hugging Face. The current adapter therefore reads the
repo's metadata and episode parquet files directly.
