# Training And Inference

This is the maintained operator guide for Open-WAM training, offline
evaluation, and simulator rollout. Commands in archived engineering notes are
historical records, not alternate launch interfaces.

## Configure Local Assets

Keep datasets, checkpoints, simulator checkouts, and output roots outside
versioned experiment YAML:

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
$EDITOR configs/local_paths.yaml
```

Select another registry with `OPEN_WAM_LOCAL_PATHS=/absolute/path/paths.yaml`.
Validate and inspect a config before allocating a GPU:

```bash
uv run open-wam-validate-config configs/experiments/<experiment>.yaml
uv run open-wam-inspect-config --cfg configs/experiments/<experiment>.yaml
```

## Supported Configs

The generic runtime covers these representative maintained families:

| Family | Experiment config |
| --- | --- |
| Method 1 exact | `parallel_stream_libero_lingbot_exact_heng_compatible.yaml` |
| Method 2 action-conditioned | `parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml` |
| Method 4 latent | `post_latent_libero_latent_local_video_conditioned.yaml` |
| Method 4 decoded | `post_decoded_libero_latent_local_video_conditioned.yaml` |
| M5 video then action | `mot_libero_latent_local_video_then_action_heng_compatible.yaml` |
| M5 action then video | `mot_libero_latent_local_action_then_video_heng_compatible.yaml` |
| M5 joint | `mot_libero_latent_local_joint_heng_compatible.yaml` |
| M5 decoupled | `mot_libero_latent_local_decoupled_same_step_heng_compatible.yaml` |
| M5 video-noisy to action | `mot_libero_latent_local_video_noisy_to_action_heng_compatible.yaml` |
| M5 action-noisy to video | `mot_libero_latent_local_action_noisy_to_video_heng_compatible.yaml` |
| M5 GJD | `mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml` |
| Video-only | `causal_video_prediction_libero_latent_local.yaml` |

## Training

All method families enter the same package-owned training runtime:

```bash
uv run --extra train open-wam-train \
  --cfg configs/experiments/<experiment>.yaml \
  --save-root runs/<run-name> \
  --devices 1
```

Use repeatable `--set section.field=value` arguments only for intentional run
overrides. Keep the resolved config with the checkpoint. External schedulers
may set process placement and retry policy, but should invoke this command
without embedding cluster paths in tracked configs.

Load an external dataset or method extension before config construction:

```bash
uv run --extra train open-wam-train \
  --extension acme_open_wam \
  --cfg /path/to/acme_experiment.yaml
```

### Initialization And Exact Resume

Use `--transformer-subdir` to initialize a new run from a video-transformer
export. Use `--checkpoint-root` for a `checkpoint_step_*` directory: it selects
`full_training_state.pt` when available and falls back to model-only state only
when no full state exists.

```bash
uv run --extra train open-wam-train \
  --cfg configs/experiments/<experiment>.yaml \
  --save-root runs/<run-name> \
  --checkpoint-root runs/<run-name>/checkpoints/checkpoint_step_N
```

For an exact continuation, confirm the source checkpoint contains
`full_training_state.pt`. This restores optimizer, scheduler, RNG, strategy,
and step state. `--resume-from` can select that file explicitly. A
`model_state.pt` checkpoint is a warm start, not an exact resume.

## Generalist Joint Denoising

Use the maintained GJD wrapper so training and rollout resolve the same
ablation semantics. M5 is the standard GJD implementation; M1 remains a
documented compatibility and diagnostic path.

```bash
bash scripts/run_gjd_libero.sh train \
  --method m5 \
  --ablation mode_token \
  --save-root runs/m5-gjd-mode-token \
  --dataset-root /path/to/libero_10 \
  --transformer-subdir /path/to/base/transformer \
  --enable-wandb \
  --wandb-project openwam-gjd
```

Valid ablations are `vanilla`, `pure_joint`, and `mode_token`. Vanilla and
mode-token use the configured real/counterfactual dynamics mixture. Pure-joint
is demo-only. Configure counterfactual train and validation latent roots in
`configs/local_paths.yaml` or with explicit `--set` overrides.

Resume through the same wrapper:

```bash
bash scripts/run_gjd_libero.sh train \
  --method m5 \
  --ablation mode_token \
  --save-root runs/m5-gjd-mode-token \
  --checkpoint-root runs/m5-gjd-mode-token/checkpoints/checkpoint_step_N
```

## Offline Evaluation

Use eval configs for dataset-level metrics:

```bash
uv run --extra eval open-wam-eval \
  --cfg configs/evals/<evaluation>.yaml \
  --device cuda:0 \
  --max-batches 16
```

Use `open-wam-sanity` when the goal is to verify loading, one train/eval path,
and rollout-style tensor flow without making a benchmark claim:

```bash
uv run --extra train open-wam-sanity \
  --cfg configs/examples/<example>.yaml \
  --device cpu \
  --max-batches 1 \
  --rollout-steps 1
```

Sanity reports intentionally inspect exactly one batch. Use `open-wam-eval`
for aggregated multi-batch metrics.

FDM/IDM ablations and simulator counterfactual renders are checkout-only
research diagnostics, not installed library APIs. Their stable commands require
an explicit checkpoint and local data inputs:

```bash
uv run --extra eval python scripts/run_joint_denoising_fdm_ablation.py \
  --checkpoint /path/to/checkpoint_step_N \
  --dataset-root /path/to/libero_10

uv run --extra sim python scripts/run_joint_denoising_fdm_counterfactual.py \
  --checkpoint /path/to/checkpoint_step_N \
  --replay-status-path /path/to/replay_status.jsonl \
  --episode-indices 0,1,2 \
  --dataset-root /path/to/libero_10
```

Use `--help` to select modes, windows, branches, and output paths. The reusable
GJD policy and counterfactual-action contracts continue to live in `open_wam`;
only the experiment orchestration and visualization live under `scripts/`.

## LIBERO Inference

The maintained M5 evaluator loads one checkpoint and executes a closed-loop
episode with the LingBot streaming VAE:

```bash
uv run --extra sim python scripts/run_libero_mot_visualization.py \
  --cfg /path/to/checkpoint_step_N/resolved_config.yaml \
  --checkpoint /path/to/checkpoint_step_N \
  --frontend-encode-mode lingbot_streaming_vae \
  --mot-inference-window-size 30 \
  --benchmark libero_10 \
  --task-id 0 \
  --episode-idx 0 \
  --max-timestep 800 \
  --max-chunks 50 \
  --startup-model-obs-frames 1 \
  --startup-env-init-steps 5 \
  --output-dir outputs/libero_m5 \
  --runtime-device cuda:0 \
  --action-device cuda:0 \
  --frontend-device cuda:0 \
  --decode-device cuda:0
```

The strict non-GJD M5 contract uses an `800` timestep and `50` chunk limit.
GJD uses the wrapper and defaults to `1500/100`:

```bash
bash scripts/run_gjd_libero.sh rollout \
  --method m5 \
  --ablation mode_token \
  --checkpoint /path/to/checkpoint_step_N \
  --task-id 0 \
  --episode-idx 0 \
  --output-dir outputs/libero_gjd
```

For a task/episode sweep, use the loaded-once batch evaluator rather than a
shell loop that reloads the checkpoint for every episode:

```bash
uv run --extra sim python scripts/run_libero_mot_batch_visualization.py \
  --cfg /path/to/checkpoint_step_N/resolved_config.yaml \
  --checkpoint /path/to/checkpoint_step_N \
  --benchmark libero_10 \
  --task-ids 0-9 \
  --episode-idxs 0-49 \
  --seed-by-episode \
  --frontend-encode-mode lingbot_streaming_vae \
  --mot-inference-window-size 30 \
  --max-timestep 800 \
  --max-chunks 50 \
  --startup-model-obs-frames 1 \
  --startup-env-init-steps 5 \
  --output-dir outputs/libero_m5_batch \
  --save-rollout-video \
  --runtime-device cuda:0 \
  --action-device cuda:0 \
  --frontend-device cuda:0 \
  --decode-device cuda:0
```

The batch runner preserves the single-episode runtime semantics and starts
fresh rollout state for each episode while retaining loaded model resources.

Method 1 uses the shared realtime sandbox:

```bash
uv run --extra sim python scripts/run_libero_realtime_sandbox.py \
  --cfg /path/to/checkpoint_step_N/resolved_config.yaml \
  --checkpoint /path/to/checkpoint_step_N \
  --task-id 0 \
  --episode-idx 0 \
  --output-dir outputs/libero_m1
```

## Generic Simulator Rollout

Use the package simulator boundary for configured RoboTwin or CALVIN adapters:

```bash
uv run --extra sim open-wam-sim-rollout \
  --cfg configs/examples/<benchmark>.yaml \
  --benchmark <robotwin-or-calvin> \
  --target-action-hz 10
```

Closed-loop reports should record target and achieved action rate, fallback
actions, task and episode identity, exact resolved config, checkpoint, success,
and video path. Never compare realtime results without also fixing planner
mode, diffusion steps, fallback policy, and horizon.

## Numerical Characterization

Changes that can affect model numerics must verify immutable fixtures and
checkpoint-backed goldens rather than regenerate expected values. The exact
six-mode M5 training, recurrent inference, four-GPU, full-state resume, and GJD
gates are documented in [MoT Refactor Characterization](mot_refactor_characterization.md).
