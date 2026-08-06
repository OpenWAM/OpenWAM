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

| Architecture / program | Experiment config |
| --- | --- |
| Parallel stream exact backend | `parallel_stream_libero_lingbot_exact.yaml` |
| Parallel stream action-conditioned profile | `parallel_stream_libero_joint_denoise.yaml` |
| Post-latent video-conditioned | `post_latent_libero_latent_local_video_conditioned.yaml` |
| Post-decoded video-conditioned | `post_decoded_libero_latent_local_video_conditioned.yaml` |
| Dual expert video then action | `dual_expert_libero_video_then_action.yaml` |
| Dual expert action then video | `dual_expert_libero_action_then_video.yaml` |
| Dual expert joint | `dual_expert_libero_joint.yaml` |
| Dual expert decoupled | `dual_expert_libero_decoupled_same_step.yaml` |
| Dual expert video-noisy to action | `dual_expert_libero_video_noisy_to_action.yaml` |
| Dual expert action-noisy to video | `dual_expert_libero_action_noisy_to_video.yaml` |
| Dual expert GJD | `dual_expert_libero_generalist_joint_denoising.yaml` |
| Video-only | `causal_video_prediction_libero_latent_local.yaml` |

Maintained config names describe the architecture and program and do not carry a
contributor-specific compatibility suffix. Retired `*_heng_compatible` and
`*_heng_eval` names still resolve to these canonical files with a deprecation
warning, so old commands remain usable during migration. Existing copied YAMLs
and checkpoint-local `resolved_config.yaml` files are loaded as written and are
never redirected by the alias resolver. Use canonical names for all new runs,
reports, and automation.

## Training

All architectures and programs enter the same package-owned training runtime:

```bash
uv run --extra train open-wam-train \
  --cfg configs/experiments/<experiment>.yaml \
  --save-root runs/<run-name>
```

Use repeatable `--set section.field=value` arguments only for intentional run
overrides. Keep the resolved config with the checkpoint. External schedulers
may set process placement and retry policy, but should invoke this command
without embedding cluster paths in tracked configs.

### Multi-GPU

Worker processes are created by the external launcher, which supplies the
runtime topology through `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`. Neither config
nor the training CLI creates workers. Launch under a process launcher:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m open_wam.cli.train \
  --cfg configs/experiments/<experiment>.yaml \
  --save-root runs/<run-name> \
  --expected-world-size 4
```

`open_wam.cli.train` is the canonical module and console-script entrypoint. Both
forms share one parser and delegate to the same training implementation.

Under Slurm, run one task per node and let `torchrun` fan out the ranks:

```bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
srun --ntasks-per-node=1 torchrun \
  --nnodes="$SLURM_NNODES" --nproc-per-node=4 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER:29500" \
  --rdzv-id="$SLURM_JOB_ID" \
  -m open_wam.cli.train \
  --cfg configs/experiments/<experiment>.yaml \
  --save-root runs/<run-name> \
  --expected-world-size "$((SLURM_NNODES * 4))"
```

Do not use bare `srun --ntasks-per-node=<gpus>` without a task-local wrapper.
Slurm exports `SLURM_PROCID`, `SLURM_LOCALID`, and `SLURM_NTASKS`, none of which
this runtime reads; it reads only `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. Every
task would see `WORLD_SIZE=1` and run as an independent single-process job. A
direct `srun` integration must map those variables inside each task, not in the
parent allocation shell. Prefer the `torchrun` recipe above.

Open-WAM never creates worker processes from config. `--expected-world-size N`
validates the topology supplied by the launcher and fails before model
construction when `WORLD_SIZE != N`. The legacy `--devices N` option remains a
compatibility alias: it is still recorded as `trainer.devices`, and explicit
CLI values now establish the same launch expectation instead of being silently
ignored. Use `--expected-world-size` in new automation.

`strategy: fsdp` shards parameters, gradients, and optimizer state across the
launched processes. A single process gets no sharding benefit regardless of how
many GPUs are visible, so single-device memory must fit the whole model plus its
optimizer state.

Load an external dataset or policy extension before config construction:

```bash
uv run --extra train open-wam-train \
  --extension acme_open_wam \
  --cfg /path/to/acme_experiment.yaml
```

For the six standard video/action programs, `policy_variant.program` is the
only public program switch. A named config is preferred for recorded runs, but
a one-off ablation can use:

```bash
uv run --extra train open-wam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --set policy_variant.program=video_then_action
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

Distributed runs use `trainer.distributed_timeout_seconds: 1800` by default.
The timeout includes rank-0 reads and writes of full-state checkpoints before
other ranks enter the next collective. Increase it for slower shared storage;
lower it only when faster failure detection is more important than large-state
resume support.

## Generalist Joint Denoising

Use the maintained GJD wrapper so training and rollout resolve the same
ablation semantics. Dual expert is the standard GJD architecture; parallel
stream remains a documented compatibility and diagnostic path.

```bash
bash scripts/run_gjd_libero.sh train \
  --architecture dual_expert \
  --ablation mode_token \
  --save-root runs/dual-expert-gjd-mode-token \
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
  --architecture dual_expert \
  --ablation mode_token \
  --save-root runs/dual-expert-gjd-mode-token \
  --checkpoint-root runs/dual-expert-gjd-mode-token/checkpoints/checkpoint_step_N
```

## Offline Evaluation

Use eval configs for dataset-level metrics:

```bash
uv run --extra eval open-wam-eval \
  --cfg configs/evals/<evaluation>.yaml \
  --device cuda:0 \
  --max-batches 16 \
  --output-json outputs/evaluation.json
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

The maintained dual-expert evaluator loads one checkpoint and executes a closed-loop
episode with the LingBot streaming VAE:

```bash
uv run --extra sim python scripts/run_libero_dual_expert_visualization.py \
  --cfg /path/to/checkpoint_step_N/resolved_config.yaml \
  --checkpoint /path/to/checkpoint_step_N \
  --frontend-encode-mode lingbot_streaming_vae \
  --dual-expert-inference-window-size 30 \
  --benchmark libero_10 \
  --task-id 0 \
  --episode-idx 0 \
  --max-timestep 800 \
  --max-chunks 50 \
  --startup-model-obs-frames 1 \
  --startup-env-init-steps 5 \
  --output-dir outputs/libero_dual_expert \
  --runtime-device cuda:0 \
  --action-device cuda:0 \
  --frontend-device cuda:0 \
  --decode-device cuda:0
```

The strict non-GJD dual-expert contract uses an `800` timestep and `50` chunk limit.
GJD uses the wrapper and defaults to `1500/100`:

```bash
bash scripts/run_gjd_libero.sh rollout \
  --architecture dual_expert \
  --ablation mode_token \
  --checkpoint /path/to/checkpoint_step_N \
  --task-id 0 \
  --episode-idx 0 \
  --output-dir outputs/libero_gjd
```

For a task/episode sweep, use the loaded-once batch evaluator rather than a
shell loop that reloads the checkpoint for every episode:

```bash
uv run --extra sim python scripts/run_libero_dual_expert_batch_visualization.py \
  --cfg /path/to/checkpoint_step_N/resolved_config.yaml \
  --checkpoint /path/to/checkpoint_step_N \
  --benchmark libero_10 \
  --task-ids 0-9 \
  --episode-idxs 0-49 \
  --seed-by-episode \
  --frontend-encode-mode lingbot_streaming_vae \
  --dual-expert-inference-window-size 30 \
  --max-timestep 800 \
  --max-chunks 50 \
  --startup-model-obs-frames 1 \
  --startup-env-init-steps 5 \
  --output-dir outputs/libero_dual_expert_batch \
  --save-rollout-video \
  --runtime-device cuda:0 \
  --action-device cuda:0 \
  --frontend-device cuda:0 \
  --decode-device cuda:0
```

The batch runner preserves the single-episode runtime semantics and starts
fresh rollout state for each episode while retaining loaded model resources.

Parallel stream uses the shared realtime sandbox:

```bash
uv run --extra sim python scripts/run_libero_realtime_sandbox.py \
  --cfg /path/to/checkpoint_step_N/resolved_config.yaml \
  --checkpoint /path/to/checkpoint_step_N \
  --task-id 0 \
  --episode-idx 0 \
  --output-dir outputs/libero_parallel_stream
```

## Generic Simulator Rollout

Use the package simulator boundary for a built-in or extension-registered
adapter:

```bash
uv run --extra sim open-wam-sim-rollout \
  --cfg configs/examples/<benchmark>.yaml \
  --benchmark <registered-name> \
  --target-action-hz 10
```

Application-owned adapters add `--extension package.module` and repeatable
`--sim-option KEY=VALUE` arguments. Standard result provenance is written by
default; use `--provenance-mode full` for a checkpoint SHA-256.

Closed-loop reports should record target and achieved action rate, fallback
actions, task and episode identity, exact resolved config, checkpoint, success,
and video path. Never compare realtime results without also fixing planner
mode, diffusion steps, fallback policy, and horizon.

## Numerical Characterization

Changes that can affect model numerics must verify immutable fixtures and
checkpoint-backed goldens rather than regenerate expected values. The exact
six-program dual-expert training, recurrent inference, four-GPU, full-state resume, and GJD
gates are documented in [DualExpert Refactor Characterization](dual_expert_refactor_characterization.md).
