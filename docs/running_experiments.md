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
Only populate keys referenced by the selected config; unrelated placeholders
may remain unchanged.
Validate and inspect a config before allocating a GPU:

```bash
uv run open-wam-validate-config configs/experiments/<experiment>.yaml
uv run open-wam-inspect-config --cfg configs/experiments/<experiment>.yaml
```

## Supported Configs

The generic runtime covers these representative maintained families:

| Architecture / program | Experiment config |
| --- | --- |
| Parallel stream six video/action programs | `parallel_stream_libero_<program>.yaml` |
| Dual expert six video/action programs | `dual_expert_libero_<program>.yaml` |
| GJD, either architecture | `<architecture>_libero_generalist_joint_denoising.yaml` |
| Dual expert conditional FDM/IDM | `dual_expert_libero_conditional_dynamics.yaml` |
| Video-only | `causal_video_prediction_libero_latent_local.yaml` |

This table identifies semantic families, not resource tiers. Use the public
tiny lifecycle in the [quickstart](quickstart.md#complete-cpu-first-run) for a
CPU first run. Full-size configs declare their actual backbone and trainer
settings in YAML; for example, the 30-layer dual-expert LIBERO references are
FSDP workloads characterized with four 48 GB GPUs. The process launcher, not
`trainer.devices`, creates the distributed workers.

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

### LIBERO Policy Planning Default

The six shipped video/action program configs use one shared full-trajectory
W64 training recipe for both `parallel_stream` and `dual_expert`. Architecture
selects model execution; it does not select the data recipe.

| Setting | Shipped value |
| --- | --- |
| Replay rows | `include_all`; no replay-status validation split |
| Segment sampler | `uniform_segment`, `1000/1000`, full segments only |
| Draw order | replacement, with uniform task/demo/trajectory powers |
| Packed geometry | chunk upper bound 4, window upper bound 64, randomized |
| Sequence semantics | `legacy_prefix_single_frame_perchunk_proprio` |
| Optimization budget | 10,000 steps, no sample-loss reweighting |

With randomized geometry, each sample draws a chunk size from `[1, 4]` and a
window size from `[4, 64]`. The sequence contract supplies the single-frame
condition latent, video-only history visibility, per-chunk additive proprio,
and prefix alignment. Do not repeat those owned fields as individual YAML or
CLI overrides; the config loader rejects ambiguous combinations.

This default follows a matched 10,000-step VTA study. On the common task subset
`{2,3,6,7,8,9}`, the full-trajectory bundle scored 94.1%, compared with
68.0-86.0% for three fixed-128 controls; its all-task score was 95.0% over 282
rollouts. The study used one training seed and changed the sampler/replay
bundle together, so it establishes the default recipe, not a causal claim for
any one field. The historical Parallel Stream exact-backend profiles remain
available for checkpoint reproduction, but they are not the default policy
program recipe.

These are YAML defaults, not runtime invariants. The generic config, data, and
policy layers neither recognize this recipe by name nor reject another
structurally valid combination. Override individual fields with `--set` or
ship another experiment YAML for an ablation.

GJD `real_joint` uses this same planning recipe and sequence contract. GJD-only
fields such as mode probabilities, timestep coupling, counterfactual source
weights, and total optimization budget remain method knobs. Conditional FDM
and IDM rows do not inherit long-horizon planning context: they keep their
target-only singleton-`t0`, text-dropped, one-video-boundary history contract.

`replay_status_policy: include_all` must remain paired with
`val_replay_status_policy: null`, `require_replay_status: false`, and
`val_require_replay_status: false`. Reusing the former failure-only validation
split would make that split empty because the failed rows are now in training.

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

Installed-package GJD training uses the generic `open-wam-train` entry point.
Parallel Stream and Dual Expert are model architectures under the same GJD
paradigm. Source checkouts also provide
`scripts/run_gjd_libero.sh` as a convenience for expanding named ablations and
launching LIBERO rollouts. The script is not part of the wheel or source
distribution and does not own model semantics.

Choose the interface by experiment intent:

| Goal | Interface | Live simulator policy? |
| --- | --- | --- |
| Compare pure FDM or IDM within a GJD study | GJD config with pure FDM/IDM overrides | No; use offline diagnostics |
| Train a run whose primary identity is FDM or IDM | `forward_dynamics` or `inverse_dynamics` | No; use offline diagnostics |
| Produce online robot actions | VTA, ATV, joint, decoupled, or a noisy-condition program | Yes |

The first two choices execute the same conditional tensor contract. They differ
only in experiment identity and tracking: a GJD run records an ablation, while
the standalone config records a fixed program.

```bash
open-wam-train \
  --config-name dual_expert_libero_generalist_joint_denoising \
  --save-root runs/dual-expert-gjd-mode-token \
  --dataset-root /path/to/libero_10 \
  --transformer-subdir /path/to/base/transformer \
  --enable-wandb \
  --wandb-project openwam-gjd \
  --set policy_variant.generalist_mode_text_token=true
```

The canonical config is vanilla GJD: joint/FDM/IDM mode probabilities are
`0.6/0.2/0.2`, the five-bucket real/counterfactual mixer is active, and the
mode token is disabled. Setting `generalist_mode_text_token=true` produces the
mode-token variant above. Pure-joint sets probabilities to `1/0/0` and
`generalist_training_paradigm=demo_only`. Pure-FDM and pure-IDM use only the
matching real-demo and counterfactual buckets; their weights are a relative
source ratio. For example, pure FDM with a `3:1` real-to-counterfactual ratio is:

```bash
FDM_PROBS='{joint: 0, action_conditioned_video: 1, video_conditioned_action: 0}'
open-wam-train \
  --config-name dual_expert_libero_generalist_joint_denoising \
  --save-root runs/dual-expert-gjd-pure-fdm \
  --set "policy_variant.generalist_denoising_mode_probs=${FDM_PROBS}" \
  --set policy_variant.generalist_mode_text_token=false \
  --set policy_variant.generalist_training_paradigm=dynamics_routed \
  --set data.generalist_dynamics_mixture.real_joint_weight=0 \
  --set data.generalist_dynamics_mixture.real_action_conditioned_video_weight=3 \
  --set data.generalist_dynamics_mixture.real_video_conditioned_action_weight=0 \
  --set data.generalist_dynamics_mixture.counterfactual_action_conditioned_video_weight=1 \
  --set data.generalist_dynamics_mixture.counterfactual_video_conditioned_action_weight=0
```

Under replacement sampling, the example draws real and counterfactual FDM
samples with a `3:1` ratio in expectation. Weights need not sum to one. Either
endpoint may be zero, but not both. A zero counterfactual weight does not
require encoded counterfactual roots. Configure positive-weight counterfactual
train and validation roots in
`configs/local_paths.yaml` or with explicit `--set` overrides.

The public `generalist_training_paradigm: dynamics_routed` setting selects this
source router and target-only data adapter. It does not mean that
counterfactual data must be active: source weights decide whether a run uses
real demonstrations, counterfactual rows, or both. The legacy input value
`mixed_dynamics` is accepted for old resolved configs and is normalized to
`dynamics_routed`; new configs and commands should use the canonical name.

Resume through the same package entry point and repeat the experiment-defining
overrides (the checkpoint-local resolved config remains the audit record):

```bash
open-wam-train \
  --config-name dual_expert_libero_generalist_joint_denoising \
  --save-root runs/dual-expert-gjd-mode-token \
  --checkpoint-root runs/dual-expert-gjd-mode-token/checkpoints/checkpoint_step_N \
  --set policy_variant.generalist_mode_text_token=true
```

### Standalone Conditional FDM And IDM

Use a fixed program when the entire experiment is conditional dynamics rather
than a GJD ablation. Both programs use the same model, attention, data
projection, scheduler, loss masks, and gradients as the corresponding 100%
GJD mode:

#### Data Prerequisites

| Training path | Required data |
| --- | --- |
| Standard VTA, ATV, joint, decoupled, or noisy-condition policy | The normal encoded demonstration root only; no counterfactual root is used |
| Conditional FDM/IDM with the default `1:1` ratio | The normal encoded demonstration root plus encoded counterfactual train and validation roots |
| Conditional FDM/IDM with counterfactual weight `0` | The normal encoded demonstration root only; counterfactual roots are not opened |

Counterfactual roots are not arbitrary videos. They must follow the encoded
target-only `t0`-plus-future contract: an observed `t0`, aligned future video
and action data, valid loss-boundary metadata, and aligned proprio when it is
available. The loader validates this contract before model execution. Set the
two roots through `configs/local_paths.yaml` or explicit `--set` overrides.

Counterfactual data is therefore required by the default config and for any
counterfactual experiment, but not by the conditional objective itself. A
real-demo-only FDM run sets
`counterfactual_action_conditioned_video_weight=0`; real-demo-only IDM sets
`counterfactual_video_conditioned_action_weight=0`.

To produce LIBERO counterfactual roots from a source checkout, first install
the `sim` dependencies and configure the local LIBERO repository. The generator
selects only successful rows from the replay-status file described in
[LIBERO Training Prerequisites](benchmarks.md#libero-training-prerequisites).
The following creates a 10,000-row train split and a disjoint 1,000-row
validation split using the maintained branch and random-`t0` defaults:

```bash
CF_ROOT=/path/to/counterfactual_dynamics
REPLAY_STATUS=/path/to/replay_status.jsonl

uv run --extra sim python scripts/build_libero_fdm_counterfactual_demo_dataset.py \
  --replay-status-path "$REPLAY_STATUS" \
  --output-dir "$CF_ROOT" --run-id train \
  --target-transitions 10000

uv run --extra sim python scripts/build_libero_fdm_counterfactual_demo_dataset.py \
  --replay-status-path "$REPLAY_STATUS" \
  --output-dir "$CF_ROOT" --run-id val \
  --target-transitions 1000 --episodes-per-task 5 \
  --t0-samples-per-episode 2 \
  --exclude-source-dataset-root "$CF_ROOT/train" --seed 1
```

Encode both raw roots with the same VAE-capable checkpoint and config used by
training:

```bash
VAE_CHECKPOINT=/path/to/checkpoint_step_N
for SPLIT in train val; do
  uv run --extra eval python scripts/encode_libero_fdm_counterfactual_dataset.py \
    --dataset-root "$CF_ROOT/$SPLIT" \
    --config configs/experiments/dual_expert_libero_conditional_dynamics.yaml \
    --checkpoint "$VAE_CHECKPOINT" --device cuda:0
done
```

Set the train and validation registry keys to
`$CF_ROOT/train/encoded_latents` and `$CF_ROOT/val/encoded_latents`. Both tools
fail on existing outputs unless replacement is requested explicitly; use their
`--help` output for sharding and larger production runs.

```bash
# FDM defaults to a 1:1 real/counterfactual source ratio.
uv run --extra train open-wam-train \
  --cfg configs/experiments/dual_expert_libero_conditional_dynamics.yaml \
  --save-root runs/dual-expert-forward-dynamics

# IDM uses the same config with one program override and the same 1:1 default.
uv run --extra train open-wam-train \
  --cfg configs/experiments/dual_expert_libero_conditional_dynamics.yaml \
  --set policy_variant.program=inverse_dynamics \
  --save-root runs/dual-expert-inverse-dynamics
```

For a non-default ratio, add the two matching source overrides. For example,
FDM `3:1` uses:

```bash
--set data.generalist_dynamics_mixture.real_action_conditioned_video_weight=3 \
--set data.generalist_dynamics_mixture.counterfactual_action_conditioned_video_weight=1
```

For IDM, use the corresponding `real_video_conditioned_action_weight` and
`counterfactual_video_conditioned_action_weight` fields.

The fixed program filters out every irrelevant mixture bucket, so the unused
pair of weights in the shared YAML has no effect. In both programs the data
layer projects real and counterfactual rows to the same target-only layout:
`V0` is an observed, unsupervised singleton t0 chunk; loss starts at `V1`; each
following chunk retains the sampled 1-4 frame geometry and can attend one most
recent clean video/proprio boundary frame. Task text and the learned GJD mode
token are absent. FDM masks action loss; IDM masks video loss.

Standalone conditional FDM/IDM inference is offline: FDM needs future clean
actions and IDM needs future clean video. The LIBERO live simulator runner
rejects these programs instead of silently substituting joint rollout. Use the
offline FDM/IDM diagnostics below with explicit condition tensors.

For FDM metrics, each evaluation row must provide `t0`, clean future actions,
and target future video. For IDM metrics, it must provide `t0`, clean future
video, and target actions. Real-demo diagnostics can project ordinary demo rows;
counterfactual metrics require the encoded counterfactual rows described above.

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
  --cfg /path/to/checkpoint_step_N/resolved_config.yaml \
  --checkpoint /path/to/checkpoint_step_N \
  --dataset-root /path/to/libero_10

uv run --extra sim python scripts/run_joint_denoising_fdm_counterfactual.py \
  --checkpoint /path/to/checkpoint_step_N \
  --replay-status-path /path/to/replay_status.jsonl \
  --episode-indices 0,1,2 \
  --dataset-root /path/to/libero_10
```

When `--mode` is omitted, a `forward_dynamics` checkpoint runs only
`forced_action_joint_fdm`, and an `inverse_dynamics` checkpoint runs only
`video_conditioned_action`. An explicitly incompatible mode fails before data
or model loading. A non-fixed GJD config retains the research behavior of
running every diagnostic mode; use repeated `--mode` arguments to narrow it.
Use `--help` for windows, branches, and output paths. The reusable GJD policy
and counterfactual-action contracts continue to live in `open_wam`; only the
experiment orchestration and visualization live under `scripts/`.

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

The six standard non-GJD dual-expert rollout programs use an `800` timestep and
`50` chunk limit. Conditional FDM/IDM is offline-only and does not use this live
rollout command. GJD LIBERO rollout is a source-checkout integration and
defaults to `1500/100`:

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
