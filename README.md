# Open-WAM

Open-WAM is a typed, extensible library for training and evaluating world
action models. It keeps visual execution stable while experiments select an
explicit policy architecture, video/action program, sequence contract,
attention runtime, cache policy, and action decoder.

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

Applications can add dataset adapters, policy architectures, action decoders,
and simulator backends without adding another trainer or copying the visual
backbone.

## What Is Maintained

- `parallel_stream`: shared-transformer video/action execution, including the
  exact LingBot-compatible backend.
- `dual_expert`: separate video and action experts with VTA, ATV, joint,
  decoupled, noisy-condition, and GJD programs.
- `post_latent` and `post_decoded`: feature-attached action baselines.
- `causal_video_prediction`: video-only prediction.
- One typed config loader, composable training runtime, checkpoint lifecycle,
  evaluator, and simulator boundary.
- Dataset adapters for public fixtures and local LIBERO, RoboTwin, CALVIN, and
  heterogeneous LeRobot inputs.

M1, M5, MoT, `mot`, and `*_heng_compatible` are compatibility labels. New
configs and code use architecture/program names.

## Core Concepts

Open-WAM treats these as orthogonal contracts:

| Contract | Responsibility |
| --- | --- |
| Architecture | Parameter topology and token execution (`parallel_stream`, `dual_expert`) |
| Program | Video/action conditioning and supervision (VTA, ATV, joint, GJD, and others) |
| Sequence semantics | Prefix, history visibility, chunk geometry, and alignment |
| Runtime backend | Packed execution, attention backend, cache writes, and denoising |
| Action decoder | Final outputs, masks, metrics, and supervised losses |

For the six standard multimodal programs,
`policy_variant.program` is the public switch. Low-level coupling values are
derived and validated at the typed config boundary.

## Install

The base package intentionally excludes heavyweight model and simulator
dependencies:

```bash
uv sync --group dev
uv run python -c "import open_wam; print(open_wam.__version__)"
```

Install only the runtime needed by the task:

```bash
uv sync --extra train
uv sync --extra eval
uv sync --extra sim
uv sync --extra docs
```

`sim` includes the model-facing simulator stack. Upstream benchmark source
checkouts such as LIBERO are installed separately.

## First Checks

Validate a config without allocating a GPU:

```bash
uv run open-wam-validate-config \
  configs/examples/public_tiny_synthetic_contract.yaml

uv run open-wam-inspect-config \
  --cfg configs/experiments/dual_expert_libero_joint.yaml
```

Run the public CPU-safe contract:

```bash
uv run --extra train open-wam-sanity \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --device cpu --max-batches 1 --rollout-steps 1
```

## Local Assets

Datasets, checkpoints, output roots, and simulator checkouts stay outside
tracked experiment YAML:

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
```

Replace placeholders in `configs/local_paths.yaml`; the file is gitignored.
Select another registry with `OPEN_WAM_LOCAL_PATHS=/absolute/path/paths.yaml`.

## Train And Resume

All policy architectures enter the same package-owned trainer:

```bash
uv run --extra train open-wam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --save-root runs/dual-expert-joint \
  --expected-world-size 1
```

Change only the program for a one-off ablation:

```bash
uv run --extra train open-wam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --set policy_variant.program=video_then_action \
  --save-root runs/dual-expert-vta
```

Resume from a checkpoint directory:

```bash
uv run --extra train open-wam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --save-root runs/dual-expert-joint \
  --checkpoint-root runs/dual-expert-joint/checkpoints/checkpoint_step_N
```

Exact resume requires `full_training_state.pt`; model-only state is a warm
start. The resolved config stored with every checkpoint is part of the
reproducibility contract.

## Evaluate And Roll Out

Use the generic evaluator for dataset metrics:

```bash
uv run --extra eval open-wam-eval \
  --cfg configs/evals/dual_expert_robotwin_smoke_eval.yaml \
  --device cuda:0
```

Use the package simulator boundary for configured benchmark adapters:

```bash
uv run --extra sim open-wam-sim-rollout \
  --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml \
  --benchmark robotwin
```

Maintained LIBERO and GJD checkpoint commands are documented in
[Training And Inference](docs/running_experiments.md). Benchmark adapters
translate observations and actions; policy sequence semantics remain in the
selected `PolicyVariant`.

## Extend

Out-of-tree packages register through `--extension module[:hook]`. Choose the
smallest extension surface:

- dataset-specific parsing: dataset adapter selected by `data.dataset_type`
- new output/loss: `ActionDecoder`
- new attention/conditioning semantics on an existing topology: runtime program
- new parameter topology or recurrent-state owner: `PolicyVariant`
- new environment: simulator adapter

Use the role-specific `open_wam.sdk` modules described in the
[Extension SDK](docs/extension_sdk.md). See the
[compatibility matrix](docs/compatibility.md) for the maintained Python,
dependency, and numerical-validation surfaces.

## Repository Layout

```text
configs/       typed experiment, evaluation, and local-path templates
docs/          public user and contributor documentation
notes/         current engineering contracts and archived roadmaps
scripts/       thin benchmark adapters and checkout-only research tools
src/open_wam/  installable library
tests/         unit, integration, simulator, and numerical parity gates
deployment/    separately tested hardware operations workspace
```

Important packages:

- `src/open_wam/configs`: typed loading, overrides, validation, and compatibility
- `src/open_wam/data`: adapters, sampling, canonical view assembly, and transforms
- `src/open_wam/models/visual_tower`: shared visual execution and runtime hooks
- `src/open_wam/models/policy_variants`: architecture-owned train/infer semantics
- `src/open_wam/models/action_decoders`: final outputs and losses
- `src/open_wam/pipelines`: shared composition boundary
- `src/open_wam/training`: generic optimization, logging, validation, and checkpoints
- `src/open_wam/evals`: generic evaluation and benchmark rollout support

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Policy Architectures And Programs](docs/policy_architectures.md)
- [Training And Inference](docs/running_experiments.md)
- [Benchmarks And Data](docs/benchmarks.md)
- [Extension SDK](docs/extension_sdk.md)
- [Testing](docs/testing.md)
- [Artifacts](docs/artifacts.md)
- [Reproducibility](docs/reproducibility.md)

## Validation Policy

Public CI checks dependency-light imports, config schema, docs, and package
surface. Local CPU tests cover typed config, data, factories, training,
checkpointing, and synthetic inference. Changes near model numerics must also
pass immutable real-checkpoint training, recurrent inference, cache-rollover,
and full-state-resume characterization; expected values are never regenerated
by a refactor.

See [Testing](docs/testing.md) and
[Dual-Expert Refactor Characterization](docs/dual_expert_refactor_characterization.md).
