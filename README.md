# Open-WAM

**OpenWAM Team, Stanford University**

[![CI](https://github.com/DaivdYuan/OpenWAM-staging-public/actions/workflows/ci.yml/badge.svg)](https://github.com/DaivdYuan/OpenWAM-staging-public/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://daivdyuan.github.io/OpenWAM-staging-public/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](CHANGELOG.md)

**Open-WAM is a research framework for training, comparing, and evaluating
video-action world models for robot learning.** It separates model topology,
video/action conditioning, sequence semantics, visual execution, and action
decoding so that controlled experiments share the same trainer and visual
stack.

[Documentation](https://daivdyuan.github.io/OpenWAM-staging-public/) |
[Quickstart](docs/quickstart.md) |
[Methods](docs/policy_architectures.md) |
[Training and evaluation](docs/running_experiments.md) |
[Extension SDK](docs/extension_sdk.md) |
[Results](docs/m5_gjd_uva_libero10_comparison.md) |
[Citation](#citation)

> **Release status:** Open-WAM 0.1.0 is pre-release Linux research software.
> The public CPU lifecycle and synthetic artifacts are self-contained. Large
> benchmark runs use separately provisioned datasets and checkpoints described
> by the [artifact contract](docs/artifacts.md).

Development began in March 2026. This repository preserves the original commit
ordering, dates, and contributor attribution. Before public distribution, the
history was rewritten to remove private infrastructure paths, operational
artifacts, and private run URLs; commit hashes therefore differ from the
internal development repository. No commits were backdated.

## Research Scope

Open-WAM provides:

- one typed train, resume, evaluation, and simulator runtime across policy
  architectures;
- six standard video/action programs plus generalist joint denoising (GJD);
- full-state checkpoint continuation and versioned run provenance;
- adapters for LIBERO, RoboTwin, CALVIN, heterogeneous LeRobot data, and
  synthetic fixtures;
- role-scoped extension APIs for datasets, policies, decoders, attention
  profiles, and simulators; and
- CPU semantic tests plus opt-in real-checkpoint GPU parity gates for changes
  near model numerics.

### Maintained Methods

| Architecture | Topology | Maintained programs |
| --- | --- | --- |
| `parallel_stream` | Video and action tokens share one transformer. | Six standard programs, GJD, and standalone conditional FDM/IDM through the exact LingBot-compatible runtime. |
| `dual_expert` | Video and action use separate transformer experts. | Six standard programs, GJD, and standalone conditional FDM/IDM. |
| `causal_video_prediction` | The visual model runs without action supervision. | Video-only prediction. |

The six standard program selectors are `video_then_action`,
`action_then_video`, `joint`, `decoupled_same_step`,
`video_noisy_to_action`, and `action_noisy_to_video`. GJD samples joint,
forward-dynamics (FDM), and inverse-dynamics (IDM) submodes within one model.
Standalone `forward_dynamics` and `inverse_dynamics` preserve the strict GJD
conditional contract: one clean t0 latent in a singleton chunk, one-frame
conditional history, no task text, and only the matching prediction loss.

Historical M1, M5, MoT, `mot`, and `*_heng_compatible` names remain accepted
at compatibility boundaries. New experiments use architecture and program
names.

## Quick Start

Open-WAM supports Linux with Python 3.11 or 3.12. Install
[`uv`](https://docs.astral.sh/uv/), then run the public CPU contract:

```bash
git clone https://github.com/DaivdYuan/OpenWAM-staging-public.git Open-WAM
cd Open-WAM

uv sync --frozen --group dev --extra train --extra eval

uv run open-wam-validate-config \
  configs/examples/public_tiny_synthetic_contract.yaml \
  configs/evals/public_tiny_synthetic_contract.yaml

uv run --extra train open-wam-sanity \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --device cpu --max-batches 1 --rollout-steps 1
```

This path requires no private data, checkpoint, GPU, or external simulator. It
checks config loading, dataset construction, a train forward pass, batch
inference, and recurrent rollout-style inference. The
[complete CPU first run](docs/quickstart.md#complete-cpu-first-run) adds exact
resume and checkpoint-backed evaluation.

Install only the runtime needed for later work:

| Task | Command |
| --- | --- |
| Config and metadata development | `uv sync --group dev` |
| Training | `uv sync --extra train` |
| Offline evaluation | `uv sync --extra eval` |
| Model-driven simulator rollout | `uv sync --extra sim` |
| Documentation | `uv sync --extra docs` |

Benchmark extras supply dependency overlays, not upstream source trees. Follow
[Benchmarks and Data](docs/benchmarks.md) before a real simulator run.

## Training

Real datasets, checkpoints, simulator checkouts, and output directories remain
outside versioned experiment YAML. Start with the local path registry:

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
uv run open-wam-inspect-config \
  --cfg configs/experiments/dual_expert_libero_joint.yaml
```

Populate only the aliases used by the selected config. The local registry is
gitignored; set `OPEN_WAM_LOCAL_PATHS=/absolute/path/paths.yaml` to keep it
elsewhere.

All architectures use `open-wam-train`. The shipped Parallel Stream and Dual
Expert LIBERO policy programs use the same validated full-trajectory W64 recipe
described in [Training and Inference](docs/running_experiments.md#libero-policy-planning-default).
The reference 30-layer configs are FSDP workloads characterized with four 48 GB GPUs:

```bash
uv run --extra train torchrun --standalone --nproc-per-node=4 \
  -m open_wam.cli.train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --save-root runs/dual-expert-joint \
  --expected-world-size 4
```

For a one-off method ablation, change the public program selector rather than
the trainer:

```bash
uv run --extra train torchrun --standalone --nproc-per-node=4 \
  -m open_wam.cli.train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --set policy_variant.program=video_then_action \
  --save-root runs/dual-expert-vta \
  --expected-world-size 4
```

Resume from a full training-state checkpoint with the same command and
`--resume-from`:

```bash
uv run --extra train torchrun --standalone --nproc-per-node=4 \
  -m open_wam.cli.train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --save-root runs/dual-expert-joint \
  --resume-from runs/dual-expert-joint/checkpoints/checkpoint_step_N \
  --expected-world-size 4
```

`--resume-from` requires `full_training_state.pt` and restores model, optimizer,
scheduler, strategy/scaler, step state, and the next sampler epoch/batch cursor.
Resumable checkpoints are written only at optimizer boundaries because partial
gradients are not serialized. Exact loader-cursor continuation also requires a
sized training dataloader. Process and stochastic dataset/worker RNG streams are
not checkpointed, so a restarted run is not bitwise identical. Use
`--initialize-weights-from` for a fresh run initialized from model weights. The
removed ambiguous `--checkpoint-root` operation always errors. Every checkpoint
stores its resolved config as an audit record; it is not merged into the
invocation config.

Conditional FDM/IDM uses the dynamics-routing data adapter. The maintained
config mixes real demonstrations with encoded counterfactual train and
validation roots; a real-demo-only ablation is also supported. Read the
[data prerequisites](docs/running_experiments.md#data-prerequisites) before
selecting `forward_dynamics` or `inverse_dynamics`.

## Evaluation And Rollout

Run offline metrics through the generic evaluator:

```bash
uv run --extra eval open-wam-eval \
  --cfg configs/evals/dual_expert_robotwin_smoke_eval.yaml \
  --checkpoint /path/to/model_state.pt \
  --device cuda:0
```

Run a configured environment through the simulator boundary:

```bash
uv run --extra sim open-wam-sim-rollout \
  --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml \
  --checkpoint /path/to/model_state.pt \
  --benchmark robotwin \
  --robotwin-task-name <task-name>
```

Benchmark adapters translate observations and actions. Sequence, attention,
cache, and denoising semantics remain owned by the selected policy. Maintained
LIBERO and GJD commands are listed in
[Training and Inference](docs/running_experiments.md).

## Recorded Results

The repository records the following LIBERO-10 rollout result for a historical
M5-labelled, canonical `dual_expert` GJD mode-token checkpoint at step 40,000:

| System | Task-aligned episodes 0-4 | Full success@1 run |
| --- | ---: | ---: |
| Open-WAM dual-expert GJD | 45/50 (90.0%) | 461/500 (92.2%) |
| Released UVA LIBERO baseline | 38/50 (76.0%) | not run |

The 50-rollout comparison is task-aligned. Offline FDM/IDM measurements use
each system's native image, target, action, and controller contracts and are
not direct scalar rankings. Read the
[result card](docs/m5_gjd_uva_libero10_comparison.md) for protocol details and
limitations. Until the real Open-WAM checkpoint entry has a public URL,
checksum, and license, these numbers are a recorded result rather than a
turnkey public reproduction claim.

## Architecture

Every built-in method follows one composition boundary:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

| Contract | Responsibility |
| --- | --- |
| `ExperimentConfig` | Typed architecture, program, data, sequence, runtime, and optimization choices. |
| `VariantPipeline` | Shared training and inference orchestration. |
| `VisualTower` | Frontend encoding, visual backbone execution, decode stages, and runtime hooks. |
| `PolicyVariant` | Parameter topology, architecture-specific packing, conditioning adapters, and recurrent state. |
| `ActionDecoder` | Final supervised outputs, masks, losses, metrics, and committed actions. |

This boundary keeps the visual stack stable while experiments vary one owned
contract at a time. See [Architecture](docs/architecture.md) and
[Policy Architectures and Programs](docs/policy_architectures.md).

## Use Open-WAM With Your System

Out-of-tree packages load through repeatable `--extension module[:hook]`
arguments. Choose the smallest owning boundary:

| Customization | Extension surface |
| --- | --- |
| Storage format, camera schema, or action/state representation | Dataset adapter selected by `data.dataset_type` |
| Learned parameters, conditioning, attention profile, or recurrent state | `PolicyVariant` |
| Final outputs, loss, sampling, or committed action count | `ActionDecoder` |
| Environment construction and observation/action translation | Simulator adapter |
| Existing method, geometry, schedule, cache, or optimizer choice | YAML only |

The packaged extension scaffold verifies registration, gradients, inference
state, and packaging before custom code is introduced:

```bash
uv run --extra train open-wam-train \
  --cfg templates/extension_method/config.yaml \
  --extension open_wam.templates.extension_method \
  --save-root runs/extension-method-smoke \
  --disable-wandb
```

Extensions import compatibility-managed contracts from the role-specific
`open_wam.sdk` modules. See the [Extension SDK](docs/extension_sdk.md) and
[cookbooks](docs/cookbooks/new_policy_architecture.md).

## Reproducibility

Evaluation, sanity, and simulator commands can emit the same versioned result
envelope with source state, exact argv, config hashes, checkpoint identity,
dataset metadata, package versions, and device details. Use full provenance to
hash a publication checkpoint:

```bash
open-wam-eval --cfg evaluation.yaml --output-json result.json \
  --provenance-mode full
```

Exact numerical claims use the locked dependency graph and documented
hardware/software stack. A refactor near model execution must pass immutable
training-step, recurrent-inference, cache-rollover, and full-state-resume
characterization; expected values are not regenerated by the refactor. See
[Reproducibility](docs/reproducibility.md),
[Compatibility](docs/compatibility.md), and [Testing](docs/testing.md).

## Repository Layout

```text
configs/       typed experiments, evaluations, examples, and path templates
docs/          public guides, experiment cards, and extension cookbooks
scripts/       thin benchmark adapters and checkout-only research tools
src/open_wam/  installable library and role-scoped SDK
tests/         unit, integration, simulator, and numerical parity gates
notes/index/   generated public consortium metadata packaged at runtime
```

## Documentation

| Topic | Guide |
| --- | --- |
| Install and first run | [Quickstart](docs/quickstart.md) |
| Runtime ownership | [Architecture](docs/architecture.md) |
| Architectures and programs | [Policy Architectures](docs/policy_architectures.md) |
| Train, resume, evaluate, and roll out | [Training and Inference](docs/running_experiments.md) |
| Dataset and simulator setup | [Benchmarks and Data](docs/benchmarks.md) |
| Custom datasets, policies, decoders, and simulators | [Extension SDK](docs/extension_sdk.md) |
| Checkpoints and manifests | [Artifacts](docs/artifacts.md) |
| Test and parity tiers | [Testing](docs/testing.md) |

## Contributing

Contributions should preserve the typed runtime boundary and add focused tests
for every changed contract. Read [CONTRIBUTING.md](CONTRIBUTING.md), the
[Code of Conduct](CODE_OF_CONDUCT.md), and the
[Security Policy](SECURITY.md) before opening a pull request.

## Citation

If Open-WAM supports your research, cite the software record in
[`CITATION.cff`](CITATION.cff):

```bibtex
@software{open_wam_2026,
  title   = {Open-WAM},
  author  = {{OpenWAM Team, Stanford University}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/DaivdYuan/OpenWAM-staging-public}
}
```

## License

Open-WAM is released under the [GNU Affero General Public License v3.0](LICENSE)
with the redistribution attribution described in [`NOTICE`](NOTICE). Covered
modified versions and network services must provide corresponding source, and
redistributed copies must preserve the Open-WAM attribution notice. Academic
work that uses Open-WAM should cite the software record in
[`CITATION.cff`](CITATION.cff).

Third-party components retain their own terms; the adapted LingBot-VA module
is distributed under Apache License 2.0. Full attributions are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).
