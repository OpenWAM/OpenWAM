<h1 align="center">OpenWAM</h1>

<p align="center"><strong>An extensible framework for video-action world models in robot learning</strong></p>

<p align="center">
  <a href="https://github.com/OpenWAM/OpenWAM/actions/workflows/ci.yml"><img src="https://github.com/OpenWAM/OpenWAM/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://openwam.github.io/OpenWAM/"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Documentation"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg" alt="Python 3.11 or 3.12"></a>
  <a href="https://github.com/OpenWAM/OpenWAM/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License: AGPL v3"></a>
</p>

<p align="center">
  <a href="https://openwam.github.io/OpenWAM/">Documentation</a> &middot;
  <a href="https://github.com/OpenWAM/OpenWAM/blob/main/docs/quickstart.md">Quickstart</a> &middot;
  <a href="https://github.com/OpenWAM/OpenWAM/blob/main/docs/policy_architectures.md">Methods</a> &middot;
  <a href="https://github.com/OpenWAM/OpenWAM/blob/main/docs/running_experiments.md">Training and evaluation</a> &middot;
  <a href="https://github.com/OpenWAM/OpenWAM/blob/main/docs/extension_sdk.md">Extension SDK</a> &middot;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  Developed by the <strong>OpenWAM Team</strong> at the
  <a href="https://svl.stanford.edu/"><strong>Stanford Vision and Learning Lab (SVL)</strong></a>.
</p>

<p align="center">
  <a href="https://www.stanford.edu/"><img src="https://raw.githubusercontent.com/OpenWAM/OpenWAM/main/docs/assets/affiliations/stanford-wordmark.png" alt="Stanford University" height="20" valign="middle"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://ai.stanford.edu/"><img src="https://raw.githubusercontent.com/OpenWAM/OpenWAM/main/docs/assets/affiliations/stanford-ai-lab.jpg" alt="Stanford Artificial Intelligence Laboratory" height="28" valign="middle"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://svl.stanford.edu/"><img src="https://raw.githubusercontent.com/OpenWAM/OpenWAM/main/docs/assets/affiliations/stanford-svl.png" alt="Stanford Vision and Learning Lab" height="28" valign="middle"></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://src.stanford.edu/"><img src="https://images.squarespace-cdn.com/content/v1/66b6b61fc5e5030973bd431f/01e83141-2062-49fa-8043-730d17b75cca/SRClogo.png" alt="Stanford Robotics Center" height="32" valign="middle"></a>
</p>

OpenWAM separates model topology, video/action conditioning, sequence
semantics, visual execution, and action decoding so that controlled experiments
share the same trainer and visual stack.

> **Release status:** OpenWAM 0.1.1 is alpha-stage Linux research software.
> The public CPU lifecycle and synthetic artifacts are self-contained. Large
> benchmark runs use separately provisioned datasets and checkpoints described
> by the [artifact contract](https://github.com/OpenWAM/OpenWAM/blob/main/docs/artifacts.md).

## Upcoming Research Release

Detailed evaluation results, trained model checkpoints, datasets, and the
OpenWAM research paper are being prepared for public release and will be
available very soon. Canonical links and integrity metadata will be added to
the [artifact documentation](https://github.com/OpenWAM/OpenWAM/blob/main/docs/artifacts.md) as each resource is published.

## Research Scope

For OpenWAM video pretraining, see the
[pretraining datasets and workflow](https://github.com/OpenWAM/OpenWAM/blob/main/docs/pretraining/index.md). The guide covers
the nine pretraining data sources, downloads, RGB multi-view composition before
VAE encoding, task text, verified object storage with a bounded cache, training,
checkpoints, and inference. These inputs are the pretraining corpus; downstream
robot policy fine-tuning and evaluation have their own dataset configurations.
Published model weights are available at
[OpenWAM-Stanford/OpenWAM-Pretraining on Hugging Face](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining).

OpenWAM provides:

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

Experiment configs and public commands use architecture and program names
directly.

## Installation

OpenWAM is available on [PyPI](https://pypi.org/project/openwam/) for Linux
with Python 3.11 or 3.12. In a virtual environment:

```bash
python -m pip install openwam
```

The base package provides configuration and metadata APIs. For model training
and evaluation, install the runtime extras:

```bash
python -m pip install 'openwam[train,eval]'
```

Use `openwam[train,pretrain]` for video data preparation and pretraining, or
`openwam[sim]` for model-driven simulator rollouts. Python imports use
`open_wam`. The optional `openwam-sdk` installation alias provides the same
implementation and extras.

The [quickstart](https://github.com/OpenWAM/OpenWAM/blob/main/docs/quickstart.md)
covers installed-package usage. For development or exact dependency
reproduction, use the frozen source checkout below. Model weights, datasets,
and external simulator source trees are provisioned separately.

## Quick Start From Source

OpenWAM supports Linux with Python 3.11 or 3.12. Install
[`uv`](https://docs.astral.sh/uv/), then run the public CPU contract:

```bash
git clone https://github.com/OpenWAM/OpenWAM.git
cd OpenWAM

uv sync --frozen --group dev --extra train --extra eval

uv run openwam-validate-config \
  configs/examples/public_tiny_synthetic_contract.yaml \
  configs/evals/public_tiny_synthetic_contract.yaml

uv run --extra train openwam-sanity \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --device cpu --max-batches 1 --rollout-steps 1
```

This path requires no private data, checkpoint, GPU, or external simulator. It
checks config loading, dataset construction, a train forward pass, batch
inference, and recurrent rollout-style inference. The
[complete CPU first run](https://github.com/OpenWAM/OpenWAM/blob/main/docs/quickstart.md#complete-cpu-first-run) adds stateful
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
[Benchmarks and Data](https://github.com/OpenWAM/OpenWAM/blob/main/docs/benchmarks.md) before a real simulator run.

## Training

Real datasets, checkpoints, simulator checkouts, and output directories remain
outside versioned experiment YAML. Start with the local path registry:

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
uv run openwam-inspect-config \
  --cfg configs/experiments/dual_expert_libero_joint.yaml
```

Populate only the aliases used by the selected config. The local registry is
gitignored; set `OPEN_WAM_LOCAL_PATHS=/absolute/path/paths.yaml` to keep it
elsewhere.

All architectures use `openwam-train`. The shipped Parallel Stream and Dual
Expert LIBERO policy programs use the same validated full-trajectory W64 recipe
described in [Training and Inference](https://github.com/OpenWAM/OpenWAM/blob/main/docs/running_experiments.md#libero-policy-planning-default).
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
[data prerequisites](https://github.com/OpenWAM/OpenWAM/blob/main/docs/running_experiments.md#data-prerequisites) before
selecting `forward_dynamics` or `inverse_dynamics`.

## Evaluation And Rollout

Run offline metrics through the generic evaluator:

```bash
uv run --extra eval openwam-eval \
  --cfg configs/evals/dual_expert_robotwin_smoke_eval.yaml \
  --checkpoint /path/to/model_state.pt \
  --device cuda:0
```

Run a configured environment through the simulator boundary:

```bash
uv run --extra sim openwam-sim-rollout \
  --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml \
  --checkpoint /path/to/model_state.pt \
  --benchmark robotwin \
  --robotwin-task-name <task-name>
```

Benchmark adapters translate observations and actions. Sequence, attention,
cache, and denoising semantics remain owned by the selected policy. Maintained
LIBERO and GJD commands are listed in
[Training and Inference](https://github.com/OpenWAM/OpenWAM/blob/main/docs/running_experiments.md).

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
contract at a time. See [Architecture](https://github.com/OpenWAM/OpenWAM/blob/main/docs/architecture.md) and
[Policy Architectures and Programs](https://github.com/OpenWAM/OpenWAM/blob/main/docs/policy_architectures.md).

## Use OpenWAM With Your System

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
uv run --extra train openwam-train \
  --cfg templates/extension_method/config.yaml \
  --extension open_wam.templates.extension_method \
  --save-root runs/extension-method-smoke \
  --disable-wandb
```

Extensions import compatibility-managed contracts from the role-specific
`open_wam.sdk` modules. See the [Extension SDK](https://github.com/OpenWAM/OpenWAM/blob/main/docs/extension_sdk.md) and
[cookbooks](https://github.com/OpenWAM/OpenWAM/blob/main/docs/cookbooks/new_policy_architecture.md).

## Reproducibility

Evaluation, sanity, and simulator commands can emit the same versioned result
envelope with source state, exact argv, config hashes, checkpoint identity,
dataset metadata, package versions, and device details. Use full provenance to
hash a publication checkpoint:

```bash
openwam-eval --cfg evaluation.yaml --output-json result.json \
  --provenance-mode full
```

Exact numerical claims use the locked dependency graph and documented
hardware/software stack. A refactor near model execution must pass immutable
training-step, recurrent-inference, cache-rollover, and full-state-resume
characterization; expected values are not regenerated by the refactor. See
[Reproducibility](https://github.com/OpenWAM/OpenWAM/blob/main/docs/reproducibility.md),
[Compatibility](https://github.com/OpenWAM/OpenWAM/blob/main/docs/compatibility.md), and [Testing](https://github.com/OpenWAM/OpenWAM/blob/main/docs/testing.md).

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
| Install and first run | [Quickstart](https://github.com/OpenWAM/OpenWAM/blob/main/docs/quickstart.md) |
| Runtime ownership | [Architecture](https://github.com/OpenWAM/OpenWAM/blob/main/docs/architecture.md) |
| Architectures and programs | [Policy Architectures](https://github.com/OpenWAM/OpenWAM/blob/main/docs/policy_architectures.md) |
| Train, resume, evaluate, and roll out | [Training and Inference](https://github.com/OpenWAM/OpenWAM/blob/main/docs/running_experiments.md) |
| Dataset and simulator setup | [Benchmarks and Data](https://github.com/OpenWAM/OpenWAM/blob/main/docs/benchmarks.md) |
| Custom datasets, policies, decoders, and simulators | [Extension SDK](https://github.com/OpenWAM/OpenWAM/blob/main/docs/extension_sdk.md) |
| Checkpoints and manifests | [Artifacts](https://github.com/OpenWAM/OpenWAM/blob/main/docs/artifacts.md) |
| Test and parity tiers | [Testing](https://github.com/OpenWAM/OpenWAM/blob/main/docs/testing.md) |

## Contributing

Contributions should preserve the typed runtime boundary and add focused tests
for every changed contract. Read [CONTRIBUTING.md](https://github.com/OpenWAM/OpenWAM/blob/main/CONTRIBUTING.md), the
[Code of Conduct](https://github.com/OpenWAM/OpenWAM/blob/main/CODE_OF_CONDUCT.md), and the
[Security Policy](https://github.com/OpenWAM/OpenWAM/blob/main/SECURITY.md) before opening a pull request.

## Citation

If OpenWAM supports your research, cite the software record in
[`CITATION.cff`](https://github.com/OpenWAM/OpenWAM/blob/main/CITATION.cff):

```bibtex
@software{open_wam_2026,
  title   = {OpenWAM},
  author  = {{OpenWAM Team}},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/OpenWAM/OpenWAM}
}
```

## License

OpenWAM is released under the [GNU Affero General Public License v3.0](https://github.com/OpenWAM/OpenWAM/blob/main/LICENSE)
with the redistribution attribution described in [`NOTICE`](https://github.com/OpenWAM/OpenWAM/blob/main/NOTICE). Covered
modified versions and network services must provide corresponding source, and
redistributed copies must preserve the OpenWAM attribution notice. Academic
work that uses OpenWAM should cite the software record in
[`CITATION.cff`](https://github.com/OpenWAM/OpenWAM/blob/main/CITATION.cff).

Third-party components retain their own terms; the adapted LingBot-VA module
is distributed under Apache License 2.0. Full attributions are listed in
[`THIRD_PARTY_NOTICES.md`](https://github.com/OpenWAM/OpenWAM/blob/main/THIRD_PARTY_NOTICES.md) and [`LICENSES/`](https://github.com/OpenWAM/OpenWAM/tree/main/LICENSES/).
Stanford, SAIL, SVL, and SRC marks are not licensed under AGPL-3.0-only and remain
the property of Stanford University.
