# Policy Architectures And Programs

Open-WAM separates model topology from experiment semantics. This distinction
keeps checkpoints understandable and lets users change a conditioning program
without selecting a different trainer or visual backbone.

## Vocabulary

| Concept | What it controls | Examples |
| --- | --- | --- |
| Architecture | Parameter topology and token execution | `parallel_stream`, `dual_expert` |
| Program | Video/action conditioning and supervision | `video_then_action`, `joint`, `generalist_joint_denoising` |
| Sequence contract | Prefix, history, chunk, and alignment semantics | single-frame prefix, target-only conditional layout |
| Runtime backend | Packed execution, cache writes, and denoising implementation | exact shared-stream, packed dual-expert |
| Action decoder | Final action outputs and supervised losses | `parallel_stream_decoder`, `dual_expert_decoder` |

An experiment config selects these contracts explicitly. The trainer and
simulator integrations do not branch on architecture nicknames.

## Built-In Architectures

| Architecture | Parameter topology | Maintained programs |
| --- | --- | --- |
| `parallel_stream` | Video and action tokens share one transformer and exact packed-stream cache lifecycle. | Six standard video/action programs and GJD; the exact LingBot backend is its primary compatibility profile. |
| `dual_expert` | Video and action have separate transformer experts that execute paired blocks. | Six standard video/action programs, GJD, and conditional FDM/IDM. |
| `causal_video_prediction` | The visual model runs without action supervision. | Video-only prediction. |

The six standard video/action programs are:

- `video_then_action`
- `action_then_video`
- `joint`
- `decoupled_same_step`
- `video_noisy_to_action`
- `action_noisy_to_video`

`generalist_joint_denoising` adds sampled joint, FDM, and IDM submodes. Its
conditional FDM/IDM layout is a separate sequence contract, not another model
architecture.

The dual-expert architecture also exposes those conditional submodes as fixed
standalone programs:

- `forward_dynamics`: clean actions condition video prediction; only video loss
  is active.
- `inverse_dynamics`: clean video conditions action prediction; only action loss
  is active.

These programs compile to exactly the same conditional runtime as one-hot GJD.
The canonical `dual_expert_libero_conditional_dynamics.yaml` config composes that
program with the existing sequence and data contracts: one clean t0 latent in
its own singleton chunk, only the most recent clean video/proprio boundary as
rolling history, the sampled 1-4 frame size for following chunks, no task text,
and only matching real-demo/counterfactual sources. They are not separate model
architectures and do not introduce another trainer, attention implementation,
or decoder.

`forward_dynamics` and `inverse_dynamics` are first-class program selectors,
but they are conditional training and offline-inference objectives. Unlike VTA,
ATV, joint, and the other standard programs, they require future clean
conditioning tensors and are not live simulator policy rollouts. The maintained
config defaults to a `1:1` real-demo/counterfactual mixture and therefore
requires encoded counterfactual train and validation roots. See the
[data prerequisites](running_experiments.md#data-prerequisites), including the
explicit real-demo-only ablation.

## Shared Execution Boundary

Every built-in architecture runs through:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

- `VariantPipeline` owns common train and inference orchestration.
- `VisualTower` owns the shared visual frontend and transformer-facing runtime.
- `PolicyVariant` owns architecture and program semantics.
- `ActionDecoder` owns final outputs and losses.

Architecture implementations do not import one another. Shared behavior lives
in typed config, sequence, attention, scheduler, and decoder-artifact
contracts.

## Selecting A Program

Maintained YAML names encode architecture and program. For a one-off ablation,
the public switch is `policy_variant.program`:

```bash
open-wam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --set policy_variant.program=video_then_action
```

The config boundary derives low-level coupling fields and rejects conflicting
program/coupling combinations.

## Extending Open-WAM

Choose the smallest extension that expresses the change:

- New output or loss only: add an `ActionDecoder`.
- New visibility or execution semantics on an existing topology: add a typed
  runtime/attention program.
- New parameter topology or recurrent-state owner: add a `PolicyVariant`.
- New source format: add a dataset adapter selected by `data.dataset_type`.

See [Add A Policy Architecture](cookbooks/new_policy_architecture.md) and the
[Extension SDK](extension_sdk.md).

## Compatibility Names

Historical names are accepted only at explicit compatibility boundaries:

| Historical input | Canonical meaning |
| --- | --- |
| M1 / Method 1 | `parallel_stream` |
| M5 / Method 5 / `mot` / MoT | `dual_expert` |
| `*_heng_compatible` config | canonical architecture/program config alias |
| `mot_decoder` | `dual_expert_decoder` |
| `lingbot_parallel_decoder` | `parallel_stream_decoder` |

Old YAML names, import paths, and checkpoint-local resolved configs continue to
load with deprecation warnings. New code, configs, runs, and documentation must
use canonical names.
