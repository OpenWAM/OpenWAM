# Architecture

OpenWAM composes models through one boundary:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

This separates shared visual execution from policy experiments. A new dataset,
conditioning program, model topology, or decoder should fit one of these roles
rather than require a separate training or inference stack.

## Runtime Ownership

| Component | Responsibility |
| --- | --- |
| `ExperimentConfig` | Typed data, model, conditioning, optimization, and runtime choices |
| `VariantPipeline` | Common input preparation, visual-stage orchestration, and decoder dispatch |
| `VisualTower` | Shared encoding, transformer-facing execution, and optional video decoding |
| `PolicyVariant` | Parameter topology, conditioning, token preparation, and recurrent inference state |
| `ActionDecoder` | Final predictions, supervised losses, and model-space action plans |

The pipeline passes typed artifacts between components. It does not inspect a
particular policy's private tensors; the decoder validates its own payload.
The visual tower executes prepared inputs and attention profiles, while the
policy determines what conditioning and supervision mean.

## Configuration Axes

Architecture, program, sequence contract, and decoder express different choices:

| Axis | Meaning | Examples |
| --- | --- | --- |
| Architecture | Parameter ownership and topology | `parallel_stream`, `dual_expert`, `causal_video_prediction` |
| Program | Video/action conditioning and supervision | `joint`, `video_then_action`, `generalist_joint_denoising` |
| Sequence contract | Prefix, history, proprioception, and supervised frames | `default`, `legacy_prefix_single_frame_perchunk_proprio` |
| Decoder | Predictions, losses, and executable action plans | `parallel_stream_decoder`, `dual_expert_decoder` |

For video-action policies, `policy_variant.program` selects the program:

```bash
openwam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --set policy_variant.program=video_then_action
```

The architecture derives compatible numerical execution and lower-level
coupling from this choice. Those derived values are not additional authored
config switches. Scheduler coupling is a separate explicit choice, not
something inferred from a sequence layout. See
[Policy Architectures and Programs](policy_architectures.md) for supported
combinations and [Training and Inference](running_experiments.md) for recipes.

Finite built-in choices become enums at the config boundary. Dataset names,
paths, and extension identifiers remain open strings. For retired fields and
checkpoint loading, consult [Compatibility](compatibility.md) and the
[migration guide](migration_0_2.md).

## Built-In Architectures

### Parallel Stream

`parallel_stream` packs video and action tokens into one shared transformer.
The policy owns stream embedding, token packing, and output projection.

### Dual Expert

`dual_expert` uses separate video and action transformer parameters. Paired
blocks execute under the chosen program's visibility rules.

### Causal Video Prediction

`causal_video_prediction` predicts video without action supervision. It can
serve as a video producer for an independent action model; composition does
not require it to create an artificial action stream.

Architecture packages do not import one another. They reuse shared sequence,
attention, scheduler, and history contracts. Changing topology does not
implicitly change conditioning semantics.

## Data Boundary

Adapters selected by `data.dataset_type` parse source-specific storage and emit
`WAMSample` or `LatentWAMSample`. The data layer owns:

- camera decoding and canonical multi-view assembly;
- action/state transforms and temporal alignment;
- sample geometry, validity, and supervised-frame metadata;
- mixed-source and dynamics-objective routing.

Models consume canonical tensors and metadata, not native camera names or
simulator storage schemas. See [Benchmarks and Data](benchmarks.md) and
[Add A Dataset](cookbooks/new_dataset.md).

## Generalist Joint Denoising

GJD is the `generalist_joint_denoising` program in either video-action
architecture. Each sample selects one objective:

| Mode | Supplied clean modality | Active loss | Task text |
| --- | --- | --- | --- |
| Joint | Neither future modality | Video and action | Retained |
| Forward dynamics (FDM) | Future action | Video only | Removed |
| Inverse dynamics (IDM) | Future video | Action only | Removed |

`data.dynamics_routing.routes` assigns sources, modes, and relative sampling
weights. Standalone `forward_dynamics` and `inverse_dynamics` use the same
conditional semantics but require routes matching their fixed objective.

Conditional samples begin with a clean t0 video frame in a singleton chunk,
with no supervised or visible fictitious t0 action. Future frames are targets.
Each conditional chunk sees the most recent clean video boundary and its
aligned proprioception, not an unrestricted demonstration prefix. The supplied
future modality occupies its clean, timestep-zero conditioning slot; only the
other modality is denoised and supervised.

Joint samples retain their configured planning history and text.
Frame-level and chunk-level proprioception are aligned to the preceding
boundary before architecture-specific token packing, preventing access to
future state. These rules are shared by training and rollout.

See [GJD recipes](running_experiments.md#generalist-joint-denoising) for source
routing, ablations, and inference commands.

## Denoising And Feature Reuse

Programs define dependencies; architectures supply computation. Shared
denoising advances schedulers and applies per-stream classifier-free guidance
(CFG). Video-then-action supplies the completed video to action generation;
action-then-video reverses that order; decoupled stages do not promote the
other modality.

`inference.use_cache` enables reuse of invariant features within a denoising
call. It does not change history visibility, scheduler clocks, or the number
of denoising updates. Features are reusable only when their attention
dependencies are invariant. A supplied clean modality is not automatically
cacheable if its queries can attend to changing tokens.

Observed and speculative history belong to the rollout session, separately
from these temporary computed features. Attention profiles define visibility;
dense and FlexAttention backends execute that same contract.

## Rollout Lifecycle

```text
Benchmark driver -> RolloutEngine -> RolloutPlanner -> VariantRolloutRunner
                                                           |
                                                    VariantPipeline
```

The driver translates environment I/O. The planner prepares a candidate using
an immutable policy session. The engine accepts a plan, executes controls,
and commits actual observations/actions before the next generation.
Predictions and unexecuted actions are speculative, not observed history.

The same lifecycle supports blocking and asynchronous execution where the
planner permits it. Fallback controls must be recorded as executed; they
cannot be hidden from model history. Temporal geometry comes from the
assembled policy, frontend, and action contract rather than script-specific
offsets.

For custom integrations, see [Rollout Contracts](rollout_contracts.md).
Independent video and action models use the same lifecycle and exchange
canonical artifacts, not model-specific caches; see
[Video-to-Action Composition](video_action_composition.md).

## Training And Checkpoints

The training runtime owns optimization, scheduling, logging, validation,
distributed execution, and checkpoint publication. Policies supply prepared
batches and predictions; decoders supply losses.

Full-state checkpoints include optimizer, scheduler, strategy/scaler, and
progress state. Weight initialization and full-state resume are distinct
operations. See [checkpoint usage](running_experiments.md#initialization-and-full-state-resume)
and [Artifacts](artifacts.md).

## Extension Boundary

Choose the smallest component that owns the change:

- Add a dataset adapter for a new source format.
- Add a policy for a new parameter topology or conditioning program.
- Add a decoder for a new output representation or loss.
- Add a simulator adapter for environment I/O.

Use role-specific `open_wam.sdk` contracts rather than private implementation
helpers. The [Extension SDK](extension_sdk.md) and cookbooks provide examples.

## Numerical Behavior

Refactors must preserve configured model semantics. Regression coverage checks
outputs, losses, gradients, optimizer updates, history reconciliation, and
checkpoint resume. Exact numerical claims additionally require matched
artifacts and hardware/software stacks; see [Testing](testing.md#numerical-regression).
