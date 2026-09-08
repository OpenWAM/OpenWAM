# Policy Architectures And Programs

OpenWAM separates model topology from experiment semantics. This distinction
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
| `parallel_stream` | Video and action tokens share one transformer and exact packed-stream cache lifecycle. | Six standard video/action programs, GJD, and conditional FDM/IDM through the maintained exact packed backend. |
| `dual_expert` | Video and action have separate transformer experts that execute paired blocks. | Six standard video/action programs, GJD, and conditional FDM/IDM. |
| `causal_video_prediction` | The visual model runs without action supervision. | Video-only prediction. |

For an action-only size reduction that preserves all 30 paired layers, see
[Smaller action experts](action_expert_sizes.md). The optional `small_500m`
profile leaves the video model and program semantics unchanged.

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

Both video/action architectures expose the conditional submodes as fixed
standalone programs:

- `forward_dynamics`: clean actions condition video prediction; only video loss
  is active.
- `inverse_dynamics`: clean video conditions action prediction; only action loss
  is active.

These programs compile to exactly the same shared training and rollout plans as
the corresponding single-mode GJD route. The plan fixes one clean t0 latent in
its own singleton chunk, the most recent clean video/proprio boundary as rolling
history, the sampled 1-4 frame size for following training chunks, no task text,
and only matching real-demo/counterfactual sources. Each architecture then packs
that plan for its own topology and cache backend. Fixed dynamics does not add a
trainer, attention implementation, or decoder.

`dual_expert_libero_conditional_dynamics.yaml` is a convenience experiment
profile, not the owner of these semantics. For either architecture, its GJD
profile can be switched atomically to `forward_dynamics` or `inverse_dynamics`
with matching `data.dynamics_routing.routes` and one fixed-mode validation
probe.

`forward_dynamics` and `inverse_dynamics` are first-class program selectors,
but they are conditional training and offline-inference objectives. Unlike VTA,
ATV, joint, and the other standard programs, they require future clean
conditioning tensors and are not live simulator policy rollouts. The maintained
config defaults to a `1:1` real-demo/counterfactual mixture and therefore
requires encoded counterfactual train and validation roots. See the
[data prerequisites](running_experiments.md#data-prerequisites). A real-only
ablation still uses those encoded roots, selecting only their rollout-local
`gt` branch.

## Shared Execution Boundary

Every built-in architecture runs through:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

- `VariantPipeline` owns common train and inference orchestration.
- `VisualTower` owns the shared visual frontend and transformer-facing runtime.
- Shared program contracts own architecture-independent conditioning and
  supervision semantics.
- `PolicyVariant` declares model-space geometry and translates those contracts
  into architecture-specific packing, execution, and recurrent state.
- `ActionDecoder` owns final outputs and losses.

Architecture implementations do not import one another. Shared behavior lives
in typed config, sequence, attention, scheduler, and decoder-artifact
contracts.

Before modules are allocated, the typed policy config returns one
`PolicyConditioningRequirements` value so the shared tower can create its
proprio and dynamics-mode adapters in deterministic checkpoint order. After
allocation, the policy returns `PolicyPipelineRequirements`; the factory checks
that its model-space geometry and conditioning match the already assembled
tower and decoder. When model and source action spaces differ, this latter
contract declares every accepted source action shape and carries the
source-channel projection into the decoder. Shared factories validate that
contract without inspecting a backend's action adapter.

Dynamics-capable programs additionally share five typed boundaries:

- `DynamicsSamplePlan` resolves program, routed objective, and sequence layout
  before an architecture prepares training tensors.
- `DynamicsTrainingPlan` owns clean/noisy slots, timesteps, text removal, and
  loss activation.
- `DynamicsRolloutRequest` carries the clean modality and committed action
  history into either recurrent backend.
- `DynamicsRolloutGeometry` resolves chunk size, attention window, history
  stream visibility, and conditional-history policy once for either backend.
- `HiddenProprioContext` preserves frame-versus-chunk sampling granularity and
  projects both architectures onto the same chunk-boundary state sequence.

Replacing `dual_expert` with `parallel_stream` therefore keeps the program,
route metadata, sequence layout, rollout request, and resolved rollout geometry
unchanged. The expected differences are parameter topology, model-space action
packing, attention execution, and cache storage. A backend may reject geometry
its native runtime cannot represent, but it must not reinterpret shared
semantics.

## Selecting A Program

Maintained YAML names encode architecture and program. For a one-off ablation,
the public switch is `policy_variant.program`:

```bash
openwam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --set policy_variant.program=video_then_action
```

The config boundary derives low-level coupling fields and rejects conflicting
program/coupling combinations.

## Extending OpenWAM

Choose the smallest extension that expresses the change:

- New output or loss only: add an `ActionDecoder`.
- New visibility or execution semantics on an existing topology: add a typed
  runtime/attention program.
- New parameter topology or recurrent-state owner: add a `PolicyVariant`.
- New source format: add a dataset adapter selected by `data.dataset_type`.

See [Add A Policy Architecture](cookbooks/new_policy_architecture.md) and the
[Extension SDK](extension_sdk.md).

## Compatibility Names

Historical naming aliases are accepted at explicit compatibility boundaries:

| Historical input | Canonical meaning |
| --- | --- |
| M1 / Method 1 | `parallel_stream` |
| M5 / Method 5 / `mot` / MoT | `dual_expert` |
| `mot_decoder` | `dual_expert_decoder` |
| `lingbot_parallel_decoder` | `parallel_stream_decoder` |

Checkpoint-local resolved configs can migrate retired semantic fields that
map unambiguously onto the current contract through
`checkpoint_runtime_compat=True`. This is not a guarantee that arbitrary
historical M1 configs remain executable. New YAML, CLI overrides, and Python
configs must use canonical semantic fields; ordinary config loading does not
translate them.
