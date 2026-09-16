# Architecture

OpenWAM composes every maintained model through one boundary:

```text
ExperimentConfig
  -> VariantPipeline
  -> VisualTower
  -> PolicyVariant
  -> ActionDecoder
```

The boundary separates shared visual execution from policy experiments. A new
dataset, attention pattern, policy architecture, or decoder should fit one of
these roles instead of adding a parallel training or inference stack.

## Configuration Axes

Video-action experiments are described through these contracts. Architecture,
program, sequence contract, and decoder are authored; each architecture
derives the compatible numerical backend from the selected program.

| Axis | Meaning | Examples |
| --- | --- | --- |
| Architecture | Parameter ownership and execution topology | `parallel_stream`, `dual_expert` |
| Program | Same-chunk video/action conditioning and supervision | `video_then_action`, `action_then_video`, `joint`, `decoupled_same_step`, `video_noisy_to_action`, `action_noisy_to_video`, `generalist_joint_denoising` |
| Sequence contract | Prefix, history, proprio, loss range, and chunk semantics | `default`, `legacy_prefix_single_frame_perchunk_proprio` |
| Numerical execution | Parameter access, packing, attention implementation, and feature reuse | shared blocks, paired blocks, call-local feature caches |
| Decoder | Final predictions, supervised losses, and rollout plan | `parallel_stream_decoder`, `dual_expert_decoder` |

`policy_variant.program` is the public switch for the six standard
video-action programs. Both built-in architectures derive the lower-level
`current_block_coupling` and reject attempts to set it directly. Parallel
Stream also derives its numerical runtime backend and whether action
conditioning is active. Model execution consumes these read-only values;
none is a second authored axis. For example:

```bash
openwam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --set policy_variant.program=video_then_action
```

Named YAMLs remain available for reproducible runs. The override above and the
matching named config resolve to the same typed program and coupling. Program
changes also restore the checkpoint-validated `independent` clock default.
Single-noisy-stream programs require that value; joint and GJD programs can
select a different clock only as an explicit ablation. Sequence contracts own
layout and history, never scheduler coupling, and the runtime never silently
replaces a configured value.

## Built-In Architectures

### Parallel Stream

`parallel_stream` packs video and action streams into one shared transformer.
Its architecture package owns stream embedding, packing, transformer execution,
and output projection. Sequence selection and rollout state are shared:

```text
src/open_wam/models/policy_variants/parallel_stream/
```

It uses the same inference driver and session contract as Dual Expert, not a
second exact runner. Checkpoint-format translation belongs at the loading
boundary, not in active rollout execution.

### Dual Expert

`dual_expert` uses separate video and action transformer parameters and
executes paired blocks under the selected visibility program. Its package owns
expert initialization, paired numerical execution, embedding, and projection:

```text
src/open_wam/models/policy_variants/dual_expert/
```

The two architecture packages must not import one another. Shared semantics
belong in `models/common`, typed policy contracts, or the data layer.

## Denoising And Feature Reuse

Program contracts define dependencies; architectures supply computation;
caches reuse computation without changing those dependencies.

`models/common/denoising.py` derives stage order from the existing coupling and
dynamics contracts. Its `denoise` function advances an existing scheduler using
an architecture-provided prediction function. It does not manage observations,
encode inputs, choose visibility, or decode actions. The shared video/action
executor owns stage transitions, scheduler clocks, and CFG. VTA promotes
completed video into clean conditioning before predicting actions; ATV does
the reverse; decoupled stages do not promote the other modality.
Conditional FDM/IDM retain their limited boundary history and drop instructions.

Paired blocks have one stable parameter owner during training and inference.
Non-owning execution views let the same modules serve single-stream and paired
computation without moving parameters between module trees or changing FSDP
boundaries. Sequence-batched and single-sample training use the same owned stack.

Packed computation and caching are independent: a coupled current chunk can
reuse invariant historical attention features. `inference.use_cache` controls
the new call-local reuse; it does not alter the selected program. The policy's
`inference_capabilities.feature_cache_scope` reports `none`, `denoising_call`, or
`rollout_session`. This is descriptive, not another config selector, and does
not imply support for a particular output subset or reconciliation operation.

The maintained policy programs and chunk-conditioned video use
`denoising_call` when caching is enabled and `none` when disabled. Disabling
feature reuse does not remove required conditioning or change scheduler clocks.
CFG uses the configured video/action guidance scales and requires negative
text embeddings when active; it is not silently disabled by a backend choice.

Call-local caches bind to the canonical attention profile and original token
positions. An invariant partition must be attention-closed: its queries cannot
depend on changing keys. A fixed IDM video input in a coupled live slot is
therefore not automatically cacheable. Each denoising stage, generated video
chunk, and CFG branch gets a fresh scope. Observation replacement, changed
conditioning, partial chunks, and window eviction never retain these caches.
Sessions own observed and speculative model-space tensors separately from
computed features. Realtime callers explicitly reconcile executed observations
before the next generation; they do not rewind architecture-specific K/V.

Action-only generation advances action time without creating video history.
Commit the corresponding observed video before generating again; the runtime
rejects a missing interval rather than relabeling older video. Commit new proprio
when extending history without aligned state rows. If replacement proprio is
omitted, only existing estimated rows for the executed prefix are retained.
A single `previous_action` is not a substitute for an explicit history commit.
Realtime observation updates cover the entire executed interval, with an
explicit model-action start/end and one raw anchor for encoding. The anchor
latent is excluded from replacement; subsequent video and proprio rows match
the executed frames. Missing observations cannot be substituted with later
frames. Shared history code owns replacement and cursor updates; architecture
adapters do not maintain a second semantic cursor.

Observed action history uses model-space controls actually executed, not
speculative controls. Its `PolicyObservedHistory` input accepts
an optional binary `action_mask` [B, T_action, 1], which keeps
startup t0 aligned without exposing a fictitious action. Realtime raw controls
use the data contract's normalization and channel mapping in both directions.
Realtime reconciliation requires raw action targets; pose-target consumers
cannot reconstruct their targets from fallback controller commands.
Speculative extensions start from a complete published session, including
generated video/actions, validity, proprio and temporal geometry.

Prediction prepares a candidate policy state. The caller adopts the returned
state only after prediction and decoding succeed; failed requests do not
advance the live session or bind previously unset temporal geometry.
Generation origins come from `PolicyInferOutput.generation_frame_start`, not
architecture-specific debug dictionaries or local packed tensor offsets.

Call-local caches retain per-layer keys, values, and completed hidden states.
After prefill, only changing tokens run through projections, attention, and
FFNs. Optional output-dependency pruning indexes the same canonical attention
law; it cannot omit keys that the requested outputs depend on. Smaller matrix
shapes can change floating-point rounding, but do not reduce the number of
denoising updates or change training execution.
Chunk-conditioned video uses the same denoising loop and cache primitive,
without introducing an artificial action stream.

Independent video producers and action consumers exchange canonical artifacts,
never model-specific K/V.

## Rollout Lifecycle

```text
Benchmark driver -> RolloutEngine -> RolloutPlanner -> VariantRolloutRunner
                                                           |
                                                    VariantPipeline
                                                           |
                                  VisualTower / PolicyVariant / ActionDecoder
```

`runtime.rollout_engine.RolloutEngine` is the control facade.
`runtime.planning_contracts.RolloutPlanner` owns model transactions over an opaque,
typed session. Its `plan()` returns a candidate; `observe()` commits an executed
interval without generating another plan. Neither publishes nor steps an environment.
`PolicyPlanner` implements this contract for a single pipeline and `RolloutAdapter`.
The engine accepts a planner and a `ControlAdapter`, not a particular model runner.
`ResolvedRolloutTemporalContract` derives frame/control origins, densities, raw
windows and interval validation from the assembled policy, frontend and inference
geometry. It is not another user-authored configuration.
`RolloutLifecycle` owns clock-free acceptance, executed history, revisions and
termination; `PlannerExecutor` owns one serialized worker and its teardown.
Its `control_stream()` yields controls and accepts typed `ControlTransition`
transitions. `run(..., step=backend.step)` drives that stream synchronously;
external APIs such as CALVIN's `step(obs, goal)` send observations to the same
stream. Terminal failure stops execution but does not count as success.
Results retain the typed terminal transition, including benchmark `info`.
Queued work is cancelled at termination/reset; running model calls are drained
before the model can be reused. An unpublished planner failure is recorded in
`planner_teardown`, not raised over the terminal result. Drain latency is
reported separately from live control time; reset is safe, not necessarily instant.
`control_stream()` returns a `RolloutControlStream`: `close()` returns the same
`RolloutResult` produced on normal completion, including cancellation/teardown.
CALVIN retains this as `last_rollout_result` after `reset()`.
Simulator throughput uses `live_wall_time_s`; `wall_time_s` additionally includes
model startup and teardown. Neither duration includes environment reset.
Partially stale candidates are rejected atomically, including suffixes that
happen to begin on a model-frame boundary. Rejected sessions are not published.
`integrations.libero_realtime.LiberoRolloutAdapter` owns simulator I/O, encoding,
proprio extraction, and action conversion. It does not choose a model backend.
The public simulator command uses `SimulatorPolicyAdapter` over normalized
`SimulatorObservation` objects. The pipeline owns normalization/mapping;
backends materialize a `ControlCommand` with both native controls and the
executed command in dataset-source coordinates, for correct history commits.
Proprio-conditioned policies require finite, exact-width model-ready states.
The adapter neither compresses missing observations nor pads/truncates feature
vectors. A no-proprio policy may omit state. Benchmark-native state conversion
remains the backend's responsibility.
The `first_frame` commit mode executes the smallest complete model-frame group,
not a single control within that group. `full_chunk` executes the complete plan.
`--zero-policy` is a separate environment wiring check, not a fake model runner.
Blocking LIBERO, including independent video/action composition, uses this same
engine. `LiberoPolicyPlanner` owns streaming VAE preparation, per-chunk RNG and
the separate producer/consumer sessions; `LiberoPolicyDriver` owns simulator I/O;
`LiberoEpisodeArtifacts` collects images and diagnostics for the existing writer.
The planner declares `supports_async=false`: a mutable streaming encoder cannot
produce speculative background candidates. Blocking plans run on the control
thread; asynchronous planners retain the single serialized worker.

An optional `max_plans` budget stops after complete executed plans. The final
nonterminal interval is observed without an additional prediction. This preserves
LIBERO's final VAE update/history warmup. Environment termination or an action
limit takes precedence: partial terminal frames are recorded, never padded or
reconciled as complete frames. Zero budgets perform no model work. A driver's
optional `termination_check` can prevent issuing a control if the environment
has already stopped; it never invents a transition or an executed action.

The lifecycle's `observed_control_end` identifies what the policy has reconciled;
`next_control_index` identifies what the environment has executed. Any interval
between them remains in the result, including terminal partial model frames.
The accepted policy session may still contain a speculative tail and is not,
by itself, a resumable environment snapshot.

`PolicyInferState` and `VariantRolloutSession` are immutable publications:

- `cursor.current_start_frame` is the exclusive prediction end, hence the next
  generation position; this has the same meaning for video-only and action policies.
- `observed_frame_end` is the exclusive end of the committed observations.
  The interval between it and the cursor is speculative.
- `revision`, history tensors, validity, proprio, decoder state, and bound
  temporal geometry travel with that publication. Tensors are treated as read-only.

Both inference and `runner.reconcile_observed_history(session=..., history=...)`
return a new session. The latter accepts `PolicyObservedHistory`; text is carried
by the request/session and is not changed as a side effect of a history commit.
The control thread alone adopts a planner candidate. A rejected candidate never
mutates the accepted session. This is semantic isolation, not global RNG rollback.

`PolicyInferContext` contains typed output, dynamics, geometry, initial-noise,
video-generation, and video-consumption requests. There is no inference `extra`
control dictionary. `PolicyInferOutput` publishes `generated_video`,
`generated_span`, and `telemetry`; `VariantRolloutStepOutput.action_plan` carries
decoder-owned model-space actions and their frame span. `aux` is diagnostic only.
An executable plan represents one environment, not a silently selected batch row.
Adapters receive that plan and its step index, not a full policy output. Planner
and executed-control receipts are typed records; call `to_record()` only when
writing artifact mappings. The pipeline applies output masks once, after raw
decoding and before the decoder's rollout-plan/state commit hooks. A decoder
that needs final executable actions should use those hooks, not duplicate the
pipeline mask in `forward_infer` or cache a second action tensor in `aux`.

`models.decoder_artifacts` owns the built-in policy-to-decoder payload contracts.
Neither side imports a concrete implementation package on the other side;
new policies can emit an existing decoder's envelope without depending on that
policy's implementation. Envelope identifiers and checkpoint tensor keys are unchanged.

The pipeline's `ActionSpaceAdapter` converts the plan to source controls and the
actual executed controls back to model-space history. Fallback is created in
environment space and recorded with its actual observation. History-freezing
and synthetic startup-action padding are not supported: neither may hide an
executed transition. The observed t0 instead has an explicitly invalid action row.

`models/common/video_action_layout.py` resolves aligned history, t0, validity,
proprio, future extent, and history retention before either architecture packs
tokens. Planning retains its configured history and language. Conditional
FDM/IDM retains one latest boundary frame and drops instructions, whether the
objective comes from a strict program or GJD. Architecture substitution does
not select different conditioning semantics.

## Runtime Ownership

### ExperimentConfig

`open_wam.configs.load_experiment_config` is the YAML-to-dataclass boundary.
Finite public choices are enums. Dataset names, paths, extension identifiers,
and free-form labels remain strings. Naming-only aliases resolve before typed
construction. Retired semantic fields are accepted only when immutable
checkpoint metadata is loaded with `checkpoint_runtime_compat=True`; authored
YAML, CLI overrides, and Python configs have one canonical field per choice.

### VariantPipeline

`VariantPipeline` orchestrates the stable sequence:

1. canonicalize input views;
2. request the visual stages needed by the policy;
3. let the policy prepare and execute architecture-specific tensors;
4. pass a typed policy output to the decoder;
5. return common train or inference outputs.

It does not inspect architecture-specific artifact keys. Policies hand decoder
payloads across `DecoderArtifactEnvelope`, and each decoder validates its own
contract and payload type.

### VisualTower

`VisualTower` owns the shared visual frontend, transformer-facing runtime
hooks, optional decode stage, and common runtime-program execution. It accepts
prepared attention profiles and runtime inputs; it does not decide policy
conditioning or supervision semantics. Multi-view VAE encoding consumes the
data layer's `ViewPlacement` contract and reconstructs the corresponding latent
canvas without benchmark names or camera-specific branches.

### PolicyVariant

A policy variant owns:

- required visual stages;
- train input preparation;
- parameter topology and runtime-program selection;
- recurrent inference state and cache reconciliation;
- architecture-specific decoder artifacts.

It does not own final supervised action losses.

### ActionDecoder

An action decoder owns final action predictions, supervised losses, decoder
state, and the model-space action plan committed by rollout. Generic rollout
code asks the decoder for a plan instead of branching on an architecture.

## Generalist Joint Denoising

GJD is the `generalist_joint_denoising` program inside either architecture. A
sample selects one `DynamicsObjective`:

| Mode | Clean supplied modality | Active loss | Task text |
| --- | --- | --- | --- |
| `joint` | neither | video and action | retained |
| `action_conditioned_video` (FDM) | action in the action-noisy slot at timestep zero | video only | removed |
| `video_conditioned_action` (IDM) | video in the video-noisy slot at timestep zero | action only | removed |

`data.dynamics_routing.routes` is the single sampling contract. Each
route names a source, mode, and relative weight; the data adapter stamps that
choice into sample metadata. The policy program is the single execution
contract: GJD consumes the routed mode, while standalone `forward_dynamics` and
`inverse_dynamics` require every active route to match their fixed mode. Adding
a new source or mode extends these typed contracts and its data/runtime adapter,
without adding a second probability field to a policy config. Custom routed
datasets expose source-specific validation through the
`open_wam.data.DynamicsSourceViewProvider` protocol; callers do not need to
subclass a built-in dataset.

Conditional real-demo and counterfactual samples share one rollout-style data
contract:

```text
latent frame 0     observed t0, clean history, no loss
latent frames 1..N future targets, supervised according to FDM or IDM mode
```

The t0 frame is always a singleton chunk. Future chunks retain the sampled GJD
geometry, including the maintained 1-to-4 frame randomization where configured.
Conditional attention exposes only the most recent clean video boundary, not a
long demonstration prefix. Both backends consume the same typed hidden-proprio
contract: frame-level state is projected onto that previous boundary, while
chunk-level state that was already sampled at a boundary is expanded over its
matching chunk. This projection happens before backend-specific token packing,
so tensor shape cannot silently change the state alignment and recorded future
state is not leaked into the current conditional chunk. The
encoded-dynamics adapter selects rollout-local
`gt` rows for real-demo conditional routes and perturbed rows for
counterfactual routes, then validates the same target-only metadata before
either architecture executes it.

The shared owner of mode semantics is
`open_wam.models.common.dynamics_objectives`. Architecture code applies those
decisions to its own packing and cache representation. Joint-mode behavior is
unchanged by the conditional target-only transform.

`parallel_stream` and `dual_expert` are maintained architecture choices under
this same GJD paradigm. Their real-joint rows use the configured planning
prefix. Conditional rows use t0 directly as frame 0 and bypass external-prefix
assembly in both backends, so both expose exactly the same clean history.
Accordingly, `require_condition_latents` requires an external prefix only for
samples without an in-sequence t0; it does not require a redundant tensor for
target-only conditional rows.
Select architecture through the experiment config:

```bash
openwam-train \
  --config-name <architecture>_libero_generalist_joint_denoising \
  --set policy_variant.generalist_mode_text_token=true
```

A source checkout additionally includes `scripts/run_gjd_libero.sh` for named
ablation expansion and LIBERO rollout orchestration. It delegates training to
the same package entry point and is intentionally absent from distributions.

## Data Boundary

Dataset-specific parsing belongs in adapters selected by `data.dataset_type`.
Adapters emit the uniform `WAMSample` or `LatentWAMSample` contract. The data
layer owns:

- camera decoding and canonical RGB assembly;
- action and state transforms;
- temporal and latent alignment;
- sample geometry and loss-range metadata;
- mixed-source and counterfactual routing.

The model receives canonical tensors and typed metadata. It does not know a
dataset's native camera names, storage format, or simulator schema.

## Attention And Cache Boundary

Common attention contracts separate visibility from execution:

- `attention_contracts` defines typed profiles and normalized choices;
- `chunked_attention_visibility` defines token-pair visibility;
- `chunked_attention` produces dense or FlexAttention representations;
- `attention_backends` executes the selected representation.

Architecture packages own their exact packed layouts and cache payloads. A
custom policy should submit a common prepared attention profile through a
runtime program instead of modifying the shared backbone.

## Training And Checkpoints

The generic training runtime owns optimizer, scheduler, logging, validation,
distributed strategy, and checkpoint lifecycle. Policy differences enter only
through configured batches and pipeline outputs.

`CheckpointManager` writes model state and, when configured, full optimizer,
scheduler, strategy/scaler, and progress state. Architecture refactors must
preserve parameter names,
registration order, state-dict keys, optimizer mapping, and recurrent cache
semantics for maintained checkpoints.

## Extension Boundary

Applications can register:

- dataset adapters with `register_dataset_adapter`;
- policy variants with `register_policy_variant`;
- action decoders with `register_action_decoder`;
- simulator backends through the simulator extension contract.

Extensions use open string identifiers and parse their own typed options.
Built-in finite choices remain enums. See [Extension SDK](extension_sdk.md) and
the cookbooks under `docs/cookbooks/`.

## Numerical Behavior

Changing a component boundary must preserve the configured model semantics.
Regression tests compare outputs, losses, gradients, optimizer updates,
recurrent caches, checkpoint resume, and rollout artifacts against immutable
references. See [Testing](testing.md#numerical-regression).
