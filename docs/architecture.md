# Architecture

Open-WAM keeps one stable top-level runtime boundary:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

The goal is to compare policy-attachment strategies without changing the
shared visual execution path for every experiment.

## Core Pieces

| Component | Responsibility |
| --- | --- |
| `ExperimentConfig` | Typed config boundary loaded from YAML. String choices are coerced into enums before runtime use. |
| `VariantPipeline` | Orchestrates data batches, visual stages, policy-variant execution, and decoder loss/output calls. |
| `VisualTower` | Owns visual preprocessing, shared frontend/core/decode hooks, and backbone-facing runtime outputs. |
| `PolicyVariant` | Defines method semantics: required visual stages, train inputs, infer state, rollout-step behavior, and reconciliation of executed observations with recurrent state. |
| `ActionDecoder` | Converts variant outputs into supervised action predictions and losses. |

## Design Principles

- Keep method differences in policy variants, runtime programs, cache policy,
  sequence semantics, schedulers, and decoders.
- Keep dataset-specific parsing inside dataset adapters selected by
  `data.dataset_type`.
- Keep canonical RGB layout construction in the data layer.
- Keep public finite choices enum-backed at the typed config boundary.
- Do not add method-named infrastructure when the abstraction is generic.

## Foundational Contracts

`open_wam.contracts` is the dependency-free layer shared by configuration,
data, models, and runtime code. It contains only standard-library value
contracts and deterministic transforms:

- `contracts.paths` owns source-root discovery and repository-relative path
  resolution.
- `contracts.video` owns WAN raw/latent temporal geometry, typed video timeline
  records, frame mapping, FPS normalization, and camera placement inside a
  canonical RGB canvas.
- `contracts.sample_metadata` owns serialized GJD metadata keys and the typed
  runtime view of sample geometry, loss ranges, and generalist mode metadata.

These contracts do not import another `open_wam` package. Higher layers may
depend on them, but they must not depend back on configuration, datasets,
models, or runtime implementations. Model code therefore consumes view
placement and sample metadata without importing dataset implementations.
Historical `runtime.paths`, `utils.video_timeline`, `utils.wan_geometry`,
`data.raw_video.ViewPlacement`, `data.sample_metadata`, and generalist metadata
keys under `configs.variant_semantics` remain identity-preserving compatibility
facades; new code imports `open_wam.contracts`.

## Installed And Checkout-Only Boundaries

The `open_wam` wheel contains maintained configuration, data, model, training,
runtime, evaluation-result, and extension contracts. Benchmark integrations
remain lazy so importing the core does not require simulator dependencies.

Reusable result schemas, rollout artifact policies, and maintained renderers
live under `open_wam.evals`. Large experiment analyses remain under `scripts/`;
in particular, `scripts/research_dynamics/` contains checkout-only FDM/IDM
diagnostics and is not a public import surface. Stable top-level scripts own
their command interfaces and import reusable policy and data behavior from the
installed package. Research tools must require machine-local checkpoints and
datasets explicitly rather than embedding private defaults.

## Configuration Contract

`open_wam.configs.load_experiment_config` is the public YAML-to-dataclass
entrypoint, and `open_wam.configs.local_paths` owns machine-local path
expansion over the shared project-path contract.
`open_wam.configs.coercion` owns reusable YAML/CLI-to-type conversion.
`open_wam.configs.sequence_contracts` owns defaults and cross-section
validation for sequence semantics. Each typed component module owns its
mapping-to-dataclass parser; the larger data section is isolated in
`open_wam.configs.data_parsing`. The root loader only reads YAML, composes those
parsers, and runs explicit cross-section checks. Historical `open_wam.utils`
loader/path imports are compatibility aliases only. Configuration parsing does
not own policy runtime behavior.

## Visual Tower Contract

The shared visual stack exposes stage-aware outputs rather than allowing policy
variants to reach into arbitrary backbone internals. Common stage families are:

- current visual features for action-conditioned policies
- post-core token or latent features for feature-attached policies
- generated future visual features for video-conditioned action heads
- decode-stage outputs for post-decoded baselines

Policy variants request stages through `required_visual_stages()` and consume
prepared inputs through explicit variant contracts.

## MoT Internal Contracts

The built-in MoT policy keeps method routing and visual execution in
`MoTPolicyVariant`, but delegates deterministic preparation to role-specific
plain contracts:

- `MoTTrainingLayout` converts typed batch metadata into loss ranges, history
  length, chunk/window geometry, and action/video supervision masks.
- `MoTConditioning` prepares condition latents, prefix layout, text/proprio
  tensors, mode tokens, and cross-attention gating. Learned conditioning
  encoders remain owned by the visual core.
- `MoTPackedInferenceLayout` validates current-chunk FDM/IDM tensor overrides.
  `MoTPackedHistory` selects one frame-aligned recurrent video/action/proprio
  window without owning or mutating policy state.
- `mot.observed_history` owns replacement of speculative packed video/action/
  proprio tails after an environment executes a chunk. Benchmark integrations
  supply canonical observations through `VariantRolloutRunner`; they do not
  mutate `MoTRuntimeState` fields.
- `build_action_grid_ids_for_sequence` owns the frame-aligned action
  coordinates shared by training and recurrent inference.
- `mot.generalist_modes` owns GJD mode selection and conditional tensor
  rewrites.
- `mot.runtime_routing` owns finite inference routes, coupling policy, and
  validated rollout overrides. Its implementation is configuration-only and
  contains no tensor execution.
- `mot.attention` owns parameter-free MoT mask and prepared-profile
  construction. Applications can select or replace these layouts without
  modifying learned runtime execution.
- `mot.cache_state` owns typed cache movement, append, retention, and
  speculative rewind without executing model parameters.
- `mot.cache_execution` owns learned video-cache prefill and action execution
  against video-only or combined video/action caches.
- `mot.dual_stream_execution` owns learned packed coupling and simultaneous
  joint video/action execution. `mot.unpacked_training` composes those
  executors for non-packed training without owning model parameters.
- `mot.runtime` is a compatibility facade for historical internal imports; new
  code imports the role-specific owners directly.
- common attention profiles own token visibility; the policy selects a profile
  and supplies its resolved layout.
- `visual_tower.shared_transformer_support` owns the reusable learned
  transformer primitives: timestep and rotary embeddings, attention and
  transformer-block execution, and FSDP-safe parameter materialization. Both
  the visual core and action-side experts consume this public implementation.
- `visual_tower.context_encoders` owns learned proprio and GJD mode
  projections. `SharedVideoTransformerCore` attaches them under stable
  checkpoint names and owns their execution lifecycle.
- `visual_tower.runtime_tensor_transport` owns parameter-free tensor,
  attention-profile, and slot-pool-state movement across block devices. The
  core binds model patch geometry but does not reimplement transport policy.
- `visual_tower.cache_lifecycle` composes backend operations into the shared
  `CacheState` lifecycle: initialization, named branches, retention, cursor
  advancement, and reset. `VisualTower` keeps the stable public facade and
  supplies its current capability and layer count; the lifecycle owns no
  modules or tensors.
- `visual_tower.runtime_backbone` owns checkpoint-source selection, one-time
  loading diagnostics, action-dimension access validation, runtime
  device/dtype normalization, and current/legacy cache reset. It borrows the
  tower-owned module for each operation and cannot register checkpoint state.
- `visual_tower.exact_runtime` owns shared exact single-stream input
  preparation, CFG duplication, dtype selection, and execution. Policy
  runtimes may select when to use it but do not reimplement these backbone
  operations.
- `parallel_stream.exact_cache` owns the typed policy-side cache context,
  write-interface selection, cache-token stream labels, scoped slot-pool
  metadata, text/CFG preparation, and attention-window compatibility checks.
  It delegates allocation and tensor execution to
  `visual_tower.exact_runtime`; it does not own backbone mechanics.
- `parallel_stream.runtime_semantics` resolves enum-backed history visibility,
  condition-latent sources, block/timestep coupling, cache-prefix visibility,
  and attention-profile selection. Policy execution and dynamics evaluation
  consume the same resolver instead of defining local compatibility rules.
- `parallel_stream.conditional_rollout` maps rollout labels to GJD modes and
  owns parameter-free conditional window/chunk, history, warmup-suffix, and
  conditioning-slice layout. Learned mode-token injection and model execution
  remain in the policy runtime.
- `parallel_stream.training_noise` owns parameter-free exact-stream noising,
  scheduler-grid adaptation, and tuple layouts for coupled timestep plans. It
  consumes the generic flow-noise plan from `models.common`; model execution
  and generalist mode selection remain in the policy runtime.
- `parallel_stream.latent_conditioning` owns validation and selection of
  first-frame and full-window clean latent conditions. Generated decoder
  windows remain in `models.policy_variants.common.video_conditioning`;
  M5 text and proprio conditioning remain in `mot.conditioning`.
- `parallel_stream.generalist_training` owns FSDP-coordinated GJD mode
  selection and policy-local joint/FDM/IDM artifact mutation. Artifact
  construction, model execution, caches, and decoder losses remain in their
  existing runtime owners; generic flow schedulers remain in `models.common`.
- `open_wam.contracts.video` owns dependency-free WAN raw/latent temporal
  mapping. `models.common.video_geometry` owns Torch-backed video token-grid
  and unpatchifying transforms used by visual execution, policy variants, and
  decoders.
- `models.common.cache_backends` owns parameter-free attention-cache policy:
  dense-mask normalization, cached-prefix visibility, packed sequence ids,
  slot retention, merged-prefix operations, and backend payloads. The visual
  cache lifecycle applies those operations to runtime state; the shared
  transformer owns projections and attention execution.

The layout, conditioning, mode, and routing helpers are plain contracts, not
model modules. They must not own parameters, buffers, visual execution, or
decoder losses, so extracting or replacing orchestration cannot change
checkpoint keys. `MoTPolicyVariant` selects these contracts and orchestrates
the learned runtime; it does not redefine their control or tensor mechanics.

## Data Contract

Dataset adapters are selected by `data.dataset_type`. One adapter identity may
provide a raw-RGB builder, a pre-encoded latent builder, or both. External
adapters register explicitly through `module[:hook]` extensions before
experiment construction.

Raw adapters normalize source records into one public batch contract:

- canonical RGB tensors with a configured camera/layout policy
- action tensors with explicit source and model dimensions
- optional state tensors
- text/task metadata when available
- adapter metadata that documents action mapping and benchmark identity

This lets LIBERO, RoboTwin, CALVIN, synthetic fixtures, and future datasets use
the same train/eval/runtime stack.

Dataset storage and sample semantics are separate responsibilities. For the
local LeRobot latent adapter, `LocalLatentRepository` in
`lerobot_v2_latent_storage` owns repository discovery, episode-row and
per-camera latent I/O, condition-payload validation, canonical latent-canvas
assembly, and bounded I/O caches;
`latent_segment_geometry` owns pure eligible-start, materialized-bound, and
loss-bound calculations; `latent_segment_materialization` combines those
bounds with raw-frame anchors and zero-order-hold latent slicing through a
public typed plan; `lerobot_v2_latent_split` owns local repository-window
discovery, replay-status filtering, explicit validation roots, and
train/validation partitioning; `lerobot_v2_latent_sampling` owns local
uniform-segment eligibility and ordering, hierarchical task/trajectory mass
tables and split-salted draws, sampling metadata, and the thin
distributed-sampler adapters; `latent_hierarchical_sampling` composes those
draws with local chunk candidates, clean-context policy, eligible start
ranges, resolved segment boundaries, and a typed diagnostic sample key;
`latent_causal_sampling` owns tensor-free causal prefix/suffix candidate
geometry, split-aware draw order, and typed raw/latent window plans;
`lerobot_v2_latent_supervision` owns deterministic local row alignment,
LingBot action-sequence assembly, adapter-specific sequence extraction, state
history, and per-frame/per-chunk proprio assembly;
`lerobot_v2_latent_segment` combines one selected materialization with those
latent and supervision tensors through `LocalLatentSegment`;
`lerobot_v2_latent_source` loads one physical window through the repository,
exposes canonical video/condition payloads, applies source frame-ID fallback,
and resolves frame-indexed task/text conditioning; and
`lerobot_v2_latent` owns profile/sample-mode and dataset-class selection,
sampling-plan invocation, mode-specific tensor materialization, final metadata,
and public `LatentWAMSample` construction. It calls repository, source, and
supervision owners directly; it does not duplicate them behind dataset-private
facades. Historical module-level helper imports remain identity aliases for
compatibility, but new extensions should use the public `open_wam.data`
contracts above.
For row-oriented robot datasets, `row_action_targets` owns raw, relative-EEF,
and absolute-joint target conversion, action mapping, normalization, and
target metadata. `sequence_packing` owns the canonical float32 padded tensor
and validity-mask layout. Each adapter still owns row decoding, empty-input
policy, and whether an overlong source sequence may be truncated.
For mixed conditional-dynamics training, `counterfactual_dynamics_dataset`
owns encoded counterfactual manifests, payload I/O, hierarchical sampling, and
target-only sample assembly. `generalist_dynamics` owns source sampling and
FDM/IDM mode routing, while `conditional_dynamics_layout` owns the
parameter-free projection of real demonstrations to the rollout-style
`t0 + future` tensor and metadata contract. Joint samples bypass that
projection.
`latent_view_assembly` owns the public, dataset-independent 1-4-view latent
canvas contract. Dataset adapters choose slots and sampling weights, then call
`assemble_latent_views`; backbones receive only the assembled canonical tensor.
For manifest-backed video pretraining, `mixed_video_catalog` owns source CSV
parsing, stream and episode records, path/FPS normalization, target-slot
validation, task merging, and physical-episode train/validation splits.
`mixed_video_decode` owns video-file materialization, timestamp clipping,
target-FPS interpolation, adaptive sizing, frame transforms, and imageio/decord
backend selection. `mixed_video_latent_storage` owns latent-sidecar
materialization, payload validation, and bounded LRU caching.
`mixed_video_planning` owns deterministic causal-window geometry, latent-view
eligibility and repetition, source weighting, epoch RNG, and source-balanced
global orders. `mixed_video` owns RGB frame-cache lifetime, RGB/latent tensor
selection and padding, latent view assembly, and final sample construction. It
calls the latent repository and window planner directly rather than exposing
dataset-private storage/cache compatibility facades.
Historical catalog, decode, and window-record imports from `mixed_video` remain
identity aliases.
For heterogeneous LeRobot consortium training,
`ConsortiumEpochOrderPlan` freezes member index groups, per-member weights,
the typed weight/random modes, and the sampling seed. It owns fixed-length
largest-remainder allocation, member-local seeded shuffling, and deterministic
global interleaving. `lerobot_consortium_storage` owns source discovery,
local/remote resolution, JSON/JSONL/Parquet reads, and optional local/cloud
write-through caches. `lerobot_consortium_catalog` owns immutable member and
channel contracts, source membership, metadata parsing, local index-snapshot
validation/refresh, and catalog construction. `lerobot_consortium_planning`
owns canonical member-ID resolution, train/validation split membership,
channel-to-slot selection, frame/slot packing decisions, and window geometry
without materializing source rows or tensors. `lerobot_consortium` owns cache
lifetime, structured-row and image decoding, action/state supervision, and
final sample construction. Historical catalog, planning, and storage imports
from `lerobot_consortium` remain identity aliases. This policy is intentionally
distinct from mixed-video source balancing, whose rounded target counts may
change epoch length.
`distributed_sampling` owns rank sharding and epoch coordination; adapters
supply weights or deterministic global index orders without embedding
distributed control flow. Equal-rank padded orders, intentionally unpadded
orders, dataset-span draw keys, and padded-span draw keys are separate
contracts rather than implicit adapter conventions. The same module owns the
stable task/trajectory/start draw primitive used by real and counterfactual
hierarchical datasets; adapters still own eligibility and probability mass.

## Operations Boundary

Open-WAM owns scheduler-agnostic train, eval, sanity, and rollout commands.
Cluster repositories own accounts, partitions, environment modules, scratch
paths, queue monitoring, and retry policy. Scheduler adapters should invoke the
same public commands and must not redefine experiment or model semantics.

## What Not To Extend

The legacy `ActionHead` and `UnifiedWAMPipeline` paradigms are intentionally
not part of the current runtime. New research should extend:

- `PolicyVariant` for method semantics
- `ActionDecoder` for supervised action outputs
- dataset adapters for new data sources
- simulator adapters for new rollout environments
- config enums and static validation for public config choices
