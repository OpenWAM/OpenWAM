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
| `PolicyVariant` | Defines method semantics: required visual stages, train inputs, infer state, and rollout-step behavior. |
| `ActionDecoder` | Converts variant outputs into supervised action predictions and losses. |

## Design Principles

- Keep method differences in policy variants, runtime programs, cache policy,
  sequence semantics, schedulers, and decoders.
- Keep dataset-specific parsing inside dataset adapters selected by
  `data.dataset_type`.
- Keep canonical RGB layout construction in the data layer.
- Keep public finite choices enum-backed at the typed config boundary.
- Do not add method-named infrastructure when the abstraction is generic.

## Configuration Contract

`open_wam.configs.coercion` owns reusable YAML/CLI-to-type conversion.
`open_wam.configs.sequence_contracts` owns defaults and cross-section
validation for sequence semantics. The YAML loader maps sections into frozen
dataclasses and retains compatibility imports, but it does not own policy
contract behavior.

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
- `build_action_grid_ids_for_sequence` owns the frame-aligned action
  coordinates shared by training and recurrent inference.
- `mot.generalist_modes` owns GJD mode selection and conditional tensor
  rewrites.
- `mot.runtime_routing` owns finite inference routes, coupling policy, and
  validated rollout overrides. Its implementation is configuration-only and
  contains no tensor execution.
- `mot.runtime` owns MoT tensor execution and cache mechanics, including
  explicit-sigma flow integration and speculative action-cache rewind.
- common attention profiles own token visibility; the policy selects a profile
  and supplies its resolved layout.

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
local LeRobot latent adapter, `lerobot_v2_latent_storage` owns repository
discovery, metadata, filenames, and payload reshaping;
`lerobot_v2_latent` owns eligible-window geometry and public
`LatentWAMSample` construction.
`distributed_sampling` owns rank sharding and epoch coordination; adapters
supply weights or deterministic global index orders without embedding
distributed control flow. Equal-rank padded orders, intentionally unpadded
orders, dataset-span draw keys, and padded-span draw keys are separate
contracts rather than implicit adapter conventions.

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
