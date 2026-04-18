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

## Visual Tower Contract

The shared visual stack exposes stage-aware outputs rather than allowing policy
variants to reach into arbitrary backbone internals. Common stage families are:

- current visual features for action-conditioned policies
- post-core token or latent features for feature-attached policies
- generated future visual features for video-conditioned action heads
- decode-stage outputs for post-decoded baselines

Policy variants request stages through `required_visual_stages()` and consume
prepared inputs through explicit variant contracts.

## Data Contract

Dataset adapters normalize raw datasets into one public batch contract:

- canonical RGB tensors with a configured camera/layout policy
- action tensors with explicit source and model dimensions
- optional state tensors
- text/task metadata when available
- adapter metadata that documents action mapping and benchmark identity

This lets LIBERO, RoboTwin, CALVIN, synthetic fixtures, and future datasets use
the same train/eval/runtime stack.

## What Not To Extend

The legacy `ActionHead` and `UnifiedWAMPipeline` paradigms are intentionally
not part of the current runtime. New research should extend:

- `PolicyVariant` for method semantics
- `ActionDecoder` for supervised action outputs
- dataset adapters for new data sources
- simulator adapters for new rollout environments
- config enums and static validation for public config choices
