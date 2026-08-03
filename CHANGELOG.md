# Changelog

Open-WAM follows semantic-versioned public surfaces for configs, CLI flags,
result schemas, artifact manifests, and checkpoint layout expectations.

## 0.1.0 - Unreleased

### Added

- Stable package commands for training, evaluation, config inspection and
  validation, sanity checks, and simulator rollout.
- Typed extension contracts for datasets, policy variants, action decoders,
  attention/runtime programs, and simulator backends.
- Public tiny synthetic fixtures, artifact manifests, experiment cards, and
  exact training/inference characterization tools.
- Dependency-light CI, package-boundary tests, static config validation, and a
  bounded source-distribution policy.

### Changed

- The maintained runtime is organized around
  `ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder`.
- Training, checkpointing, evaluation, rollout, data adaptation, attention,
  and cache execution now have explicit package owners and typed contracts.
- Heavy model, training, simulator, visualization, and deployment dependencies
  are isolated behind optional extras; the base install requires only PyYAML.
- Finite public configuration choices are enum-backed at the typed config
  boundary while experiment YAMLs remain string-friendly.

### Deprecated

- Legacy `action_head` config sections remain accepted but should be migrated to
  `policy_variant` plus `action_decoder`.
- Legacy root scripts remain only as compatibility adapters where a maintained
  package command exists.

### Removed

- Private cluster orchestration and machine-specific launch supervision.
- Superseded Method 2/3 implementations, unreachable model prototypes, and
  stale deployment diagnostics.
- Duplicate launchers and research-only utilities without a maintained runtime
  owner.

### Fixed

- Result envelopes protect reserved schema keys from legacy metadata
  collisions.
- Deployment recording imports no longer require OpenCV at collection/import
  time.
- Four-rank FSDP synchronization includes trainable root parameters outside
  leaf block stacks.
- Installed commands resolve package-owned runtimes without depending on a
  source checkout.
