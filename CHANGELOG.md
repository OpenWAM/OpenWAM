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
- Role-scoped `open_wam.sdk` modules for extending config, data, policy,
  decoder, simulator, and result contracts without importing implementation
  internals.
- Versioned runtime provenance in structured evaluation, sanity, and simulator
  result envelopes.
- Dependency-light CI, package-boundary tests, static config validation, and a
  bounded source-distribution policy.
- A required Python 3.11/3.12 semantic-contract CI gate that includes unmarked
  tests while excluding only explicitly marked hardware/data suites.

### Changed

- lerobot_v2_latent_local dataset preflight now discovers every LeRobot
  bundle under the configured local_root before model construction; it
  raises DatasetArtifactPreflightError when no bundle can be found rather
  than deferring the failure to first-batch dataloader construction. Configs
  with non-standard roots that previously failed lazily now fail loudly at
  preflight.
- The maintained runtime is organized around
  `ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder`.
- Training, checkpointing, evaluation, rollout, data adaptation, attention,
  and cache execution now have explicit package owners and typed contracts.
- Heavy model, training, simulator, visualization, and deployment dependencies
  are isolated behind optional extras; the base install requires only PyYAML.
- Canonical configs, examples, evaluation wrappers, and extension templates
  are installed package resources and resolve independently of the caller's
  working directory.
- Runtime checkpoint loading is key-strict by default; partial loading requires
  an explicit migration-diagnostic policy.
- Simulator construction is registry-driven and receives only typed,
  immutable factory context rather than CLI implementation state.
- Finite public configuration choices are enum-backed at the typed config
  boundary while experiment YAMLs remain string-friendly.

### Deprecated

- Legacy root scripts remain only as compatibility adapters where a maintained
  package command exists.

### Removed

- Private cluster orchestration and machine-specific launch supervision.
- Legacy `action_head`, `post_latent`, and `post_decoded` policy surfaces;
  policies now cross the explicit `policy_variant` plus `action_decoder`
  boundary.
- Contract-only presets, the sampled-evaluation study, duplicate superseded
  configs, and completed roadmap archives.
- Superseded traditional Method 2 and experimental Method 3 implementations,
  unreachable model prototypes, and stale deployment diagnostics.
- Duplicate launchers and research-only utilities without a maintained runtime
  owner.

### Fixed

- Empty `--checkpoint-root` inputs fail before model and CUDA initialization;
  distributed checkpoint publication reports rank-zero filesystem failures to
  peers and uses the configured distributed timeout.
- Result envelopes protect reserved schema keys from legacy metadata
  collisions.
- Deployment recording imports no longer require OpenCV at collection/import
  time.
- Four-rank FSDP synchronization includes trainable root parameters outside
  leaf block stacks.
- Tensor-bearing artifacts use restricted deserialization by default, with the
  sole legacy NumPy-pickle boundary requiring an explicit trust policy.
- Result JSON writes are atomic and protect the versioned envelope contract.
- Installed commands resolve package-owned runtimes without depending on a
  source checkout.
