# Changelog

OpenWAM follows semantic-versioned public surfaces for configs, CLI flags,
result schemas, artifact manifests, and checkpoint layout expectations.

## 0.2.0 - 2026-09-17

This is an explicit pre-1.0 compatibility boundary, not a drop-in patch for
0.1.1. See the [migration guide](docs/migration_0_2.md). Model weights,
datasets, and simulator installations remain separate artifacts.

### Changed

- Unified video/action inference under shared denoising, layout, history, and
  call-local feature-cache owners for Parallel Stream, Dual Expert, GJD,
  conditional dynamics, and chunk-conditioned causal video.
- Maintained closed-loop rollouts share a typed planner, immutable session,
  control lifecycle, and engine. Benchmark I/O stays outside model execution.
- Inference uses explicit per-stream CFG and stage-ordered noise draws.
  Equal seeds do not imply identical traces from retired runners. Parallel
  inference preserves the generated chunk extent instead of interpolating
  actions to the training horizon; training decoding is unchanged.
- Single-rank `trainer.strategy=fsdp` now uses FSDP, including configured mixed
  precision and optional CPU offload. Select `single_device` explicitly to
  retain the previous unwrapped execution path.

### Breaking SDK And Rollout Changes

- `PolicyInferContext.extra` is replaced by typed input fields.
  `PolicyInferState` is immutable; its cursor owns `step_index`, and the generic
  `cache` field is removed.
- Simulator adapters return `ControlTransition` instead of
  `SimulatorStepResult` and implement `materialize_control` returning both
  executable and dataset-source actions in a `ControlCommand`.
- Decoder rollout commits return new state instead of mutating their input.
  The default plan consumes a single environment's full predicted action chunk,
  not the legacy `aux["current_action"]` convention.
- The architecture-owned exact runner and old private inference executors are
  retired. Use the shared policy runner and maintained LIBERO policy commands.
- Realtime fallback-freeze/quarantine options are removed. Actually executed
  controls and observations enter history; blocking and fixed-rate scheduling
  remain explicit alternatives, not equivalent replacements for hiding fallback.

### Fixed

- Single-rank FSDP no longer silently ignores CPU offload; process-group
  ownership and teardown are explicit.
- Explicit CPU FSDP training retains a CPU device mesh even on CUDA hosts,
  rather than letting PyTorch choose an accelerator implicitly.
- LIBERO action-budget termination retains its budget reason rather than
  reporting it as native environment termination.
- GPU architecture tests distinguish the Parallel training horizon from the
  shorter inference chunk.

### Security And Validation

- The characterized third-party dependency lock is unchanged. The proposed
  Torch/Setuptools upgrade is deferred, not a vulnerability fix.
- Existing dependency-audit exceptions retain their scope and expiry.
  The maintainer renewed the Accelerate acceptance for 0.2.0; reassess before
  2026-10-09. The other current exceptions expire 2026-11-06. Only trusted artifacts are
  supported under the [artifact-trust policy](SECURITY.md#artifact-trust).
- Release validation passed 3,551 distinct tests plus 18 subtests, including
  all 38 fresh-cache two-rank FSDP cases. A full VTA checkpoint completed one
  representative LIBERO rollout successfully using split frontend placement;
  this is not an aggregate benchmark claim. Another 44 external-asset/opt-in
  tests were not run. Numerical reproducibility requires the recorded dependency
  and hardware stack, not just a matching package version.

## 0.1.1 - 2026-09-09

### Added

- PyPI release tooling for the `openwam` SDK and exact-version installation
  aliases, with isolated installation tests and project-scoped Trusted
  Publishers. The initial publication targets `openwam` and `open-wam`.
- Condensed public change snapshots in staging-to-production promotion PRs.

### Security

- Documented, maintainer-authorized acceptance of the malicious-checkpoint
  risk in `accelerate==1.13.0` / `CVE-2026-69112`, expiring 2026-10-09.
  This is not a vulnerability fix; see [Artifact Trust](SECURITY.md#artifact-trust).
  The dependency audit remains enforced, and the base SDK does not install
  Accelerate. Model code and third-party dependency versions are unchanged.

## 0.1.0 - 2026-09-07

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

- Training checkpoint operations are explicit: the historically ambiguous
  `--checkpoint-root` option now fails with migration guidance to
  `--initialize-weights-from` or `--resume-from`.
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

- Full-state checkpoints preserve an explicit next sampler/batch cursor, reject
  partially accumulated gradients, and resolve standard run roots through
  their `checkpoints/` directory. Older full-state files without the explicit
  cursor remain usable for weight initialization but are rejected for resume.
- Distributed checkpoint publication reports rank-zero filesystem failures to
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
