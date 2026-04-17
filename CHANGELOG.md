# Changelog

Open-WAM follows semantic-versioned public surfaces for configs, CLI flags,
result schemas, artifact manifests, and checkpoint layout expectations.

## 0.1.0 - Unreleased

### Added

- Open-source readiness docs and issue/PR templates.
- Static no-Torch CI tier.
- Minimal-core dependency plan and package import-safety work.
- Static config validator entrypoint: `open-wam-validate-config`.
- Public tiny synthetic contract fixture.
- Artifact and experiment card templates.

### Changed

- Heavy runtime dependencies are moving behind optional extras.
- Package and CLI imports should stay dependency-light until runtime execution.

### Deprecated

- Legacy `action_head` config sections remain accepted but should be migrated to
  `policy_variant` plus `action_decoder`.

### Removed

- Nothing.

### Fixed

- Result envelopes protect reserved schema keys from legacy metadata
  collisions.
- Deployment recording imports no longer require OpenCV at collection/import
  time.
