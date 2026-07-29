# Open-WAM Documentation

Open-WAM is a research framework for studying world-action-model policy
attachments while keeping the shared visual backbone stable. The public docs
focus on reproducible usage, extension points, and benchmark contracts. Internal
engineering notes are not published as part of this site.

## Start Here

- [Quickstart](quickstart.md): install, validate configs, and run CPU-safe smoke checks.
- [Architecture](architecture.md): the stable runtime boundary and core abstractions.
- [Method Families](method_families.md): how the current policy variants fit together.
- [Benchmarks and Data](benchmarks.md): LIBERO, RoboTwin, CALVIN, and synthetic fixtures.
- [Running Experiments](running_experiments.md): train, eval, sanity, and rollout workflows.

## Research Extension

- [Extension SDK](extension_sdk.md): registry surfaces for datasets, policy variants, and decoders.
- [Cookbooks](cookbooks/new_method.md): concrete recipes for adding new research components.
- [Artifacts](artifacts.md): checkpoint manifests, local path aliases, and artifact cards.
- [Reproducibility](reproducibility.md): result envelopes, experiment cards, and tracking policy.
- [M5 GJD vs UVA LIBERO-10](m5_gjd_uva_libero10_comparison.md): task-aligned rollout comparison and route smoke.

## Contributor Operations

- [CLI Reference](cli.md): package-owned commands and legacy script policy.
- [Testing](testing.md): CI tiers, pytest markers, and resource gates.
- [MoT Refactor Characterization](mot_refactor_characterization.md): opt-in
  real-checkpoint training and streaming-inference regression gate.
- [GitHub Pages](github_pages.md): how this site is built and deployed.
- [Release Process](release.md): versioning, packaging checks, and release checklist.
