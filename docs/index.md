# OpenWAM Documentation

<p align="center">
  <a href="https://www.stanford.edu/"><img src="assets/affiliations/stanford-wordmark.png" alt="Stanford University" height="45" hspace="14" valign="middle"></a>
  <a href="https://ai.stanford.edu/"><img src="assets/affiliations/stanford-ai-lab.jpg" alt="Stanford Artificial Intelligence Laboratory" height="74" hspace="14" valign="middle"></a>
  <a href="https://svl.stanford.edu/"><img src="assets/affiliations/stanford-svl.png" alt="Stanford Vision and Learning Lab" height="74" hspace="14" valign="middle"></a>
</p>

<p align="center"><strong>OpenWAM is developed by the OpenWAM Team in the <a href="https://svl.stanford.edu/">Stanford Vision and Learning Lab (SVL)</a>, a research group of the <a href="https://ai.stanford.edu/">Stanford Artificial Intelligence Laboratory (SAIL)</a> at <a href="https://www.stanford.edu/">Stanford University</a>.</strong></p>

OpenWAM is an extensible library for training and evaluating world action
models while keeping the shared visual backbone stable. The public docs focus
on reproducible usage, typed extension points, and benchmark contracts.

## Start Here

- [Quickstart](quickstart.md): install, validate configs, and run CPU-safe smoke checks.
- [Architecture](architecture.md): the stable runtime boundary and core abstractions.
- [Compatibility](compatibility.md): supported platforms, SDK surface, and numerical gates.
- [Policy Architectures And Programs](policy_architectures.md): how topology,
  conditioning programs, and decoders compose.
- [Benchmarks and Data](benchmarks.md): LIBERO, RoboTwin, CALVIN, and synthetic fixtures.
- [Training and Inference](running_experiments.md): maintained train, resume,
  eval, sanity, and rollout commands.

## Extension Points

- [Extension SDK](extension_sdk.md): registry surfaces for datasets, policy variants, and decoders.
- [Cookbooks](cookbooks/new_policy_architecture.md): concrete recipes for adding new research components.
- [Artifacts](artifacts.md): checkpoint manifests, local path aliases, and artifact cards.
- [Reproducibility](reproducibility.md): result envelopes, experiment cards, and tracking policy.

## Contributor Operations

- [CLI Reference](cli.md): package-owned commands and legacy script policy.
- [Testing](testing.md): CI tiers, pytest markers, and resource gates.
- [DualExpert Refactor Characterization](dual_expert_refactor_characterization.md): opt-in
  real-checkpoint training and streaming-inference regression gate.
- [GitHub Pages](github_pages.md): how this site is built and deployed.
- [Release Process](release.md): versioning, packaging checks, and release checklist.
