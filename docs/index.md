# OpenWAM Documentation

<p align="center">
  Developed by the <strong>OpenWAM Team</strong> at the
  <a href="https://svl.stanford.edu/"><strong>Stanford Vision and Learning Lab (SVL)</strong></a>.
</p>

<div class="openwam-affiliations" aria-label="Stanford affiliations">
  <a href="https://www.stanford.edu/"><img class="openwam-affiliation-wordmark" src="assets/affiliations/stanford-wordmark.png" alt="Stanford University" height="20"></a>
  <a href="https://ai.stanford.edu/"><img class="openwam-affiliation-sail" src="assets/affiliations/stanford-ai-lab.jpg" alt="Stanford Artificial Intelligence Laboratory" height="28"></a>
  <a href="https://svl.stanford.edu/"><img class="openwam-affiliation-svl" src="assets/affiliations/stanford-svl.png" alt="Stanford Vision and Learning Lab" height="28"></a>
  <a href="https://src.stanford.edu/"><img class="openwam-affiliation-src" src="assets/affiliations/stanford-src.webp" alt="Stanford Robotics Center" height="32"></a>
</div>

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

## Reference

- [CLI Reference](cli.md): commands and examples.
- [Testing](testing.md): local tests and numerical comparisons for extensions.
- [Releases](release.md): package versions and reproducible installation.
