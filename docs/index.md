# OpenWAM Documentation

<p align="center">
  Heng&nbsp;Yu<sup>*</sup>, David&nbsp;D.&nbsp;Yuan<sup>*</sup>, Juze&nbsp;Zhang<sup>*</sup>, Changan&nbsp;Chen, Yao&nbsp;Feng,<br>
  Michelle&nbsp;Baldonado, Steve&nbsp;Cousins, Li&nbsp;Fei-Fei, Jiajun&nbsp;Wu, Ehsan&nbsp;Adeli
</p>
<p align="center">Stanford University<br><sup>*</sup> Equal contribution</p>

<div class="openwam-affiliations" aria-label="Stanford affiliations">
  <a href="https://www.stanford.edu/"><img class="openwam-affiliation-wordmark" src="assets/affiliations/stanford-wordmark.png" alt="Stanford University" height="20"></a>
  <a href="https://ai.stanford.edu/"><img class="openwam-affiliation-sail" src="assets/affiliations/stanford-ai-lab.jpg" alt="Stanford Artificial Intelligence Laboratory" height="28"></a>
  <a href="https://svl.stanford.edu/"><img class="openwam-affiliation-svl" src="assets/affiliations/stanford-svl.png" alt="Stanford Vision and Learning Lab" height="28"></a>
  <a href="https://stai.stanford.edu/"><img class="openwam-affiliation-stai" src="assets/affiliations/stanford-stai.png" alt="Stanford Translational AI (STAI) Lab" height="32"></a>
  <a href="https://src.stanford.edu/"><img class="openwam-affiliation-src" src="assets/affiliations/stanford-src.webp" alt="Stanford Robotics Center" height="32"></a>
</div>

OpenWAM is an extensible and composable library for training and evaluating world action
models while keeping the shared visual backbone stable. Start with a CPU example, then choose a training, evaluation, or extension workflow.

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

- [Research paper (PDF)](https://arxiv.org/pdf/2610.07922): framework, methods, and experimental results.
- [CLI Reference](cli.md): commands and examples.
- [Rollout Contracts](rollout_contracts.md): sessions, executed history, and control integration.
- [Testing](testing.md): local tests and numerical comparisons for extensions.
- [Releases](release.md): package versions and reproducible installation.
