# Baselines

This directory contains integrations for external baselines that should remain
outside the Open-WAM method stack. Baseline runners may use Open-WAM utilities
for datasets, simulators, local path handling, and reporting, but they should
not add method-specific branches to `VariantPipeline`, `VisualTower`,
`PolicyVariant`, or `ActionDecoder`.

Current baselines:

- `lingbot_va/`: read-only LingBot-VA LIBERO-10 rollout runner using Heng's
  original `VA_Server` semantics.
