# Baselines

This directory contains integrations for external baselines that should remain
outside the OpenWAM method stack. Baseline runners may use OpenWAM utilities
for datasets, simulators, local path handling, and reporting, but they should
not add method-specific branches to `VariantPipeline`, `VisualTower`,
`PolicyVariant`, or `ActionDecoder`.

Current baselines:

- `lingbot_va/`: read-only upstream LingBot-VA LIBERO-LONG checkpoint runner
  for `libero_10` baseline evaluation, plus upstream-style RobotWin server/client
  wrappers for the released RobotWin checkpoint.
