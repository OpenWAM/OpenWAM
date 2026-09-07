# Public Tiny Fixture

This fixture is intentionally structural. It is small enough to keep in the
repo and exists to exercise public contracts without private datasets,
checkpoints, GPUs, or simulators.

Use it with:

```bash
openwam-validate-config configs/examples/public_tiny_synthetic_contract.yaml
```

Torch-backed train/eval smoke tests should use this fixture only in a gated CPU
tier, not in default static PR CI.
