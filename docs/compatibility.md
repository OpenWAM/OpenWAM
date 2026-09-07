# Compatibility Matrix

OpenWAM is pre-1.0 Linux research software. Compatibility claims are split
between API compatibility and exact numerical characterization.

## Maintained Matrix

| Surface | Maintained contract |
| --- | --- |
| Operating system | Linux |
| Python | 3.11 and 3.12 |
| CPU semantic tests | both Python versions, dependencies from `uv.lock` |
| GPU characterization | documented CUDA host stack and immutable golden artifacts |
| Simulator integrations | fake-adapter CI; real environments on documented external stacks |
| Python SDK | `open_wam.sdk.config`, `.data`, `.policy`, `.simulator`, `.results` |

Installed distributions include a PEP 561 `py.typed` marker. Static type
information is therefore available to downstream packages without a separate
stub distribution; the stable import boundary remains the role-specific SDK.

`pyproject.toml` dependency ranges describe install compatibility. Exact
training gradients and rollout outputs are characterized only with the locked
dependency graph and the hardware/software stack recorded by the
characterization report. Use `uv sync --frozen` when reproducing those claims.
Changing PyTorch, Diffusers, Transformers, CUDA, attention backends, or GPU
models requires re-characterization even when installation remains supported.

The supported Diffusers range is intentionally capped below `0.38`. That
release changed WAN RMS normalization precision and does not reproduce the
characterized bf16 checkpoint outputs. OpenWAM's lock currently selects
`0.37.1`; raising the cap is a numerical migration and requires the real-model
GPU characterization gate, not only an import or unit test.

## Required Gates

The required CPU gate includes unmarked tests; marker selection cannot silently
exclude the core suite:

```bash
uv sync --frozen --group dev --extra full
CUDA_VISIBLE_DEVICES="" uv run pytest --strict-markers -q \
  -m "not (gpu or sim or data or slow)"
```

Model-facing changes additionally run the immutable GPU fixture-replay,
training-step, and inference-step gates described in
[Dual-Expert Refactor Characterization](dual_expert_refactor_characterization.md).
Real simulator and private-data tests are resource gates and are never implied
by the CPU suite.

## Compatibility Policy

Typed config fields, checkpoint compatibility, result schemas, and SDK exports
are fail-closed. The low-level checkpoint loader is strict by default. Standard
evaluation accepts a checkpoint superset: every current runtime tensor must be
present with the exact shape, while checkpoint-only tensors from a removed
optional component are reported and ignored. This is distinct from
`--allow-partial-checkpoint`, which may leave current runtime tensors missing
and is only an explicit migration diagnostic. New finite config choices use
enums, while extension identifiers and source names remain open strings.

Historical broad package facades remain available during the pre-1.0
migration. They are not a guarantee that every implementation helper is a
stable API. Deprecations are documented before removal.
