# Cookbook: Add A Simulator Adapter

Use this when a benchmark has a live environment that can be stepped in closed
loop.

## Files To Touch

- `src/open_wam/integrations/`: add an adapter behind lazy imports.
- `src/open_wam/integrations/contracts.py`: extend shared metadata only when
  the existing contract is insufficient.
- `scripts/run_sim_realtime_sandbox.py`: add CLI wiring only if the benchmark
  needs user-facing flags.
- `configs/examples/`: add a tiny config or documented template.
- `docs/cards/`: add a simulator card.
- `tests/`: add fake-adapter contract tests; keep real simulator tests gated.

## Contract

Adapters should provide:

- reset with task/episode/seed inputs
- observation-to-RGB extraction
- observation-to-state extraction
- model-action to environment-action conversion
- step result with reward/done/info
- success predicate
- optional render frame
- close

## Validation

Default CI should use fake adapters only. Real simulator checks belong in
self-hosted, scheduled, or label-gated jobs that upload videos and metrics.

```bash
uv run --extra train pytest tests/test_sim_benchmark.py -q
```
