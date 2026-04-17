# Cookbook: Add A New Dataset

Use this when raw storage, metadata, action/state keys, or camera layout differ
from existing adapters.

## Files To Touch

- `src/open_wam/data/`: add a dataset adapter that returns `WAMSample`.
- `src/open_wam/data/factory.py`: register the adapter by `data.dataset_type`.
- `configs/examples/`: add one public or template config.
- `configs/local_paths.sample.yaml`: add placeholder aliases only when local
  roots are needed.
- `tests/fixtures/`: add tiny public structural fixtures when possible.
- `tests/`: add adapter and action-mapping tests.

## Contract

Dataset-specific parsing stays inside the adapter. Public outputs stay uniform:

- canonical RGB views keyed by model camera names
- `actions` and optional `action_mask`
- `state` and optional `state_mask`
- `task_text`
- metadata explaining source episode/window identity

Canonical RGB layout construction belongs in the data layer, not in the visual
backbone.

## Validation

```bash
open-wam-validate-config configs/examples/<dataset_sanity>.yaml
uv run --extra train pytest tests/<dataset_test>.py -q
```

If the dataset needs private paths, tests should skip with an actionable
message unless a public fixture is being used.
