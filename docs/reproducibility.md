# Reproducibility

Every public result should be traceable to:

- git commit
- command
- experiment or eval config
- checkpoint artifact id or local path alias
- dataset artifact id or local path alias
- benchmark adapter
- device
- random seed
- result schema version

Use [experiment_cards.md](experiment_cards.md) for method-family result cards
and `configs/artifacts.sample.yaml` for artifact layout metadata.

## Result Schema

New structured result files should include:

```json
{
  "schema_version": "open_wam.result.v1",
  "command": "open-wam-eval",
  "config": "configs/evals/<evaluation>.yaml",
  "checkpoint": null,
  "benchmark": null,
  "device": "cpu",
  "seed": 0,
  "metrics": {},
  "artifacts": {}
}
```

When changing result schemas, write both old and new fields for one
compatibility window. Remove legacy fields only in a later legacy-removal PR.

## WandB

WandB is optional. When enabled, use stable naming:

- project: `openwam-<benchmark-or-method>`
- group: `<method-family>/<benchmark>/<dataset-or-artifact>`
- run name: `<config-name>_<short-commit>_<timestamp-or-step>`
- tags: method family, benchmark, dataset type, checkpoint source
