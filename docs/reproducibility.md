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

`open-wam-eval`, `open-wam-sanity`, and `open-wam-sim-rollout` write the same
versioned envelope when a JSON output is requested. The nested
`open_wam.provenance.v1` record contains the source commit and dirty state,
exact argv, config and resolved-config hashes, checkpoint identity, dataset
metadata hashes, Python/platform details, and model-stack package versions.
Source commit fields are populated only when the imported package is actually
running from that Git checkout; a wheel never borrows identity from an
unrelated checkout in the working directory and instead relies on its package
version plus the release artifact checksum.

Use standard provenance for routine runs. It records checkpoint path, size,
and modification time without reading a multi-gigabyte artifact. Use
`--provenance-mode full` for publication artifacts; it additionally records
the checkpoint SHA-256. Config and discovered dataset metadata are always
hashed.

```bash
open-wam-eval --cfg evaluation.yaml --output-json result.json \
  --provenance-mode full
```

Use [experiment_cards.md](experiment_cards.md) for architecture/program result cards
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
  "artifacts": {},
  "provenance": {
    "schema_version": "open_wam.provenance.v1",
    "mode": "standard"
  }
}
```

When changing result schemas, write both old and new fields for one
compatibility window. Remove legacy fields only in a later legacy-removal PR.

## WandB

WandB is optional. When enabled, use stable naming:

- project: `openwam-<benchmark-or-workload>`
- group: `<benchmark>/<architecture>/<program-or-profile>`
- run name: `<config-name>_<short-commit>_<timestamp-or-step>`
- tags: architecture, program, benchmark, dataset type, checkpoint source
