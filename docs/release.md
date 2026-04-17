# Release Hygiene

Open-WAM versions public interfaces more strictly than internal research code.

## Public Surfaces

Treat these as compatibility-managed:

- package version
- config schema and enum names
- CLI command names and stable flags
- result envelope schema
- artifact manifest fields
- checkpoint layout expectations
- documented benchmark adapter contracts

## Versioning Policy

- Patch: bug fixes, docs, new cards, compatible config aliases.
- Minor: new methods, datasets, decoders, optional extras, or compatible CLI
  additions.
- Major: removing deprecated config fields, changing result schemas, changing
  checkpoint layout expectations, or changing method semantics.

## Release Checklist

```bash
OPEN_WAM_CI_NO_TORCH=1 python scripts/ci_basic_sanity.py
python scripts/validate_configs_static.py configs/experiments configs/evals configs/examples --quiet
python -m build
python -m twine check dist/*
```

Before tagging:

- `CHANGELOG.md` is updated.
- Artifact and experiment cards list `last_validated_commit` or explicitly
  state that validation is pending.
- Static CI and minimal-package CI pass.
- Any CPU/GPU/simulator validation claims are linked in cards.
- Deprecations are documented before removals.

Packaging checks are release or manually triggered CI. They are not part of the
default static PR tier.
