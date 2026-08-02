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

## Source Archive

The wheel contains only `src/open_wam`. The source distribution is also a
bounded public artifact, not a repository snapshot. Its allowlist contains:

- package source;
- public configs;
- public docs;
- extension templates;
- required project metadata.

Checkout-only deployment, research scripts, tests, baselines, notes, caches,
and machine-local config are intentionally excluded. Both basic CI and
`scripts/check_release_metadata.py` enforce the allowlist and reject private
mount paths in the selected text files.

Before tagging:

- `CHANGELOG.md` is updated.
- Artifact and experiment cards list `last_validated_commit` or explicitly
  state that validation is pending.
- Static CI and minimal-package CI pass.
- Any CPU/GPU/simulator validation claims are linked in cards.
- Deprecations are documented before removals.

Artifact build and Twine checks are release or manually triggered CI. The
source-archive allowlist and private-path checks are part of the default static
PR tier.
