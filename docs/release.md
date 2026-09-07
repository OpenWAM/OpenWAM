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
- role-specific `open_wam.sdk` exports

## Versioning Policy

- Patch: bug fixes, docs, new cards, compatible config aliases.
- Minor: new methods, datasets, decoders, optional extras, or compatible CLI
  additions.
- Major: removing deprecated config fields, changing result schemas, changing
  checkpoint layout expectations, or changing method semantics.

## Release Checklist

```bash
OPEN_WAM_CI_NO_TORCH=1 python scripts/ci_basic_sanity.py
uv run --frozen --group audit python scripts/check_dependency_audit.py
python scripts/check_release_metadata.py --release
python scripts/validate_configs_static.py configs/experiments configs/evals configs/examples --quiet
python scripts/build_docs_site.py --output .docs_site
mkdocs build --strict
python -m build
python scripts/check_release_metadata.py --dist-dir dist
python -m twine check dist/*
```

## Distribution Resources

The wheel includes the package plus read-only canonical configs, examples,
local-path samples, extension templates, the consortium metadata snapshot, and
third-party license material.
Config resolution prefers explicit files and source-checkout configs, then
falls back to packaged resources. The source distribution is also a bounded
public artifact, not a repository snapshot. Its allowlist contains:

- package source;
- public configs;
- public docs;
- extension templates and the consortium metadata snapshot;
- required project metadata;
- project and third-party license/notice files.

Hatch always places the root `.gitignore` in source distributions. Open-WAM
therefore includes that generic file explicitly in the allowlist and scans it
with the rest of the public text surface.

The release is AGPL-3.0-only. Redistributed copies also preserve the
attribution in `NOTICE`; scholarly citation guidance is machine-readable in
`CITATION.cff`.

Packaged consortium snapshots contain public repository metadata only. The
release check rejects private records and drift between the repo list,
inventory, and contract catalog.

Checkout-only research scripts, tests, baselines, caches, and machine-local
config are intentionally excluded. The four generated files under
`notes/index/` remain because the consortium data adapter consumes that
bounded metadata contract at runtime. Both basic CI and
`scripts/check_release_metadata.py` enforce the allowlist and reject private
mount paths in the selected text files.

Installed-package experiment docs use package CLIs and packaged configs.
Commands under `scripts/` are labeled source-checkout integrations and must
not be the only documented way to train a packaged model.

Before tagging:

- `CHANGELOG.md` is updated.
- Artifact and experiment cards list `last_validated_commit` or explicitly
  state that validation is pending.
- Static CI and minimal-package CI pass.
- Any CPU/GPU/simulator validation claims are linked in cards.
- Deprecations are documented before removals.
- GitHub private vulnerability reporting is enabled for the public repository,
  and the reporting URL in `SECURITY.md` is tested while logged out.
- The frozen full environment passes the dependency audit. Any exception is
  exact-version bound, justified in `.github/dependency-audit-exceptions.toml`,
  and unexpired.
- At least one non-fixture model artifact has a public HTTPS URL, SHA-256, and
  license; the synthetic fixture does not satisfy this gate.

Artifact build and Twine checks are release or manually triggered CI. The
source-archive allowlist and private-path checks are part of the default static
PR tier.
