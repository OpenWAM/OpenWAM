# Release Hygiene

OpenWAM versions public interfaces more strictly than internal research code.

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

## PyPI SDK And Installation Aliases

`openwam` is the canonical distribution; `open_wam` is the Python import.
The official `open-wam`, `openwam-sdk`, and `open-wam-sdk` metapackages install
the exact same version of `openwam`. Each forwards every canonical extra,
including `[train]`, `[eval]`, and `[pretrain]`. They contain no Python modules
or console scripts, so installing them together or uninstalling an alias
does not overwrite or remove the implementation.

PyPI normalizes underscores and periods to hyphens: `open_wam` installs
`open-wam`, not `openwam`. Prefer the canonical name in new documentation and
requirements. These are functional installation aliases, not empty name
reservations. Account registration, TestPyPI publication, and pending Trusted
Publishers do not reserve names on the production index.

The first PyPI publication is a separate release operation; merging the
packaging workflow does not mean the packages are already available. Build
and exercise the release locally before publishing:

```bash
python -m pip install 'build>=1.2,<2' 'twine>=6' 'tomli-w>=1,<2' PyYAML pytest packaging
python scripts/build_pypi_distributions.py --out-dir dist
python scripts/check_release_metadata.py --dist-dir dist/core
python scripts/check_release_metadata.py --dist-dir dist/aliases
python -m twine check --strict dist/core/* dist/aliases/*
python -m pip download --only-binary=:all: --dest dist/dependencies dist/core/*.whl
OPENWAM_DISTRIBUTIONS_DIR="$PWD/dist" python -m pytest -q tests/test_pypi_distributions.py
```

Use a fresh output directory. The builder reads version, extras, and shared
metadata from `pyproject.toml`; alias definitions do not duplicate dependency
lists. Each wheel is built from its source distribution. Installation tests
use isolated environments outside the checkout, check all four spellings and
co-installation, inspect every extra, and exercise all installed CLI parsers.
They do not establish GPU parity for a newly resolved dependency stack.

### Publisher Setup

On PyPI, verify the maintainer account's email, enable 2FA, and configure
pending GitHub Trusted Publishers under **Account > Publishing**. Use owner
`OpenWAM`, repository `OpenWAM`, and workflow filename `publish-pypi.yml` for
each project, with distinct environments:

| Project / workflow `project` input | PyPI environment | TestPyPI environment |
| --- | --- | --- |
| `openwam` | `pypi` | `testpypi` |
| `open-wam` | `pypi-open-wam` | `testpypi-open-wam` |
| `openwam-sdk` | `pypi-openwam-sdk` | `testpypi-openwam-sdk` |
| `open-wam-sdk` | `pypi-open-wam-sdk` | `testpypi-open-wam-sdk` |

PyPI rejects identical pending-publisher identities across project names and
allows at most three pending publishers per account at once. Register and
publish `openwam` first, then register the remaining aliases as slots become
available. Publishing only `openwam` and `open-wam` is supported; unselected
projects require no publisher or GitHub environment. An existing `OpenWAM`
entry already covers `openwam`; scope its environment to `pypi`, not `(Any)`.
These pending-publisher restrictions are enforced by
[PyPI's registration handler](https://github.com/pypi/warehouse/blob/main/warehouse/accounts/views.py).

Repeat on TestPyPI using its separate account and the environments above.
For existing projects, add publishers in their project settings instead.
Add a trusted backup project Owner after the first publication. No PyPI API
token or password is stored in GitHub: publishing uses OIDC.

Before publishing a selected project, create its matching GitHub
environment in `OpenWAM/OpenWAM`, require maintainer approval, and restrict
deployments to `v*` tags. GitHub environment protections are repository
settings, not created by the workflow YAML. Protect version tags against
replacement and deletion. See [PyPI's publisher setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

### Release Procedure

1. Merge and validate the public changes. Set the intended new version in
   `pyproject.toml`, update `CITATION.cff`, `CHANGELOG.md`, and the attribution
   version in `NOTICE` and its release check together. Use a PEP 440 suffix
   such as `0.2.0a1` for an alpha; a GitHub prerelease flag alone does not mark
   a PyPI version as a prerelease. Do not reuse the existing `v0.1.0` tag for
   newer code. Date the release metadata when actually preparing the release.
2. Run the release checklist, including the dependency audit. Review dependency
   bounds and the supported Python/CUDA environment. `pip install` does not
   apply `uv.lock`; exact model reproduction still uses the frozen checkout.
   Do not waive security or numerical parity checks to obtain project names.
3. Tag the reviewed production commit with `v<project.version>`. In Actions,
   run **publish-pypi** against that tag with `index=testpypi`, `project=openwam`.
   The workflow rejects staging/branch publication, checks tag/version agreement and main
   ancestry, and runs the package and dependency gates before approval.
4. Inspect the artifacts and approve the matching TestPyPI environment. Once
   `openwam` succeeds, dispatch again for each desired alias, using the same
   tag and index with its `project` value. Install the candidate using
   TestPyPI **only** for the selected OpenWAM distributions;
   provision third-party dependencies from the normal index separately. Avoid
   mixing indexes with `--extra-index-url` when verifying package provenance.
5. Run the same tag with `index=pypi`, `project=openwam`, and approve production
   publication. Wait for success before dispatching each desired alias. The
   upload job has no checkout or build step: it selects only the requested
   project's wheel and source archive from the tested artifacts. Building and
   testing all aliases does not publish them.
6. Verify the selected project pages, versions, extras, and fresh installs. Record
   the release and supported environment in the public documentation.

Each dispatch publishes one project. To start with the two installation spellings,
publish `project=openwam`, then `project=open-wam`; SDK aliases can wait.
If interrupted, rerun the same project, index, and tag; existing files are
skipped so the remaining uploads can finish. Do not move
the tag or silently replace a bad release: inspect published files and use a
new version for corrections. Publication only starts on manual dispatch;
neither a main push nor a tag push uploads anything by itself.

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

Hatch always places the root `.gitignore` in source distributions. OpenWAM
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
- Any model advertised as public has an HTTPS URL, SHA-256, and license.
  Unpublished model entries keep all three fields null.

Artifact build and Twine checks are release or manually triggered CI. The
source-archive allowlist and private-path checks are part of the default static
PR tier.
