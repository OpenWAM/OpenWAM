# GitHub Pages Documentation Site

Open-WAM publishes documentation through a generated MkDocs source tree. The
tracked public docs live in `docs/`; research notes live in `notes/` and are
staged into a sanitized `engineering-notes/` section before publication.

## Local Preview

```bash
uv sync --extra docs
uv run --extra docs python scripts/build_docs_site.py --output .docs_site
uv run --extra docs mkdocs serve
```

The generated `.docs_site/` directory and final `site/` directory are
gitignored. Rebuild `.docs_site/` after editing `docs/` or `notes/`.

## Publication Flow

The `pages` GitHub Actions workflow runs on pushes to `main` and can also be
started manually. It:

- installs only MkDocs, not the Open-WAM package
- stages public docs and sanitized notes with `scripts/build_docs_site.py`
- asserts Torch is not importable in the docs job
- builds the static site with `mkdocs build --clean`
- uploads and deploys the generated `site/` artifact through GitHub Pages

The PR `ci` workflow also has a `docs-site` job that builds the same site
without deploying it.

## Sanitization Rules

The staging script rewrites machine-local paths before publication:

- repo-local absolute paths become `${OPEN_WAM_REPO}`
- private checkpoint roots become `${OPEN_WAM_ARTIFACT_ROOT}`
- private dataset roots become `${OPEN_WAM_DATA_ROOT}`
- output artifacts become `${OPEN_WAM_OUTPUT_ROOT}/...`

The build fails if known private cluster roots, AFS roots, home-directory roots,
or local usernames remain in the generated site source.

## Required GitHub Setting

Repository maintainers need to enable GitHub Pages with **GitHub Actions** as
the source:

`Settings -> Pages -> Build and deployment -> Source -> GitHub Actions`

After this is enabled, the `pages` workflow will deploy the site from `main`.
