# GitHub Pages Documentation Site

Open-WAM publishes documentation through a generated MkDocs source tree. The
tracked public site source is the explicit allowlist in
`scripts/build_docs_site.py`. Its pages live in `docs/`; internal research and
engineering notes under `notes/` are intentionally not published to GitHub
Pages.

## Local Preview

```bash
uv sync --extra docs
uv run --extra docs python scripts/build_docs_site.py --output .docs_site
uv run --extra docs mkdocs serve
```

The generated `.docs_site/` directory and final `site/` directory are
gitignored. Rebuild `.docs_site/` after editing `docs/`.

## Publication Flow

The `pages` GitHub Actions workflow runs on pushes to `main` and can also be
started manually. It:

- installs only MkDocs, not the Open-WAM package
- stages curated public docs with `scripts/build_docs_site.py`
- asserts Torch is not importable in the docs job
- builds the static site with `mkdocs build --clean`
- uploads and deploys the generated `site/` artifact through GitHub Pages

The PR `ci` workflow also has a `docs-site` job that builds the same site
without deploying it.

## Publication Rules

The Pages site is a public user and contributor manual, not a dump of internal
engineering notes. Publish durable docs under `docs/` and keep raw notes under
`notes/`.

If a note becomes useful for outside users, distill it into a public doc page
with stable commands, placeholders, and current repo paths. Do not publish raw
run logs, local machine paths, private checkpoint locations, or superseded
planning records.

The build fails if a page under `docs/` has not been classified in the public
allowlist, if a local link is missing or escapes the generated site, or if known
private cluster roots, AFS roots, home-directory roots, or local usernames
remain in the generated source.

## Required GitHub Setting

Repository maintainers need to enable GitHub Pages with **GitHub Actions** as
the source:

`Settings -> Pages -> Build and deployment -> Source -> GitHub Actions`

After this is enabled, the `pages` workflow will deploy the site from `main`.
