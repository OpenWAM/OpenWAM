# Security Policy

## Supported Versions

Open-WAM is pre-1.0 research software. Security fixes target the current
`main` branch unless maintainers explicitly announce a release branch.

## Reporting A Vulnerability

Do not open a public issue for vulnerabilities that expose credentials, private
dataset paths, checkpoint access tokens, or remote-code execution surfaces.

Send a private report to the maintainers with:

- affected commit or release
- reproduction steps
- impacted command or package
- whether private credentials, datasets, or checkpoints are involved
- suggested mitigation, if known

Use the repository's
[private vulnerability report](https://github.com/DaivdYuan/OpenWAM-staging-public/security/advisories/new)
form. If that form is unavailable, do not share sensitive reproduction
artifacts in a public issue; wait until a private maintainer contact is listed.

## Scope

In scope:

- dependency or import behavior that can execute untrusted code unexpectedly
- unsafe handling of local credentials, WandB tokens, or Hugging Face tokens
- accidental disclosure of private paths in public samples or docs
- CI or packaging changes that publish private artifacts

## Artifact Trust

Treat checkpoints, latent tensors, NumPy archives, and simulator datasets as
untrusted inputs. Maintained tensor loaders use PyTorch's restricted
`weights_only=True` mode, and maintained NumPy readers disable object-pickle
loading. A restricted loader failure is not a reason to call unrestricted
`torch.load` or `pickle.load`.

The only maintained legacy exception is explicitly configured CALVIN language
annotation object arrays. Set the `trusted_legacy` policy only for a local
artifact whose origin and integrity have been independently verified. Never
enable that policy for downloaded or user-supplied files.

The frozen full environment is checked with `pip-audit` in pull requests and
release checks. Narrow exceptions live in
`.github/dependency-audit-exceptions.toml`; each is bound to one package,
version, advisory, rationale, and expiration date. New, stale, or expired
exceptions fail CI.

Out of scope:

- expected failures from missing optional simulator packages
- model quality issues without a security impact
- simulator crashes from unsupported local installations
