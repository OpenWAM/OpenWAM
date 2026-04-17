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

## Scope

In scope:

- dependency or import behavior that can execute untrusted code unexpectedly
- unsafe handling of local credentials, WandB tokens, or Hugging Face tokens
- accidental disclosure of private paths in public samples or docs
- CI or packaging changes that publish private artifacts

Out of scope:

- expected failures from missing optional simulator packages
- model quality issues without a security impact
- simulator crashes from unsupported local installations
