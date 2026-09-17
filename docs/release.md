# Releases

OpenWAM is alpha-stage research software for Linux with Python 3.11 or 3.12.
See the [changelog](https://github.com/OpenWAM/OpenWAM/blob/main/CHANGELOG.md)
for changes and the
[release page](https://github.com/OpenWAM/OpenWAM/releases) for version notes.

## Install A Release

```bash
python -m pip install openwam
```

Add the extras needed for your workload, such as `openwam[train,eval]` or
`openwam[train,pretrain]`. The base package provides configuration and metadata
APIs without the model stack. Python imports use `open_wam`.

`openwam-sdk` is an optional installation alias for the same version of
`openwam`, including its extras. Prefer `openwam` in new requirements files.
See [Quickstart](quickstart.md) for setup and a complete CPU example.

## 0.2.0 Draft

This checkout prepares **0.2.0**, which is not yet published. It unifies
inference and rollout execution and fixes one-rank FSDP, but also changes
documented SDK contracts. Read [Migrating To 0.2](migration_0_2.md) before
upgrading an extension or reproducing an older rollout.

The latest published release is still 0.1.1; the install commands below
deliberately pin that available version. The package remains alpha-stage
research software. A plain `0.2.0` is not a PEP 440 prerelease; a GitHub
prerelease flag or Alpha classifier alone does not make pip treat it as one.
An index prerelease would need a separately approved suffix such as `0.2.0a1`.

## Reproduce An Environment

Pin the package version to keep the OpenWAM API fixed:

```bash
python -m pip install 'openwam[train,eval]==0.1.1'
```

This still resolves third-party dependency ranges. For exact numerical
comparisons, use the corresponding source tag, `uv sync --frozen`, and the
hardware/software stack recorded with the result. See
[Compatibility](compatibility.md) and [Reproducibility](reproducibility.md).

Model weights, datasets, and simulator source trees are separate from the
Python package. Follow [Artifacts](artifacts.md) for setup and read the
[security policy](https://github.com/OpenWAM/OpenWAM/blob/main/SECURITY.md#artifact-trust)
before loading third-party artifacts.
