# Artifacts And Checkpoints

OpenWAM separates public experiment configs from machine-local artifact paths.

## Local Path Registry

Use `configs/local_paths.yaml` for local paths. Start from:

```bash
cp configs/local_paths.sample.yaml configs/local_paths.yaml
```

That file is gitignored. It can also live outside the repo:

```bash
OPEN_WAM_LOCAL_PATHS=/path/to/local_paths.yaml uv run openwam-eval ...
```

## Artifact Manifest

`configs/artifacts.sample.yaml` documents the public manifest schema for data
and checkpoints. Machine-local or private manifests should use
`configs/artifacts.yaml`, which is gitignored.

- `artifact_id`
- `architecture`
- `variant`
- `benchmark`
- `config`
- `local_path_alias`
- `expected_layout`
- `download_url`
- `checksum`
- `license`
- `source`
- `notes`

Entries with `download_url: null` are layout documentation only. They should not
be advertised as reproducible public checkpoints until hosting, checksum, and
license fields are filled.

Final release validation rejects partially published model entries. A public
entry must have an HTTPS download URL, SHA-256 checksum, and license; an
unpublished entry keeps all three fields null. The tiny synthetic entry proves
structure and execution only; it is not evidence of model quality.

Checkpoint and latent tensor files are loaded through the restricted
`weights_only=True` PyTorch path. OpenWAM does not automatically retry unsafe
pickle deserialization. Legacy CALVIN object-array language annotations require
the explicit `trusted_legacy` policy and must only come from a trusted local
dataset.

Manifest schema v2 uses `architecture`. The loader still accepts the retired
`method_family` key in private manifests, but new manifests should not emit it.

The `public-tiny-synthetic-contract` entry is an exception in purpose: it is a
checked-in structural fixture under `tests/fixtures/public_tiny/`, not a real
model checkpoint. It exists so public validation can exercise artifact layout
checks without private or large files.

## Checkpoint Layout Convention

Full training checkpoint roots should use this layout when possible:

```text
checkpoint_step_N/
  full_training_state.pt
  model_state.pt
  transformer/
    config.json
    diffusion_pytorch_model.safetensors
```

Transformer-only runtime paths may point directly at `checkpoint_step_N/transformer`.
Code that accepts checkpoint roots should also accept roots containing a
`transformer/` child when possible.

A usable transformer export contains a valid JSON object in `config.json` and
either one nonempty `diffusion_pytorch_model.safetensors` file or a
`diffusion_pytorch_model.safetensors.index.json` whose `weight_map` references
only present, nonempty shards. Runtime, checkpoint, evaluation, and CLI paths
all enforce this same contract; a directory containing only `config.json` is
not a model artifact.
