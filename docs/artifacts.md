# Artifacts And Checkpoints

OpenWAM separates public experiment configs from machine-local artifact paths.

## Released Models

- [Video-pretrained backbone](https://huggingface.co/OpenWAM-Stanford/OpenWAM-Pretraining): causal robot-video prediction.
- [VTA for LIBERO-10](https://huggingface.co/OpenWAM-Stanford/OpenWAM-VTA-LIBERO10): the step-15,000 video-then-action policy for LIBERO-Long (LIBERO-10), including both video and action experts.
- [Mixed IDM CF64k](https://huggingface.co/OpenWAM-Stanford/OpenWAM-IDM-Mixed-CF64k): inverse dynamics post-training with 60% counterfactual data and 40% real demonstrations.
- [CF-only IDM CF64k](https://huggingface.co/OpenWAM-Stanford/OpenWAM-IDM-CF64k): inverse dynamics post-training with counterfactual data only.

Each release includes `model_state.pt`, `resolved_config.yaml`, and
`SHA256SUMS`. These are weights-only releases, not resumable training states.
Frontend assets and datasets are installed separately. The VTA and IDM
checkpoints are released under AGPL-3.0-only, with upstream attribution included.
Use each checkpoint's bundled `resolved_config.yaml` to preserve its inference
settings rather than substituting a repository training config.

Use the [pretraining guide](pretraining/training.md#7-download-the-released-pretraining-weights)
for the video backbone and [video-to-action composition](video_action_composition.md)
to pair an IDM with a compatible video producer. The IDM cards identify the
video predictors and evaluation protocols behind their reported results;
the IDMs are action components, not standalone planning policies.

## Counterfactual Data

[LIBERO-CF64k](https://huggingface.co/datasets/OpenWAM-Stanford/LIBERO-CF64k)
contains 64,000 counterfactual motion segments from 10,000 branching contexts,
balanced across the ten LIBERO-Long tasks. Alternative action sequences from
shared starting states support learning both inverse and forward dynamics.

Each segment has agent-view and wrist-camera RGB, actions, proprioception,
and a boundary simulator state. The raw format uses 128 controls and 129
observations at 20 Hz, spanning 6.4 seconds. Data is licensed under CC BY 4.0;
the reader code is licensed separately under AGPL-3.0-only.

Raw data, a reader, and integrity checks are publicly available. The download
includes 125 video shards, 20 context-state shards, sample and context indices,
and a shard checksum manifest. The full dataset is approximately 388 GB.

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

A downloadable model entry includes an HTTPS URL, SHA-256 checksum, and license.
The tiny synthetic entry demonstrates structure and execution only; it is not
evidence of model quality.

Checkpoint and latent tensor files are loaded through the restricted
`weights_only=True` PyTorch path. OpenWAM does not automatically retry unsafe
pickle deserialization. Legacy CALVIN object-array language annotations require
the explicit `trusted_legacy` policy and must only come from a trusted local
dataset.

Only load artifacts from trusted sources. Restricted tensor loading does not
validate every external model asset or shard index; read the
[artifact trust policy](https://github.com/OpenWAM/OpenWAM/blob/main/SECURITY.md#artifact-trust).

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
