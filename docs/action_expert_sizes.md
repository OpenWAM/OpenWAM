# Smaller action experts without reducing video depth

The dual-expert architecture supports an optional `small_500m` action expert.
It changes only action residual and feed-forward widths. The video model,
one-to-one layer pairing, attention head geometry, VTA ordering, masks,
history visibility, text conditioning and dynamic token batching are unchanged.
No existing experiment opts in automatically.

| Action expert | Layers | Residual width | FFN width | Attention width | Parameters (20D action) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Configured width-2048 expert | 30 | 2048 | 8192 | 24 × 128 | 2,563,098,644 |
| `small_500m` | 30 | 576 | 2048 | 24 × 128 | 503,254,484 |

Counts include every action-expert parameter: all transformer layers, action
input/output projections, time conditioning, text projection and hidden-context
projection. A 7D action interface has 503,239,495 parameters. The new expert has
about 80.4% fewer parameters; this is **not** an 80.4% whole-model memory or
throughput improvement. Video computation and attention geometry remain large.

## Selecting the model

In an existing VTA recipe, change only the following policy settings:

```yaml
policy_variant:
  name: dual_expert
  program: video_then_action
  action_expert_size: small_500m
  action_expert_init_mode: video_weight_interpolate
  num_action_layers: 30
  action_hidden_size: null
  action_ffn_dim: null
```

Keep all other existing sections/settings. Null dimensions are materialized as
576 and 2048 in the typed configuration and checkpoint `resolved_config.yaml`.
Alternatively specify those exact dimensions. Remove/clear old explicit widths
(such as 2048/8192): conflicting settings are rejected, never silently ignored.
For CLI overrides, pass the size, both null width overrides and initialization
mode in the same invocation. `action_expert_size: configured` is the unchanged
default and preserves existing manual dimensions or backbone-matched defaults.

For the maintained LIBERO VTA recipe, use the standard OpenWAM entrypoint after
configuring local dataset and model paths:

```bash
torchrun --standalone --nproc-per-node=4 -m open_wam.cli.train \
  --cfg configs/experiments/dual_expert_libero_video_then_action.yaml \
  --save-root runs/libero-vta-small-500m \
  --expected-world-size 4 \
  --set policy_variant.action_expert_size=small_500m \
  --set policy_variant.action_hidden_size=null \
  --set policy_variant.action_ffn_dim=null \
  --set policy_variant.action_expert_init_mode=video_weight_interpolate \
  --set trainer.enable_wandb=false
```

This example leaves LIBERO's action representation, language conditioning,
sampling, resolution, and batch schedule unchanged. To opt into dynamic token
batching separately, see [Variable-Length Training Batches](variable_length_training_batches.md).

The named 500M profile is validated for the current 30-layer, 3072-wide video
backbone with 3072-wide attention. Different video depths/widths are rejected
rather than silently dropping action layers or mislabeling another parameter
count as 500M. Custom geometries remain available through `configured`, and
the pipeline still requires one action layer per video layer.

## Initialization and checkpoints

`video_weight_interpolate` reuses the existing dimension-resizing initializer
on each corresponding video layer; it does not skip or merge layers and is not
a claim of distilled or equivalent model quality. `random` is also allowed.
`video_weight_copy` is rejected for this profile because widths differ.

Old full-size action checkpoints and optimizer states cannot be resumed into
this narrower expert. Start a separate run with a compatible video-only
initialization source, or an explicitly prepared conversion; do not point the
whole-model initializer at an old large-action checkpoint. Future small-model
checkpoints must travel with their `resolved_config.yaml` and updated runtime.
Selecting this model does not change text-conditioning settings. If task text
is disabled, retain that setting together with dropout 0 and guidance scale 1.

Validation covers actual meta-module counts and projection shapes, configuration
and checkpoint-YAML round trips, conflict rejection, and scaled-width VTA/Joint
dynamic-token prediction/loss/gradient/update parity with activation checkpointing
on and off. Full-size multi-GPU throughput and downstream task quality require
separate experiments; no training job is switched by adding this option.
