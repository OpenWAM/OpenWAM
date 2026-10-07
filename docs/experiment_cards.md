# Experiment Cards

Experiment cards are the public reproducibility layer for maintained policy
architectures and programs. Each card should point to an artifact manifest entry once a public
checkpoint exists.

## Required Fields

- architecture
- program or variant profile
- benchmark and task split
- train config
- eval config
- rollout command, if applicable
- checkpoint artifact id or local path alias
- dataset artifact id or local path alias
- hardware
- expected metrics
- known limitations

## Architecture Matrix

| Architecture | Current programs/profiles | Public card status |
| --- | --- | --- |
| `parallel_stream` | six video/action programs; GJD | scaffolded; public checkpoint pending |
| `dual_expert` | six video/action programs; GJD | scaffolded; public checkpoint pending |
| `causal_video_prediction` | video-only causal prediction | task-level card pending; see [released pretraining weights](pretraining/training.md#7-download-the-released-pretraining-weights) |
| fixture | `public_tiny_synthetic_contract` | public structural fixture card added |

## Current Cards

- [Public Tiny Synthetic Contract](cards/public_tiny_synthetic_contract.md)

## Template

```yaml
architecture: parallel_stream
program: video_then_action
benchmark: libero_10
train_config: configs/experiments/parallel_stream_libero_video_then_action.yaml
eval_config: null
checkpoint_artifact_id: parallel-stream-libero-video-then-action
dataset_artifact_id: null
hardware:
  gpu: null
  num_gpus: null
expected_metrics:
  mean_action_mse: null
  rollout_success_rate: null
commands:
  train: null
  eval: null
  rollout: null
limitations:
  - Public checkpoint hosting is not filled yet.
```
