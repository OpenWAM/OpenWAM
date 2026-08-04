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
| `parallel_stream` | exact backend; six video/action programs; GJD | layout card added; public checkpoint pending |
| `dual_expert` | six video/action programs; GJD | scaffolded; public checkpoint pending |
| `post_latent`, `post_decoded` | video-conditioned decoder | scaffolded; public checkpoint pending |
| fixture | `public_tiny_synthetic_contract` | public structural fixture card added |

## Current Cards

- `docs/cards/public_tiny_synthetic_contract.md`
- `docs/cards/parallel_stream_libero_exact_step400.md`

## Template

```yaml
architecture: parallel_stream
variant: exact
benchmark: libero_10
train_config: configs/experiments/parallel_stream_libero_lingbot_exact.yaml
eval_config: configs/evals/parallel_stream_libero_lingbot_exact_eval.yaml
checkpoint_artifact_id: parallel-stream-libero-exact-step400
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
