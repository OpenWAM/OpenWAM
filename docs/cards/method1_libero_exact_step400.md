# Method 1 LIBERO Exact Step 400/1100 Layout Card

## Identity

- family: method1
- variant: `parallel_stream_lingbot_exact`
- benchmark: `libero_10`
- config: `configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml`
- eval config: `configs/evals/parallel_stream_libero_lingbot_exact_heng_eval.yaml`
- artifact id: `method1-libero-exact-step400`

## Dimensions

- cameras: `observation.images.agentview_rgb`, `observation.images.eye_in_hand_rgb`
- canonical RGB: `128 x 256`
- source action dim: `7`
- model action dim: `30`
- action horizon: `16`
- state dim: `8`

## Resources

- requires private or pending-public LIBERO latent dataset
- requires private or pending-public checkpoint export
- requires Torch runtime for eval/rollout
- simulator rollout requires LIBERO setup outside the default CI tier

## Validation

```bash
open-wam-validate-config \
  configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml \
  configs/evals/parallel_stream_libero_lingbot_exact_heng_eval.yaml
```

Expected outcome: static config validation passes. Public checkpoint hosting,
checksum, and license are still pending.

## Limitations

- This card documents the expected layout and command surface only.
- It is not a public reproducibility claim until artifact hosting and checksums
  are filled in.
