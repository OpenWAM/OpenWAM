# Method 3 LIBERO Video Sequence Step 800 Layout Card

## Identity

- family: method3
- variant: `video_sequence_policy`
- benchmark: `libero_10`
- config: `configs/experiments/video_sequence_policy_libero_latent_local_random_subwindow.yaml`
- eval config: `configs/evals/video_sequence_policy_libero_heng_eval.yaml`
- artifact id: `method3-libero-video-sequence-step800`

## Dimensions

- cameras: `observation.images.agentview_rgb`, `observation.images.eye_in_hand_rgb`
- canonical RGB: `128 x 256`
- source action dim: `7`
- action horizon: config-defined local policy window
- state dim: `8`

## Resources

- requires private or pending-public LIBERO latent dataset
- requires private or pending-public checkpoint export
- requires Torch runtime for eval/rollout
- simulator rollout requires LIBERO setup outside the default CI tier

## Validation

```bash
open-wam-validate-config \
  configs/experiments/video_sequence_policy_libero_latent_local_random_subwindow.yaml \
  configs/evals/video_sequence_policy_libero_heng_eval.yaml
```

Expected outcome: static config validation passes. Public checkpoint hosting,
checksum, and license are still pending.

## Limitations

- This card documents the expected layout and command surface only.
- It is not a public reproducibility claim until artifact hosting and checksums
  are filled in.
