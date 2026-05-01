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

## Local Rollout

For a verified causal-realtime rollout against an already-trained checkpoint,
use the ablation wrapper. The `live_async_history_first_startup_hold` profile
combined with `--replan-low-watermark-actions auto` resolves K=8 for Method 1
(verified in PR #58: K=4/10/12 fail; only K=8 succeeds at 2 Hz):

```bash
uv run --extra sim python scripts/run_libero_realtime_ablation.py \
  --cases method1_exact_step400:configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml:/absolute/path/to/checkpoint_step_400 \
  --profiles live_async_history_first_startup_hold \
  --target-action-hz 2 --max-actions 520 \
  --fallback-history-policy freeze_until_clean_chunk \
  --replan-low-watermark-actions auto \
  --output-dir outputs/libero_method1_recommended \
  --suffix recommended_live_causal \
  --runtime-device cuda:0 --frontend-device cuda:0 --decode-device cuda:0
```

Local rollout prerequisites — see
[docs/quickstart.md](../quickstart.md) "LIBERO Local Rollout Setup":

- `[sim]` (or `[full]`) extra installed; `[libero]` alone is not enough.
- Upstream LIBERO cloned and `pip install -e ../LIBERO` (with an empty
  `touch ../LIBERO/libero/__init__.py` first to fix the namespace package).
- `~/.libero/config.yaml` populated with `benchmark_root` / `bddl_files` /
  `init_states` / `datasets` / `assets` keys (without `_folder` suffix).
- `configs/local_paths.yaml` populated with the
  `parallel_stream_exact_libero_step_400` alias pointing at the checkpoint
  directory.

Smoke-test the pipeline with `--max-actions 30 --suffix smoke` before the
full 520-action run. The model load plus first plan takes roughly 6 seconds;
steady-state inference holds 2 Hz with replan latency ~3 seconds running in
a background thread while the env keeps stepping at the target rate.
Steady-state GPU footprint is ~13 GB on a 32 GB card.

## Limitations

- This card documents the expected layout and command surface only.
- It is not a public reproducibility claim until artifact hosting and checksums
  are filled in.
