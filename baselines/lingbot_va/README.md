# LingBot-VA LIBERO-10 Baseline

This baseline runs the prior LingBot-VA model as an external reference system
while keeping Open-WAM methods untouched. The runner imports LingBot-VA source
read-only, stages compatible model roots under the output directory, and uses
Open-WAM's LIBERO init-state loader so rollout inputs match the current
Open-WAM LIBERO-10 evaluation setup.

## Contract

What is intentionally shared with Open-WAM rollouts:

- LIBERO benchmark/task/episode identifiers.
- LIBERO pruned init-state loading through `open_wam.integrations`.
- 128x128 `agentview` and wrist RGB observations.
- 7D LIBERO environment actions.
- Per-episode and per-chunk seeding.
- Video artifacts and JSON summaries for every rollout.

What remains LingBot-VA-owned:

- The model object is Heng's original `VA_Server`.
- The chunk lifecycle is `reset -> infer chunk -> execute chunk -> warmup KV`.
- The first chunk skips frame group 0, matching the LingBot client.
- The model checkpoint is loaded from LingBot/Heng-format components.

The wrapper also normalizes the LIBERO action config to Heng's successful
server contract: EEF channels `0..5`, gripper channel `28`,
`action_snr_shift=1.0`, and matching quantile stats. This is a wrapper-level
runtime override because the checked-in `previous_works/lingbot-va` config can
drift from Heng's successful checkout; the LingBot source files themselves are
not edited.

The baseline is comparable to Open-WAM `blocking_control` rollouts. It is not a
live realtime scheduler, because Heng's original server waits for each full
chunk and warmup before continuing.

## Read-Only Source Rule

Do not edit `previous_works/lingbot-va` or any external LingBot checkout. The
runner only adds source paths to `sys.path` and writes staged symlinks,
converted transformer copies, videos, traces, and summaries under the selected
`output_dir`.

If a transformer was exported by Open-WAM with local conditioner key names, the
baseline creates a converted copy under:

```text
<output_dir>/checkpoints/<checkpoint_name>/_heng_transformer_converted/
```

The source checkpoint is not modified.

## CLI

One checkpoint:

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python -m baselines.lingbot_va.run_libero10_baseline \
    --source-repo previous_works/lingbot-va \
    --pretrained-root /path/to/lingbot-va-base \
    --transformer-dir /path/to/checkpoint/transformer \
    --checkpoint-name lingbot_va_libero10_step600 \
    --benchmark libero_10 \
    --task-ids 0:10 \
    --episode-indices 0 \
    --max-timestep 1000 \
    --output-dir outputs/lingbot_va_baseline \
    --seed 0
```

Suite file:

```bash
export LINGBOT_VA_BASE_ROOT=/path/to/lingbot-va-base
export LINGBOT_VA_LIBERO10_TRANSFORMER=/path/to/checkpoint/transformer

PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  python -m baselines.lingbot_va.run_libero10_baseline \
    --suite baselines/lingbot_va/suites/libero10_env_template.yaml
```

## Artifacts

Each suite writes:

- `results.jsonl`: append-only per-episode records.
- `summary.json`: aggregate result and full row list.
- `summary.md`: compact table for notes or PR descriptions.
- `load_reports/<checkpoint>.json`: loaded component hashes and runtime config.
- `rollouts/.../*.mp4`: side-by-side agentview/wrist videos with chunk colors.
- `rollouts/.../*_chunks.json`: per-chunk infer/warmup timing and boundaries.

## Roadmap

1. Exact LingBot-VA chunk-by-chunk LIBERO-10 parity is established.
2. The bounded all-task LIBERO-10 suite has been run for task IDs `0:10`,
   episode `0`, seed `0`, max timestep `1000`.
3. The best native checkpoint tested so far is
   `train_out_heng_libero_10_0323_60fps/checkpoint_step_600`, with `9/10`
   successes on that suite.
4. Task 8 remains the residual failure. Additional probes at 60fps step550 and
   `libero_all_0325_60fps_videoonly` step850 also failed task 8 at the horizon.
5. Open-WAM-exported Method-1 checkpoints remain parity controls, not the
   LingBot-VA baseline.
6. Only after the exact baseline is stable, add an optional realtime wrapper
   that maps LingBot chunks into the shared Open-WAM sandbox scheduler.

## Current Result

Validation date: 2026-04-29.

Native LingBot-VA checkpoint:

```text
/path/to/private-resource
```

Suite:

- benchmark: `libero_10`
- task IDs: `0:10`
- episode index: `0`
- seed: `0`
- max timestep: `1000`

Result: `9/10` successes. Task 8, `put both moka pots on the stove`, failed at
`env_timestep=1009`.

Artifacts:

```text
outputs/lingbot_va_baseline_libero10_ep0_20260429/summary.md
outputs/lingbot_va_baseline_libero10_ep0_20260429/summary.json
outputs/lingbot_va_baseline_libero10_ep0_20260429/rollouts/
```
