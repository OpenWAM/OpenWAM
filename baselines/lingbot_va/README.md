# LingBot-VA LIBERO-LONG Baseline

This baseline runs the upstream open-source LingBot-VA LIBERO checkpoint as an
external reference system while keeping Open-WAM methods untouched.

The default baseline is intentionally narrow:

- source checkout: `previous_works/lingbot-va`, read-only
- released checkpoint: `robbyant/lingbot-va-posttrain-libero-long`
- released checkpoint revision: `0e89d1e753019988aba484e8da2dc0810e264d9f`
- benchmark: `libero_10`, the LIBERO-LONG task suite
- control loop: upstream `evaluation/libero/client.py` semantics
- horizon: upstream `env.timestep < 800`

No local checkpoint substitution, Open-WAM-exported transformer, or
action-channel compatibility override is part of this baseline.

## Download

Place the released checkpoint outside the repo:

```bash
export LINGBOT_BASELINE_PYTHON=/path/to/lingbot-va-env/bin/python
export LINGBOT_VA_MODEL_ROOT=/path/to/lingbot-va-posttrain-libero-long

"$LINGBOT_BASELINE_PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="robbyant/lingbot-va-posttrain-libero-long",
    local_dir=os.environ["LINGBOT_VA_MODEL_ROOT"],
)
PY
```

The model root must contain `vae/`, `text_encoder/`, `tokenizer/`, and
`transformer/`.

## Run

Full upstream-style LIBERO-10 evaluation is 10 tasks x 50 init states:

```bash
export LINGBOT_BASELINE_PYTHON=/path/to/lingbot-va-env/bin/python
export LINGBOT_VA_MODEL_ROOT=/path/to/lingbot-va-posttrain-libero-long

PYTHONPATH=src:outputs/lingbot_va_pydeps \
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  "$LINGBOT_BASELINE_PYTHON" \
  -m baselines.lingbot_va.run_libero10_baseline \
    --suite baselines/lingbot_va/suites/libero10_env_template.yaml
```

Single smoke episode:

```bash
PYTHONPATH=src:outputs/lingbot_va_pydeps \
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  "$LINGBOT_BASELINE_PYTHON" \
  -m baselines.lingbot_va.run_libero10_baseline \
    --source-repo previous_works/lingbot-va \
    --model-root "$LINGBOT_VA_MODEL_ROOT" \
    --checkpoint-name lingbot_va_posttrain_libero_long \
    --benchmark libero_10 \
    --task-ids 0 \
    --episode-indices 0 \
    --max-timestep 800 \
    --video-fps 60 \
    --output-dir outputs/lingbot_va_posttrain_libero_long_smoke \
    --continue-on-error
```

## Contract

The runner imports upstream `VA_Server` and preserves the LIBERO client loop:

- initialize env from `benchmark_instance.get_task_init_states(task_id)`
- `model.infer(dict(reset=True, prompt=prompt))`
- infer one chunk from the latest observation
- skip frame group `0` only on the first chunk
- execute every action in the returned chunk in order
- append key frames after every `action_per_frame` actions
- warm the KV cache with `compute_kv_cache=True, imagine=False, state=action`
- stop when the env reports success or reaches timestep `800`

Open-WAM only provides argument parsing, resumable episode grids/manifests,
JSON summaries, and optional rollout video rendering.
The suite's `runtime.renderer_profile: online_rollout` pins this online baseline
to EGL before upstream simulator modules are imported.

## Artifacts

Each run writes:

- `results.jsonl`: append-only per-episode records
- `summary.json`: aggregate result and full row list
- `summary.md`: compact table
- `load_reports/<checkpoint>.json`: model-root provenance and runtime config
- `rollouts/.../*_chunks.json`: per-chunk timing and boundaries
- `rollouts/.../*.mp4`: optional side-by-side videos with chunk colors

Long runs can set `runtime.resume: true`; completed rows are skipped by
`(checkpoint, benchmark, task_id, episode_idx, seed, sample_id)`.

The default suite leaves `seed` empty to match the upstream client, which does
not seed the policy RNG. Pass `--seed` only for explicit reproducibility
ablations.

Two-GPU full evaluations can be merged after both shards finish:

```bash
PYTHONPATH=src:outputs/lingbot_va_pydeps \
  "$LINGBOT_BASELINE_PYTHON" \
  -m baselines.lingbot_va.summarize_results \
  --results-jsonl \
    outputs/lingbot_va_libero10/gpu0_tasks0_4/results.jsonl \
    outputs/lingbot_va_libero10/gpu1_tasks5_9/results.jsonl \
  --expect-count 500 \
  --expect-benchmark libero_10 \
  --expect-task-ids 0:10 \
  --expect-episode-indices 0:50 \
  --require-null-seed \
  --require-unique \
  --require-hf-revision 0e89d1e753019988aba484e8da2dc0810e264d9f \
  --require-model-root "$LINGBOT_VA_MODEL_ROOT" \
  --output-dir outputs/lingbot_va_libero10/combined
```

## RobotWin Evaluation

LingBot-VA's RobotWin evaluation is a separate upstream blocking-control path:
the client requests one action chunk, steps the simulator through that chunk,
warms the model KV cache from the executed key frames, then requests the next
chunk. It is comparable to our blocking-control rollouts, not realtime
`freeze_until_clean_chunk` or async scheduling policies.

The wrappers below do not edit `previous_works/lingbot-va`; they execute the
upstream RobotWin server/client code with local path and output-root patches.

Start one model server per GPU:

```bash
PYTHONPATH=src:outputs/lingbot_va_pydeps \
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  "$LINGBOT_BASELINE_PYTHON" \
  -m baselines.lingbot_va.run_robotwin_server \
    --source-repo previous_works/lingbot-va \
    --model-root /path/to/lingbot-va-posttrain-robotwin \
    --save-root outputs/lingbot_va_robotwin/server_gpu0 \
    --port 29056
```

Run the upstream default task for 100 expert-verified episodes:

```bash
PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 ROBOTWIN_ROOT=/path/to/RoboTwin \
  /path/to/robotwin/python \
  -m baselines.lingbot_va.run_robotwin_client \
    --source-repo previous_works/lingbot-va \
    --save-root outputs/lingbot_va_robotwin/adjust_bottle_100ep \
    --task-name adjust_bottle \
    --test-num 100 \
    --seed 0 \
    --port 29056
```

Artifacts are written under the client `--save-root`:

- `stseed-10000/metrics/<task>/res.json`: incremental success count
- `stseed-10000/visualization/<task>/*.mp4`: per-episode comparison videos
- `eval_result/<task>/ACT/demo_clean/0/<timestamp>/_result.txt`: upstream result
  file
