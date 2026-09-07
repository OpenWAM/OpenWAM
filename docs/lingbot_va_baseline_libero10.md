# LingBot-VA LIBERO-10 Baseline

Status: active baseline integration under `baselines/lingbot_va`.

This baseline is the upstream open-source LingBot-VA LIBERO-LONG checkpoint
evaluated on `libero_10`. It is intentionally outside the OpenWAM method
runtime.

## Contract

Default baseline scope:

- source checkout: `previous_works/lingbot-va`, read-only
- checkpoint: `robbyant/lingbot-va-posttrain-libero-long`
- checkpoint revision: `0e89d1e753019988aba484e8da2dc0810e264d9f`
- local asset root: selected by `LINGBOT_VA_MODEL_ROOT`
- benchmark: `libero_10`
- task IDs: `0:10`
- episode indices: `0:50`
- policy RNG seed: unset, matching upstream client behavior
- horizon: `env.timestep < 800`
- video FPS when rendered: `60`

The runner preserves the upstream LIBERO client loop while adding resumable
OpenWAM result collection:

- `benchmark_instance.get_task_init_states(task_id)` for init states
- `model.infer(dict(reset=True, prompt=prompt))`
- infer one chunk from the latest observation
- skip frame group `0` only for the first chunk
- execute the returned actions in order
- warm KV cache with `compute_kv_cache=True, imagine=False, state=action`

No local checkpoint substitution, transformer override, or action-channel
override is part of the baseline.

## Download

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

## Full Evaluation Command

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

For long metric runs, shard by disjoint task ranges and merge the JSONL files:

```bash
PYTHONPATH=src:outputs/lingbot_va_pydeps \
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  "$LINGBOT_BASELINE_PYTHON" \
  -m baselines.lingbot_va.run_libero10_baseline \
    --source-repo previous_works/lingbot-va \
    --model-root "$LINGBOT_VA_MODEL_ROOT" \
    --checkpoint-name lingbot_va_posttrain_libero_long \
    --hf-revision 0e89d1e753019988aba484e8da2dc0810e264d9f \
    --benchmark libero_10 --task-ids 0:5 --episode-indices 0:50 \
    --max-timestep 800 --video-fps 60 \
    --output-dir outputs/lingbot_va_posttrain_libero_long_libero10_full_20260429/gpu0_tasks0_4 \
    --continue-on-error --resume --no-render-video

PYTHONPATH=src:outputs/lingbot_va_pydeps \
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
  "$LINGBOT_BASELINE_PYTHON" \
  -m baselines.lingbot_va.run_libero10_baseline \
    --source-repo previous_works/lingbot-va \
    --model-root "$LINGBOT_VA_MODEL_ROOT" \
    --checkpoint-name lingbot_va_posttrain_libero_long \
    --hf-revision 0e89d1e753019988aba484e8da2dc0810e264d9f \
    --benchmark libero_10 --task-ids 5:10 --episode-indices 0:50 \
    --max-timestep 800 --video-fps 60 \
    --output-dir outputs/lingbot_va_posttrain_libero_long_libero10_full_20260429/gpu1_tasks5_9 \
    --continue-on-error --resume --no-render-video
```

Merge:

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

The suite writes:

- `results.jsonl`
- `summary.json`
- `summary.md`
- `load_reports/*.json`
- per-rollout chunk traces under `rollouts/`
- optional chunk-colored rollout videos under `rollouts/`
