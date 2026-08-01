# CLI Reference

Open-WAM exposes package-owned console commands and keeps legacy root scripts as
compatibility wrappers.

## Stable Commands

| Command | Purpose | Legacy path |
| --- | --- | --- |
| `open-wam-train` | Train from an experiment YAML | `scripts/train.py` |
| `open-wam-eval` | Offline eval from experiment or eval YAML | `scripts/eval.py` |
| `open-wam-inspect-config` | Load and print typed config | `scripts/inspect_config.py` |
| `open-wam-validate-config` | Static YAML validation without model imports | `scripts/validate_configs_static.py` |
| `open-wam-sanity` | Run quantified load/train/eval/rollout-style sanity checks | `scripts/run_benchmark_pipeline_sanity.py` |
| `open-wam-sim-rollout` | Run closed-loop RoboTwin/CALVIN rollout when simulators are installed | `scripts/run_sim_realtime_sandbox.py` |

`open-wam-sim-rollout` executes entirely from the installed package. Its
legacy script path is a thin adapter to the same parser and runtime.

## Compatibility Policy

New docs should prefer `open-wam-*` commands. Existing `scripts/...` commands
must remain callable until they have:

1. a package-owned replacement
2. compatibility tests or dry-run/help equivalence
3. a documented deprecation warning
4. a later explicit removal PR

For complete train, full-state resume, offline evaluation, and benchmark
rollout examples, use [Training and Inference](running_experiments.md). It is
the canonical operator guide; archived engineering notes are not command
references.

## Command Examples

Installed extensions load before experiment construction. Repeat
`--extension module[:hook]` when a config uses out-of-tree dataset, policy, or
decoder registrations:

```bash
uv run --extra train open-wam-train \
  --extension acme_open_wam \
  --cfg configs/experiments/acme_joint.yaml
```

```bash
uv run open-wam-validate-config configs/examples/public_tiny_synthetic_contract.yaml
```

```bash
uv run --extra train open-wam-train --cfg configs/experiments/parallel_stream_robotwin_smoke.yaml
```

```bash
uv run --extra eval open-wam-eval \
  --cfg configs/evals/parallel_stream_robotwin_smoke.yaml \
  --device cpu \
  --max-batches 1
```

```bash
uv run --extra sim open-wam-sim-rollout \
  --cfg configs/examples/calvin_npz_raw7_sanity.yaml \
  --benchmark calvin \
  --max-steps 10 \
  --zero-policy
```
