# CLI Reference

OpenWAM provides six console commands for configuration, training, and inference.

## Stable Commands

| Command | Purpose |
| --- | --- |
| `openwam-train` | Train from an experiment YAML |
| `openwam-eval` | Offline evaluation from experiment or eval YAML |
| `openwam-inspect-config` | Load and print typed config |
| `openwam-validate-config` | Static YAML validation without model imports |
| `openwam-sanity` | Check loading, training, evaluation, and rollout-style inference |
| `openwam-sim-rollout` | Run a registered simulator adapter in closed loop |

All six commands work from an installed package. Use `--help` for their options.
The `uv run` examples below assume a source checkout; after a pip installation
with the relevant extras, invoke the commands directly without that prefix.

For complete train, full-state resume, offline evaluation, and benchmark
rollout examples, use [Training and Inference](running_experiments.md). It is
the complete usage guide.

## Command Examples

Installed extensions load before experiment construction. Repeat
`--extension module[:hook]` when a config uses out-of-tree dataset, policy, or
decoder registrations:

```bash
uv run --extra train openwam-train \
  --extension acme_open_wam \
  --cfg /path/to/acme_joint.yaml
```

```bash
uv run openwam-validate-config configs/examples/public_tiny_synthetic_contract.yaml
```

`openwam-validate-config` checks authored experiment and evaluation YAML. Do
not use it to lint checkpoint-generated `resolved_config.yaml` files: those
artifacts serialize typed defaults, including fields intentionally omitted from
an authored method config. Load such files through the runtime checkpoint path,
which applies checkpoint compatibility when required. For a current-schema
artifact, `openwam-inspect-config` can display the typed configuration without
treating it as authored YAML.

```bash
uv run --extra train openwam-train \
  --cfg configs/examples/public_tiny_synthetic_contract.yaml \
  --save-root runs/public-tiny \
  --disable-wandb
```

```bash
uv run --extra eval openwam-eval \
  --cfg configs/evals/public_tiny_synthetic_contract.yaml \
  --checkpoint runs/public-tiny/checkpoints/checkpoint_step_1/model_state.pt \
  --device cpu \
  --max-batches 1 \
  --output-json outputs/eval.json \
  --provenance-mode full
```

These two commands form the data-free train/eval example. Benchmark configs
require the local artifacts and resources documented in
[Training and Inference](running_experiments.md).

```bash
uv run --extra sim openwam-sim-rollout \
  --cfg configs/examples/calvin_npz_raw7_sanity.yaml \
  --benchmark calvin \
  --max-steps 10 \
  --zero-policy
```

Checkpoint loading is strict by default. `--allow-partial-checkpoint` permits
missing or unexpected model keys only for an intentional migration diagnostic;
results produced under that opt-in should not be reported as normal evals.

Third-party simulator adapters use `--extension`, an application-owned
`--benchmark` identifier, and repeatable `--sim-option KEY=VALUE` values. See
the [simulator adapter cookbook](cookbooks/new_simulator_adapter.md).

## Source-Checkout Utilities

Model conversion, dataset generation, research diagnostics, and specialized
artifact visualization can require a source checkout. In particular, the Wan
conversion scripts and `scripts/generate_video_only_rollout.py` are not installed
console commands and are not included in release distributions. Their guides
label them as source-checkout integrations. They may compose package APIs, but
they are not stable package entrypoints.
