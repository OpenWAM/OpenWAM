# Benchmarks And Data

Open-WAM keeps benchmark-specific loading behind adapters while exposing one
uniform model-facing batch contract.

## Supported Sources

| Source | Status | Primary use |
| --- | --- | --- |
| Public tiny synthetic | Checked-in fixture | CI-safe loader/eval contract smoke tests. |
| LIBERO | Dataset and simulator paths | Manipulation policy training, evaluation, and realtime rollout experiments. |
| RoboTwin | Dataset and simulator adapter path | Simulated robotic manipulation with configurable action schema. |
| CALVIN | Dataset and simulator adapter path | Simulated language-conditioned manipulation with 7D relative actions. |

Private datasets, local simulator checkouts, and large checkpoints should be
provided through the local path registry, not hard-coded in public configs.

## Action Dimensions

Benchmarks expose different native action spaces. The model-facing action
dimension is configured separately from the source action dimension.

| Benchmark | Common source action | Model-facing examples |
| --- | --- | --- |
| LIBERO | 7D EEF delta plus gripper | 7D or sparse 30D mapping depending on config. |
| RoboTwin | 16D or 30D modes | Native 16D, native 30D, or mapped sparse 30D. |
| CALVIN | 7D `rel_actions` | Native 7D or sparse 30D compatibility mapping. |

Action mapping should be explicit in the dataset adapter/config. A model should
not infer missing dimensions silently.

## Visual Layout

The data layer builds canonical RGB layouts before the visual backbone sees the
batch. Public configs should make these choices visible:

- camera names
- camera count
- frame window
- target image size
- layout policy
- channel order

This keeps visual packing controlled across methods and benchmarks.

## Public Fixture

The public tiny synthetic fixture exists to test infrastructure, not model
quality. It is useful for:

- static config validation
- loader construction
- CPU eval smoke checks
- artifact manifest layout validation
- new contributor onboarding

Run it with:

```bash
open-wam-validate-config \
  configs/examples/public_tiny_synthetic_contract.yaml \
  configs/evals/public_tiny_synthetic_contract.yaml

open-wam-eval \
  --cfg configs/evals/public_tiny_synthetic_contract.yaml \
  --device cpu \
  --max-batches 1
```

Use real benchmark cards and experiment cards for claims about policy quality.
