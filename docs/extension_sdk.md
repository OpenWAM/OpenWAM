# Extension SDK

Open-WAM extensions are ordinary installed Python modules. They register
role-based components without modifying the Open-WAM source tree.

## Loading Extensions

Every maintained runtime command accepts repeatable extension specs:

```bash
open-wam-train \
  --extension acme_open_wam \
  --extension research_runtime.install:register \
  --cfg experiment.yaml
```

An extension spec is `module[:hook]`; the default hook is
`register_open_wam`. Hooks run once per process and in command-line order.
Import or registration failures stop startup before configuration is
constructed.

```python
from open_wam.data import register_dataset_adapter

from .dataset import build_train_val


def register_open_wam() -> None:
    register_dataset_adapter(
        "acme_robot",
        raw_builder=build_train_val,
        description="ACME robot demonstrations.",
    )
```

The same `--extension` contract is available on `open-wam-eval`,
`open-wam-sanity`, and `open-wam-sim-rollout`. The module must be installed in
the active environment or otherwise importable on `PYTHONPATH`.

## Current Registries

- dataset adapters: `open_wam.data.register_dataset_adapter`
- policy variant builders: `open_wam.pipelines.POLICY_VARIANT_BUILDERS`
- action decoder builders: `open_wam.pipelines.ACTION_DECODER_BUILDERS`

The active architectural boundary remains:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

## Adding A Dataset

1. Implement a dataset adapter that returns `WAMSample`.
2. Keep source-specific parsing inside the adapter.
3. Build canonical RGB layout in the data layer.
4. Register raw and/or latent builders under one stable `dataset_type`.
5. Add a focused adapter test and one config-loader smoke.

Use `data.adapter_options` for source-specific, open-ended settings. Keep
camera layout, action/state schema, sampling, and other shared semantics in
their typed `data` fields.

An adapter may expose:

- `raw_builder`: returns raw-RGB `WAMSample` datasets
- `latent_builder`: returns pre-encoded `LatentWAMSample` datasets
- both builders under one key when a source supports both paths

Registration rejects accidental replacement. Use a globally unique
`dataset_type`; `replace=True` is reserved for intentional process-local
overrides.

### Distributed Sampling

A training dataset may implement
`build_train_sampler(*, world_size, rank)`. Prefer the shared contracts in
`open_wam.data`:

- `WeightedReplacementDistributedSampler` for seeded weighted draws
- `EpochOrderDistributedSampler` for a dataset-provided global order padded to
  equal rank lengths; set `geometry_from_order=True` when weighting changes
  the epoch length
- `EpochOffsetDistributedSampler` for deterministic draw keys interpreted by
  the dataset
- `PaddedEpochOffsetDistributedSampler` when draw-key epochs must not overlap
  after equal-rank padding
- `UnpaddedEpochOrderDistributedSampler` only when the training strategy
  explicitly supports unequal rank lengths

Construct one deterministic global order, then shard it by rank. Dataset
adapters should own source weights and index interpretation, while these
samplers own distributed coordination.

## Adding A Policy Variant

1. Add or extend a typed policy config.
2. Implement `PolicyVariant` methods:
   `required_visual_stages`, `prepare_train_inputs`, `forward_train`,
   `prepare_infer_state`, and `forward_infer_step`.
3. Register a builder in `POLICY_VARIANT_BUILDERS`.
4. Add construction parity tests before migrating existing methods.
5. Keep old enum/config names as aliases during the migration window.

## Adding An Action Decoder

1. Add or extend a typed decoder config.
2. Implement the `ActionDecoder` contract.
3. Register a builder in `ACTION_DECODER_BUILDERS`.
4. Add loss/output shape tests.
5. Keep result schemas backward compatible when adding new outputs.

## Contract Rules

- Registration hooks configure contracts; they must not start jobs or mutate
  global training state.
- Use `open_wam.configs` coercion helpers for enum-backed extension settings;
  keep cross-section defaults and validation in a named config contract.
- Dataset parsing remains in data adapters.
- Policy semantics remain in `PolicyVariant`.
- Shared visual execution remains in `VisualTower`.
- Final supervised outputs and losses remain in `ActionDecoder`.
- Public finite choices are enum-backed; dataset names, row keys, paths, and
  extension labels remain open strings.

## Cookbooks

- `docs/cookbooks/new_method.md`
- `docs/cookbooks/new_action_decoder.md`
- `docs/cookbooks/new_dataset.md`
- `docs/cookbooks/new_simulator_adapter.md`
- `docs/cookbooks/reproduce_result.md`
