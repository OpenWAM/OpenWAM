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

## Registration APIs

- dataset adapters: `open_wam.data.register_dataset_adapter`
- policy variants: `open_wam.pipelines.register_policy_variant`
- action decoders: `open_wam.pipelines.register_action_decoder`

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

## Policy And Decoder Envelopes

Application-owned policies and decoders use a typed outer envelope. The
extension identifier is intentionally an open string; all built-in finite
choices remain enums.

```yaml
policy_variant:
  name: extension
  extension_type: acme.block_sparse_policy
  hidden_size: 1536
  attach_site: within_visual_core
  options:
    block_size: 64
    history_chunks: 8

action_decoder:
  name: extension
  extension_type: acme.flow_action_decoder
  hidden_size: 1536
  action_dim: 7
  action_horizon: 16
  options:
    loss: smooth_l1
```

`options` is copied into the frozen config envelope and must be a mapping with
string keys. Parse it into an application-owned typed dataclass inside the
builder. Shared data, backbone, training, and inference semantics stay in their
normal typed config sections.

## Adding A Policy Variant

1. Implement the `PolicyVariant` contract in the extension package.
2. Define its required methods:
   `required_visual_stages`, `prepare_train_inputs`, `forward_train`,
   `prepare_infer_state`, and `forward_infer_step`.
3. Register a builder under the YAML `extension_type`.
4. Add config, construction, gradient, and recurrent-inference tests.

```python
from open_wam.configs import ExtensionPolicyConfig
from open_wam.pipelines import register_policy_variant

from .policy import AcmePolicy, AcmePolicyOptions


def build_policy(experiment):
    config = experiment.policy_variant
    assert isinstance(config, ExtensionPolicyConfig)
    options = AcmePolicyOptions.from_mapping(config.options)
    return AcmePolicy(config=config, options=options)


def register_open_wam() -> None:
    register_policy_variant(
        "acme.block_sparse_policy",
        build_policy,
        description="ACME block-sparse policy.",
    )
```

## Adding An Action Decoder

1. Implement the `ActionDecoder` contract in the extension package.
2. Keep final output construction, supervised loss, and action sampling in the
   decoder.
3. Register a builder under the YAML `extension_type`.
4. Add loss/output shape tests.
5. Keep result schemas backward compatible when adding outputs.

```python
from open_wam.configs import ExtensionActionDecoderConfig
from open_wam.pipelines import register_action_decoder

from .decoder import AcmeActionDecoder, AcmeDecoderOptions


def build_decoder(experiment):
    config = experiment.action_decoder
    assert isinstance(config, ExtensionActionDecoderConfig)
    options = AcmeDecoderOptions.from_mapping(config.options)
    return AcmeActionDecoder(config=config, options=options)


def register_open_wam() -> None:
    register_action_decoder("acme.flow_action_decoder", build_decoder)
```

The policy and decoder snippets are standalone examples. When one module owns
both, call both registration functions from the same `register_open_wam` hook.

## Custom Attention

Attention visibility is data passed through the policy/runtime boundary, not a
backbone subclass. A custom policy can build a
`open_wam.models.common.PreparedAttentionProfile` and pass it in
`VisualCoreInput.attention_profile` through the dense runtime program:

```python
from open_wam.models.visual_tower import (
    RuntimeStepInput,
    VisualCoreInput,
    build_dense_runtime_program,
)

result = visual_tower.execute_runtime_step(
    RuntimeStepInput(
        program=build_dense_runtime_program(),
        core_input=VisualCoreInput(
            tokens=tokens,
            attention_profile=prepared_profile,
        ),
    )
)
```

The profile may provide dense boolean masks, FlexAttention block masks, or
both. Keep sequence packing and mask construction in the policy extension;
the visual tower owns kernel selection and backbone execution. The exact
dual-stream M1/M5 programs are checkpoint-compatibility contracts with fixed
layout semantics, not general attention extension points.

### Cache Policy

`open_wam.models.common.cache_backends` contains the parameter-free cache
operations used by the shared visual runtime. Custom policies that retain the
built-in cache formats can reuse:

- `prepare_sdpa_mask` and `prepend_cached_prefix_mask`;
- `resolve_slot_pool_prefix_visibility`;
- `packed_slot_pool_query_sequence_ids`;
- `retained_slot_pool_indices_for_current_write`;
- `merge_attention_cache_entries`.

Attention profiles decide which tokens may interact. Cache policy decides how
already-computed keys and values are represented, retained, and prepended.
Neither contract owns learned parameters. A new cache representation still
requires a backend integration; do not encode its retention rules inside a
policy variant or transformer block.

## Contract Rules

- Registration hooks configure contracts; they must not start jobs or mutate
  global training state.
- Use globally unique extension identifiers. Duplicate registration fails
  unless the caller explicitly requests a process-local replacement.
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
