# Extension SDK

Open-WAM extensions are ordinary installed Python modules. They register
role-based components without modifying the Open-WAM source tree.

## Compatibility Boundary

New integrations should import from these role-specific modules:

- `open_wam.sdk.config`: typed config envelopes, loading, and resource resolution
- `open_wam.sdk.data`: sample contracts and dataset registration
- `open_wam.sdk.policy`: policy, decoder, attention, and visual runtime contracts
- `open_wam.sdk.simulator`: simulator protocol and factory registration
- `open_wam.sdk.results`: versioned results and provenance

These modules are the compatibility-managed Python SDK. Historical broad
facades such as `open_wam.configs`, `open_wam.data`, and
`open_wam.pipelines` remain import-compatible during the pre-1.0 migration,
but their complete symbol sets are not a promise that every implementation
helper is stable. Modules below `open_wam.models.*` are internal unless a
contract is re-exported by `open_wam.sdk.policy`.

The wheel ships a `py.typed` marker, so type checkers consume annotations from
these SDK modules directly. Public stability still follows the role-specific
SDK boundary above; typing visibility does not make internal implementation
modules compatibility-managed.

The base install supports config and result tooling. Install `open-wam[torch]`
for dataset, policy, decoder, and attention extensions; use
`open-wam[train]`, `open-wam[eval]`, or `open-wam[sim]` for the corresponding
runnable command. An extension package should declare the narrowest extra its
runtime actually needs.

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
from open_wam.sdk.data import register_dataset_adapter

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

The packaged `templates/extension_method/` example is runnable and imports
Open-WAM only through these SDK modules. From a source checkout:

```bash
uv run --extra train open-wam-train \
  --cfg templates/extension_method/config.yaml \
  --extension templates.extension_method.extension
```

Its normalized visual-token policy and masked-MSE decoder are deliberately
small. Use them to verify registration, gradients, inference state, and
packaging before replacing one component at a time.

## Registration APIs

- dataset adapters: `open_wam.sdk.data.register_dataset_adapter`
- policy variants: `open_wam.sdk.policy.register_policy_variant`
- action decoders: `open_wam.sdk.policy.register_action_decoder`
- simulator adapters: `open_wam.sdk.simulator.register_simulator_adapter`

The active architectural boundary remains:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

## Choosing The Extension Level

Use the first level that can express the change:

1. **Config only:** select existing programs, layouts, schedules, cache
   policies, samplers, or decoder behavior in YAML.
2. **SDK extension:** register an application-owned dataset adapter, policy
   variant, action decoder, or simulator adapter from an installed package.
3. **Core contribution:** add a shared visual backend, exact sequence family,
   cache representation, training backend, or other reusable runtime contract.

Do not create a new architecture for a mask, loss, or data-layout change. Do
not monkeypatch the shared runtime when the behavior needs a typed in-tree
contract.

| Desired change | Owning boundary | Supported path |
| --- | --- | --- |
| Change an existing program, geometry, loss weight, or optimizer setting | `ExperimentConfig` | YAML or `--set`; no Python required |
| Read a new storage format, camera schema, or action/state representation | dataset adapter | Register `raw_builder` and/or `latent_builder` through `open_wam.sdk.data` |
| Check dataset-owned files before startup | dataset adapter | Register an `artifact_resolver` |
| Change dataset mixing, weighting, or distributed sample order | dataset and sampler | Implement the dataset sampling contract; shared advanced samplers remain provisional infrastructure |
| Add policy parameters, dense sequence semantics, or recurrent state | `PolicyVariant` | Register an extension policy through `open_wam.sdk.policy` |
| Add dense attention visibility over the shared core | `PolicyVariant` and attention profile | Submit a `PreparedAttentionProfile` through the dense runtime program |
| Change final action outputs, losses, sampling, or committed action count | `ActionDecoder` | Register an extension decoder through `open_wam.sdk.policy` |
| Add a simulator backend | `SimulatorBackend` | Register a factory through `open_wam.sdk.simulator`; the standard simulator CLI resolves the registered benchmark |
| Add an exact packed sequence family or cache tensor representation | `VisualTower` runtime | Contribute a generic in-tree contract and parity tests; there is no runtime-backend registry |
| Replace the visual frontend, backbone, or decode stack | `VisualTower` | Contribute in-tree and preserve checkpoint contracts |
| Add an optimizer, strategy, loop policy, checkpoint format, or log sink | training infrastructure | Select built-ins by config; new reusable implementations are currently in-tree contributions |
| Add an offline metric or application report | application evaluator | Build an application command around typed pipeline outputs |

## Core Boundary Ownership

### ExperimentConfig

Start from the nearest maintained YAML and change one ownership axis at a
time. Shared finite choices stay in typed config fields. Dataset-specific
settings belong in `data.adapter_options`; policy- and decoder-specific
settings belong in their extension `options` mappings and should be parsed
into frozen application dataclasses by the extension builder.

Do not subclass `ExperimentConfig` in an extension package: the built-in YAML
loader will not discover that subclass. A new shared finite choice requires an
in-tree enum and named cross-section validation. Application-specific choices
remain in the open extension envelope.

### VariantPipeline

`VariantPipeline` is composition infrastructure, not a plugin slot. It owns the
common train/inference order and connects one preprocessor, `VisualTower`,
`PolicyVariant`, and `ActionDecoder`. There is no
`register_variant_pipeline` API.

Customize through the adjacent contracts: request visual work from the policy,
prepare policy inputs and decoder artifacts there, and implement final outputs
and losses in the decoder. If those contracts cannot express a reusable
behavior, extend the pipeline contract in-tree and add train and recurrent
inference coverage for every maintained architecture.

### VisualTower

The tower owns the shared frontend, visual core, decode stage, runtime
execution, and cache lifecycle. Policies receive frontend output and may
request `"core"` and/or `"decode"` from `required_visual_stages()`.

Use the dense runtime plus `PreparedAttentionProfile` for custom visibility.
Adding a `RuntimeProgramSpec` name does not register an executor: a new exact
packing format, cache representation, or visual backbone requires an in-tree
runtime implementation and checkpoint/parity coverage. There is no
`register_visual_tower` API.

### PolicyVariant

Use a policy extension for application-owned learned parameters,
packing/conditioning semantics, runtime-program selection, or recurrent
inference state. Pass policy-specific outputs through a typed
`DecoderArtifactEnvelope`; do not expose decoder-private tensors through
unstructured pipeline keys. See [Adding A Policy Variant](#adding-a-policy-variant).

### ActionDecoder

Use a decoder extension when policy topology remains valid but final
prediction, supervision, sampling, or rollout commitment changes. Keep
simulator-space conversion in the simulator adapter. See
[Adding An Action Decoder](#adding-an-action-decoder).

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
- `artifact_resolver`: declares required files and directories for startup preflight
- both builders under one key when a source supports both paths

Use `DatasetArtifactRequirement` instead of putting benchmark-specific path
checks in the trainer. The runtime evaluates these requirements before model
construction and reports the config owner and remediation for every missing
required artifact. Optional requirements are retained in run-start provenance.

```python
from open_wam.sdk.data import DatasetArtifactKind, DatasetArtifactRequirement


def resolve_artifacts(config):
    return (
        DatasetArtifactRequirement(
            name="episode manifest",
            path=config.local_root,
            kind=DatasetArtifactKind.DIRECTORY,
            required=True,
            config_path="data.local_root",
            purpose="the ACME adapter discovers episodes below this root",
        ),
    )
```

Registration rejects accidental replacement. Use a globally unique
`dataset_type`; `replace=True` is reserved for intentional process-local
overrides.

### Distributed Sampling

A training dataset may implement
`build_train_sampler(*, world_size, rank)`. Prefer the shared contracts in
`open_wam.data` when an adapter needs advanced distributed sampling. Those
sampler implementations are provisional infrastructure rather than part of
the narrow extension SDK:

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
from open_wam.sdk.config import ExtensionPolicyConfig
from open_wam.sdk.policy import register_policy_variant

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
from open_wam.sdk.config import ExtensionActionDecoderConfig
from open_wam.sdk.policy import register_action_decoder

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

## Adding A Simulator

Simulator extensions register a factory under an application-owned benchmark
identifier. The factory receives only immutable generic options and local-path
aliases; it does not depend on Open-WAM's CLI parser.

```python
from open_wam.sdk.simulator import (
    SimulatorFactoryContext,
    register_simulator_adapter,
)

from .simulator import AcmeSimulator


def build_simulator(context: SimulatorFactoryContext) -> AcmeSimulator:
    return AcmeSimulator(
        endpoint=context.options["endpoint"],
        asset_root=context.local_paths.get("simulators.acme_root"),
    )


def register_open_wam() -> None:
    register_simulator_adapter("acme", build_simulator)
```

Invoke it without changing the Open-WAM repository:

```bash
open-wam-sim-rollout \
  --extension acme_open_wam \
  --benchmark acme \
  --sim-option endpoint=localhost:5000 \
  --cfg experiment.yaml
```

The returned object implements `SimulatorBackend`. Task, episode, and seed are
passed through `EpisodeSpec` at reset time; application-specific construction
values belong in repeatable `--sim-option KEY=VALUE` settings.

## Custom Attention

Attention visibility is data passed through the policy/runtime boundary, not a
backbone subclass. A custom policy can build a
`PreparedAttentionProfile` and pass it in
`VisualCoreInput.attention_profile` through the dense runtime program:

```python
from open_wam.sdk.policy import (
    PreparedAttentionProfile,
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
the shared visual runtime applies backend-ready profiles and owns backbone
execution. The exact parallel-stream and dual-expert backends are checkpoint
compatibility contracts with fixed layout semantics, not general attention
extension points.

The following built-in role modules are useful when contributing to Open-WAM
itself, but are not stable extension APIs. The built-in attention
implementation has three parameter-free roles:

- `open_wam.models.common.attention_contracts` owns profile records, coupling
  names, and semantic normalization;
- `open_wam.models.common.chunked_attention` assembles the maintained packed
  chunk masks; and
- `open_wam.models.common.attention_backends` selects and executes dense SDPA
  or FlexAttention representations.

`open_wam.models.common.attention_profiles` remains a historical import
facade. New integrations should depend on the role module matching what they
extend, or on the stable `open_wam.models.common` public exports above.

Built-in DualExpert checkpoint layouts are narrower policy-internal contracts:

- `open_wam.models.policy_variants.dual_expert.attention_unpacked` owns dense layouts
  used by unpacked training and joint denoising;
- `open_wam.models.policy_variants.dual_expert.attention_packed` owns exact packed
  coupling profiles; and
- `open_wam.models.policy_variants.dual_expert.attention_cached` owns split-cache
  action inference layouts.

`open_wam.models.policy_variants.dual_expert.attention` remains a compatibility
facade. Extensions implementing a new attention paradigm should normally
construct a common `PreparedAttentionProfile`; depend on a DualExpert role only when
the extension deliberately implements that exact built-in sequence layout.

Built-in DualExpert runtime controls are also split by parameter-free role:

- `open_wam.models.policy_variants.dual_expert.runtime_routes` selects a typed runtime
  route;
- `open_wam.models.policy_variants.dual_expert.rollout_geometry` resolves chunk,
  history, cache, and action-execution geometry;
- `open_wam.models.policy_variants.dual_expert.coupling_semantics` resolves block and
  timestep coupling; and
- `open_wam.models.policy_variants.dual_expert.inference_backend` validates and
  restores the inference backend selected by a route.

`open_wam.models.policy_variants.dual_expert.runtime_routing` is a compatibility
facade. These modules document the fixed built-in checkpoint contract; they
are not a registration API. A custom policy should express its behavior through
its `PolicyVariant`, runtime program, prepared attention profile, and decoder
rather than adding architecture-specific branches to these DualExpert owners.

### Shared Transformer Primitives

`open_wam.models.visual_tower` exports the shared Wan-style transformer
building blocks used by the visual core and action-side experts. Their
canonical owners are:

- `shared_transformer_support` for `SharedTransformerAttention` and
  `SharedTransformerBlock`, including the stable attention-backend patch
  point;
- `shared_transformer_embeddings` for timestep and rotary positional
  embeddings plus rotary application;
- `shared_transformer_layout` for chunk-slice and split-segment tensor
  helpers;
- `runtime_parameter_ops` for FSDP-safe linear, normalization, and
  feed-forward helpers.

The package-root exports and historical `shared_transformer_support` aggregate
remain compatible. New implementation code should import the role owner it
uses. These functions own learned transformer execution or the explicit tensor
operations supporting it, not sequence visibility or cache retention.
Extensions should normally submit an attention profile through
`VisualCoreInput`; use the lower-level primitives only when implementing a
genuinely new reusable block outside the built-in core.

### Cache Policy

The parameter-free cache API is split by role:

- `open_wam.models.common.cache_backend_contracts` defines backend specs,
  payload records, and backend selection;
- `open_wam.models.common.cache_layout_policy` defines attention-mask,
  prefix-visibility, packed-sequence, slot-retention, and prefix-merge policy;
- `open_wam.models.common.cache_backend_lifecycle` defines payload allocation,
  mutation, reset, and materialization.

`open_wam.models.common.cache_backends` remains a historical import and pickle
facade. New integrations should import the role module that owns the operation.
Custom policies that retain the built-in cache formats can reuse:

- `prepare_sdpa_mask` and `prepend_cached_prefix_mask`;
- `resolve_slot_pool_prefix_visibility`;
- `packed_slot_pool_query_sequence_ids`;
- `retained_slot_pool_indices_for_current_write`;
- `merge_attention_cache_entries`.

Custom runtime programs can use `init_cache_backend_payload`,
`update_slot_pool_layer_state`, `clear_cache_backend_payload`, and
`materialize_cache_backend_entries` from `cache_backend_lifecycle` without
depending on transformer execution. The payload types come from
`cache_backend_contracts`; do not duplicate their tensor-layout conventions in
a policy variant.

`open_wam.models.visual_tower.RuntimeCacheLifecycle` composes those backend
operations into initialization, named-branch, retention, cursor-advance, and
reset operations over the public `CacheState` contract. `VisualTower` exposes
the same operations as its stable runtime facade and supplies the current
backbone capability and layer count dynamically. The lifecycle is a frozen
plain object, not an `nn.Module`, so using or replacing it cannot add
checkpoint keys.

Runtime-backbone checkpoint selection and operational compatibility live in
`open_wam.models.visual_tower.runtime_backbone`. These helpers borrow the
tower-owned module rather than wrapping it, so custom runtime programs can
reuse access validation, device normalization, and cache reset without
creating a second parameter owner.

Attention profiles decide which tokens may interact. Cache policy decides how
already-computed keys and values are represented, retained, and prepended.
Neither contract owns learned parameters. A new cache representation still
requires a backend integration; do not encode its retention rules inside a
policy variant or transformer block.

## Training And Checkpoints

Built-in optimizer, scheduler, precision, strategy, loop, checkpoint, and
logging choices are config driven. They are not SDK registries. An application
may call a `VariantPipeline` from its own research loop, but then it owns
distributed coordination, exact resume, validation, and logging. Reusable
behavior belongs in a generic typed in-tree training contract, not a policy- or
benchmark-named trainer branch.

Registered policy and decoder modules participate in normal model and
full-training-state checkpoints. Treat parameter names, module registration
order, tensor shapes, optimizer mapping, and recurrent cache semantics as
compatibility contracts. A topology change creates a new checkpoint contract;
an orchestration-only change must preserve output, loss, gradient, optimizer,
and recurrent-inference parity.

## Extension Workflow

1. Copy the nearest maintained config and identify the owning boundary.
2. Keep application code in an installable package outside Open-WAM's built-in
   implementation directories.
3. Parse every open `options` mapping into a frozen application dataclass.
4. Implement the smallest supported contract and register it from one
   side-effect-free hook.
5. Pass the same `--extension module[:hook]` to every train, eval, sanity, and
   rollout command that constructs the component.
6. Test config parsing, shapes and masks, one forward/backward update,
   checkpoint load, and recurrent inference. Add distributed and simulator
   tiers when the component uses them.
7. Retain the resolved config and artifact manifest for every reported run.

## Contract Rules

- Registration hooks configure contracts; they must not start jobs or mutate
  global training state.
- Use globally unique extension identifiers. Duplicate registration fails
  unless the caller explicitly requests a process-local replacement.
- Use `open_wam.sdk.config.coerce_fields` for enum-backed extension settings;
  keep cross-section defaults and validation in a named config contract.
- Dataset parsing remains in data adapters.
- Policy semantics remain in `PolicyVariant`.
- Shared visual execution remains in `VisualTower`.
- Final supervised outputs and losses remain in `ActionDecoder`.
- Public finite choices are enum-backed; dataset names, row keys, paths, and
  extension labels remain open strings.

## Cookbooks

- `docs/cookbooks/new_policy_architecture.md`
- `docs/cookbooks/new_action_decoder.md`
- `docs/cookbooks/new_dataset.md`
- `docs/cookbooks/new_simulator_adapter.md`
- `docs/cookbooks/reproduce_result.md`
