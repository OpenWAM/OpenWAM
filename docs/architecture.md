# Architecture

Open-WAM composes every maintained model through one boundary:

```text
ExperimentConfig
  -> VariantPipeline
  -> VisualTower
  -> PolicyVariant
  -> ActionDecoder
```

The boundary separates shared visual execution from policy experiments. A new
dataset, attention pattern, policy architecture, or decoder should fit one of
these roles instead of adding a parallel training or inference stack.

## Configuration Axes

Video-action experiments are described by independent choices:

| Axis | Meaning | Examples |
| --- | --- | --- |
| Architecture | Parameter ownership and execution topology | `parallel_stream`, `dual_expert` |
| Program | Same-chunk video/action conditioning and supervision | `video_then_action`, `action_then_video`, `joint`, `decoupled_same_step`, `video_noisy_to_action`, `action_noisy_to_video`, `generalist_joint_denoising` |
| Sequence contract | Prefix, history, proprio, loss range, and chunk semantics | `default`, `legacy_prefix_single_frame_perchunk_proprio` |
| Numerical backend | Packing, attention implementation, cache writes, and denoising execution | exact packed stream, packed dual expert, split cache |
| Decoder | Final predictions, supervised losses, and rollout plan | `parallel_stream_decoder`, `dual_expert_decoder` |

`policy_variant.program` is the public switch for the six standard
video-action programs. The loader derives the lower-level
`current_block_coupling`; users do not need to set both. For example:

```bash
open-wam-train \
  --cfg configs/experiments/dual_expert_libero_joint.yaml \
  --set policy_variant.program=video_then_action
```

Named YAMLs remain available for reproducible runs. The override above and the
matching named config resolve to the same typed program and coupling.

## Built-In Architectures

### Parallel Stream

`parallel_stream` packs video and action streams into one shared transformer.
Its architecture package owns stream packing, exact attention layouts, cache
lifecycle, proprio insertion, and recurrent inference:

```text
src/open_wam/models/policy_variants/parallel_stream/
```

The `lingbot_exact` runtime name denotes a checkpoint-compatible numerical
backend. It is not a separate architecture and does not determine the
video/action program.

### Dual Expert

`dual_expert` uses separate video and action transformer parameters and
executes paired blocks under the selected visibility program. Its package owns
expert initialization, packed and split-cache execution, recurrent history,
and checkpoint-compatible block routing:

```text
src/open_wam/models/policy_variants/dual_expert/
```

The two architecture packages must not import one another. Shared semantics
belong in `models/common`, typed policy contracts, or the data layer.

## Runtime Ownership

### ExperimentConfig

`open_wam.configs.load_experiment_config` is the YAML-to-dataclass boundary.
Finite public choices are enums. Dataset names, paths, extension identifiers,
and free-form labels remain strings. Compatibility normalization accepts old
config names and fields before typed construction; runtime code consumes only
canonical values.

### VariantPipeline

`VariantPipeline` orchestrates the stable sequence:

1. canonicalize input views;
2. request the visual stages needed by the policy;
3. let the policy prepare and execute architecture-specific tensors;
4. pass a typed policy output to the decoder;
5. return common train or inference outputs.

It does not inspect architecture-specific artifact keys. Policies hand decoder
payloads across `DecoderArtifactEnvelope`, and each decoder validates its own
contract and payload type.

### VisualTower

`VisualTower` owns the shared visual frontend, transformer-facing runtime
hooks, optional decode stage, and common runtime-program execution. It accepts
prepared attention profiles and runtime inputs; it does not decide policy
conditioning or supervision semantics.

### PolicyVariant

A policy variant owns:

- required visual stages;
- train input preparation;
- parameter topology and runtime-program selection;
- recurrent inference state and cache reconciliation;
- architecture-specific decoder artifacts.

It does not own final supervised action losses.

### ActionDecoder

An action decoder owns final action predictions, supervised losses, decoder
state, and the model-space action plan committed by rollout. Generic rollout
code asks the decoder for a plan instead of branching on an architecture.

## Generalist Joint Denoising

GJD is the `generalist_joint_denoising` program inside either architecture. A
sample selects one `GeneralistDenoisingMode`:

| Mode | Clean supplied modality | Active loss | Task text |
| --- | --- | --- | --- |
| `joint` | neither | video and action | retained |
| `action_conditioned_video` (FDM) | action in the action-noisy slot at timestep zero | video only | removed |
| `video_conditioned_action` (IDM) | video in the video-noisy slot at timestep zero | action only | removed |

Conditional real-demo and counterfactual samples share one rollout-style data
contract:

```text
latent frame 0     observed t0, clean history, no loss
latent frames 1..N future targets, supervised according to FDM or IDM mode
```

The t0 frame is always a singleton chunk. Future chunks retain the sampled GJD
geometry, including the maintained 1-to-4 frame randomization where configured.
Conditional attention exposes only the most recent clean video boundary, not a
long demonstration prefix. The data layer projects real-demo conditional
samples to this target-only layout and validates counterfactual metadata before
either architecture executes it.

The shared owner of mode semantics is
`open_wam.models.common.joint_conditioning`. Architecture code applies those
decisions to its own packing and cache representation. Joint-mode behavior is
unchanged by the conditional target-only transform.

`parallel_stream` and `dual_expert` are maintained architecture choices under
this same GJD paradigm. Their real-joint rows use the configured planning
prefix, while conditional FDM/IDM rows bypass that prefix and use the shared
target-only contract above. Select architecture through the experiment config:

```bash
open-wam-train \
  --config-name <architecture>_libero_generalist_joint_denoising \
  --set policy_variant.generalist_mode_text_token=true
```

A source checkout additionally includes `scripts/run_gjd_libero.sh` for named
ablation expansion and LIBERO rollout orchestration. It delegates training to
the same package entry point and is intentionally absent from distributions.

## Data Boundary

Dataset-specific parsing belongs in adapters selected by `data.dataset_type`.
Adapters emit the uniform `WAMSample` or `LatentWAMSample` contract. The data
layer owns:

- camera decoding and canonical RGB assembly;
- action and state transforms;
- temporal and latent alignment;
- sample geometry and loss-range metadata;
- mixed-source and counterfactual routing.

The model receives canonical tensors and typed metadata. It does not know a
dataset's native camera names, storage format, or simulator schema.

## Attention And Cache Boundary

Common attention contracts separate visibility from execution:

- `attention_contracts` defines typed profiles and normalized choices;
- `chunked_attention_visibility` defines token-pair visibility;
- `chunked_attention` produces dense or FlexAttention representations;
- `attention_backends` executes the selected representation.

Architecture packages own their exact packed layouts and cache payloads. A
custom policy should submit a common prepared attention profile through a
runtime program instead of modifying the shared backbone.

## Training And Checkpoints

The generic training runtime owns optimizer, scheduler, logging, validation,
distributed strategy, and checkpoint lifecycle. Policy differences enter only
through configured batches and pipeline outputs.

`CheckpointManager` writes model state and, when configured, exact-resume
training state. Architecture refactors must preserve parameter names,
registration order, state-dict keys, optimizer mapping, and recurrent cache
semantics for maintained checkpoints.

## Extension Boundary

Applications can register:

- dataset adapters with `register_dataset_adapter`;
- policy variants with `register_policy_variant`;
- action decoders with `register_action_decoder`;
- simulator backends through the simulator extension contract.

Extensions use open string identifiers and parse their own typed options.
Built-in finite choices remain enums. See [Extension SDK](extension_sdk.md) and
the cookbooks under `docs/cookbooks/`.

## Compatibility Policy

Historical `mot`, `MoT*`, `M1`, `M5`, and `*_heng_compatible` names are input
compatibility labels only. They resolve to canonical architecture names at the
config, import, or launcher boundary. Maintained implementation code, new
configs, run metadata, artifact manifests, and documentation use
`parallel_stream`, `dual_expert`, and explicit program names.

Compatibility may change labels and metadata, but it must not change model
numerics. The strict characterization suite compares outputs, losses,
gradients, optimizer updates, recurrent caches, checkpoint resume, and rollout
artifacts against immutable checkpoint-backed goldens.
