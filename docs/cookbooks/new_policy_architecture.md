# Cookbook: Add A Policy Architecture

Use this when an experiment needs a new parameter topology or policy-owned
runtime while retaining Open-WAM's shared visual and training boundaries. If
only visibility or supervision changes, prefer a runtime program. If only the
final loss changes, add an action decoder instead.

## Extension Package

Keep application code outside the Open-WAM source tree:

```text
acme_open_wam/
  __init__.py
  config.py
  policy.py
  registration.py
```

Implement `PolicyVariant` in `policy.py`. Parse the open `options` mapping into
an application-owned frozen dataclass in `config.py`, then register a builder:

```python
from open_wam.sdk.config import ExtensionPolicyConfig
from open_wam.sdk.policy import register_policy_variant

from .config import AcmePolicyOptions
from .policy import AcmePolicy


def build_policy(experiment):
    config = experiment.policy_variant
    assert isinstance(config, ExtensionPolicyConfig)
    return AcmePolicy(
        config=config,
        options=AcmePolicyOptions.from_mapping(config.options),
    )


def register_open_wam() -> None:
    register_policy_variant("acme.policy", build_policy)
```

Select it without changing Open-WAM's finite built-in enum:

```yaml
policy_variant:
  name: extension
  extension_type: acme.policy
  hidden_size: 1536
  attach_site: post_visual_core
  options:
    history_frames: 8
```

## Contract

The extension must fit:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

Implement the policy hooks it uses:

- `required_visual_stages`
- `prepare_train_inputs`
- `forward_train`
- `prepare_infer_state`
- `forward_infer_step`

The frontend output is always available. Include `PolicyVisualStage.CORE` in
`required_visual_stages()` only when the policy needs the shared dense visual
core. The pipeline rejects unknown stage names before execution. Use the optional `initialize_for_training()` and
`reconcile_observed_history()` hooks instead of adding pipeline branches.

Set `proprio_context_mode` or `dynamics_mode_context_enabled` in the extension
policy config when the policy needs those shared-tower adapters. They are
configured before any policy modules are allocated. Override
`pipeline_requirements()` to validate model-space geometry and expose accepted
source-action shapes plus any channel projection when a backend maps source
actions into a wider model space. The factory verifies that the runtime
declaration matches the dataset, config, tower, and decoder;
architecture-specific dimension logic does not belong in the generic factory.

Use `DecoderArtifactEnvelope` for architecture-specific policy-to-decoder
tensors. Keep sequence layout, recurrent state, and cache transitions in the
policy; keep final supervised outputs and losses in the decoder.

For custom visibility, submit a `PreparedAttentionProfile` through the dense
runtime program. A new exact packed sequence family or visual backbone is not
an out-of-tree policy extension; it requires a shared in-tree runtime contract.

## Avoid New Infrastructure

- Do not add an architecture-specific trainer or top-level pipeline.
- Do not bypass `VisualTower` with a private backbone copy.
- Do not inspect experiment names in runtime code.
- Do not mutate built-in registries directly; use registration functions.
- Do not reintroduce `ActionHead` or `UnifiedWAMPipeline`.

## Validation

```bash
open-wam-validate-config experiment.yaml
open-wam-train --extension acme_open_wam.registration --cfg experiment.yaml
```

Add deterministic tests for config parsing, one forward/backward update, and
one recurrent inference step. Use a labeled GPU or simulator tier for tests
that require local assets.
