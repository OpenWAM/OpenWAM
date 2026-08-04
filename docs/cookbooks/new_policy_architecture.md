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
from open_wam.configs import ExtensionPolicyConfig
from open_wam.pipelines import register_policy_variant

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

Use `DecoderArtifactEnvelope` for architecture-specific policy-to-decoder
tensors. Keep sequence layout, recurrent state, and cache transitions in the
policy; keep final supervised outputs and losses in the decoder.

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
