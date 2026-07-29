# Cookbook: Add A New Method

Use this when the research idea changes policy semantics while keeping the
shared visual tower.

## Extension Package

Keep the implementation outside the Open-WAM source tree:

```text
acme_open_wam/
  __init__.py
  config.py
  policy.py
  registration.py
```

Implement `PolicyVariant` in `policy.py`. Parse the open `options` mapping into
an application-owned frozen dataclass in `config.py`. Register the builder in
the module hook:

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

Select it without adding a core enum:

```yaml
policy_variant:
  name: extension
  extension_type: acme.policy
  hidden_size: 1536
  attach_site: post_visual_core
  options:
    history_frames: 8
```

## Files Not To Touch By Default

- Do not add method-specific top-level runtime classes.
- Do not bypass `VisualTower`.
- Do not reintroduce `ActionHead` or `UnifiedWAMPipeline`.
- Do not edit unrelated method configs.
- Do not mutate `POLICY_VARIANT_BUILDERS`; use the registration function.

## Contract

The new method should fit:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

Implement:

- `required_visual_stages`
- `prepare_train_inputs`
- `forward_train`
- `prepare_infer_state`
- `forward_infer_step`

## Validation

```bash
open-wam-validate-config experiment.yaml
open-wam-train --extension acme_open_wam.registration --cfg experiment.yaml
```

Test one deterministic train step through gradients and one recurrent inference
step before relying on the extension in an experiment. Keep GPU or simulator
validation in a labeled or self-hosted tier.
