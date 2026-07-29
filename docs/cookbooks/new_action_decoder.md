# Cookbook: Add An Action Decoder

Use this when the policy attachment stays the same but the supervised action
output, loss, or sampling backend changes.

## Extension Package

Implement `ActionDecoder` in an installed application package, then register a
builder:

```python
from open_wam.configs import ExtensionActionDecoderConfig
from open_wam.pipelines import register_action_decoder

from .config import AcmeDecoderOptions
from .decoder import AcmeActionDecoder


def build_decoder(experiment):
    config = experiment.action_decoder
    assert isinstance(config, ExtensionActionDecoderConfig)
    return AcmeActionDecoder(
        config=config,
        options=AcmeDecoderOptions.from_mapping(config.options),
    )


def register_open_wam() -> None:
    register_action_decoder("acme.action_decoder", build_decoder)
```

Select it in YAML:

```yaml
action_decoder:
  name: extension
  extension_type: acme.action_decoder
  hidden_size: 1536
  action_dim: 7
  action_horizon: 16
  options:
    loss: smooth_l1
```

## Contract

The decoder owns final supervised outputs and losses. It should not own visual
execution or policy-variant semantics.

Required checks:

- action output shape is `[B, H_action, D_action]`
- loss honors action masks
- inference uses the configured sampler/step count
- inactive action channels are handled according to the data-layer mapping

## Validation

```bash
open-wam-validate-config experiment.yaml
open-wam-train --extension acme_open_wam.registration --cfg experiment.yaml
```
