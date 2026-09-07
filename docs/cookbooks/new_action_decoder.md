# Cookbook: Add An Action Decoder

Use this when the policy attachment stays the same but the supervised action
output, loss, or sampling backend changes.

## Extension Package

Implement `ActionDecoder` in an installed application package, then register a
builder:

```python
from open_wam.sdk.config import ExtensionActionDecoderConfig
from open_wam.sdk.policy import register_action_decoder

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

`ActionDecoder.build_rollout_plan()` is the inference-to-environment boundary.
The default implementation releases the full `[H_action, D_action]` prediction
and also supports the standard cached `current_action` output. A decoder with a
different cache or commit policy should override `build_rollout_plan()` and
return an `ActionDecoderRolloutPlan` containing detached float32 CPU actions.
Override `commit_rollout_plan()` when releasing multiple actions must advance
decoder-owned state. Benchmark integrations consume this typed plan and must
not interpret decoder-private `aux` keys.

Required checks:

- action output shape is `[B, H_action, D_action]`
- loss honors action masks
- inference uses the configured sampler/step count
- inactive action channels are handled according to the data-layer mapping
- rollout-plan actions have shape `[steps, D_action]` on CPU in float32
- cached decoder state advances past every action released by the plan

## Validation

```bash
openwam-validate-config experiment.yaml
openwam-train --extension acme_open_wam.registration --cfg experiment.yaml
```
