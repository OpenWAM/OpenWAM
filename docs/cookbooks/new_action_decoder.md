# Cookbook: Add An Action Decoder

Use this when the policy attachment stays the same but the supervised action
output, loss, or sampling backend changes.

## Files To Touch

- `src/open_wam/configs/enums.py`: add one `ActionDecoderName`.
- `src/open_wam/configs/action_decoder.py`: add a typed decoder config.
- `src/open_wam/models/action_decoders/`: implement the decoder.
- `src/open_wam/pipelines/factory.py`: register the decoder builder.
- `configs/examples/`: add one tiny config that uses the decoder.
- `tests/`: add loss/output shape tests and static config validation.

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
open-wam-validate-config configs/examples/<decoder_smoke>.yaml
uv run --extra train pytest tests/<decoder_test>.py -q
```
