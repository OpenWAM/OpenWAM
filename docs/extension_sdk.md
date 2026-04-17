# Extension SDK

Open-WAM extension points should be role-based, not method-name-based.

## Current Registries

- dataset builders: `open_wam.data.register_dataset_builder`
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
4. Register the builder with `register_dataset_builder(dataset_type, builder)`.
5. Add a focused adapter test and one config-loader smoke.

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

## Compatibility Rule

New registry paths can become the default immediately, but old central factory
branches, config names, and script commands should remain as compatibility
shims until a later legacy-removal PR.

## Cookbooks

- `docs/cookbooks/new_method.md`
- `docs/cookbooks/new_action_decoder.md`
- `docs/cookbooks/new_dataset.md`
- `docs/cookbooks/new_simulator_adapter.md`
- `docs/cookbooks/reproduce_result.md`
