# Cookbook: Add A New Method

Use this when the research idea changes policy semantics while keeping the
shared visual tower.

## Files To Touch

- `src/open_wam/configs/enums.py`: add one `PolicyVariantName` value when the
  variant is a public finite choice.
- `src/open_wam/configs/policy_variant.py`: add a typed config dataclass.
- `src/open_wam/models/policy_variants/`: implement the `PolicyVariant`.
- `src/open_wam/pipelines/factory.py`: register a builder while registry
  migration is in progress.
- `configs/examples/`: add one tiny smoke config.
- `tests/`: add config, factory, and shape tests.
- `docs/cards/`: add an experiment or fixture card.

## Files Not To Touch By Default

- Do not add method-specific top-level runtime classes.
- Do not bypass `VisualTower`.
- Do not reintroduce `ActionHead` or `UnifiedWAMPipeline`.
- Do not edit unrelated method configs.

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
open-wam-validate-config configs/examples/<new_method_smoke>.yaml
uv run --extra train pytest tests/<new_method_test>.py -q
```

Keep GPU or simulator validation in a labeled or self-hosted tier.
