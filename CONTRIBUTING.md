# Contributing To OpenWAM

OpenWAM is organized around this runtime boundary:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

Keep changes compatible with that boundary unless the PR explicitly proposes an
architecture change.

## Runtime Compatibility

Follow the [compatibility policy](docs/compatibility.md). Refactors preserve
supported configurations, checkpoint loading, and model behavior. Intentional
breaking changes require a migration guide; do not silently revive retired
fields or add compatibility branches to the active runtime.

For model-facing changes, check visual/action packing, denoising schedules,
attention, history, losses, and outputs. Test full-state resume when parameter
ownership or checkpoint handling changes. Document any intentional numerical
difference and its reproducibility limits.

## Extension Style

- Prefer registries, runtime programs, schedulers, decoders, and typed adapters
  over method-named infrastructure.
- Keep dataset-specific parsing inside dataset adapters selected by
  `data.dataset_type`.
- Keep canonical RGB packing in the data layer.
- Add enum-backed public config choices in `src/open_wam/configs/enums.py`.
- Compare enum members in Python code instead of raw strings for enum-backed
  fields.

## Local Paths And Private Artifacts

Do not commit machine-local dataset roots, private checkpoint paths, WandB
tokens, Hugging Face tokens, or simulator checkout paths.

Use:

- `configs/local_paths.sample.yaml` for public placeholders
- `configs/local_paths.yaml` for machine-local values, which is gitignored
- `OPEN_WAM_LOCAL_PATHS=/path/to/local_paths.yaml` to point at a different
  registry
- `configs/artifacts.sample.yaml` for public artifact manifest structure

## Testing Tiers

Use pytest markers to communicate required resources:

- `unit`: no GPU, no external data, no simulator
- `smoke`: small CPU-safe integration path
- `gpu`: requires CUDA
- `sim`: requires LIBERO, RoboTwin, or CALVIN simulator setup
- `data`: requires non-fixture local datasets
- `slow`: long-running train/eval/rollout
- `integration`: cross-component behavior that is larger than a unit test

Default local check:

```bash
CUDA_VISIBLE_DEVICES="" uv run --frozen --extra full pytest --strict-markers -q \
  -m "not (gpu or sim or data or slow)"
```

This is the same required CPU semantic gate used by CI. For a faster edit loop,
pass explicit test paths before the marker expression. GPU/sim/data tests should
skip clearly unless their documented resource gate is set.

For extension work, start from the cookbooks under `docs/cookbooks/` and add a
static config validation command:

```bash
uv run openwam-validate-config configs/examples/<your_config>.yaml
```

## Documentation And Tests

Write public documentation for a reader using or extending the current package.
Keep the README an entry point, guides task-oriented, and API contracts in the
SDK/reference pages. Changelogs describe release impact, not review discussions,
private experiments, test counts, or helper-by-helper refactor inventories.
Internal run records belong outside the public documentation.

Prefer tests of observable behavior: outputs, errors, gradients, state changes,
and supported imports. Parameterize meaningful input boundaries instead of
duplicating fixtures. Source inspection is useful for dependency rules and
import safety, not for freezing private helper names or incidental file layout.
Do not replace numerical or compatibility regression checks with prose.

## Pull Request Checklist

- State whether the PR is docs-only, test-only, packaging-only, wrapper-only,
  runtime-modernization, behavior-change, or legacy-removal.
- List the command/config/checkpoint surfaces touched.
- Explain old path -> new path compatibility if any public command or config
  name changes.
- Run `git diff --check`.
- Run the relevant pytest tier.
- For behavior changes, include before/after numbers and exact commands.
