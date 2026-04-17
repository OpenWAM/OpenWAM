# Contributing To Open-WAM

Open-WAM is organized around this runtime boundary:

```text
ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder
```

Keep changes compatible with that boundary unless the PR explicitly proposes an
architecture change.

## Runtime Compatibility

Runtime modernization is allowed and encouraged. The compatibility rule is:
introduce the new path first, keep old experiment paths working through
wrappers or aliases, warn before deprecating, and remove legacy only in a later
explicit removal PR.

When changing runtime, scripts, configs, or result schemas:

- keep existing experiment YAMLs loadable
- keep current checkpoint layouts loadable
- keep existing root `scripts/...` commands callable through wrappers or clear
  migration messages
- keep old config fields accepted while introducing new names
- preserve visual packing, action packing, denoising-step semantics, cache
  semantics, scheduler semantics, and policy outputs unless the PR is explicitly
  a behavior change
- write both old and new result fields for one compatibility window if an output
  schema changes

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
uv run --extra train pytest -m "unit or smoke or integration"
```

GPU/sim/data tests should skip clearly unless their documented resource gate is
set.

For extension work, start from the cookbooks under `docs/cookbooks/` and add a
static config validation command:

```bash
uv run open-wam-validate-config configs/examples/<your_config>.yaml
```

## Pull Request Checklist

- State whether the PR is docs-only, test-only, packaging-only, wrapper-only,
  runtime-modernization, behavior-change, or legacy-removal.
- List the command/config/checkpoint surfaces touched.
- Explain old path -> new path compatibility if any public command or config
  name changes.
- Run `git diff --check`.
- Run the relevant pytest tier.
- For behavior changes, include before/after numbers and exact commands.
