# Agent Style Guide

This file is the repo-level style guide for agent and human contributors.

It is intentionally practical: when making code changes, prefer the patterns
below unless there is a strong repo-specific reason to do otherwise.

## Core Architecture

- Preserve the current top-level runtime boundary:
  `ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder`
- Keep the shared visual stack stable across policy experiments.
- Express method differences through shared contracts such as:
  - runtime programs
  - sequence semantics
  - cache policy
  - schedulers
  - decoders
- Do not reintroduce deprecated `ActionHead` / `UnifiedWAMPipeline` paradigms.

## Abstraction Style

- Prefer generic, composable abstractions over method-named infrastructure.
- Avoid classes like `ExactMethod1Trainer` when the real concept is something
  more general such as:
  - batch adapter
  - loop policy
  - strategy backend
  - checkpoint manager
  - log sink
  - runtime program
- If a feature starts from one benchmark or one method, still name the shared
  abstraction after its role, not after the first use case.

## Config Style

- If a config field is a finite public choice, model it as an explicit enum.
- Keep experiment YAMLs string-friendly; convert those strings into enums at the
  typed config boundary.
- Compare enum members in Python code. Do not add new string-literal equality
  checks for enum-backed config fields.
- Keep genuinely open-ended values as plain strings:
  - dataset names
  - paths
  - row keys
  - free-form labels
- Group experiment YAML knobs by purpose and annotate sections with short,
  readable comments.

## Enum Rules

- Add new enum-backed config choices in `src/open_wam/configs/enums.py`.
- Add a short docstring or comment when the meaning is not obvious.
- When extending a frozen config dataclass, use the shared config coercion
  helpers rather than repeating inline mutation boilerplate.
- If a runtime/datapath consumes an enum-backed field, use the enum in that
  consumer too, not just in the config declaration.

## Data Layer

- Keep dataset-specific parsing inside dataset adapters selected by
  `data.dataset_type`.
- The public data contract should stay uniform across sources.
- Canonical RGB layout construction belongs in the data layer, not in the
  backbone.
- If supervision is transformed from raw dataset state/action, keep that logic
  explicit and typed.

## Variant and Runtime Boundaries

- `PolicyVariant` owns variant semantics.
- `VisualTower` owns shared visual execution and backbone-facing runtime hooks.
- `ActionDecoder` owns final supervised outputs and losses.
- If a new variant needs custom behavior, first ask whether it can be expressed
  through:
  - `required_visual_stages()`
  - prepared inputs
  - runtime-program selection
  - decoder changes
- Only add a new top-level abstraction when the shared contracts are no longer
  enough.

## Training Infrastructure

- Keep training runtime pieces generic and decoupled.
- Logging, checkpointing, scheduling, optimizer construction, and trainability
  controls should remain reusable across variants.
- Prefer config-driven behavior over variant-specific branching in the trainer.
- If a training behavior is method-specific today, try to express it as a
  general runtime or config knob before adding a special-case codepath.

## Notes and Docs

- Top-level `notes/` should describe the repo as it exists now.
- Historical plans and finished roadmaps belong under
  `notes/finished_roadmaps/`.
- When architecture changes, update beginner-facing notes, not just deep-dive
  internals.
- Do not leave docs describing removed paths as if they are still active.

## Testing Expectations

- Add or update focused tests for the surfaces you change.
- Prefer small, direct tests near the changed contract:
  - config loader tests
  - runtime/control tests
  - variant pipeline tests
  - dataset adapter tests
- For training/runtime changes, run at least one smoke path when feasible.
- If a change affects exact method-1 behavior, preserve the existing smoke and
  parity-oriented coverage.

## Branch and PR Workflow

- Start each PR from a fresh branch created off the latest `main`.
- Do not develop PR-sized work directly on `main`.
- Use conventional branch names such as `feat/...`, `fix/...`, `docs/...`,
  `refactor/...`, or `test/...`.
- Use a meaningful branch name, a clear PR title, and a detailed PR
  description that explains:
  - what changed
  - why it changed
  - user or developer impact
  - validation that was run
- Do not add `codex` branding or prefixes to branch names or PR titles.
- Before marking a PR ready for review, resolve all actionable Copilot comments
  and other automated review comments that are visible to you.
- If you notice unresolved Copilot comments while working on a branch, fix them
  or clearly surface the remaining blocker before handing the PR off.

## Edit Hygiene

- Do not revert unrelated user changes.
- Keep changes localized and typed where possible.
- Prefer readability over cleverness.
- If introducing a temporary compatibility path, mark it clearly and keep the
  new main path clean.

## When In Doubt

- Favor current repo structure over historical precedent.
- Favor explicit contracts over hidden conventions.
- Favor generic naming over benchmark-specific naming.
- Favor enum-backed public choices over magic strings.
