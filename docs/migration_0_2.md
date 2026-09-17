# Migrating From 0.1.x To 0.2

**Version 0.2.0** is an explicit pre-1.0
compatibility boundary. Keep 0.1.1 pinned until downstream extensions and
recorded experiments have been validated against the new contracts.

The architecture remains
`ExperimentConfig -> VariantPipeline -> VisualTower -> PolicyVariant -> ActionDecoder`.
The change consolidates inference and closed-loop execution; it does not add
another trainer or require a dependency upgrade.

## Inference Context And State

Import extension contracts through `open_wam.sdk.policy`.

| 0.1.x usage | 0.2 contract |
| --- | --- |
| `PolicyInferContext(extra={...})` | Use explicit `task_text`, `metadata`, `sample_seed`, `initial_video_noise`, `initial_action_noise`, and request fields as appropriate. |
| Mutating a context after construction | Contexts are frozen; construct a new context or use `dataclasses.replace`. |
| `PolicyInferState(step_index=..., cache=...)` | The immutable state owns a cursor, variant state, decoder state, geometry, revision, and observed-frame boundary. Consume the returned `next_state`; do not synthesize a cursor from an old step counter. |
| `state.step_index = ...` | `step_index` is a read-only view of `cursor.block_index`. Variant/session logic publishes the advanced cursor. |
| `state.bind_temporal_geometry(geometry)` | Use the returned `state.with_temporal_geometry(geometry)`; it validates rather than mutating the input. |
| Arbitrary shared `state.cache` payloads | Keep recurrent state in the typed variant/decoder session. Built-in feature caching is local to a denoising call, not a session-wide K/V store. |

For example, task text and row metadata are explicit batch-aligned inputs:

```python
from open_wam.sdk.policy import PolicyInferContext

context = PolicyInferContext(
    task_text=("put the mug on the plate",),
    metadata=({"episode_id": 0},),
    sample_seed=0,
)
```

This illustrates input construction, not a complete model call. A policy may
also require model-ready proprioception and resolved temporal geometry.
Do not move retired execution flags into `metadata`; metadata is not an
escape hatch for untyped runtime controls.

Published state tensors must also be treated as read-only: a frozen dataclass
does not stop in-place tensor mutation. Observed history is committed explicitly
after execution, independently of speculative predictions.

## Action Decoders

`ActionDecoder.build_rollout_plan(output)` now requires an action prediction
with shape `[1, steps, channels]` and returns the full chunk as detached FP32
CPU actions. It rejects multiple environment rows instead of silently taking
the first, and it does not interpret `aux["current_action"]`.

Override the hook for a custom release/slicing policy. Override
`commit_rollout_plan(state, plan)` by **returning** the new decoder state,
without modifying the input. An old override that mutates state and returns
`None` must be migrated, not reused unchanged.

The pipeline applies the output action mask before decoder plan hooks.
Integration code converts the plan from model space into the dataset action
schema and executable simulator controls. See the
[decoder cookbook](cookbooks/new_action_decoder.md).

## Simulator Extensions

Import `ControlCommand` and `ControlTransition` from
`open_wam.sdk.simulator`; `SimulatorStepResult` is no longer exported.

Replace `action_from_model_action(...)` with
`materialize_control(source_action, *, data_config) -> ControlCommand`.
The input is already in the dataset's source action schema, not the raw model
output. Return both the executable `action` and the corresponding
`source_action` after any clipping/conversion. History must record what was
executed, not an unclipped prediction.

`step(action)` still takes the executable action array, not the command
object. It returns `ControlTransition[SimulatorObservation]` with separate
`done`, `success`, reward, and info fields. Do not infer success from episode
termination or label a scheduler action budget as native environment
termination. The shared engine records the budget reason separately.

Proprio-conditioned policies require finite, exact-width, model-ready
`SimulatorObservation.state`; native EEF/qpos conversion belongs in the
adapter. See the [simulator cookbook](cookbooks/new_simulator_adapter.md).

## Rollout Entry Points And History

Use `scripts/run_libero_policy.py` and
`scripts/run_libero_policy_batch.py` for source-checkout blocking LIBERO
evaluation. For fixed-rate LIBERO use `scripts/run_libero_realtime_sandbox.py`;
for registered simulator adapters use `openwam-sim-rollout`.
Consult each maintained parser rather than forwarding flags from a retired
architecture-named visualization script.

The retired `LingbotExactRunner` and architecture-private executors are not
compatibility aliases. Package integrations use `VariantRolloutRunner` with
the shared planner/lifecycle/engine; see
[Extension SDK](extension_sdk.md) and [Architecture](architecture.md).

Fallback-freeze/quarantine policies, including `freeze_until_clean_chunk`,
are removed. Actual fallback controls and observations enter canonical
history. There is no semantics-preserving flag substitution that hides those
transitions. Use blocking control when waiting without stepping is intended,
or fixed-rate execution when the simulator must keep running, and record that
scheduling choice with the result.

## Numerical And Training Expectations

- Startup uses one observed frame with a masked action placeholder; only
  future predictions are returned. Cache enablement does not choose startup
  semantics.
- Inference honors explicit per-stream CFG and draws noise in stage order for
  updated modalities. The same global seed can produce a different action
  trace from a retired runner. Controlled numerical comparisons must also
  match per-modality noise, observations, guidance, and geometry.
- Parallel inference preserves its requested chunk length and configured
  dtype/device. It no longer interpolates a short action chunk to the training
  horizon or silently converts the model to BF16. Training decoding is
  unchanged.
- `inference.use_cache=true` reuses dependency-invariant features within a
  denoising call. It does not promise persistent full-prefix equivalence.
  The maintained causal chunk-conditioned preset keeps caching off; explicit
  cache-on use needs checkpoint/task validation.
- `trainer.strategy=fsdp` now applies with one worker, including mixed
  precision and optional CPU offload. To preserve an older one-rank unwrapped
  run, explicitly select `trainer.strategy=single_device`. Check training
  resume and numerics before changing strategies. Worker count and effective
  batch size remain launcher/config responsibilities.
- Explicit `trainer.accelerator=cpu` FSDP uses a CPU mesh even when CUDA is
  visible. This is separate from GPU FSDP's optional CPU offload, which keeps
  its CUDA compute mesh.

This release does not introduce a new weight format or require tensor
conversion solely for the execution refactor. That is not a guarantee of
arbitrary old checkpoint/config compatibility or identical rollouts.
Retain strict checkpoint loading, inspect its compatibility report, and test
full-state resume separately from weight initialization.

Training recipes and third-party dependency pins are not changed by the
release-preparation branch. The release includes intentional inference and
one-rank FSDP behavior changes already merged into the implementation, plus
an explicit-CPU FSDP placement correction found during release testing.

## Upgrade Checklist

1. Preserve the old environment, checkpoint, resolved config, seed, and
   baseline artifacts before changing versions.
2. Migrate SDK extensions and launchers using the contracts above; validate
   authored configs and checkpoint loading without partial-load bypasses.
3. Run one forward/backward update and full-state save/resume for training
   integrations, plus a complete representative checkpoint rollout.
4. Record output geometry, execution budgets, renderer, success, action traces,
   and resolved guidance. Do not interpret an import or tiny-model pass as
   unchanged benchmark success.
5. Use the corresponding source revision and frozen lock for characterization;
   a wheel install resolves allowed dependency ranges instead.

The draft does not claim new task-success rates or universal old/new bitwise
parity. See [Compatibility](compatibility.md), [Testing](testing.md), and
[Releases](release.md). Deferred upgrades leave existing, expiring
dependency-risk exceptions in force; they do not fix the vulnerabilities.
