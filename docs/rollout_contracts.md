# Rollout Contracts

Use these contracts when integrating a policy with a simulator or an external
control loop. For ready-to-run commands, see
[Training and Inference](running_experiments.md#libero-inference); for model
composition, see [Video-to-Action Composition](video_action_composition.md).

## Ownership

| Role | Owns |
| --- | --- |
| Benchmark driver and control adapter | Environment I/O and native control conversion |
| `RolloutEngine` | Plan acceptance, control execution, termination, and history commits |
| `RolloutPlanner` | Model transactions over a typed, opaque session |
| `VariantRolloutRunner` | Pipeline inference and observed-history reconciliation |
| `ActionDecoder` | Model-space action plans and decoder state |

A planner's `plan()` returns a candidate; `observe()` commits an executed
interval without generating another plan. Neither steps the environment.
`PolicyPlanner` implements this for a single pipeline. Blocking LIBERO and
independent video/action composition use the same engine.

## Session And Output

`PolicyInferState` and `VariantRolloutSession` are immutable publications.
Treat their tensors as read-only. Inference and
`runner.reconcile_observed_history(session=..., history=...)` return a new
session; the control loop adopts it only after the operation succeeds.

| Field | Meaning |
| --- | --- |
| `cursor.current_start_frame` | Exclusive prediction end, hence the next generation position |
| `observed_frame_end` | Exclusive end of committed observations |
| `revision` | Session revision carried with tensors and bound temporal geometry |

The interval between observed and prediction ends is speculative.
Rejected candidates do not mutate the accepted session. This isolates model
state, not global random-number streams.

`PolicyInferContext` carries typed geometry, conditioning, initial-noise, and
output requests. `PolicyInferOutput` publishes `generated_video`,
`generated_span`, `generation_frame_start`, and telemetry. Use those fields
for alignment, not diagnostic `aux` dictionaries.

`VariantRolloutStepOutput.action_plan` contains decoder-owned model-space
actions and their frame span for one environment. The pipeline applies output
masks before decoder rollout-plan and state-commit hooks. Decoders needing final
executable actions should use those hooks rather than duplicate output masking.

## Observed History

Commit the controls actually executed and the observations covering their full
interval. Do not substitute speculative actions, a lone `previous_action`,
or later images for missing observations.

`PolicyObservedHistory.action_mask` is an optional binary
`[B, T_action, 1]` mask. It permits an aligned t0 slot without exposing a
fictitious startup action. Realtime encoding uses a raw anchor observation;
its anchor latent is excluded from the replacement interval.

The pipeline's `ActionSpaceAdapter` converts model actions to source controls
and executed source controls back to model-space history. Simulator backends
return a `ControlCommand` with native controls and dataset-source controls.
Fallback is created in environment space and recorded with its actual
observation. Realtime reconciliation requires raw action targets: pose-target
consumers cannot reconstruct their targets from fallback controller commands.

After partial execution, replace the speculative interval with the executed
prefix and align the next generation position to that prefix. Action-only
generation creates no video history: supply the corresponding observed video
before generating again.

Proprio-conditioned policies require finite, exact-width model-ready states.
An adapter must not pad or truncate state vectors to make them fit. During
reconciliation, omitted replacement proprio retains only existing estimates
for the executed prefix; it cannot fill newly extended history. Text remains
part of the request/session and does not change as a side effect of a commit.

## Temporal Geometry

`ResolvedRolloutTemporalContract` derives frame/control origins, densities,
raw windows, and interval validation from the assembled policy and frontend.
It is not another user-authored config.

`first_frame` execution means the smallest complete model-frame group, not
one low-level control. `full_chunk` executes the complete plan. Model chunk
size, requested generation extent, and executed prefix are distinct:
short execution must not relabel unused predictions as observed history.

Conditional FDM/IDM uses one latest video boundary and aligned proprio, without
instructions. Joint planning retains configured history and text. Changing
architecture must not reinterpret those rules.

## Scheduling And Termination

`RolloutEngine.control_stream()` yields controls and accepts typed
`ControlTransition` observations. `run(..., step=backend.step)` drives it
synchronously; external step APIs can use the same stream. Terminal failure
stops execution without counting as success.

Only planners declaring asynchronous support may run in the background.
The LIBERO streaming-VAE planner is blocking because its encoder is mutable.
Partially stale candidates are rejected atomically rather than reusing a
conveniently aligned suffix.

`max_plans` stops after complete executed plans, observing the final
nonterminal interval without another prediction. Environment termination and
action limits take precedence. Terminal partial model frames are recorded,
not padded into complete observations. Zero budgets do no model work.

On termination/reset, queued work is cancelled and running model work is
drained before reuse. Teardown errors are recorded in `planner_teardown`
rather than replacing the terminal result. Stream `close()` returns a
`RolloutResult`; CALVIN retains it as `last_rollout_result` after reset.

## Results And Timing

Results retain the terminal transition and benchmark information.
`observed_control_end` identifies reconciled history; `next_control_index`
identifies executed controls. Any gap, including a terminal partial frame,
remains recorded. A policy session alone is not a resumable environment snapshot.

`live_wall_time_s` measures live control time. `wall_time_s` also includes
model startup and teardown; neither includes environment reset. Report teardown
latency separately when comparing control throughput.

Test custom integrations for partial execution, action-space conversion,
missing observations, rejected candidates, and terminal cleanup. See
[Testing](testing.md) and [Add A Simulator Adapter](cookbooks/new_simulator_adapter.md).
