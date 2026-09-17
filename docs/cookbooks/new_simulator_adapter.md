# Cookbook: Add A Simulator Adapter

Use an installed extension package when a benchmark has a live environment
that can be stepped in closed loop. No OpenWAM source edit is required.

## Implement The Contract

Implement `SimulatorBackend` from `open_wam.sdk.simulator`. The adapter owns
environment construction and lifecycle, observation extraction, source-action
conversion, stepping, success detection, rendering, and close.

Keep dependency-heavy simulator imports inside the factory so importing the
extension hook remains lightweight.

```python
from open_wam.sdk.simulator import (
    SimulatorFactoryContext,
    register_simulator_adapter,
)


def build_adapter(context: SimulatorFactoryContext):
    from acme_sim import Environment

    from .adapter import AcmeAdapter

    environment = Environment(
        asset_root=context.local_paths.get("simulators.acme_root"),
        endpoint=context.options.get("endpoint", "local"),
    )
    return AcmeAdapter(environment)


def register_open_wam() -> None:
    register_simulator_adapter("acme", build_adapter)
```

The backend should expose reset with `EpisodeSpec`, return
`SimulatorObservation`, and return `ControlTransition` from each step. `done`
means terminal; `success` independently records task success. A timeout is not
a success. Both types, plus `ControlCommand`, are available from the simulator SDK.

`SimulatorObservation.state` is already in the configured model-ready state
encoding, with shape `[state_dim]` and finite values. Convert native robot
state in your backend. Policies with proprio conditioning require a state at
every consumed observation; missing rows and wrong-width vectors raise errors.
No-proprio policies may omit state. Short startup histories repeat the known
initial state along time, never pad missing state features with zeros.

`materialize_control(source_action, data_config=...)` converts a dataset-source
action into `ControlCommand(action=..., source_action=...)`. The first field is
passed to `step`; the second records that executed command in dataset-source
coordinates. Apply clipping or discrete gripper conversion before returning
the pair. Do not normalize or map dimensions here: `pipeline.action_adapter`
owns those conversions in both directions.

The shared controller reconciles actual images, actions, and proprio before
replanning. `--action-commit-mode first_frame` executes one complete model-frame
action group; `full_chunk` executes the whole prediction. Camera/action timing
must agree with the encoder stride. It is an error to supply a partial frame
and relabel it as a complete one.
The result retains a typed `termination` with the last transition and its
`info`, distinguishing task success, environment failure and the control limit.
JSON summaries report the reason; application-specific `info` stays on the
typed result so arbitrary simulator objects are not implicitly serialized.
`achieved_action_hz` is executed controls divided by `live_wall_time_s`, including
live replanning waits, pacing, stepping and rendering. `wall_time_s` additionally
includes model startup and planner teardown, not environment reset. Unused planner
errors and drain duration are separate fields and cannot change task success.

## Run It

```bash
openwam-sim-rollout \
  --extension acme_open_wam \
  --benchmark acme \
  --sim-option endpoint=localhost:5000 \
  --cfg experiment.yaml \
  --checkpoint checkpoint_step_1000
```

Use `--sim-option KEY=VALUE` for application-owned construction values. Put
machine paths in `configs/local_paths.yaml`; the factory receives the resolved
mapping through `context.local_paths`.

## Validation

Test the factory and backend against fake environments in ordinary CI. Add one
closed-loop smoke that checks reset, action conversion, step, success, render,
and close. Real simulator checks belong in self-hosted, scheduled, or
label-gated jobs that retain result JSON and videos.

Only built-in integrations maintained by this repository belong under
`src/open_wam/integrations/`. A third-party benchmark should remain an
out-of-tree extension.
