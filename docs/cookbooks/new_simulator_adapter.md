# Cookbook: Add A Simulator Adapter

Use an installed extension package when a benchmark has a live environment
that can be stepped in closed loop. No OpenWAM source edit is required.

## Implement The Contract

Implement `SimulatorBackend` from `open_wam.sdk.simulator`. The adapter owns
environment construction and lifecycle, observation extraction, model-action
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
`SimulatorObservation`, translate model actions, and return
`SimulatorStepResult` from each step.

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
