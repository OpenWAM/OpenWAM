# Running Experiments

Open-WAM exposes package-owned CLIs for the supported train, eval, sanity, and
rollout pathways. Legacy root scripts remain compatibility wrappers, but new
documentation should prefer `open-wam-*` commands.

## Static Validation First

Validate configs before launching compute:

```bash
open-wam-validate-config configs/experiments/<experiment>.yaml
open-wam-validate-config configs/evals/<eval>.yaml
```

The static validator does not import model code. It is meant to catch missing
sections, enum typos, bad path placeholders, and incompatible public config
choices before GPU time is allocated.

## Training

Training uses the same generic stack across method families:

```bash
open-wam-train --cfg configs/experiments/<experiment>.yaml
```

Before real training, check:

- local dataset paths are configured through `configs/local_paths.yaml`
- checkpoint/artifact aliases are present when required
- the selected optional extras are installed
- WandB or local tracking policy is documented for the run
- the config has an experiment card if it is intended to be reproducible

## Offline Evaluation

Use eval configs for batch-level metrics:

```bash
open-wam-eval \
  --cfg configs/evals/<eval>.yaml \
  --device cpu \
  --max-batches 1
```

For real policies, switch the device and batch limits according to the
available compute. Result files should use the versioned result envelope
described in [Reproducibility](reproducibility.md).

## Sanity Checks

Use sanity configs to check the complete loader, model, and decoder pathway
without claiming benchmark performance:

```bash
open-wam-sanity \
  --cfg configs/examples/<example>.yaml \
  --device cpu \
  --max-batches 1 \
  --rollout-steps 1
```

GPU or simulator sanity checks should be marked and documented as resource
gated. They should skip clearly when the required resource is missing.

## Realtime And Simulator Rollouts

Closed-loop rollouts must avoid future information. The simulator should step
forward in wall-clock-aware time, and the policy should only consume
observations available at the current control step.

Use the shared rollout command when a simulator adapter is configured:

```bash
open-wam-sim-rollout \
  --cfg configs/examples/<example>.yaml \
  --target-action-hz 10
```

Report:

- target action Hz
- achieved non-fallback action Hz
- fallback action count
- rollout success/failure
- output video path
- exact checkpoint/config used

Do not compare realtime results without documenting planner mode, diffusion
step count, fallback policy, and simulator task/episode identity.
