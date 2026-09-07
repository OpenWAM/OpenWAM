# Cookbook: Reproduce A Result

Use this when turning an internal run into a public or collaborator-facing
reproduction recipe.

## Required Inputs

- experiment config
- eval config
- artifact manifest entry
- artifact or checkpoint card
- dataset or local path registry entry
- expected metrics
- exact validation command
- hardware/resource requirements

## Steps

1. Add or update the config under `configs/experiments/`.
2. Add or update an eval wrapper under `configs/evals/`.
3. Add an artifact manifest entry in `configs/artifacts.sample.yaml` if the
   artifact is public or layout-documented.
4. Add a card under `docs/cards/`.
5. Run static validation:

```bash
openwam-validate-config configs/experiments/<config>.yaml configs/evals/<eval>.yaml
```

6. Run the smallest honest runtime check that matches the claim:
   CPU batch eval, GPU eval, or real simulator rollout.
7. Record metrics, commit, command, and limitations in the card.

## Rule

Do not describe a private checkpoint as publicly reproducible. If hosting,
checksum, or license is missing, state that the card documents layout only.
