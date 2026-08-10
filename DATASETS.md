# Open-WAM Datasets

Open-WAM keeps dataset-specific storage and parsing behind adapters selected by
`data.dataset_type`. Every adapter produces the same model-facing contracts:

- canonical camera views or pre-encoded view latents
- aligned actions and optional action masks
- aligned state/proprioception and optional masks
- task text
- source and temporal metadata

The maintained source matrix, required local artifacts, and benchmark setup
live in [Benchmarks And Data](docs/benchmarks.md). This file is intentionally a
short entry point so supported adapters are not confused with prospective
datasets or historical architecture layers.

## Start Here

- Use `synthetic_multiview` and
  `configs/examples/public_tiny_synthetic_contract.yaml` for the data-free CPU
  lifecycle.
- Use [the dataset cookbook](docs/cookbooks/new_dataset.md) to register an
  application-owned adapter without modifying the trainer or visual backbone.
- Use `configs/local_paths.sample.yaml` for machine-local roots. Populate only
  keys referenced by the selected experiment config.
- Keep canonical RGB layout and action/state transformation in the adapter;
  policy variants consume the uniform contract.

Dataset and simulator availability are separate. Offline training and
evaluation need a dataset adapter; closed-loop evaluation additionally needs a
registered simulator backend and its upstream environment.
