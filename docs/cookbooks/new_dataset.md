# Cookbook: Add A New Dataset

Use this when raw storage, metadata, action/state keys, or camera layout differ
from existing adapters.

## Files To Touch

An external dataset does not require an Open-WAM source change. Create an
installable module containing:

- a `torch.utils.data.Dataset` that returns `WAMSample`
- a train/validation builder
- a `register_open_wam()` hook
- focused adapter tests and a tiny structural fixture

## Contract

Dataset-specific parsing stays inside the adapter. Public outputs stay uniform:

- canonical RGB views keyed by model camera names
- `actions` and optional `action_mask`
- `state` and optional `state_mask`
- `task_text`
- metadata explaining source episode/window identity

Canonical RGB layout construction belongs in the data layer, not in the visual
backbone.

## Minimal Adapter

```python
from torch.utils.data import Dataset

from open_wam.configs import DataConfig
from open_wam.data import WAMSample, register_dataset_adapter


class AcmeDataset(Dataset[WAMSample]):
    def __init__(self, config: DataConfig, *, split: str) -> None:
        self.root = config.local_root
        self.options = config.adapter_options
        self.split = split

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> WAMSample:
        # Decode source rows, apply configured transforms, and return canonical
        # [T, H, W, 3] uint8 views plus aligned actions/state.
        ...


def build_train_val(config: DataConfig):
    return AcmeDataset(config, split="train"), AcmeDataset(config, split="val")


def register_open_wam() -> None:
    register_dataset_adapter(
        "acme_robot",
        raw_builder=build_train_val,
        description="ACME robot demonstrations.",
    )
```

When pre-encoded samples are available, add a `latent_builder` returning
datasets of `LatentWAMSample` under the same `dataset_type`.

For row-oriented robot data, reuse
`open_wam.data.build_row_action_targets` and
`open_wam.data.pack_temporal_sequence`. The adapter supplies
`extract_sequence`, which owns source-key resolution, empty-row handling, and
the decision to truncate. The shared packer builds the float32 padded tensor
and validity mask; the target transform applies the configured raw,
relative-EEF, or absolute-joint representation, normalization, mapping, and
metadata contract. A custom `pack_sequence` callback remains available only
for storage contracts that cannot use the canonical layout.

## Configuration

```yaml
data:
  dataset_name: acme
  dataset_type: acme_robot
  local_root: /datasets/acme
  camera_names: [front]
  latent_camera_names: [front]
  adapter_options:
    rgb_key: observation.images.front
    timestamp_tolerance_us: 100
```

`adapter_options` is passed unchanged to the builder. It can also be adjusted
without editing YAML:

```bash
open-wam-train \
  --extension acme_open_wam \
  --cfg experiment.yaml \
  --set data.adapter_options.timestamp_tolerance_us=200
```

Prefer typed shared fields over `adapter_options` whenever the setting affects
model-facing shapes, action/state semantics, temporal alignment, view layout,
or sampling.

For task-balanced hierarchical sampling, reuse
`open_wam.data.draw_hierarchical_sample_index`. The adapter computes eligible
windows and their probability mass; each task exposes windows with
`mass_within_task`, `start_min`, and `start_max`. The shared primitive owns
stable seed mixing and the task, window, and inclusive-start RNG order.

## Validation

```bash
open-wam-validate-config configs/examples/<dataset_sanity>.yaml
open-wam-sanity \
  --extension acme_open_wam \
  --cfg configs/examples/<dataset_sanity>.yaml \
  --max-batches 1
pytest tests/<dataset_test>.py -q
```

If the dataset needs private paths, tests should skip with an actionable
message unless a public fixture is being used.
