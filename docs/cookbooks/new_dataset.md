# Cookbook: Add A New Dataset

Use this when raw storage, metadata, action/state keys, or camera layout differ
from existing adapters.

## Files To Touch

An external dataset does not require an OpenWAM source change. Create an
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

from open_wam.sdk.config import DataConfig
from open_wam.sdk.data import (
    DatasetArtifactKind,
    DatasetArtifactRequirement,
    WAMSample,
    register_dataset_adapter,
)


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


def resolve_artifacts(config: DataConfig):
    return (
        DatasetArtifactRequirement(
            name="dataset root",
            path=config.local_root,
            kind=DatasetArtifactKind.DIRECTORY,
            required=True,
            config_path="data.local_root",
            purpose="the ACME adapter reads episodes from this directory",
        ),
    )


def register_open_wam() -> None:
    register_dataset_adapter(
        "acme_robot",
        raw_builder=build_train_val,
        artifact_resolver=resolve_artifacts,
        description="ACME robot demonstrations.",
    )
```

The resolver is adapter-owned and must remain tensor-free. It runs before model
allocation, may return any sequence of requirements, and should declare only
filesystem dependencies actually consumed by the adapter. Keep optional files
`required=False`; their availability is recorded without blocking startup.

When pre-encoded samples are available, add a `latent_builder` returning
datasets of `LatentWAMSample` under the same `dataset_type`.

Build `LatentWAMSample` directly for a new storage contract. Existing in-tree
adapters also use the helpers below, but these `open_wam.data` implementation
APIs are provisional and are not part of the compatibility-managed extension
SDK. External packages should own equivalent storage logic or request that a
generally useful contract be promoted through `open_wam.sdk.data`.

| Provisional helper | In-tree role |
| --- | --- |
| `LocalLatentSampleSourceLoader` and `LocalLatentRepository` | Read local LeRobot episode rows and per-camera latents. |
| `LocalLatentSegmentAssembler` | Align a selected latent range with action, state, and proprio tensors. |
| `LatentCausalPrefixSuffixWindowPlanner` | Plan split-aware causal prefix/suffix windows. |
| `LocalLatentHierarchicalSegmentPlan` | Resolve hierarchical fixed-segment sampling geometry. |
| `LocalLatentTrainValWindowPlanner` | Resolve replay-aware train and validation membership. |
| `build_row_action_targets` and `pack_temporal_sequence` | Build canonical row-oriented targets and masks. |

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
openwam-train \
  --extension acme_open_wam \
  --cfg experiment.yaml \
  --set data.adapter_options.timestamp_tolerance_us=200
```

Prefer typed shared fields over `adapter_options` whenever the setting affects
model-facing shapes, action/state semantics, temporal alignment, view layout,
or sampling.

In-tree adapters can use the provisional
`open_wam.data.draw_hierarchical_sample_index`. The adapter computes eligible
windows and their probability mass; each task exposes windows with
`mass_within_task`, `start_min`, and `start_max`. The shared primitive owns
stable seed mixing and the task, window, and inclusive-start RNG order.

## Validation

```bash
openwam-validate-config configs/examples/<dataset_sanity>.yaml
openwam-sanity \
  --extension acme_open_wam \
  --cfg configs/examples/<dataset_sanity>.yaml \
  --max-batches 1
pytest tests/<dataset_test>.py -q
```

If the dataset needs private paths, tests should skip with an actionable
message unless a public fixture is being used.
