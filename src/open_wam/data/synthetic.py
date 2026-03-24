from __future__ import annotations

import torch
from torch.utils.data import Dataset

from open_wam.configs import DataConfig

from .contracts import WAMBatch, WAMSample, collate_wam_samples


class SyntheticWindowDataset(Dataset[WAMSample]):
    """Config-driven synthetic dataset for smoke tests and dry runs.

    The goal is not to mimic any single benchmark. The goal is to exercise the
    full data contract using whatever view layout, action schema, and state
    schema are declared in the experiment config.

    This class is useful whenever collaborators want to test a new camera
    layout, action schema, or head contract before a real adapter exists.
    """

    def __init__(self, data_config: DataConfig, length: int, task_text: str | None = None) -> None:
        self.data_config = data_config
        self.length = length
        self.task_text = task_text or f"synthetic task for {data_config.dataset_name}"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> WAMSample:
        action_schema = self.data_config.action_schema
        return WAMSample(
            views=build_synthetic_views(self.data_config, batch_size=None),
            actions=torch.randn(action_schema.action_horizon, action_schema.action_dim),
            action_mask=torch.ones(action_schema.action_horizon, action_schema.action_dim),
            state=torch.randn(action_schema.state_horizon, action_schema.state_dim),
            state_mask=torch.ones(action_schema.state_horizon, action_schema.state_dim),
            task_text=self.task_text,
            metadata={"sample_index": index, "dataset_name": self.data_config.dataset_name},
        )


def build_synthetic_views(
    data_config: DataConfig,
    batch_size: int | None,
    num_frames: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build random RGB views from the declared config layout.

    If `batch_size` is `None`, this returns per-sample tensors `[T, H, W, 3]`.
    Otherwise it returns batched tensors `[B, T, H, W, 3]`.
    """

    resolved_num_frames = num_frames or data_config.num_frames
    views: dict[str, torch.Tensor] = {}
    for view in data_config.view_layout:
        shape = (resolved_num_frames, view.height, view.width, 3)
        if batch_size is not None:
            shape = (batch_size, *shape)
        views[view.source_name] = torch.randint(0, 255, shape, dtype=torch.uint8)
    return views


def build_synthetic_batch(
    data_config: DataConfig,
    batch_size: int,
    task_text: str | None = None,
) -> WAMBatch:
    """Build one synthetic batch that respects the configured schemas."""

    dataset = SyntheticWindowDataset(data_config=data_config, length=batch_size, task_text=task_text)
    samples = [dataset[index] for index in range(batch_size)]
    return collate_wam_samples(samples)
