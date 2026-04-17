from __future__ import annotations

import torch
from torch.utils.data import Dataset

from open_wam.configs import DataConfig

from .action_mapping import (
    action_mapping_is_active,
    apply_action_mapping,
    resolve_action_source_dim,
)
from .latent_contracts import LatentWAMBatch, LatentWAMSample, collate_latent_wam_samples
from .synthetic import build_synthetic_metadata


class SyntheticLatentWindowDataset(Dataset[LatentWAMSample]):
    """Synthetic latent-first dataset for runtime and train-loop smoke tests."""

    def __init__(self, data_config: DataConfig, length: int, task_text: str | None = None) -> None:
        self.data_config = data_config
        self.length = length
        self.task_text = task_text or f"synthetic latent task for {data_config.dataset_name}"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LatentWAMSample:
        action_schema = self.data_config.action_schema
        latent_height = max(1, self.data_config.canonical_height // 16)
        latent_width = max(1, self.data_config.canonical_width // 16)
        if action_mapping_is_active(self.data_config.action_mapping):
            source_dim = resolve_action_source_dim(self.data_config.action_mapping, fallback_dim=action_schema.action_dim)
            source_actions = torch.randn(action_schema.action_horizon, source_dim)
            source_mask = torch.ones_like(source_actions)
            mapped = apply_action_mapping(
                source_actions,
                source_mask,
                self.data_config.action_mapping,
                target_dim=action_schema.action_dim,
            )
            actions = mapped.actions
            action_mask = mapped.action_mask
            metadata = {
                **build_synthetic_metadata(self.data_config, index=index),
                **mapped.metadata,
            }
        else:
            actions = torch.randn(action_schema.action_horizon, action_schema.action_dim)
            action_mask = torch.ones(action_schema.action_horizon, action_schema.action_dim)
            metadata = build_synthetic_metadata(self.data_config, index=index)
        return LatentWAMSample(
            video_latents=torch.randn(48, self.data_config.num_frames, latent_height, latent_width),
            actions=actions,
            action_mask=action_mask,
            state=torch.randn(action_schema.state_horizon, action_schema.state_dim),
            state_mask=torch.ones(action_schema.state_horizon, action_schema.state_dim),
            task_text=self.task_text,
            metadata=metadata,
        )


def build_synthetic_latent_batch(
    data_config: DataConfig,
    batch_size: int,
    task_text: str | None = None,
) -> LatentWAMBatch:
    """Build one synthetic latent batch that respects the configured schemas."""

    dataset = SyntheticLatentWindowDataset(data_config=data_config, length=batch_size, task_text=task_text)
    samples = [dataset[index] for index in range(batch_size)]
    return collate_latent_wam_samples(samples)
