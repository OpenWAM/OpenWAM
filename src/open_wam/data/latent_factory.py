from __future__ import annotations

from torch.utils.data import Dataset

from open_wam.configs import DataConfig

from .latent_contracts import LatentWAMSample
from .latent_synthetic import SyntheticLatentWindowDataset
from .lerobot_v2_latent import build_local_lerobot_latent_train_val_datasets
from .mixed_video import build_mixed_video_latent_train_val_datasets
from .registries import (
    DATASET_ADAPTERS,
    register_latent_dataset_builder,
)


def build_train_val_latent_datasets(
    data_config: DataConfig,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    """Build train/val latent datasets from the config-defined source type."""

    builder = DATASET_ADAPTERS.require_latent_builder(data_config.dataset_type)
    return builder(data_config)


def _build_synthetic_latent_datasets(
    data_config: DataConfig,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    return (
        SyntheticLatentWindowDataset(data_config, length=8),
        SyntheticLatentWindowDataset(data_config, length=2),
    )


register_latent_dataset_builder("synthetic_latent", _build_synthetic_latent_datasets)
register_latent_dataset_builder("synthetic_robotwin", _build_synthetic_latent_datasets)
register_latent_dataset_builder("synthetic_multiview", _build_synthetic_latent_datasets)
register_latent_dataset_builder("lerobot_v2_latent_local", build_local_lerobot_latent_train_val_datasets)
register_latent_dataset_builder("mixed_video", build_mixed_video_latent_train_val_datasets)
