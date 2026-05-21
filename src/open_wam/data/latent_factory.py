from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import Dataset

from open_wam.configs import DataConfig

from .latent_contracts import LatentWAMSample
from .lerobot_v2_latent import build_local_lerobot_latent_train_val_datasets
from .mixed_video import build_mixed_video_latent_train_val_datasets
from .latent_synthetic import SyntheticLatentWindowDataset


LatentDatasetPairBuilder = Callable[[DataConfig], tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]]

_LATENT_DATASET_BUILDERS: dict[str, LatentDatasetPairBuilder] = {}


def register_latent_dataset_builder(dataset_type: str, builder: LatentDatasetPairBuilder) -> None:
    """Register one train/val latent dataset builder for a source type."""

    _LATENT_DATASET_BUILDERS[dataset_type] = builder


def build_train_val_latent_datasets(
    data_config: DataConfig,
) -> tuple[Dataset[LatentWAMSample], Dataset[LatentWAMSample]]:
    """Build train/val latent datasets from the config-defined source type."""

    try:
        builder = _LATENT_DATASET_BUILDERS[data_config.dataset_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_LATENT_DATASET_BUILDERS))
        raise ValueError(
            f"Unsupported latent dataset_type '{data_config.dataset_type}'. "
            f"Registered latent dataset types: {supported}"
        ) from exc
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
