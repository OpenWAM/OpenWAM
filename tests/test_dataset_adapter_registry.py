from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from torch.utils.data import Dataset

from open_wam.configs import GenericDataConfig
from open_wam.data import (
    DATASET_ADAPTERS,
    build_train_val_datasets,
    build_train_val_latent_datasets,
    register_dataset_adapter,
    register_dataset_builder,
)
from open_wam.utils import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class _MarkerDataset(Dataset):
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        raise IndexError(index)


def test_one_adapter_can_expose_raw_and_latent_builders() -> None:
    dataset_type = "test_raw_and_latent_adapter"
    observed_options = []

    def build_raw(config):
        observed_options.append(("raw", config.adapter_options))
        return _MarkerDataset("raw_train"), _MarkerDataset("raw_val")

    def build_latent(config):
        observed_options.append(("latent", config.adapter_options))
        return _MarkerDataset("latent_train"), _MarkerDataset("latent_val")

    register_dataset_adapter(
        dataset_type,
        raw_builder=build_raw,
        description="Test adapter.",
    )
    register_dataset_adapter(dataset_type, latent_builder=build_latent)
    config = GenericDataConfig(
        dataset_name="test",
        dataset_type=dataset_type,
        adapter_options={"row_key": "observation.rgb"},
    )

    raw_train, raw_val = build_train_val_datasets(config)
    latent_train, latent_val = build_train_val_latent_datasets(config)

    assert (raw_train.marker, raw_val.marker) == ("raw_train", "raw_val")
    assert (latent_train.marker, latent_val.marker) == ("latent_train", "latent_val")
    assert observed_options == [
        ("raw", {"row_key": "observation.rgb"}),
        ("latent", {"row_key": "observation.rgb"}),
    ]
    spec = DATASET_ADAPTERS.require(dataset_type)
    assert spec.raw_builder is build_raw
    assert spec.latent_builder is build_latent
    assert spec.description == "Test adapter."


def test_dataset_adapter_rejects_accidental_builder_replacement() -> None:
    dataset_type = "test_duplicate_adapter"

    def first_builder(config):
        del config
        return _MarkerDataset("first_train"), _MarkerDataset("first_val")

    def second_builder(config):
        del config
        return _MarkerDataset("second_train"), _MarkerDataset("second_val")

    register_dataset_builder(dataset_type, first_builder)

    with pytest.raises(ValueError, match="already has a raw builder"):
        register_dataset_builder(dataset_type, second_builder)

    register_dataset_builder(dataset_type, second_builder, replace=True)
    config = GenericDataConfig(dataset_name="test", dataset_type=dataset_type)
    train, val = build_train_val_datasets(config)
    assert (train.marker, val.marker) == ("second_train", "second_val")


def test_dataset_adapter_errors_report_only_compatible_capabilities() -> None:
    dataset_type = "test_latent_only_adapter"

    def build_latent(config):
        del config
        return _MarkerDataset("latent_train"), _MarkerDataset("latent_val")

    register_dataset_adapter(dataset_type, latent_builder=build_latent)

    with pytest.raises(ValueError, match="Unsupported raw dataset_type") as exc_info:
        build_train_val_datasets(
            GenericDataConfig(dataset_name="test", dataset_type=dataset_type)
        )

    assert dataset_type not in str(exc_info.value).split("Registered raw dataset types:", 1)[1]


def test_config_loader_preserves_adapter_options(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["data"]["dataset_name"] = "custom"
    raw["data"]["dataset_type"] = "acme_robot_dataset"
    raw["data"]["adapter_options"] = {
        "rgb_key": "observation.images.front",
        "nested": {"timestamp_tolerance_us": 100},
    }
    config_path = tmp_path / "custom_dataset.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(config_path)

    assert config.data.adapter_options == raw["data"]["adapter_options"]


def test_data_config_rejects_non_mapping_adapter_options() -> None:
    with pytest.raises(TypeError, match="data.adapter_options"):
        GenericDataConfig(adapter_options=["not", "a", "mapping"])  # type: ignore[arg-type]
