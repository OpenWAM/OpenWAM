from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

from open_wam.configs import (
    CalvinDataConfig,
    GenericDataConfig,
    LeRobotConsortiumDataConfig,
    LiberoDataConfig,
    RobotWinDataConfig,
)
from open_wam.configs import data as data_config_facade
from open_wam.configs import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_data_config_defaults_use_all_valid_training_episodes() -> None:
    configs = (
        GenericDataConfig(),
        RobotWinDataConfig(),
        LiberoDataConfig(),
        CalvinDataConfig(),
        LeRobotConsortiumDataConfig(),
    )

    assert {config.train_fraction for config in configs} == {1.0}


def test_libero_parallel_stream_config_uses_all_valid_training_episodes() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )

    assert config.data.train_fraction == 1.0


def test_public_data_config_type_hints_resolve_from_canonical_owners() -> None:
    for name in data_config_facade.__all__:
        config_type = getattr(data_config_facade, name)
        if not isinstance(config_type, type):
            continue
        assert get_type_hints(config_type)
