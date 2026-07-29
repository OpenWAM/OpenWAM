from __future__ import annotations

from pathlib import Path

from open_wam.configs import (
    CalvinDataConfig,
    GenericDataConfig,
    LeRobotConsortiumDataConfig,
    LiberoDataConfig,
    RobotWinDataConfig,
)
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


def test_libero_lingbot_exact_config_uses_all_valid_training_episodes() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
    )

    assert config.data.train_fraction == 1.0
