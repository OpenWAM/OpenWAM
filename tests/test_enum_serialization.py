from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from open_wam.configs import JointDenoiseTrainingMode
from open_wam.configs.enums import serialize_enum_values
from open_wam.utils import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_serialize_enum_values_converts_mapping_keys_for_yaml() -> None:
    payload = {
        JointDenoiseTrainingMode.JOINT: {
            JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION: 0.25,
        },
    }

    serialized = serialize_enum_values(payload)

    assert serialized == {"joint": {"video_conditioned_action": 0.25}}
    yaml.safe_dump(serialized, sort_keys=False)


def test_generalist_joint_denoising_config_serializes_for_checkpoint_yaml() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
    )

    serialized = serialize_enum_values(asdict(config))

    yaml.safe_dump(serialized, sort_keys=False)
