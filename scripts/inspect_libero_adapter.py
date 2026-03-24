from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import LeRobotV2WindowDataset, build_lerobot_train_val_episode_split, collate_wam_samples, load_lerobot_v2_metadata
from open_wam.utils import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        "--config",
        dest="config",
        type=str,
        default="configs/experiments/contract_only_libero.yaml",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    metadata = load_lerobot_v2_metadata(config.data.repo_id, cache_dir=config.data.cache_dir)
    print("repo_id", metadata.repo_id)
    print("codebase_version", metadata.codebase_version)
    print("fps", metadata.fps)
    print("total_episodes", metadata.total_episodes)
    print("feature_keys", sorted(metadata.features.keys()))

    train_episodes, val_episodes = build_lerobot_train_val_episode_split(config.data)
    print("train_episodes", len(train_episodes))
    print("val_episodes", len(val_episodes))

    dataset = LeRobotV2WindowDataset(config.data, episodes=train_episodes[:2])
    print("dataset_windows", len(dataset))

    first = dataset[0]
    print("sample.views.image", tuple(first.views["image"].shape))
    print("sample.views.wrist_image", tuple(first.views["wrist_image"].shape))
    print("sample.actions", tuple(first.actions.shape))
    print("sample.actions[0]", first.actions[0].tolist())
    print("sample.state", tuple(first.state.shape) if first.state is not None else None)
    print("sample.task_text", first.task_text)
    print("sample.metadata", first.metadata)

    batch = collate_wam_samples([dataset[0], dataset[1]])
    print("batch.views.image", tuple(batch.views["image"].shape))
    print("batch.views.wrist_image", tuple(batch.views["wrist_image"].shape))
    print("batch.actions", tuple(batch.actions.shape))
    print("batch.state", tuple(batch.state.shape) if batch.state is not None else None)


if __name__ == "__main__":
    main()
