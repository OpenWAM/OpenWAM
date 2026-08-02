"""Physical-episode train/validation splitting for mixed video."""

from __future__ import annotations

from collections import defaultdict
import random

from open_wam.configs import MixedVideoDataConfig

from .mixed_video_catalog_contracts import MixedVideoCatalog, MixedVideoEpisodeRecord


def split_mixed_video_episodes(
    data_config: MixedVideoDataConfig,
    catalog: MixedVideoCatalog,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split physical episodes while keeping timestamp clips together."""

    group_to_episode_keys: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for episode in catalog.episodes:
        group_to_episode_keys[_physical_episode_group_key(episode)].append(
            episode.key
        )
    group_keys = list(group_to_episode_keys)
    rng = random.Random(int(data_config.split_seed))
    rng.shuffle(group_keys)
    train_count = int(len(group_keys) * float(data_config.train_fraction))
    train_count = (
        min(max(train_count, 1), len(group_keys))
        if group_keys
        else 0
    )
    train_group_list = group_keys[:train_count]
    val_group_list = group_keys[train_count:]
    if data_config.max_train_episodes is not None:
        train_group_list = train_group_list[: data_config.max_train_episodes]
    if data_config.max_val_episodes is not None:
        val_group_list = val_group_list[: data_config.max_val_episodes]
    train_keys = [
        episode_key
        for group_key in train_group_list
        for episode_key in group_to_episode_keys[group_key]
    ]
    val_keys = [
        episode_key
        for group_key in val_group_list
        for episode_key in group_to_episode_keys[group_key]
    ]
    if not val_keys and train_group_list:
        val_keys = list(group_to_episode_keys[train_group_list[0]])
    return tuple(sorted(train_keys)), tuple(sorted(val_keys))


def _physical_episode_group_key(
    episode: MixedVideoEpisodeRecord,
) -> tuple[object, ...]:
    path_keys = tuple(
        sorted({stream.clip.path_key for stream in episode.streams})
    )
    return (
        episode.source_id,
        episode.dataset_id,
        episode.episode_index,
        path_keys,
    )


__all__ = ["split_mixed_video_episodes"]
