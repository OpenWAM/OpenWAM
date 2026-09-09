"""Physical-episode train/validation splitting for mixed video."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random

from open_wam.configs import MixedVideoDataConfig

from .mixed_video_catalog_contracts import MixedVideoCatalog, MixedVideoEpisodeRecord


def split_mixed_video_episodes(
    data_config: MixedVideoDataConfig,
    catalog: MixedVideoCatalog,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep physical episodes together and preserve explicit-ID assignments.

    New manifests use a globally namespaced physical_episode_key shared by all
    their camera, multi view, and timestamp variants. Its hash depends only on the split
    seed and identity, so adding episodes or moving data does not reshuffle it.
    Legacy manifests retain their historical shuffle and small-dataset fallback.
    """

    group_to_episode_keys: dict[tuple[object, ...], list[str]] = defaultdict(list)
    for episode in catalog.episodes:
        group_to_episode_keys[_physical_episode_group_key(episode)].append(
            episode.key
        )
    explicit_keys = sorted(
        key for key in group_to_episode_keys if len(key) == 2
    )
    group_keys = [key for key in group_to_episode_keys if len(key) != 2]
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
    threshold = int(float(data_config.train_fraction) * (1 << 64))
    for key in explicit_keys:
        encoded = json.dumps(
            [int(data_config.split_seed), key[1]], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        score = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
        (train_group_list if score < threshold else val_group_list).append(key)
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
    legacy_train_groups = [key for key in train_group_list if len(key) != 2]
    if not val_keys and legacy_train_groups:
        val_keys = list(group_to_episode_keys[legacy_train_groups[0]])
    return tuple(sorted(train_keys)), tuple(sorted(val_keys))


def _physical_episode_group_key(
    episode: MixedVideoEpisodeRecord,
) -> tuple[object, ...]:
    physical_key = episode.physical_episode_key
    if physical_key is not None:
        return ("physical_episode_key", physical_key)
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
