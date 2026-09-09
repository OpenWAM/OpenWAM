from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from open_wam.configs import MixedVideoDataConfig, MixedVideoSourceConfig, MixedVideoSourceFormat
from open_wam.data.mixed_video_catalog_assembly import load_mixed_video_catalog
from open_wam.data.mixed_video_catalog_contracts import MixedVideoCatalog
from open_wam.data.mixed_video_catalog_split import split_mixed_video_episodes


def _catalog(tmp_path: Path, count: int = 30, *, variants: bool = False):
    rows = []
    for i in range(count):
        for variant in (("single", "multi_view", "clip") if variants else ("single",)):
            rows.append({
                "dataset_id": "official/repository", "episode_index": i,
                "clip_id": variant, "stream_index": 0, "stream_key": "front",
                "target_slot_key": "observation.images.slot0",
                "latent_path": f"{variant}/episode_{i}.pt", "length_frames": 5,
                "width": 4, "height": 4, "physical_episode_key": f"official/repository/episode_{i}",
            })
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = MixedVideoDataConfig(
        video_sources=(MixedVideoSourceConfig(source_id="source", manifest_csv=str(path), source_format=MixedVideoSourceFormat.LATENT),),
        camera_names=("observation.images.slot0",), latent_camera_names=("observation.images.slot0",),
        train_fraction=0.6, split_seed=19,
    )
    return config, load_mixed_video_catalog(config)


def test_single_multi_view_and_timestamp_variants_share_one_physical_split(tmp_path):
    config, catalog = _catalog(tmp_path, variants=True)
    train, val = map(set, split_mixed_video_episodes(config, catalog))
    assert train and val and train.isdisjoint(val)
    for i in range(30):
        siblings = {ep.key for ep in catalog.episodes if ep.episode_index == i}
        assert len(siblings) == 3
        assert siblings <= train or siblings <= val
    assert {ep.physical_episode_key for ep in catalog.episodes} == {
        f"official/repository/episode_{i}" for i in range(30)
    }


def test_append_reorder_relocation_and_new_view_do_not_change_existing_split(tmp_path):
    config, catalog = _catalog(tmp_path)
    initial = split_mixed_video_episodes(config, catalog)
    _, expanded = _catalog(tmp_path, count=80, variants=True)
    relocated = replace(expanded, episodes=tuple(
        replace(ep, streams=tuple(replace(stream, latent_path=Path("/new/cache") / stream.latent_path.name) for stream in ep.streams))
        for ep in reversed(expanded.episodes)
    ))
    later = split_mixed_video_episodes(config, relocated)
    originals = {ep.key for ep in catalog.episodes}
    assert tuple(set(keys) & originals for keys in later) == tuple(map(set, initial))


def test_distinct_repositories_reusing_episode_numbers_remain_distinct(tmp_path):
    config, catalog = _catalog(tmp_path, count=1)
    first = catalog.episodes[0]
    second = replace(first, key="other", dataset_id="other/repository", streams=(
        replace(first.streams[0], physical_episode_key="other/repository/episode_0"),
    ))
    from open_wam.data.mixed_video_catalog_split import _physical_episode_group_key
    assert _physical_episode_group_key(first) != _physical_episode_group_key(second)
    # Identities are global: source aliases and derived paths do not change them.
    alias = replace(first, key="alias", source_id="second-source", dataset_id="derived", streams=(
        replace(first.streams[0], latent_path=Path("different.pt")),
    ))
    assert _physical_episode_group_key(first) == _physical_episode_group_key(alias)


def test_explicit_split_never_copies_training_to_validation(tmp_path):
    config, catalog = _catalog(tmp_path, count=1)
    train, val = split_mixed_video_episodes(replace(config, train_fraction=1.0), catalog)
    assert train == (catalog.episodes[0].key,)
    assert val == ()


def test_legacy_fallback_does_not_include_explicit_identity(tmp_path):
    config, catalog = _catalog(tmp_path, count=2)
    legacy = replace(catalog.episodes[0], streams=(replace(catalog.episodes[0].streams[0], physical_episode_key=None),))
    mixed = MixedVideoCatalog(episodes=(legacy, catalog.episodes[1]))
    train, val = split_mixed_video_episodes(replace(config, train_fraction=1.0), mixed)
    assert set(train) == {ep.key for ep in mixed.episodes}
    assert val == (legacy.key,)


@pytest.mark.parametrize("second_id", [None, "other/episode"])
def test_conflicting_or_partial_identity_inside_one_episode_is_rejected(tmp_path, second_id):
    config, catalog = _catalog(tmp_path, count=1)
    episode = catalog.episodes[0]
    bad = replace(episode, streams=episode.streams + (replace(episode.streams[0], physical_episode_key=second_id),))
    with pytest.raises(ValueError, match="physical_episode_key"):
        split_mixed_video_episodes(config, MixedVideoCatalog(episodes=(bad,)))


def test_episode_caps_retain_whole_physical_groups(tmp_path):
    config, catalog = _catalog(tmp_path, variants=True)
    train, val = split_mixed_video_episodes(replace(config, max_train_episodes=2, max_val_episodes=1), catalog)
    assert len(train) == 6 and len(val) == 3
    assert set(train).isdisjoint(val)
