"""Preparation and inspection use declared formats, not dataset or clip names."""

import json
from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import MixedVideoDataConfig, MixedVideoSourceConfig
from open_wam.data.mixed_video_catalog_assembly import load_mixed_video_catalog
from open_wam.data.preparation import build_manifest, prepare_dataset


def _single_rows(tmp_path, *, source, raw_type, evidence=None):
    root = tmp_path / "latents"
    path = root / "multi_view/latents/chunk-000/front/episode_000000_0_18.pth"
    path.parent.mkdir(parents=True)
    torch.save(
        dict(
            latent=torch.zeros(3, 8, 8, 48, dtype=torch.float16),
            latent_layout="THWC",
            latents_normalized=True,
            video_num_frames=9,
            video_height=128,
            video_width=128,
            fps=15.0,
            ori_fps=30.0,
            **(evidence or {}),
        ),
        path,
    )
    episode = prepare_dataset.make_row(
        source,
        "multi_view",
        0,
        ["front"],
        {"front": 18},
        dict(type=raw_type, fps=30, uri="s3://archive/h5_franka_3rgb/demo.tar.gz"),
        "move left",
        "native",
    )
    index = tmp_path / "episodes.jsonl"
    index.write_text(json.dumps(episode) + "\n")
    return list(
        build_manifest.single_rows(
            SimpleNamespace(
                source=source,
                episodes=index,
                latent_root=root,
            )
        )
    )


@pytest.mark.parametrize("source", ["VPT-06", "custom_robomind"])
@pytest.mark.parametrize(
    "raw_type", ["robomind_official_archive", "robomind_failure_hdf5"]
)
def test_rgb_evidence_required_independently_of_source_label(
    tmp_path, source, raw_type
):
    with pytest.raises(ValueError, match="RGB"):
        _single_rows(tmp_path, source=source, raw_type=raw_type)


@pytest.mark.parametrize("source", ["VPT-06", "custom_robomind"])
def test_certified_failure_payload_keeps_its_source_label(tmp_path, source):
    rows = _single_rows(
        tmp_path,
        source=source,
        raw_type="robomind_failure_hdf5",
        evidence=dict(
            color_policy="robomind_failure_standard_jpeg_v1", source_rgb_sha256="a" * 64
        ),
    )
    assert rows[0]["source_id"] == source


@pytest.mark.parametrize("raw_type", [None, "", "unknown_hdf5"])
def test_admission_requires_known_raw_format(tmp_path, raw_type):
    with pytest.raises(ValueError, match="raw_source.type"):
        _single_rows(tmp_path, source="custom_source", raw_type=raw_type)


@pytest.mark.parametrize("source", ["VPT-06", "custom_source"])
@pytest.mark.parametrize("raw_type", ["lerobot", "egoexo_aligned"])
def test_non_robomind_formats_do_not_inherit_rgb_policy_from_label(
    tmp_path, source, raw_type
):
    assert len(_single_rows(tmp_path, source=source, raw_type=raw_type)) == 1


def test_inspection_filters_manifest_augmentation_not_clip_name(tmp_path):
    from open_wam.cli.video_pretraining_infer import _select_sample

    single = _single_rows(tmp_path, source="custom_source", raw_type="lerobot")[0]
    assert single["clip_id"].startswith("multi_view/")
    rows = [
        single,
        dict(single, clip_id="arbitrary_identifier", augmentation="multi_view"),
        dict(single, clip_id="multi_view/undeclared", augmentation=""),
    ]
    manifest = tmp_path / "manifest.csv"
    build_manifest.write_csv(manifest, rows)
    cfg = MixedVideoDataConfig(
        video_sources=(
            MixedVideoSourceConfig(
                source_id="custom_source",
                manifest_csv=str(manifest),
                source_format="latent",
            ),
        )
    )
    catalog = load_mixed_video_catalog(cfg)
    episodes = {ep.clip_id: ep for ep in catalog.episodes}
    assert [episodes[row["clip_id"]].streams[0].augmentation for row in rows] == [
        "single_view",
        "multi_view",
        None,
    ]

    class Dataset:
        episode_records = {ep.key: ep for ep in catalog.episodes}
        sample_index = [
            SimpleNamespace(episode_key=episodes[row["clip_id"]].key) for row in rows
        ]

        def __len__(self):
            return len(self.sample_index)

        def __getitem__(self, index):
            return rows[index]["clip_id"]

    dataset = Dataset()
    assert (
        _select_sample(dataset, source_id=None, sample_index=0, multi_view_only=True)
        == "arbitrary_identifier"
    )
    assert (
        _select_sample(dataset, source_id=None, sample_index=0, multi_view_only=False)
        == single["clip_id"]
    )
    assert (
        _select_sample(
            dataset, source_id="custom_source", sample_index=2, multi_view_only=False
        )
        == "multi_view/undeclared"
    )
    with pytest.raises(IndexError, match="No matching sample"):
        _select_sample(dataset, source_id="other", sample_index=0, multi_view_only=True)
    with pytest.raises(IndexError, match="No matching sample"):
        _select_sample(dataset, source_id=None, sample_index=1, multi_view_only=True)
