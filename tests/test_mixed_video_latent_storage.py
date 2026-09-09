"""Stored axis order and optional spatial hints stay inside the data adapter."""

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import (
    BatchingConfig,
    CausalPrefixSuffixBucketConfig,
    MixedVideoDataConfig,
    MixedVideoSourceConfig,
    SampleConstructionConfig,
)
from open_wam.data.mixed_video import MixedVideoLatentWindowDataset
from open_wam.data.mixed_video_catalog_assembly import load_mixed_video_catalog
from open_wam.data.mixed_video_latent_storage import load_mixed_video_latent_tensor
from open_wam.data.preparation.build_manifest import write_csv
from open_wam.training.data_loading import _build_variable_length_loader


@pytest.mark.parametrize("layout", ["tensor", "implicit", "CTHW", "THWC"])
def test_latent_reader_preserves_values_and_normalizes_declared_axes(tmp_path, layout):
    expected = torch.arange(7 * 11 * 2 * 4, dtype=torch.float16).reshape(7, 11, 2, 4)
    stored = expected.permute(1, 2, 3, 0).contiguous() if layout == "THWC" else expected
    payload = {"latent": stored}
    if layout not in ("tensor", "implicit"):
        payload["latent_layout"] = layout
    path = tmp_path / "latent.pt"
    torch.save(stored if layout == "tensor" else payload, path)
    loaded = load_mixed_video_latent_tensor(path, key="latent")
    torch.testing.assert_close(loaded, expected.float(), rtol=0, atol=0)
    assert loaded.is_contiguous()


@pytest.mark.parametrize("layout", ["TCHW", "", None, 123])
def test_latent_reader_rejects_unknown_explicit_layout(tmp_path, layout):
    path = tmp_path / "latent.pt"
    torch.save({"latent": torch.zeros(7, 11, 2, 4), "latent_layout": layout}, path)
    with pytest.raises(ValueError, match="latent_layout"):
        load_mixed_video_latent_tensor(path, key="latent")


@pytest.mark.parametrize(
    "shape_bucketed,has_grid,batch_size",
    [
        (False, False, 1),
        (False, False, 2),
        (False, True, 2),
        (True, True, 1),
        (True, False, 1),
    ],
)
def test_length_bucketing_only_requests_opted_in_spatial_metadata(
    tmp_path, shape_bucketed, has_grid, batch_size
):
    path = tmp_path / "latent.pt"
    torch.save(torch.arange(48 * 3 * 8 * 8).reshape(48, 3, 8, 8).float(), path)
    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        [
            dict(
                episode_index=i,
                clip_id=str(i),
                length_frames=3,
                latent_path=str(path),
                **({"height": 8, "width": 8} if has_grid else {}),
            )
            for i in range(3)
        ],
    )
    data = MixedVideoDataConfig(
        video_sources=(
            MixedVideoSourceConfig(
                source_id="custom",
                manifest_csv=str(manifest),
                source_format="latent",
            ),
        ),
        num_frames=3,
        sample_stride=3,
        num_workers=0,
        target_observation_fps=None,
        sample_construction=SampleConstructionConfig(
            mode="causal_prefix_suffix",
            num_frames=3,
            action_horizon=0,
            state_horizon=0,
            causal_prefix_suffix_buckets=(CausalPrefixSuffixBucketConfig(1, 2),),
        ),
        shape_bucketed_batching=shape_bucketed,
        batching=BatchingConfig(mode="bucket"),
    )
    catalog = load_mixed_video_catalog(data)
    dataset = MixedVideoLatentWindowDataset(
        data,
        catalog=catalog,
        split="val",
        episode_keys=tuple(ep.key for ep in catalog.episodes),
    )
    if shape_bucketed and not has_grid:
        with pytest.raises(ValueError, match="height/width"):
            loader = _build_variable_length_loader(
                SimpleNamespace(data=data),
                dataset,
                None,
                batch_size=batch_size,
                shuffle=False,
                train=False,
            )
            next(iter(loader))
        return
    assert callable(dataset.batching_shape_hint) == shape_bucketed
    loader = _build_variable_length_loader(
        SimpleNamespace(data=data),
        dataset,
        None,
        batch_size=batch_size,
        shuffle=False,
        train=False,
    )
    batches = list(loader)
    assert sum(batch.video_latents.shape[0] for batch in batches) == len(dataset) == 3
    assert batches[-1].video_latents.shape[0] == 1
    assert all(batch.video_latents.shape[1:] == (48, 3, 8, 8) for batch in batches)
