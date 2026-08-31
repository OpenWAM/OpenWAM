from __future__ import annotations

import json
import shutil

import pytest
import torch

from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.contracts import identify_video_latent_space
from open_wam.models.visual_tower.frontend import SharedVideoFrontend
from open_wam.pipelines import require_compatible_video_latent_spaces


def _write_artifact(root, *, config: dict, weights: bytes) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "diffusion_pytorch_model.safetensors").write_bytes(weights)


def _artifact_with_weights(root, *, weights: bytes):
    _write_artifact(root, config={"latent_channels": 48}, weights=weights)
    return root


def _identify(root):
    return identify_video_latent_space(
        root,
        encoder_family="diffusers.AutoencoderKLWan@test",
        encoding_contract="open_wam.wan_vae_latents.test",
    )


def test_identical_mirrored_artifacts_share_one_latent_space_identity(tmp_path) -> None:
    source = tmp_path / "source" / "vae"
    mirror = tmp_path / "mirror" / "vae"
    _write_artifact(
        source,
        config={"scaling_factor": 1.0, "latent_channels": 48},
        weights=b"identical-weights",
    )
    shutil.copytree(source, mirror)

    assert _identify(source) == _identify(mirror)
    assert str(source) not in _identify(source).artifact_sha256
    assert require_compatible_video_latent_spaces(
        _identify(source), _identify(mirror)
    )["artifact_sha256"] == _identify(source).artifact_sha256


def test_config_formatting_does_not_change_latent_space_identity(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    config = {"latent_channels": 48, "scaling_factor": 1.0}
    _write_artifact(left, config=config, weights=b"weights")
    _write_artifact(right, config=config, weights=b"weights")
    (right / "config.json").write_text(
        '{"scaling_factor":1.0,"latent_channels":48}\n',
        encoding="utf-8",
    )

    assert _identify(left) == _identify(right)


def test_weight_mutation_at_the_same_path_changes_identity(tmp_path) -> None:
    artifact = tmp_path / "vae"
    _write_artifact(
        artifact,
        config={"latent_channels": 48},
        weights=b"first-weights",
    )
    before = _identify(artifact)

    (artifact / "diffusion_pytorch_model.safetensors").write_bytes(
        b"other-weights"
    )
    after = _identify(artifact)

    assert before != after
    assert before.weights_sha256 != after.weights_sha256


def test_equal_geometry_with_different_weights_is_not_the_same_latent_space(
    tmp_path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    config = {"latent_channels": 48, "spatial_stride": 16}
    _write_artifact(left, config=config, weights=b"left")
    _write_artifact(right, config=config, weights=b"right")

    left_identity = _identify(left)
    right_identity = _identify(right)

    assert left_identity.config_sha256 == right_identity.config_sha256
    assert left_identity != right_identity
    with pytest.raises(ValueError, match="different latent spaces"):
        require_compatible_video_latent_spaces(left_identity, right_identity)


def test_composition_rejects_unidentified_latent_spaces() -> None:
    with pytest.raises(ValueError, match="artifact-backed latent-space identity"):
        require_compatible_video_latent_spaces(None, None)


def test_visual_frontend_propagates_its_latent_space_identity(tmp_path) -> None:
    identity = _identify(
        _artifact_with_weights(tmp_path / "vae", weights=b"frontend-weights")
    )
    frontend = SharedVideoFrontend(
        SharedVideoTransformerConfig(
            input_channels=3,
            latent_channels=4,
            latent_stride=1,
            patch_size_t=1,
            patch_size_h=1,
            patch_size_w=1,
            hidden_size=8,
            num_heads=1,
        )
    )
    frontend.reference_assets.latent_space_identity = identity

    output = frontend.from_video_latents(torch.randn(1, 4, 2, 2, 2))

    assert output.latent_space_identity == identity
