from __future__ import annotations

from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.context_encoders import (
    GeneralistModeContextEncoder,
    ProprioContextEncoder,
    ProprioHiddenContextEncoder,
)
from open_wam.models.visual_tower import replica_core as replica_core_module
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


def _small_core() -> SharedVideoTransformerCore:
    config = SharedVideoTransformerConfig(
        latent_channels=4,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        text_dim=8,
        freq_dim=4,
    )
    return SharedVideoTransformerCore(config=config, action_dim=2)


def test_context_encoder_compatibility_exports_preserve_identity() -> None:
    assert replica_core_module.ProprioContextEncoder is ProprioContextEncoder
    assert replica_core_module.ProprioHiddenContextEncoder is ProprioHiddenContextEncoder
    assert replica_core_module.GeneralistModeContextEncoder is GeneralistModeContextEncoder


def test_context_encoder_attachment_names_preserve_checkpoint_keys() -> None:
    core = _small_core()
    core.configure_proprio_context_encoder(enabled=True, state_dim=3)
    core.configure_proprio_hidden_context_encoder(enabled=True, state_dim=3)
    core.configure_generalist_mode_context_encoder(enabled=True)

    keys = set(core.state_dict())

    assert {
        "proprio_context_encoder.proj.weight",
        "proprio_context_encoder.proj.bias",
        "proprio_hidden_context_encoder.proj.weight",
        "proprio_hidden_context_encoder.proj.bias",
        "generalist_mode_context_encoder.embedding.weight",
    } <= keys
