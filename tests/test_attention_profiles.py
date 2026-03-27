from __future__ import annotations

import torch

from open_wam.models.common.attention_profiles import build_chunked_temporal_exact_attention_profile
from open_wam.models.policy_variants.parallel_stream.reference_runtime import get_mesh_id
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.visual_tower.replica_core import SharedVideoTransformerCore


def test_build_chunked_temporal_exact_attention_profile_dense_masks() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 2, 2, 2, 2),
        action_shape=(1, 3, 2, 1, 1),
        padded_length=2,
        chunk_size=1,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=4,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
    )

    assert profile.self_attention_mask is not None
    assert profile.cross_attention_mask is not None
    assert profile.self_attention_mask.shape == (22, 22)
    assert profile.cross_attention_mask.shape == (22, 4)

    # Query = noisy latent token on frame 0, KV = clean latent token on frame 0.
    # Noise cannot see same-frame clean tokens.
    assert bool(profile.self_attention_mask[0, 8].item()) is False
    # Query = noisy latent token on frame 1, KV = clean latent token on frame 0.
    # Noise can see earlier clean frames.
    assert bool(profile.self_attention_mask[4, 8].item()) is True
    # Query = clean latent token on frame 0 can see itself.
    assert bool(profile.self_attention_mask[8, 8].item()) is True
    # Padded rows/cols are fully masked out.
    assert bool(profile.self_attention_mask[-1].any().item()) is False
    assert bool(profile.self_attention_mask[:, -1].any().item()) is False
    assert bool(profile.cross_attention_mask[-1].any().item()) is False


def test_chunked_temporal_exact_attention_profile_uses_patchified_frame_ids() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 2, 4, 2, 2),
        action_shape=(1, 3, 2, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=8,
        patch_size=(2, 1, 1),
        text_token_count=2,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
    )

    assert profile.self_attention_mask is not None
    # 2 patchified video frames -> 8 latent tokens, doubled for noisy/clean,
    # plus 2 action frames doubled for noisy/clean.
    assert profile.self_attention_mask.shape == (20, 20)


def test_replica_core_exact_forward_train_supports_flex_profile_cpu_fallback() -> None:
    config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        hidden_size=64,
        num_layers=1,
        num_heads=8,
        latent_channels=4,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
        text_dim=16,
        freq_dim=16,
        ffn_dim=128,
        attn_mode="flex",
    )
    core = SharedVideoTransformerCore(config, action_dim=3)

    latent_grid_id = get_mesh_id(2, 2, 2, t=0, action=False, device=torch.device("cpu"))[None]
    action_grid_id = get_mesh_id(2, 1, 1, t=1, action=True, device=torch.device("cpu"))[None]
    input_dict = {
        "latent_dict": {
            "timesteps": torch.zeros(1, 2, dtype=torch.float32),
            "noisy_latents": torch.randn(1, 4, 2, 2, 2),
            "targets": torch.randn(1, 4, 2, 2, 2),
            "latent": torch.randn(1, 4, 2, 2, 2),
            "cond_timesteps": torch.zeros(1, 2, dtype=torch.float32),
            "grid_id": latent_grid_id,
            "text_emb": torch.randn(1, 4, 16),
        },
        "action_dict": {
            "timesteps": torch.zeros(1, 2, dtype=torch.float32),
            "noisy_latents": torch.randn(1, 3, 2, 1, 1),
            "targets": torch.randn(1, 3, 2, 1, 1),
            "latent": torch.randn(1, 3, 2, 1, 1),
            "cond_timesteps": torch.zeros(1, 2, dtype=torch.float32),
            "grid_id": action_grid_id,
            "text_emb": torch.randn(1, 4, 16),
        },
        "chunk_size": 1,
        "window_size": 8,
    }

    latent_pred, action_pred = core.forward_train(input_dict)

    assert latent_pred.shape == (1, 8, 4)
    assert action_pred.shape == (1, 2, 3)
