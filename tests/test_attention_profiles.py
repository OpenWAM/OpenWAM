from __future__ import annotations

import torch

from open_wam.models.common.attention_profiles import (
    build_chunked_temporal_exact_attention_profile,
    normalize_chunked_temporal_exact_coupling,
)
from open_wam.models.common.packed_token_layout import (
    PackedTokenKind,
    PackedTokenStream,
    build_exact_video_action_token_layout,
)
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


def test_chunked_temporal_exact_cross_mask_limits_per_chunk_proprio_tokens() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(2, 1, 6, 1, 1),
        action_shape=(2, 1, 6, 1, 1),
        padded_length=2,
        chunk_size=2,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=6,
        base_text_token_count=3,
        proprio_context_token_count=3,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
    )

    assert profile.cross_attention_mask is not None
    mask = profile.cross_attention_mask
    # Query layout starts with sample-0 latent noisy frames 0..5, then sample-1
    # latent noisy frames 0..5. Context layout is sample-major text rows.
    sample0_chunk0_query = 0
    sample0_chunk1_query = 2
    sample1_chunk0_query = 6
    sample0_text = torch.arange(0, 6)
    sample1_text = torch.arange(6, 12)

    assert mask.shape == (50, 12)
    assert mask[sample0_chunk0_query, sample0_text].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert mask[sample0_chunk1_query, sample0_text].tolist() == [
        True,
        True,
        True,
        False,
        True,
        False,
    ]
    assert bool(mask[sample0_chunk0_query, sample1_text].any().item()) is False
    assert mask[sample1_chunk0_query, sample1_text].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert bool(mask[-1].any().item()) is False


def test_chunked_temporal_exact_prefix_condition_is_history_context() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 5, 1, 1),
        action_shape=(1, 1, 4, 1, 1),
        padded_length=0,
        chunk_size=2,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=1,
        current_block_coupling="decoupled_same_step",
        history_stream_visibility="video_only",
        prefix_condition_frames=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
    )

    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    video_noisy_start = 0
    video_clean_start = 5
    action_noisy_start = 10
    prefix_clean = video_clean_start
    first_target_video_noisy = video_noisy_start + 1
    first_target_action_noisy = action_noisy_start
    first_target_action_clean = 14

    assert bool(mask[first_target_video_noisy, prefix_clean].item()) is True
    assert bool(mask[first_target_action_noisy, prefix_clean].item()) is True
    assert bool(mask[first_target_video_noisy, first_target_action_clean].item()) is False


def test_chunked_temporal_exact_chunk_origin_keeps_context_frame_out_of_first_target_chunk() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 5, 1, 1),
        action_shape=(1, 1, 5, 1, 1),
        padded_length=0,
        chunk_size=4,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=3,
        base_text_token_count=1,
        proprio_context_token_count=2,
        chunk_origin_frame=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
    )

    assert profile.cross_attention_mask is not None
    mask = profile.cross_attention_mask
    # Frame 0 is a prefix-context chunk (-1): it sees text, but no deprecated
    # per-target proprio text token. Frames 1 and 4 are both in generated chunk 0.
    assert mask[0, :3].tolist() == [True, False, False]
    assert mask[1, :3].tolist() == [True, True, False]
    assert mask[4, :3].tolist() == [True, True, False]
    assert profile.metadata["chunk_origin_frame"] == 1


def test_chunked_temporal_exact_action_context_mask_hides_startup_action_tokens() -> None:
    action_context_mask = torch.ones(1, 1, 5, 4, 1)
    action_context_mask[:, :, 0] = 0.0
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 5, 1, 1),
        action_shape=(1, 1, 5, 4, 1),
        padded_length=0,
        chunk_size=4,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=1,
        chunk_origin_frame=1,
        action_context_mask=action_context_mask,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
        preserve_video_pretrain_history=True,
    )

    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    latent_token_count = 5
    action_token_count = 20
    action_noisy_start = latent_token_count * 2
    action_clean_start = action_noisy_start + action_token_count
    query_action_frame1 = action_noisy_start + 4
    kv_video_clean_frame0 = latent_token_count
    kv_action_noisy_frame0 = action_noisy_start
    kv_action_clean_frame0 = action_clean_start

    assert bool(mask[query_action_frame1, kv_video_clean_frame0].item()) is True
    assert bool(mask[query_action_frame1, kv_action_noisy_frame0].item()) is False
    assert bool(mask[query_action_frame1, kv_action_clean_frame0].item()) is False
    # Invalid/context action tokens are hidden as keys, but remain safe query
    # rows so FlexAttention never sees an all-masked query during strict
    # one-frame startup training.
    assert bool(mask[kv_action_noisy_frame0].any().item()) is True
    assert bool(mask[:, kv_action_noisy_frame0].any().item()) is False
    assert bool(mask[:, kv_action_clean_frame0].any().item()) is False
    assert profile.cross_attention_mask is not None
    assert bool(profile.cross_attention_mask[kv_action_noisy_frame0].any().item()) is True
    assert profile.metadata["invalid_action_context_tokens"] == 4


def test_history_stream_visibility_video_only_filters_all_action_history() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 2, 1, 1),
        action_shape=(1, 1, 2, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
        current_block_coupling="video_then_action",
        history_stream_visibility="video_only",
    )

    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    latent_token_count = 2
    action_token_count = 2
    action_noisy_start = latent_token_count * 2
    action_clean_start = action_noisy_start + action_token_count
    query_video_frame1 = 1
    query_action_frame1 = action_noisy_start + 1
    kv_video_clean_frame0 = latent_token_count
    kv_action_clean_frame0 = action_clean_start

    assert bool(mask[query_video_frame1, kv_video_clean_frame0].item()) is True
    assert bool(mask[query_video_frame1, kv_action_clean_frame0].item()) is False
    assert bool(mask[query_action_frame1, kv_video_clean_frame0].item()) is True
    assert bool(mask[query_action_frame1, kv_action_clean_frame0].item()) is False
    assert profile.metadata["history_stream_visibility"] == "video_only"


def test_chunked_temporal_exact_coupling_accepts_profile_aliases() -> None:
    assert normalize_chunked_temporal_exact_coupling("lingbot_chunked_exact") == "video_then_action"
    assert normalize_chunked_temporal_exact_coupling("chunked_temporal_exact_joint") == "joint"


def test_packed_token_layout_separates_query_and_kv_validity() -> None:
    action_context_mask = torch.ones(1, 1, 5, 4, 1)
    action_context_mask[:, :, 0] = 0.0

    layout = build_exact_video_action_token_layout(
        batch_size=1,
        latent_frames=5,
        latent_height=1,
        latent_width=1,
        action_frames=5,
        action_height=4,
        action_width=1,
        patch_size=(1, 1, 1),
        chunk_size=4,
        chunk_origin_frame=1,
        current_block_coupling="video_then_action",
        device=torch.device("cpu"),
        action_context_mask=action_context_mask,
    )

    latent_token_count = 5
    action_token_count = 20
    action_noisy_start = latent_token_count * 2
    action_clean_start = action_noisy_start + action_token_count

    assert layout.valid_for_loss[:latent_token_count].all()
    assert not layout.valid_for_loss[latent_token_count : latent_token_count * 2].any()
    assert int(layout.token_kind[action_noisy_start]) == int(PackedTokenKind.ACTION_NOISY)
    assert int(layout.stream_id[action_noisy_start]) == int(PackedTokenStream.ACTION)
    assert layout.valid_as_query[action_noisy_start : action_noisy_start + 4].all()
    assert not layout.valid_as_kv[action_noisy_start : action_noisy_start + 4].any()
    assert not layout.valid_as_kv[action_clean_start : action_clean_start + 4].any()
    assert layout.valid_as_kv[action_noisy_start + 4 : action_noisy_start + 8].all()
    assert not layout.valid_for_loss[action_noisy_start : action_noisy_start + 4].any()
    assert layout.valid_for_loss[action_noisy_start + 4 : action_noisy_start + 8].all()
    assert not layout.valid_for_loss[action_clean_start:].any()
    assert layout.valid_for_loss.shape == layout.valid_as_kv.shape


def test_chunked_temporal_exact_cross_mask_keeps_cfg_rows_isolated() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(2, 1, 2, 1, 1),
        action_shape=(2, 1, 2, 1, 1),
        padded_length=0,
        chunk_size=2,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=3,
        base_text_token_count=2,
        proprio_context_token_count=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
    )

    assert profile.cross_attention_mask is not None
    mask = profile.cross_attention_mask
    conditional_query = 0
    unconditional_query = 2
    conditional_context = torch.arange(0, 3)
    unconditional_context = torch.arange(3, 6)

    assert mask[conditional_query, conditional_context].tolist() == [True, True, True]
    assert bool(mask[conditional_query, unconditional_context].any().item()) is False
    assert mask[unconditional_query, unconditional_context].tolist() == [True, True, True]
    assert bool(mask[unconditional_query, conditional_context].any().item()) is False


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


def _tiny_exact_mask(current_block_coupling: str) -> torch.Tensor:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 2, 1, 1),
        action_shape=(1, 1, 2, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
        current_block_coupling=current_block_coupling,
    )
    assert profile.self_attention_mask is not None
    return profile.self_attention_mask


def test_chunked_temporal_exact_video_then_action_couples_action_to_new_video() -> None:
    mask = _tiny_exact_mask("video_then_action")
    v_noisy_0, v_clean_0, a_noisy_0, a_clean_0 = 0, 2, 4, 6

    assert bool(mask[a_noisy_0, v_clean_0].item()) is True
    assert bool(mask[v_noisy_0, a_clean_0].item()) is False
    assert bool(mask[v_noisy_0, a_noisy_0].item()) is False


def test_chunked_temporal_exact_action_then_video_couples_video_to_new_action() -> None:
    mask = _tiny_exact_mask("action_then_video")
    v_noisy_0, v_clean_0, a_noisy_0, a_clean_0 = 0, 2, 4, 6

    assert bool(mask[v_noisy_0, a_clean_0].item()) is True
    assert bool(mask[a_noisy_0, v_clean_0].item()) is False
    assert bool(mask[v_noisy_0, a_noisy_0].item()) is False


def test_chunked_temporal_exact_joint_couples_same_block_noisy_streams() -> None:
    mask = _tiny_exact_mask("joint")
    v_noisy_0, v_clean_0, a_noisy_0 = 0, 2, 4

    assert bool(mask[v_noisy_0, a_noisy_0].item()) is True
    assert bool(mask[a_noisy_0, v_noisy_0].item()) is True
    assert bool(mask[a_noisy_0, v_clean_0].item()) is False


def test_chunked_temporal_exact_video_noisy_to_action_is_one_way() -> None:
    mask = _tiny_exact_mask("video_noisy_to_action")
    v_noisy_0, v_clean_0, a_noisy_0, a_clean_0 = 0, 2, 4, 6

    assert bool(mask[a_noisy_0, v_noisy_0].item()) is True
    assert bool(mask[v_noisy_0, a_noisy_0].item()) is False
    assert bool(mask[a_noisy_0, v_clean_0].item()) is False
    assert bool(mask[v_noisy_0, a_clean_0].item()) is False


def test_chunked_temporal_exact_action_noisy_to_video_is_one_way() -> None:
    mask = _tiny_exact_mask("action_noisy_to_video")
    v_noisy_0, v_clean_0, a_noisy_0, a_clean_0 = 0, 2, 4, 6

    assert bool(mask[v_noisy_0, a_noisy_0].item()) is True
    assert bool(mask[a_noisy_0, v_noisy_0].item()) is False
    assert bool(mask[v_noisy_0, a_clean_0].item()) is False
    assert bool(mask[a_noisy_0, v_clean_0].item()) is False


def test_chunked_temporal_exact_decoupled_hides_same_step_cross_stream_context() -> None:
    mask = _tiny_exact_mask("decoupled_same_step")
    v_noisy_0, v_clean_0, a_noisy_0, a_clean_0 = 0, 2, 4, 6

    assert bool(mask[a_noisy_0, v_clean_0].item()) is False
    assert bool(mask[v_noisy_0, a_clean_0].item()) is False
    assert bool(mask[v_noisy_0, a_noisy_0].item()) is False
    assert bool(mask[a_noisy_0, a_noisy_0].item()) is True
    assert bool(mask[v_clean_0, a_clean_0].item()) is False
    assert bool(mask[a_clean_0, v_clean_0].item()) is False


def test_preserve_video_pretrain_history_filters_video_queries_only() -> None:
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 4, 1, 1),
        action_shape=(1, 1, 4, 1, 1),
        padded_length=0,
        chunk_size=2,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
        current_block_coupling="joint",
        preserve_video_pretrain_history=True,
    )
    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    # Layout for four frames: V_noisy [0:4], V_clean [4:8], A_noisy [8:12], A_clean [12:16].
    v_noisy_chunk1, v_clean_chunk1, a_noisy_chunk1 = 2, 6, 10
    v_clean_history, a_clean_history = 4, 12

    assert bool(mask[v_noisy_chunk1, v_clean_history].item()) is True
    assert bool(mask[v_noisy_chunk1, a_clean_history].item()) is False
    assert bool(mask[v_clean_chunk1, v_clean_history].item()) is True
    assert bool(mask[v_clean_chunk1, a_clean_history].item()) is False
    assert bool(mask[a_noisy_chunk1, v_clean_history].item()) is True
    assert bool(mask[a_noisy_chunk1, a_clean_history].item()) is True
    assert profile.metadata["preserve_video_pretrain_history"] is True


def test_preserve_video_pretrain_history_keeps_staged_current_condition_visible() -> None:
    video_then_action = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 2, 1, 1),
        action_shape=(1, 1, 2, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
        current_block_coupling="video_then_action",
        preserve_video_pretrain_history=True,
    )
    action_then_video = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 2, 1, 1),
        action_shape=(1, 1, 2, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=8,
        patch_size=(1, 1, 1),
        text_token_count=1,
        device=torch.device("cpu"),
        build_dense_masks=True,
        build_flex_masks=False,
        current_block_coupling="action_then_video",
        preserve_video_pretrain_history=True,
    )
    assert video_then_action.self_attention_mask is not None
    assert action_then_video.self_attention_mask is not None
    v_noisy_0, v_clean_0, a_noisy_0, a_clean_0 = 0, 2, 4, 6

    assert bool(video_then_action.self_attention_mask[a_noisy_0, v_clean_0].item()) is True
    assert bool(video_then_action.self_attention_mask[v_noisy_0, a_clean_0].item()) is False
    assert bool(action_then_video.self_attention_mask[v_noisy_0, a_clean_0].item()) is True
    assert bool(action_then_video.self_attention_mask[a_noisy_0, v_clean_0].item()) is False


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


def test_replica_core_appends_deprecated_text_token_proprio_context_tokens() -> None:
    config = SharedVideoTransformerConfig(
        implementation="shared_transformer",
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        text_dim=8,
        freq_dim=8,
        ffn_dim=64,
    )
    core = SharedVideoTransformerCore(config, action_dim=3, state_dim=5)
    core.configure_proprio_context_encoder(enabled=True, state_dim=5)
    text_emb = torch.randn(2, 4, 8)
    proprio = torch.randn(2, 3, 5)

    appended = core.append_proprio_context_tokens(text_emb, proprio)  # deprecated helper

    assert appended.shape == (2, 7, 8)
    assert torch.allclose(appended[:, :4], text_emb)
    # The encoder is zero-initialized so adding the new conditioning path does
    # not perturb old checkpoints until it is trained.
    assert torch.allclose(appended[:, 4:], torch.zeros_like(appended[:, 4:]))
