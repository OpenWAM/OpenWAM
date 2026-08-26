from __future__ import annotations

import torch

from open_wam.configs import SharedVideoTransformerConfig
from open_wam.models.common.video_geometry import unpatchify_video_sequence
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.visual_tower.exact_runtime import (
    build_reference_mesh_id,
    clear_exact_prediction_cache,
    initialize_exact_runtime_cache,
    prepare_exact_single_stream_forward_input,
    prepare_exact_single_stream_input,
    repeat_exact_single_stream_input_for_cfg,
    resolve_runtime_module_dtype,
    run_exact_single_stream_forward,
)


def test_reference_runtime_compatibility_names_alias_canonical_owners() -> None:
    assert reference_runtime.data_seq_to_patch is unpatchify_video_sequence
    assert reference_runtime.get_mesh_id is build_reference_mesh_id
    assert reference_runtime._clear_exact_prediction_cache is clear_exact_prediction_cache
    assert reference_runtime.initialize_reference_cache is initialize_exact_runtime_cache
    assert (
        reference_runtime.prepare_reference_forward_input
        is prepare_exact_single_stream_forward_input
    )
    assert (
        reference_runtime.prepare_reference_single_stream_input
        is prepare_exact_single_stream_input
    )
    assert (
        reference_runtime.repeat_input_for_cfg
        is repeat_exact_single_stream_input_for_cfg
    )
    assert reference_runtime.reference_runtime_dtype is resolve_runtime_module_dtype
    assert (
        reference_runtime.run_reference_single_stream_forward
        is run_exact_single_stream_forward
    )


def test_unpatchify_video_sequence_preserves_values_and_gradients() -> None:
    token_predictions = torch.arange(32, dtype=torch.float64).reshape(1, 8, 4)
    token_predictions.requires_grad_(True)

    actual = unpatchify_video_sequence(
        (1, 2, 2),
        token_predictions,
        2,
        4,
        4,
        batch_size=1,
    )

    expected = token_predictions.reshape(1, 2, 2, 2, 1, 2, 2, 1)
    expected = expected.permute(0, 7, 1, 4, 2, 5, 3, 6)
    expected = expected.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    actual.square().sum().backward()
    torch.testing.assert_close(
        token_predictions.grad,
        2.0 * token_predictions.detach(),
        rtol=0.0,
        atol=0.0,
    )


def test_prepare_exact_single_stream_input_preserves_reference_grid_dtypes() -> None:
    config = SharedVideoTransformerConfig(
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    text_emb = torch.zeros(1, 2, 3)

    video_input = prepare_exact_single_stream_input(
        latents=torch.zeros(1, 4, 2, 4, 4),
        timestep=3.0,
        text_emb=text_emb,
        frame_st_id=5,
        backbone_config=config,
        action_mode=False,
    )
    action_input = prepare_exact_single_stream_input(
        latents=torch.zeros(1, 7, 2, 4, 1),
        timestep=3.0,
        text_emb=text_emb,
        frame_st_id=5,
        backbone_config=config,
        action_mode=True,
    )

    assert video_input["grid_id"].dtype == torch.int64
    assert action_input["grid_id"].dtype == torch.float32
    assert video_input["grid_id"].shape == (1, 4, 8)
    assert action_input["grid_id"].shape == (1, 4, 8)


def test_prepare_exact_single_stream_input_conditions_the_complete_prefix() -> None:
    """Inference must label every clean observed frame exactly as training does."""

    config = SharedVideoTransformerConfig(
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    latents = torch.arange(1 * 4 * 6 * 4 * 4, dtype=torch.float32).reshape(
        1, 4, 6, 4, 4
    )
    condition = torch.full((1, 4, 3, 4, 4), 7.0)

    prepared = prepare_exact_single_stream_input(
        latents=latents,
        timestep=500.0,
        text_emb=torch.zeros(1, 2, 3),
        frame_st_id=0,
        backbone_config=config,
        action_mode=False,
        cond=condition,
    )

    torch.testing.assert_close(
        prepared["noisy_latents"][:, :, :3],
        condition,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        prepared["noisy_latents"][:, :, 3:],
        latents[:, :, 3:],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(prepared["timesteps"][:, :3]).item() == 0
    assert torch.all(prepared["timesteps"][:, 3:] == 500.0)
