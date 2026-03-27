from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from open_wam.data import build_synthetic_batch
from open_wam.models.policy_variants import PolicyInferContext
from open_wam.models.video_backbone.contracts import CacheUpdateMetadata
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_register_attached_runtime_owns_sequence_assembly() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/register_attached_robotwin_smoke.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)
    visual_outputs = pipeline.prepare_visual_outputs(batch.views)

    variant = pipeline.policy_variant
    runtime = variant.runtime

    train_layout = runtime.build_layout(visual_outputs, include_clean_video_prefix=True)
    infer_layout = runtime.build_layout(visual_outputs, include_clean_video_prefix=False)

    assert train_layout.has_clean_video_prefix is True
    assert infer_layout.has_clean_video_prefix is False
    assert train_layout.clean_video_sequence_length == visual_outputs.frontend.token_grid.sequence_length
    assert infer_layout.clean_video_sequence_length == 0

    noisy_visual_outputs = runtime.build_noisy_frontend_outputs(
        pipeline.visual_tower,
        visual_outputs=visual_outputs,
        noisy_video_latents=torch.randn_like(visual_outputs.frontend.video_latents),
    )
    result = runtime.run_core(
        visual_tower=pipeline.visual_tower,
        visual_outputs=visual_outputs,
        clean_video_prefix_tokens=visual_outputs.frontend.video_tokens,
        noisy_video_tokens=noisy_visual_outputs.frontend.video_tokens,
        action_inputs=batch.actions,
        state_inputs=batch.state,
        video_timesteps=torch.zeros(2, config.data.num_frames),
        action_timesteps=torch.zeros(2, config.data.action_schema.action_horizon),
        current_start_frame=0,
        cache_update_metadata=CacheUpdateMetadata(
            current_start_frame=0,
            update_kv_cache=True,
            update_cross_attention_cache=True,
        ),
    )

    assert result.video_hidden.shape == (
        2,
        visual_outputs.frontend.token_grid.sequence_length,
        config.policy_variant.hidden_size,
    )
    assert result.action_hidden.shape == (
        2,
        config.data.action_schema.action_horizon,
        config.policy_variant.hidden_size,
    )
    assert result.layout.has_clean_video_prefix is True
    assert result.cache_state.supported is True
    assert result.cache_state.update_metadata.update_kv_cache is True
    assert len(result.cache_state.self_attention_kv) == config.backbone.num_layers
    assert result.cache_state.capability == "self_attn_plus_cross_attn"
    assert result.cache_state.self_attention_kv[0].key is not None
    assert result.cache_state.self_attention_kv[0].value is not None
    assert result.cache_state.cross_attention_kv[0].key is not None
    assert result.cache_state.cross_attention_kv[0].value is not None
    assert result.aux["structured_block_mode"] == "register_explicit"
    assert result.aux["structured_attention_mode"] == "register_explicit"
    assert result.aux["structured_attention_kernel"] == "branchwise_explicit"
    assert result.aux["structured_cache_kernel"] == "branchwise_rollout_explicit"
    assert result.aux["stream_output_head_family"] == "structured_joint_flow"
    assert result.aux["structured_attention_internal_mask"] is True
    assert result.aux["structured_frequency_mode"] == "stream_local"
    assert result.aux["structured_has_clean_prefix_frequencies"] is True
    assert result.aux["structured_has_action_frequencies"] is True
    assert result.aux["structured_has_state_frequencies"] is True
    assert result.aux["structured_register_frame_shift"] == 1.0
    assert result.aux["structured_time_layout"] == "video_action_state"
    assert result.aux["structured_position_layout"] == "clean_prefix"
    assert result.aux["structured_current_start_frame"] == 0
    assert result.aux["structured_observed_prefix_frames"] == 1
    assert result.aux["structured_action_register_length"] == config.data.action_schema.action_horizon
    assert result.aux["structured_state_register_length"] == config.data.action_schema.state_horizon
    assert result.aux["structured_clean_prefix_length"] == visual_outputs.frontend.token_grid.sequence_length
    assert result.projected_outputs["video_patch_flow"].shape == (
        2,
        visual_outputs.frontend.token_grid.sequence_length,
        pipeline.visual_tower.core.proj_out.out_features,
    )
    assert result.projected_outputs["action_flow"].shape == (
        2,
        config.data.action_schema.action_horizon,
        config.data.action_schema.action_dim,
    )
    assert not hasattr(variant, "action_encoder")
    assert not hasattr(variant, "state_encoder")
    assert not hasattr(variant, "register_role_embedding")
    assert not hasattr(variant, "video_flow_head")
    assert not hasattr(variant, "action_flow_head")


def test_register_attached_inference_cache_persists_video_history() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/register_attached_robotwin_smoke.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)

    step_one = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))
    first_cache = step_one.policy_output.next_state.cache
    tokens_per_frame = step_one.visual_outputs.frontend.token_grid.tokens_per_frame
    cached_video_tokens_per_step = tokens_per_frame

    assert first_cache.self_attention_kv[0].key is not None
    assert first_cache.self_attention_kv[0].key.shape[2] == cached_video_tokens_per_step
    assert first_cache.update_metadata.max_cached_frames is None
    assert first_cache.self_attention_kv[0].metadata["segment_token_lengths"] == (tokens_per_frame,)

    step_two = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state),
        infer_state=step_one.policy_output.next_state,
    )
    second_cache = step_two.policy_output.next_state.cache

    assert second_cache.self_attention_kv[0].key is not None
    assert second_cache.self_attention_kv[0].key.shape[2] == cached_video_tokens_per_step * 2
    assert second_cache.cached_frames == 2
    assert second_cache.self_attention_kv[0].metadata["segment_token_lengths"] == (
        tokens_per_frame,
        tokens_per_frame,
    )
    assert step_two.policy_output.aux["structured_attention_full_cache_prefix"] is True
    assert step_two.policy_output.aux["core_aux"]["structured_cache_kernel"] == "branchwise_rollout_explicit"


def test_register_attached_default_cache_policy_prefills_then_freezes() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/register_attached_robotwin_smoke.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)

    step_one = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))
    first_cache = step_one.policy_output.next_state.cache

    assert first_cache.update_metadata.update_kv_cache is False
    assert first_cache.update_metadata.update_cross_attention_cache is False
    assert first_cache.update_metadata.cfg_mode in {"joint", "joint_cfg"}


def test_register_attached_cfg_keeps_separate_conditioned_and_unconditioned_cache_branches() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/register_attached_robotwin_smoke.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)
    visual_outputs = pipeline.prepare_visual_outputs(batch.views)
    text_dim = config.backbone.text_dim
    visual_outputs.frontend.conditioning.text_context = torch.randn(
        batch.actions.shape[0], 4, text_dim, device=batch.actions.device
    )
    visual_outputs.frontend.conditioning.negative_text_context = torch.randn(
        batch.actions.shape[0], 4, text_dim, device=batch.actions.device
    )
    variant = pipeline.policy_variant
    variant.inference_config = replace(variant.inference_config, guidance_scale=2.0)
    infer_state = variant.prepare_infer_state(
        pipeline.visual_tower,
        visual_outputs,
        PolicyInferContext(state=batch.state),
    )
    output = variant.forward_infer_step(
        pipeline.visual_tower,
        visual_outputs,
        PolicyInferContext(state=batch.state),
        infer_state,
    )
    cache_state = output.next_state.cache

    conditioned = cache_state.branch_states["conditioned"]
    unconditioned = cache_state.branch_states["unconditioned"]

    assert conditioned.self_attention_kv[0].key is not None
    assert unconditioned.self_attention_kv[0].key is not None
    assert conditioned.cross_attention_kv[0].key is not None
    assert unconditioned.cross_attention_kv[0].key is not None
    assert conditioned.self_attention_kv[0].key.data_ptr() != unconditioned.self_attention_kv[0].key.data_ptr()
    assert conditioned.cross_attention_kv[0].key.data_ptr() != unconditioned.cross_attention_kv[0].key.data_ptr()


def test_register_attached_inference_keeps_observed_prefix_frame_fixed() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/register_attached_robotwin_smoke.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)

    output = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))
    predicted_latents = output.policy_output.aux["predicted_latents"]
    observed_latents = output.visual_outputs.frontend.video_latents

    assert isinstance(predicted_latents, torch.Tensor)
    assert torch.allclose(predicted_latents[:, :, 0], observed_latents[:, :, 0])
