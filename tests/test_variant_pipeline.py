from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from open_wam.configs import ParallelStreamPolicyConfig
from open_wam.data import build_synthetic_batch, build_synthetic_latent_batch
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_pipeline(config_path: Path) -> tuple:
    config = load_experiment_config(config_path)
    # Joint diffusion variants now run the shared core at every denoising step.
    # Keep the pipeline test on a short smoke rollout so it stays focused on
    # shape/contract coverage rather than full reference-timing parity.
    if not (
        isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        and config.policy_variant.reference_profile is not None
    ):
        config = replace(
            config,
            inference=replace(
                config.inference,
                video_num_inference_steps=min(config.inference.video_num_inference_steps, 2),
                action_num_inference_steps=min(config.inference.action_num_inference_steps, 2),
                joint_num_inference_steps=(
                    min(config.inference.joint_num_inference_steps, 2)
                    if config.inference.joint_num_inference_steps is not None
                    else None
                ),
            ),
        )
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(actions=batch.actions, action_mask=batch.action_mask, state=batch.state)
    return config, pipeline, batch, train_batch


@pytest.mark.parametrize(
    ("config_name", "expected_horizon"),
    [
        ("post_latent_robotwin.yaml", 6),
        ("post_decoded_robotwin.yaml", 6),
        ("register_attached_robotwin_smoke.yaml", 6),
        ("parallel_stream_robotwin_smoke.yaml", 8),
    ],
)
def test_variant_pipeline_train_and_infer_shapes(config_name: str, expected_horizon: int) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config, pipeline, batch, train_batch = _build_pipeline(config_path)

    sequence_train_batch = PolicyTrainBatch(
        actions=train_batch.actions,
        action_mask=train_batch.action_mask,
        state=train_batch.state,
        extra={"task_text": batch.task_text},
    )
    train_output = pipeline.forward_train(batch.views, sequence_train_batch)
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )

    assert train_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )
    assert infer_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )


@pytest.mark.parametrize(
    ("config_name", "expected_horizon"),
    [
        ("post_latent_libero_smoke.yaml", 6),
        ("post_decoded_libero_smoke.yaml", 6),
        ("video_sequence_policy_libero_smoke.yaml", 6),
        ("register_attached_libero_smoke.yaml", 6),
        ("parallel_stream_libero_raw_smoke.yaml", 16),
    ],
)
def test_raw_libero_variant_pipeline_train_and_infer_shapes(config_name: str, expected_horizon: int) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config, pipeline, batch, train_batch = _build_pipeline(config_path)

    sequence_train_batch = PolicyTrainBatch(
        actions=train_batch.actions,
        action_mask=train_batch.action_mask,
        state=train_batch.state,
        extra={"task_text": batch.task_text},
    )
    train_output = pipeline.forward_train(batch.views, sequence_train_batch)
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )

    assert train_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )
    assert infer_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )


@pytest.mark.parametrize(
    ("config_name", "expected_horizon"),
    [
        ("post_latent_robotwin.yaml", 6),
        ("parallel_stream_robotwin_smoke.yaml", 8),
    ],
)
def test_variant_pipeline_train_from_latents_shapes(config_name: str, expected_horizon: int) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    config = load_experiment_config(config_path)
    pipeline = build_variant_pipeline_from_config(config)
    latent_batch = build_synthetic_latent_batch(config.data, batch_size=2)
    train_batch = PolicyTrainBatch(
        actions=latent_batch.actions,
        action_mask=latent_batch.action_mask,
        state=latent_batch.state,
        extra={
            "task_text": latent_batch.task_text,
            "metadata": latent_batch.metadata,
            "state_mask": latent_batch.state_mask,
        },
    )

    train_output = pipeline.forward_train_from_latents(
        latent_batch.video_latents,
        train_batch,
        canonical_video=latent_batch.canonical_video,
        text_context=latent_batch.text_context,
        negative_text_context=latent_batch.negative_text_context,
    )

    assert train_output.decoder_output.action_pred.shape == (
        2,
        expected_horizon,
        config.action_decoder.action_dim,
    )


@pytest.mark.parametrize(
    ("config_name", "expected_source_stage"),
    [
        ("post_latent_robotwin.yaml", "core"),
        ("post_decoded_robotwin.yaml", "decode"),
    ],
)
def test_post_visual_variants_emit_decoder_sequence_context(
    config_name: str,
    expected_source_stage: str,
) -> None:
    config_path = REPO_ROOT / "configs/experiments" / config_name
    _, pipeline, batch, train_batch = _build_pipeline(config_path)

    train_output = pipeline.forward_train(batch.views, train_batch)
    infer_output = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))

    train_context = train_output.policy_output.decoder_sequence_context
    infer_context = infer_output.policy_output.decoder_sequence_context

    assert train_context is not None
    assert infer_context is not None
    assert train_context.sequence_layout["family"] == "video_feature_policy"
    assert infer_context.sequence_layout["family"] == "video_feature_policy"
    assert train_context.source_stage == expected_source_stage
    assert infer_context.source_stage == expected_source_stage
    assert train_context.frame_count is not None and train_context.frame_count > 0
    assert infer_context.frame_count is not None and infer_context.frame_count > 0


def test_video_sequence_policy_pipeline_emits_core_token_sequence_context(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "video_sequence_policy_robotwin"
    raw["policy_variant"]["name"] = "video_sequence_policy"
    raw["policy_variant"]["attach_site"] = "post_visual_core"
    raw["policy_variant"].pop("pooling_mode", None)
    raw["policy_variant"].pop("query_count", None)
    raw["policy_variant"].pop("use_state_projection", None)
    raw["action_decoder"]["name"] = "vpp_decoder"
    raw["action_decoder"]["num_sampling_steps"] = 2
    raw["action_decoder"]["rollout_chunk_steps"] = 2

    config_path = tmp_path / "video_sequence_policy_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config, pipeline, batch, train_batch = _build_pipeline(config_path)
    sequence_train_batch = PolicyTrainBatch(
        actions=train_batch.actions,
        action_mask=train_batch.action_mask,
        state=train_batch.state,
        extra={"task_text": batch.task_text},
    )

    train_output = pipeline.forward_train(batch.views, sequence_train_batch)
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )

    assert train_output.decoder_output.action_pred.shape == (2, 6, config.action_decoder.action_dim)
    assert infer_output.decoder_output.action_pred.shape == (2, 6, config.action_decoder.action_dim)

    train_context = train_output.policy_output.decoder_sequence_context
    infer_context = infer_output.policy_output.decoder_sequence_context

    assert train_context is not None
    assert infer_context is not None
    assert train_context.sequence_layout["family"] == "video_sequence_policy"
    assert infer_context.sequence_layout["family"] == "video_sequence_policy"
    assert train_context.source_stage == "core"
    assert infer_context.source_stage == "core"
    assert train_context.state_sequence is not None
    assert infer_context.state_sequence is not None
    assert infer_output.decoder_output.aux["sampled_new_chunk"] is True

    second_infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
        infer_state=infer_output.policy_output.next_state,
    )
    assert second_infer_output.decoder_output.aux["sampled_new_chunk"] is False
    assert second_infer_output.policy_output.next_state.decoder_state is not None
    assert torch.allclose(
        second_infer_output.decoder_output.action_pred,
        infer_output.decoder_output.action_pred,
    )


def test_video_sequence_policy_exact_vpp_knobs_pipeline_runs(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "video_sequence_policy_exact_robotwin"
    raw["policy_variant"]["name"] = "video_sequence_policy"
    raw["policy_variant"]["attach_site"] = "post_visual_core"
    raw["policy_variant"].pop("pooling_mode", None)
    raw["policy_variant"].pop("query_count", None)
    raw["policy_variant"].pop("use_state_projection", None)
    raw["action_decoder"]["name"] = "vpp_decoder"
    raw["action_decoder"]["num_sampling_steps"] = 2
    raw["action_decoder"]["rollout_chunk_steps"] = 2
    raw["action_decoder"]["temporal_compression_adapter_family"] = "video_former_3d"
    raw["action_decoder"]["sequence_denoiser_family"] = "film_diffusion_transformer"

    config_path = tmp_path / "video_sequence_policy_exact_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config, pipeline, batch, train_batch = _build_pipeline(config_path)
    sequence_train_batch = PolicyTrainBatch(
        actions=train_batch.actions,
        action_mask=train_batch.action_mask,
        state=train_batch.state,
        extra={"task_text": batch.task_text},
    )

    train_output = pipeline.forward_train(batch.views, sequence_train_batch)
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )

    assert train_output.decoder_output.action_pred.shape == (2, 6, config.action_decoder.action_dim)
    assert infer_output.decoder_output.action_pred.shape == (2, 6, config.action_decoder.action_dim)
    assert infer_output.policy_output.decoder_sequence_context is not None
    assert infer_output.decoder_output.aux["sampled_new_chunk"] is True
