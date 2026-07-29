from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from open_wam.configs import ParallelStreamPolicyConfig, VideoConditionSource
from open_wam.data import build_synthetic_batch, build_synthetic_latent_batch
from open_wam.models.policy_variants.common.layouts import tokens_to_frame_major
from open_wam.models.policy_variants.common.video_conditioning import (
    resolve_video_condition_frame_start,
    resolve_video_condition_sample_seed,
)
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
        ("post_latent_robotwin_video_conditioned.yaml", 6),
        ("post_decoded_robotwin_video_conditioned.yaml", 6),
        ("parallel_stream_robotwin_smoke.yaml", 8),
        ("mot_robotwin_smoke.yaml", 8),
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
        ("post_latent_robotwin_video_conditioned.yaml", 6),
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


def test_method4_rgb_video_conditioning_rejects_latent_entrypoint() -> None:
    config_path = REPO_ROOT / "configs/experiments/post_decoded_robotwin_video_conditioned.yaml"
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

    try:
        pipeline.forward_train_from_latents(
            latent_batch.video_latents,
            train_batch,
            canonical_video=latent_batch.canonical_video,
            text_context=latent_batch.text_context,
            negative_text_context=latent_batch.negative_text_context,
        )
    except ValueError as exc:
        assert "rgb_video" in str(exc)
        assert "raw RGB views" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected rgb-video method-4 conditioning to reject latent-only entrypoints.")


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


@pytest.mark.parametrize("config_name", ["post_latent_robotwin.yaml", "post_decoded_robotwin.yaml"])
def test_method4_video_conditioned_decoder_runs_when_action_decoder_is_omitted(
    tmp_path: Path,
    config_name: str,
) -> None:
    source_path = REPO_ROOT / "configs/experiments" / config_name
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = f"implicit_{source_path.stem}_video_conditioned"
    raw.pop("action_decoder", None)

    config_path = tmp_path / f"{raw['name']}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config, pipeline, batch, train_batch = _build_pipeline(config_path)
    train_output = pipeline.forward_train(batch.views, train_batch)
    infer_output = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))

    assert train_output.decoder_output.action_pred.shape == (2, 6, config.action_decoder.action_dim)
    assert infer_output.decoder_output.action_pred.shape == (2, 6, config.action_decoder.action_dim)

    train_context = train_output.policy_output.decoder_sequence_context
    infer_context = infer_output.policy_output.decoder_sequence_context
    assert train_context is not None and train_context.video_condition_window is not None
    assert infer_context is not None and infer_context.video_condition_window is not None
    assert train_context.video_condition_window.local_window_tokens.shape[1] == 4
    assert infer_context.video_condition_window.local_window_tokens.shape[1] == 4
    assert train_context.video_condition_window.current_frame_index == 0
    assert infer_context.video_condition_window.current_frame_index == 0
    assert train_context.video_condition_window.current_action_index == 0
    assert infer_context.video_condition_window.current_action_index == 0
    assert train_context.video_condition_window.action_chunk_anchor_mode == "current_plus_future"
    assert infer_context.video_condition_window.action_chunk_anchor_mode == "current_plus_future"
    assert infer_context.video_condition_window.source_stage == "generated_future"
    assert infer_context.video_condition_window.metadata["source_family"] == "generated_future_video_tokens"
    assert infer_context.video_condition_window.metadata["uses_future_ground_truth"] is False
    assert infer_context.video_condition_window.metadata["observed_prefix_anchor"] == "start"
    assert infer_output.policy_output.aux["video_condition_source"] == "generated_future_video_tokens"
    assert infer_output.policy_output.aux["video_condition_uses_future_ground_truth"] is False
    assert infer_output.policy_output.aux["predicted_latents"].shape[2] == 3
    if config.policy_variant.video_condition_input_space == "rgb_video":
        assert train_context.video_condition_window.metadata["source_family"] == "encoded_rgb_frontend_video_tokens"
        assert train_output.visual_outputs.frontend.input_source == "canonical_rgb"
        assert infer_output.visual_outputs.frontend.input_source == "canonical_rgb"
    else:
        assert train_context.video_condition_window.metadata["source_family"] == "frontend_video_tokens"
    train_frontend_frame_tokens = tokens_to_frame_major(
        train_output.visual_outputs.frontend.video_tokens,
        train_output.visual_outputs.frontend.token_grid,
    )
    infer_frontend_frame_tokens = tokens_to_frame_major(
        infer_output.visual_outputs.frontend.video_tokens,
        infer_output.visual_outputs.frontend.token_grid,
    )
    assert torch.allclose(
        train_context.video_condition_window.local_window_tokens[:, 0],
        train_frontend_frame_tokens[:, 0],
    )
    assert torch.allclose(
        infer_context.video_condition_window.local_window_tokens[:, 0],
        infer_frontend_frame_tokens[:, 0],
    )
    assert train_output.decoder_output.aux["action_chunk_anchor_mode"] == "current_plus_future"
    assert infer_output.decoder_output.aux["action_chunk_anchor_mode"] == "current_plus_future"
    assert infer_output.decoder_output.aux["current_frame_index"].item() == 0.0
    assert infer_output.decoder_output.aux["current_action_index"].item() == 0.0


def test_method4_generated_video_condition_training_uses_predicted_latents(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "post_latent_video_conditioned_generated_train"
    raw["policy_variant"]["train_video_condition_source"] = "generated_future"
    raw["training"]["video_num_train_timesteps"] = 10
    raw["inference"]["video_num_inference_steps"] = 1

    config_path = tmp_path / "post_latent_video_conditioned_generated_train.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config, pipeline, batch, train_batch = _build_pipeline(config_path)
    generated_train_batch = PolicyTrainBatch(
        actions=train_batch.actions,
        action_mask=train_batch.action_mask,
        state=train_batch.state,
        extra={
            "metadata": (
                {"window_start_frame": 7, "sample_seed": 1234},
                {"window_start_frame": 7, "sample_seed": 1234},
            ),
        },
    )

    train_output = pipeline.forward_train(batch.views, generated_train_batch)

    assert config.policy_variant.train_video_condition_source == VideoConditionSource.GENERATED_FUTURE
    train_context = train_output.policy_output.decoder_sequence_context
    assert train_context is not None and train_context.video_condition_window is not None
    assert train_context.video_condition_window.source_stage == "generated_future"
    assert train_context.video_condition_window.metadata["source_family"] == "generated_future_video_tokens"
    assert train_context.video_condition_window.metadata["uses_future_ground_truth"] is False
    assert train_context.video_condition_window.metadata["frame_start"] == 7
    assert train_context.video_condition_window.metadata["observed_prefix_anchor"] == "start"
    assert train_context.video_condition_window.metadata["sample_seed"] == 1234
    assert train_context.video_condition_window.local_window_tokens.requires_grad is False
    assert train_output.policy_output.aux["video_condition_source"] == "generated_future_video_tokens"
    assert train_output.policy_output.aux["video_condition_uses_future_ground_truth"] is False
    assert train_output.policy_output.aux["predicted_latents"].shape[2] == 3


def test_generated_video_condition_frame_start_rejects_mixed_batch_offsets() -> None:
    batch = PolicyTrainBatch(
        actions=torch.zeros(2, 1, 1),
        extra={
            "metadata": (
                {"window_start_frame": 3},
                {"window_start_frame": 4},
            ),
        },
    )

    with pytest.raises(ValueError, match="share one absolute frame start"):
        resolve_video_condition_frame_start(batch)


def test_generated_video_condition_sample_seed_rejects_mixed_batch_seeds() -> None:
    batch = PolicyTrainBatch(
        actions=torch.zeros(2, 1, 1),
        extra={
            "metadata": (
                {"window_start_frame": 3, "sample_seed": 11},
                {"window_start_frame": 3, "sample_seed": 12},
            ),
        },
    )

    with pytest.raises(ValueError, match="share one deterministic conditioning seed"):
        resolve_video_condition_sample_seed(batch)


@pytest.mark.parametrize("config_name", ["post_latent_robotwin.yaml", "post_decoded_robotwin.yaml"])
def test_method4_generated_video_condition_infer_uses_provided_sample_seed(
    tmp_path: Path,
    config_name: str,
) -> None:
    source_path = REPO_ROOT / "configs/experiments" / config_name
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = f"seeded_{source_path.stem}_video_conditioned"
    raw.pop("action_decoder", None)

    config_path = tmp_path / f"{raw['name']}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    _, pipeline, batch, _ = _build_pipeline(config_path)
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(
            state=batch.state,
            extra={
                "video_condition_source": "generated_future",
                "video_condition_observed_prefix_anchor": "end",
                "video_condition_sample_seed": 1234,
            },
        ),
    )

    infer_context = infer_output.policy_output.decoder_sequence_context
    assert infer_context is not None and infer_context.video_condition_window is not None
    assert infer_context.video_condition_window.metadata["sample_seed"] == 1234
    assert infer_context.video_condition_window.metadata["observed_prefix_anchor"] == "end"
    infer_frontend_frame_tokens = tokens_to_frame_major(
        infer_output.visual_outputs.frontend.video_tokens,
        infer_output.visual_outputs.frontend.token_grid,
    )
    assert torch.allclose(
        infer_context.video_condition_window.local_window_tokens[:, 0],
        infer_frontend_frame_tokens[:, -1],
    )


def test_method4_video_conditioned_decoder_reuses_chunk_when_configured(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "post_latent_video_conditioned_chunk_reuse"
    raw["action_decoder"] = {
        "name": "video_conditioned_action_decoder",
        "rollout_chunk_steps": 2,
    }

    config_path = tmp_path / "post_latent_video_conditioned_chunk_reuse.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    _, pipeline, batch, _ = _build_pipeline(config_path)
    first = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))
    second = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state),
        infer_state=first.policy_output.next_state,
    )

    assert first.decoder_output.aux["sampled_new_chunk"] is True
    assert second.decoder_output.aux["sampled_new_chunk"] is False
    assert torch.allclose(second.decoder_output.action_pred, first.decoder_output.action_pred)
    assert second.decoder_output.aux["current_action_index"].item() == 1.0
    assert torch.allclose(
        second.decoder_output.aux["current_action"],
        first.decoder_output.action_pred[:, 1],
    )


def test_method4_current_frame_regression_direct_latent_train_path_uses_current_frame_only() -> None:
    config_path = REPO_ROOT / "configs/experiments/post_latent_libero_latent_local_current_frame_regression.yaml"
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

    assert train_output.visual_outputs is None
    assert train_output.policy_output.aux["direct_train_mode"] == "current_frame_regression"
    assert train_output.policy_output.policy_features.shape == (2, 0, config.action_decoder.hidden_size)
    assert train_output.decoder_output.action_pred.shape == (2, 1, config.action_decoder.action_dim)
    assert train_output.decoder_output.aux["video_condition_input_space"] == "video_latent"
    assert train_output.decoder_output.aux["current_action_index"].item() == 0.0


def test_method4_current_frame_regression_direct_rgb_train_path_uses_current_frame_only() -> None:
    config_path = REPO_ROOT / "configs/experiments/post_decoded_robotwin_current_frame_regression.yaml"
    config, pipeline, batch, train_batch = _build_pipeline(config_path)

    train_output = pipeline.forward_train(batch.views, train_batch)

    assert train_output.visual_outputs is None
    assert train_output.policy_output.aux["direct_train_mode"] == "current_frame_regression"
    assert train_output.policy_output.policy_features.shape == (2, 0, config.action_decoder.hidden_size)
    assert train_output.decoder_output.action_pred.shape == (2, 1, config.action_decoder.action_dim)
    assert train_output.decoder_output.aux["video_condition_input_space"] == "rgb_video"
    assert train_output.decoder_output.aux["current_action_index"].item() == 0.0


def test_method4_current_frame_regression_rejects_rollout_inference() -> None:
    config_path = REPO_ROOT / "configs/experiments/post_decoded_robotwin_current_frame_regression.yaml"
    _, pipeline, batch, _ = _build_pipeline(config_path)

    with pytest.raises(ValueError, match="train-only"):
        pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))


def test_post_latent_core_layer_visual_readout_pipeline_runs(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "post_latent_core_layer"
    raw["policy_variant"]["visual_readout"] = {
        "source_family": "core_layer_tokens",
        "layer_index": 0,
    }
    raw["backbone"]["num_layers"] = 2

    config_path = tmp_path / "post_latent_core_layer.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    _, pipeline, batch, train_batch = _build_pipeline(config_path)
    train_output = pipeline.forward_train(batch.views, train_batch)
    infer_output = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))

    assert train_output.policy_output.decoder_sequence_context is not None
    assert infer_output.policy_output.decoder_sequence_context is not None
    assert train_output.policy_output.decoder_sequence_context.source_stage == "core_layer_0"
    assert infer_output.policy_output.decoder_sequence_context.source_stage == "core_layer_0"
    assert train_output.policy_output.decoder_sequence_context.sequence_layout["source_family"] == "core_layer_tokens"


def test_post_decoded_multi_layer_visual_readout_pipeline_runs(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_decoded_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "post_decoded_multi_layer"
    raw["policy_variant"]["visual_readout"] = {
        "source_family": "core_multi_layer_tokens",
        "layer_indices": [0, 1],
        "fusion_mode": "concat_project",
    }
    raw["backbone"]["num_layers"] = 2

    config_path = tmp_path / "post_decoded_multi_layer.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    _, pipeline, batch, train_batch = _build_pipeline(config_path)
    train_output = pipeline.forward_train(batch.views, train_batch)
    infer_output = pipeline.forward_infer_step(batch.views, PolicyInferContext(state=batch.state))

    assert train_output.policy_output.decoder_sequence_context is not None
    assert infer_output.policy_output.decoder_sequence_context is not None
    assert train_output.policy_output.decoder_sequence_context.source_stage == "core_multi_layer"
    assert infer_output.policy_output.decoder_sequence_context.source_stage == "core_multi_layer"
    assert train_output.policy_output.decoder_sequence_context.sequence_layout["source_family"] == "core_multi_layer_tokens"


def test_causal_video_prediction_pipeline_trains_from_latents(tmp_path: Path) -> None:
    del tmp_path
    config_path = REPO_ROOT / "configs/experiments/causal_video_prediction_robotwin_smoke.yaml"
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

    assert train_output.decoder_output.action_pred.shape == (2, 0, config.action_decoder.action_dim)
    assert "latent_mse" in train_output.decoder_output.metrics
    assert "predicted_latents" in train_output.decoder_output.aux

    infer_batch = build_synthetic_latent_batch(config.data, batch_size=1)
    infer_output = pipeline.forward_infer_step_from_latents(
        infer_batch.video_latents,
        PolicyInferContext(
            state=infer_batch.state,
            extra={
                "task_text": infer_batch.task_text,
                "metadata": infer_batch.metadata,
            },
        ),
        canonical_video=infer_batch.canonical_video,
        text_context=infer_batch.text_context,
        negative_text_context=infer_batch.negative_text_context,
    )

    assert infer_output.decoder_output.action_pred.shape == (1, 0, config.action_decoder.action_dim)
    assert "predicted_latents" in infer_output.decoder_output.aux
