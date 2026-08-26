from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from open_wam.configs import (
    CausalVideoPredictionPolicyConfig,
    InferenceConfig,
    TextConditioningMode,
    TrainingConfig,
    load_experiment_config,
)
from open_wam.data import LatentWAMBatch
from open_wam.evals.video_prediction import rollout_causal_video_prediction
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.models.policy_variants.causal_video_prediction import (
    CausalVideoPredictionPolicyVariant,
)
from open_wam.models.policy_variants.video_flow_artifacts import (
    VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
    VideoFlowTrainArtifacts,
)
from open_wam.models.video_backbone.contracts import (
    ChunkMetadata,
    ConditioningState,
    TokenGridMetadata,
)
from open_wam.models.visual_tower import VisualFrontendOutput, VisualStageOutputs
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.training import (
    LatentBatchAdapter,
    PipelineTrainStepExecutor,
    apply_training_component_controls,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAUSAL_VIDEO_GOLDEN_ROOT = REPO_ROOT / "tests" / "fixtures" / "causal_video_prediction"
TRAINING_STEP_GOLDEN = CAUSAL_VIDEO_GOLDEN_ROOT / "training_step_v1.safetensors"
MULTICHUNK_ROLLOUT_GOLDEN = (
    CAUSAL_VIDEO_GOLDEN_ROOT / "multichunk_rollout_v1.safetensors"
)


def test_task_prompt_causal_video_prediction_rejects_missing_conditioning() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(
            text_conditioning_mode=TextConditioningMode.TASK_PROMPT
        ),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
    )

    with torch.no_grad(), pytest.raises(ValueError, match="non-empty task instruction"):
        variant._validate_text_conditioning(
            text_context=torch.ones(1, 2, 3),
            task_text=("",),
            batch_size=1,
        )
    with torch.no_grad(), pytest.raises(ValueError, match="requires task-prompt text"):
        variant._validate_text_conditioning(
            text_context=None,
            task_text=("move the object",),
            batch_size=1,
        )
    with torch.no_grad(), pytest.raises(ValueError, match="all-zero text embedding"):
        variant._validate_text_conditioning(
            text_context=torch.zeros(1, 2, 3),
            task_text=("move the object",),
            batch_size=1,
        )

    variant._validate_text_conditioning(
        text_context=torch.ones(1, 2, 3),
        task_text=("move the object",),
        batch_size=1,
    )


def test_causal_video_cfg_requires_shape_matched_finite_negative_text() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(
            text_conditioning_mode=TextConditioningMode.TASK_PROMPT
        ),
        training_config=TrainingConfig(text_condition_dropout_prob=0.1),
        inference_config=InferenceConfig(guidance_scale=2.0),
    )
    positive = torch.ones(1, 2, 3)

    with pytest.raises(ValueError, match="requires negative text embeddings"):
        variant._validate_text_conditioning(
            text_context=positive,
            task_text=("move the object",),
            batch_size=1,
            require_negative_text=True,
        )
    with pytest.raises(ValueError, match="identical shapes"):
        variant._validate_text_conditioning(
            text_context=positive,
            negative_text_context=torch.zeros(1, 1, 3),
            task_text=("move the object",),
            batch_size=1,
            require_negative_text=True,
        )
    with pytest.raises(ValueError, match="finite values"):
        variant._validate_text_conditioning(
            text_context=positive,
            negative_text_context=torch.full_like(positive, float("nan")),
            task_text=("move the object",),
            batch_size=1,
            require_negative_text=True,
        )
    with pytest.raises(ValueError, match="all-zero text embedding"):
        variant._validate_text_conditioning(
            text_context=positive,
            negative_text_context=torch.zeros_like(positive),
            task_text=("move the object",),
            batch_size=1,
            require_negative_text=True,
        )

    variant._validate_text_conditioning(
        text_context=positive,
        negative_text_context=torch.full_like(positive, -0.25),
        task_text=("move the object",),
        batch_size=1,
        require_negative_text=True,
    )


def test_causal_video_prediction_maps_raw_wan_windows_to_latent_layouts() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
    )

    layouts = variant._resolve_layouts(
        metadata=(
            {
                "observed_prefix_frames": 2,
                "future_suffix_frames": 6,
                "valid_video_frames": 8,
                "padded_video_frames": 16,
            },
        ),
        available_frames=5,
        frame_mapping={
            "kind": "wan_temporal_downsample",
            "raw_frames": 16,
            "latent_frames": 4,
        },
    )

    assert len(layouts) == 1
    assert layouts[0].observed_frames == 1
    assert layouts[0].future_frames == 1
    assert layouts[0].total_frames == 2


def test_causal_video_prediction_keeps_latent_layouts_in_identity_mapping() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
    )

    layouts = variant._resolve_layouts(
        metadata=(
            {
                "observed_prefix_frames": 2,
                "future_suffix_frames": 6,
                "valid_video_frames": 8,
            },
        ),
        available_frames=8,
        frame_mapping={"kind": "identity", "raw_frames": 8, "latent_frames": 8},
    )

    assert len(layouts) == 1
    assert layouts[0].observed_frames == 2
    assert layouts[0].future_frames == 6
    assert layouts[0].total_frames == 8


class _CaptureVideoFlowTower:
    def __init__(self) -> None:
        self.attention_mask: torch.Tensor | None = None

    def predict_video_flow(self, **kwargs):
        self.attention_mask = kwargs.get("attention_mask")
        return torch.zeros_like(kwargs["noisy_latents"])


def test_causal_video_prediction_masks_padded_tokens_during_train_rollout() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(),
        training_config=TrainingConfig(video_num_train_timesteps=8),
        inference_config=InferenceConfig(),
    )
    video_latents = torch.randn(2, 48, 6, 2, 2)
    token_grid = TokenGridMetadata(
        num_frames=6,
        latent_height=2,
        latent_width=2,
        patch_size=(1, 2, 2),
        patches_per_frame_h=1,
        patches_per_frame_w=1,
        tokens_per_frame=1,
        sequence_length=6,
    )
    frontend = VisualFrontendOutput(
        canonical_video=torch.zeros(2, 3, 6, 32, 32),
        video_latents=video_latents,
        video_tokens=torch.zeros(2, 6, 4),
        input_source="video_latents",
        token_grid=token_grid,
        chunk=ChunkMetadata(
            chunk_start_frame=0,
            chunk_num_frames=6,
            frame_stride=1,
            chunk_type="dense_video_chunk",
        ),
        conditioning=ConditioningState(
            supported=True,
            text_context=torch.zeros(2, 4, 16),
            metadata={},
        ),
    )
    tower = _CaptureVideoFlowTower()

    rollout = variant._build_train_rollout(
        visual_tower=tower,  # type: ignore[arg-type]
        visual_outputs=VisualStageOutputs(frontend=frontend),
        text_context=frontend.conditioning.text_context,
        metadata=(
            {
                "observed_prefix_frames": 2,
                "future_suffix_frames": 2,
                "valid_video_frames": 4,
                "padded_video_frames": 6,
            },
            {
                "observed_prefix_frames": 1,
                "future_suffix_frames": 5,
                "valid_video_frames": 6,
            },
        ),
    )

    assert tower.attention_mask is not None
    assert tower.attention_mask.shape == (2, 6, 6)
    assert torch.all(tower.attention_mask[0, :, :4])
    assert not torch.any(tower.attention_mask[0, :, 4:])
    assert torch.all(tower.attention_mask[1])
    future_loss_mask = rollout["future_loss_mask"]
    assert torch.all(future_loss_mask[0, :, 2:4])
    assert not torch.any(future_loss_mask[0, :, 4:])


def _tiny_causal_video_pipeline(
    *,
    text_conditioning_mode: TextConditioningMode = TextConditioningMode.TASK_PROMPT,
    text_condition_dropout_prob: float = 0.0,
):
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )
    config = replace(
        config,
        backbone=replace(
            config.backbone,
            pretrained_model_name_or_path=None,
            load_reference_core_weights=False,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            attention_head_dim=8,
            ffn_dim=32,
            text_dim=8,
            freq_dim=8,
        ),
        policy_variant=replace(
            config.policy_variant,
            hidden_size=16,
            text_conditioning_mode=text_conditioning_mode,
        ),
        action_decoder=replace(config.action_decoder, hidden_size=16),
        training=replace(
            config.training,
            video_num_train_timesteps=8,
            text_condition_dropout_prob=text_condition_dropout_prob,
        ),
        inference=replace(
            config.inference,
            video_num_inference_steps=2,
            # The immutable numerical fixture predates the maintained CFG preset.
            guidance_scale=1.0,
        ),
        trainer=replace(
            config.trainer,
            strategy="single_device",
            accelerator="cpu",
            precision="32-true",
        ),
    )
    torch.manual_seed(1337)
    pipeline = build_variant_pipeline_from_config(config)
    report = apply_training_component_controls(pipeline, config.training)
    return config, pipeline, report


def _causal_video_inputs() -> tuple[torch.Tensor, torch.Tensor, PolicyTrainBatch]:
    latents = torch.linspace(
        -1.0,
        1.0,
        1 * 48 * 5 * 2 * 4,
        dtype=torch.float32,
    ).reshape(1, 48, 5, 2, 4)
    text_context = torch.linspace(0.1, 0.8, 1 * 3 * 8).reshape(1, 3, 8)
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 0, 7),
        action_mask=torch.zeros(1, 0, 7),
        state=torch.zeros(1, 0, 8),
        extra={
            "task_text": ("move object",),
            "metadata": (
                {
                    "observed_prefix_frames": 2,
                    "future_suffix_frames": 3,
                    "valid_video_frames": 5,
                    "padded_video_frames": 5,
                },
            ),
            "state_mask": torch.zeros(1, 0, 8),
        },
    )
    return latents, text_context, batch


def _assert_tensor_golden(
    path: Path,
    actual: dict[str, torch.Tensor],
    *,
    metadata: dict[str, str],
) -> None:
    expected = load_file(path, device="cpu")
    assert set(actual) == set(expected)
    for name in sorted(expected):
        actual_tensor = actual[name].detach().cpu()
        expected_tensor = expected[name]
        assert actual_tensor.dtype == expected_tensor.dtype, name
        assert tuple(actual_tensor.shape) == tuple(expected_tensor.shape), name
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=1e-5,
            atol=2e-6,
            msg=lambda message, tensor_name=name: f"{tensor_name}: {message}",
        )
    with safe_open(path, framework="pt", device="cpu") as handle:
        assert handle.metadata() == metadata


@contextmanager
def _deterministic_cpu_math():
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)


def test_causal_video_training_step_has_strict_numerical_gradient_parity() -> None:
    with _deterministic_cpu_math():
        _, pipeline, report = _tiny_causal_video_pipeline()
        latents, text_context, batch = _causal_video_inputs()

        torch.manual_seed(4242)
        output = pipeline.forward_train_from_latents(
            latents,
            batch,
            text_context=text_context,
            negative_text_context=torch.zeros_like(text_context),
        )
        loss = output.decoder_output.loss
        loss.backward()
        trainable = [
            (name, parameter)
            for name, parameter in pipeline.named_parameters()
            if parameter.requires_grad
        ]
        gradients = [
            (name, parameter.grad)
            for name, parameter in trainable
            if parameter.grad is not None
        ]
        optimizer = torch.optim.SGD(
            (parameter for _, parameter in trainable),
            lr=1e-3,
        )
        optimizer.step()

    assert output.policy_output.decoder_artifacts is not None
    train_artifacts = output.policy_output.decoder_artifacts.require(
        contract=VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
        payload_type=VideoFlowTrainArtifacts,
    )
    torch.testing.assert_close(
        train_artifacts.predicted_latents.detach(),
        output.decoder_output.aux["predicted_latents"],
        rtol=0.0,
        atol=0.0,
    )
    assert not {
        "flow_pred",
        "flow_targets",
        "predicted_latents",
        "target_latents",
        "timesteps",
        "scheduler",
        "future_loss_mask",
    }.intersection(output.policy_output.aux)

    parity_tensors = {
        "loss.total": loss.detach(),
        "prediction.predicted_latents": output.decoder_output.aux[
            "predicted_latents"
        ].detach(),
    }
    parity_tensors.update(
        (f"gradient.{name}", gradient.detach()) for name, gradient in gradients
    )
    parity_tensors.update(
        (f"updated_parameter.{name}", parameter.detach())
        for name, parameter in trainable
    )

    assert len(trainable) == 42
    assert len(gradients) == 42
    assert [name for name, _ in gradients] == [name for name, _ in trainable]
    assert report.trainable_parameters == 12_288
    assert loss.item() == pytest.approx(3.283013105392456, rel=1e-7, abs=1e-7)
    assert (
        hashlib.sha256(
            "\n".join(sorted(name for name, _ in trainable)).encode("utf-8")
        ).hexdigest()
        == "a6ac5260b51dbc60decb0c8931faa8e32488bd25c294a65e367e114538c46be0"
    )
    _assert_tensor_golden(
        TRAINING_STEP_GOLDEN,
        parity_tensors,
        metadata={
            "schema_version": "open_wam.causal_video_training_step.v1",
            "model_seed": "1337",
            "forward_seed": "4242",
            "optimizer": "SGD(lr=0.001)",
            "torch_num_threads": "1",
        },
    )


def test_causal_video_text_dropout_uses_blank_text_and_preserves_source() -> None:
    with _deterministic_cpu_math():
        config, pipeline, _ = _tiny_causal_video_pipeline(
            text_condition_dropout_prob=0.5
        )
        latents, text_context, policy_batch = _causal_video_inputs()
        blank_text = torch.full_like(text_context, -0.25)
        latent_batch = LatentWAMBatch(
            video_latents=latents,
            actions=policy_batch.actions,
            action_mask=policy_batch.action_mask,
            state=policy_batch.state,
            state_mask=policy_batch.extra["state_mask"],
            task_text=policy_batch.extra["task_text"],
            text_context=text_context,
            negative_text_context=blank_text,
            metadata=policy_batch.extra["metadata"],
        )
        executor = PipelineTrainStepExecutor(
            pipeline=pipeline,
            batch_adapter=LatentBatchAdapter(),
            training_config=config.training,
        )

        torch.manual_seed(0)
        prepared = executor.batch_adapter.prepare(latent_batch)
        dropped = executor._apply_text_condition_dropout(prepared)
        torch.testing.assert_close(
            dropped.policy_batch.source_text_context,
            text_context,
        )
        assert dropped.text_context is not None
        torch.testing.assert_close(dropped.text_context, blank_text)

        torch.manual_seed(0)
        result = executor.forward_train(latent_batch)
        assert torch.isfinite(result.loss)

        missing_blank_batch = replace(latent_batch, negative_text_context=None)
        with pytest.raises(ValueError, match="requires negative text embeddings"):
            executor.forward_train(missing_blank_batch)

        invalid_batch = replace(
            latent_batch,
            text_context=torch.zeros_like(text_context),
        )
        with pytest.raises(ValueError, match="all-zero text embedding"):
            executor.forward_train(invalid_batch)


def test_causal_video_training_requires_one_metadata_row_per_sample() -> None:
    with _deterministic_cpu_math():
        _, pipeline, _ = _tiny_causal_video_pipeline()
        latents, text_context, policy_batch = _causal_video_inputs()
        batch_size = 2
        batched_latents = latents.expand(batch_size, -1, -1, -1, -1).clone()
        batched_text = text_context.expand(batch_size, -1, -1).clone()
        batched_policy = replace(
            policy_batch,
            actions=policy_batch.actions.expand(batch_size, -1, -1).clone(),
            action_mask=policy_batch.action_mask.expand(batch_size, -1, -1).clone(),
            state=policy_batch.state.expand(batch_size, -1, -1).clone(),
            extra={
                **policy_batch.extra,
                "task_text": ("move object",) * batch_size,
                "state_mask": policy_batch.extra["state_mask"]
                .expand(batch_size, -1, -1)
                .clone(),
            },
        )

        with pytest.raises(ValueError, match="metadata cardinality"):
            pipeline.forward_train_from_latents(
                batched_latents,
                batched_policy,
                text_context=batched_text,
                negative_text_context=torch.zeros_like(batched_text),
            )

        valid_policy = replace(
            batched_policy,
            extra={
                **batched_policy.extra,
                "metadata": policy_batch.extra["metadata"] * batch_size,
            },
        )
        torch.manual_seed(4242)
        output = pipeline.forward_train_from_latents(
            batched_latents,
            valid_policy,
            text_context=batched_text,
            negative_text_context=torch.zeros_like(batched_text),
        )

    assert torch.isfinite(output.decoder_output.loss)


def test_causal_video_inference_rejects_batched_latents_explicitly() -> None:
    _, pipeline, _ = _tiny_causal_video_pipeline()
    latents, text_context, policy_batch = _causal_video_inputs()
    batch_size = 2
    batched_latents = latents.expand(batch_size, -1, -1, -1, -1).clone()
    batched_text = text_context.expand(batch_size, -1, -1).clone()
    context = PolicyInferContext(
        extra={
            "task_text": ("move object",) * batch_size,
            "metadata": policy_batch.extra["metadata"] * batch_size,
        }
    )

    with pytest.raises(ValueError, match="supports batch size 1; got 2"):
        pipeline.forward_infer_step_from_latents(
            batched_latents,
            context,
            text_context=batched_text,
            negative_text_context=torch.zeros_like(batched_text),
        )


def test_disabled_causal_video_conditioning_uses_blank_text_for_train_and_infer() -> None:
    with _deterministic_cpu_math():
        _, pipeline, _ = _tiny_causal_video_pipeline(
            text_conditioning_mode=TextConditioningMode.DISABLED
        )
        latents, positive_text, policy_batch = _causal_video_inputs()
        positive_text = torch.full_like(positive_text, float("nan"))
        blank_text = torch.full_like(positive_text, -0.25)
        policy_batch = replace(
            policy_batch,
            extra={**policy_batch.extra, "task_text": (None,)},
        )

        torch.manual_seed(4242)
        train_output = pipeline.forward_train_from_latents(
            latents,
            policy_batch,
            text_context=positive_text,
            negative_text_context=blank_text,
        )
        train_conditioning = train_output.visual_outputs.frontend.conditioning
        torch.testing.assert_close(
            train_conditioning.text_context,
            blank_text,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            train_conditioning.negative_text_context,
            blank_text,
            rtol=0.0,
            atol=0.0,
        )
        assert torch.isfinite(train_output.decoder_output.loss)

        rollout_batch = LatentWAMBatch(
            video_latents=latents,
            actions=policy_batch.actions,
            action_mask=policy_batch.action_mask,
            state=policy_batch.state,
            state_mask=policy_batch.extra["state_mask"],
            task_text=(None,),
            text_context=positive_text * 2.0,
            negative_text_context=blank_text,
            metadata=policy_batch.extra["metadata"],
        )
        torch.manual_seed(9001)
        rollout = rollout_causal_video_prediction(
            pipeline,
            rollout_batch,
            num_chunks=1,
        )

    assert torch.isfinite(rollout.predicted_latents).all()


def test_causal_video_multichunk_rollout_preserves_full_generated_context() -> None:
    with _deterministic_cpu_math():
        _, pipeline, _ = _tiny_causal_video_pipeline()
        latents, text_context, _ = _causal_video_inputs()
        batch = LatentWAMBatch(
            video_latents=latents,
            actions=torch.zeros(1, 0, 7),
            action_mask=torch.zeros(1, 0, 7),
            state=torch.zeros(1, 0, 8),
            state_mask=torch.zeros(1, 0, 8),
            task_text=("move object",),
            text_context=text_context,
            negative_text_context=torch.zeros_like(text_context),
            metadata=(
                {
                    "observed_prefix_frames": 2,
                    "future_suffix_frames": 3,
                    "valid_video_frames": 5,
                },
            ),
        )

        torch.manual_seed(9001)
        result = rollout_causal_video_prediction(pipeline, batch, num_chunks=2)

    assert result.context_latent_frames == (2, 5, 8)
    torch.testing.assert_close(
        result.predicted_latents[:, :, :2],
        latents[:, :, :2],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(result.predicted_latents).all()
    _assert_tensor_golden(
        MULTICHUNK_ROLLOUT_GOLDEN,
        {
            "prediction.predicted_latents": result.predicted_latents,
            "metric.first_chunk_future_mse": torch.tensor(
                result.first_chunk_future_mse
            ),
        },
        metadata={
            "schema_version": "open_wam.causal_video_multichunk_rollout.v1",
            "model_seed": "1337",
            "rollout_seed": "9001",
            "num_chunks": "2",
            "torch_num_threads": "1",
        },
    )
