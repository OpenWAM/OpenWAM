from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from open_wam.configs import (
    CausalVideoPredictionPolicyConfig,
    CausalVideoProgram,
    InferenceConfig,
    TextConditioningMode,
    TrainingConfig,
    load_experiment_config,
)
from open_wam.data import LatentWAMBatch
from open_wam.evals.video_prediction import rollout_causal_video_prediction
from open_wam.models.policy_variants import (
    PolicyExecutionCommit,
    PolicyInferContext,
    PolicyInferenceOutputRequest,
    PolicyInferState,
    PolicyObservedHistory,
    PolicyOutputModality,
    PolicyPreparedInputs,
    PolicyRecurrentHistoryPolicy,
    PolicyTemporalSpan,
    PolicyTrainBatch,
    PolicyVideoGenerationRequest,
)
from open_wam.models.policy_variants.causal_video_prediction import (
    CausalVideoPredictionPolicyVariant,
)
from open_wam.models.policy_variants.observed_video_history import (
    ObservedVideoHistoryState,
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
            program=CausalVideoProgram.PREFIX_SUFFIX,
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
            program=CausalVideoProgram.PREFIX_SUFFIX,
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
        config=CausalVideoPredictionPolicyConfig(program=CausalVideoProgram.PREFIX_SUFFIX),
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
        config=CausalVideoPredictionPolicyConfig(program=CausalVideoProgram.PREFIX_SUFFIX),
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


def test_causal_video_inference_publishes_only_future_frames_for_composition() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(
            program=CausalVideoProgram.PREFIX_SUFFIX
        ),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
    )
    latents = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1, 1)
    generated_future = torch.full((1, 1, 3, 1, 1), 17.0)

    class _Tower:
        def generate_conditioned_future_latents(self, **kwargs):
            assert kwargs["observed_prefix"].shape[2] == 2
            assert kwargs["future_template"].shape[2] == 3
            return generated_future

    conditioning = SimpleNamespace(
        text_context=torch.ones(1, 2, 3),
        negative_text_context=None,
        metadata={
            "video_frame_mapping": {
                "kind": "identity",
                "raw_frames": 5,
                "latent_frames": 5,
            }
        },
    )
    visual_outputs = SimpleNamespace(
        frontend=SimpleNamespace(
            video_latents=latents,
            conditioning=conditioning,
            latent_space_identity=None,
        )
    )
    context = PolicyInferContext(
        output_request=PolicyInferenceOutputRequest.video_only(),
        extra={
            "task_text": ("move the object",),
            "metadata": (
                {
                    "observed_prefix_frames": 2,
                    "future_suffix_frames": 3,
                    "valid_video_frames": 5,
                },
            ),
        },
    )

    output = variant.forward_infer_step(
        _Tower(),  # type: ignore[arg-type]
        visual_outputs,  # type: ignore[arg-type]
        context,
        PolicyInferState(),
    )

    assert variant.inference_capabilities.native_modalities == frozenset(
        {PolicyOutputModality.VIDEO}
    )
    assert output.generated_video is not None
    torch.testing.assert_close(output.generated_video.latents, generated_future)
    artifacts = output.decoder_artifacts.payload
    assert artifacts.predicted_latents.shape[2] == 5
    torch.testing.assert_close(artifacts.predicted_latents[:, :, :2], latents[:, :, :2])


def test_causal_video_composition_synthesizes_future_template_from_one_observation() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(
            program=CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO,
            noisy_video_condition_prob=0.0,
        ),
        training_config=TrainingConfig(window_size=8),
        inference_config=InferenceConfig(frame_chunk_size=4),
    )
    observed = torch.randn(1, 3, 1, 2, 2)
    generated = torch.randn(1, 3, 4, 2, 2)

    class _Tower:
        def generate_chunked_conditioned_video_latents(self, **kwargs):
            torch.testing.assert_close(kwargs["observed_history"], observed)
            assert kwargs["future_template"].shape == generated.shape
            assert bool((kwargs["future_template"] == 0).all())
            assert kwargs["history_frame_start"] == 0
            assert kwargs["chunk_origin_frame"] == 0
            assert kwargs["window_size"] == 30
            return generated

    visual_outputs = SimpleNamespace(
        frontend=SimpleNamespace(
            video_latents=observed,
            conditioning=SimpleNamespace(
                text_context=torch.ones(1, 2, 3),
                negative_text_context=None,
                metadata={},
            ),
            latent_space_identity=None,
        )
    )
    output = variant.forward_infer_step(
        _Tower(),  # type: ignore[arg-type]
        visual_outputs,  # type: ignore[arg-type]
        PolicyInferContext(
            extra={"task_text": ("move the object",)},
            video_generation=PolicyVideoGenerationRequest(
                frame_count=4,
                attention_window_size=30,
            ),
        ),
        PolicyInferState(),
    )

    assert output.generated_video is not None
    assert output.generated_video.frame_start == 1
    assert output.generation_frame_start == 1
    torch.testing.assert_close(output.generated_video.latents, generated)
    assert output.decoder_artifacts.payload.predicted_latents.shape[2] == 5
    assert (
        variant.inference_capabilities.recurrent_history_policy
        is PolicyRecurrentHistoryPolicy.EXPLICIT_RECONCILIATION
    )

    next_observed = torch.randn(1, 3, 4, 2, 2)

    with pytest.raises(RuntimeError, match="reconciled"):
        variant.forward_infer_step(
            _Tower(),  # type: ignore[arg-type]
            visual_outputs,  # type: ignore[arg-type]
            PolicyInferContext(
                extra={"task_text": ("move the object",)},
                video_generation=PolicyVideoGenerationRequest(
                    frame_count=4,
                    attention_window_size=30,
                ),
            ),
            output.next_state,
        )

    history_output = variant.reconcile_observed_history(
        PolicyObservedHistory(
            video_latents=next_observed,
            observation_frame_count=16,
            inference_window_size=30,
            rollout_frame_chunk_size=4,
            execution_commit=PolicyExecutionCommit(
                speculative_span=PolicyTemporalSpan(start_frame=1, frame_count=4),
                executed_frame_count=4,
            ),
        ),
        output.next_state,
    )
    assert history_output.applied is True
    assert history_output.next_state is not None

    class _NextTower:
        def generate_chunked_conditioned_video_latents(self, **kwargs):
            torch.testing.assert_close(
                kwargs["observed_history"],
                torch.cat([observed, next_observed], dim=2),
            )
            assert kwargs["history_frame_start"] == 0
            assert kwargs["chunk_origin_frame"] == 0
            assert kwargs["window_size"] == 30
            return generated

    next_visual_outputs = SimpleNamespace(
        frontend=SimpleNamespace(
            video_latents=next_observed,
            conditioning=visual_outputs.frontend.conditioning,
            latent_space_identity=None,
        )
    )
    next_output = variant.forward_infer_step(
        _NextTower(),  # type: ignore[arg-type]
        next_visual_outputs,  # type: ignore[arg-type]
        PolicyInferContext(
            extra={"task_text": ("move the object",)},
            video_generation=PolicyVideoGenerationRequest(
                frame_count=4,
                attention_window_size=30,
            ),
        ),
        history_output.next_state,
    )

    assert next_output.generated_video is not None
    assert next_output.generated_video.frame_start == 5
    assert next_output.generation_frame_start == 5
    assert next_output.next_state.cursor.current_start_frame == 5


def test_causal_video_composition_reconciles_repeated_short_requests() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(
            program=CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO,
            noisy_video_condition_prob=0.0,
        ),
        training_config=TrainingConfig(window_size=30),
        inference_config=InferenceConfig(frame_chunk_size=4),
    )
    initial_observed = torch.randn(1, 3, 1, 2, 2)
    next_observed = torch.randn(1, 3, 2, 2, 2)
    generated = torch.randn(1, 3, 2, 2, 2)
    tower_calls: list[dict[str, object]] = []

    class _Tower:
        def generate_chunked_conditioned_video_latents(self, **kwargs):
            tower_calls.append(kwargs)
            assert kwargs["future_template"].shape[2] == 2
            assert kwargs["chunk_size"] == 4
            return generated

    conditioning = SimpleNamespace(
        text_context=torch.ones(1, 2, 3),
        negative_text_context=None,
        metadata={},
    )

    def visual_outputs(video_latents: torch.Tensor):
        return SimpleNamespace(
            frontend=SimpleNamespace(
                video_latents=video_latents,
                conditioning=conditioning,
                latent_space_identity=None,
            )
        )

    context = PolicyInferContext(
        extra={"task_text": ("move the object",)},
        video_generation=PolicyVideoGenerationRequest(
            frame_count=2,
            attention_window_size=30,
        ),
    )
    first_output = variant.forward_infer_step(
        _Tower(),  # type: ignore[arg-type]
        visual_outputs(initial_observed),  # type: ignore[arg-type]
        context,
        PolicyInferState(),
    )
    assert first_output.generated_video is not None
    assert first_output.generated_video.frame_start == 1
    assert first_output.next_state.cursor.chunk_size == 2
    assert isinstance(
        first_output.next_state.variant_state,
        ObservedVideoHistoryState,
    )
    assert first_output.next_state.variant_state.model_frame_chunk_size == 4

    reconciled = variant.reconcile_observed_history(
        PolicyObservedHistory(
            video_latents=next_observed,
            observation_frame_count=8,
            inference_window_size=30,
            rollout_frame_chunk_size=2,
            execution_commit=PolicyExecutionCommit(
                speculative_span=PolicyTemporalSpan(start_frame=1, frame_count=2),
                executed_frame_count=2,
            ),
        ),
        first_output.next_state,
    )
    assert reconciled.applied is True
    assert reconciled.next_state is not None

    second_output = variant.forward_infer_step(
        _Tower(),  # type: ignore[arg-type]
        visual_outputs(next_observed),  # type: ignore[arg-type]
        context,
        reconciled.next_state,
    )
    assert second_output.generated_video is not None
    assert second_output.generated_video.frame_start == 3
    assert second_output.next_state.cursor.chunk_size == 2
    torch.testing.assert_close(
        tower_calls[1]["observed_history"],
        torch.cat([initial_observed, next_observed], dim=2),
    )


def test_native_chunked_causal_inference_honors_reconciliation_capability() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(
            program=CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO,
            noisy_video_condition_prob=0.0,
        ),
        training_config=TrainingConfig(window_size=30),
        inference_config=InferenceConfig(frame_chunk_size=4),
    )
    first_prefix = torch.randn(1, 3, 1, 2, 2)
    next_observed = torch.randn(1, 3, 4, 2, 2)
    generated = torch.randn(1, 3, 4, 2, 2)
    calls: list[dict[str, object]] = []

    class _Tower:
        def generate_chunked_conditioned_video_latents(self, **kwargs):
            calls.append(kwargs)
            return generated

    def visual_outputs(video_latents: torch.Tensor):
        return SimpleNamespace(
            frontend=SimpleNamespace(
                video_latents=video_latents,
                conditioning=SimpleNamespace(
                    text_context=torch.ones(1, 2, 3),
                    negative_text_context=None,
                    metadata={},
                ),
                latent_space_identity=None,
            )
        )

    context = PolicyInferContext(
        extra={
            "task_text": ("move the object",),
            "metadata": (
                {
                    "observed_prefix_frames": 1,
                    "future_suffix_frames": 4,
                    "frame_shift": 0,
                    "chunk_origin_frame": 0,
                },
            ),
        }
    )
    first_output = variant.forward_infer_step(
        _Tower(),  # type: ignore[arg-type]
        visual_outputs(
            torch.cat([first_prefix, torch.zeros_like(generated)], dim=2)
        ),  # type: ignore[arg-type]
        context,
        PolicyInferState(),
    )

    assert first_output.generation_frame_start == 1
    assert isinstance(
        first_output.next_state.variant_state,
        ObservedVideoHistoryState,
    )
    assert calls[0]["history_frame_start"] == -1
    torch.testing.assert_close(calls[0]["observed_history"], first_prefix)

    reconciled = variant.reconcile_observed_history(
        PolicyObservedHistory(
            video_latents=next_observed,
            observation_frame_count=16,
            inference_window_size=30,
            rollout_frame_chunk_size=4,
            execution_commit=PolicyExecutionCommit(
                speculative_span=PolicyTemporalSpan(start_frame=1, frame_count=4),
                executed_frame_count=4,
            ),
        ),
        first_output.next_state,
    )
    assert reconciled.next_state is not None

    second_output = variant.forward_infer_step(
        _Tower(),  # type: ignore[arg-type]
        visual_outputs(
            torch.cat([next_observed[:, :, -1:], torch.zeros_like(generated)], dim=2)
        ),  # type: ignore[arg-type]
        context,
        reconciled.next_state,
    )

    assert second_output.generation_frame_start == 5
    assert calls[1]["history_frame_start"] == -1
    torch.testing.assert_close(
        calls[1]["observed_history"],
        torch.cat([first_prefix, next_observed], dim=2),
    )


def test_prefix_suffix_composition_advances_from_each_real_observation_chunk() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(
            program=CausalVideoProgram.PREFIX_SUFFIX
        ),
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(),
    )
    generated = torch.randn(1, 3, 4, 2, 2)
    observed_chunks = (
        torch.randn(1, 3, 1, 2, 2),
        torch.randn(1, 3, 4, 2, 2),
    )
    expected_frame_starts = (0, 1)

    class _Tower:
        def __init__(self, expected_frame_start: int) -> None:
            self.expected_frame_start = expected_frame_start

        def generate_conditioned_future_latents(self, **kwargs):
            assert kwargs["frame_start"] == self.expected_frame_start
            assert kwargs["future_template"].shape == generated.shape
            return generated

    state = PolicyInferState()
    generated_frame_starts = []
    for observed, expected_frame_start in zip(
        observed_chunks, expected_frame_starts, strict=True
    ):
        visual_outputs = SimpleNamespace(
            frontend=SimpleNamespace(
                video_latents=observed,
                conditioning=SimpleNamespace(
                    text_context=torch.ones(1, 2, 3),
                    negative_text_context=None,
                    metadata={},
                ),
                latent_space_identity=None,
            )
        )
        output = variant.forward_infer_step(
            _Tower(expected_frame_start),  # type: ignore[arg-type]
            visual_outputs,  # type: ignore[arg-type]
            PolicyInferContext(
                extra={"task_text": ("move the object",)},
                video_generation=PolicyVideoGenerationRequest(frame_count=4),
            ),
            state,
        )
        assert output.generated_video is not None
        generated_frame_starts.append(output.generated_video.frame_start)
        state = output.next_state

    assert generated_frame_starts == [1, 5]
    assert state.cursor.current_start_frame == 5


class _CaptureVideoFlowTower:
    def __init__(self) -> None:
        self.attention_mask: torch.Tensor | None = None

    def predict_video_flow(self, **kwargs):
        self.attention_mask = kwargs.get("attention_mask")
        return torch.zeros_like(kwargs["noisy_latents"])


class _CaptureChunkedVideoTower:
    def __init__(self) -> None:
        self.kwargs = None

    def predict_chunked_conditioned_video_flow(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros_like(kwargs["noisy_latents"])


def test_chunked_conditioned_video_training_uses_external_prefix_and_full_target() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(
            program=CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO,
            noisy_video_condition_prob=0.0,
            use_activation_checkpointing=True,
        ),
        training_config=TrainingConfig(
            video_num_train_timesteps=8,
            chunk_size=2,
            window_size=4,
        ),
        inference_config=InferenceConfig(frame_chunk_size=2),
    )
    target = torch.randn(1, 48, 4, 2, 2)
    condition = torch.full_like(target, 7.0)
    frontend = VisualFrontendOutput(
        canonical_video=torch.zeros(1, 3, 4, 32, 32),
        video_latents=target,
        video_tokens=torch.zeros(1, 4, 4),
        input_source="video_latents",
        token_grid=TokenGridMetadata(
            num_frames=4,
            latent_height=2,
            latent_width=2,
            patch_size=(1, 2, 2),
            patches_per_frame_h=1,
            patches_per_frame_w=1,
            tokens_per_frame=1,
            sequence_length=4,
        ),
        chunk=ChunkMetadata(
            chunk_start_frame=0,
            chunk_num_frames=4,
            frame_stride=1,
            chunk_type="dense_video_chunk",
        ),
        conditioning=ConditioningState(
            supported=True,
            text_context=torch.ones(1, 2, 8),
            negative_text_context=torch.ones(1, 2, 8),
            metadata={},
        ),
    )
    batch = PolicyTrainBatch(
        actions=torch.zeros(1, 0, 7),
        action_mask=torch.zeros(1, 0, 7),
        state=torch.zeros(1, 0, 8),
        extra={
            "task_text": ("move object",),
            "condition_latents": condition,
            "metadata": (
                {
                    "sampled_chunk_size": 2,
                    "sampled_window_size": 4,
                    "frame_shift": 9,
                    "latent_loss_frame_start": 1,
                    "latent_loss_frame_end": 3,
                },
            ),
        },
    )
    tower = _CaptureChunkedVideoTower()

    output = variant.forward_train(
        tower,  # type: ignore[arg-type]
        VisualStageOutputs(frontend=frontend),
        PolicyPreparedInputs(batch=batch),
    )

    assert tower.kwargs is not None
    assert tower.kwargs["chunk_size"] == 2
    assert tower.kwargs["window_size"] == 4
    assert tower.kwargs["frame_start"] == 8
    assert tower.kwargs["prefix_condition_frames"] == 1
    assert tower.kwargs["use_activation_checkpointing"] is True
    artifacts = output.decoder_artifacts.require(
        contract=VIDEO_FLOW_DECODER_ARTIFACT_CONTRACT,
        payload_type=VideoFlowTrainArtifacts,
    )
    assert artifacts.target_latents.shape[2] == 5
    torch.testing.assert_close(artifacts.target_latents[:, :, :1], condition[:, :, :1])
    torch.testing.assert_close(artifacts.target_latents[:, :, 1:], target)
    expected_loss_mask = torch.tensor([0, 0, 1, 1, 0], dtype=target.dtype)
    torch.testing.assert_close(
        artifacts.future_loss_mask.flatten(),
        expected_loss_mask,
    )
    torch.testing.assert_close(
        output.metrics["future_frame_count"],
        torch.tensor(2.0),
    )
    torch.testing.assert_close(
        tower.kwargs["noisy_latents"][:, :, :1],
        condition[:, :, :1],
    )


def test_causal_video_prediction_masks_padded_tokens_during_train_rollout() -> None:
    variant = CausalVideoPredictionPolicyVariant(
        config=CausalVideoPredictionPolicyConfig(program=CausalVideoProgram.PREFIX_SUFFIX),
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


def _tiny_chunked_conditioned_video_pipeline():
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/causal_video_prediction_libero_chunked_conditioned.yaml"
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
            max_text_tokens=3,
            freq_dim=8,
        ),
        policy_variant=replace(config.policy_variant, hidden_size=16),
        action_decoder=replace(config.action_decoder, hidden_size=16),
        training=replace(
            config.training,
            video_num_train_timesteps=8,
            text_condition_dropout_prob=0.0,
        ),
        inference=replace(
            config.inference,
            video_num_inference_steps=2,
            guidance_scale=1.0,
            frame_chunk_size=2,
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
    apply_training_component_controls(pipeline, config.training)
    return config, pipeline


def test_chunked_conditioned_video_pipeline_trains_and_evaluates_without_actions() -> None:
    config, pipeline = _tiny_chunked_conditioned_video_pipeline()
    target = torch.randn(1, 48, 4, 2, 4)
    condition = torch.randn_like(target)
    text = torch.randn(1, 3, 8)
    negative_text = torch.randn_like(text)
    policy_batch = PolicyTrainBatch(
        actions=torch.zeros(1, 0, 7),
        action_mask=torch.zeros(1, 0, 7),
        state=torch.zeros(1, 0, 8),
        extra={
            "task_text": ("move object",),
            "condition_latents": condition,
            "metadata": (
                {
                    "sampled_chunk_size": 2,
                    "sampled_window_size": 4,
                    "frame_shift": 0,
                },
            ),
        },
    )

    torch.manual_seed(4242)
    output = pipeline.forward_train_from_latents(
        target,
        policy_batch,
        text_context=text,
        negative_text_context=negative_text,
    )
    assert torch.isfinite(output.decoder_output.loss)
    output.decoder_output.loss.backward()
    runtime = pipeline.visual_tower.core
    assert runtime.patch_embedding_mlp.weight.grad is not None
    assert runtime.action_embedder.weight.grad is None
    assert runtime.action_proj_out.weight.grad is None

    latent_batch = LatentWAMBatch(
        video_latents=target,
        actions=torch.zeros(1, 0, 7),
        action_mask=torch.zeros(1, 0, 7),
        state=torch.zeros(1, 0, 8),
        task_text=("move object",),
        text_context=text,
        negative_text_context=negative_text,
        condition_latents=condition,
        metadata=({"frame_shift": 0},),
    )
    torch.manual_seed(101)
    rollout = rollout_causal_video_prediction(
        pipeline,
        latent_batch,
        num_chunks=2,
    )
    assert rollout.observed_latent_frames == 1
    assert rollout.future_latent_frames == 2
    assert rollout.context_latent_frames == (1, 3, 5)
    assert rollout.predicted_latents.shape[2] == 5
    assert torch.isfinite(torch.tensor(rollout.first_chunk_future_mse))
    assert config.policy_variant.program == CausalVideoProgram.CHUNKED_CONDITIONED_VIDEO
    assert config.policy_variant.use_activation_checkpointing is True


def test_chunked_conditioned_video_generation_commits_each_chunk_to_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pipeline = _tiny_chunked_conditioned_video_pipeline()
    calls: list[dict[str, object]] = []

    def capture_prediction(**kwargs):
        calls.append(
            {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        )
        return torch.zeros_like(kwargs["noisy_latents"])

    monkeypatch.setattr(
        pipeline.visual_tower,
        "predict_chunked_conditioned_video_flow",
        capture_prediction,
    )
    prefix = torch.randn(1, 48, 1, 2, 4)
    future = torch.zeros(1, 48, 4, 2, 4)

    generated = pipeline.visual_tower.generate_chunked_conditioned_video_latents(
        observed_history=prefix,
        future_template=future,
        text_context=torch.randn(1, 3, 8),
        negative_text_context=None,
        history_frame_start=0,
        chunk_size=2,
        window_size=4,
        chunk_origin_frame=0,
        num_inference_steps=1,
        num_train_timesteps=8,
        sigma_shift=5.0,
        guidance_scale=1.0,
        sample_seed=17,
    )

    assert generated.shape == future.shape
    assert [call["noisy_latents"].shape[2] for call in calls] == [3, 5]
    assert [call["chunk_origin_frame"] for call in calls] == [0, 0]
    second_noisy = calls[1]["noisy_latents"]
    second_condition = calls[1]["condition_latents"]
    assert isinstance(second_noisy, torch.Tensor)
    assert isinstance(second_condition, torch.Tensor)
    torch.testing.assert_close(second_noisy[:, :, :3], second_condition[:, :, :3])
    assert not torch.any(second_condition[:, :, 3:])
    second_timesteps = calls[1]["timesteps"]
    assert isinstance(second_timesteps, torch.Tensor)
    assert not torch.any(second_timesteps[:, :3])
    assert torch.all(second_timesteps[:, 3:] > 0)


def test_chunked_conditioned_video_generation_respects_partial_chunk_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pipeline = _tiny_chunked_conditioned_video_pipeline()
    calls: list[dict[str, object]] = []

    def capture_prediction(**kwargs):
        calls.append(
            {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        )
        return torch.zeros_like(kwargs["noisy_latents"])

    monkeypatch.setattr(
        pipeline.visual_tower,
        "predict_chunked_conditioned_video_flow",
        capture_prediction,
    )
    # One external condition plus three real target frames means the next target
    # completes the current four-frame chunk before a new chunk can begin.
    observed_history = torch.randn(1, 48, 4, 2, 4)
    future = torch.zeros(1, 48, 4, 2, 4)

    generated = pipeline.visual_tower.generate_chunked_conditioned_video_latents(
        observed_history=observed_history,
        future_template=future,
        text_context=torch.randn(1, 3, 8),
        negative_text_context=None,
        history_frame_start=0,
        chunk_size=4,
        window_size=30,
        chunk_origin_frame=0,
        num_inference_steps=1,
        num_train_timesteps=8,
        sigma_shift=5.0,
        guidance_scale=1.0,
        sample_seed=17,
    )

    assert generated.shape == future.shape
    # The interrupted first block is reconstructed from its pre-block history;
    # its already-executed prefix is not exposed as clean same-block context.
    first_condition = calls[0]["condition_latents"]
    assert isinstance(first_condition, torch.Tensor)
    torch.testing.assert_close(
        first_condition[:, :, :1],
        observed_history[:, :, :1],
    )
    assert not torch.any(first_condition[:, :, 1:])
    # The second pass over-generates the complete next model block and returns
    # only its requested three-frame prefix.
    assert [int(call["noisy_latents"].shape[2]) for call in calls] == [5, 9]
    first_noisy = calls[0]["noisy_latents"]
    second_noisy = calls[1]["noisy_latents"]
    assert isinstance(first_noisy, torch.Tensor)
    assert isinstance(second_noisy, torch.Tensor)
    torch.testing.assert_close(generated[:, :, :1], first_noisy[:, :, 4:5])
    torch.testing.assert_close(generated[:, :, 1:], second_noisy[:, :, 5:8])
    assert [call["window_size"] for call in calls] == [30, 30]
    assert [call["frame_start"] for call in calls] == [0, 0]


def test_chunked_conditioned_video_generation_overgenerates_short_final_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pipeline = _tiny_chunked_conditioned_video_pipeline()
    calls: list[torch.Tensor] = []

    def capture_prediction(**kwargs):
        calls.append(kwargs["noisy_latents"].clone())
        return torch.zeros_like(kwargs["noisy_latents"])

    monkeypatch.setattr(
        pipeline.visual_tower,
        "predict_chunked_conditioned_video_flow",
        capture_prediction,
    )
    observed = torch.randn(1, 48, 1, 2, 4)
    requested = torch.zeros(1, 48, 3, 2, 4)

    generated = pipeline.visual_tower.generate_chunked_conditioned_video_latents(
        observed_history=observed,
        future_template=requested,
        text_context=torch.randn(1, 3, 8),
        negative_text_context=None,
        history_frame_start=0,
        chunk_size=4,
        window_size=30,
        chunk_origin_frame=0,
        num_inference_steps=1,
        num_train_timesteps=8,
        sigma_shift=5.0,
        guidance_scale=1.0,
        sample_seed=17,
    )

    assert generated.shape == requested.shape
    assert [int(call.shape[2]) for call in calls] == [5]


def test_chunked_conditioned_video_generation_physically_bounds_w30_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pipeline = _tiny_chunked_conditioned_video_pipeline()
    calls: list[dict[str, object]] = []

    def capture_prediction(**kwargs):
        calls.append(
            {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        )
        return torch.zeros_like(kwargs["noisy_latents"])

    monkeypatch.setattr(
        pipeline.visual_tower,
        "predict_chunked_conditioned_video_flow",
        capture_prediction,
    )
    observed = torch.randn(1, 48, 101, 2, 4)
    requested = torch.zeros(1, 48, 8, 2, 4)

    generated = pipeline.visual_tower.generate_chunked_conditioned_video_latents(
        observed_history=observed,
        future_template=requested,
        text_context=torch.randn(1, 3, 8),
        negative_text_context=None,
        history_frame_start=0,
        chunk_size=4,
        window_size=30,
        chunk_origin_frame=0,
        num_inference_steps=1,
        num_train_timesteps=8,
        sigma_shift=5.0,
        guidance_scale=1.0,
        sample_seed=17,
    )

    assert generated.shape == requested.shape
    # 60 visible target-history frames + one external condition + four current.
    assert [int(call["noisy_latents"].shape[2]) for call in calls] == [65, 65]
    assert [call["frame_start"] for call in calls] == [40, 44]
    assert [call["chunk_origin_frame"] for call in calls] == [0, 0]

    calls.clear()
    generated = pipeline.visual_tower.generate_chunked_conditioned_video_latents(
        # The final three target frames are an interrupted current block. They
        # are retained in state but removed before the full block is regenerated.
        observed_history=torch.randn(1, 48, 104, 2, 4),
        future_template=torch.zeros(1, 48, 1, 2, 4),
        text_context=torch.randn(1, 3, 8),
        negative_text_context=None,
        history_frame_start=0,
        chunk_size=4,
        window_size=30,
        chunk_origin_frame=0,
        num_inference_steps=1,
        num_train_timesteps=8,
        sigma_shift=5.0,
        guidance_scale=1.0,
        sample_seed=17,
    )
    assert generated.shape[2] == 1
    assert [int(call["noisy_latents"].shape[2]) for call in calls] == [65]
    assert calls[0]["frame_start"] == 40


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
