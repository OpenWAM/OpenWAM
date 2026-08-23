from __future__ import annotations

from pathlib import Path

import torch

from open_wam.configs import load_experiment_config
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.data import build_synthetic_batch
from open_wam.models.common import AttentionProfileSpec, PreparedAttentionProfile
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.models.visual_tower import (
    RuntimeProgramSpec,
    RuntimeSequenceFamily,
    RuntimeStepInput,
    VisualCoreInput,
    build_chunked_dual_stream_exact_train_program,
    build_dense_runtime_program,
)
from open_wam.models.visual_tower.core import PackedSequenceVisualCore
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.utils.config_overrides import apply_config_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_program_coerces_sequence_family_to_typed_contract() -> None:
    program = RuntimeProgramSpec(
        name="extension_dense",
        sequence_family="dense_default",
    )

    assert program.sequence_family is RuntimeSequenceFamily.DENSE


def test_exact_runtime_program_executes_on_shared_backbone() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    assert not hasattr(pipeline.policy_variant, "action_embedder")
    assert not hasattr(pipeline.policy_variant, "video_flow_head")
    assert not hasattr(pipeline.policy_variant, "action_flow_head")
    batch = build_synthetic_batch(config.data, batch_size=2)
    visual_outputs = pipeline.prepare_visual_outputs(batch.views)
    train_batch = PolicyTrainBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
        extra={"task_text": batch.task_text, "metadata": batch.metadata},
    )
    prepared_inputs = pipeline.policy_variant.prepare_train_inputs(
        visual_outputs, train_batch
    )
    train_artifacts = prepared_inputs.variant_inputs["parallel_train_artifacts"]
    assert "lingbot_train_artifacts" not in prepared_inputs.variant_inputs

    runtime_backbone = pipeline.visual_tower.get_runtime_backbone(
        action_dim=config.data.action_schema.action_dim
    )
    step_output = runtime_backbone.execute_runtime_step(
        RuntimeStepInput(
            program=build_chunked_dual_stream_exact_train_program(
                attention_profile_name=train_artifacts.input_dict[
                    "attention_profile_name"
                ],
            ),
            payload=train_artifacts.input_dict,
        )
    )

    assert step_output.projected_outputs["video_prediction"].ndim == 3
    assert step_output.projected_outputs["action_prediction"].ndim == 3
    assert step_output.aux["runtime_program"] == "chunked_dual_stream_exact_train"
    assert step_output.aux["sequence_family"] == "chunked_dual_stream_exact"


def test_exact_runtime_program_owns_packed_proprio_conditioning() -> None:
    config = apply_config_overrides(
        load_experiment_config(
            REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
        ),
        {
            "backbone.text_dim": 16,
            "backbone.max_text_tokens": 4,
            "policy_variant.proprio_context_mode": "per_chunk_additive",
        },
    )
    pipeline = build_variant_pipeline_from_config(config)
    video_latents = torch.randn(1, 48, 4, 2, 2)
    state_frames = torch.randn(1, 4, config.data.action_schema.state_dim)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 8, config.data.action_schema.action_dim),
        action_mask=torch.ones(1, 8, config.data.action_schema.action_dim),
        state=state_frames[:, :1],
        extra={"proprio_context_frames": state_frames},
    )

    output = pipeline.forward_train_from_latents(
        video_latents,
        batch,
        text_context=torch.randn(1, 4, 16),
    )
    output.policy_output.policy_features.square().mean().backward()

    encoder = pipeline.visual_tower.core.proprio_hidden_context_encoder
    assert encoder is not None
    assert encoder.proj.weight.grad is not None
    assert torch.isfinite(encoder.proj.weight.grad).all()


def test_dense_runtime_accepts_application_prepared_attention_profile() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    profile = PreparedAttentionProfile(
        spec=AttentionProfileSpec(
            name="application_diagonal",
            family="application",
            backend="dense",
        ),
        self_attention_mask=torch.eye(4, dtype=torch.bool),
    )

    step_output = pipeline.visual_tower.execute_runtime_step(
        RuntimeStepInput(
            program=build_dense_runtime_program(),
            core_input=VisualCoreInput(
                tokens=torch.randn(1, 4, config.backbone.hidden_size),
                attention_profile=profile,
            ),
        )
    )

    assert step_output.tokens is not None
    assert step_output.tokens.shape == (1, 4, config.backbone.hidden_size)
    assert step_output.aux["runtime_program"] == "dense_default"
    assert step_output.aux["sequence_family"] == "dense_default"


def test_dense_runtime_executes_through_lightweight_core() -> None:
    config = SharedVideoTransformerConfig(
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        attention_head_dim=4,
    )
    core = PackedSequenceVisualCore(config)

    step_output = core.execute_runtime_step(
        RuntimeStepInput(
            program=build_dense_runtime_program(),
            core_input=VisualCoreInput(tokens=torch.randn(1, 3, 8)),
        )
    )

    assert step_output.tokens is not None
    assert step_output.tokens.shape == (1, 3, 8)
    assert step_output.aux["runtime_program"] == "dense_default"
    assert step_output.aux["sequence_family"] == "dense_default"
