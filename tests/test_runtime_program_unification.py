from __future__ import annotations

from pathlib import Path

import torch

from open_wam.data import build_synthetic_batch
from open_wam.models.common import AttentionProfileSpec, PreparedAttentionProfile
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.models.visual_tower import (
    RuntimeStepInput,
    VisualCoreInput,
    build_chunked_dual_stream_exact_train_program,
    build_dense_runtime_program,
)
from open_wam.pipelines import build_variant_pipeline_from_config
from open_wam.configs import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    prepared_inputs = pipeline.policy_variant.prepare_train_inputs(visual_outputs, train_batch)
    train_artifacts = prepared_inputs.variant_inputs["lingbot_train_artifacts"]

    runtime_backbone = pipeline.visual_tower.get_runtime_backbone(
        action_dim=config.data.action_schema.action_dim
    )
    step_output = runtime_backbone.execute_runtime_step(
        RuntimeStepInput(
            program=build_chunked_dual_stream_exact_train_program(
                attention_profile_name=train_artifacts.input_dict["attention_profile_name"],
                cache_backend_name="slot_pool_exact",
            ),
            payload=train_artifacts.input_dict,
        )
    )

    assert step_output.projected_outputs["video_prediction"].ndim == 3
    assert step_output.projected_outputs["action_prediction"].ndim == 3
    assert step_output.aux["runtime_program"] == "chunked_dual_stream_exact_train"
    assert step_output.aux["sequence_family"] == "chunked_dual_stream_exact"


def test_dense_runtime_accepts_application_prepared_attention_profile() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
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
