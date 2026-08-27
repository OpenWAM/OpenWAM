from __future__ import annotations

from pathlib import Path

import torch

from open_wam.configs import (
    CurrentBlockCoupling,
    HistoryStreamVisibility,
    load_experiment_config,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.data import build_synthetic_batch
from open_wam.models.common import AttentionProfileSpec, PreparedAttentionProfile
from open_wam.models.policy_variants import PolicyTrainBatch
from open_wam.models.policy_variants.dual_expert.attention import (
    build_dual_expert_packed_coupling_attention_profile,
)
from open_wam.models.policy_variants.dual_expert.dual_stream_execution import (
    forward_dual_expert_packed_coupling_denoise,
)
from open_wam.models.policy_variants.dual_expert.modules import (
    DualExpertActionExpert,
)
from open_wam.models.visual_tower import (
    RuntimeProgramSpec,
    RuntimeSequenceFamily,
    RuntimeStepInput,
    VisualCoreInput,
    VisualTower,
    build_chunked_conditioned_video_runtime_program,
    build_chunked_dual_stream_exact_train_program,
    build_dense_runtime_program,
)
from open_wam.models.visual_tower.core import PackedSequenceVisualCore
from open_wam.models.visual_tower.grid_ids import build_mesh_id
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


def test_conditioned_video_runtime_matches_vta_video_marginal_and_skips_actions(
    monkeypatch,
) -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    )
    pipeline = build_variant_pipeline_from_config(config)
    runtime = pipeline.visual_tower.get_runtime_backbone(
        action_dim=config.data.action_schema.action_dim
    )
    runtime.eval()
    device = next(runtime.parameters()).device
    video = torch.randn(1, 48, 3, 2, 2, device=device)
    video_condition = torch.randn_like(video)
    action = torch.randn(
        1,
        config.data.action_schema.action_dim,
        2,
        2,
        1,
        device=device,
    )
    text = torch.randn(
        1,
        config.backbone.max_text_tokens,
        config.backbone.text_dim,
        device=device,
    )
    video_grid = build_mesh_id(
        f=3,
        h=1,
        w=1,
        t=0,
        f_shift=-1,
        action=False,
        device=device,
    )[None]
    action_grid = build_mesh_id(
        f=2,
        h=2,
        w=1,
        t=1,
        f_shift=0,
        action=True,
        device=device,
    )[None]
    video_payload = {
        "noisy_latents": video,
        "latent": video_condition,
        "text_emb": text,
        "grid_id": video_grid,
        "timesteps": torch.tensor([[0.0, 900.0, 400.0]], device=device),
        "cond_timesteps": torch.zeros(1, 3, device=device),
    }
    common_payload = {
        "chunk_size": 1,
        "window_size": 4,
        "chunk_origin_frame": 0,
        "prefix_condition_frames": 1,
    }
    with torch.no_grad():
        vta_output = runtime.execute_runtime_step(
            RuntimeStepInput(
                program=build_chunked_dual_stream_exact_train_program(
                    attention_profile_name="chunked_temporal_exact"
                ),
                payload={
                    **common_payload,
                    "attention_profile_name": "chunked_temporal_exact",
                    "history_stream_visibility": "video_only",
                    "latent_dict": video_payload,
                    "action_dict": {
                        "noisy_latents": action,
                        "latent": torch.zeros_like(action),
                        "text_emb": text,
                        "grid_id": action_grid,
                        "timesteps": torch.ones(1, 2, device=device) * 500,
                        "cond_timesteps": torch.zeros(1, 2, device=device),
                        "actions_mask": torch.ones_like(action),
                    },
                },
            )
        )

    embed_calls: list[str] = []
    original_input_embed = runtime._input_embed

    def capture_input_embed(tensor, *, input_type):
        embed_calls.append(input_type)
        return original_input_embed(tensor, input_type=input_type)

    def reject_action_projection(*args, **kwargs):
        raise AssertionError("conditioned-video runtime used the action head")

    monkeypatch.setattr(runtime, "_input_embed", capture_input_embed)
    monkeypatch.setattr(runtime.action_proj_out, "forward", reject_action_projection)
    with torch.no_grad():
        video_output = runtime.execute_runtime_step(
            RuntimeStepInput(
                program=build_chunked_conditioned_video_runtime_program(),
                payload={
                    **common_payload,
                    "latent_dict": video_payload,
                    "stage": "train",
                },
            )
        )

    assert embed_calls == ["latent", "latent"]
    assert set(video_output.projected_outputs) == {"video_prediction"}
    torch.testing.assert_close(
        video_output.projected_outputs["video_prediction"],
        vta_output.projected_outputs["video_prediction"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_conditioned_video_runtime_matches_m5_dual_expert_video_branch() -> None:
    torch.manual_seed(9)
    config = SharedVideoTransformerConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=8,
        max_text_tokens=3,
        freq_dim=8,
        load_reference_core_weights=False,
        load_text_conditioning=False,
        load_wan_vae_frontend=False,
    )
    tower = VisualTower(config, action_dim=4, state_dim=3)
    action_expert = DualExpertActionExpert(
        hidden_size=16,
        action_dim=4,
        num_layers=1,
        num_heads=2,
        attention_head_dim=8,
        ffn_dim=32,
        text_dim=8,
        freq_dim=8,
    )
    tower.eval()
    action_expert.eval()

    noisy_video = torch.randn(1, 48, 3, 2, 2)
    condition_video = torch.randn_like(noisy_video)
    video_timesteps = torch.tensor([[0.0, 900.0, 400.0]])
    condition_timesteps = torch.zeros_like(video_timesteps)
    text_context = torch.randn(1, 3, 8)
    packed_actions = torch.randn(1, 4, 4)
    packed_action_timesteps = torch.tensor([[500.0, 500.0, 0.0, 0.0]])
    packed_action_grid = build_mesh_id(
        f=2,
        h=2,
        w=1,
        t=1.0,
        f_shift=0.0,
        action=True,
    )[None]
    packed_action_pre = action_expert.pre_dit(
        action_tokens=packed_actions,
        timestep=packed_action_timesteps,
        context=text_context,
        action_grid_ids=packed_action_grid,
    )
    m5_profile = build_dual_expert_packed_coupling_attention_profile(
        num_video_frames=3,
        video_tokens_per_frame=1,
        num_action_frames=2,
        action_tokens_per_frame=1,
        chunk_size_frames=1,
        attention_window_size=4,
        device=torch.device("cpu"),
        current_block_coupling=CurrentBlockCoupling.VIDEO_THEN_ACTION,
        history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
        prefix_condition_frames=1,
        build_dense_masks=True,
        build_flex_masks=False,
    )

    with torch.no_grad():
        m5_video, _ = forward_dual_expert_packed_coupling_denoise(
            visual_tower=tower,
            noisy_video_latents=noisy_video,
            clean_video_latents=condition_video,
            noisy_video_timesteps=video_timesteps,
            clean_video_timesteps=condition_timesteps,
            action_expert=action_expert,
            packed_action_pre=packed_action_pre,
            attention_profile=m5_profile,
            text_context=text_context,
            frame_start=-1,
            use_activation_checkpointing=False,
            prefer_flex_attention=False,
        )
        video_only = tower.predict_chunked_conditioned_video_flow(
            noisy_latents=noisy_video,
            condition_latents=condition_video,
            timesteps=video_timesteps,
            condition_timesteps=condition_timesteps,
            text_context=text_context,
            chunk_size=1,
            window_size=4,
            frame_start=-1,
            prefix_condition_frames=1,
            stage="train",
        )

    torch.testing.assert_close(video_only, m5_video, rtol=1e-5, atol=1e-5)


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
