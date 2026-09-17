from __future__ import annotations
from open_wam.models.common.flow_schedule import FlowMatchScheduler as FlowMatchScheduler
from open_wam.models.visual_tower.exact_runtime import repeat_exact_single_stream_input_for_cfg as repeat_input_for_cfg
from open_wam.models.visual_tower.exact_runtime import run_exact_single_stream_forward as run_reference_single_stream_forward
import open_wam.models.common.dynamics_objectives as owner_dynamics_objectives

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_wam.configs import (
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DynamicsObjective,
    HistoryStreamVisibility,
    InferenceConfig,
    JointTimestepCoupling,
    ParallelStreamPolicyConfig,
    ProprioContextMode,
    TrainingConfig,
    VideoActionProgram,
    VideoActionSequenceContract,
)
from open_wam.contracts import (
    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY,
    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON,
    DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY,
    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY,
    DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY,
    DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE,
    DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY,
    DYNAMICS_ROUTING_MODE_METADATA_KEY,
    DYNAMICS_ROUTING_SOURCE_METADATA_KEY,
    SampleConstructionMetadata,
)
from open_wam.models.action_decoders.parallel_stream_decoder import (
    ParallelStreamActionDecoder,
)
from open_wam.models.common import (
    SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION,
    SlotPoolLayerState,
    build_chunked_temporal_exact_attention_profile,
)
from open_wam.models.common.cache_backend_contracts import SlotPoolCachePayload
from open_wam.models.common.flow_matching import (
    FlowMatchScheduler as SharedFlowMatchScheduler,
)
from open_wam.models.policy_variants.contracts import (
    DecoderArtifactEnvelope,
    PolicyTemporalGeometry,
    PolicyTrainBatch,
    PolicyTrainOutput,
)
from open_wam.models.decoder_artifacts import (
    PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT,
    ParallelDecoderTrainArtifacts,
)
from open_wam.models.policy_variants.parallel_stream.inference_conditioning import (
    append_generalist_mode_text_context,
)
from open_wam.models.policy_variants.parallel_stream.training_exact_artifacts import (
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_exact_train_artifacts,
)
from open_wam.models.policy_variants.parallel_stream.training_prefix_artifacts import (
    prepare_parallel_prefix_condition_exact_train_artifacts,
)
from open_wam.models.policy_variants.parallel_stream.variant import (
    ParallelStreamPolicyVariant,
)
from open_wam.models.video_backbone.config import (
    LingbotCompatibleVideoBackboneConfig,
)
from open_wam.models.video_backbone.contracts import (
    CacheState,
    ChunkMetadata,
    ConditioningState,
    TokenGridMetadata,
)
from open_wam.models.visual_tower import (
    RuntimeStepOutput,
)
from open_wam.models.visual_tower.contracts import (
    VisualFrontendOutput,
    VisualStageOutputs,
)
from open_wam.models.visual_tower.replica_core import (
    SharedVideoTransformerCore,
    _retained_slot_pool_indices_for_current_write,
)
from open_wam.models.visual_tower.tower import VisualTower


def _generalist_sample_metadata(
    mode: DynamicsObjective,
    *,
    source: str = "real_demo",
    drop_text: bool | None = None,
) -> SampleConstructionMetadata:
    metadata: dict[str, object] = {
        DYNAMICS_ROUTING_MODE_METADATA_KEY: mode.value,
        DYNAMICS_ROUTING_SOURCE_METADATA_KEY: source,
    }
    if drop_text is not None:
        metadata[DYNAMICS_ROUTING_DROP_TEXT_METADATA_KEY] = drop_text
    if mode.is_conditional:
        metadata.update(
            {
                DYNAMICS_CONDITIONAL_LAYOUT_METADATA_KEY: (
                    DYNAMICS_CONDITIONAL_LAYOUT_TARGET_ONLY_T0_PLUS_FUTURE
                ),
                DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_METADATA_KEY: (
                    DYNAMICS_CONDITIONAL_CHUNK_LAYOUT_T0_SINGLETON
                ),
                DYNAMICS_CONDITIONAL_HISTORY_POLICY_METADATA_KEY: (
                    DYNAMICS_CONDITIONAL_HISTORY_PREVIOUS_BOUNDARY_VIDEO_ONLY
                ),
                "history_frames": 1,
                "loss_frame_start": 1,
                "latent_loss_frame_start": 1,
                "action_loss_frame_start": 1,
                "chunk_origin_frame": 1,
                "target_observation_frame_in_sample": 0,
                "singleton_chunk_frame": 0,
                "context_prefix_frames_in_sample": 1,
            }
        )
    resolved = SampleConstructionMetadata.from_mapping(metadata)
    assert resolved is not None
    return resolved


def test_repeat_input_for_cfg_preserves_hidden_context() -> None:
    input_dict = {
        "noisy_latents": torch.randn(2, 3, 1, 2, 2),
        "text_emb": torch.randn(2, 4, 8),
        "grid_id": torch.zeros(2, 4, 1),
        "timesteps": torch.zeros(2, 1),
        "hidden_context": torch.randn(2, 4, 8),
    }
    negative_text_emb = torch.randn(2, 4, 8)

    repeated = repeat_input_for_cfg(input_dict, negative_text_emb=negative_text_emb)

    assert repeated["hidden_context"].shape == (4, 4, 8)
    torch.testing.assert_close(
        repeated["hidden_context"][:2], input_dict["hidden_context"]
    )
    torch.testing.assert_close(
        repeated["hidden_context"][2:], input_dict["hidden_context"]
    )


def test_m1_reference_runtime_uses_shared_flow_match_scheduler() -> None:
    assert FlowMatchScheduler is SharedFlowMatchScheduler


class _FakeReferenceTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = (1, 2, 2)
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.cache_batch_sizes: dict[str, int] = {}
        self.cache_layouts: dict[str, tuple[int, int]] = {}
        self.cache_attn_windows: dict[str, int] = {}
        self.cleared_pred_cache_names: list[str] = []
        self.last_text_emb: torch.Tensor | None = None
        self.last_noisy_latents: torch.Tensor | None = None
        self.runtime_cache_states: dict[str, CacheState] = {}

    def clear_cache(self, cache_name: str) -> None:
        self.cache_batch_sizes.pop(cache_name, None)

    def clear_pred_cache(self, cache_name: str) -> None:
        self.cleared_pred_cache_names.append(cache_name)

    def clear_runtime_cache_state(self, cache_name: str) -> None:
        self.clear_cache(cache_name)
        self.runtime_cache_states.pop(cache_name, None)

    def clear_runtime_prediction_cache(self, cache_name: str) -> None:
        self.clear_pred_cache(cache_name)

    def get_runtime_cache_state(self, cache_name: str) -> CacheState | None:
        return self.runtime_cache_states.get(cache_name)

    def replace_runtime_cache_state(
        self,
        cache_name: str,
        cache_state: CacheState,
    ) -> None:
        self.runtime_cache_states[cache_name] = cache_state

    def initialize_runtime_cache_backend(
        self,
        cache_name: str,
        **kwargs,
    ) -> None:
        self.create_empty_cache(cache_name, **kwargs)
        backend_name = str(kwargs.get("backend_name", "slot_pool_exact"))
        self.runtime_cache_states[cache_name] = CacheState(
            supported=True,
            current_start_frame=0,
            cached_frames=0,
            chunk_size=0,
            capability="fake_exact_runtime",
            backend_name=backend_name,
            backend_payload=SlotPoolCachePayload(
                layer_states=(SlotPoolLayerState(),),
                batch_size=int(kwargs["batch_size"]),
                metadata={
                    "attn_window": int(kwargs["attn_window"]),
                },
            ),
            payload={
                "attn_window": int(kwargs["attn_window"]),
            },
        )

    def execute_runtime_step(self, step_input) -> RuntimeStepOutput:
        if step_input.payload is None:
            raise ValueError("Fake exact runtime requires a payload.")
        return RuntimeStepOutput(
            tokens=self.forward(
                step_input.payload,
                update_cache=step_input.update_cache,
                cache_name=step_input.cache_name,
                action_mode=step_input.action_mode,
            )
        )

    def create_empty_cache(
        self,
        cache_name: str,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        backend_name: str = "lingbot_slot_pool",
        prefix_visibility_mode: str = "full_history",
    ) -> None:
        del device, dtype, backend_name, prefix_visibility_mode
        self.cache_batch_sizes[cache_name] = batch_size
        self.cache_layouts[cache_name] = (
            latent_token_per_chunk,
            action_token_per_chunk,
        )
        self.cache_attn_windows[cache_name] = int(attn_window)

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        batch_size = input_dict["noisy_latents"].shape[0]
        self.last_text_emb = input_dict["text_emb"].detach().clone()
        self.last_noisy_latents = input_dict["noisy_latents"].detach().clone()
        if update_cache and cache_name in self.cache_batch_sizes:
            assert batch_size == self.cache_batch_sizes[cache_name]
        latents = input_dict["noisy_latents"]
        if action_mode:
            return (
                latents.squeeze(-1)
                .permute(0, 2, 3, 1)
                .reshape(batch_size, -1, latents.shape[1])
            )
        patch_t, patch_h, patch_w = self.patch_size
        return (
            latents.view(
                batch_size,
                latents.shape[1],
                latents.shape[2] // patch_t,
                patch_t,
                latents.shape[3] // patch_h,
                patch_h,
                latents.shape[4] // patch_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(batch_size, -1, latents.shape[1] * patch_t * patch_h * patch_w)
        )


class _GradTrackingTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.grad_enabled_during_forward: bool | None = None

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ) -> torch.Tensor:
        del update_cache, cache_name, action_mode
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        return input_dict["noisy_latents"] * self.weight

    def execute_runtime_step(self, step_input) -> RuntimeStepOutput:
        if step_input.payload is None:
            raise ValueError("Fake exact runtime requires a payload.")
        return RuntimeStepOutput(
            tokens=self.forward(
                step_input.payload,
                update_cache=step_input.update_cache,
                cache_name=step_input.cache_name,
                action_mode=step_input.action_mode,
            )
        )


def test_deprecated_text_token_proprio_encoder_is_default_off_and_zero_init() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    text_emb = torch.randn(2, 5, 16)
    assert core.proprio_context_encoder is None
    assert (
        core.append_proprio_context_tokens(text_emb, torch.randn(2, 8)) is text_emb
    )  # deprecated helper

    core.configure_proprio_context_encoder(enabled=True, state_dim=8)
    assert core.proprio_context_encoder is not None
    assert "proprio_context_encoder.proj.weight" in core.state_dict()
    appended = core.append_proprio_context_tokens(
        text_emb, torch.randn(2, 8)
    )  # deprecated helper

    assert appended.shape == (2, 6, 16)
    assert torch.equal(appended[:, :5], text_emb)
    assert torch.equal(appended[:, 5:], torch.zeros_like(appended[:, 5:]))


def test_generalist_mode_context_encoder_is_default_off_and_small_random_init() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    text_emb = torch.randn(2, 5, 16)
    assert core.generalist_mode_context_encoder is None
    assert core.append_generalist_mode_context_token(text_emb, "joint") is text_emb

    core.configure_generalist_mode_context_encoder(enabled=True)
    assert core.generalist_mode_context_encoder is not None
    assert "generalist_mode_context_encoder.embedding.weight" in core.state_dict()
    appended = core.append_generalist_mode_context_token(
        text_emb,
        ["joint", "video_conditioned_action"],
    )

    assert appended.shape == (2, 6, 16)
    assert torch.equal(appended[:, :5], text_emb)
    assert not torch.equal(appended[:, 5:], torch.zeros_like(appended[:, 5:]))
    assert float(appended[:, 5:].detach().abs().max()) < 0.2
    assert not torch.equal(appended[0, 5], appended[1, 5])


def test_generalist_mode_context_rejects_out_of_range_tensor_indices() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    core.configure_generalist_mode_context_encoder(enabled=True)

    with pytest.raises(ValueError, match="Generalist mode tensor indices"):
        core.append_generalist_mode_context_token(
            torch.randn(1, 5, 16),
            torch.tensor([3]),
        )


def test_generalist_mode_context_injection_preserves_cfg_negative_branch() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    core.configure_generalist_mode_context_encoder(enabled=True)
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        frame_chunk_size=2,
        action_per_frame=2,
        generalist_mode_text_token=True,
    )
    text_emb = torch.randn(1, 4, 16)
    negative_text_emb = torch.randn(1, 4, 16)

    appended, appended_negative = append_generalist_mode_text_context(
        core,
        policy_config=policy_config,
        text_emb=text_emb,
        negative_text_emb=negative_text_emb,
        mode=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
    )

    assert appended.shape == (1, 5, 16)
    assert appended_negative is not None
    assert appended_negative.shape == (1, 5, 16)
    assert torch.equal(appended[:, :4], text_emb)
    assert torch.equal(appended_negative[:, :4], negative_text_emb)
    assert not torch.equal(appended[:, 4:], torch.zeros_like(appended[:, 4:]))
    assert not torch.equal(
        appended_negative[:, 4:], torch.zeros_like(appended_negative[:, 4:])
    )
    torch.testing.assert_close(appended[:, 4:], appended_negative[:, 4:])
    assert (
        owner_dynamics_objectives.resolve_dynamics_objective(
            "action_conditioned_video"
        )
        == DynamicsObjective.ACTION_CONDITIONED_VIDEO
    )
    assert (
        owner_dynamics_objectives.resolve_dynamics_objective("joint")
        == DynamicsObjective.JOINT
    )


@pytest.mark.parametrize(
    ("rollout_mode", "expected_mode"),
    [
        ("joint", DynamicsObjective.JOINT),
        ("action_conditioned_video", DynamicsObjective.ACTION_CONDITIONED_VIDEO),
        ("video_conditioned_action", DynamicsObjective.VIDEO_CONDITIONED_ACTION),
    ],
)
def test_generalist_mode_context_maps_rollout_modes(
    rollout_mode: str,
    expected_mode: DynamicsObjective,
) -> None:
    assert (
        owner_dynamics_objectives.resolve_dynamics_objective(rollout_mode)
        == expected_mode
    )


def test_generalist_mode_context_rejects_unknown_rollout_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported dynamics objective"):
        owner_dynamics_objectives.resolve_dynamics_objective(
            "unknown_rollout_mode"
        )


def test_generalist_conditional_local_window_sees_one_previous_video_frame_only() -> (
    None
):
    profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 8, 1, 1),
        action_shape=(1, 1, 8, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=3,
        patch_size=(1, 1, 1),
        text_token_count=0,
        chunk_origin_frame=0,
        device=torch.device("cpu"),
        build_dense_masks=True,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
    )
    assert profile.self_attention_mask is not None
    mask = profile.self_attention_mask
    latent_tokens = 8
    action_tokens = 8
    current_video_noisy_frame4 = 4
    current_video_clean_frame4 = latent_tokens + 4
    current_action_noisy_frame4 = 2 * latent_tokens + 4
    previous_video_clean_frame3 = latent_tokens + 3
    older_video_clean_frame2 = latent_tokens + 2
    previous_action_clean_frame3 = 2 * latent_tokens + action_tokens + 3
    current_action_clean_frame4 = 2 * latent_tokens + action_tokens + 4

    assert mask[current_video_noisy_frame4, previous_video_clean_frame3]
    assert not mask[current_video_noisy_frame4, older_video_clean_frame2]
    assert not mask[current_video_noisy_frame4, previous_action_clean_frame3]
    assert mask[current_action_noisy_frame4, previous_video_clean_frame3]
    assert not mask[current_action_noisy_frame4, older_video_clean_frame2]
    assert not mask[current_action_noisy_frame4, previous_action_clean_frame3]
    assert not mask[current_video_noisy_frame4, current_video_clean_frame4]
    assert not mask[current_action_noisy_frame4, current_action_clean_frame4]

    too_narrow_profile = build_chunked_temporal_exact_attention_profile(
        latent_shape=(1, 1, 8, 1, 1),
        action_shape=(1, 1, 8, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=2,
        patch_size=(1, 1, 1),
        text_token_count=0,
        chunk_origin_frame=0,
        device=torch.device("cpu"),
        build_dense_masks=True,
        current_block_coupling=CurrentBlockCoupling.JOINT,
        history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
    )
    assert too_narrow_profile.self_attention_mask is not None
    assert not too_narrow_profile.self_attention_mask[
        current_action_noisy_frame4, previous_video_clean_frame3
    ]


def test_generalist_conditional_rollout_modes_use_one_frame_history_window() -> None:
    inference_config = InferenceConfig(attention_window_size=30)

    assert (
        owner_dynamics_objectives.dynamics_objective_attention_window_size(
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            fallback_window_size=inference_config.attention_window_size,
        )
        == 3
    )
    assert (
        owner_dynamics_objectives.dynamics_objective_attention_window_size(
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
            fallback_window_size=inference_config.attention_window_size,
        )
        == 3
    )
    assert (
        owner_dynamics_objectives.dynamics_objective_rollout_chunk_size(
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            fallback_chunk_size=4,
        )
        == 1
    )
    assert (
        owner_dynamics_objectives.dynamics_objective_attention_window_size(
            DynamicsObjective.JOINT,
            fallback_window_size=inference_config.attention_window_size,
        )
        == 30
    )


def test_generalist_mode_context_requires_configured_encoder() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        frame_chunk_size=2,
        action_per_frame=2,
        generalist_mode_text_token=True,
    )

    with pytest.raises(ValueError, match="append exactly one token"):
        append_generalist_mode_text_context(
            core,
            policy_config=policy_config,
            text_emb=torch.randn(1, 4, 16),
            negative_text_emb=None,
            mode=DynamicsObjective.JOINT,
        )




def test_visual_tower_configures_deprecated_text_token_proprio_encoder_before_runtime_load() -> (
    None
):
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    tower = VisualTower(backbone_config, action_dim=4, state_dim=8)
    tower.configure_policy_conditioning(
        proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,
        dynamics_mode_context_enabled=False,
    )

    assert isinstance(tower.core, SharedVideoTransformerCore)
    assert tower.core.proprio_context_encoder is not None
    assert "proprio_context_encoder.proj.weight" in tower.core.state_dict()


def test_parallel_variant_configures_generalist_mode_encoder() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        frame_chunk_size=2,
        action_per_frame=2,
        generalist_mode_text_token=True,
        proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        backbone_config,
        TrainingConfig(chunk_size=2, window_size=8),
        InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )
    tower = VisualTower(backbone_config, action_dim=4, state_dim=8)

    requirements = variant.pipeline_requirements(
        default_action_dim=4,
        default_action_horizon=4,
        default_state_dim=8,
    )
    assert requirements.action_dim == 4
    assert requirements.action_horizon == 4
    tower.configure_policy_conditioning(
        proprio_context_mode=requirements.proprio_context_mode,
        dynamics_mode_context_enabled=requirements.dynamics_mode_context_enabled,
    )

    assert tower.core.generalist_mode_context_encoder is not None
    assert tower.core.proprio_context_encoder is not None
    assert "generalist_mode_context_encoder.embedding.weight" in tower.core.state_dict()


def test_visual_tower_loads_exported_generalist_mode_encoder_when_preconfigured(
    tmp_path: Path,
) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        pretrained_model_name_or_path=str(tmp_path),
        load_reference_core_weights=True,
    )
    probe_core = SharedVideoTransformerCore(backbone_config, action_dim=4, state_dim=8)
    probe_core.configure_generalist_mode_context_encoder(enabled=True)
    exported_state = {
        # Marks the safetensors file as an OpenWAM exported runtime backbone.
        "time_conditioner.time_proj.weight": probe_core.state_dict()[
            "time_conditioner.time_proj.weight"
        ].clone(),
        "generalist_mode_context_encoder.embedding.weight": torch.arange(
            3 * 16,
            dtype=torch.float32,
        ).reshape(3, 16),
    }
    save_file(exported_state, transformer_dir / "diffusion_pytorch_model.safetensors")

    tower = VisualTower(backbone_config, action_dim=4, state_dim=8)
    tower.configure_policy_conditioning(
        proprio_context_mode=ProprioContextMode.NONE,
        dynamics_mode_context_enabled=True,
    )
    tower.initialize_configured_weights()

    assert tower.core.generalist_mode_context_encoder is not None
    assert torch.equal(
        tower.core.generalist_mode_context_encoder.embedding.weight.detach().cpu(),
        exported_state["generalist_mode_context_encoder.embedding.weight"],
    )
    assert "generalist_mode_context_encoder.embedding.weight" in (
        tower.reference_core_load_report.loaded_keys
    )


def test_deprecated_proprio_context_appending_preserves_existing_text_tokens() -> None:
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        action_dim=4,
        state_dim=8,
    )
    core.configure_proprio_context_encoder(enabled=True, state_dim=8)

    text_emb = torch.ones(2, 5, 16)
    appended = core.append_proprio_context_tokens(
        text_emb, torch.randn(2, 8)
    )  # deprecated helper

    assert appended.shape == (2, 6, 16)
    assert torch.equal(appended[:, :5], text_emb)


def test_deprecated_proprio_context_appending_runs_exact_single_stream_forward() -> (
    None
):
    torch.manual_seed(0)
    core = SharedVideoTransformerCore(
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
            patch_size_t=1,
            patch_size_h=2,
            patch_size_w=2,
        ),
        action_dim=4,
        state_dim=8,
    ).eval()
    text_emb = torch.randn(1, 6, 16)
    input_dict = {
        "noisy_latents": torch.randn(1, 48, 1, 2, 2),
        "text_emb": text_emb,
        "grid_id": torch.zeros(1, 4, 1),
        "timesteps": torch.zeros(1, 1),
    }

    core.configure_proprio_context_encoder(enabled=True, state_dim=8)
    appended_input = dict(input_dict)
    appended_input["text_emb"] = (
        core.append_proprio_context_tokens(  # deprecated helper
            text_emb,
            torch.randn(1, 8),
        )
    )
    with_proprio = run_reference_single_stream_forward(
        core,
        input_dict=appended_input,
        update_cache=0,
        cache_name="parity",
        action_mode=False,
        guidance_scale=1.0,
        negative_text_emb=None,
    )

    assert with_proprio.shape == (1, 4, 48)
    assert torch.isfinite(with_proprio).all()




def test_deprecated_parallel_stream_text_token_proprio_adds_state_to_train_artifacts() -> (
    None
):
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        backbone_config,
        TrainingConfig(chunk_size=2, window_size=8),
        InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        action_horizon=4,
        num_frames=2,
    )
    video_latents = torch.randn(1, 48, 2, 2, 2)
    visual_outputs = VisualStageOutputs(
        frontend=VisualFrontendOutput(
            canonical_video=torch.empty(1, 3, 2, 8, 8),
            video_latents=video_latents,
            video_tokens=torch.empty(1, 0, 32),
            input_source="latents",
            token_grid=TokenGridMetadata(
                num_frames=2,
                latent_height=2,
                latent_width=2,
                patch_size=(1, 1, 1),
                patches_per_frame_h=2,
                patches_per_frame_w=2,
                tokens_per_frame=4,
                sequence_length=8,
            ),
            chunk=ChunkMetadata(
                chunk_start_frame=0,
                chunk_num_frames=2,
                frame_stride=1,
                chunk_type="test",
            ),
            conditioning=ConditioningState(
                supported=True,
                text_context=torch.zeros(1, 512, 16),
            ),
        )
    )
    proprio_context_state = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    proprio_context_state_mask = torch.ones_like(proprio_context_state)
    proprio_context_state_mask[:, 1, 3:] = 0
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 4, 4),
        state=torch.full((1, 2, 8), -1.0),
        extra={
            "proprio_context_state": proprio_context_state,
            "proprio_context_state_mask": proprio_context_state_mask,
        },
    )

    prepared = variant.prepare_train_inputs(visual_outputs, batch)
    artifacts = prepared.variant_inputs["parallel_train_artifacts"]

    torch.testing.assert_close(
        artifacts.input_dict["proprio_state"],
        proprio_context_state * proprio_context_state_mask,
    )


























def test_action_conditioned_reference_profile_validates_inference_step_counts() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )

    with pytest.raises(ValueError, match="action_num_inference_steps"):
        ParallelStreamPolicyVariant(
            ParallelStreamPolicyConfig(
                hidden_size=32,
                program=VideoActionProgram.JOINT,
                reference_profile="libero_joint",
                frame_chunk_size=4,
                action_per_frame=4,
            ),
            backbone_config=backbone_config,
            training_config=TrainingConfig(chunk_size=4, window_size=30),
            inference_config=InferenceConfig(
                frame_chunk_size=4,
                video_num_inference_steps=20,
                action_num_inference_steps=50,
                guidance_scale=5.0,
                action_guidance_scale=1.0,
            ),
            action_dim=30,
            action_horizon=16,
            num_frames=4,
        )


def test_reference_profile_validates_effective_session_geometry() -> None:
    variant = ParallelStreamPolicyVariant(
        ParallelStreamPolicyConfig(
            hidden_size=32,
            program=VideoActionProgram.JOINT,
            reference_profile="libero_joint",
            frame_chunk_size=4,
            action_per_frame=4,
        ),
        backbone_config=LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
        ),
        training_config=TrainingConfig(chunk_size=4, window_size=30),
        inference_config=InferenceConfig(
            frame_chunk_size=4,
            attention_window_size=30,
            video_num_inference_steps=20,
            action_num_inference_steps=20,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
        ),
        action_dim=30,
        action_horizon=16,
        num_frames=4,
    )

    with pytest.raises(ValueError, match="attn_window"):
        variant._resolve_inference_config(
            PolicyTemporalGeometry(
                frame_chunk_size=4,
                attention_window_size=17,
            )
        )






def test_exact_train_artifacts_default_to_flex_attention_profile() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        train_attn_mode=None,
        infer_attn_mode=None,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    video_latents = torch.randn(1, 48, 2, 8, 8)
    actions = torch.randn(1, 4, 4)
    action_mask = torch.ones_like(actions, dtype=torch.bool)
    text_emb = torch.randn(1, 512, 16)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        text_emb=text_emb,
    )

    assert artifacts.input_dict["attention_profile_name"] == "chunked_temporal_exact"


def test_generalist_action_conditioned_override_drops_text_and_masks_action_loss() -> (
    None
):
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        implementation="shared_transformer",
        attn_mode="torch",
        train_attn_mode=None,
        infer_attn_mode=None,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        frame_chunk_size=2,
        action_per_frame=2,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=8)
    video_latents = torch.randn(1, 48, 2, 8, 8)
    actions = torch.randn(1, 4, 4)
    action_mask = torch.ones_like(actions, dtype=torch.bool)
    text_emb = torch.randn(1, 512, 16)

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=action_mask,
        text_emb=text_emb,
        sample_metadata=_generalist_sample_metadata(
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            source="counterfactual_dynamics",
            drop_text=True,
        ),
    )

    assert (
        artifacts.input_dict["joint_denoise_training_mode"]
        == "action_conditioned_video"
    )
    assert (
        artifacts.input_dict["joint_denoise_training_mode_override"]
        == "action_conditioned_video"
    )
    assert artifacts.input_dict["joint_denoise_text_dropped"] is True
    assert (
        artifacts.input_dict["generalist_training_source"] == "counterfactual_dynamics"
    )
    assert torch.equal(
        artifacts.input_dict["latent_dict"]["text_emb"], torch.zeros_like(text_emb)
    )
    assert torch.equal(
        artifacts.input_dict["action_dict"]["text_emb"], torch.zeros_like(text_emb)
    )
    assert artifacts.input_dict["action_dict"]["loss_mask"].sum().item() == 0
    assert artifacts.input_dict["latent_dict"]["loss_mask"].sum().item() > 0




def test_reference_single_stream_forward_runs_in_inference_mode() -> None:
    transformer = _GradTrackingTransformer()
    input_dict = {
        "noisy_latents": torch.randn(1, 1, 1, 1, 1),
        "text_emb": torch.zeros(1, 226, 16),
        "grid_id": torch.zeros(1, 4, 1),
        "timesteps": torch.zeros(1, 1),
    }

    output = run_reference_single_stream_forward(
        transformer,
        input_dict=input_dict,
        update_cache=0,
        cache_name="test",
        action_mode=False,
        guidance_scale=1.0,
        negative_text_emb=None,
    )

    assert transformer.grad_enabled_during_forward is False
    assert output.requires_grad is False


def test_parallel_exact_train_artifacts_accept_contextual_overrides() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=4,
        action_per_frame=4,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 8, 8, 16)
    actions = torch.randn(1, 32, 30)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
        loss_frame_start=4,
        loss_frame_end=8,
        frame_shift=7,
    )

    assert artifacts.input_dict["chunk_size"] == 2
    assert artifacts.input_dict["window_size"] == 4
    assert artifacts.input_dict["loss_frame_start"] == 4
    assert artifacts.input_dict["loss_frame_end"] == 8
    assert artifacts.input_dict["frame_shift"] == 7
    assert artifacts.input_dict["latent_dict"]["loss_mask"][:, :, :4].sum().item() == 0
    assert torch.all(artifacts.input_dict["latent_dict"]["loss_mask"][:, :, 4:8] == 1)
    assert artifacts.input_dict["action_dict"]["loss_mask"][:, :, :4].sum().item() == 0
    assert torch.all(artifacts.input_dict["action_dict"]["loss_mask"][:, :, 4:8] == 1)
    assert float(artifacts.input_dict["latent_dict"]["grid_id"][0, 0, 0].item()) == 7.0
    assert torch.isclose(
        artifacts.input_dict["action_dict"]["grid_id"][0, 0, 0],
        torch.tensor(7.2),
    )


def test_parallel_exact_train_artifacts_prefer_full_condition_latents() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        noisy_video_condition_prob=0.0,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.zeros(1, 48, 2, 8, 16)
    condition_latents = torch.full_like(video_latents, 4.0)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=torch.randn(1, 4, 30),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    assert artifacts.input_dict["video_condition_source"] == "condition_latents"
    torch.testing.assert_close(
        artifacts.input_dict["latent_dict"]["latent"], condition_latents, rtol=0, atol=0
    )


def test_parallel_exact_train_artifacts_can_use_single_frame_context_condition_latents() -> (
    None
):
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        noisy_video_condition_prob=0.0,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.zeros(1, 48, 3, 8, 16)
    video_latents[:, :, 1:] = 2.0
    condition_latents = torch.full_like(video_latents, 7.0)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=torch.randn(1, 6, 30),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
        loss_frame_start=1,
        loss_frame_end=3,
    )

    latent_dict = artifacts.input_dict["latent_dict"]
    assert artifacts.input_dict["video_condition_source"] == "context_condition_latents"
    torch.testing.assert_close(
        latent_dict["latent"][:, :, :1], condition_latents[:, :, :1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        latent_dict["latent"][:, :, 1:], video_latents[:, :, 1:], rtol=0, atol=0
    )
    assert torch.equal(
        latent_dict["cond_timesteps"][:, :1],
        torch.zeros_like(latent_dict["cond_timesteps"][:, :1]),
    )


def test_parallel_exact_train_artifacts_require_single_frame_context_condition_latents() -> (
    None
):
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=4)

    with pytest.raises(
        ValueError, match="single_frame_condition_latent.*requires `condition_latents`"
    ):
        prepare_parallel_exact_train_artifacts(
            backbone_config=backbone_config,
            policy_config=policy_config,
            training_config=training_config,
            video_latents=torch.zeros(1, 48, 3, 8, 16),
            condition_latents=None,
            actions=torch.randn(1, 6, 30),
            action_mask=None,
            text_emb=torch.randn(1, 512, 16),
            chunk_size_override=2,
            window_size_override=4,
            loss_frame_start=1,
            loss_frame_end=3,
        )


def test_parallel_prefix_condition_train_artifacts_match_legacy_prefix_semantics() -> (
    None
):
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
        noisy_video_condition_prob=1.0,
    )
    training_config = TrainingConfig(chunk_size=2, window_size=4)
    video_latents = torch.randn(1, 3, 4, 2, 2)
    condition_latents = torch.full_like(video_latents, 9.0)
    actions = torch.randn(1, 8, 5)

    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
        frame_shift=5,
    )

    latent_dict = artifacts.input_dict["latent_dict"]
    action_dict = artifacts.input_dict["action_dict"]
    assert latent_dict["noisy_latents"].shape[2] == 5
    assert action_dict["noisy_latents"].shape[2] == 4
    torch.testing.assert_close(
        latent_dict["noisy_latents"][:, :, :1], condition_latents[:, :, :1]
    )
    torch.testing.assert_close(
        latent_dict["latent"][:, :, :1], condition_latents[:, :, :1]
    )
    assert latent_dict["cond_timesteps"][:, :1].sum().item() == 0
    assert latent_dict["cond_timesteps"][:, 1:].sum().item() > 0
    assert not torch.allclose(latent_dict["latent"][:, :, 1:], video_latents)
    assert latent_dict["loss_mask"][:, :, :1].sum().item() == 0
    assert latent_dict["loss_mask"][:, :, 1:].sum().item() > 0
    assert action_dict["loss_mask"].sum().item() == action_dict["loss_mask"].numel()
    assert artifacts.input_dict["prefix_condition_frames"] == 1
    assert artifacts.input_dict["latent_loss_frame_start"] == 1
    assert artifacts.input_dict["action_loss_frame_start"] == 0
    assert artifacts.input_dict["frame_shift"] == 5


def test_parallel_prefix_condition_train_artifacts_honor_shared_video_schedule() -> (
    None
):
    torch.manual_seed(17)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        frame_chunk_size=2,
        action_per_frame=2,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
        joint_timestep_coupling=JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=500,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        condition_latents=torch.randn(1, 3, 1, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    input_dict = artifacts.input_dict
    video_target_timesteps = input_dict["latent_dict"]["timesteps"][0, 1:]
    action_timesteps = input_dict["action_dict"]["timesteps"][0]
    video_target_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        video_target_timesteps
    )
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(action_timesteps)

    assert (
        input_dict["joint_timestep_coupling"]
        == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE.value
    )
    assert input_dict["coupled_action_video_timesteps"] is True
    assert input_dict["latent_dict"]["timesteps"][0, 0].item() == 0
    torch.testing.assert_close(action_timesteps, video_target_timesteps)
    torch.testing.assert_close(action_sigmas, video_target_sigmas)


def test_parallel_prefix_condition_train_artifacts_honor_match_sigma_coupling() -> None:
    torch.manual_seed(19)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        condition_latents=torch.randn(1, 3, 1, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    input_dict = artifacts.input_dict
    video_target_timesteps = input_dict["latent_dict"]["timesteps"][0, 1:]
    action_timesteps = input_dict["action_dict"]["timesteps"][0]
    video_target_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        video_target_timesteps
    )
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(action_timesteps)

    assert (
        input_dict["joint_timestep_coupling"] == JointTimestepCoupling.MATCH_SIGMA.value
    )
    assert input_dict["coupled_action_video_timesteps"] is True
    assert input_dict["latent_dict"]["timesteps"][0, 0].item() == 0
    torch.testing.assert_close(action_sigmas, video_target_sigmas, atol=2e-3, rtol=0.0)


def test_parallel_prefix_condition_generalist_joint_is_pure_joint_metadata() -> None:
    torch.manual_seed(23)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        frame_chunk_size=2,
        action_per_frame=2,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        condition_latents=torch.randn(1, 3, 1, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    input_dict = artifacts.input_dict
    assert input_dict["prefix_condition_frames"] == 1
    assert input_dict["joint_denoise_training_mode"] == DynamicsObjective.JOINT.value
    assert "generalist_denoising_mode_probs" not in input_dict
    assert input_dict["video_condition_source"] == "condition_latents_prefix"
    assert input_dict["joint_denoise_shared_sigmas"].shape == (4,)


def test_parallel_generalist_sequence_contract_accepts_routed_conditional_modes() -> (
    None
):
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        frame_chunk_size=2,
        action_per_frame=2,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
    )

    assert policy_config.sequence_contract == (
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )


def test_legacy_prefix_variant_preserves_frame_aligned_proprio_state() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        backbone_config,
        TrainingConfig(chunk_size=2, window_size=4),
        InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        action_horizon=8,
        num_frames=4,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    visual_outputs = VisualStageOutputs(
        frontend=VisualFrontendOutput(
            canonical_video=torch.empty(1, 3, 4, 8, 8),
            video_latents=video_latents,
            video_tokens=torch.empty(1, 0, 32),
            input_source="latents",
            token_grid=TokenGridMetadata(
                num_frames=4,
                latent_height=2,
                latent_width=2,
                patch_size=(1, 1, 1),
                patches_per_frame_h=2,
                patches_per_frame_w=2,
                tokens_per_frame=4,
                sequence_length=16,
            ),
            chunk=ChunkMetadata(
                chunk_start_frame=0,
                chunk_num_frames=4,
                frame_stride=1,
                chunk_type="test",
            ),
            conditioning=ConditioningState(
                supported=True,
                text_context=torch.zeros(1, 512, 16),
            ),
        )
    )
    prefix_state = torch.full((1, 8), 9.0)
    frame_state = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 8, 4),
        state=prefix_state,
        extra={
            "condition_latents": torch.full_like(video_latents, 3.0),
            "proprio_context_frames": frame_state,
            "metadata": ({"sampled_chunk_size": 2, "sampled_window_size": 4},),
        },
    )

    prepared = variant.prepare_train_inputs(visual_outputs, batch)
    input_dict = prepared.variant_inputs["parallel_train_artifacts"].input_dict

    assert input_dict["per_chunk_proprio_state_granularity"] == "frame"
    assert input_dict["per_chunk_proprio_state"].shape == (1, 5, 8)
    torch.testing.assert_close(
        input_dict["per_chunk_proprio_state"][:, :1], prefix_state[:, None]
    )
    torch.testing.assert_close(
        input_dict["per_chunk_proprio_state"][:, 1:], frame_state
    )


@pytest.mark.parametrize(
    ("mode", "uses_planning_prefix"),
    (
        (DynamicsObjective.JOINT, True),
        (DynamicsObjective.ACTION_CONDITIONED_VIDEO, False),
        (DynamicsObjective.VIDEO_CONDITIONED_ACTION, False),
    ),
)
def test_parallel_gjd_routes_planning_and_conditional_layouts_by_mode(
    mode: DynamicsObjective,
    uses_planning_prefix: bool,
) -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        frame_chunk_size=2,
        action_per_frame=2,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        proprio_context_mode=ProprioContextMode.PER_CHUNK_ADDITIVE,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        history_stream_visibility=HistoryStreamVisibility.VIDEO_ONLY,
        use_condition_latents=True,
        require_condition_latents=True,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        backbone_config,
        TrainingConfig(chunk_size=2, window_size=4),
        InferenceConfig(frame_chunk_size=2),
        action_dim=4,
        action_horizon=8,
        num_frames=4,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    text_context = torch.ones(1, 512, 16)
    visual_outputs = VisualStageOutputs(
        frontend=VisualFrontendOutput(
            canonical_video=torch.empty(1, 3, 4, 8, 8),
            video_latents=video_latents,
            video_tokens=torch.empty(1, 0, 32),
            input_source="latents",
            token_grid=TokenGridMetadata(
                num_frames=4,
                latent_height=2,
                latent_width=2,
                patch_size=(1, 1, 1),
                patches_per_frame_h=2,
                patches_per_frame_w=2,
                tokens_per_frame=4,
                sequence_length=16,
            ),
            chunk=ChunkMetadata(
                chunk_start_frame=0,
                chunk_num_frames=4,
                frame_stride=1,
                chunk_type="test",
            ),
            conditioning=ConditioningState(
                supported=True,
                text_context=text_context,
            ),
        )
    )
    conditional = mode != DynamicsObjective.JOINT
    metadata = {
        **_generalist_sample_metadata(mode, drop_text=conditional).raw,
        "sampled_chunk_size": 2,
        "sampled_window_size": 4,
        "context_prefix_frames_in_sample": 0,
    }
    action_mask = torch.ones(1, 8, 4)
    if conditional:
        metadata.update(
            {
                "context_prefix_frames_in_sample": 1,
                "loss_frame_end": 4,
                "latent_loss_frame_end": 4,
                "action_loss_frame_end": 4,
            }
        )
        action_mask[:, :2] = 0
    batch = PolicyTrainBatch(
        actions=torch.randn(1, 8, 4),
        action_mask=action_mask,
        state=torch.full((1, 1, 8), 9.0),
        extra={
            "condition_latents": torch.full_like(video_latents, 3.0),
            "proprio_context_frames": torch.randn(1, 4, 8),
            "metadata": (metadata,),
        },
    )

    prepared = variant.prepare_train_inputs(visual_outputs, batch)
    input_dict = prepared.variant_inputs["parallel_train_artifacts"].input_dict

    assert input_dict["joint_denoise_training_mode"] == mode.value
    assert bool(input_dict.get("prefix_condition_frames")) is uses_planning_prefix
    assert input_dict["latent_dict"]["noisy_latents"].shape[2] == (
        5 if uses_planning_prefix else 4
    )
    assert input_dict["joint_denoise_text_dropped"] is conditional
    assert (
        bool(torch.count_nonzero(input_dict["latent_dict"]["text_emb"]) == 0)
        is conditional
    )
    if conditional:
        assert input_dict["singleton_chunk_frame"] == 0
        assert input_dict["chunk_origin_frame"] == 1
        assert input_dict["generalist_conditional_history_chunks"] == 1
        assert (
            input_dict["history_stream_visibility"]
            == HistoryStreamVisibility.VIDEO_ONLY.value
        )
    else:
        assert input_dict["per_chunk_proprio_state"].shape[1] == 5


def test_parallel_exact_train_artifacts_split_video_and_action_loss_masks() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=2,
        patch_size_w=2,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=4,
        action_per_frame=4,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 8, 8, 16)
    actions = torch.randn(1, 32, 30)

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        latent_loss_frame_start=0,
        latent_loss_frame_end=5,
        action_loss_frame_start=0,
        action_loss_frame_end=8,
    )

    assert artifacts.input_dict["loss_frame_start"] == 0
    assert artifacts.input_dict["loss_frame_end"] == 8
    assert artifacts.input_dict["latent_loss_frame_start"] == 0
    assert artifacts.input_dict["latent_loss_frame_end"] == 5
    assert artifacts.input_dict["action_loss_frame_start"] == 0
    assert artifacts.input_dict["action_loss_frame_end"] == 8
    assert torch.all(artifacts.input_dict["latent_dict"]["loss_mask"][:, :, :5] == 1)
    assert artifacts.input_dict["latent_dict"]["loss_mask"][:, :, 5:].sum().item() == 0
    assert torch.all(artifacts.input_dict["action_dict"]["loss_mask"] == 1)


def test_parallel_stream_decoder_ignores_history_frames_outside_loss_mask() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 3, 2, 1, 1)
    actions = torch.randn(1, 4, 5)
    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        loss_frame_start=1,
        loss_frame_end=2,
    )

    target_action_pred = (
        artifacts.input_dict["action_dict"]["targets"]
        .squeeze(-1)
        .permute(0, 2, 3, 1)
        .reshape(1, 4, 5)
    )
    corrupted_action_pred = target_action_pred.clone()
    corrupted_action_pred[:, :2] += 100.0

    target_latent_pred = (
        artifacts.input_dict["latent_dict"]["targets"]
        .permute(0, 2, 3, 4, 1)
        .reshape(1, 2, 3)
    )
    corrupted_latent_pred = target_latent_pred.clone()
    corrupted_latent_pred[:, :1] += 100.0

    decoder = ParallelStreamActionDecoder(
        hidden_size=32, action_dim=5, action_horizon=4
    )
    output = decoder.forward_train(
        PolicyTrainOutput(
            policy_features=corrupted_action_pred,
            metrics={},
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT,
                payload=ParallelDecoderTrainArtifacts(
                    latent_pred=corrupted_latent_pred,
                    runtime=artifacts,
                    loss_weights={"latent": 0.0, "action": 1.0},
                    patch_size=(1, 1, 1),
                ),
                dynamics_objective=DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            ),
        ),
        PolicyTrainBatch(actions=actions),
    )

    assert torch.isclose(output.loss, torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(output.metrics["action_mse"], torch.tensor(0.0), atol=1e-5)


def test_parallel_stream_decoder_accepts_prefix_video_action_frame_mismatch() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        sequence_contract=VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
        context_condition_latent_source=ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        use_condition_latents=True,
        require_condition_latents=True,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=4,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    condition_latents = torch.full_like(video_latents, 9.0)
    actions = torch.randn(1, 8, 5)
    artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        condition_latents=condition_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=4,
    )

    assert artifacts.input_dict["latent_dict"]["timesteps"].shape == (1, 5)
    assert artifacts.input_dict["action_dict"]["timesteps"].shape == (1, 4)
    target_action_pred = (
        artifacts.input_dict["action_dict"]["targets"]
        .squeeze(-1)
        .permute(0, 2, 3, 1)
        .reshape(1, 8, 5)
    )
    target_latent_pred = (
        artifacts.input_dict["latent_dict"]["targets"]
        .permute(0, 2, 3, 4, 1)
        .reshape(1, 20, 3)
    )

    decoder = ParallelStreamActionDecoder(
        hidden_size=32, action_dim=5, action_horizon=8
    )
    output = decoder.forward_train(
        PolicyTrainOutput(
            policy_features=target_action_pred,
            metrics={},
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT,
                payload=ParallelDecoderTrainArtifacts(
                    latent_pred=target_latent_pred,
                    runtime=artifacts,
                    loss_weights={"latent": 1.0, "action": 1.0},
                    patch_size=(1, 1, 1),
                ),
            ),
        ),
        PolicyTrainBatch(actions=actions),
    )

    assert torch.isfinite(output.loss)
    assert torch.isclose(output.loss, torch.tensor(0.0), atol=1e-5)


def test_parallel_action_conditioned_train_artifacts_accept_contextual_overrides() -> (
    None
):
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.JOINT,
        frame_chunk_size=4,
        action_per_frame=4,
        video_action_condition_source="noisy_action",
        history_stream_visibility="video_only",
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 48, 6, 8, 8),
        actions=torch.randn(1, 24, 30),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        window_size_override=5,
        loss_frame_start=4,
        loss_frame_end=6,
        frame_shift=9,
        chunk_origin_frame=4,
        singleton_chunk_frame=3,
    )

    assert artifacts.input_dict["chunk_size"] == 2
    assert artifacts.input_dict["window_size"] == 5
    assert artifacts.input_dict["loss_frame_start"] == 4
    assert artifacts.input_dict["loss_frame_end"] == 6
    assert artifacts.input_dict["frame_shift"] == 9
    assert artifacts.input_dict["chunk_origin_frame"] == 4
    assert artifacts.input_dict["singleton_chunk_frame"] == 3
    assert (
        artifacts.input_dict["attention_profile_name"] == "chunked_temporal_exact_joint"
    )








def test_slot_pool_deferred_eviction_keeps_prefix_visible_during_update_attention() -> (
    None
):
    layer_state = SlotPoolLayerState(
        slot_ids=torch.arange(4, dtype=torch.long),
        slot_mask=torch.ones(4, dtype=torch.bool),
    )
    valid = torch.arange(4, dtype=torch.long)

    retained_without_defer = _retained_slot_pool_indices_for_current_write(
        layer_state,
        valid=valid,
        current_token_count=4,
        update_mode=1,
    )
    assert retained_without_defer.numel() == 0

    layer_state.metadata[SLOT_POOL_DEFER_EVICTION_UNTIL_AFTER_WRITE_ATTENTION] = True
    retained_with_defer = _retained_slot_pool_indices_for_current_write(
        layer_state,
        valid=valid,
        current_token_count=4,
        update_mode=1,
    )
    assert torch.equal(retained_with_defer, valid)










def test_parallel_action_conditioned_train_artifacts_can_force_clean_video_condition() -> (
    None
):
    torch.manual_seed(0)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.JOINT,
        frame_chunk_size=4,
        action_per_frame=4,
        video_action_condition_source="noisy_action",
        noisy_video_condition_prob=1.0,
    )
    training_config = TrainingConfig(
        chunk_size=4,
        window_size=64,
        video_num_train_timesteps=10,
        action_num_train_timesteps=10,
    )
    video_latents = torch.randn(1, 48, 6, 8, 8)
    actions = torch.randn(1, 24, 30)

    augmented = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    forced_clean = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        force_clean_video_condition=True,
    )

    assert (
        torch.count_nonzero(augmented.input_dict["latent_dict"]["cond_timesteps"]) > 0
    )
    assert (
        torch.count_nonzero(forced_clean.input_dict["latent_dict"]["cond_timesteps"])
        == 0
    )
    assert torch.allclose(
        forced_clean.input_dict["latent_dict"]["latent"], video_latents
    )
    assert forced_clean.input_dict["force_clean_video_condition"] is True




def test_standard_joint_training_couples_video_and_action_noise_clarity() -> None:
    torch.manual_seed(11)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=JointTimestepCoupling.MATCH_SIGMA,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    actions = torch.randn(1, 8, 5)

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict
    video_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        input_dict["latent_dict"]["timesteps"][0]
    )
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(
        input_dict["action_dict"]["timesteps"][0]
    )

    assert input_dict["coupled_action_video_timesteps"] is True
    assert torch.allclose(video_sigmas, action_sigmas, atol=2e-3, rtol=0.0)


def test_standard_joint_training_can_share_video_scheduler_clock() -> None:
    torch.manual_seed(13)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=500,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict
    video_timesteps = input_dict["latent_dict"]["timesteps"][0]
    action_timesteps = input_dict["action_dict"]["timesteps"][0]
    video_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(video_timesteps)
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(action_timesteps)
    video_weights = artifacts.latent_scheduler.training_weight(
        video_timesteps.flatten()
    )
    action_weights = artifacts.action_scheduler.training_weight(
        action_timesteps.flatten()
    )

    assert (
        input_dict["joint_timestep_coupling"]
        == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE.value
    )
    assert input_dict["coupled_action_video_timesteps"] is True
    torch.testing.assert_close(action_timesteps, video_timesteps)
    torch.testing.assert_close(action_sigmas, video_sigmas)
    torch.testing.assert_close(action_weights, video_weights)


def test_standard_joint_training_can_match_scheduler_index_without_matching_sigma() -> (
    None
):
    torch.manual_seed(12)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.JOINT,
        frame_chunk_size=2,
        action_per_frame=2,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=JointTimestepCoupling.MATCH_INDEX,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
        video_sigma_shift=5.0,
        action_sigma_shift=1.0,
    )

    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    input_dict = artifacts.input_dict
    video_timesteps = input_dict["latent_dict"]["timesteps"][0]
    action_timesteps = input_dict["action_dict"]["timesteps"][0]
    video_ids = torch.argmin(
        (artifacts.latent_scheduler.timesteps[:, None] - video_timesteps[None]).abs(),
        dim=0,
    )
    action_ids = torch.argmin(
        (artifacts.action_scheduler.timesteps[:, None] - action_timesteps[None]).abs(),
        dim=0,
    )
    video_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(video_timesteps)
    action_sigmas = artifacts.action_scheduler.sigma_for_timesteps(action_timesteps)

    assert (
        input_dict["joint_timestep_coupling"] == JointTimestepCoupling.MATCH_INDEX.value
    )
    assert input_dict["coupled_action_video_timesteps"] is False
    assert torch.equal(video_ids, action_ids)
    assert not torch.allclose(video_sigmas, action_sigmas, atol=2e-3, rtol=0.0)


def test_video_then_action_uses_declared_independent_noise_schedule() -> None:
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = ParallelStreamPolicyConfig(
        hidden_size=32,
        program=VideoActionProgram.VIDEO_THEN_ACTION,
        frame_chunk_size=2,
        action_per_frame=2,
        joint_timestep_coupling=JointTimestepCoupling.INDEPENDENT,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=1000,
        action_num_train_timesteps=1000,
    )

    artifacts = prepare_parallel_exact_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=torch.randn(1, 3, 4, 2, 2),
        actions=torch.randn(1, 8, 5),
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )

    assert artifacts.input_dict["coupled_action_video_timesteps"] is False
    assert (
        artifacts.input_dict["joint_timestep_coupling"]
        == JointTimestepCoupling.INDEPENDENT.value
    )


def test_shared_video_schedule_inference_uses_video_timestep_directly() -> None:
    video_scheduler = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
    )
    video_scheduler.set_timesteps(20)

    step_index = 1
    shared_sigma = video_scheduler.sigmas[step_index]
    shared_sigma_next = video_scheduler.next_sigma(step_index)
    shared_action_timestep = video_scheduler.timesteps[step_index]
    model_output = torch.ones(1, 1, 1, 1, 1)
    sample = torch.zeros_like(model_output)

    shared_action_step = video_scheduler.step_with_sigmas(
        model_output,
        sigma=shared_sigma,
        sigma_next=shared_sigma_next,
        sample=sample,
    )

    torch.testing.assert_close(
        video_scheduler.sigma_for_timesteps(shared_action_timestep), shared_sigma
    )
    torch.testing.assert_close(
        shared_action_step, model_output * (shared_sigma_next - shared_sigma)
    )


def test_coupled_inference_steps_action_on_shared_video_sigma_schedule() -> None:
    video_scheduler = FlowMatchScheduler(
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=1000,
    )
    action_scheduler = FlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=500,
    )
    video_scheduler.set_timesteps(20)
    action_scheduler.set_timesteps(20)

    step_index = 1
    shared_sigma = video_scheduler.sigmas[step_index]
    shared_sigma_next = video_scheduler.next_sigma(step_index)
    model_output = torch.ones(1, 1, 1, 1, 1)
    sample = torch.zeros_like(model_output)

    action_lookup_scheduler = FlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=500,
    )
    action_lookup_scheduler.set_timesteps(500)

    coupled_action_timestep = action_lookup_scheduler.timestep_matching_sigma(
        shared_sigma
    )
    coupled_action_step = action_scheduler.step_with_sigmas(
        model_output,
        sigma=shared_sigma,
        sigma_next=shared_sigma_next,
        sample=sample,
    )
    independent_action_step = action_scheduler.step(
        model_output,
        action_scheduler.timesteps[step_index],
        sample,
    )

    assert torch.allclose(
        action_lookup_scheduler.sigma_for_timesteps(coupled_action_timestep),
        shared_sigma,
        atol=2e-3,
        rtol=0.0,
    )
    assert not torch.allclose(
        coupled_action_timestep, video_scheduler.timesteps[step_index]
    )
    assert torch.allclose(
        coupled_action_step, model_output * (shared_sigma_next - shared_sigma)
    )
    assert not torch.allclose(coupled_action_step, independent_action_step)


def _generalist_policy_config(
    *,
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA,
    generalist_mode_text_token: bool = False,
) -> ParallelStreamPolicyConfig:
    return ParallelStreamPolicyConfig(
        hidden_size=32,
        frame_chunk_size=2,
        action_per_frame=2,
        video_action_condition_source="noisy_action",
        joint_timestep_coupling=joint_timestep_coupling,
        program=VideoActionProgram.GENERALIST_JOINT_DENOISING,
        generalist_mode_text_token=generalist_mode_text_token,
    )


def _small_generalist_artifacts(
    mode: DynamicsObjective,
    *,
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA,
    drop_text_conditioning: bool | None = None,
):
    torch.manual_seed(7)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    policy_config = _generalist_policy_config(
        joint_timestep_coupling=joint_timestep_coupling
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=20,
        action_num_train_timesteps=20,
    )
    video_latents = torch.randn(1, 3, 4, 2, 2)
    actions = torch.randn(1, 8, 5)
    artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=policy_config,
        training_config=training_config,
        video_latents=video_latents,
        actions=actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
        chunk_size_override=2,
        sample_metadata=_generalist_sample_metadata(
            mode,
            drop_text=drop_text_conditioning,
        ),
    )
    action_latents = actions.reshape(1, 4, 2, 5).permute(0, 3, 1, 2).unsqueeze(-1)
    return artifacts, video_latents, action_latents


def test_parallel_variant_appends_generalist_mode_token_before_deprecated_text_token_proprio() -> (
    None
):
    artifacts, _, _ = _small_generalist_artifacts(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO
    )
    original_text = artifacts.input_dict["latent_dict"]["text_emb"]
    policy_config = replace(
        _generalist_policy_config(generalist_mode_text_token=True),
        proprio_context_mode=ProprioContextMode.TEXT_CONTEXT_TOKEN,  # deprecated compatibility
    )
    variant = ParallelStreamPolicyVariant(
        policy_config,
        LingbotCompatibleVideoBackboneConfig(
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            attention_head_dim=8,
            text_dim=16,
            freq_dim=8,
            patch_size_t=1,
            patch_size_h=1,
            patch_size_w=1,
        ),
        TrainingConfig(chunk_size=2, window_size=8),
        InferenceConfig(frame_chunk_size=2),
        action_dim=5,
        action_horizon=8,
        num_frames=4,
    )
    core = SharedVideoTransformerCore(
        variant.backbone_config,
        action_dim=5,
        state_dim=8,
    )
    core.configure_generalist_mode_context_encoder(enabled=True)
    core.configure_proprio_context_encoder(enabled=True, state_dim=8)

    mode_count = variant.conditioning.append_generalist_mode_text_token(
        core,
        artifacts,
    )
    base_text_token_count = int(
        artifacts.input_dict["latent_dict"]["text_emb"].shape[1]
    )
    appended_text = core.append_proprio_context_tokens(  # deprecated helper
        artifacts.input_dict["latent_dict"]["text_emb"],
        torch.randn(1, 8),
    )

    assert mode_count == 1
    assert base_text_token_count == int(original_text.shape[1]) + 1
    assert appended_text.shape[1] == int(original_text.shape[1]) + 2
    assert (
        artifacts.input_dict["generalist_mode_text_token"] == "action_conditioned_video"
    )
    assert artifacts.input_dict["generalist_mode_text_token_count"] == 1
    assert torch.equal(
        artifacts.input_dict["latent_dict"]["text_emb"],
        artifacts.input_dict["action_dict"]["text_emb"],
    )
    assert torch.equal(
        artifacts.input_dict["latent_dict"]["text_emb"][:, :-1], original_text
    )
    assert not torch.equal(
        artifacts.input_dict["latent_dict"]["text_emb"][:, -1:],
        torch.zeros_like(artifacts.input_dict["latent_dict"]["text_emb"][:, -1:]),
    )


def test_generalist_joint_denoising_action_conditioned_video_uses_clean_action_slot() -> (
    None
):
    artifacts, video_latents, action_latents = _small_generalist_artifacts(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO
    )
    input_dict = artifacts.input_dict

    assert input_dict["joint_denoise_training_mode"] == "action_conditioned_video"
    assert torch.equal(input_dict["action_dict"]["noisy_latents"], action_latents)
    assert torch.all(input_dict["action_dict"]["timesteps"] == 0)
    assert torch.all(input_dict["action_dict"]["targets"] == 0)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 0)
    assert torch.all(input_dict["latent_dict"]["loss_mask"] == 1)
    assert torch.equal(input_dict["latent_dict"]["latent"], video_latents)
    assert torch.equal(input_dict["action_dict"]["latent"], action_latents)
    assert input_dict["chunk_size"] == 2
    assert input_dict["window_size"] == 3
    assert (
        input_dict["history_stream_visibility"]
        == HistoryStreamVisibility.VIDEO_ONLY.value
    )
    assert input_dict["conditional_history_policy"] == "previous_boundary_video_only"
    assert input_dict["generalist_conditional_history_chunks"] == 1
    shared_sigmas = input_dict["joint_denoise_shared_sigmas"]
    latent_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        input_dict["latent_dict"]["timesteps"][0]
    )
    assert torch.allclose(latent_sigmas, shared_sigmas, atol=2e-3, rtol=0.0)


def test_generalist_joint_denoising_video_conditioned_action_uses_clean_video_slot() -> (
    None
):
    artifacts, video_latents, _ = _small_generalist_artifacts(
        DynamicsObjective.VIDEO_CONDITIONED_ACTION
    )
    input_dict = artifacts.input_dict

    assert input_dict["joint_denoise_training_mode"] == "video_conditioned_action"
    assert torch.equal(input_dict["latent_dict"]["noisy_latents"], video_latents)
    assert torch.all(input_dict["latent_dict"]["timesteps"] == 0)
    assert torch.all(input_dict["latent_dict"]["targets"] == 0)
    assert torch.all(input_dict["latent_dict"]["loss_mask"] == 0)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 1)
    assert torch.equal(input_dict["latent_dict"]["latent"], video_latents)
    assert torch.count_nonzero(input_dict["action_dict"]["latent"]) > 0
    assert input_dict["chunk_size"] == 2
    assert input_dict["window_size"] == 3
    assert (
        input_dict["history_stream_visibility"]
        == HistoryStreamVisibility.VIDEO_ONLY.value
    )
    assert input_dict["conditional_history_policy"] == "previous_boundary_video_only"
    assert input_dict["generalist_conditional_history_chunks"] == 1
    shared_sigmas = input_dict["joint_denoise_shared_sigmas"]
    expected_action_timesteps = artifacts.action_scheduler.timestep_matching_sigma(
        shared_sigmas
    )
    assert torch.equal(
        input_dict["action_dict"]["timesteps"][0], expected_action_timesteps
    )


def test_generalist_joint_denoising_joint_mode_matches_standard_m1_joint_artifacts() -> (
    None
):
    artifacts, video_latents, action_latents = _small_generalist_artifacts(
        DynamicsObjective.JOINT
    )
    input_dict = artifacts.input_dict

    torch.manual_seed(7)
    backbone_config = LingbotCompatibleVideoBackboneConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        patch_size_t=1,
        patch_size_h=1,
        patch_size_w=1,
    )
    standard_policy = replace(
        _generalist_policy_config(),
        program=VideoActionProgram.JOINT,
    )
    training_config = TrainingConfig(
        chunk_size=2,
        window_size=8,
        video_num_train_timesteps=20,
        action_num_train_timesteps=20,
    )
    standard_video_latents = torch.randn(1, 3, 4, 2, 2)
    standard_actions = torch.randn(1, 8, 5)
    standard_artifacts = prepare_parallel_action_conditioned_train_artifacts(
        backbone_config=backbone_config,
        policy_config=standard_policy,
        training_config=training_config,
        video_latents=standard_video_latents,
        actions=standard_actions,
        action_mask=None,
        text_emb=torch.randn(1, 512, 16),
    )
    standard_input = standard_artifacts.input_dict

    assert input_dict["joint_denoise_training_mode"] == "joint"
    assert torch.all(input_dict["latent_dict"]["loss_mask"] == 1)
    assert torch.all(input_dict["action_dict"]["loss_mask"] == 1)
    assert not torch.equal(input_dict["latent_dict"]["noisy_latents"], video_latents)
    assert not torch.equal(input_dict["action_dict"]["noisy_latents"], action_latents)
    assert torch.equal(video_latents, standard_video_latents)
    assert torch.equal(
        input_dict["latent_dict"]["noisy_latents"],
        standard_input["latent_dict"]["noisy_latents"],
    )
    assert torch.equal(
        input_dict["latent_dict"]["latent"], standard_input["latent_dict"]["latent"]
    )
    assert torch.equal(
        input_dict["latent_dict"]["targets"], standard_input["latent_dict"]["targets"]
    )
    assert torch.equal(
        input_dict["latent_dict"]["timesteps"],
        standard_input["latent_dict"]["timesteps"],
    )
    assert torch.equal(
        input_dict["action_dict"]["noisy_latents"],
        standard_input["action_dict"]["noisy_latents"],
    )
    assert torch.equal(
        input_dict["action_dict"]["latent"], standard_input["action_dict"]["latent"]
    )
    assert torch.equal(
        input_dict["action_dict"]["targets"], standard_input["action_dict"]["targets"]
    )
    assert torch.equal(
        input_dict["action_dict"]["timesteps"],
        standard_input["action_dict"]["timesteps"],
    )
    assert "generalist_conditional_history_chunks" not in input_dict

    shared_sigmas = input_dict["joint_denoise_shared_sigmas"]
    assert shared_sigmas.shape == (4,)
    assert torch.all(shared_sigmas >= 0)
    assert torch.all(shared_sigmas <= 1)
    latent_sigmas = artifacts.latent_scheduler.sigma_for_timesteps(
        input_dict["latent_dict"]["timesteps"][0]
    )
    expected_action_timesteps = artifacts.action_scheduler.timestep_matching_sigma(
        shared_sigmas
    )
    assert torch.allclose(latent_sigmas, shared_sigmas, atol=2e-3, rtol=0.0)
    assert torch.equal(
        input_dict["action_dict"]["timesteps"][0], expected_action_timesteps
    )


def test_generalist_joint_denoising_conditional_modes_drop_text_by_default() -> None:
    joint_artifacts, _, _ = _small_generalist_artifacts(DynamicsObjective.JOINT)
    assert joint_artifacts.input_dict["joint_denoise_text_dropped"] is False
    assert (
        torch.count_nonzero(joint_artifacts.input_dict["latent_dict"]["text_emb"]) > 0
    )

    for mode in (
        DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        DynamicsObjective.VIDEO_CONDITIONED_ACTION,
    ):
        artifacts, _, _ = _small_generalist_artifacts(mode)
        input_dict = artifacts.input_dict

        assert input_dict["joint_denoise_text_dropped"] is True
        assert torch.equal(
            input_dict["latent_dict"]["text_emb"],
            torch.zeros_like(input_dict["latent_dict"]["text_emb"]),
        )
        assert torch.equal(
            input_dict["action_dict"]["text_emb"],
            torch.zeros_like(input_dict["action_dict"]["text_emb"]),
        )


def test_generalist_joint_denoising_rejects_conditional_text_override() -> None:
    with pytest.raises(ValueError, match="always removes task text"):
        _small_generalist_artifacts(
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
            drop_text_conditioning=False,
        )


def test_parallel_stream_decoder_logs_generalist_mode_sums_and_counts() -> None:
    artifacts, _, _ = _small_generalist_artifacts(
        DynamicsObjective.ACTION_CONDITIONED_VIDEO
    )
    action_targets = (
        artifacts.input_dict["action_dict"]["targets"]
        .squeeze(-1)
        .permute(0, 2, 3, 1)
        .reshape(1, 8, 5)
    )
    latent_targets = (
        artifacts.input_dict["latent_dict"]["targets"]
        .permute(0, 2, 3, 4, 1)
        .reshape(1, 16, 3)
    )
    decoder = ParallelStreamActionDecoder(
        hidden_size=32, action_dim=5, action_horizon=8
    )

    output = decoder.forward_train(
        PolicyTrainOutput(
            policy_features=action_targets,
            metrics={},
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT,
                payload=ParallelDecoderTrainArtifacts(
                    latent_pred=latent_targets,
                    runtime=artifacts,
                    loss_weights={"latent": 1.0, "action": 1.0},
                    patch_size=(1, 1, 1),
                ),
                dynamics_objective=(DynamicsObjective.ACTION_CONDITIONED_VIDEO),
            ),
        ),
        PolicyTrainBatch(actions=torch.zeros(1, 8, 5)),
    )

    assert output.metrics["joint_denoise/action_conditioned_video/count"].item() == 1.0
    assert output.metrics["joint_denoise/joint/count"].item() == 0.0
    assert (
        "joint_denoise/action_conditioned_video/action_flow_loss_sum" in output.metrics
    )
    assert "joint_denoise/action_conditioned_video/action_mse_sum" in output.metrics
    assert torch.equal(
        output.metrics["joint_denoise/action_conditioned_video/action_mse_sum"],
        output.metrics["joint_denoise/action_conditioned_video/action_flow_loss_sum"],
    )
    assert output.metrics["joint_denoise/action_loss_active"].item() == 0.0
    assert output.metrics["joint_denoise/latent_loss_active"].item() == 1.0
