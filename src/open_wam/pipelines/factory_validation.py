"""Runtime validation and shared dimension resolution for pipeline construction."""

from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    BackboneImplementation,
    DualExpertRuntimeMode,
    ExperimentConfig,
    ParallelRuntimeMode,
    ProprioContextMode,
)
from open_wam.configs.policy_contracts import (
    CausalVideoPredictionPolicyConfig,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.data.action_mapping import validate_action_mapping_preflight
from open_wam.models.policy_variants.parallel_stream.action_adapter import (
    build_action_adapter_spec,
)
from open_wam.models.video_backbone import normalize_backbone_implementation

_PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES = {
    ParallelRuntimeMode.LINGBOT_EXACT,
    ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
}


def _resolve_parallel_stream_model_action_dim(config: ExperimentConfig) -> int:
    if (
        isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        and config.policy_variant.runtime_mode in _PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES
    ):
        return config.action_decoder.action_dim
    return config.data.action_schema.action_dim


def _resolve_proprio_context_state_dim(config: ExperimentConfig) -> int | None:
    if not isinstance(config.policy_variant, (DualExpertPolicyConfig, ParallelStreamPolicyConfig)):
        return None
    mode = ProprioContextMode(config.policy_variant.proprio_context_mode)
    if mode != ProprioContextMode.TEXT_CONTEXT_TOKEN:
        return None
    state_dim = int(config.data.action_schema.state_dim)
    if state_dim <= 0:
        raise ValueError(
            "Deprecated proprio_context_mode=text_context_token requires positive data.action_schema.state_dim."
        )
    return state_dim


def _resolve_proprio_hidden_context_state_dim(config: ExperimentConfig) -> int | None:
    if not isinstance(config.policy_variant, ParallelStreamPolicyConfig):
        return None
    mode = ProprioContextMode(config.policy_variant.proprio_context_mode)
    if mode != ProprioContextMode.PER_CHUNK_ADDITIVE:
        return None
    state_dim = int(config.data.action_schema.state_dim)
    if state_dim <= 0:
        raise ValueError("proprio_context_mode=per_chunk_additive requires positive data.action_schema.state_dim.")
    return state_dim


def validate_experiment_config(config: ExperimentConfig) -> None:
    action_schema = config.data.action_schema
    validate_action_mapping_preflight(
        config.data.action_mapping,
        action_schema_dim=action_schema.action_dim,
    )
    if (
        isinstance(
            config.policy_variant,
            (
                ParallelStreamPolicyConfig,
                CausalVideoPredictionPolicyConfig,
                DualExpertPolicyConfig,
            ),
        )
        and normalize_backbone_implementation(config.backbone.implementation)
        != BackboneImplementation.SHARED_TRANSFORMER
    ):
        raise ValueError(
            "Policy variants in the current repo all require the shared transformer backbone so they run "
            f"through the same LingBot-compatible visual core, got "
            f"backbone.implementation={config.backbone.implementation!r}."
        )
    if isinstance(config.policy_variant, ParallelStreamPolicyConfig):
        if config.policy_variant.runtime_mode not in _PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES:
            raise ValueError(
                "Parallel-stream policies currently require a supported exact runtime, "
                f"got policy_variant.runtime_mode={config.policy_variant.runtime_mode!r}."
            )
        expected_horizon = config.data.num_frames * config.policy_variant.action_per_frame
        if action_schema.action_horizon != expected_horizon:
            raise ValueError(
                "Parallel-stream config requires `action_horizon == num_frames * action_per_frame`, "
                f"got action_horizon={action_schema.action_horizon}, num_frames={config.data.num_frames}, "
                f"action_per_frame={config.policy_variant.action_per_frame}."
            )
        if config.action_decoder.name != ActionDecoderName.PARALLEL_STREAM:
            raise ValueError(
                "Parallel-stream policies require `action_decoder.name = parallel_stream_decoder`."
            )
        if config.action_decoder.action_horizon != action_schema.action_horizon:
            raise ValueError(
                "Parallel-stream policies require `action_decoder.action_horizon` to match "
                "`data.action_schema.action_horizon`, "
                f"got decoder={config.action_decoder.action_horizon}, data={action_schema.action_horizon}."
            )
        model_action_dim = _resolve_parallel_stream_model_action_dim(config)
        adapter_spec = build_action_adapter_spec(config.policy_variant, model_action_dim=model_action_dim)
        if adapter_spec is None and action_schema.action_dim != model_action_dim:
            raise ValueError(
                "Exact LingBot runtime needs an action adapter when dataset and model action dims differ, "
                f"got data action_dim={action_schema.action_dim} and model action_dim={model_action_dim}."
            )
        if (
            adapter_spec is not None
            and action_schema.action_dim != model_action_dim
            and adapter_spec.raw_action_dim != action_schema.action_dim
        ):
            raise ValueError(
                "Exact LingBot action adapter raw action dim must match `data.action_schema.action_dim` "
                "when dataset and model action dims differ, "
                f"got adapter raw_action_dim={adapter_spec.raw_action_dim}, "
                f"data action_dim={action_schema.action_dim}, model action_dim={model_action_dim}."
            )
    if isinstance(config.policy_variant, DualExpertPolicyConfig):
        if action_schema.action_horizon <= 0:
            raise ValueError("Dual-expert policies require `data.action_schema.action_horizon > 0`.")
        if (
            config.policy_variant.runtime_mode
            in {DualExpertRuntimeMode.JOINT_DENOISE, DualExpertRuntimeMode.NON_JOINT_TWO_STREAM}
            and config.policy_variant.video_prefix_frames >= config.data.num_frames
        ):
            raise ValueError(
                "Dual-expert two-stream policies require `video_prefix_frames < data.num_frames`, "
                f"got video_prefix_frames={config.policy_variant.video_prefix_frames}, "
                f"data.num_frames={config.data.num_frames}, "
                f"runtime_mode={config.policy_variant.runtime_mode!r}."
            )
        if config.policy_variant.num_action_layers != config.backbone.num_layers:
            raise ValueError(
                "Dual-expert policies currently require `policy_variant.num_action_layers == backbone.num_layers` "
                "so the action expert stays layer-aligned with the video expert, "
                f"got num_action_layers={config.policy_variant.num_action_layers}, "
                f"backbone.num_layers={config.backbone.num_layers}."
            )
__all__ = ["validate_experiment_config"]
