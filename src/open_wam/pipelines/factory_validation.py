"""Runtime validation and shared dimension resolution for pipeline construction."""

from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    BackboneImplementation,
    BatchAdapterName,
    ExperimentConfig,
    MoTRuntimeMode,
    ParallelRuntimeMode,
    ProprioContextMode,
    VideoConditionInputSpace,
    VideoConditionSource,
    VideoConditionTrainMode,
)
from open_wam.configs.policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
)
from open_wam.configs.policy_mot import MoTPolicyConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.data.action_mapping import validate_action_mapping_preflight
from open_wam.models.policy_variants.parallel_stream.action_adapter import (
    build_action_adapter_spec,
)
from open_wam.models.video_backbone import normalize_backbone_implementation

_PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES = {
    ParallelRuntimeMode.LINGBOT_EXACT,
    ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
    ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
    ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
}


def _resolve_parallel_stream_model_action_dim(config: ExperimentConfig) -> int:
    if (
        isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        and config.policy_variant.runtime_mode in _PARALLEL_STREAM_EXACT_MODEL_ACTION_MODES
    ):
        return config.action_decoder.action_dim
    return config.data.action_schema.action_dim


def _resolve_proprio_context_state_dim(config: ExperimentConfig) -> int | None:
    if not isinstance(config.policy_variant, (MoTPolicyConfig, ParallelStreamPolicyConfig)):
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
                PostLatentPolicyConfig,
                PostDecodedPolicyConfig,
                CausalVideoPredictionPolicyConfig,
                MoTPolicyConfig,
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
                "Parallel-stream method 1 now only supports LingBot-exact semantics, "
                f"got policy_variant.runtime_mode={config.policy_variant.runtime_mode!r}."
            )
        expected_horizon = config.data.num_frames * config.policy_variant.action_per_frame
        if action_schema.action_horizon != expected_horizon:
            raise ValueError(
                "Parallel-stream config requires `action_horizon == num_frames * action_per_frame`, "
                f"got action_horizon={action_schema.action_horizon}, num_frames={config.data.num_frames}, "
                f"action_per_frame={config.policy_variant.action_per_frame}."
            )
        if config.action_decoder.name != ActionDecoderName.LINGBOT_PARALLEL:
            raise ValueError(
                "Parallel-stream method 1 requires `action_decoder.name = lingbot_parallel_decoder`."
            )
        if config.action_decoder.action_horizon != action_schema.action_horizon:
            raise ValueError(
                "Parallel-stream method 1 requires `action_decoder.action_horizon` to match "
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
    if isinstance(config.policy_variant, MoTPolicyConfig):
        if action_schema.action_horizon <= 0:
            raise ValueError("MoT method 5 requires `data.action_schema.action_horizon > 0`.")
        if (
            config.policy_variant.runtime_mode
            in {MoTRuntimeMode.JOINT_DENOISE, MoTRuntimeMode.NON_JOINT_TWO_STREAM}
            and config.policy_variant.video_prefix_frames >= config.data.num_frames
        ):
            raise ValueError(
                "MoT two-stream method 5 requires `video_prefix_frames < data.num_frames`, "
                f"got video_prefix_frames={config.policy_variant.video_prefix_frames}, "
                f"data.num_frames={config.data.num_frames}, "
                f"runtime_mode={config.policy_variant.runtime_mode!r}."
            )
        if config.policy_variant.num_action_layers != config.backbone.num_layers:
            raise ValueError(
                "MoT method 5 currently requires `policy_variant.num_action_layers == backbone.num_layers` "
                "so the action expert stays layer-aligned with the video expert, "
                f"got num_action_layers={config.policy_variant.num_action_layers}, "
                f"backbone.num_layers={config.backbone.num_layers}."
            )
    if (
        isinstance(config.policy_variant, (PostLatentPolicyConfig, PostDecodedPolicyConfig))
        and config.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    ):
        direct_train_mode = config.action_decoder.train_mode == VideoConditionTrainMode.CURRENT_FRAME_REGRESSION
        if config.policy_variant.local_video_window_frames > config.data.num_frames:
            raise ValueError(
                "Method-4 video-conditioned decoding requires `policy_variant.local_video_window_frames <= data.num_frames`, "
                f"got local_video_window_frames={config.policy_variant.local_video_window_frames}, "
                f"data.num_frames={config.data.num_frames}."
            )
        if config.action_decoder.action_horizon <= 0:
            raise ValueError("Method-4 video-conditioned decoding requires `action_horizon > 0`.")
        if not direct_train_mode and int(config.policy_variant.current_video_frame_index) != 0:
            raise ValueError(
                "Method-4 rollout-window decoding currently supports only `current_video_frame_index = 0`. "
                "Non-zero sliding-window alignment is not implemented yet."
            )
        if (
            direct_train_mode
            and config.policy_variant.train_video_condition_source == VideoConditionSource.GENERATED_FUTURE
        ):
            raise ValueError(
                "Method-4 generated-future video conditioning is only supported for rollout-window diffusion "
                "training. `current_frame_regression` bypasses the policy-variant window builder."
            )
        if direct_train_mode:
            if config.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT:
                if config.trainer.batch_adapter != BatchAdapterName.LATENTS:
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `video_latent` input requires the latent batch "
                        "adapter so training sees dataset video latents directly, "
                        f"got trainer.batch_adapter={config.trainer.batch_adapter!r}."
                    )
                if config.data.dataset_type != "lerobot_v2_latent_local":
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `video_latent` input is currently maintained "
                        "only for latent-local datasets, "
                        f"got data.dataset_type={config.data.dataset_type!r}."
                    )
            if config.policy_variant.video_condition_input_space == VideoConditionInputSpace.RGB_VIDEO:
                if config.trainer.batch_adapter != BatchAdapterName.VIEWS:
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `rgb_video` input requires the view batch "
                        "adapter so training sees raw RGB frames directly, "
                        f"got trainer.batch_adapter={config.trainer.batch_adapter!r}."
                    )
                if config.data.dataset_type == "lerobot_v2_latent_local":
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `rgb_video` input requires raw RGB dataset "
                        "windows. Latent-local datasets enter through precomputed latents, so use "
                        "`video_latent` there."
                    )
                if config.action_decoder.use_text_conditioning:
                    raise ValueError(
                        "Method-4 `current_frame_regression` with `rgb_video` input and the view batch adapter "
                        "does not currently provide text embeddings. Set `action_decoder.use_text_conditioning=false` "
                        "for this mode."
                    )
        elif (
            config.policy_variant.video_condition_input_space == VideoConditionInputSpace.RGB_VIDEO
            and config.data.dataset_type == "lerobot_v2_latent_local"
        ):
            raise ValueError(
                "Method-4 `rgb_video` conditioning requires raw RGB to enter through the shared frontend/VAE path. "
                "Latent-local datasets enter from precomputed latents, so use `video_latent` conditioning there."
            )
        if not direct_train_mode and config.data.dataset_type in {"lerobot_v2", "libero_hdf5"}:
            raise ValueError(
                "Method-4 video-conditioned current-action decoding is currently aligned only for latent-local "
                "`standard_policy_window` style data. Raw LIBERO / raw LeRobot adapters anchor actions at the "
                "last observed frame, so apples-to-apples current-action method-4 runs need a deliberate raw-data "
                "alignment pass first. Use the latent-local method-4 configs or keep the explicit legacy "
                "method-4 decoders on raw LIBERO for now."
            )


__all__ = ["validate_experiment_config"]
