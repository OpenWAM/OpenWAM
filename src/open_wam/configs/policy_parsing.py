"""Raw policy-variant parsing into typed configuration owners."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import enums as config_enums
from .backbone import SharedVideoTransformerConfig
from .coercion import (
    coerce_bool as _coerce_bool,
)
from .coercion import (
    coerce_enum as _coerce_enum,
)
from .coercion import (
    coerce_enum_tuple as _coerce_enum_tuple,
)
from .coercion import (
    coerce_optional_enum as _coerce_optional_enum,
)
from .data_contracts import DataConfig
from .inference import InferenceConfig
from .policy_compatibility import normalize_video_action_policy_fields
from .policy_contracts import (
    CausalVideoPredictionPolicyConfig,
    ExtensionPolicyConfig,
    PolicyVariantConfig,
)
from .policy_dual_expert import DualExpertPolicyConfig
from .policy_parallel_stream import ParallelStreamPolicyConfig
from .policy_video_action import resolve_video_action_program_semantics
from .training import TrainingConfig


def parse_policy_variant_config(
    policy_variant_raw: Mapping[str, Any],
    data_config: DataConfig,
    backbone_config: SharedVideoTransformerConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> PolicyVariantConfig:
    resolved_raw = normalize_video_action_policy_fields(policy_variant_raw)
    if not resolved_raw:
        raise ValueError(
            "Experiment config requires an explicit `policy_variant` mapping. "
            "Choose `parallel_stream`, `dual_expert`, `causal_video_prediction`, "
            "or a registered `extension`."
        )

    name = _coerce_enum(
        config_enums.PolicyVariantName,
        resolved_raw.get("name"),
    )
    hidden_size = resolved_raw.get("hidden_size", backbone_config.hidden_size)
    if name == config_enums.PolicyVariantName.EXTENSION:
        return ExtensionPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            extension_type=resolved_raw.get("extension_type", ""),
            options=resolved_raw.get("options", {}),
        )
    if name == config_enums.PolicyVariantName.CAUSAL_VIDEO_PREDICTION:
        return CausalVideoPredictionPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
        )
    if name == config_enums.PolicyVariantName.DUAL_EXPERT:
        removed_fields = {
            "runtime_mode",
            "current_block_coupling",
            "video_can_attend_action",
        }.intersection(resolved_raw)
        if removed_fields:
            fields = ", ".join(
                f"policy_variant.{field}" for field in sorted(removed_fields)
            )
            raise ValueError(
                f"{fields} cannot be authored for Dual Expert; select "
                "`policy_variant.program` instead. Legacy checkpoint metadata is "
                "migrated only by the checkpoint loader."
            )
        if resolved_raw.get("program") is None:
            raise ValueError(
                "DualExpert policy requires an explicit `policy_variant.program`."
            )
        program = _coerce_enum(
            config_enums.VideoActionProgram,
            resolved_raw["program"],
        )
        preset = _coerce_optional_enum(
            config_enums.DualExpertPreset,
            resolved_raw.get("preset"),
        )
        generalist_denoising_mode_probs = resolved_raw.get(
            "generalist_denoising_mode_probs"
        )
        dual_expert_joint_timestep_coupling_default = (
            config_enums.JointTimestepCoupling.INDEPENDENT
            if generalist_denoising_mode_probs is not None
            else config_enums.JointTimestepCoupling.MATCH_SIGMA
        )
        dual_expert_defaults: dict[str, Any] = {}
        if preset == config_enums.DualExpertPreset.FASTWAM:
            dual_expert_defaults = {
                "condition_mode": config_enums.DualExpertConditionMode.FIRST_FRAME,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.DualExpertPreset.FASTWAM_JOINT:
            dual_expert_defaults = {
                "condition_mode": config_enums.DualExpertConditionMode.FULL_VIDEO,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.DualExpertPreset.FASTWAM_IDM:
            dual_expert_defaults = {
                "condition_mode": config_enums.DualExpertConditionMode.TEACHER_FORCING_COND_VIDEO,
                "teacher_forcing_video_noise_prob": 0.5,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.DualExpertPreset.FASTWAM_NON_JOINT:
            dual_expert_defaults = {
                "condition_mode": config_enums.DualExpertConditionMode.TEACHER_FORCING_COND_VIDEO,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        context_condition_latent_source = _coerce_enum(
            config_enums.ContextConditionLatentSource,
            resolved_raw.get(
                "context_condition_latent_source",
                config_enums.ContextConditionLatentSource.VIDEO_LATENTS,
            ),
        )
        return DualExpertPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            preset=preset,
            condition_mode=_coerce_enum(
                config_enums.DualExpertConditionMode,
                resolved_raw.get(
                    "condition_mode",
                    dual_expert_defaults.get(
                        "condition_mode",
                        config_enums.DualExpertConditionMode.FIRST_FRAME,
                    ),
                ),
            ),
            action_expert_init_mode=_coerce_enum(
                config_enums.DualExpertActionExpertInitMode,
                resolved_raw.get(
                    "action_expert_init_mode",
                    config_enums.DualExpertActionExpertInitMode.VIDEO_WEIGHT_COPY,
                ),
            ),
            video_prefix_frames=resolved_raw.get(
                "video_prefix_frames",
                dual_expert_defaults.get("video_prefix_frames", 1),
            ),
            teacher_forcing_video_noise_prob=resolved_raw.get(
                "teacher_forcing_video_noise_prob",
                dual_expert_defaults.get("teacher_forcing_video_noise_prob", 0.5),
            ),
            noisy_video_condition_prob=resolved_raw.get("noisy_video_condition_prob", 0.5),
            num_action_layers=resolved_raw.get("num_action_layers", backbone_config.num_layers),
            action_hidden_size=resolved_raw.get("action_hidden_size"),
            action_ffn_dim=resolved_raw.get("action_ffn_dim"),
            program=program,
            use_text_conditioning=resolved_raw.get("use_text_conditioning", True),
            use_state_conditioning=resolved_raw.get("use_state_conditioning", False),
            proprio_context_mode=_coerce_enum(
                config_enums.ProprioContextMode,
                resolved_raw.get("proprio_context_mode", config_enums.ProprioContextMode.NONE),
            ),
            history_stream_visibility=_coerce_enum(
                config_enums.HistoryStreamVisibility,
                resolved_raw.get(
                    "history_stream_visibility",
                    config_enums.HistoryStreamVisibility.FULL,
                ),
            ),
            context_condition_latent_source=context_condition_latent_source,
            use_activation_checkpointing=resolved_raw.get("use_activation_checkpointing", False),
            use_condition_latents=(
                True
                if context_condition_latent_source
                == config_enums.ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
                else bool(resolved_raw.get("use_condition_latents", True))
            ),
            require_condition_latents=(
                True
                if context_condition_latent_source
                == config_enums.ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
                else bool(resolved_raw.get("require_condition_latents", False))
            ),
            sequence_contract=_coerce_enum(
                config_enums.VideoActionSequenceContract,
                resolved_raw.get(
                    "sequence_contract",
                    config_enums.VideoActionSequenceContract.DEFAULT,
                ),
            ),
            generalist_denoising_mode_probs=generalist_denoising_mode_probs,
            generalist_mode_text_token=_coerce_bool(
                resolved_raw.get("generalist_mode_text_token", False),
                field_name="policy_variant.generalist_mode_text_token",
            ),
            joint_timestep_coupling=_coerce_enum(
                config_enums.JointTimestepCoupling,
                resolved_raw.get(
                    "joint_timestep_coupling",
                    dual_expert_joint_timestep_coupling_default,
                ),
            ),
            generalist_training_paradigm=_coerce_enum(
                config_enums.GeneralistTrainingParadigm,
                resolved_raw.get(
                    "generalist_training_paradigm",
                    config_enums.GeneralistTrainingParadigm.DEMO_ONLY,
                ),
            ),
        )
    if name == config_enums.PolicyVariantName.PARALLEL_STREAM:
        default_action_per_frame = max(
            1,
            data_config.action_schema.action_horizon // max(1, data_config.num_frames),
        )
        runtime_mode = _coerce_enum(
            config_enums.ParallelRuntimeMode,
            resolved_raw.get("runtime_mode", config_enums.ParallelRuntimeMode.LINGBOT_EXACT),
        )
        program, current_block_coupling = resolve_video_action_program_semantics(
            program=resolved_raw.get("program"),
            current_block_coupling=resolved_raw.get("current_block_coupling"),
        )
        preserve_video_pretrain_history = bool(
            resolved_raw.get("preserve_video_pretrain_history", False)
        )
        history_stream_visibility = _coerce_enum(
            config_enums.HistoryStreamVisibility,
            resolved_raw.get(
                "history_stream_visibility",
                (
                    config_enums.HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
                    if preserve_video_pretrain_history
                    else config_enums.HistoryStreamVisibility.FULL
                ),
            ),
        )
        context_condition_latent_source = _coerce_enum(
            config_enums.ContextConditionLatentSource,
            resolved_raw.get(
                "context_condition_latent_source",
                config_enums.ContextConditionLatentSource.VIDEO_LATENTS,
            ),
        )
        proprio_context_mode = _coerce_enum(
            config_enums.ProprioContextMode,
            resolved_raw.get("proprio_context_mode", config_enums.ProprioContextMode.NONE),
        )
        if proprio_context_mode == config_enums.ProprioContextMode.PER_CHUNK_ADDITIVE:
            exact_runtime_mode = runtime_mode in {
                config_enums.ParallelRuntimeMode.LINGBOT_EXACT,
                config_enums.ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            }
            if exact_runtime_mode and current_block_coupling is None:
                raise ValueError(
                    "proprio_context_mode=per_chunk_additive requires "
                    "a video/action program with explicit chunk semantics."
                )
            if not exact_runtime_mode:
                raise ValueError(
                    "proprio_context_mode=per_chunk_additive is only supported for "
                    "exact parallel-stream runtime modes."
                )
        use_condition_latents = (
            True
            if context_condition_latent_source
            == config_enums.ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
            else bool(resolved_raw.get("use_condition_latents", True))
        )
        require_condition_latents = (
            True
            if context_condition_latent_source
            == config_enums.ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
            else bool(resolved_raw.get("require_condition_latents", False))
        )
        sequence_order = tuple(
            resolved_raw.get(
                "sequence_order",
                (
                    config_enums.ParallelSequenceComponent.VIDEO_NOISY,
                    config_enums.ParallelSequenceComponent.VIDEO_CONDITION,
                    config_enums.ParallelSequenceComponent.ACTION_NOISY,
                    config_enums.ParallelSequenceComponent.ACTION_CONDITION,
                ),
            )
        )
        variant_profile = _coerce_enum(
            config_enums.ParallelStreamVariantProfile,
            resolved_raw.get("variant_profile", config_enums.ParallelStreamVariantProfile.STANDARD),
        )
        parallel_joint_timestep_coupling_default = (
            config_enums.JointTimestepCoupling.INDEPENDENT
            if variant_profile == config_enums.ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
            else config_enums.JointTimestepCoupling.MATCH_SIGMA
        )
        return ParallelStreamPolicyConfig(
            hidden_size=hidden_size,
            runtime_mode=runtime_mode,
            variant_profile=variant_profile,
            reference_profile=resolved_raw.get("reference_profile"),
            frame_chunk_size=resolved_raw.get("frame_chunk_size", inference_config.frame_chunk_size),
            action_per_frame=resolved_raw.get("action_per_frame", default_action_per_frame),
            attn_window=resolved_raw.get("attn_window", training_config.window_size),
            sequence_order=_coerce_enum_tuple(config_enums.ParallelSequenceComponent, sequence_order),
            mask_mode=_coerce_enum(
                config_enums.ParallelMaskMode,
                resolved_raw.get("mask_mode", config_enums.ParallelMaskMode.LINGBOT_CHUNKED),
            ),
            cache_mode=_coerce_enum(
                config_enums.ParallelCacheMode,
                resolved_raw.get("cache_mode", config_enums.ParallelCacheMode.METADATA_ONLY),
            ),
            noisy_video_condition_prob=resolved_raw.get("noisy_video_condition_prob", 0.5),
            video_condition_on_action=resolved_raw.get("video_condition_on_action", False),
            video_action_condition_source=_coerce_enum(
                config_enums.ParallelActionConditionSource,
                resolved_raw.get(
                    "video_action_condition_source",
                    config_enums.ParallelActionConditionSource.NOISY_ACTION,
                ),
            ),
            video_action_attention_scope=_coerce_enum(
                config_enums.ParallelActionAttentionScope,
                resolved_raw.get(
                    "video_action_attention_scope",
                    config_enums.ParallelActionAttentionScope.BLOCK_LOCAL,
                ),
            ),
            joint_timestep_coupling=_coerce_enum(
                config_enums.JointTimestepCoupling,
                resolved_raw.get(
                    "joint_timestep_coupling",
                    parallel_joint_timestep_coupling_default,
                ),
            ),
            generalist_denoising_mode_probs=resolved_raw.get(
                "generalist_denoising_mode_probs"
            ),
            generalist_training_paradigm=_coerce_enum(
                config_enums.GeneralistTrainingParadigm,
                resolved_raw.get(
                    "generalist_training_paradigm",
                    config_enums.GeneralistTrainingParadigm.DEMO_ONLY,
                ),
            ),
            generalist_mode_text_token=_coerce_bool(
                resolved_raw.get("generalist_mode_text_token", False),
                field_name="policy_variant.generalist_mode_text_token",
            ),
            program=program,
            current_block_coupling=current_block_coupling,
            preserve_video_pretrain_history=preserve_video_pretrain_history,
            history_stream_visibility=history_stream_visibility,
            context_condition_latent_source=context_condition_latent_source,
            use_condition_latents=use_condition_latents,
            proprio_context_mode=proprio_context_mode,
            require_condition_latents=require_condition_latents,
            sequence_contract=_coerce_enum(
                config_enums.VideoActionSequenceContract,
                resolved_raw.get(
                    "sequence_contract",
                    config_enums.VideoActionSequenceContract.DEFAULT,
                ),
            ),
            temporal_position_mode=_coerce_enum(
                config_enums.TemporalPositionMode,
                resolved_raw.get(
                    "temporal_position_mode",
                    config_enums.TemporalPositionMode.GLOBAL_SHIFTED,
                ),
            ),
            used_action_channel_ids=tuple(resolved_raw.get("used_action_channel_ids", ())),
            inverse_used_action_channel_ids=tuple(resolved_raw.get("inverse_used_action_channel_ids", ())),
            action_norm_method=_coerce_enum(
                config_enums.ActionNormMethod,
                resolved_raw.get("action_norm_method", config_enums.ActionNormMethod.PROFILE),
            ),
            norm_q01=tuple(resolved_raw.get("norm_q01", ())),
            norm_q99=tuple(resolved_raw.get("norm_q99", ())),
        )
    raise ValueError(f"Unsupported policy variant '{name}'.")


__all__ = [
    "parse_policy_variant_config",
]
