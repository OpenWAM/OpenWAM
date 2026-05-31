from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Collection, Mapping, TypeVar

from open_wam.configs.enums import StrEnum
from open_wam.configs import (
    ActionDecoderConfig,
    DecodedFeatureActionDecoderConfig,
    LingbotParallelActionDecoderConfig,
    MLPActionDecoderConfig,
    MoTActionDecoderConfig,
    VideoConditionedActionDecoderConfig,
    CausalPrefixSuffixBucketConfig,
    CausalVideoPredictionPolicyConfig,
    MoTPolicyConfig,
    ParallelStreamPolicyConfig,
    PolicyVariantConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    RegisterActionDecoderConfig,
    RegisterAttachedPolicyConfig,
    VPPActionDecoderConfig,
    VideoOnlyActionDecoderConfig,
    VideoSequencePolicyConfig,
    ActionMappingConfig,
    ActionNormalizationConfig,
    ConsortiumChannelMappingConfig,
    ConsortiumCloudCacheConfig,
    ConsortiumEpisodeSelectionConfig,
    ConsortiumLocalCacheConfig,
    ConsortiumMemberConfig,
    ActionSchemaConfig,
    ActionTargetConfig,
    CalvinDataConfig,
    DataConfig,
    ExperimentConfig,
    GenericDataConfig,
    GeneralistDynamicsMixtureConfig,
    LeRobotConsortiumDataConfig,
    LiberoDataConfig,
    MixedVideoDataConfig,
    MixedVideoResizeBinConfig,
    MixedVideoSourceConfig,
    MixedVideoViewCombinationConfig,
    RobotWinDataConfig,
    SampleConstructionConfig,
    TrainerConfig,
    AuxiliaryValidationTaskConfig,
    ValidationConfig,
    ViewLayoutConfig,
    VisualReadoutConfig,
)
import open_wam.configs.enums as config_enums
from open_wam.configs.inference import InferenceConfig
from open_wam.configs.training import TrainingConfig
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig, normalize_backbone_implementation
from .local_paths import read_yaml_with_local_paths
from .video_timeline import VideoFrameMapping

EnumT = TypeVar("EnumT", bound=StrEnum)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    return read_yaml_with_local_paths(path)


def _raw_enum_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    return value


def _set_contract_default(
    mapping: dict[str, Any],
    *,
    key: str,
    value: Any,
    path: str,
    contract: config_enums.ParallelSequenceContract,
) -> None:
    existing = mapping.get(key)
    if key in mapping and _raw_enum_value(existing) != _raw_enum_value(value):
        raise ValueError(
            f"`policy_variant.parallel_sequence_contract={contract.value}` requires "
            f"`{path}={_raw_enum_value(value)}`, got {existing!r}."
        )
    mapping[key] = value


_LEGACY_PREFIX_PARALLEL_RUNTIME_MODES = frozenset(
    {
        config_enums.ParallelRuntimeMode.LINGBOT_EXACT,
        config_enums.ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
    }
)


def _apply_parallel_sequence_contract(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand sequence contracts into raw defaults before typed parsing."""

    normalized = dict(raw)
    policy_variant_raw = normalized.get("policy_variant")
    if not isinstance(policy_variant_raw, dict):
        return normalized
    policy_variant_raw = dict(policy_variant_raw)
    normalized["policy_variant"] = policy_variant_raw

    contract = _coerce_enum(
        config_enums.ParallelSequenceContract,
        policy_variant_raw.get(
            "parallel_sequence_contract",
            config_enums.ParallelSequenceContract.DEFAULT,
        ),
    )
    if contract == config_enums.ParallelSequenceContract.DEFAULT:
        return normalized

    policy_name = _coerce_enum(
        config_enums.PolicyVariantName,
        policy_variant_raw.get("name", config_enums.PolicyVariantName.POST_LATENT),
    )
    if policy_name not in {
        config_enums.PolicyVariantName.PARALLEL_STREAM,
        config_enums.PolicyVariantName.MOT,
    }:
        raise ValueError(
            f"`policy_variant.parallel_sequence_contract={contract.value}` is only supported for "
            "`policy_variant.name` in {'parallel_stream', 'mot'}."
        )

    if contract == config_enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
        if policy_name == config_enums.PolicyVariantName.PARALLEL_STREAM:
            runtime_mode = _coerce_enum(
                config_enums.ParallelRuntimeMode,
                policy_variant_raw.get("runtime_mode", config_enums.ParallelRuntimeMode.LINGBOT_EXACT),
            )
            if runtime_mode not in _LEGACY_PREFIX_PARALLEL_RUNTIME_MODES:
                allowed = ", ".join(f"'{mode.value}'" for mode in sorted(_LEGACY_PREFIX_PARALLEL_RUNTIME_MODES))
                raise ValueError(
                    "`policy_variant.parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` only "
                    f"supports `policy_variant.runtime_mode` in {{{allowed}}}, got {runtime_mode.value!r}."
                )
        else:
            runtime_mode = _coerce_enum(
                config_enums.MoTRuntimeMode,
                policy_variant_raw.get("runtime_mode", config_enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM),
            )
            if runtime_mode != config_enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM:
                raise ValueError(
                    "`policy_variant.parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` "
                    "requires `policy_variant.runtime_mode=non_joint_two_stream` for MoT/M5."
                )

    if contract not in {
        config_enums.ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        config_enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        raise ValueError(f"Unsupported `policy_variant.parallel_sequence_contract={contract.value}`.")

    if (
        contract == config_enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
        and policy_name == config_enums.PolicyVariantName.MOT
        and "joint_timestep_coupling" not in policy_variant_raw
    ):
        policy_variant_raw["joint_timestep_coupling"] = config_enums.JointTimestepCoupling.INDEPENDENT

    for key, value in (
        ("proprio_context_mode", config_enums.ProprioContextMode.PER_CHUNK_ADDITIVE),
        (
            "context_condition_latent_source",
            config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        ),
        ("history_stream_visibility", config_enums.ParallelHistoryStreamVisibility.VIDEO_ONLY),
        ("use_condition_latents", True),
        ("require_condition_latents", True),
    ):
        _set_contract_default(
            policy_variant_raw,
            key=key,
            value=value,
            path=f"policy_variant.{key}",
            contract=contract,
        )

    data_raw = normalized.get("data")
    if not isinstance(data_raw, dict):
        data_raw = {}
    else:
        data_raw = dict(data_raw)
    normalized["data"] = data_raw
    sample_construction_raw = data_raw.get("sample_construction")
    if not isinstance(sample_construction_raw, dict):
        sample_construction_raw = {}
    else:
        sample_construction_raw = dict(sample_construction_raw)
    data_raw["sample_construction"] = sample_construction_raw

    common_sample_defaults = {
        "condition_source_frame_offset": -1,
        "start_padding_frames": 0,
    }
    if contract == config_enums.ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO:
        sample_defaults = {
            **common_sample_defaults,
            "target_alignment": config_enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT,
            "rollout_context_policy": config_enums.RolloutContextPolicy.ONE_FRAME,
        }
    else:
        sample_defaults = {
            **common_sample_defaults,
            "target_alignment": config_enums.SampleTargetAlignment.LEGACY,
        }
    for key, value in sample_defaults.items():
        _set_contract_default(
            sample_construction_raw,
            key=key,
            value=value,
            path=f"data.sample_construction.{key}",
            contract=contract,
        )

    return normalized


def _apply_checkpoint_runtime_compat(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop stale resolved-config fields that are invalid in authored YAML."""

    normalized = dict(raw)
    data_raw = normalized.get("data")
    if not isinstance(data_raw, dict):
        return normalized
    data_raw = dict(data_raw)
    normalized["data"] = data_raw

    sample_construction_raw = data_raw.get("sample_construction")
    if not isinstance(sample_construction_raw, dict):
        return normalized
    sample_construction_raw = dict(sample_construction_raw)
    data_raw["sample_construction"] = sample_construction_raw

    target_alignment = _raw_enum_value(sample_construction_raw.get("target_alignment"))
    if target_alignment == config_enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT.value:
        for key in ("context_prefix_policy", "context_prefix_frames"):
            sample_construction_raw.pop(key, None)

    mode = _raw_enum_value(sample_construction_raw.get("mode"))
    if mode == config_enums.WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT.value:
        for key in (
            "segment_min_frames",
            "segment_max_frames",
            "randomize_segment_length",
            "randomize_segment_start",
            "require_full_segment",
            "sample_weight_mode",
            "sample_weight_length_power",
        ):
            sample_construction_raw.pop(key, None)

    return normalized


def _coerce_enum(enum_cls: type[EnumT], value: EnumT | str) -> EnumT:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _coerce_optional_enum(enum_cls: type[EnumT], value: EnumT | str | None) -> EnumT | None:
    if value is None:
        return None
    return _coerce_enum(enum_cls, value)


def _coerce_enum_tuple(enum_cls: type[EnumT], values: tuple[EnumT | str, ...] | list[EnumT | str]) -> tuple[EnumT, ...]:
    return tuple(_coerce_enum(enum_cls, value) for value in values)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"`{field_name}` must be boolean, got {value!r}.")


def _coerce_strict_chunk_size(name: str, value: Any) -> int:
    """Coerce strict rollout chunk-size fields with a clear config error."""

    try:
        if isinstance(value, bool):
            raise TypeError
        coerced = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "`sample_construction.target_alignment=next_after_context` requires integer chunk-size fields; "
            f"got {name}={value!r}."
        ) from None
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            "`sample_construction.target_alignment=next_after_context` requires integer chunk-size fields; "
            f"got {name}={value!r}."
        )
    return coerced


def _load_consortium_channel_mappings(raw_value: Any) -> tuple[ConsortiumChannelMappingConfig, ...]:
    mappings_raw = raw_value or ()
    if not isinstance(mappings_raw, (list, tuple)):
        raise ValueError("Expected `channel_mappings` to be a list of mappings.")
    return tuple(
        ConsortiumChannelMappingConfig(
            source_name=str(item["source_name"]),
            target_slot=str(item["target_slot"]),
        )
        for item in mappings_raw
    )


def _load_consortium_episode_selection(raw_value: Any) -> tuple[ConsortiumEpisodeSelectionConfig, ...]:
    selections_raw = raw_value or ()
    if not isinstance(selections_raw, (list, tuple)):
        raise ValueError("Expected explicit consortium episode selections to be a list.")
    return tuple(
        ConsortiumEpisodeSelectionConfig(
            member_id=str(item["member_id"]),
            episode_indices=tuple(int(value) for value in item.get("episode_indices", ())),
        )
        for item in selections_raw
    )


def _load_consortium_members(raw_value: Any) -> tuple[ConsortiumMemberConfig, ...]:
    members_raw = raw_value or ()
    if not isinstance(members_raw, (list, tuple)):
        raise ValueError("Expected `consortium_members` to be a list of member mappings.")
    members: list[ConsortiumMemberConfig] = []
    for item in members_raw:
        members.append(
            ConsortiumMemberConfig(
                member_id=item.get("member_id"),
                repo_id=item.get("repo_id"),
                local_root=item.get("local_root"),
                enabled=item.get("enabled", True),
                source_group=item.get("source_group"),
                include_channels=tuple(item.get("include_channels", ())),
                channel_mappings=_load_consortium_channel_mappings(item.get("channel_mappings")),
                sampling_weight=item.get("sampling_weight"),
            )
        )
    return tuple(members)


def _load_mixed_video_sources(raw_value: Any) -> tuple[MixedVideoSourceConfig, ...]:
    sources_raw = raw_value or ()
    if not isinstance(sources_raw, (list, tuple)):
        raise ValueError("Expected `video_sources` to be a list of source mappings.")
    sources: list[MixedVideoSourceConfig] = []
    for item in sources_raw:
        if not isinstance(item, dict):
            raise ValueError("Expected each `video_sources` entry to be a mapping.")
        sources.append(
            MixedVideoSourceConfig(
                source_id=str(item["source_id"]),
                manifest_csv=str(item["manifest_csv"]),
                repo_id=item.get("repo_id"),
                local_root=item.get("local_root"),
                latent_root=item.get("latent_root"),
                source_format=_coerce_enum(
                    config_enums.MixedVideoSourceFormat,
                    item.get("source_format", "rgb"),
                ),
                latent_key=str(item.get("latent_key", "video_latents")),
                enabled=item.get("enabled", True),
                source_group=item.get("source_group"),
                include_streams=tuple(item.get("include_streams", ())),
                channel_mappings=_load_consortium_channel_mappings(item.get("channel_mappings")),
                sampling_weight=item.get("sampling_weight"),
            )
        )
    return tuple(sources)


def _load_mixed_video_resize_bins(raw_value: Any) -> tuple[MixedVideoResizeBinConfig, ...] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, (list, tuple)):
        raise ValueError("Expected `decode_resize_bins` to be a list of bin mappings.")
    bins: list[MixedVideoResizeBinConfig] = []
    for item in raw_value:
        if not isinstance(item, dict):
            raise ValueError("Expected each `decode_resize_bins` entry to be a mapping.")
        bins.append(
            MixedVideoResizeBinConfig(
                name=str(item["name"]),
                aspect_width=int(item["aspect_width"]),
                aspect_height=int(item["aspect_height"]),
                target_height=int(item["target_height"]),
                target_width=int(item["target_width"]),
                max_pixels=item.get("max_pixels"),
            )
        )
    return tuple(bins)


def _load_mixed_video_view_combinations(raw_value: Any) -> tuple[MixedVideoViewCombinationConfig, ...]:
    combinations_raw = raw_value or ()
    if not isinstance(combinations_raw, (list, tuple)):
        raise ValueError("Expected `latent_view_combinations` to be a list of mappings.")
    combinations: list[MixedVideoViewCombinationConfig] = []
    for item in combinations_raw:
        if not isinstance(item, dict):
            raise ValueError("Expected each `latent_view_combinations` entry to be a mapping.")
        if "name" not in item:
            raise ValueError("Expected each `latent_view_combinations` entry to define `name`.")
        combinations.append(
            MixedVideoViewCombinationConfig(
                name=str(item["name"]),
                slots=tuple(str(value) for value in item.get("slots", ())),
                sampling_weight=float(item.get("sampling_weight", 1.0)),
                source_ids=tuple(str(value) for value in item.get("source_ids", ())),
                enabled=item.get("enabled", True),
            )
        )
    return tuple(combinations)


def _load_mixed_video_fit_mode(
    data_raw: dict[str, Any],
    data_defaults: MixedVideoDataConfig,
) -> config_enums.MixedVideoFrameFitMode:
    if "decode_fit_mode" in data_raw:
        return _coerce_enum(config_enums.MixedVideoFrameFitMode, data_raw["decode_fit_mode"])
    if "decode_center_crop" in data_raw:
        return (
            config_enums.MixedVideoFrameFitMode.CENTER_CROP
            if bool(data_raw["decode_center_crop"])
            else config_enums.MixedVideoFrameFitMode.LETTERBOX_PAD
        )
    return data_defaults.decode_fit_mode


def _validate_mixed_video_wan_causal_buckets(
    *,
    data_config: DataConfig,
    backbone_config: SharedVideoTransformerConfig,
    policy_variant_config: PolicyVariantConfig,
    trainer_config: TrainerConfig,
) -> None:
    if not isinstance(data_config, MixedVideoDataConfig):
        return
    if not isinstance(policy_variant_config, CausalVideoPredictionPolicyConfig):
        return
    if trainer_config.batch_adapter != config_enums.BatchAdapterName.VIEWS:
        return
    if not backbone_config.load_wan_vae_frontend:
        return
    sample_construction = data_config.sample_construction
    if sample_construction.mode != config_enums.WindowSamplingMode.CAUSAL_PREFIX_SUFFIX:
        return

    invalid_buckets: list[str] = []
    max_raw_span = int(sample_construction.num_frames)
    for index, bucket in enumerate(sample_construction.effective_causal_prefix_suffix_buckets):
        raw_observed_frames = int(bucket.observed_frames)
        raw_future_frames = int(bucket.future_frames)
        raw_total_frames = raw_observed_frames + raw_future_frames
        if raw_total_frames > max_raw_span:
            invalid_buckets.append(
                f"#{index} observed_frames={raw_observed_frames} future_frames={raw_future_frames} "
                f"exceeds sample_construction.num_frames={max_raw_span}"
            )
            continue
        try:
            VideoFrameMapping.wan_causal_prefix_suffix(
                raw_observed_frames=raw_observed_frames,
                raw_future_frames=raw_future_frames,
            )
        except ValueError:
            invalid_buckets.append(
                f"#{index} observed_frames={raw_observed_frames} future_frames={raw_future_frames} "
                "maps to zero future Wan latent targets"
            )
    if invalid_buckets:
        formatted = "\n".join(f"- {item}" for item in invalid_buckets)
        raise ValueError(
            "Mixed-video causal buckets with the Wan VAE frontend must produce at least one future latent target. "
            "Wan fresh-clip encoding maps raw frames as frame 0 plus complete 4-frame groups; choose buckets such "
            f"as observed_frames=1, future_frames=4 instead of 1+3.\n{formatted}"
        )


def _load_visual_readout_config(raw_value: Any) -> VisualReadoutConfig | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("Expected `visual_readout` to be a mapping.")
    layer_index = raw_value.get("layer_index")
    layer_indices_raw = raw_value.get("layer_indices")
    if layer_indices_raw is None:
        layer_indices: tuple[int, ...] = ()
    elif isinstance(layer_indices_raw, (list, tuple)):
        layer_indices = tuple(int(value) for value in layer_indices_raw)
    else:
        raise ValueError("Expected `visual_readout.layer_indices` to be a list or tuple.")
    return VisualReadoutConfig(
        source_family=_coerce_enum(
            config_enums.VisualReadoutSourceFamily,
            raw_value["source_family"],
        ),
        layer_index=(None if layer_index is None else int(layer_index)),
        layer_indices=layer_indices,
        fusion_mode=_coerce_enum(
            config_enums.VisualReadoutFusionMode,
            raw_value.get("fusion_mode", config_enums.VisualReadoutFusionMode.NONE),
        ),
        diffusion_extract_timestep=int(raw_value.get("diffusion_extract_timestep", 20)),
        diffusion_extract_step_time=int(raw_value.get("diffusion_extract_step_time", 1)),
    )


def _load_action_mapping_config(raw_value: Any, defaults: ActionMappingConfig) -> ActionMappingConfig:
    raw = raw_value or {}
    if not isinstance(raw, dict):
        raise ValueError("Expected `data.action_mapping` to be a mapping.")
    normalization = _load_action_normalization_config(
        raw.get("normalization", None),
        defaults.normalization,
        field_path="data.action_mapping.normalization",
    )
    return ActionMappingConfig(
        mode=_coerce_enum(
            config_enums.ActionMappingMode,
            raw.get("mode", defaults.mode),
        ),
        source_dim=raw.get("source_dim", defaults.source_dim),
        target_dim=raw.get("target_dim", defaults.target_dim),
        source_to_target_indices=tuple(
            int(value) for value in raw.get("source_to_target_indices", defaults.source_to_target_indices)
        ),
        active_target_indices=tuple(
            int(value) for value in raw.get("active_target_indices", defaults.active_target_indices)
        ),
        inactive_value=float(raw.get("inactive_value", defaults.inactive_value)),
        loss_mask_mode=_coerce_enum(
            config_enums.ActionMappingLossMaskMode,
            raw.get("loss_mask_mode", defaults.loss_mask_mode),
        ),
        sampler_mask_mode=_coerce_enum(
            config_enums.ActionMappingSamplerMaskMode,
            raw.get("sampler_mask_mode", defaults.sampler_mask_mode),
        ),
        normalization=normalization,
    )


def _load_action_normalization_config(
    raw_value: Any,
    defaults: ActionNormalizationConfig,
    *,
    field_path: str,
) -> ActionNormalizationConfig:
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError(f"Expected `{field_path}` to be a mapping.")
    return ActionNormalizationConfig(
        mode=_coerce_enum(
            config_enums.ActionNormalizationMode,
            raw_value.get("mode", defaults.mode),
        ),
        mean=tuple(float(value) for value in raw_value.get("mean", defaults.mean)),
        std=tuple(float(value) for value in raw_value.get("std", defaults.std)),
        q01=tuple(float(value) for value in raw_value.get("q01", defaults.q01)),
        q99=tuple(float(value) for value in raw_value.get("q99", defaults.q99)),
        lower=tuple(float(value) for value in raw_value.get("lower", defaults.lower)),
        upper=tuple(float(value) for value in raw_value.get("upper", defaults.upper)),
        clip_min=raw_value.get("clip_min", defaults.clip_min),
        clip_max=raw_value.get("clip_max", defaults.clip_max),
    )


def _load_generalist_dynamics_mixture_config(
    raw_value: Any,
    defaults: GeneralistDynamicsMixtureConfig,
) -> GeneralistDynamicsMixtureConfig:
    raw = raw_value or {}
    if not isinstance(raw, dict):
        raise ValueError("Expected `data.generalist_dynamics_mixture` to be a mapping.")
    return GeneralistDynamicsMixtureConfig(
        train_latent_root=raw.get("train_latent_root", defaults.train_latent_root),
        val_latent_root=raw.get("val_latent_root", defaults.val_latent_root),
        allow_train_latent_root_for_val=raw.get(
            "allow_train_latent_root_for_val",
            defaults.allow_train_latent_root_for_val,
        ),
        real_joint_weight=raw.get("real_joint_weight", defaults.real_joint_weight),
        real_action_conditioned_video_weight=raw.get(
            "real_action_conditioned_video_weight",
            defaults.real_action_conditioned_video_weight,
        ),
        real_video_conditioned_action_weight=raw.get(
            "real_video_conditioned_action_weight",
            defaults.real_video_conditioned_action_weight,
        ),
        counterfactual_action_conditioned_video_weight=raw.get(
            "counterfactual_action_conditioned_video_weight",
            defaults.counterfactual_action_conditioned_video_weight,
        ),
        counterfactual_video_conditioned_action_weight=raw.get(
            "counterfactual_video_conditioned_action_weight",
            defaults.counterfactual_video_conditioned_action_weight,
        ),
        conditional_history_frames=raw.get("conditional_history_frames", defaults.conditional_history_frames),
        seed=raw.get("seed", defaults.seed),
        length_multiplier=raw.get("length_multiplier", defaults.length_multiplier),
    )


def _load_validation_config(raw_value: Any) -> ValidationConfig:
    raw = raw_value or {}
    if not isinstance(raw, dict):
        raise ValueError("Expected `validation` to be a mapping.")
    tasks_raw = raw.get("auxiliary_tasks", ())
    if tasks_raw is None:
        tasks_raw = ()
    if not isinstance(tasks_raw, (list, tuple)):
        raise ValueError("Expected `validation.auxiliary_tasks` to be a list.")
    tasks: list[AuxiliaryValidationTaskConfig] = []
    for item in tasks_raw:
        if not isinstance(item, dict):
            raise ValueError("Expected each `validation.auxiliary_tasks` entry to be a mapping.")
        if "name" not in item:
            raise ValueError("Expected each `validation.auxiliary_tasks` entry to include `name`.")
        tasks.append(
            AuxiliaryValidationTaskConfig(
                name=item["name"],
                mode_override=_coerce_optional_enum(
                    config_enums.JointDenoiseTrainingMode,
                    item.get("mode_override"),
                ),
                dataset_split=_coerce_enum(
                    config_enums.DataSplit,
                    item.get("dataset_split", config_enums.DataSplit.VAL),
                ),
                source=_coerce_enum(
                    config_enums.AuxiliaryValidationSource,
                    item.get("source", config_enums.AuxiliaryValidationSource.DATASET),
                ),
                max_batches=item.get("max_batches", 16),
                report_prefix=item.get("report_prefix"),
                drop_text_conditioning=item.get("drop_text_conditioning"),
                enabled=item.get("enabled", True),
            )
        )
    return ValidationConfig(auxiliary_tasks=tuple(tasks))


def _load_policy_variant_config(
    policy_variant_raw: dict[str, Any],
    action_head_raw: dict[str, Any],
    data_config: DataConfig,
    backbone_config: SharedVideoTransformerConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
) -> PolicyVariantConfig:
    resolved_raw = dict(policy_variant_raw)
    compatibility_mode = False
    if not resolved_raw:
        compatibility_mode = True
        resolved_raw = {
            "name": config_enums.PolicyVariantName.POST_LATENT,
            "hidden_size": action_head_raw.get("hidden_size", backbone_config.hidden_size),
            "attach_site": config_enums.AttachSite.POST_VISUAL_CORE,
            "pooling_mode": config_enums.PoolingMode.COMPAT_GLOBAL_MEAN,
            "use_state_projection": True,
            "compatibility_mode": True,
        }

    name = _coerce_enum(
        config_enums.PolicyVariantName,
        resolved_raw.get("name", config_enums.PolicyVariantName.POST_LATENT),
    )
    hidden_size = resolved_raw.get("hidden_size", backbone_config.hidden_size)
    if name == config_enums.PolicyVariantName.POST_LATENT:
        return PostLatentPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            pooling_mode=_coerce_enum(
                config_enums.PoolingMode,
                resolved_raw.get(
                    "pooling_mode",
                    (
                        config_enums.PoolingMode.COMPAT_GLOBAL_MEAN
                        if compatibility_mode
                        else config_enums.PoolingMode.PER_FRAME_MEAN
                    ),
                ),
            ),
            query_count=resolved_raw.get("query_count", 0),
            temporal_projection=_coerce_enum(
                config_enums.TemporalProjection,
                resolved_raw.get("temporal_projection", config_enums.TemporalProjection.INTERPOLATE),
            ),
            use_state_projection=resolved_raw.get("use_state_projection", True),
            compatibility_mode=resolved_raw.get("compatibility_mode", compatibility_mode),
            video_condition_input_space=_coerce_enum(
                config_enums.VideoConditionInputSpace,
                resolved_raw.get("video_condition_input_space", config_enums.VideoConditionInputSpace.VIDEO_LATENT),
            ),
            train_video_condition_source=_coerce_enum(
                config_enums.VideoConditionSource,
                resolved_raw.get("train_video_condition_source", config_enums.VideoConditionSource.LOCAL_WINDOW),
            ),
            action_chunk_anchor_mode=_coerce_enum(
                config_enums.ActionChunkAnchorMode,
                resolved_raw.get(
                    "action_chunk_anchor_mode",
                    config_enums.ActionChunkAnchorMode.CURRENT_PLUS_FUTURE,
                ),
            ),
            local_video_window_frames=resolved_raw.get("local_video_window_frames", 4),
            current_video_frame_index=resolved_raw.get("current_video_frame_index", 0),
            visual_readout=_load_visual_readout_config(resolved_raw.get("visual_readout")),
        )
    if name == config_enums.PolicyVariantName.POST_DECODED:
        return PostDecodedPolicyConfig(
            hidden_size=hidden_size,
            decode_feature_mode=_coerce_enum(
                config_enums.DecodeFeatureMode,
                resolved_raw.get("decode_feature_mode", config_enums.DecodeFeatureMode.FRAME_TOKEN_SEQUENCE),
            ),
            pooling_mode=_coerce_enum(
                config_enums.PoolingMode,
                resolved_raw.get("pooling_mode", config_enums.PoolingMode.PER_FRAME_MEAN),
            ),
            temporal_projection=_coerce_enum(
                config_enums.TemporalProjection,
                resolved_raw.get("temporal_projection", config_enums.TemporalProjection.INTERPOLATE),
            ),
            use_state_projection=resolved_raw.get("use_state_projection", True),
            video_condition_input_space=_coerce_enum(
                config_enums.VideoConditionInputSpace,
                resolved_raw.get("video_condition_input_space", config_enums.VideoConditionInputSpace.RGB_VIDEO),
            ),
            train_video_condition_source=_coerce_enum(
                config_enums.VideoConditionSource,
                resolved_raw.get("train_video_condition_source", config_enums.VideoConditionSource.LOCAL_WINDOW),
            ),
            action_chunk_anchor_mode=_coerce_enum(
                config_enums.ActionChunkAnchorMode,
                resolved_raw.get(
                    "action_chunk_anchor_mode",
                    config_enums.ActionChunkAnchorMode.CURRENT_PLUS_FUTURE,
                ),
            ),
            local_video_window_frames=resolved_raw.get("local_video_window_frames", 4),
            current_video_frame_index=resolved_raw.get("current_video_frame_index", 0),
            visual_readout=_load_visual_readout_config(resolved_raw.get("visual_readout")),
        )
    if name == config_enums.PolicyVariantName.VIDEO_SEQUENCE_POLICY:
        return VideoSequencePolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            temporal_projection=_coerce_enum(
                config_enums.TemporalProjection,
                resolved_raw.get("temporal_projection", config_enums.TemporalProjection.INTERPOLATE),
            ),
            visual_readout=_load_visual_readout_config(resolved_raw.get("visual_readout")),
            visual_state_source=_coerce_enum(
                config_enums.VisualStateSource,
                resolved_raw.get("visual_state_source", config_enums.VisualStateSource.DENOISED_VIDEO_TOKENS),
            ),
            visual_denoise_ratio=resolved_raw.get("visual_denoise_ratio", 1.0),
            use_state_context=resolved_raw.get("use_state_context", True),
            use_goal_context=resolved_raw.get("use_goal_context", True),
        )
    if name == config_enums.PolicyVariantName.CAUSAL_VIDEO_PREDICTION:
        return CausalVideoPredictionPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
        )
    if name == config_enums.PolicyVariantName.MOT:
        preset = _coerce_optional_enum(
            config_enums.MoTPreset,
            resolved_raw.get("preset"),
        )
        mot_generalist_training_mode_probs = resolved_raw.get("mot_generalist_training_mode_probs")
        mot_joint_timestep_coupling_default = (
            config_enums.JointTimestepCoupling.INDEPENDENT
            if mot_generalist_training_mode_probs is not None
            else config_enums.JointTimestepCoupling.MATCH_SIGMA
        )
        mot_defaults: dict[str, Any] = {}
        if preset == config_enums.MoTPreset.FASTWAM:
            mot_defaults = {
                "runtime_mode": config_enums.MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE,
                "condition_mode": config_enums.MoTConditionMode.FIRST_FRAME,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.MoTPreset.FASTWAM_JOINT:
            mot_defaults = {
                "runtime_mode": config_enums.MoTRuntimeMode.JOINT_DENOISE,
                "condition_mode": config_enums.MoTConditionMode.FULL_VIDEO,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.MoTPreset.FASTWAM_IDM:
            mot_defaults = {
                "runtime_mode": config_enums.MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE,
                "condition_mode": config_enums.MoTConditionMode.TEACHER_FORCING_COND_VIDEO,
                "teacher_forcing_video_noise_prob": 0.5,
                "video_prefix_frames": 1,
            }
        elif preset == config_enums.MoTPreset.FASTWAM_NON_JOINT:
            # Method-1-non-joint-aligned two-stream MoT: both video and action
            # run through a history-clean / current-noisy split, and the mask
            # disallows same-chunk noisy-to-noisy cross-stream attention.
            mot_defaults = {
                "runtime_mode": config_enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM,
                "condition_mode": config_enums.MoTConditionMode.TEACHER_FORCING_COND_VIDEO,
                "teacher_forcing_video_noise_prob": 0.0,
                "video_prefix_frames": 1,
            }
        context_condition_latent_source = _coerce_enum(
            config_enums.ParallelContextConditionLatentSource,
            resolved_raw.get(
                "context_condition_latent_source",
                config_enums.ParallelContextConditionLatentSource.VIDEO_LATENTS,
            ),
        )
        return MoTPolicyConfig(
            hidden_size=hidden_size,
            attach_site=_coerce_enum(
                config_enums.AttachSite,
                resolved_raw.get("attach_site", config_enums.AttachSite.POST_VISUAL_CORE),
            ),
            preset=preset,
            runtime_mode=_coerce_enum(
                config_enums.MoTRuntimeMode,
                resolved_raw.get(
                    "runtime_mode",
                    mot_defaults.get("runtime_mode", config_enums.MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE),
                ),
            ),
            condition_mode=_coerce_enum(
                config_enums.MoTConditionMode,
                resolved_raw.get(
                    "condition_mode",
                    mot_defaults.get("condition_mode", config_enums.MoTConditionMode.FIRST_FRAME),
                ),
            ),
            action_expert_init_mode=_coerce_enum(
                config_enums.MoTActionExpertInitMode,
                resolved_raw.get(
                    "action_expert_init_mode",
                    config_enums.MoTActionExpertInitMode.VIDEO_WEIGHT_COPY,
                ),
            ),
            video_prefix_frames=resolved_raw.get("video_prefix_frames", mot_defaults.get("video_prefix_frames", 1)),
            teacher_forcing_video_noise_prob=resolved_raw.get(
                "teacher_forcing_video_noise_prob",
                mot_defaults.get("teacher_forcing_video_noise_prob", 0.5),
            ),
            noisy_video_condition_prob=resolved_raw.get("noisy_video_condition_prob", 0.5),
            num_action_layers=resolved_raw.get("num_action_layers", backbone_config.num_layers),
            action_hidden_size=resolved_raw.get("action_hidden_size"),
            action_ffn_dim=resolved_raw.get("action_ffn_dim"),
            video_can_attend_action=resolved_raw.get("video_can_attend_action", True),
            current_block_coupling=(
                _coerce_enum(config_enums.CurrentBlockCoupling, resolved_raw["current_block_coupling"])
                if "current_block_coupling" in resolved_raw
                else None
            ),
            use_text_conditioning=resolved_raw.get("use_text_conditioning", True),
            use_state_conditioning=resolved_raw.get("use_state_conditioning", False),
            proprio_context_mode=_coerce_enum(
                config_enums.ProprioContextMode,
                resolved_raw.get("proprio_context_mode", config_enums.ProprioContextMode.NONE),
            ),
            history_stream_visibility=_coerce_enum(
                config_enums.ParallelHistoryStreamVisibility,
                resolved_raw.get(
                    "history_stream_visibility",
                    config_enums.ParallelHistoryStreamVisibility.FULL,
                ),
            ),
            context_condition_latent_source=context_condition_latent_source,
            use_activation_checkpointing=resolved_raw.get("use_activation_checkpointing", False),
            use_condition_latents=(
                True
                if context_condition_latent_source
                == config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
                else bool(resolved_raw.get("use_condition_latents", True))
            ),
            require_condition_latents=(
                True
                if context_condition_latent_source
                == config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
                else bool(resolved_raw.get("require_condition_latents", False))
            ),
            parallel_sequence_contract=_coerce_enum(
                config_enums.ParallelSequenceContract,
                resolved_raw.get(
                    "parallel_sequence_contract",
                    config_enums.ParallelSequenceContract.DEFAULT,
                ),
            ),
            mot_generalist_training_mode_probs=mot_generalist_training_mode_probs,
            generalist_mode_text_token=_coerce_bool(
                resolved_raw.get("generalist_mode_text_token", False),
                field_name="policy_variant.generalist_mode_text_token",
            ),
            joint_timestep_coupling=_coerce_enum(
                config_enums.JointTimestepCoupling,
                resolved_raw.get(
                    "joint_timestep_coupling",
                    mot_joint_timestep_coupling_default,
                ),
            ),
            couple_action_to_video_timesteps=resolved_raw.get("couple_action_to_video_timesteps"),
            generalist_training_paradigm=_coerce_enum(
                config_enums.GeneralistTrainingParadigm,
                resolved_raw.get(
                    "generalist_training_paradigm",
                    config_enums.GeneralistTrainingParadigm.DEMO_ONLY,
                ),
            ),
        )
    if name == config_enums.PolicyVariantName.REGISTER_ATTACHED:
        return RegisterAttachedPolicyConfig(
            hidden_size=hidden_size,
            num_frame_per_block=resolved_raw.get("num_frame_per_block", 1),
            num_action_per_block=resolved_raw.get("num_action_per_block", 1),
            num_state_per_block=resolved_raw.get("num_state_per_block", 1),
            max_chunk_size=resolved_raw.get("max_chunk_size", inference_config.frame_chunk_size),
            register_layout=_coerce_enum(
                config_enums.RegisterLayout,
                resolved_raw.get("register_layout", config_enums.RegisterLayout.ACTION_THEN_STATE),
            ),
            mask_mode=_coerce_enum(
                config_enums.RegisterMaskMode,
                resolved_raw.get("mask_mode", config_enums.RegisterMaskMode.DREAMZERO_BLOCKWISE),
            ),
            use_state_encoder=resolved_raw.get("use_state_encoder", True),
            action_encoder_type=_coerce_enum(
                config_enums.StreamEncoderType,
                resolved_raw.get("action_encoder_type", config_enums.StreamEncoderType.MLP),
            ),
            state_encoder_type=_coerce_enum(
                config_enums.StreamEncoderType,
                resolved_raw.get("state_encoder_type", config_enums.StreamEncoderType.MLP),
            ),
            couple_action_to_video_blocks=resolved_raw.get("couple_action_to_video_blocks", True),
            structured_block_mode=_coerce_enum(
                config_enums.StructuredBlockMode,
                resolved_raw.get("structured_block_mode", config_enums.StructuredBlockMode.REGISTER_EXPLICIT),
            ),
            structured_time_layout=_coerce_enum(
                config_enums.StructuredTimeLayout,
                resolved_raw.get("structured_time_layout", config_enums.StructuredTimeLayout.VIDEO_ACTION_STATE),
            ),
            structured_frequency_mode=_coerce_enum(
                config_enums.StructuredFrequencyMode,
                resolved_raw.get("structured_frequency_mode", config_enums.StructuredFrequencyMode.STREAM_LOCAL),
            ),
            structured_teacher_forcing_layout=_coerce_enum(
                config_enums.StructuredTeacherForcingLayout,
                resolved_raw.get(
                    "structured_teacher_forcing_layout",
                    config_enums.StructuredTeacherForcingLayout.CLEAN_PREFIX,
                ),
            ),
            structured_attention_kernel=_coerce_enum(
                config_enums.StructuredAttentionKernel,
                resolved_raw.get(
                    "structured_attention_kernel",
                    config_enums.StructuredAttentionKernel.BRANCHWISE_EXPLICIT,
                ),
            ),
            structured_cache_kernel=_coerce_enum(
                config_enums.StructuredCacheKernel,
                resolved_raw.get(
                    "structured_cache_kernel",
                    config_enums.StructuredCacheKernel.BRANCHWISE_ROLLOUT_EXPLICIT,
                ),
            ),
            stream_input_adapter_family=_coerce_enum(
                config_enums.StreamInputAdapterFamily,
                resolved_raw.get(
                    "stream_input_adapter_family",
                    config_enums.StreamInputAdapterFamily.STRUCTURED_REGISTER_STREAMS,
                ),
            ),
            stream_output_head_family=_coerce_enum(
                config_enums.StreamOutputHeadFamily,
                resolved_raw.get(
                    "stream_output_head_family",
                    config_enums.StreamOutputHeadFamily.STRUCTURED_JOINT_FLOW,
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
        current_block_coupling = (
            _coerce_enum(
                config_enums.CurrentBlockCoupling,
                resolved_raw["current_block_coupling"],
            )
            if "current_block_coupling" in resolved_raw
            else None
        )
        preserve_video_pretrain_history = bool(
            resolved_raw.get("preserve_video_pretrain_history", False)
        )
        history_stream_visibility = _coerce_enum(
            config_enums.ParallelHistoryStreamVisibility,
            resolved_raw.get(
                "history_stream_visibility",
                (
                    config_enums.ParallelHistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
                    if preserve_video_pretrain_history
                    else config_enums.ParallelHistoryStreamVisibility.FULL
                ),
            ),
        )
        context_condition_latent_source = _coerce_enum(
            config_enums.ParallelContextConditionLatentSource,
            resolved_raw.get(
                "context_condition_latent_source",
                config_enums.ParallelContextConditionLatentSource.VIDEO_LATENTS,
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
            compact_runtime_mode = runtime_mode in {
                config_enums.ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
                config_enums.ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
            }
            if exact_runtime_mode and current_block_coupling is None:
                raise ValueError(
                    "proprio_context_mode=per_chunk_additive requires "
                    "`policy_variant.current_block_coupling` for Method-1 chunk semantics."
                )
            if not exact_runtime_mode and not compact_runtime_mode:
                raise ValueError(
                    "proprio_context_mode=per_chunk_additive is only supported for "
                    "LingBot exact and compact current-frame Method-1 runtime modes."
                )
        use_condition_latents = (
            True
            if context_condition_latent_source
            == config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
            else bool(resolved_raw.get("use_condition_latents", True))
        )
        require_condition_latents = (
            True
            if context_condition_latent_source
            == config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
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
            couple_action_to_video_timesteps=resolved_raw.get("couple_action_to_video_timesteps"),
            joint_denoise_training_mode_probs=resolved_raw.get("joint_denoise_training_mode_probs"),
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
            current_block_coupling=current_block_coupling,
            preserve_video_pretrain_history=preserve_video_pretrain_history,
            history_stream_visibility=history_stream_visibility,
            context_condition_latent_source=context_condition_latent_source,
            use_condition_latents=use_condition_latents,
            proprio_context_mode=proprio_context_mode,
            require_condition_latents=require_condition_latents,
            parallel_sequence_contract=_coerce_enum(
                config_enums.ParallelSequenceContract,
                resolved_raw.get(
                    "parallel_sequence_contract",
                    config_enums.ParallelSequenceContract.DEFAULT,
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


def _load_action_decoder_config(
    action_decoder_raw: dict[str, Any],
    policy_variant_config: PolicyVariantConfig,
    data_config: DataConfig,
    backbone_config: SharedVideoTransformerConfig,
) -> ActionDecoderConfig:
    resolved_raw = dict(action_decoder_raw)
    if not resolved_raw:
        if policy_variant_config.name == config_enums.PolicyVariantName.REGISTER_ATTACHED:
            resolved_raw["name"] = config_enums.ActionDecoderName.REGISTER
        elif policy_variant_config.name == config_enums.PolicyVariantName.MOT:
            resolved_raw["name"] = config_enums.ActionDecoderName.MOT
        elif policy_variant_config.name == config_enums.PolicyVariantName.CAUSAL_VIDEO_PREDICTION:
            resolved_raw["name"] = config_enums.ActionDecoderName.VIDEO_ONLY
        elif policy_variant_config.name == config_enums.PolicyVariantName.VIDEO_SEQUENCE_POLICY:
            resolved_raw["name"] = config_enums.ActionDecoderName.VPP
        elif policy_variant_config.name == config_enums.PolicyVariantName.POST_DECODED:
            resolved_raw["name"] = config_enums.ActionDecoderName.VIDEO_CONDITIONED
        elif (
            policy_variant_config.name == config_enums.PolicyVariantName.POST_LATENT
            and isinstance(policy_variant_config, PostLatentPolicyConfig)
            and not policy_variant_config.compatibility_mode
        ):
            resolved_raw["name"] = config_enums.ActionDecoderName.VIDEO_CONDITIONED
        elif (
            policy_variant_config.name == config_enums.PolicyVariantName.PARALLEL_STREAM
            and isinstance(policy_variant_config, ParallelStreamPolicyConfig)
            and policy_variant_config.runtime_mode
            in {
                config_enums.ParallelRuntimeMode.LINGBOT_EXACT,
                config_enums.ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
                config_enums.ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
                config_enums.ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
            }
        ):
            resolved_raw["name"] = config_enums.ActionDecoderName.LINGBOT_PARALLEL
        else:
            resolved_raw["name"] = config_enums.ActionDecoderName.MLP

    name = _coerce_enum(config_enums.ActionDecoderName, resolved_raw["name"])
    if name == config_enums.ActionDecoderName.MLP and policy_variant_config.name == config_enums.PolicyVariantName.MOT:
        # Compatibility path for early MoT YAMLs that used `mlp_decoder` as a placeholder.
        name = config_enums.ActionDecoderName.MOT
    hidden_size = resolved_raw.get(
        "hidden_size",
        (
            backbone_config.hidden_size
            if name == config_enums.ActionDecoderName.VIDEO_CONDITIONED
            else policy_variant_config.hidden_size
        ),
    )
    action_dim = resolved_raw.get("action_dim", data_config.action_schema.action_dim)
    action_horizon = resolved_raw.get("action_horizon", data_config.action_schema.action_horizon)
    dropout = resolved_raw.get("dropout", 0.0)

    if name == config_enums.ActionDecoderName.MLP:
        return MLPActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.REGISTER:
        return RegisterActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.DECODED_FEATURE:
        return DecodedFeatureActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.VIDEO_CONDITIONED:
        if isinstance(policy_variant_config, (PostLatentPolicyConfig, PostDecodedPolicyConfig)):
            input_space = policy_variant_config.video_condition_input_space
            anchor_mode = policy_variant_config.action_chunk_anchor_mode
        else:
            input_space = config_enums.VideoConditionInputSpace.VIDEO_LATENT
            anchor_mode = config_enums.ActionChunkAnchorMode.CURRENT_PLUS_FUTURE
        return VideoConditionedActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
            context_dim=resolved_raw.get("context_dim", backbone_config.hidden_size),
            text_context_dim=resolved_raw.get("text_context_dim", backbone_config.text_dim),
            state_dim=resolved_raw.get("state_dim", data_config.action_schema.state_dim),
            freq_dim=resolved_raw.get("freq_dim", backbone_config.freq_dim),
            num_layers=resolved_raw.get("num_layers", backbone_config.num_layers),
            num_heads=resolved_raw.get("num_heads", backbone_config.num_heads),
            attention_head_dim=resolved_raw.get(
                "attention_head_dim",
                backbone_config.attention_head_dim or (backbone_config.hidden_size // backbone_config.num_heads),
            ),
            ffn_dim=resolved_raw.get(
                "ffn_dim",
                backbone_config.ffn_dim or (backbone_config.hidden_size * backbone_config.mlp_ratio),
            ),
            cross_attn_norm=resolved_raw.get("cross_attn_norm", backbone_config.cross_attn_norm),
            eps=resolved_raw.get("eps", backbone_config.latent_norm_eps),
            input_space=_coerce_enum(
                config_enums.VideoConditionInputSpace,
                resolved_raw.get("input_space", input_space),
            ),
            train_mode=_coerce_enum(
                config_enums.VideoConditionTrainMode,
                resolved_raw.get(
                    "train_mode",
                    config_enums.VideoConditionTrainMode.ROLLOUT_WINDOW_DIFFUSION,
                ),
            ),
            action_chunk_anchor_mode=_coerce_enum(
                config_enums.ActionChunkAnchorMode,
                resolved_raw.get("action_chunk_anchor_mode", anchor_mode),
            ),
            action_expert_init_mode=_coerce_enum(
                config_enums.ActionExpertInitMode,
                resolved_raw.get(
                    "action_expert_init_mode",
                    config_enums.ActionExpertInitMode.VIDEO_WEIGHT_COPY,
                ),
            ),
            rollout_chunk_steps=resolved_raw.get("rollout_chunk_steps", 1),
            direct_latent_channels=resolved_raw.get("direct_latent_channels", backbone_config.latent_channels),
            direct_rgb_patch_size=resolved_raw.get("direct_rgb_patch_size", 16),
            use_text_conditioning=resolved_raw.get("use_text_conditioning", True),
            use_state_conditioning=resolved_raw.get("use_state_conditioning", True),
        )
    if name == config_enums.ActionDecoderName.VPP:
        return VPPActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
            temporal_compression_adapter_family=_coerce_enum(
                config_enums.TemporalCompressionAdapterFamily,
                resolved_raw.get(
                    "temporal_compression_adapter_family",
                    config_enums.TemporalCompressionAdapterFamily.TEMPORAL_LATENT_RESAMPLER_3D,
                ),
            ),
            sequence_denoiser_family=_coerce_enum(
                config_enums.SequenceDenoiserFamily,
                resolved_raw.get(
                    "sequence_denoiser_family",
                    config_enums.SequenceDenoiserFamily.GENERIC_TRANSFORMER,
                ),
            ),
            goal_conditioning_adapter_family=_coerce_enum(
                config_enums.GoalConditioningAdapterFamily,
                resolved_raw.get(
                    "goal_conditioning_adapter_family",
                    config_enums.GoalConditioningAdapterFamily.PASSTHROUGH,
                ),
            ),
            state_sequence_adapter_family=_coerce_enum(
                config_enums.StateSequenceAdapterFamily,
                resolved_raw.get(
                    "state_sequence_adapter_family",
                    config_enums.StateSequenceAdapterFamily.LINEAR,
                ),
            ),
            action_generation_backend=_coerce_enum(
                config_enums.ActionGenerationBackendFamily,
                resolved_raw.get(
                    "action_generation_backend",
                    config_enums.ActionGenerationBackendFamily.EDM_DIFFUSION,
                ),
            ),
            diffusion_noise_schedule=_coerce_enum(
                config_enums.DiffusionNoiseSchedule,
                resolved_raw.get(
                    "diffusion_noise_schedule",
                    config_enums.DiffusionNoiseSchedule.EXPONENTIAL,
                ),
            ),
            diffusion_sampler=_coerce_enum(
                config_enums.DiffusionSampler,
                resolved_raw.get("diffusion_sampler", config_enums.DiffusionSampler.DDIM),
            ),
            num_sampling_steps=resolved_raw.get("num_sampling_steps"),
            rollout_chunk_steps=resolved_raw.get("rollout_chunk_steps"),
            compressed_tokens_per_frame=resolved_raw.get("compressed_tokens_per_frame", 2),
            compression_depth=resolved_raw.get("compression_depth", 2),
            temporal_compression_max_frames=resolved_raw.get("temporal_compression_max_frames", 32),
            num_heads=resolved_raw.get("num_heads", 8),
            encoder_layers=resolved_raw.get("encoder_layers", 2),
            decoder_layers=resolved_raw.get("decoder_layers", 2),
            sigma_data=resolved_raw.get("sigma_data", 0.5),
            sigma_min=resolved_raw.get("sigma_min", 0.001),
            sigma_max=resolved_raw.get("sigma_max", 80.0),
        )
    if name == config_enums.ActionDecoderName.LINGBOT_PARALLEL:
        return LingbotParallelActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
            recovered_osc_loss_weight=resolved_raw.get("recovered_osc_loss_weight", 0.0),
            recovered_osc_position_scale=resolved_raw.get(
                "recovered_osc_position_scale",
                LingbotParallelActionDecoderConfig.recovered_osc_position_scale,
            ),
            recovered_osc_rotation_scale=resolved_raw.get(
                "recovered_osc_rotation_scale",
                LingbotParallelActionDecoderConfig.recovered_osc_rotation_scale,
            ),
        )
    if name == config_enums.ActionDecoderName.MOT:
        return MoTActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    if name == config_enums.ActionDecoderName.VIDEO_ONLY:
        return VideoOnlyActionDecoderConfig(
            hidden_size=hidden_size,
            action_dim=action_dim,
            action_horizon=action_horizon,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported action decoder '{name}'.")


def _validate_cross_config_contracts(
    *,
    data_config: DataConfig,
    policy_variant_config: PolicyVariantConfig,
) -> None:
    if not isinstance(policy_variant_config, (ParallelStreamPolicyConfig, MoTPolicyConfig)):
        return
    if (
        policy_variant_config.context_condition_latent_source
        != config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    ):
        return
    condition_source_frame_offset = int(data_config.sample_construction.condition_source_frame_offset)
    if condition_source_frame_offset != -1:
        raise ValueError(
            "`policy_variant.context_condition_latent_source=single_frame_condition_latent` requires "
            "`data.sample_construction.condition_source_frame_offset=-1` so the clean context latent is encoded "
            "from the raw frame immediately before the target latent span. Offset 0 can expose the first target "
            "raw frame and is not a safe default; use a separate explicit ablation path if that behavior is intended."
        )


def validate_experiment_config_runtime_contract(config: ExperimentConfig) -> ExperimentConfig:
    """Validate cross-section runtime contracts after YAML and CLI overrides."""

    if (
        isinstance(config.policy_variant, MoTPolicyConfig)
        and config.policy_variant.mot_generalist_training_mode_probs is not None
        and (int(config.data.train_batch_size) != 1 or int(config.data.val_batch_size) != 1)
    ):
        raise ValueError(
            "`policy_variant.mot_generalist_training_mode_probs` currently requires "
            "`data.train_batch_size = data.val_batch_size = 1` because M5 GJD samples one mode per "
            "segment/forward pass and forced per-sample metadata is only unambiguous for rank-local batch size 1."
        )

    if getattr(config.policy_variant, "generalist_training_paradigm", None) == config_enums.GeneralistTrainingParadigm.MIXED_DYNAMICS:
        if config.trainer.batch_adapter != config_enums.BatchAdapterName.LATENTS:
            raise ValueError(
                "`policy_variant.generalist_training_paradigm=mixed_dynamics` requires "
                "`trainer.batch_adapter=latents` because the mixed-dynamics source mixture wraps latent datasets."
            )
        sample_construction = config.data.sample_construction
        if sample_construction.sample_order_mode == config_enums.SampleOrderMode.REPLACEMENT:
            raise ValueError(
                "`data.sample_construction.sample_order_mode=replacement` is not supported with "
                "`policy_variant.generalist_training_paradigm=mixed_dynamics` because the mixed-dynamics "
                "wrapper owns source sampling and would bypass the local-latent replacement sampler."
            )
        if sample_construction.sample_weight_mode != config_enums.SampleWeightMode.UNIFORM:
            raise ValueError(
                "`data.sample_construction.sample_weight_mode` must be `uniform` with "
                "`policy_variant.generalist_training_paradigm=mixed_dynamics` because the mixed-dynamics "
                "wrapper owns source sampling and would bypass local-latent sample weights."
            )

    if config.data.sample_construction.target_alignment == config_enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT:
        strict_chunk_sources = {
            "data.sample_construction.chunk_size": config.data.sample_construction.chunk_size,
            "training.chunk_size": config.training.chunk_size,
            "inference.frame_chunk_size": config.inference.frame_chunk_size,
        }
        policy_frame_chunk_size = getattr(config.policy_variant, "frame_chunk_size", None)
        if policy_frame_chunk_size is not None:
            strict_chunk_sources["policy_variant.frame_chunk_size"] = policy_frame_chunk_size
        invalid_chunk_sources = {
            name: value
            for name, value in strict_chunk_sources.items()
            if _coerce_strict_chunk_size(name, value) != 4
        }
        if invalid_chunk_sources:
            joined = ", ".join(f"{name}={value}" for name, value in sorted(invalid_chunk_sources.items()))
            raise ValueError(
                "`sample_construction.target_alignment=next_after_context` requires fixed 4-frame chunks "
                f"across data/training/inference/policy; got {joined}."
            )

    _validate_cross_config_contracts(
        data_config=config.data,
        policy_variant_config=config.policy_variant,
    )
    return config


def apply_parallel_sequence_contract(
    config: ExperimentConfig,
    *,
    explicit_override_keys: Collection[str] | None = None,
) -> ExperimentConfig:
    """Apply typed sequence-contract defaults after config overrides."""

    policy_variant = config.policy_variant
    contract = _coerce_enum(
        config_enums.ParallelSequenceContract,
        getattr(policy_variant, "parallel_sequence_contract", config_enums.ParallelSequenceContract.DEFAULT),
    )
    if contract == config_enums.ParallelSequenceContract.DEFAULT:
        return validate_experiment_config_runtime_contract(config)

    policy_name = _coerce_enum(config_enums.PolicyVariantName, policy_variant.name)
    if policy_name not in {
        config_enums.PolicyVariantName.PARALLEL_STREAM,
        config_enums.PolicyVariantName.MOT,
    }:
        raise ValueError(
            f"`policy_variant.parallel_sequence_contract={contract.value}` is only supported for "
            "`policy_variant.name` in {'parallel_stream', 'mot'}."
        )

    if contract == config_enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
        if policy_name == config_enums.PolicyVariantName.PARALLEL_STREAM:
            runtime_mode = _coerce_enum(
                config_enums.ParallelRuntimeMode,
                getattr(policy_variant, "runtime_mode", config_enums.ParallelRuntimeMode.LINGBOT_EXACT),
            )
            if runtime_mode not in _LEGACY_PREFIX_PARALLEL_RUNTIME_MODES:
                allowed = ", ".join(f"'{mode.value}'" for mode in sorted(_LEGACY_PREFIX_PARALLEL_RUNTIME_MODES))
                raise ValueError(
                    "`policy_variant.parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` only "
                    f"supports `policy_variant.runtime_mode` in {{{allowed}}}, got {runtime_mode.value!r}."
                )
        else:
            runtime_mode = _coerce_enum(
                config_enums.MoTRuntimeMode,
                getattr(policy_variant, "runtime_mode", config_enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM),
            )
            if runtime_mode != config_enums.MoTRuntimeMode.NON_JOINT_TWO_STREAM:
                raise ValueError(
                    "`policy_variant.parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` "
                    "requires `policy_variant.runtime_mode=non_joint_two_stream` for MoT/M5."
                )

    if contract not in {
        config_enums.ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO,
        config_enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO,
    }:
        raise ValueError(f"Unsupported `policy_variant.parallel_sequence_contract={contract.value}`.")

    policy_updates: dict[str, Any] = {
        "proprio_context_mode": config_enums.ProprioContextMode.PER_CHUNK_ADDITIVE,
        "context_condition_latent_source": config_enums.ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT,
        "history_stream_visibility": config_enums.ParallelHistoryStreamVisibility.VIDEO_ONLY,
    }
    if hasattr(policy_variant, "use_condition_latents"):
        policy_updates["use_condition_latents"] = True
    if hasattr(policy_variant, "require_condition_latents"):
        policy_updates["require_condition_latents"] = True
    if (
        contract == config_enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
        and hasattr(policy_variant, "joint_timestep_coupling")
    ):
        explicit_keys = set(explicit_override_keys or ())
        contract_set_by_cli = "policy_variant.parallel_sequence_contract" in explicit_keys
        if (
            contract_set_by_cli
            and "policy_variant.joint_timestep_coupling" not in explicit_keys
            and policy_variant.joint_timestep_coupling == config_enums.JointTimestepCoupling.MATCH_SIGMA
        ):
            policy_updates["joint_timestep_coupling"] = config_enums.JointTimestepCoupling.INDEPENDENT
    updated_policy_variant = replace(policy_variant, **policy_updates)

    sample_updates: dict[str, Any] = {
        "condition_source_frame_offset": -1,
        "start_padding_frames": 0,
    }
    if contract == config_enums.ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO:
        sample_updates.update(
            {
                "target_alignment": config_enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT,
                "rollout_context_policy": config_enums.RolloutContextPolicy.ONE_FRAME,
            }
        )
    else:
        sample_updates.update(
            {
                "target_alignment": config_enums.SampleTargetAlignment.LEGACY,
            }
        )
    updated_sample_construction = replace(config.data.sample_construction, **sample_updates)
    updated_data = replace(config.data, sample_construction=updated_sample_construction)
    return validate_experiment_config_runtime_contract(
        replace(config, data=updated_data, policy_variant=updated_policy_variant)
    )


_PARALLEL_SEQUENCE_CONTRACT_MANAGED_OVERRIDE_KEYS = frozenset(
    {
        "policy_variant.proprio_context_mode",
        "policy_variant.context_condition_latent_source",
        "policy_variant.history_stream_visibility",
        "policy_variant.use_condition_latents",
        "policy_variant.require_condition_latents",
        "policy_variant.joint_timestep_coupling",
        "data.sample_construction.target_alignment",
        "data.sample_construction.rollout_context_policy",
        "data.sample_construction.condition_source_frame_offset",
        "data.sample_construction.start_padding_frames",
    }
)


def validate_parallel_sequence_contract_override_keys(
    overrides: Mapping[str, Any],
    *,
    contract_value: Any | None = None,
) -> None:
    """Reject ambiguous CLI overrides of fields owned by a sequence contract."""

    resolved_contract_value = overrides.get("policy_variant.parallel_sequence_contract", contract_value)
    if resolved_contract_value is None:
        return
    contract = _coerce_enum(config_enums.ParallelSequenceContract, resolved_contract_value)
    if contract == config_enums.ParallelSequenceContract.DEFAULT:
        return
    managed_keys = set(_PARALLEL_SEQUENCE_CONTRACT_MANAGED_OVERRIDE_KEYS)
    if contract == config_enums.ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO:
        managed_keys.discard("policy_variant.joint_timestep_coupling")
    conflicting_keys = sorted(key for key in overrides if key in managed_keys)
    if conflicting_keys:
        joined = ", ".join(f"`{key}`" for key in conflicting_keys)
        raise ValueError(
            f"`policy_variant.parallel_sequence_contract={contract.value}` owns {joined}; "
            "drop the contract or drop the individual override(s)."
        )


def load_experiment_config(path: str | Path, *, checkpoint_runtime_compat: bool = False) -> ExperimentConfig:
    """Load one root experiment YAML into the typed config boundary."""

    raw = _read_yaml(path)
    if checkpoint_runtime_compat:
        raw = _apply_checkpoint_runtime_compat(raw)
    raw = _apply_parallel_sequence_contract(raw)
    data_raw = raw.get("data", {})
    action_schema_raw = data_raw.get("action_schema", {})
    action_target_raw = data_raw.get("action_target", {})
    sample_construction_raw = data_raw.get("sample_construction", {})
    dataset_name = data_raw.get("dataset_name", "robotwin")
    dataset_type = data_raw.get("dataset_type")

    # Resolve defaults in two stages:
    # 1. known benchmark presets such as RobotWin and LIBERO
    # 2. a fully generic multiview fallback for custom sources
    #
    # This keeps new-source onboarding mostly declarative. A collaborator can
    # often add a new dataset config by specifying `dataset_type`, camera names,
    # layouts, and action/state schema in YAML without editing the loader.
    if dataset_name == "libero":
        if dataset_type == "libero_hdf5":
            data_defaults = LiberoDataConfig(dataset_type="libero_hdf5", repo_id=None)
        else:
            data_defaults = LiberoDataConfig()
        data_config_cls = LiberoDataConfig
    elif dataset_type == "lerobot_consortium" or dataset_name == "lerobot_consortium":
        data_defaults = LeRobotConsortiumDataConfig()
        data_config_cls = LeRobotConsortiumDataConfig
    elif dataset_type == "mixed_video" or dataset_name == "mixed_video":
        data_defaults = MixedVideoDataConfig(
            video_sources=_load_mixed_video_sources(data_raw.get("video_sources")),
        )
        data_config_cls = MixedVideoDataConfig
    elif dataset_name == "calvin" or dataset_type == "calvin_npz":
        data_defaults = CalvinDataConfig()
        data_config_cls = CalvinDataConfig
    elif dataset_name == "robotwin":
        data_defaults = RobotWinDataConfig()
        data_config_cls = RobotWinDataConfig
    elif dataset_type == "lerobot_v2":
        data_defaults = GenericDataConfig(dataset_name=dataset_name, dataset_type="lerobot_v2")
        data_config_cls = GenericDataConfig
    else:
        resolved_type = dataset_type or "synthetic_multiview"
        data_defaults = GenericDataConfig(dataset_name=dataset_name, dataset_type=resolved_type)
        data_config_cls = GenericDataConfig

    sample_construction_mode = _coerce_enum(
        config_enums.WindowSamplingMode,
        sample_construction_raw.get("mode", data_defaults.sample_construction.mode),
    )
    sample_target_alignment = _coerce_enum(
        config_enums.SampleTargetAlignment,
        sample_construction_raw.get(
            "target_alignment",
            data_defaults.sample_construction.target_alignment,
        ),
    )
    if sample_target_alignment == config_enums.SampleTargetAlignment.NEXT_AFTER_CONTEXT:
        legacy_context_keys = [key for key in ("context_prefix_policy", "context_prefix_frames") if key in sample_construction_raw]
        if legacy_context_keys:
            joined = ", ".join(f"`{key}`" for key in legacy_context_keys)
            raise ValueError(
                "`sample_construction.target_alignment=next_after_context` uses "
                "`rollout_context_policy` / `rollout_context_frames`; remove legacy context fields: "
                f"{joined}."
            )
    if sample_construction_mode == config_enums.WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT:
        legacy_hierarchical_keys = (
            "segment_min_frames",
            "segment_max_frames",
            "randomize_segment_length",
            "randomize_segment_start",
            "require_full_segment",
            "sample_weight_mode",
            "sample_weight_length_power",
        )
        present_legacy_keys = [key for key in legacy_hierarchical_keys if key in sample_construction_raw]
        if present_legacy_keys:
            joined = ", ".join(f"`{key}`" for key in present_legacy_keys)
            raise ValueError(
                "`sample_construction.mode=hierarchical_fixed_segment` uses `segment_frames`, padding policies, "
                f"and hierarchical powers; remove legacy fields: {joined}."
            )

    default_view_layout = [
        {
            "source_name": view.source_name,
            "canonical_name": view.canonical_name,
            "top": view.top,
            "left": view.left,
            "height": view.height,
            "width": view.width,
        }
        for view in data_defaults.view_layout
    ]
    view_layout_raw = data_raw.get("view_layout", default_view_layout)
    view_layout = tuple(
        ViewLayoutConfig(
            source_name=view["source_name"],
            canonical_name=view.get("canonical_name", view["source_name"]),
            top=view["top"],
            left=view["left"],
            height=view["height"],
            width=view["width"],
        )
        for view in view_layout_raw
    )

    common_data_kwargs = dict(
        dataset_name=dataset_name,
        dataset_type=data_raw.get("dataset_type", data_defaults.dataset_type),
        repo_id=data_raw.get("repo_id", data_defaults.repo_id),
        local_root=data_raw.get("local_root", data_defaults.local_root),
        val_local_root=data_raw.get("val_local_root", data_defaults.val_local_root),
        empty_text_embedding_path=data_raw.get(
            "empty_text_embedding_path",
            data_defaults.empty_text_embedding_path,
        ),
        latent_root=data_raw.get("latent_root", data_defaults.latent_root),
        latent_subdir=data_raw.get("latent_subdir", data_defaults.latent_subdir),
        latent_window_profile=_coerce_enum(
            config_enums.LatentWindowProfile,
            data_raw.get("latent_window_profile", data_defaults.latent_window_profile),
        ),
        latent_temporal_layout=_coerce_enum(
            config_enums.LatentTemporalLayout,
            data_raw.get("latent_temporal_layout", data_defaults.latent_temporal_layout),
        ),
        split=_coerce_enum(config_enums.DataSplit, data_raw.get("split", data_defaults.split)),
        cache_dir=data_raw.get("cache_dir", data_defaults.cache_dir),
        camera_names=tuple(data_raw.get("camera_names", data_defaults.camera_names)),
        latent_camera_names=tuple(data_raw.get("latent_camera_names", data_defaults.latent_camera_names)),
        canonical_height=data_raw.get("canonical_height", data_defaults.canonical_height),
        canonical_width=data_raw.get("canonical_width", data_defaults.canonical_width),
        view_layout=view_layout,
        num_frames=data_raw.get("num_frames", data_defaults.num_frames),
        frame_stride=data_raw.get("frame_stride", data_defaults.frame_stride),
        sample_stride=data_raw.get("sample_stride", data_defaults.sample_stride),
        episode_cache_size=data_raw.get("episode_cache_size", data_defaults.episode_cache_size),
        train_fraction=data_raw.get("train_fraction", data_defaults.train_fraction),
        split_seed=data_raw.get("split_seed", data_defaults.split_seed),
        max_train_episodes=data_raw.get("max_train_episodes", data_defaults.max_train_episodes),
        max_val_episodes=data_raw.get("max_val_episodes", data_defaults.max_val_episodes),
        replay_status_path=data_raw.get("replay_status_path", data_defaults.replay_status_path),
        val_replay_status_path=data_raw.get("val_replay_status_path", data_defaults.val_replay_status_path),
        replay_status_policy=_coerce_enum(
            config_enums.ReplayStatusPolicy,
            data_raw.get("replay_status_policy", data_defaults.replay_status_policy),
        ),
        require_replay_status=data_raw.get("require_replay_status", data_defaults.require_replay_status),
        val_replay_status_policy=_coerce_optional_enum(
            config_enums.ReplayStatusPolicy,
            data_raw.get("val_replay_status_policy", data_defaults.val_replay_status_policy),
        ),
        val_require_replay_status=data_raw.get(
            "val_require_replay_status",
            data_defaults.val_require_replay_status,
        ),
        train_batch_size=data_raw.get("train_batch_size", data_defaults.train_batch_size),
        val_batch_size=data_raw.get("val_batch_size", data_defaults.val_batch_size),
        num_workers=data_raw.get("num_workers", data_defaults.num_workers),
        action_schema=ActionSchemaConfig(
            action_dim=action_schema_raw.get("action_dim", data_defaults.action_schema.action_dim),
            action_horizon=action_schema_raw.get("action_horizon", data_defaults.action_schema.action_horizon),
            state_dim=action_schema_raw.get("state_dim", data_defaults.action_schema.state_dim),
            state_horizon=action_schema_raw.get("state_horizon", data_defaults.action_schema.state_horizon),
        ),
        action_target=ActionTargetConfig(
            representation=_coerce_enum(
                config_enums.ActionTargetRepresentation,
                action_target_raw.get("representation", data_defaults.action_target.representation),
            ),
            source_key=action_target_raw.get("source_key", data_defaults.action_target.source_key),
            pose_source_key=action_target_raw.get("pose_source_key", data_defaults.action_target.pose_source_key),
            state_encoding=_coerce_enum(
                config_enums.ActionTargetStateEncoding,
                action_target_raw.get("state_encoding", data_defaults.action_target.state_encoding),
            ),
            reference_source=_coerce_enum(
                config_enums.ActionTargetReferenceSource,
                action_target_raw.get("reference_source", data_defaults.action_target.reference_source),
            ),
            rotation_representation=_coerce_enum(
                config_enums.RotationRepresentation,
                action_target_raw.get("rotation_representation", data_defaults.action_target.rotation_representation),
            ),
            include_gripper=action_target_raw.get("include_gripper", data_defaults.action_target.include_gripper),
            gripper_representation=_coerce_enum(
                config_enums.GripperRepresentation,
                action_target_raw.get("gripper_representation", data_defaults.action_target.gripper_representation),
            ),
            gripper_action_index=action_target_raw.get(
                "gripper_action_index",
                data_defaults.action_target.gripper_action_index,
            ),
            gripper_position_source_key=action_target_raw.get(
                "gripper_position_source_key",
                data_defaults.action_target.gripper_position_source_key,
            ),
            joint_position_source_key=action_target_raw.get(
                "joint_position_source_key",
                data_defaults.action_target.joint_position_source_key,
            ),
            joint_position_normalization=_load_action_normalization_config(
                action_target_raw.get("joint_position_normalization"),
                data_defaults.action_target.joint_position_normalization,
                field_path="data.action_target.joint_position_normalization",
            ),
            normalization=_load_action_normalization_config(
                action_target_raw.get("normalization"),
                data_defaults.action_target.normalization,
                field_path="data.action_target.normalization",
            ),
        ),
        action_mapping=_load_action_mapping_config(
            data_raw.get("action_mapping"),
            data_defaults.action_mapping,
        ),
        sample_construction=SampleConstructionConfig(
            mode=sample_construction_mode,
            anchor_policy=_coerce_enum(
                config_enums.AnchorPolicy,
                sample_construction_raw.get(
                    "anchor_policy",
                    data_defaults.sample_construction.anchor_policy,
                ),
            ),
            num_frames=sample_construction_raw.get(
                "num_frames",
                data_defaults.sample_construction.num_frames,
            ),
            action_horizon=sample_construction_raw.get(
                "action_horizon",
                data_defaults.sample_construction.action_horizon,
            ),
            state_horizon=sample_construction_raw.get(
                "state_horizon",
                data_defaults.sample_construction.state_horizon,
            ),
            state_anchor_mode=_coerce_enum(
                config_enums.SampleStateAnchorMode,
                sample_construction_raw.get(
                    "state_anchor_mode",
                    data_defaults.sample_construction.state_anchor_mode,
                ),
            ),
            frame_stride=sample_construction_raw.get(
                "frame_stride",
                data_defaults.sample_construction.frame_stride,
            ),
            chunk_size=sample_construction_raw.get(
                "chunk_size",
                data_defaults.sample_construction.chunk_size,
            ),
            window_size=sample_construction_raw.get(
                "window_size",
                data_defaults.sample_construction.window_size,
            ),
            predict_blocks_per_sample=sample_construction_raw.get(
                "predict_blocks_per_sample",
                data_defaults.sample_construction.predict_blocks_per_sample,
            ),
            randomize_geometry=sample_construction_raw.get(
                "randomize_geometry",
                data_defaults.sample_construction.randomize_geometry,
            ),
            allow_next_after_context_random_geometry=sample_construction_raw.get(
                "allow_next_after_context_random_geometry",
                data_defaults.sample_construction.allow_next_after_context_random_geometry,
            ),
            target_alignment=sample_target_alignment,
            rollout_context_policy=_coerce_enum(
                config_enums.RolloutContextPolicy,
                sample_construction_raw.get(
                    "rollout_context_policy",
                    data_defaults.sample_construction.rollout_context_policy,
                ),
            ),
            rollout_context_frames=sample_construction_raw.get(
                "rollout_context_frames",
                data_defaults.sample_construction.rollout_context_frames,
            ),
            segment_frames=sample_construction_raw.get(
                "segment_frames",
                data_defaults.sample_construction.segment_frames,
            ),
            segment_min_frames=sample_construction_raw.get(
                "segment_min_frames",
                data_defaults.sample_construction.segment_min_frames,
            ),
            segment_max_frames=sample_construction_raw.get(
                "segment_max_frames",
                data_defaults.sample_construction.segment_max_frames,
            ),
            segment_length_stride=sample_construction_raw.get(
                "segment_length_stride",
                data_defaults.sample_construction.segment_length_stride,
            ),
            segment_locality_block_size=sample_construction_raw.get(
                "segment_locality_block_size",
                data_defaults.sample_construction.segment_locality_block_size,
            ),
            randomize_segment_length=sample_construction_raw.get(
                "randomize_segment_length",
                data_defaults.sample_construction.randomize_segment_length,
            ),
            randomize_segment_start=sample_construction_raw.get(
                "randomize_segment_start",
                data_defaults.sample_construction.randomize_segment_start,
            ),
            require_full_segment=sample_construction_raw.get(
                "require_full_segment",
                data_defaults.sample_construction.require_full_segment,
            ),
            start_padding_frames=sample_construction_raw.get(
                "start_padding_frames",
                data_defaults.sample_construction.start_padding_frames,
            ),
            condition_source_frame_offset=sample_construction_raw.get(
                "condition_source_frame_offset",
                data_defaults.sample_construction.condition_source_frame_offset,
            ),
            context_prefix_policy=_coerce_enum(
                config_enums.SegmentContextPolicy,
                sample_construction_raw.get(
                    "context_prefix_policy",
                    data_defaults.sample_construction.context_prefix_policy,
                ),
            ),
            context_prefix_frames=sample_construction_raw.get(
                "context_prefix_frames",
                data_defaults.sample_construction.context_prefix_frames,
            ),
            tail_padding_policy=_coerce_enum(
                config_enums.TailPaddingPolicy,
                sample_construction_raw.get(
                    "tail_padding_policy",
                    data_defaults.sample_construction.tail_padding_policy,
                ),
            ),
            padded_target_policy=_coerce_enum(
                config_enums.PaddedTargetPolicy,
                sample_construction_raw.get(
                    "padded_target_policy",
                    data_defaults.sample_construction.padded_target_policy,
                ),
            ),
            task_start_power=sample_construction_raw.get(
                "task_start_power",
                data_defaults.sample_construction.task_start_power,
            ),
            demo_count_power=sample_construction_raw.get(
                "demo_count_power",
                data_defaults.sample_construction.demo_count_power,
            ),
            trajectory_start_power=sample_construction_raw.get(
                "trajectory_start_power",
                data_defaults.sample_construction.trajectory_start_power,
            ),
            sample_weight_mode=_coerce_enum(
                config_enums.SampleWeightMode,
                sample_construction_raw.get(
                    "sample_weight_mode",
                    data_defaults.sample_construction.sample_weight_mode,
                ),
            ),
            sample_order_mode=_coerce_enum(
                config_enums.SampleOrderMode,
                sample_construction_raw.get(
                    "sample_order_mode",
                    data_defaults.sample_construction.sample_order_mode,
                ),
            ),
            sample_weight_length_power=sample_construction_raw.get(
                "sample_weight_length_power",
                data_defaults.sample_construction.sample_weight_length_power,
            ),
            sample_weight_min=sample_construction_raw.get(
                "sample_weight_min",
                data_defaults.sample_construction.sample_weight_min,
            ),
            sample_weight_max=sample_construction_raw.get(
                "sample_weight_max",
                data_defaults.sample_construction.sample_weight_max,
            ),
            causal_prefix_suffix_buckets=tuple(
                CausalPrefixSuffixBucketConfig(
                    observed_frames=int(bucket["observed_frames"]),
                    future_frames=int(bucket["future_frames"]),
                )
                for bucket in sample_construction_raw.get(
                    "causal_prefix_suffix_buckets",
                    tuple(
                        {
                            "observed_frames": bucket.observed_frames,
                            "future_frames": bucket.future_frames,
                        }
                        for bucket in data_defaults.sample_construction.causal_prefix_suffix_buckets
                    ),
                )
            ),
        ),
        generalist_dynamics_mixture=_load_generalist_dynamics_mixture_config(
            data_raw.get("generalist_dynamics_mixture"),
            data_defaults.generalist_dynamics_mixture,
        ),
    )
    if data_config_cls is LeRobotConsortiumDataConfig:
        common_data_kwargs.update(
            consortium_members=_load_consortium_members(data_raw.get("consortium_members")),
            channel_selection_mode=_coerce_enum(
                config_enums.ConsortiumChannelSelectionMode,
                data_raw.get("channel_selection_mode", data_defaults.channel_selection_mode),
            ),
            required_channels=tuple(data_raw.get("required_channels", data_defaults.required_channels)),
            channel_mappings=_load_consortium_channel_mappings(data_raw.get("channel_mappings")),
            view_packing_mode=_coerce_enum(
                config_enums.ConsortiumViewPackingMode,
                data_raw.get("view_packing_mode", data_defaults.view_packing_mode),
            ),
            frame_packing_order=_coerce_enum(
                config_enums.ConsortiumFramePackingOrder,
                data_raw.get("frame_packing_order", data_defaults.frame_packing_order),
            ),
            missing_channel_policy=_coerce_enum(
                config_enums.ConsortiumMissingChannelPolicy,
                data_raw.get("missing_channel_policy", data_defaults.missing_channel_policy),
            ),
            random_mode=_coerce_enum(
                config_enums.ConsortiumRandomMode,
                data_raw.get("random_mode", data_defaults.random_mode),
            ),
            weight_mode=_coerce_enum(
                config_enums.ConsortiumWeightMode,
                data_raw.get("weight_mode", data_defaults.weight_mode),
            ),
            sampling_seed=data_raw.get("sampling_seed", data_defaults.sampling_seed),
            split_mode=_coerce_enum(
                config_enums.ConsortiumSplitMode,
                data_raw.get("split_mode", data_defaults.split_mode),
            ),
            explicit_train_episodes=_load_consortium_episode_selection(data_raw.get("explicit_train_episodes")),
            explicit_val_episodes=_load_consortium_episode_selection(data_raw.get("explicit_val_episodes")),
            local_cache=ConsortiumLocalCacheConfig(
                mode=_coerce_enum(
                    config_enums.ConsortiumCacheMode,
                    (data_raw.get("local_cache", {}) or {}).get("mode", data_defaults.local_cache.mode),
                ),
                root=(data_raw.get("local_cache", {}) or {}).get("root", data_defaults.local_cache.root),
            ),
            cloud_cache=ConsortiumCloudCacheConfig(
                mode=_coerce_enum(
                    config_enums.ConsortiumCacheMode,
                    (data_raw.get("cloud_cache", {}) or {}).get("mode", data_defaults.cloud_cache.mode),
                ),
                backend=_coerce_enum(
                    config_enums.ConsortiumCloudCacheBackend,
                    (data_raw.get("cloud_cache", {}) or {}).get("backend", data_defaults.cloud_cache.backend),
                ),
                root=(data_raw.get("cloud_cache", {}) or {}).get("root", data_defaults.cloud_cache.root),
            ),
        )
    if data_config_cls is MixedVideoDataConfig:
        resize_bins = _load_mixed_video_resize_bins(data_raw.get("decode_resize_bins"))
        common_data_kwargs.update(
            video_sources=_load_mixed_video_sources(data_raw.get("video_sources")),
            latent_encoding_mode=_coerce_enum(
                config_enums.MixedVideoLatentEncodingMode,
                data_raw.get("latent_encoding_mode", data_defaults.latent_encoding_mode),
            ),
            latent_view_combinations=_load_mixed_video_view_combinations(data_raw.get("latent_view_combinations")),
            decode_size_mode=_coerce_enum(
                config_enums.MixedVideoDecodeSizeMode,
                data_raw.get("decode_size_mode", data_defaults.decode_size_mode),
            ),
            decode_resize_bins=data_defaults.decode_resize_bins if resize_bins is None else resize_bins,
            decode_height=int(data_raw.get("decode_height", data_defaults.decode_height)),
            decode_width=int(data_raw.get("decode_width", data_defaults.decode_width)),
            decode_fit_mode=_load_mixed_video_fit_mode(data_raw, data_defaults),
            decode_center_crop=bool(data_raw.get("decode_center_crop", data_defaults.decode_center_crop)),
            decode_allow_upscale=bool(data_raw.get("decode_allow_upscale", data_defaults.decode_allow_upscale)),
            target_observation_fps=(
                None
                if data_raw.get("target_observation_fps", data_defaults.target_observation_fps) is None
                else float(data_raw.get("target_observation_fps", data_defaults.target_observation_fps))
            ),
            missing_observation_fps=float(
                data_raw.get("missing_observation_fps", data_defaults.missing_observation_fps)
            ),
            missing_stream_policy=_coerce_enum(
                config_enums.MixedVideoMissingStreamPolicy,
                data_raw.get("missing_stream_policy", data_defaults.missing_stream_policy),
            ),
            random_mode=_coerce_enum(
                config_enums.MixedVideoRandomMode,
                data_raw.get("random_mode", data_defaults.random_mode),
            ),
            weight_mode=_coerce_enum(
                config_enums.MixedVideoWeightMode,
                data_raw.get("weight_mode", data_defaults.weight_mode),
            ),
            sampling_seed=data_raw.get("sampling_seed", data_defaults.sampling_seed),
        )

    # `data_config_cls` may be a benchmark-specific preset or the generic
    # fallback. In both cases, the instantiated object carries the exact view
    # layout and action/state schema that the rest of the code should trust.
    data_config = data_config_cls(
        **common_data_kwargs,
    )
    backbone_raw = raw.get("backbone", {})
    backbone_defaults = SharedVideoTransformerConfig()
    pretrained_model_name_or_path = backbone_raw.get("pretrained_model_name_or_path")
    load_wan_vae_frontend = backbone_raw.get("load_wan_vae_frontend")
    if load_wan_vae_frontend is None:
        load_wan_vae_frontend = pretrained_model_name_or_path is not None
    load_text_conditioning = backbone_raw.get("load_text_conditioning")
    if load_text_conditioning is None:
        load_text_conditioning = pretrained_model_name_or_path is not None
    load_reference_core_weights = backbone_raw.get("load_reference_core_weights")
    if load_reference_core_weights is None:
        load_reference_core_weights = False

    backbone_config = SharedVideoTransformerConfig(
        input_channels=backbone_raw.get("input_channels", backbone_defaults.input_channels),
        latent_channels=backbone_raw.get("latent_channels", backbone_defaults.latent_channels),
        latent_stride=backbone_raw.get("latent_stride", backbone_defaults.latent_stride),
        patch_size_t=backbone_raw.get("patch_size_t", backbone_defaults.patch_size_t),
        patch_size_h=backbone_raw.get("patch_size_h", backbone_defaults.patch_size_h),
        patch_size_w=backbone_raw.get("patch_size_w", backbone_defaults.patch_size_w),
        implementation=normalize_backbone_implementation(
            backbone_raw.get("implementation", backbone_defaults.implementation)
        ),
        hidden_size=backbone_raw.get("hidden_size", backbone_defaults.hidden_size),
        num_layers=backbone_raw.get("num_layers", backbone_defaults.num_layers),
        num_heads=backbone_raw.get("num_heads", backbone_defaults.num_heads),
        attention_head_dim=backbone_raw.get("attention_head_dim"),
        mlp_ratio=backbone_raw.get("mlp_ratio", backbone_defaults.mlp_ratio),
        ffn_dim=backbone_raw.get("ffn_dim"),
        text_dim=backbone_raw.get("text_dim", backbone_defaults.text_dim),
        freq_dim=backbone_raw.get("freq_dim", backbone_defaults.freq_dim),
        cross_attn_norm=backbone_raw.get("cross_attn_norm", backbone_defaults.cross_attn_norm),
        rope_max_seq_len=backbone_raw.get("rope_max_seq_len", backbone_defaults.rope_max_seq_len),
        latent_norm_eps=backbone_raw.get("latent_norm_eps", backbone_defaults.latent_norm_eps),
        attn_mode=_coerce_enum(config_enums.AttentionMode, backbone_raw.get("attn_mode", backbone_defaults.attn_mode)),
        train_attn_mode=_coerce_optional_enum(
            config_enums.AttentionMode,
            backbone_raw.get("train_attn_mode", backbone_defaults.train_attn_mode),
        ),
        infer_attn_mode=_coerce_optional_enum(
            config_enums.AttentionMode,
            backbone_raw.get("infer_attn_mode", backbone_defaults.infer_attn_mode),
        ),
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        transformer_subdir=backbone_raw.get("transformer_subdir", backbone_defaults.transformer_subdir),
        vae_subdir=backbone_raw.get("vae_subdir", backbone_defaults.vae_subdir),
        text_encoder_subdir=backbone_raw.get("text_encoder_subdir", backbone_defaults.text_encoder_subdir),
        tokenizer_subdir=backbone_raw.get("tokenizer_subdir", backbone_defaults.tokenizer_subdir),
        max_text_tokens=backbone_raw.get("max_text_tokens", backbone_defaults.max_text_tokens),
        load_wan_vae_frontend=load_wan_vae_frontend,
        load_text_conditioning=load_text_conditioning,
        load_reference_core_weights=load_reference_core_weights,
        reference_core_init_mode=_coerce_enum(
            config_enums.ReferenceCoreInitMode,
            backbone_raw.get("reference_core_init_mode", backbone_defaults.reference_core_init_mode),
        ),
        reference_norm2_source_path=backbone_raw.get(
            "reference_norm2_source_path",
            backbone_defaults.reference_norm2_source_path,
        ),
        exported_runtime_action_init_mode=_coerce_enum(
            config_enums.ExportedRuntimeActionInitMode,
            backbone_raw.get(
                "exported_runtime_action_init_mode",
                backbone_defaults.exported_runtime_action_init_mode,
            ),
        ),
        reference_assets_device_policy=_coerce_enum(
            config_enums.ReferenceAssetsDevicePolicy,
            backbone_raw.get(
                "reference_assets_device_policy",
                backbone_defaults.reference_assets_device_policy,
            ),
        ),
        reference_model_path=backbone_raw.get("reference_model_path"),
    )

    training_raw = raw.get("training", {})
    training_config = TrainingConfig(
        video_num_train_timesteps=training_raw.get("video_num_train_timesteps", 1000),
        action_num_train_timesteps=training_raw.get("action_num_train_timesteps", 1000),
        video_sigma_shift=training_raw.get("video_sigma_shift", 5.0),
        action_sigma_shift=training_raw.get("action_sigma_shift", 1.0),
        use_teacher_forcing=training_raw.get("use_teacher_forcing", False),
        chunk_size=training_raw.get("chunk_size", 2),
        window_size=training_raw.get("window_size", 8),
        optimizer_name=_coerce_enum(
            config_enums.OptimizerName,
            training_raw.get("optimizer_name", "adamw"),
        ),
        scheduler_name=_coerce_enum(
            config_enums.SchedulerName,
            training_raw.get("scheduler_name", "constant"),
        ),
        learning_rate=training_raw.get("learning_rate", 1e-4),
        beta1=training_raw.get("beta1", 0.9),
        beta2=training_raw.get("beta2", 0.999),
        weight_decay=training_raw.get("weight_decay", 0.0),
        warmup_steps=training_raw.get("warmup_steps", 0),
        gradient_accumulation_steps=training_raw.get("gradient_accumulation_steps", 1),
        max_grad_norm=training_raw.get("max_grad_norm"),
        num_steps=training_raw.get("num_steps"),
        text_condition_dropout_prob=training_raw.get("text_condition_dropout_prob", 0.0),
        enabled_objectives=tuple(training_raw.get("enabled_objectives", ("action", "latent"))),
        latent_loss_weight=training_raw.get("latent_loss_weight", 1.0),
        action_loss_weight=training_raw.get("action_loss_weight", 1.0),
        sample_loss_weight_mode=_coerce_enum(
            config_enums.SampleLossWeightMode,
            training_raw.get("sample_loss_weight_mode", "none"),
        ),
        sample_loss_weight_reference_steps=training_raw.get("sample_loss_weight_reference_steps"),
        sample_loss_weight_min=training_raw.get("sample_loss_weight_min"),
        sample_loss_weight_max=training_raw.get("sample_loss_weight_max"),
        trainable_components=tuple(training_raw.get("trainable_components", ("all",))),
        frozen_components=tuple(training_raw.get("frozen_components", ())),
    )

    inference_raw = raw.get("inference", {})
    joint_cfg_application = _coerce_optional_enum(
        config_enums.JointCfgApplication,
        inference_raw.get("joint_cfg_application"),
    )
    if joint_cfg_application == config_enums.JointCfgApplication.VIDEO_ONLY:
        video_cfg_mode = config_enums.CFGMode.GUIDED
        action_cfg_mode = config_enums.CFGMode.CONDITIONED
    elif joint_cfg_application == config_enums.JointCfgApplication.JOINT:
        video_cfg_mode = config_enums.CFGMode.GUIDED
        action_cfg_mode = config_enums.CFGMode.GUIDED
    else:
        video_cfg_mode = _coerce_enum(
            config_enums.CFGMode,
            inference_raw.get("video_cfg_mode", "guided"),
        )
        action_cfg_mode = _coerce_enum(
            config_enums.CFGMode,
            inference_raw.get("action_cfg_mode", "conditioned"),
        )

    joint_cache_warmup_source = inference_raw.get("joint_cache_warmup_source")
    if joint_cache_warmup_source == "dreamzero_reference_block":
        resolved_warmup_source = config_enums.CacheWarmupSource.REFERENCE_VIDEO
        initial_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = inference_raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = inference_raw.get("joint_cache_rollout_warmup_frames")
    elif joint_cache_warmup_source == "reference_video":
        resolved_warmup_source = config_enums.CacheWarmupSource.REFERENCE_VIDEO
        initial_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_initial_warmup_anchor", "full"),
        )
        initial_warmup_frames = inference_raw.get("joint_cache_initial_warmup_frames")
        rollout_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_rollout_warmup_anchor", "full"),
        )
        rollout_warmup_frames = inference_raw.get("joint_cache_rollout_warmup_frames")
    elif joint_cache_warmup_source in {None, "none"}:
        resolved_warmup_source = (
            config_enums.CacheWarmupSource.REFERENCE_VIDEO
            if joint_cache_warmup_source is None
            else config_enums.CacheWarmupSource.NONE
        )
        initial_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = inference_raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = inference_raw.get("joint_cache_rollout_warmup_frames")
    else:
        resolved_warmup_source = _coerce_enum(config_enums.CacheWarmupSource, joint_cache_warmup_source)
        initial_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_initial_warmup_anchor", "start"),
        )
        initial_warmup_frames = inference_raw.get("joint_cache_initial_warmup_frames", 1)
        rollout_warmup_anchor = _coerce_enum(
            config_enums.WarmupAnchor,
            inference_raw.get("joint_cache_rollout_warmup_anchor", "end"),
        )
        rollout_warmup_frames = inference_raw.get("joint_cache_rollout_warmup_frames")

    inference_config = InferenceConfig(
        video_num_inference_steps=inference_raw.get("video_num_inference_steps", 25),
        action_num_inference_steps=inference_raw.get("action_num_inference_steps", 50),
        joint_num_inference_steps=inference_raw.get("joint_num_inference_steps"),
        joint_sampler=_coerce_enum(
            config_enums.JointSampler,
            inference_raw.get("joint_sampler", "unipc"),
        ),
        video_cfg_mode=video_cfg_mode,
        action_cfg_mode=action_cfg_mode,
        joint_cfg_application=joint_cfg_application,
        joint_cache_update_mode=_coerce_enum(
            config_enums.CacheUpdateMode,
            inference_raw.get("joint_cache_update_mode", "warmup_only"),
        ),
        joint_cache_warmup_source=resolved_warmup_source,
        joint_cache_initial_warmup_anchor=initial_warmup_anchor,
        joint_cache_initial_warmup_frames=initial_warmup_frames,
        joint_cache_rollout_warmup_anchor=rollout_warmup_anchor,
        joint_cache_rollout_warmup_frames=rollout_warmup_frames,
        joint_observed_video_prefix_frames=inference_raw.get("joint_observed_video_prefix_frames", 1),
        frame_chunk_size=inference_raw.get("frame_chunk_size", 2),
        use_cache=inference_raw.get("use_cache", True),
        guidance_scale=inference_raw.get("guidance_scale", 1.0),
        action_guidance_scale=inference_raw.get("action_guidance_scale", 1.0),
        video_exec_step=inference_raw.get("video_exec_step", -1),
        joint_dynamic_cache_schedule=inference_raw.get("joint_dynamic_cache_schedule", False),
        joint_num_dit_steps=inference_raw.get("joint_num_dit_steps", 8),
        joint_dit_step_mask=(
            tuple(bool(value) for value in inference_raw["joint_dit_step_mask"])
            if inference_raw.get("joint_dit_step_mask") is not None
            else None
        ),
        joint_enable_prediction_reuse=inference_raw.get("joint_enable_prediction_reuse", False),
        joint_prediction_reuse_thresholds=tuple(
            float(value) for value in inference_raw.get("joint_prediction_reuse_thresholds", (0.95, 0.93))
        ),
        joint_prediction_reuse_countdowns=tuple(
            int(value) for value in inference_raw.get("joint_prediction_reuse_countdowns", (4, 2))
        ),
    )

    # Legacy configs may still provide an `action_head` block. The current
    # runtime no longer instantiates a separate head stack, but we still read
    # those fields here to derive the equivalent variant/decoder defaults.
    action_head_raw = raw.get("action_head", {})
    policy_variant_config = _load_policy_variant_config(
        policy_variant_raw=raw.get("policy_variant", {}),
        action_head_raw=action_head_raw,
        data_config=data_config,
        backbone_config=backbone_config,
        training_config=training_config,
        inference_config=inference_config,
    )
    _validate_cross_config_contracts(
        data_config=data_config,
        policy_variant_config=policy_variant_config,
    )
    action_decoder_config = _load_action_decoder_config(
        action_decoder_raw=raw.get("action_decoder", {}),
        policy_variant_config=policy_variant_config,
        data_config=data_config,
        backbone_config=backbone_config,
    )

    trainer_raw = raw.get("trainer", {})
    trainer_config = TrainerConfig(
        max_epochs=trainer_raw.get("max_epochs", 1),
        limit_train_batches=trainer_raw.get("limit_train_batches", 2),
        limit_val_batches=trainer_raw.get("limit_val_batches", 1),
        validation_interval=trainer_raw.get("validation_interval"),
        log_every_n_steps=trainer_raw.get("log_every_n_steps", 1),
        accelerator=_coerce_enum(
            config_enums.TrainerAccelerator,
            trainer_raw.get("accelerator", "cpu"),
        ),
        devices=trainer_raw.get("devices", 1),
        precision=_coerce_enum(
            config_enums.TrainerPrecision,
            trainer_raw.get("precision", "32-true"),
        ),
        enable_checkpointing=trainer_raw.get("enable_checkpointing", False),
        enable_model_summary=trainer_raw.get("enable_model_summary", False),
        runtime=_coerce_enum(
            config_enums.TrainerRuntimeName,
            trainer_raw.get("runtime", "lightning"),
        ),
        batch_adapter=_coerce_enum(
            config_enums.BatchAdapterName,
            trainer_raw.get("batch_adapter", "views"),
        ),
        loop_policy=_coerce_enum(
            config_enums.LoopPolicyName,
            trainer_raw.get("loop_policy", "epochs"),
        ),
        strategy=_coerce_enum(
            config_enums.StrategyName,
            trainer_raw.get("strategy", "lightning"),
        ),
        default_root_dir=trainer_raw.get("default_root_dir"),
        checkpoint_dir=trainer_raw.get("checkpoint_dir"),
        save_interval=trainer_raw.get("save_interval"),
        checkpoint_mode=_coerce_enum(
            config_enums.CheckpointMode,
            trainer_raw.get("checkpoint_mode", "full_training_state"),
        ),
        max_checkpoints_to_keep=trainer_raw.get("max_checkpoints_to_keep"),
        export_runtime_backbone=trainer_raw.get("export_runtime_backbone", False),
        resume_from=trainer_raw.get("resume_from"),
        enable_jsonl_logging=trainer_raw.get("enable_jsonl_logging", False),
        metrics_filename=trainer_raw.get("metrics_filename", "metrics.jsonl"),
        enable_wandb=trainer_raw.get("enable_wandb", False),
        wandb_project=trainer_raw.get("wandb_project"),
        wandb_entity=trainer_raw.get("wandb_entity"),
        wandb_mode=_coerce_enum(
            config_enums.WandBMode,
            trainer_raw.get("wandb_mode", "disabled"),
        ),
        run_name=trainer_raw.get("run_name"),
    )
    validation_config = _load_validation_config(raw.get("validation", {}))
    _validate_mixed_video_wan_causal_buckets(
        data_config=data_config,
        backbone_config=backbone_config,
        policy_variant_config=policy_variant_config,
        trainer_config=trainer_config,
    )

    return apply_parallel_sequence_contract(ExperimentConfig(
        name=raw.get("name", "unnamed_experiment"),
        data=data_config,
        backbone=backbone_config,
        policy_variant=policy_variant_config,
        action_decoder=action_decoder_config,
        training=training_config,
        inference=inference_config,
        trainer=trainer_config,
        validation=validation_config,
    ))
