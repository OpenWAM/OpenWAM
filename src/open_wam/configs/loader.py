from __future__ import annotations

from pathlib import Path
from typing import Any

from open_wam.contracts import VideoFrameMapping

from . import enums as config_enums
from .action_decoder import parse_action_decoder_config
from .backbone import (
    SharedVideoTransformerConfig,
    parse_shared_video_transformer_config,
)
from .coercion import raw_enum_value as _raw_enum_value
from .data_contracts import DataConfig
from .data_mixed_video import MixedVideoDataConfig
from .data_parsing import parse_data_config
from .experiment import ExperimentConfig
from .inference import parse_inference_config
from .local_paths import read_yaml_with_local_paths
from .policy_compatibility import normalize_video_action_config_fields
from .policy_contracts import CausalVideoPredictionPolicyConfig, PolicyVariantConfig
from .policy_parsing import parse_policy_variant_config
from .sequence_contracts import (
    apply_video_action_sequence_contract,
    expand_video_action_sequence_contract,
    validate_experiment_config_runtime_contract,
    validate_policy_data_sequence_contract,
    validate_video_action_sequence_contract_override_keys,
)
from .trainer import TrainerConfig, parse_trainer_config
from .training import parse_training_config
from .validation import parse_validation_config

# These names were historically importable from this module.
apply_parallel_sequence_contract = apply_video_action_sequence_contract
validate_parallel_sequence_contract_override_keys = (
    validate_video_action_sequence_contract_override_keys
)

_SEQUENCE_CONTRACT_COMPATIBILITY_EXPORTS = (
    validate_experiment_config_runtime_contract,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    return read_yaml_with_local_paths(path)


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


def load_experiment_config(path: str | Path, *, checkpoint_runtime_compat: bool = False) -> ExperimentConfig:
    """Load one root experiment YAML into the typed config boundary."""

    raw = _read_yaml(path)
    if checkpoint_runtime_compat:
        raw = _apply_checkpoint_runtime_compat(raw)
    raw = normalize_video_action_config_fields(raw)
    raw = expand_video_action_sequence_contract(raw)
    data_config = parse_data_config(raw.get("data", {}))
    backbone_config = parse_shared_video_transformer_config(raw.get("backbone", {}))

    training_config = parse_training_config(raw.get("training", {}))

    inference_config = parse_inference_config(raw.get("inference", {}))

    # Legacy configs may still provide an `action_head` block. The current
    # runtime no longer instantiates a separate head stack, but we still read
    # those fields here to derive the equivalent variant/decoder defaults.
    action_head_raw = raw.get("action_head", {})
    policy_variant_config = parse_policy_variant_config(
        policy_variant_raw=raw.get("policy_variant", {}),
        action_head_raw=action_head_raw,
        data_config=data_config,
        backbone_config=backbone_config,
        training_config=training_config,
        inference_config=inference_config,
    )
    validate_policy_data_sequence_contract(
        data_config=data_config,
        policy_variant_config=policy_variant_config,
    )
    action_decoder_config = parse_action_decoder_config(
        action_decoder_raw=raw.get("action_decoder", {}),
        policy_variant_config=policy_variant_config,
        data_config=data_config,
        backbone_config=backbone_config,
    )

    trainer_config = parse_trainer_config(raw.get("trainer", {}))
    validation_config = parse_validation_config(raw.get("validation", {}))
    _validate_mixed_video_wan_causal_buckets(
        data_config=data_config,
        backbone_config=backbone_config,
        policy_variant_config=policy_variant_config,
        trainer_config=trainer_config,
    )

    return apply_video_action_sequence_contract(
        ExperimentConfig(
            name=raw.get("name", "unnamed_experiment"),
            data=data_config,
            backbone=backbone_config,
            policy_variant=policy_variant_config,
            action_decoder=action_decoder_config,
            training=training_config,
            inference=inference_config,
            trainer=trainer_config,
            validation=validation_config,
        )
    )
