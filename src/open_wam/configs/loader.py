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
from .checkpoint_compatibility import apply_checkpoint_runtime_compat
from .config_paths import resolve_experiment_config_reference
from .data_contracts import DataConfig
from .data_mixed_video import MixedVideoDataConfig
from .data_parsing import parse_data_config
from .experiment import EXPERIMENT_CONFIG_SCHEMA_VERSION, ExperimentConfig
from .inference import parse_inference_config
from .local_paths import read_yaml_with_local_paths
from .policy_compatibility import normalize_video_action_config_fields
from .policy_contracts import CausalVideoPredictionPolicyConfig, PolicyVariantConfig
from .policy_parsing import parse_policy_variant_config
from .resolution import resolve_experiment_config
from .sequence_contracts import (
    expand_video_action_sequence_contract,
    validate_experiment_config_runtime_contract,
    validate_policy_data_sequence_contract,
    validate_video_action_sequence_contract_override_keys,
)
from .trainer import TrainerConfig, parse_trainer_config
from .training import parse_training_config
from .validation import parse_validation_config

_SEQUENCE_CONTRACT_COMPATIBILITY_EXPORTS = (
    validate_experiment_config_runtime_contract,
    validate_video_action_sequence_contract_override_keys,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    return read_yaml_with_local_paths(resolve_experiment_config_reference(path))


def _resolve_config_schema(
    raw: dict[str, Any],
    *,
    checkpoint_runtime_compat: bool,
) -> dict[str, Any]:
    """Resolve current configs and explicitly identified checkpoint metadata."""

    declared = raw.get("schema_version")
    if declared is None:
        if not checkpoint_runtime_compat:
            return raw
        migrated = apply_checkpoint_runtime_compat(raw)
        migrated["schema_version"] = EXPERIMENT_CONFIG_SCHEMA_VERSION
        return migrated
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise TypeError("`schema_version` must be an integer.")
    version = declared
    if version == EXPERIMENT_CONFIG_SCHEMA_VERSION:
        return raw
    if version in (0, 1):
        if not checkpoint_runtime_compat:
            raise ValueError(
                f"Experiment config schema_version {version} requires migration; "
                "load it through the explicit checkpoint compatibility path."
            )
        migrated = apply_checkpoint_runtime_compat(raw)
        migrated["schema_version"] = EXPERIMENT_CONFIG_SCHEMA_VERSION
        return migrated
    raise ValueError(
        f"Unsupported experiment config schema_version {version}; expected "
        f"{EXPERIMENT_CONFIG_SCHEMA_VERSION}."
    )


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
    if (
        sample_construction.mode
        != config_enums.WindowSamplingMode.CAUSAL_PREFIX_SUFFIX
    ):
        return

    invalid_buckets: list[str] = []
    max_raw_span = int(sample_construction.num_frames)
    for index, bucket in enumerate(
        sample_construction.effective_causal_prefix_suffix_buckets
    ):
        raw_observed_frames = int(bucket.observed_frames)
        raw_future_frames = int(bucket.future_frames)
        raw_total_frames = raw_observed_frames + raw_future_frames
        if raw_total_frames > max_raw_span:
            invalid_buckets.append(
                f"#{index} observed_frames={raw_observed_frames} "
                f"future_frames={raw_future_frames} exceeds "
                f"sample_construction.num_frames={max_raw_span}"
            )
            continue
        try:
            VideoFrameMapping.wan_causal_prefix_suffix(
                raw_observed_frames=raw_observed_frames,
                raw_future_frames=raw_future_frames,
            )
        except ValueError:
            invalid_buckets.append(
                f"#{index} observed_frames={raw_observed_frames} "
                f"future_frames={raw_future_frames} maps to zero future Wan "
                "latent targets"
            )
    if invalid_buckets:
        formatted = "\n".join(f"- {item}" for item in invalid_buckets)
        raise ValueError(
            "Mixed-video causal buckets with the Wan VAE frontend must produce "
            "at least one future latent target. Wan fresh-clip encoding maps raw "
            "frames as frame 0 plus complete 4-frame groups; choose buckets such "
            "as observed_frames=1, future_frames=4 instead of 1+3.\n"
            f"{formatted}"
        )


def load_experiment_config(
    path: str | Path,
    *,
    checkpoint_runtime_compat: bool = False,
) -> ExperimentConfig:
    """Load one root experiment YAML into the typed config boundary."""

    raw = _resolve_config_schema(
        _read_yaml(path),
        checkpoint_runtime_compat=checkpoint_runtime_compat,
    )
    raw = normalize_video_action_config_fields(raw)
    raw = expand_video_action_sequence_contract(raw)
    data_config = parse_data_config(raw.get("data", {}))
    backbone_config = parse_shared_video_transformer_config(raw.get("backbone", {}))
    training_config = parse_training_config(raw.get("training", {}))
    inference_config = parse_inference_config(raw.get("inference", {}))
    policy_variant_config = parse_policy_variant_config(
        policy_variant_raw=raw.get("policy_variant", {}),
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
    )
    trainer_config = parse_trainer_config(raw.get("trainer", {}))
    validation_config = parse_validation_config(raw.get("validation", {}))
    _validate_mixed_video_wan_causal_buckets(
        data_config=data_config,
        backbone_config=backbone_config,
        policy_variant_config=policy_variant_config,
        trainer_config=trainer_config,
    )

    return validate_experiment_config_runtime_contract(
        resolve_experiment_config(
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
    )
