from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from open_wam.configs import ExperimentConfig
from open_wam.utils import apply_config_overrides
from open_wam.utils.config_loader import apply_parallel_sequence_contract


class MoTTrainingProfile(StrEnum):
    """Ground-truth M5 training geometry supplied by the original trainer."""

    RANDOM_SEGMENT = "random_segment"
    FULL_SEGMENT_W64 = "full_segment_w64"


class GJDTrainingMode(StrEnum):
    JOINT = "joint"
    FDM = "action_conditioned_video"
    IDM = "video_conditioned_action"


@dataclass(frozen=True)
class MoTMethodSpec:
    asset_id: str
    config_name: str
    coupling: str
    gjd_ablation: str | None = None
    mode_token: bool = False

    @property
    def is_gjd(self) -> bool:
        return self.gjd_ablation is not None


@dataclass(frozen=True)
class TrainingScenario:
    scenario_id: str
    fixture_id: str
    mode: GJDTrainingMode | None = None
    source: str = "real_demo"


@dataclass(frozen=True)
class InferenceScenario:
    scenario_id: str
    mode: GJDTrainingMode | None = None


@dataclass(frozen=True)
class MoTInferenceContract:
    frontend_encode_mode: str = "lingbot_streaming_vae"
    inference_window_size: int = 30
    startup_model_obs_frames: int = 1
    startup_env_init_steps: int = 5
    model_frame_chunk_size: int = 4
    action_per_frame: int = 4
    action_horizon: int = 16
    execute_action_steps: int = 16
    max_timestep: int = 800
    max_chunks: int = 50


NON_GJD_METHODS: tuple[MoTMethodSpec, ...] = (
    MoTMethodSpec(
        asset_id="mot_video_then_action",
        config_name="mot_libero_latent_local_video_then_action_heng_compatible",
        coupling="video_then_action",
    ),
    MoTMethodSpec(
        asset_id="mot_action_then_video",
        config_name="mot_libero_latent_local_action_then_video_heng_compatible",
        coupling="action_then_video",
    ),
    MoTMethodSpec(
        asset_id="mot_joint",
        config_name="mot_libero_latent_local_joint_heng_compatible",
        coupling="joint",
    ),
    MoTMethodSpec(
        asset_id="mot_decoupled_same_step",
        config_name="mot_libero_latent_local_decoupled_same_step_heng_compatible",
        coupling="decoupled_same_step",
    ),
    MoTMethodSpec(
        asset_id="mot_video_noisy_to_action",
        config_name="mot_libero_latent_local_video_noisy_to_action_heng_compatible",
        coupling="video_noisy_to_action",
    ),
    MoTMethodSpec(
        asset_id="mot_action_noisy_to_video",
        config_name="mot_libero_latent_local_action_noisy_to_video_heng_compatible",
        coupling="action_noisy_to_video",
    ),
)

GJD_METHODS: tuple[MoTMethodSpec, ...] = (
    MoTMethodSpec(
        asset_id="gjd_vanilla",
        config_name="mot_libero_latent_local_generalist_joint_denoising_heng_compatible",
        coupling="joint",
        gjd_ablation="vanilla",
    ),
    MoTMethodSpec(
        asset_id="gjd_pure_joint",
        config_name="mot_libero_latent_local_generalist_joint_denoising_heng_compatible",
        coupling="joint",
        gjd_ablation="pure_joint",
    ),
    MoTMethodSpec(
        asset_id="gjd_mode_token",
        config_name="mot_libero_latent_local_generalist_joint_denoising_heng_compatible",
        coupling="joint",
        gjd_ablation="mode_token",
        mode_token=True,
    ),
)

ALL_METHODS: tuple[MoTMethodSpec, ...] = NON_GJD_METHODS + GJD_METHODS
METHOD_BY_ASSET_ID = {method.asset_id: method for method in ALL_METHODS}
# No vanilla checkpoint exists for the current mixed-dynamics contract, and the
# current private asset set intentionally does not treat historical VNA/ANV
# checkpoints as strict numerical baselines. All three remain in static
# syntax/semantics coverage.
EXACT_CHECKPOINT_METHODS: tuple[MoTMethodSpec, ...] = tuple(
    method
    for method in ALL_METHODS
    if method.asset_id
    not in {
        "mot_video_noisy_to_action",
        "mot_action_noisy_to_video",
        "gjd_vanilla",
    }
)

# Shared-runtime sentinels deliberately avoid multiplying expensive
# infrastructure gates across every policy program.
CACHE_ROLLOVER_ASSET_IDS = ("mot_joint", "gjd_mode_token")
FULL_STATE_RESUME_ASSET_ID = "gjd_mode_token"

NON_GJD_TRAINING_SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario("random_segment", "mot_random_segment"),
    TrainingScenario("full_segment_w64", "mot_full_segment_w64"),
)

GJD_CONDITIONAL_TRAINING_SCENARIOS: tuple[TrainingScenario, ...] = (
    TrainingScenario("real_joint", "gjd_real_joint", GJDTrainingMode.JOINT),
    TrainingScenario("real_fdm", "gjd_real_fdm", GJDTrainingMode.FDM),
    TrainingScenario("real_idm", "gjd_real_idm", GJDTrainingMode.IDM),
    TrainingScenario(
        "counterfactual_fdm",
        "gjd_counterfactual_fdm",
        GJDTrainingMode.FDM,
        source="counterfactual_dynamics",
    ),
    TrainingScenario(
        "counterfactual_idm",
        "gjd_counterfactual_idm",
        GJDTrainingMode.IDM,
        source="counterfactual_dynamics",
    ),
)

GJD_INFERENCE_SCENARIOS: tuple[InferenceScenario, ...] = (
    InferenceScenario("joint", GJDTrainingMode.JOINT),
    InferenceScenario("fdm", GJDTrainingMode.FDM),
    InferenceScenario("idm", GJDTrainingMode.IDM),
)

DEFAULT_INFERENCE_CONTRACT = MoTInferenceContract()
GJD_INFERENCE_CONTRACT = MoTInferenceContract(max_timestep=1500, max_chunks=100)
_UNRESOLVED_ENVIRONMENT_VARIABLE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})")


def training_scenarios_for(method: MoTMethodSpec) -> tuple[TrainingScenario, ...]:
    if not method.is_gjd:
        return NON_GJD_TRAINING_SCENARIOS
    if method.gjd_ablation == "pure_joint":
        return GJD_CONDITIONAL_TRAINING_SCENARIOS[:1]
    return GJD_CONDITIONAL_TRAINING_SCENARIOS


def inference_scenarios_for(method: MoTMethodSpec) -> tuple[InferenceScenario, ...]:
    if not method.is_gjd:
        return (InferenceScenario("default"),)
    return GJD_INFERENCE_SCENARIOS


def ground_truth_training_overrides(profile: MoTTrainingProfile) -> dict[str, Any]:
    """Return the expanded non-GJD command contract without launcher behavior."""

    overrides: dict[str, Any] = {
        "data.replay_status_policy": "include_all",
        "data.val_replay_status_policy": None,
        "data.require_replay_status": False,
        "data.val_require_replay_status": False,
        "policy_variant.parallel_sequence_contract": (
            "legacy_prefix_single_frame_perchunk_proprio"
        ),
        "policy_variant.noisy_video_condition_prob": 0.5,
        "data.sample_construction.mode": "uniform_segment",
        "data.sample_construction.sample_order_mode": "replacement",
        "data.sample_construction.randomize_geometry": True,
        "data.sample_construction.segment_locality_block_size": 1,
        "data.sample_construction.require_full_segment": True,
        "data.sample_construction.task_start_power": 0.0,
        "data.sample_construction.demo_count_power": 0.0,
        "data.sample_construction.trajectory_start_power": 0.0,
        "data.sample_construction.sample_weight_mode": "uniform",
        "training.sample_loss_weight_mode": "none",
        "trainer.checkpoint_mode": "model_only",
        "trainer.save_interval": 500,
        "trainer.max_checkpoints_to_keep": 3,
    }
    if profile == MoTTrainingProfile.RANDOM_SEGMENT:
        overrides.update(
            {
                "data.sample_construction.segment_min_frames": 64,
                "data.sample_construction.segment_max_frames": 256,
                "data.sample_construction.segment_length_stride": 4,
                "data.sample_construction.randomize_segment_length": True,
                "data.sample_construction.randomize_segment_start": True,
                "training.num_steps": 5000,
            }
        )
        return overrides
    if profile == MoTTrainingProfile.FULL_SEGMENT_W64:
        overrides.update(
            {
                "data.sample_construction.segment_min_frames": 1000,
                "data.sample_construction.segment_max_frames": 1000,
                "data.sample_construction.segment_length_stride": 1,
                "data.sample_construction.window_size": 64,
                "data.sample_construction.randomize_segment_length": False,
                "data.sample_construction.randomize_segment_start": False,
                "training.window_size": 64,
                "training.num_steps": 10000,
            }
        )
        return overrides
    raise ValueError(f"Unsupported MoT training profile {profile!r}.")


def apply_ground_truth_training_profile(
    config: ExperimentConfig,
    profile: MoTTrainingProfile,
) -> ExperimentConfig:
    overrides = ground_truth_training_overrides(profile)
    config = apply_config_overrides(config, overrides)
    return apply_parallel_sequence_contract(
        config,
        explicit_override_keys=set(overrides),
    )


def gjd_ablation_overrides(method: MoTMethodSpec) -> dict[str, Any]:
    if not method.is_gjd:
        raise ValueError(f"Method {method.asset_id!r} is not a GJD ablation.")
    vanilla_probs = {
        "joint": 0.6,
        "action_conditioned_video": 0.2,
        "video_conditioned_action": 0.2,
    }
    if method.gjd_ablation == "vanilla":
        return {
            "policy_variant.mot_generalist_training_mode_probs": vanilla_probs,
            "policy_variant.generalist_mode_text_token": False,
        }
    if method.gjd_ablation == "pure_joint":
        return {
            "policy_variant.mot_generalist_training_mode_probs": {
                "joint": 1.0,
                "action_conditioned_video": 0.0,
                "video_conditioned_action": 0.0,
            },
            "policy_variant.generalist_mode_text_token": False,
            "policy_variant.generalist_training_paradigm": "demo_only",
            "data.sample_construction.sample_order_mode": "replacement",
            "data.generalist_dynamics_mixture.train_latent_root": None,
            "data.generalist_dynamics_mixture.val_latent_root": None,
        }
    if method.gjd_ablation == "mode_token":
        return {
            "policy_variant.mot_generalist_training_mode_probs": vanilla_probs,
            "policy_variant.generalist_mode_text_token": True,
        }
    raise ValueError(f"Unsupported GJD ablation {method.gjd_ablation!r}.")


def apply_gjd_ablation(
    config: ExperimentConfig,
    method: MoTMethodSpec,
) -> ExperimentConfig:
    return apply_config_overrides(config, gjd_ablation_overrides(method))


@dataclass(frozen=True)
class CharacterizationAssets:
    dataset_root: Path
    base_model_root: Path
    video_transformer_root: Path
    empty_text_embedding: Path
    counterfactual_train_root: Path | None
    counterfactual_val_root: Path | None
    checkpoints: dict[str, CheckpointAsset]

    def checkpoint_for(self, asset_id: str) -> Path:
        try:
            return self.checkpoints[asset_id].model_state
        except KeyError as exc:
            raise KeyError(
                f"Asset manifest is missing checkpoint {asset_id!r}."
            ) from exc

    def checkpoint_config_for(self, asset_id: str) -> Path:
        try:
            return self.checkpoints[asset_id].resolved_config
        except KeyError as exc:
            raise KeyError(
                f"Asset manifest is missing checkpoint {asset_id!r}."
            ) from exc


@dataclass(frozen=True)
class CheckpointAsset:
    """Model weights and the resolved training contract that produced them."""

    model_state: Path
    resolved_config: Path
    accepted_origin_mismatch_fields: tuple[str, ...] = ()


def load_characterization_assets(path: str | Path) -> CharacterizationAssets:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected a YAML mapping in {manifest_path}.")
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError(
            f"Unsupported characterization asset schema in {manifest_path}."
        )

    paths = raw.get("paths")
    checkpoints = raw.get("checkpoints")
    if not isinstance(paths, Mapping) or not isinstance(checkpoints, Mapping):
        raise TypeError(
            "Characterization assets require `paths` and `checkpoints` mappings."
        )

    def required_path(key: str) -> Path:
        value = paths.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Characterization assets require paths.{key}.")
        return _expanded_path(value, base=manifest_path.parent)

    def optional_path(key: str) -> Path | None:
        value = paths.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"Expected paths.{key} to be a non-empty path or null.")
        return _expanded_path(value, base=manifest_path.parent)

    resolved_checkpoints: dict[str, CheckpointAsset] = {}
    for asset_id, value in checkpoints.items():
        if asset_id not in METHOD_BY_ASSET_ID:
            raise ValueError(f"Unknown characterization checkpoint id {asset_id!r}.")
        model_value: Any = value
        resolved_config_value: Any = None
        accepted_origin_mismatch_fields: tuple[str, ...] = ()
        if isinstance(value, Mapping):
            model_value = value.get("model_state")
            resolved_config_value = value.get("resolved_config")
            raw_accepted_fields = value.get("accepted_origin_mismatch_fields", ())
            if not isinstance(raw_accepted_fields, (tuple, list)) or not all(
                isinstance(field, str) and field for field in raw_accepted_fields
            ):
                raise ValueError(
                    f"Expected checkpoints.{asset_id}."
                    "accepted_origin_mismatch_fields to be a list of dotted "
                    "config-field names."
                )
            accepted_origin_mismatch_fields = tuple(dict.fromkeys(raw_accepted_fields))
        if not isinstance(model_value, str) or not model_value:
            raise ValueError(
                f"Expected checkpoint {asset_id!r} to be a path or a mapping "
                "with `model_state`."
            )
        model_state = resolve_model_checkpoint(
            _expanded_path(model_value, base=manifest_path.parent)
        )
        explicit_resolved_config = None
        if resolved_config_value is not None:
            if not isinstance(resolved_config_value, str) or not resolved_config_value:
                raise ValueError(
                    f"Expected checkpoints.{asset_id}.resolved_config to be a "
                    "non-empty path."
                )
            explicit_resolved_config = _expanded_path(
                resolved_config_value,
                base=manifest_path.parent,
            )
        resolved_checkpoints[str(asset_id)] = CheckpointAsset(
            model_state=model_state,
            resolved_config=resolve_checkpoint_config(
                model_state,
                explicit=explicit_resolved_config,
            ),
            accepted_origin_mismatch_fields=accepted_origin_mismatch_fields,
        )

    return CharacterizationAssets(
        dataset_root=required_path("dataset_root"),
        base_model_root=required_path("base_model_root"),
        video_transformer_root=required_path("video_transformer_root"),
        empty_text_embedding=required_path("empty_text_embedding"),
        counterfactual_train_root=optional_path("counterfactual_train_root"),
        counterfactual_val_root=optional_path("counterfactual_val_root"),
        checkpoints=resolved_checkpoints,
    )


def resolve_model_checkpoint(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    model_state = candidate / "model_state.pt"
    if model_state.is_file():
        return model_state
    raise FileNotFoundError(f"Expected model_state.pt at {candidate}.")


def resolve_checkpoint_config(
    model_state: str | Path,
    *,
    explicit: str | Path | None = None,
) -> Path:
    """Resolve the saved config without inferring semantics from tensor keys."""

    candidate = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else Path(model_state).expanduser().resolve().parent / "resolved_config.yaml"
    )
    if candidate.is_dir():
        candidate = candidate / "resolved_config.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "Checkpoint characterization requires the resolved training config. "
        f"Expected resolved_config.yaml at {candidate}; provide "
        "`resolved_config` explicitly in the asset manifest when weights were "
        "copied separately."
    )


def _expanded_path(value: str, *, base: Path) -> Path:
    expanded_value = os.path.expandvars(os.path.expanduser(value))
    unresolved = _UNRESOLVED_ENVIRONMENT_VARIABLE.search(expanded_value)
    if unresolved is not None:
        raise ValueError(
            "Characterization asset path contains an unresolved environment "
            f"variable {unresolved.group(0)!r}: {value!r}."
        )
    expanded = Path(expanded_value)
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve()
