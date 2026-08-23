from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

import open_wam.configs.loader as config_loader_module
from open_wam.configs import (
    EXPERIMENT_CONFIG_SCHEMA_VERSION,
    ActionDecoderName,
    ActionNormalizationMode,
    ActionTargetRepresentation,
    AnchorPolicy,
    AttentionMode,
    AuxiliaryValidationSource,
    BatchAdapterName,
    CausalPrefixSuffixBucketConfig,
    CausalVideoPredictionPolicyConfig,
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    DeprecatedPolicyConfigFieldWarning,
    DualExpertPolicyConfig,
    DualExpertPreset,
    DynamicsObjective,
    DynamicsSource,
    ExportedRuntimeActionInitMode,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    LatentWindowProfile,
    LingbotParallelActionDecoderConfig,
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoFrameFitMode,
    MixedVideoLatentEncodingMode,
    MixedVideoMissingStreamPolicy,
    MixedVideoRandomMode,
    MixedVideoSourceFormat,
    MixedVideoWeightMode,
    PaddedTargetPolicy,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    ProprioContextMode,
    ReplayStatusPolicy,
    RolloutContextPolicy,
    SampleConstructionConfig,
    SampleLossWeightMode,
    SampleOrderMode,
    SampleTargetAlignment,
    SampleWeightMode,
    SegmentContextPolicy,
    TailPaddingPolicy,
    TrainingComponentSelector,
    VideoActionProgram,
    VideoActionSequenceContract,
    WindowSamplingMode,
    load_experiment_config,
    normalize_video_action_policy_fields,
    read_yaml_with_local_paths,
)
from open_wam.models.policy_variants.parallel_stream.variant import (
    ParallelStreamPolicyVariant,
)
from open_wam.utils.config_overrides import apply_config_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]

PARALLEL_STREAM_PROGRAM_CONFIG_NAMES = (
    "parallel_stream_libero_action_noisy_to_video.yaml",
    "parallel_stream_libero_action_then_video.yaml",
    "parallel_stream_libero_decoupled_same_step.yaml",
    "parallel_stream_libero_generalist_joint_denoising.yaml",
    "parallel_stream_libero_joint.yaml",
    "parallel_stream_libero_video_noisy_to_action.yaml",
    "parallel_stream_libero_video_then_action.yaml",
)


def _instantiate_parallel_stream_variant(config) -> ParallelStreamPolicyVariant:
    assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
    return ParallelStreamPolicyVariant(
        config.policy_variant,
        config.backbone,
        config.training,
        config.inference,
        action_dim=config.action_decoder.action_dim,
        action_horizon=config.action_decoder.action_horizon,
        num_frames=config.data.num_frames,
    )


def test_legacy_parallel_decoder_name_normalizes_to_canonical(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["name"] = "legacy_parallel_decoder_name"
    raw["action_decoder"]["name"] = "lingbot_parallel_decoder"
    config_path = tmp_path / "legacy_parallel_decoder_name.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(
        DeprecatedPolicyConfigFieldWarning,
        match="parallel_stream_decoder",
    ):
        config = load_experiment_config(config_path)

    assert config.action_decoder.name == ActionDecoderName.PARALLEL_STREAM


def test_legacy_parallel_decoder_python_config_normalizes_to_canonical() -> None:
    config = LingbotParallelActionDecoderConfig(
        name=ActionDecoderName.LINGBOT_PARALLEL,
        hidden_size=8,
        action_dim=7,
        action_horizon=4,
    )

    assert config.name == ActionDecoderName.PARALLEL_STREAM


@pytest.mark.parametrize(
    "field_name",
    [
        "couple_action_to_video_timesteps",
        "parallel_sequence_contract",
        "preserve_video_pretrain_history",
    ],
)
def test_authored_policy_semantic_aliases_are_rejected(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"policy_variant.{field_name}.*retired"):
        normalize_video_action_policy_fields({field_name: None}, warn=False)


@pytest.mark.parametrize(
    ("field_name", "canonical_name", "legacy_value"),
    [
        ("couple_action_to_video_timesteps", "joint_timestep_coupling", False),
        (
            "parallel_sequence_contract",
            "sequence_contract",
            "legacy_prefix_single_frame_perchunk_proprio",
        ),
    ],
)
def test_authored_config_loader_rejects_semantic_aliases(
    tmp_path: Path,
    field_name: str,
    canonical_name: str,
    legacy_value: object,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"].pop(canonical_name)
    raw["policy_variant"][field_name] = legacy_value
    config_path = tmp_path / "authored.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=f"policy_variant.{field_name}.*retired"):
        load_experiment_config(config_path)


@pytest.mark.parametrize(
    ("legacy_value", "expected"),
    [
        (True, JointTimestepCoupling.MATCH_SIGMA),
        (False, JointTimestepCoupling.INDEPENDENT),
    ],
)
def test_checkpoint_loader_migrates_boolean_timestep_coupling(
    tmp_path: Path,
    legacy_value: bool,
    expected: JointTimestepCoupling,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"].pop("joint_timestep_coupling")
    raw["policy_variant"]["couple_action_to_video_timesteps"] = legacy_value
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(
        DeprecatedPolicyConfigFieldWarning,
        match="couple_action_to_video_timesteps",
    ):
        config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert config.policy_variant.joint_timestep_coupling is expected


def test_current_checkpoint_schema_bypasses_historical_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["schema_version"] = EXPERIMENT_CONFIG_SCHEMA_VERSION
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        config_loader_module,
        "apply_checkpoint_runtime_compat",
        lambda _: pytest.fail("current checkpoint config must not be migrated"),
    )

    config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert config.name == raw["name"]


def test_resolved_config_filename_does_not_select_checkpoint_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        config_loader_module,
        "apply_checkpoint_runtime_compat",
        lambda _: pytest.fail("filename must not select checkpoint migration"),
    )

    config = load_experiment_config(config_path)

    assert config.name


@pytest.mark.parametrize("schema_version", [True, "1", 1.0])
def test_config_schema_version_requires_an_integer(
    tmp_path: Path,
    schema_version: object,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["schema_version"] = schema_version
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TypeError, match="schema_version.*integer"):
        load_experiment_config(config_path)


def test_checkpoint_loader_migrates_sequence_contract_alias(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    sequence_contract = raw["policy_variant"].pop("sequence_contract")
    raw["policy_variant"]["parallel_sequence_contract"] = sequence_contract
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(
        DeprecatedPolicyConfigFieldWarning,
        match="parallel_sequence_contract",
    ):
        config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert config.policy_variant.sequence_contract.value == sequence_contract


def test_checkpoint_loader_migrates_history_visibility_alias(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["sequence_contract"] = "default"
    raw["policy_variant"]["preserve_video_pretrain_history"] = True
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(
        DeprecatedPolicyConfigFieldWarning,
        match="preserve_video_pretrain_history",
    ):
        config = load_experiment_config(
            config_path,
            checkpoint_runtime_compat=True,
        )

    assert (
        config.policy_variant.history_stream_visibility
        == HistoryStreamVisibility.VIDEO_QUERIES_VIDEO_ONLY
    )


def test_checkpoint_history_alias_defers_to_sequence_contract(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["preserve_video_pretrain_history"] = True
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(
        DeprecatedPolicyConfigFieldWarning,
        match="preserve_video_pretrain_history",
    ):
        config = load_experiment_config(
            config_path,
            checkpoint_runtime_compat=True,
        )

    assert (
        config.policy_variant.history_stream_visibility
        == HistoryStreamVisibility.VIDEO_ONLY
    )


def test_checkpoint_history_alias_defers_to_explicit_canonical_visibility(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["sequence_contract"] = "default"
    raw["policy_variant"]["history_stream_visibility"] = "full"
    raw["policy_variant"]["preserve_video_pretrain_history"] = True
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(
        DeprecatedPolicyConfigFieldWarning,
        match="preserve_video_pretrain_history",
    ):
        config = load_experiment_config(
            config_path,
            checkpoint_runtime_compat=True,
        )

    assert (
        config.policy_variant.history_stream_visibility
        == HistoryStreamVisibility.FULL
    )


def test_checkpoint_loader_rejects_conflicting_timestep_alias(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["joint_timestep_coupling"] = "independent"
    raw["policy_variant"]["couple_action_to_video_timesteps"] = True
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Conflicting policy config fields"):
        load_experiment_config(config_path, checkpoint_runtime_compat=True)


@pytest.mark.parametrize("legacy_value", [0, 1, "true", (), {}])
def test_checkpoint_loader_rejects_non_boolean_timestep_alias(
    tmp_path: Path,
    legacy_value: object,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"].pop("joint_timestep_coupling")
    raw["policy_variant"]["couple_action_to_video_timesteps"] = legacy_value
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(TypeError, match="must be a boolean"):
        load_experiment_config(config_path, checkpoint_runtime_compat=True)


def test_policy_dataclass_has_no_legacy_timestep_constructor_route() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ParallelStreamPolicyConfig(
            program=VideoActionProgram.JOINT,
            couple_action_to_video_timesteps=True,  # type: ignore[call-arg]
        )


def test_parallel_policy_dataclass_has_no_legacy_history_constructor_route() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ParallelStreamPolicyConfig(
            program=VideoActionProgram.JOINT,
            preserve_video_pretrain_history=True,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("config_name", "expected_proprio_mode"),
    [
        (
            "parallel_stream_libero_video_then_action.yaml",
            ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
        (
            "parallel_stream_libero_action_then_video.yaml",
            ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
        (
            "parallel_stream_libero_joint.yaml",
            ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
        (
            "dual_expert_libero_video_then_action.yaml",
            ProprioContextMode.PER_CHUNK_ADDITIVE,
        ),
    ],
)
def test_raw_action_targets_do_not_imply_a_proprio_conditioning_mode(
    config_name: str,
    expected_proprio_mode: ProprioContextMode,
) -> None:
    config = load_experiment_config(REPO_ROOT / "configs" / "experiments" / config_name)

    assert config.data.action_schema.action_dim == 7
    assert config.data.action_target.representation == ActionTargetRepresentation.RAW
    assert config.data.action_target.source_key == "action"
    assert config.data.action_target.normalization.mode == ActionNormalizationMode.NONE
    assert config.data.action_target.joint_position_normalization.mode == ActionNormalizationMode.NONE
    assert (
        getattr(config.policy_variant, "proprio_context_mode", ProprioContextMode.NONE)
        == expected_proprio_mode
    )
    assert getattr(config.action_decoder, "recovered_osc_loss_weight", 0.0) == 0.0


def test_implicit_legacy_policy_config_is_rejected(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/examples/public_tiny_synthetic_contract.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw.pop("policy_variant")
    raw["action_head"] = {"name": "legacy"}
    config_path = tmp_path / "implicit_legacy_policy.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="policy_variant"):
        load_experiment_config(config_path)


def test_new_variant_yaml_configs_load() -> None:
    causal = load_experiment_config(REPO_ROOT / "configs/experiments/causal_video_prediction_mixed_video.yaml")
    dual_expert = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml")
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin.yaml")
    smoke_parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")

    assert isinstance(causal.policy_variant, CausalVideoPredictionPolicyConfig)
    assert isinstance(dual_expert.policy_variant, DualExpertPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(smoke_parallel.policy_variant, ParallelStreamPolicyConfig)
    assert dual_expert.policy_variant.preset == DualExpertPreset.FASTWAM
    assert dual_expert.policy_variant.program == VideoActionProgram.VIDEO_THEN_ACTION
    assert dual_expert.policy_variant.condition_mode == "first_frame"
    assert dual_expert.policy_variant.use_condition_latents is True
    assert dual_expert.training.trainable_components == (TrainingComponentSelector.POLICY_VARIANT_ACTION_EXPERT,)
    assert parallel.backbone.implementation == "shared_transformer"
    assert smoke_parallel.backbone.implementation == "shared_transformer"
    assert parallel.backbone.train_attn_mode == "flex"
    assert parallel.backbone.infer_attn_mode == "torch"
    assert parallel.policy_variant.runtime_mode == "lingbot_exact"
    assert parallel.action_decoder.name == "parallel_stream_decoder"
    assert parallel.backbone.hidden_size == 3072
    assert smoke_parallel.policy_variant.runtime_mode == "lingbot_exact"
    assert smoke_parallel.action_decoder.name == "parallel_stream_decoder"
    assert smoke_parallel.backbone.reference_model_path is None


def test_dynamics_routing_knob_loads_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "dynamics_routed.yaml"
    config_path.write_text(
        """
name: dynamics_routed
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  train_batch_size: 1
  val_batch_size: 1
  sample_construction:
    sample_order_mode: replacement
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  dynamics_routing:
    train_latent_root: /tmp/counterfactual_train/encoded_latents
    val_latent_root: /tmp/counterfactual_val/encoded_latents
    allow_train_latent_root_for_val: false
    routes:
      - {source: real_demo, mode: joint, weight: 0.6}
      - {source: real_demo, mode: action_conditioned_video, weight: 0.1}
      - {source: real_demo, mode: video_conditioned_action, weight: 0.1}
      - {source: counterfactual_dynamics, mode: action_conditioned_video, weight: 0.1}
      - {source: counterfactual_dynamics, mode: video_conditioned_action, weight: 0.1}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: latents
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT
    assert config.data.dynamics_routing.train_latent_root == "/tmp/counterfactual_train/encoded_latents"
    assert config.data.dynamics_routing.allow_train_latent_root_for_val is False
    routes = config.data.dynamics_routing.active_routes
    assert len(routes) == 5
    assert routes[0].source is DynamicsSource.REAL_DEMO
    assert routes[0].mode is DynamicsObjective.JOINT
    assert routes[0].weight == pytest.approx(0.6)
    assert config.data.dynamics_routing.mode_probabilities() == {
        DynamicsObjective.JOINT: pytest.approx(0.6),
        DynamicsObjective.ACTION_CONDITIONED_VIDEO: pytest.approx(0.2),
        DynamicsObjective.VIDEO_CONDITIONED_ACTION: pytest.approx(0.2),
    }


@pytest.mark.parametrize(
    "field_name",
    ["generalist_training_paradigm", "dynamics_routing_requirement"],
)
def test_authored_legacy_routing_marker_is_rejected(
    tmp_path: Path,
    field_name: str,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"][field_name] = "mixed_dynamics"
    config_path = tmp_path / "legacy_mixed_dynamics.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{field_name}.*retired"):
        load_experiment_config(config_path)


def test_checkpoint_legacy_mixed_dynamics_marker_is_consumed(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_training_paradigm"] = "mixed_dynamics"
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert not hasattr(config.policy_variant, "dynamics_routing_requirement")
    assert config.data.dynamics_routing.active_routes


def test_authored_legacy_routing_keys_are_rejected(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_training_paradigm"] = "mixed_dynamics"
    policy_path = tmp_path / "legacy_policy_key.yaml"
    policy_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="generalist_training_paradigm.*retired"):
        load_experiment_config(policy_path)

    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["generalist_dynamics_mixture"] = raw["data"].pop(
        "dynamics_routing"
    )
    data_path = tmp_path / "legacy_data_key.yaml"
    data_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="generalist_dynamics_mixture.*retired"):
        load_experiment_config(data_path)


def test_checkpoint_legacy_routing_keys_and_source_migrate_once(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    canonical_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    legacy_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    legacy_raw["policy_variant"]["generalist_training_paradigm"] = "mixed_dynamics"
    legacy_routing = legacy_raw["data"].pop("dynamics_routing")
    legacy_raw["data"]["generalist_dynamics_mixture"] = legacy_routing
    for route in legacy_routing["routes"]:
        if route["source"] == "counterfactual_dynamics":
            route["source"] = "counterfactual"

    canonical_path = tmp_path / "canonical.yaml"
    canonical_path.write_text(
        yaml.safe_dump(canonical_raw, sort_keys=False), encoding="utf-8"
    )
    checkpoint_path = tmp_path / "resolved_config.yaml"
    checkpoint_path.write_text(
        yaml.safe_dump(legacy_raw, sort_keys=False), encoding="utf-8"
    )

    canonical = load_experiment_config(canonical_path)
    migrated = load_experiment_config(
        checkpoint_path,
        checkpoint_runtime_compat=True,
    )

    assert migrated == canonical


def test_checkpoint_missing_sample_order_uses_replacement_default(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["sample_construction"].pop("sample_order_mode")
    checkpoint_path = tmp_path / "resolved_config.yaml"
    checkpoint_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    config = load_experiment_config(
        checkpoint_path,
        checkpoint_runtime_compat=True,
    )

    assert (
        config.data.sample_construction.sample_order_mode
        is SampleOrderMode.REPLACEMENT
    )


def test_dynamics_routing_accepts_replacement_sample_order(tmp_path: Path) -> None:
    config_path = tmp_path / "dynamics_routed_replacement_order.yaml"
    config_path.write_text(
        """
name: dynamics_routed_replacement_order
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  train_batch_size: 1
  val_batch_size: 1
  sample_construction:
    mode: uniform_segment
    sample_order_mode: replacement
    chunk_size: 4
    window_size: 64
    randomize_geometry: true
    sample_weight_mode: uniform
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  dynamics_routing:
    train_latent_root: /tmp/counterfactual_train/encoded_latents
    val_latent_root: /tmp/counterfactual_val/encoded_latents
    routes:
      - {source: real_demo, mode: joint, weight: 1.0}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: latents
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
    assert config.data.sample_construction.chunk_size == 4
    assert config.data.sample_construction.window_size == 64
    assert config.data.sample_construction.randomize_geometry is True
    assert config.data.sample_construction.sample_weight_mode == SampleWeightMode.UNIFORM
    assert config.data.dynamics_routing.train_latent_root == "/tmp/counterfactual_train/encoded_latents"
    assert config.data.dynamics_routing.val_latent_root == "/tmp/counterfactual_val/encoded_latents"
    assert len(config.data.dynamics_routing.active_routes) == 1


def test_dynamics_routing_rejects_epoch_order_override() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )

    with pytest.raises(ValueError, match="sample_order_mode.*replacement"):
        apply_config_overrides(
            config,
            {"data.sample_construction.sample_order_mode": "epoch_order"},
        )


def test_dynamics_routing_rejects_views_batch_adapter(tmp_path: Path) -> None:
    config_path = tmp_path / "dynamics_routed_views_adapter.yaml"
    config_path.write_text(
        """
name: dynamics_routed_views_adapter
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  train_batch_size: 1
  val_batch_size: 1
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  dynamics_routing:
    train_latent_root: /tmp/counterfactual_train/encoded_latents
    val_latent_root: /tmp/counterfactual_val/encoded_latents
    routes:
      - {source: real_demo, mode: joint, weight: 1.0}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: views
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch_adapter=latents"):
        load_experiment_config(config_path)


def test_dynamics_routing_rejects_non_uniform_sample_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "dynamics_routed_weighted_source.yaml"
    config_path.write_text(
        """
name: dynamics_routed_weighted_source
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  train_batch_size: 1
  val_batch_size: 1
  sample_construction:
    mode: uniform_segment
    sample_order_mode: replacement
    sample_weight_mode: valid_action_steps
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  dynamics_routing:
    train_latent_root: /tmp/counterfactual_train/encoded_latents
    val_latent_root: /tmp/counterfactual_val/encoded_latents
    routes:
      - {source: real_demo, mode: joint, weight: 1.0}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
action_decoder:
  name: parallel_stream_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: latents
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_weight_mode"):
        load_experiment_config(config_path)


def test_auxiliary_validation_tasks_load_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "auxiliary_validation.yaml"
    config_path.write_text(
        """
name: auxiliary_validation
data:
  dataset_name: robotwin
  train_batch_size: 1
  val_batch_size: 1
  sample_construction:
    sample_order_mode: replacement
  dynamics_routing:
    routes:
      - {source: real_demo, mode: action_conditioned_video, weight: 1.0}
      - {source: real_demo, mode: video_conditioned_action, weight: 1.0}
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: generalist_joint_denoising
action_decoder:
  name: parallel_stream_decoder
validation:
  auxiliary_tasks:
    - name: fdm_val
      mode_override: action_conditioned_video
      dataset_split: val
      source: counterfactual_dynamics_if_available
      max_batches: 16
      report_prefix: val_fdm
    - name: idm_val
      mode_override: video_conditioned_action
      source: counterfactual_dynamics_if_available
      max_batches: 8
      report_prefix: val_idm
trainer:
  accelerator: cpu
  batch_adapter: latents
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    fdm, idm = config.validation.auxiliary_tasks
    assert fdm.name == "fdm_val"
    assert fdm.mode_override == DynamicsObjective.ACTION_CONDITIONED_VIDEO
    assert fdm.source == AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS_IF_AVAILABLE
    assert fdm.phase == "val_fdm"
    assert fdm.should_drop_text is True
    assert idm.mode_override == DynamicsObjective.VIDEO_CONDITIONED_ACTION
    assert idm.max_batches == 8
    assert idm.should_drop_text is True


def test_conditional_auxiliary_validation_rejects_text_drop_override(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["validation"] = {
        "auxiliary_tasks": [
            {
                "name": "idm_val",
                "mode_override": "video_conditioned_action",
                "drop_text_conditioning": False,
            }
        ]
    }
    config_path = tmp_path / "conditional_text_override.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Conditional FDM/IDM validation always removes task text",
    ):
        load_experiment_config(config_path)


def test_strict_dynamics_auxiliary_validation_rejects_text_drop_override(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["validation"] = {
        "auxiliary_tasks": [
            {
                "name": "fdm_val",
                "source": "real_demo",
                "drop_text_conditioning": False,
            }
        ]
    }
    config_path = tmp_path / "strict_conditional_text_override.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="always removes task text"):
        load_experiment_config(config_path)


def test_conditional_auxiliary_validation_rejects_unprojected_mixed_dataset() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )

    with pytest.raises(ValueError, match="active routes contain another objective"):
        apply_config_overrides(
            config,
            {
                "validation.auxiliary_tasks": [
                    {
                        "name": "fdm_mixed_dataset",
                        "mode_override": "action_conditioned_video",
                        "source": "dataset",
                        "max_batches": 1,
                    }
                ]
            },
        )


def test_conditional_auxiliary_validation_requires_active_router() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )

    with pytest.raises(ValueError, match="requires active `data.dynamics_routing.routes`"):
        apply_config_overrides(
            config,
            {"data.dynamics_routing.routes": []},
        )


def test_absolute_joint_position_action_target_loads_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "absolute_joint_libero.yaml"
    config_path.write_text(
        """
name: absolute_joint_libero
data:
  dataset_name: libero
  action_schema:
    action_dim: 8
    action_horizon: 4
    state_dim: 7
  action_target:
    representation: absolute_joint_position
    source_key: action
    joint_position_source_key: robot0_joint_pos
    gripper_position_source_key: custom_gripper_qpos
    include_gripper: true
    gripper_representation: action_command
    gripper_action_index: -1
    joint_position_normalization:
      mode: joint_limits
      lower: [-2.0, -1.0, -3.0, -2.5, -2.0, -1.5, -1.0]
      upper: [2.0, 1.0, 3.0, 2.5, 2.0, 1.5, 1.0]
      clip_min: -1.0
      clip_max: 1.0
    normalization:
      mode: gaussian
      mean: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
      std: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  program: joint
action_decoder:
  name: parallel_stream_decoder
  action_dim: 8
  action_horizon: 4
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.data.action_target.representation == ActionTargetRepresentation.ABSOLUTE_JOINT_POSITION
    assert config.data.action_target.source_key == "action"
    assert config.data.action_target.joint_position_source_key == "robot0_joint_pos"
    assert config.data.action_target.gripper_position_source_key == "custom_gripper_qpos"
    assert config.data.action_target.joint_position_normalization.mode == ActionNormalizationMode.JOINT_LIMITS
    assert config.data.action_target.joint_position_normalization.lower[0] == -2.0
    assert config.data.action_target.normalization.mode == ActionNormalizationMode.GAUSSIAN
    assert config.data.action_target.normalization.mean[-1] == 0.7


def test_mixed_video_config_loads_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_video.yaml"
    manifest_path = tmp_path / "mixed_manifest.csv"
    config_path.write_text(
        f"""
name: mixed_video_smoke
data:
  dataset_name: mixed_video
  dataset_type: mixed_video
  video_sources:
    - source_id: bridge_video
      manifest_csv: {manifest_path}
      local_root: {tmp_path}
      latent_root: {tmp_path / "latents"}
      source_format: rgb_and_latent
      latent_key: encoded_latents
      sampling_weight: 2.0
  camera_names: [observation.images.slot0]
  latent_camera_names: [observation.images.slot0]
  latent_encoding_mode: per_view
  latent_view_combinations:
    - name: slot0_only
      slots: [observation.images.slot0]
      sampling_weight: 2.5
      source_ids: [bridge_video]
  canonical_height: 8
  canonical_width: 8
  view_layout:
    - source_name: observation.images.slot0
      canonical_name: observation.images.slot0
      top: 0
      left: 0
      height: 8
      width: 8
  num_frames: 4
  frame_stride: 1
  sample_stride: 1
  train_batch_size: 1
  val_batch_size: 1
  decode_size_mode: aspect_ratio_bins
  decode_resize_bins:
    - name: square_8
      aspect_width: 1
      aspect_height: 1
      target_height: 8
      target_width: 8
      max_pixels: 64
    - name: four_three
      aspect_width: 4
      aspect_height: 3
      target_height: 12
      target_width: 16
  decode_height: 8
  decode_width: 8
  decode_fit_mode: letterbox_pad
  target_observation_fps: 15.0
  missing_observation_fps: 30.0
  missing_stream_policy: zero_fill
  random_mode: within_source
  weight_mode: proportional_then_manual_scale
  action_schema:
    action_dim: 1
    action_horizon: 0
    state_dim: 1
    state_horizon: 0
  sample_construction:
    mode: causal_prefix_suffix
    num_frames: 4
    action_horizon: 0
    state_horizon: 0
    causal_prefix_suffix_buckets:
      - observed_frames: 1
        future_frames: 3
backbone:
  implementation: shared_transformer
policy_variant:
  name: causal_video_prediction
action_decoder:
  name: video_only_decoder
  action_dim: 1
  action_horizon: 0
trainer:
  accelerator: cpu
  batch_adapter: views
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert isinstance(config.data, MixedVideoDataConfig)
    assert config.data.video_sources[0].source_id == "bridge_video"
    assert config.data.video_sources[0].source_format == MixedVideoSourceFormat.RGB_AND_LATENT
    assert config.data.video_sources[0].latent_root == str(tmp_path / "latents")
    assert config.data.video_sources[0].latent_key == "encoded_latents"
    assert config.data.video_sources[0].sampling_weight == 2.0
    assert config.data.latent_encoding_mode == MixedVideoLatentEncodingMode.PER_VIEW
    assert config.data.latent_view_combinations[0].name == "slot0_only"
    assert config.data.latent_view_combinations[0].slots == ("observation.images.slot0",)
    assert config.data.latent_view_combinations[0].sampling_weight == 2.5
    assert config.data.latent_view_combinations[0].source_ids == ("bridge_video",)
    assert config.data.decode_size_mode == MixedVideoDecodeSizeMode.ASPECT_RATIO_BINS
    assert config.data.decode_resize_bins[0].name == "square_8"
    assert config.data.decode_resize_bins[1].target_width == 16
    assert config.data.decode_height == 8
    assert config.data.decode_fit_mode == MixedVideoFrameFitMode.LETTERBOX_PAD
    assert config.data.target_observation_fps == 15.0
    assert config.data.missing_observation_fps == 30.0
    assert config.data.missing_stream_policy == MixedVideoMissingStreamPolicy.ZERO_FILL
    assert config.data.random_mode == MixedVideoRandomMode.WITHIN_SOURCE
    assert config.data.weight_mode == MixedVideoWeightMode.PROPORTIONAL_THEN_MANUAL_SCALE
    assert config.trainer.batch_adapter == BatchAdapterName.VIEWS


def test_mixed_video_wan_causal_buckets_reject_zero_future_latents(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_video_bad_wan_bucket.yaml"
    manifest_path = tmp_path / "mixed_manifest.csv"
    config_path.write_text(
        f"""
name: mixed_video_bad_wan_bucket
data:
  dataset_name: mixed_video
  dataset_type: mixed_video
  video_sources:
    - source_id: bridge_video
      manifest_csv: {manifest_path}
      local_root: {tmp_path}
  camera_names: [observation.images.slot0]
  latent_camera_names: [observation.images.slot0]
  canonical_height: 8
  canonical_width: 8
  view_layout:
    - source_name: observation.images.slot0
      canonical_name: observation.images.slot0
      top: 0
      left: 0
      height: 8
      width: 8
  num_frames: 4
  train_batch_size: 1
  val_batch_size: 1
  action_schema:
    action_dim: 1
    action_horizon: 0
    state_dim: 1
    state_horizon: 0
  sample_construction:
    mode: causal_prefix_suffix
    num_frames: 4
    action_horizon: 0
    state_horizon: 0
    causal_prefix_suffix_buckets:
      - observed_frames: 1
        future_frames: 3
backbone:
  implementation: shared_transformer
  load_wan_vae_frontend: true
policy_variant:
  name: causal_video_prediction
action_decoder:
  name: video_only_decoder
  action_dim: 1
  action_horizon: 0
trainer:
  accelerator: cpu
  batch_adapter: views
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one future latent target"):
        load_experiment_config(config_path)


def test_mixed_video_wan_causal_bucket_fallback_is_validated(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_video_bad_implicit_wan_bucket.yaml"
    manifest_path = tmp_path / "mixed_manifest.csv"
    config_path.write_text(
        f"""
name: mixed_video_bad_implicit_wan_bucket
data:
  dataset_name: mixed_video
  dataset_type: mixed_video
  video_sources:
    - source_id: bridge_video
      manifest_csv: {manifest_path}
      local_root: {tmp_path}
  camera_names: [observation.images.slot0]
  latent_camera_names: [observation.images.slot0]
  canonical_height: 8
  canonical_width: 8
  view_layout:
    - source_name: observation.images.slot0
      canonical_name: observation.images.slot0
      top: 0
      left: 0
      height: 8
      width: 8
  num_frames: 4
  train_batch_size: 1
  val_batch_size: 1
  action_schema:
    action_dim: 1
    action_horizon: 0
    state_dim: 1
    state_horizon: 0
  sample_construction:
    mode: causal_prefix_suffix
    num_frames: 4
    action_horizon: 0
    state_horizon: 0
backbone:
  implementation: shared_transformer
  load_wan_vae_frontend: true
policy_variant:
  name: causal_video_prediction
action_decoder:
  name: video_only_decoder
  action_dim: 1
  action_horizon: 0
trainer:
  accelerator: cpu
  batch_adapter: views
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="observed_frames=2 future_frames=2"):
        load_experiment_config(config_path)


def test_mixed_video_aspect_ratio_bins_reject_multi_sample_batches(tmp_path: Path) -> None:
    manifest_path = tmp_path / "mixed_manifest.csv"
    config_path = tmp_path / "mixed_video_bad_batch.yaml"
    config_path.write_text(
        f"""
name: mixed_video_bad_batch
data:
  dataset_name: mixed_video
  dataset_type: mixed_video
  video_sources:
    - source_id: bridge_video
      manifest_csv: {manifest_path}
      local_root: {tmp_path}
  camera_names: [observation.images.slot0]
  latent_camera_names: [observation.images.slot0]
  train_batch_size: 2
  val_batch_size: 1
  decode_size_mode: aspect_ratio_bins
  action_schema:
    action_dim: 1
    action_horizon: 0
    state_dim: 1
    state_horizon: 0
  sample_construction:
    mode: causal_prefix_suffix
    num_frames: 16
    action_horizon: 0
    state_horizon: 0
backbone:
  implementation: shared_transformer
policy_variant:
  name: causal_video_prediction
action_decoder:
  name: video_only_decoder
  action_dim: 1
  action_horizon: 0
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="train_batch_size=1"):
        load_experiment_config(config_path)


def test_raw_libero_smoke_variant_yaml_configs_load() -> None:
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_raw_smoke.yaml")
    parallel_action_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_action_conditioned_smoke.yaml"
    )

    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(parallel_action_conditioned.policy_variant, ParallelStreamPolicyConfig)

    assert parallel.data.dataset_name == "libero"
    assert parallel_action_conditioned.data.dataset_name == "libero"

    assert parallel.data.action_schema.action_horizon == 16
    assert parallel.action_decoder.name == ActionDecoderName.PARALLEL_STREAM
    assert parallel_action_conditioned.policy_variant.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    assert parallel_action_conditioned.policy_variant.video_condition_on_action is True
    assert parallel_action_conditioned.policy_variant.video_action_condition_source == "noisy_action"
    assert parallel_action_conditioned.policy_variant.video_action_attention_scope == "block_local"

def test_dual_expert_policy_yaml_config_loads(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "dual_expert_robotwin"
    raw["policy_variant"]["video_prefix_frames"] = 1
    raw["policy_variant"]["num_action_layers"] = 4
    raw["policy_variant"]["action_hidden_size"] = 768
    raw["policy_variant"]["action_ffn_dim"] = 1024
    raw["policy_variant"]["use_state_conditioning"] = True
    raw["policy_variant"]["proprio_context_mode"] = "per_chunk_additive"

    config_path = tmp_path / "dual_expert_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.name == "dual_expert"
    assert config.policy_variant.program == VideoActionProgram.VIDEO_THEN_ACTION
    assert config.policy_variant.video_prefix_frames == 1
    assert config.policy_variant.num_action_layers == 4
    assert config.policy_variant.action_hidden_size == 768
    assert config.policy_variant.action_ffn_dim == 1024
    assert config.policy_variant.use_state_conditioning is True
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert config.action_decoder.name == ActionDecoderName.DUAL_EXPERT


def test_checkpoint_loader_migrates_explicit_dual_expert_coupling_to_program(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"].pop("program")
    raw["policy_variant"]["runtime_mode"] = "non_joint_two_stream"
    raw["policy_variant"]["current_block_coupling"] = "action_then_video"
    raw["policy_variant"]["video_can_attend_action"] = False
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert config.policy_variant.program == VideoActionProgram.ACTION_THEN_VIDEO
    assert (
        config.policy_variant.current_block_coupling
        == CurrentBlockCoupling.ACTION_THEN_VIDEO
    )
    assert not hasattr(config.policy_variant, "runtime_mode")
    assert not hasattr(config.policy_variant, "video_can_attend_action")


def test_checkpoint_loader_preserves_generalist_program_during_legacy_migration(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"].pop("program")
    raw["policy_variant"]["runtime_mode"] = "non_joint_two_stream"
    raw["policy_variant"]["current_block_coupling"] = "joint"
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert (
        config.policy_variant.program == VideoActionProgram.GENERALIST_JOINT_DENOISING
    )


def test_checkpoint_loader_consumes_derived_parallel_execution_fields(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    canonical_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    checkpoint_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    checkpoint_raw["policy_variant"].update(
        {
            "runtime_mode": "lingbot_exact_action_conditioned",
            "current_block_coupling": "joint",
            "variant_profile": "generalist_joint_denoising",
            "video_condition_on_action": True,
        }
    )
    canonical_path = tmp_path / "canonical.yaml"
    canonical_path.write_text(
        yaml.safe_dump(canonical_raw, sort_keys=False),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "resolved_config.yaml"
    checkpoint_path.write_text(
        yaml.safe_dump(checkpoint_raw, sort_keys=False),
        encoding="utf-8",
    )

    canonical = load_experiment_config(canonical_path)
    migrated = load_experiment_config(
        checkpoint_path,
        checkpoint_runtime_compat=True,
    )

    assert migrated == canonical


def test_checkpoint_loader_uses_explicit_routes_over_stale_gjd_routing_marker(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    canonical_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    stale_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    stale_policy = stale_raw["policy_variant"]
    stale_policy["name"] = "mot"
    stale_policy.pop("program")
    stale_policy["runtime_mode"] = "non_joint_two_stream"
    stale_policy["current_block_coupling"] = "joint"
    stale_policy["video_can_attend_action"] = False
    stale_policy["mot_generalist_training_mode_probs"] = {
        "joint": 0.6,
        "action_conditioned_video": 0.2,
        "video_conditioned_action": 0.2,
    }
    stale_policy["generalist_training_paradigm"] = "demo_only"

    canonical_path = tmp_path / "canonical.yaml"
    canonical_path.write_text(
        yaml.safe_dump(canonical_raw, sort_keys=False), encoding="utf-8"
    )
    stale_path = tmp_path / "resolved_config.yaml"
    stale_path.write_text(yaml.safe_dump(stale_raw, sort_keys=False), encoding="utf-8")

    canonical = load_experiment_config(canonical_path)
    with pytest.warns(Warning) as emitted:
        migrated = load_experiment_config(
            stale_path,
            checkpoint_runtime_compat=True,
        )

    assert {warning.category for warning in emitted} == {
        DeprecatedPolicyConfigFieldWarning,
        UserWarning,
    }
    assert migrated == canonical


def test_checkpoint_loader_migrates_legacy_source_weights_only_for_mixed_data(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    canonical_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    legacy_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    legacy_policy = legacy_raw["policy_variant"]
    legacy_policy["name"] = "mot"
    legacy_policy.pop("program")
    legacy_policy.update(
        {
            "runtime_mode": "non_joint_two_stream",
            "current_block_coupling": "joint",
            "video_can_attend_action": False,
            "mot_generalist_training_mode_probs": {
                "joint": 0.6,
                "action_conditioned_video": 0.2,
                "video_conditioned_action": 0.2,
            },
            "generalist_training_paradigm": "mixed_dynamics",
        }
    )
    legacy_routing = legacy_raw["data"]["dynamics_routing"]
    legacy_routing.pop("routes")
    legacy_routing.update(
        {
            "real_joint_weight": 0.6,
            "real_action_conditioned_video_weight": 0.1,
            "real_video_conditioned_action_weight": 0.1,
            "counterfactual_action_conditioned_video_weight": 0.1,
            "counterfactual_video_conditioned_action_weight": 0.1,
        }
    )
    canonical_path = tmp_path / "canonical.yaml"
    canonical_path.write_text(
        yaml.safe_dump(canonical_raw, sort_keys=False), encoding="utf-8"
    )
    legacy_path = tmp_path / "resolved_config.yaml"
    legacy_path.write_text(
        yaml.safe_dump(legacy_raw, sort_keys=False), encoding="utf-8"
    )

    canonical = load_experiment_config(canonical_path)
    with pytest.warns(DeprecatedPolicyConfigFieldWarning):
        migrated = load_experiment_config(
            legacy_path,
            checkpoint_runtime_compat=True,
        )

    assert migrated == canonical


def test_checkpoint_loader_normalizes_routed_epoch_order_sampling(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["sample_construction"]["sample_order_mode"] = "epoch_order"
    expected_routes = raw["data"]["dynamics_routing"]["routes"]
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(UserWarning, match="routed checkpoint sampling.*replacement"):
        config = load_experiment_config(
            config_path,
            checkpoint_runtime_compat=True,
        )

    assert (
        config.data.sample_construction.sample_order_mode
        == SampleOrderMode.REPLACEMENT
    )
    assert [
        {
            "source": route.source.value,
            "mode": route.mode.value,
            "weight": route.weight,
        }
        for route in config.data.dynamics_routing.routes
    ] == expected_routes


def test_checkpoint_loader_preserves_epoch_order_without_active_routes(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["sample_construction"]["sample_order_mode"] = "epoch_order"
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(
        config_path,
        checkpoint_runtime_compat=True,
    )

    assert config.data.sample_construction.sample_order_mode == SampleOrderMode.EPOCH_ORDER


def test_checkpoint_loader_ignores_inert_legacy_weights_for_standard_program(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_video_then_action.yaml"
    canonical_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    legacy_raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    legacy_policy = legacy_raw["policy_variant"]
    legacy_policy["name"] = "mot"
    legacy_policy.pop("program")
    legacy_policy.update(
        {
            "runtime_mode": "non_joint_two_stream",
            "current_block_coupling": "video_then_action",
            "video_can_attend_action": False,
            "mot_generalist_training_mode_probs": None,
            "generalist_training_paradigm": "demo_only",
        }
    )
    legacy_raw["data"]["generalist_dynamics_mixture"] = {
        "train_latent_root": None,
        "val_latent_root": None,
        "allow_train_latent_root_for_val": False,
        "real_joint_weight": 0.6,
        "real_action_conditioned_video_weight": 0.1,
        "real_video_conditioned_action_weight": 0.1,
        "counterfactual_action_conditioned_video_weight": 0.1,
        "counterfactual_video_conditioned_action_weight": 0.1,
        "conditional_history_frames": 16,
        "seed": 0,
        "length_multiplier": 1.0,
    }
    canonical_path = tmp_path / "canonical.yaml"
    canonical_path.write_text(
        yaml.safe_dump(canonical_raw, sort_keys=False), encoding="utf-8"
    )
    legacy_path = tmp_path / "resolved_config.yaml"
    legacy_path.write_text(
        yaml.safe_dump(legacy_raw, sort_keys=False), encoding="utf-8"
    )

    canonical = load_experiment_config(canonical_path)
    with pytest.warns(DeprecatedPolicyConfigFieldWarning):
        migrated = load_experiment_config(
            legacy_path,
            checkpoint_runtime_compat=True,
        )

    assert migrated == canonical


def test_checkpoint_loader_maps_demo_only_conditional_modes_to_real_routes(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    policy = raw["policy_variant"]
    policy["generalist_training_paradigm"] = "demo_only"
    policy["generalist_denoising_mode_probs"] = {
        "joint": 0.6,
        "action_conditioned_video": 0.2,
        "video_conditioned_action": 0.2,
    }
    routing = raw["data"]["dynamics_routing"]
    routing.pop("routes")
    routing.update(
        {
            "real_joint_weight": 0.6,
            "real_action_conditioned_video_weight": 0.1,
            "real_video_conditioned_action_weight": 0.1,
            "counterfactual_action_conditioned_video_weight": 0.1,
            "counterfactual_video_conditioned_action_weight": 0.1,
        }
    )
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert [
        (route.source.value, route.mode.value, route.weight)
        for route in config.data.dynamics_routing.routes
    ] == [
        ("real_demo", "joint", 0.6),
        ("real_demo", "action_conditioned_video", 0.2),
        ("real_demo", "video_conditioned_action", 0.2),
    ]


def test_checkpoint_loader_rejects_enabled_marker_without_routes(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_training_paradigm"] = "mixed_dynamics"
    raw["data"]["dynamics_routing"]["routes"] = []
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="no positive source/objective route"):
        load_experiment_config(config_path, checkpoint_runtime_compat=True)


def test_checkpoint_loader_rejects_strict_program_legacy_gjd_weights(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_training_paradigm"] = "mixed_dynamics"
    routing = raw["data"]["dynamics_routing"]
    routing.pop("routes")
    routing.update(
        {
            "real_action_conditioned_video_weight": 1.0,
            "real_video_conditioned_action_weight": 1.0,
        }
    )
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Strict forward- and inverse-dynamics"):
        load_experiment_config(config_path, checkpoint_runtime_compat=True)


def test_checkpoint_loader_does_not_rewrite_unknown_gjd_routing_metadata(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_training_paradigm"] = "unsupported"
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        load_experiment_config(config_path, checkpoint_runtime_compat=True)


def test_authored_gjd_config_rejects_retired_demo_only_marker(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_training_paradigm"] = "demo_only"
    config_path = tmp_path / "authored.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="generalist_training_paradigm.*retired"):
        load_experiment_config(config_path)


def test_checkpoint_loader_preserves_pure_joint_demo_only_routing(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_training_paradigm"] = "demo_only"
    raw["policy_variant"]["generalist_denoising_mode_probs"] = {
        "joint": 1.0,
        "action_conditioned_video": 0.0,
        "video_conditioned_action": 0.0,
    }
    routing = raw["data"]["dynamics_routing"]
    routing.pop("routes")
    routing.update(
        {
            "real_joint_weight": 0.6,
            "real_action_conditioned_video_weight": 0.1,
            "real_video_conditioned_action_weight": 0.1,
            "counterfactual_action_conditioned_video_weight": 0.1,
            "counterfactual_video_conditioned_action_weight": 0.1,
        }
    )
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(UserWarning, match="no active dynamics routes"):
        config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert config.data.dynamics_routing.routes == ()
    assert config.policy_variant.program == VideoActionProgram.GENERALIST_JOINT_DENOISING
    assert not any(task.enabled for task in config.validation.auxiliary_tasks)


def test_checkpoint_loader_does_not_misclassify_parallel_joint_probability_defaults(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    policy = raw["policy_variant"]
    policy.pop("program")
    policy.update(
        {
            "runtime_mode": "lingbot_exact_action_conditioned",
            "current_block_coupling": "joint",
            "variant_profile": "standard",
            "video_condition_on_action": True,
            "joint_denoise_training_mode_probs": {
                "joint": 1.0,
                "action_conditioned_video": 0.0,
                "video_conditioned_action": 0.0,
            },
            "generalist_training_paradigm": "demo_only",
        }
    )
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert config.policy_variant.program == VideoActionProgram.JOINT
    assert config.data.dynamics_routing.routes == ()


def test_checkpoint_loader_uses_parallel_profile_for_pure_joint_gjd(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["dynamics_routing"]["routes"] = []
    policy = raw["policy_variant"]
    policy.pop("program")
    policy.update(
        {
            "runtime_mode": "lingbot_exact_action_conditioned",
            "current_block_coupling": "joint",
            "variant_profile": "generalist_joint_denoising",
            "video_condition_on_action": True,
            "joint_denoise_training_mode_probs": {
                "joint": 1.0,
                "action_conditioned_video": 0.0,
                "video_conditioned_action": 0.0,
            },
            "generalist_training_paradigm": "demo_only",
        }
    )
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.warns(UserWarning, match="no active dynamics routes"):
        config = load_experiment_config(config_path, checkpoint_runtime_compat=True)

    assert config.policy_variant.program == VideoActionProgram.GENERALIST_JOINT_DENOISING
    assert config.data.dynamics_routing.routes == ()
    assert not any(task.enabled for task in config.validation.auxiliary_tasks)


def test_checkpoint_loader_rejects_routed_metadata_for_explicit_standard_program(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["generalist_denoising_mode_probs"] = {
        "joint": 0.6,
        "action_conditioned_video": 0.2,
        "video_conditioned_action": 0.2,
    }
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicts with its explicit policy program"):
        load_experiment_config(config_path, checkpoint_runtime_compat=True)


def test_checkpoint_loader_rejects_runtime_only_dual_expert_metadata(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"].pop("program")
    raw["policy_variant"]["runtime_mode"] = "non_joint_two_stream"
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError, match="does not uniquely define attention semantics"
    ):
        load_experiment_config(config_path, checkpoint_runtime_compat=True)


def test_dual_expert_joint_policy_allows_shared_video_schedule(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "dual_expert_shared_video_schedule_robotwin"
    raw["policy_variant"]["program"] = "joint"
    raw["policy_variant"]["joint_timestep_coupling"] = "shared_video_schedule"

    config_path = tmp_path / "dual_expert_shared_video_schedule_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE


def test_dual_expert_policy_preset_applies_fastwam_joint_defaults(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "dual_expert_fastwam_joint_robotwin"
    raw["policy_variant"]["preset"] = "fastwam_joint"

    config_path = tmp_path / "dual_expert_fastwam_joint_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.preset == DualExpertPreset.FASTWAM_JOINT
    assert config.policy_variant.condition_mode == "full_video"
    assert config.action_decoder.name == ActionDecoderName.DUAL_EXPERT
    assert config.policy_variant.teacher_forcing_video_noise_prob == 0.0
    assert config.policy_variant.video_prefix_frames == 1


def test_dual_expert_policy_preset_allows_explicit_overrides(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_robotwin_smoke.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "dual_expert_fastwam_override_robotwin"
    raw["policy_variant"]["preset"] = "fastwam"
    raw["policy_variant"]["condition_mode"] = "teacher_forcing_cond_video"
    raw["policy_variant"]["teacher_forcing_video_noise_prob"] = 0.2
    raw["policy_variant"]["video_prefix_frames"] = 3

    config_path = tmp_path / "dual_expert_fastwam_override_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.preset == DualExpertPreset.FASTWAM
    assert config.policy_variant.condition_mode == "teacher_forcing_cond_video"
    assert config.policy_variant.teacher_forcing_video_noise_prob == 0.2
    assert config.policy_variant.video_prefix_frames == 3


def test_parallel_stream_video_then_action_libero_yaml_config_loads() -> None:
    lingbot_libero = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )

    assert isinstance(lingbot_libero.policy_variant, ParallelStreamPolicyConfig)
    assert lingbot_libero.data.dataset_type == "lerobot_v2_latent_local"
    assert Path(lingbot_libero.data.local_root).name == "libero_10"
    assert lingbot_libero.data.empty_text_embedding_path.endswith("empty_emb.pt")
    assert lingbot_libero.data.camera_names == (
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    )
    assert lingbot_libero.data.latent_camera_names == (
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    )
    assert lingbot_libero.data.action_target.source_key == "action"
    assert lingbot_libero.data.action_target.pose_source_key == "observation.state"
    assert lingbot_libero.data.latent_window_profile == LatentWindowProfile.EXACT_CHUNKED_WINDOW
    assert lingbot_libero.data.replay_status_policy == ReplayStatusPolicy.INCLUDE_ALL
    assert lingbot_libero.data.val_replay_status_policy is None
    assert lingbot_libero.data.require_replay_status is False
    assert lingbot_libero.data.val_require_replay_status is False
    assert lingbot_libero.training.learning_rate == 1e-5
    assert lingbot_libero.training.gradient_accumulation_steps == 10
    assert lingbot_libero.training.num_steps == 10000
    assert lingbot_libero.training.enabled_objectives == ("latent", "action")
    assert lingbot_libero.training.action_loss_weight == 1.0
    assert lingbot_libero.training.trainable_components == ("visual_tower.runtime_backbone",)
    assert lingbot_libero.training.sample_loss_weight_mode == SampleLossWeightMode.NONE
    assert lingbot_libero.trainer.runtime == "composable"
    assert lingbot_libero.trainer.batch_adapter == "latents"
    assert lingbot_libero.trainer.loop_policy == "steps"
    assert lingbot_libero.trainer.strategy == "fsdp"
    assert lingbot_libero.trainer.save_interval == 100
    assert lingbot_libero.trainer.enable_wandb is True
    assert lingbot_libero.trainer.wandb_project == "openwam-parallel-stream-libero"
def test_m1_non_generalist_configs_instantiate_reference_variant() -> None:
    config_paths = [
        REPO_ROOT / "configs/experiments" / name
        for name in PARALLEL_STREAM_PROGRAM_CONFIG_NAMES
    ]
    config_paths = [path for path in config_paths if "generalist_joint_denoising" not in path.name]
    assert len(config_paths) == 6

    for config_path in config_paths:
        config = load_experiment_config(config_path)
        assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        assert (
            config.policy_variant.program
            != VideoActionProgram.GENERALIST_JOINT_DENOISING
        )
        _instantiate_parallel_stream_variant(config)


def test_m1_generalist_joint_denoising_keeps_guidance_scale_strict() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    config = replace(config, inference=replace(config.inference, guidance_scale=1.0))

    with pytest.raises(ValueError, match="guidance_scale"):
        _instantiate_parallel_stream_variant(config)


def test_parallel_stream_generalist_joint_denoising_yaml_config_loads() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )

    assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
    assert config.policy_variant.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    assert (
        config.policy_variant.program
        == VideoActionProgram.GENERALIST_JOINT_DENOISING
    )
    assert config.policy_variant.current_block_coupling == "joint"
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT
    assert config.policy_variant.attn_window == 30
    assert config.policy_variant.sequence_contract == (
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.context_condition_latent_source == (
        ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.mode == WindowSamplingMode.UNIFORM_SEGMENT
    assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
    assert config.data.sample_construction.window_size == 64
    assert config.data.sample_construction.segment_min_frames == 1000
    assert config.data.sample_construction.segment_max_frames == 1000
    assert config.data.sample_construction.require_full_segment is True
    assert config.data.sample_construction.condition_source_frame_offset == -1
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.LEGACY
    assert config.data.dynamics_routing.train_latent_root is not None
    assert config.data.dynamics_routing.val_latent_root is not None
    assert config.training.window_size == 64
    assert config.training.sample_loss_weight_mode == SampleLossWeightMode.NONE
    probs = config.data.dynamics_routing.mode_probabilities()
    assert probs[DynamicsObjective.JOINT] == pytest.approx(0.6)
    assert probs[DynamicsObjective.ACTION_CONDITIONED_VIDEO] == pytest.approx(0.2)
    assert probs[DynamicsObjective.VIDEO_CONDITIONED_ACTION] == pytest.approx(0.2)
    assert config.action_decoder.name == ActionDecoderName.PARALLEL_STREAM


def test_m5_generalist_joint_denoising_defaults_independent_joint_coupling(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"].pop("joint_timestep_coupling", None)

    config_path = tmp_path / "m5_generalist_default_coupling.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT


@pytest.mark.parametrize(
    ("program", "expected_mode"),
    [
        (
            VideoActionProgram.FORWARD_DYNAMICS,
            DynamicsObjective.ACTION_CONDITIONED_VIDEO,
        ),
        (
            VideoActionProgram.INVERSE_DYNAMICS,
            DynamicsObjective.VIDEO_CONDITIONED_ACTION,
        ),
    ],
)
def test_dual_expert_conditional_dynamics_config_derives_fixed_gjd_contract(
    tmp_path: Path,
    program: VideoActionProgram,
    expected_mode: DynamicsObjective,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"]["program"] = program.value
    for route in raw["data"]["dynamics_routing"]["routes"]:
        route["mode"] = expected_mode.value
    config_path = tmp_path / f"conditional_{program.value}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.program is program
    assert config.policy_variant.current_block_coupling == CurrentBlockCoupling.JOINT
    assert config.policy_variant.generalist_mode_text_token is False
    assert config.policy_variant.sequence_contract == (
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT
    assert config.data.dynamics_routing.mode_probabilities() == {
        mode: float(mode is expected_mode) for mode in DynamicsObjective
    }
    assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
    assert config.data.sample_construction.condition_source_frame_offset == -1
    assert config.data.sample_construction.start_padding_frames == 0
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.LEGACY
    assert config.data.sample_construction.window_size == 64
    assert config.training.window_size == 64
    assert config.data.train_batch_size == 1
    assert config.data.val_batch_size == 1


def test_dual_expert_conditional_dynamics_rejects_conflicting_route(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["dynamics_routing"]["routes"][0]["mode"] = (
        "video_conditioned_action"
    )
    config_path = tmp_path / "conditional_fdm_conflicting_route.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="accepts only.*action_conditioned_video"):
        load_experiment_config(config_path)


def test_dual_expert_conditional_dynamics_keeps_gjd_training_contract() -> None:
    gjd = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    conditional = load_experiment_config(
        REPO_ROOT / "configs/experiments/dual_expert_libero_conditional_dynamics.yaml"
    )

    assert conditional.backbone == gjd.backbone
    assert conditional.action_decoder == gjd.action_decoder
    assert conditional.training == gjd.training
    assert conditional.inference == gjd.inference
    assert conditional.trainer == gjd.trainer
    assert replace(
        conditional.data,
        dynamics_routing=gjd.data.dynamics_routing,
    ) == gjd.data
    conditional_policy = asdict(conditional.policy_variant)
    gjd_policy = asdict(gjd.policy_variant)
    for field_name in ("program",):
        conditional_policy.pop(field_name)
        gjd_policy.pop(field_name)
    assert conditional_policy == gjd_policy


def test_m5_generalist_joint_denoising_rejects_multi_sample_batches(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["data"]["train_batch_size"] = 2

    config_path = tmp_path / "m5_generalist_bad_batch.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="Generalist and conditional-dynamics.*train_batch_size"):
        load_experiment_config(config_path)


def test_m5_generalist_joint_denoising_mode_text_token_flag_loads(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["generalist_mode_text_token"] = True

    config_path = tmp_path / "m5_generalist_mode_text_token.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.policy_variant.generalist_mode_text_token is True


def test_m5_generalist_mode_text_token_rejects_non_gjd_config(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["generalist_mode_text_token"] = True

    config_path = tmp_path / "m5_non_gjd_mode_text_token.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="generalist_mode_text_token"):
        load_experiment_config(config_path)


def test_m1_generalist_joint_denoising_mode_text_token_flag_loads(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["generalist_mode_text_token"] = True

    config_path = tmp_path / "generalist_joint_mode_text_token.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.policy_variant.generalist_mode_text_token is True


def test_m1_generalist_joint_denoising_mode_text_token_string_false_loads_false(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["generalist_mode_text_token"] = "false"

    config_path = tmp_path / "generalist_joint_mode_text_token_string_false.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.policy_variant.generalist_mode_text_token is False


def test_m1_generalist_mode_text_token_rejects_non_generalist_profile(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["generalist_mode_text_token"] = True

    config_path = tmp_path / "non_generalist_mode_text_token.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="generalist_mode_text_token"):
        load_experiment_config(config_path)


def test_parallel_program_configs_use_canonical_video_only_history() -> None:
    config_paths = [
        REPO_ROOT / "configs/experiments" / name
        for name in PARALLEL_STREAM_PROGRAM_CONFIG_NAMES
    ]
    assert len(config_paths) == 7
    for config_path in config_paths:
        config = load_experiment_config(config_path)
        assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        assert (
            config.policy_variant.history_stream_visibility
            == HistoryStreamVisibility.VIDEO_ONLY
        )


def test_m1_generalist_joint_denoising_derives_mode_probabilities_from_routes() -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    config = load_experiment_config(source_path)

    probs = config.data.dynamics_routing.mode_probabilities()
    assert probs[DynamicsObjective.JOINT] == pytest.approx(0.6)
    assert probs[DynamicsObjective.ACTION_CONDITIONED_VIDEO] == pytest.approx(0.2)
    assert probs[DynamicsObjective.VIDEO_CONDITIONED_ACTION] == pytest.approx(0.2)


def test_m1_generalist_joint_denoising_rejects_all_zero_routes(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    for route in raw["data"]["dynamics_routing"]["routes"]:
        route["weight"] = 0.0

    config_path = tmp_path / "generalist_joint_bad_probs.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="at least one positive weight"):
        load_experiment_config(config_path)


@pytest.mark.parametrize("bad_value", [True, "1.0", float("nan"), float("inf")])
def test_m1_generalist_joint_denoising_rejects_invalid_route_weights(
    tmp_path: Path,
    bad_value: object,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["data"]["dynamics_routing"]["routes"][1]["weight"] = bad_value

    config_path = tmp_path / "generalist_joint_non_numeric_probs.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="finite real number|finite and non-negative"):
        load_experiment_config(config_path)


def test_dynamics_routing_routes_reject_unknown_fields(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["dynamics_routing"]["routes"][0]["ratio"] = 0.6
    config_path = tmp_path / "generalist_joint_unknown_route_field.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"routes\[0\].*unknown fields: ratio"):
        load_experiment_config(config_path)


def test_dynamics_routing_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["dynamics_routing"]["lenght_multiplier"] = 2.0
    config_path = tmp_path / "generalist_joint_unknown_routing_field.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields: lenght_multiplier"):
        load_experiment_config(config_path)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("seed", True, "seed.*integer"),
        ("seed", 1.5, "seed.*integer"),
        ("length_multiplier", True, "length_multiplier.*finite positive number"),
        ("length_multiplier", "2", "length_multiplier.*finite positive number"),
        ("train_latent_root", 42, "train_latent_root.*string path or null"),
    ],
)
def test_dynamics_routing_rejects_invalid_scalar_types(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
    message: str,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/dual_expert_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["dynamics_routing"][field_name] = bad_value
    config_path = tmp_path / f"generalist_joint_bad_{field_name}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_experiment_config(config_path)


def test_parallel_generalist_joint_denoising_rejects_multi_sample_batches(
    tmp_path: Path,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["data"]["train_batch_size"] = 2
    config_path = tmp_path / "parallel_generalist_bad_batch.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Generalist and conditional-dynamics.*train_batch_size"):
        load_experiment_config(config_path)


def test_local_path_registry_overrides_sample_aliases(monkeypatch, tmp_path: Path) -> None:
    local_paths_path = tmp_path / "local_paths.yaml"
    with local_paths_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                    "paths": {
                        "datasets": {"libero_heng_root": "/tmp/custom_libero_root"},
                        "models": {"lingbot_va_base": "/tmp/custom_model_root"},
                    }
            },
            handle,
            sort_keys=False,
        )
    monkeypatch.setenv("OPEN_WAM_LOCAL_PATHS", str(local_paths_path))

    config = load_experiment_config(REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml")

    assert config.data.local_root == "/tmp/custom_libero_root"
    assert config.data.empty_text_embedding_path.endswith("empty_emb.pt")
    assert config.backbone.pretrained_model_name_or_path == "/tmp/custom_model_root"
    assert config.backbone.transformer_subdir == "transformer"


def test_local_path_registry_override_can_reference_sample_aliases(monkeypatch, tmp_path: Path) -> None:
    local_paths_path = tmp_path / "local_paths.yaml"
    with local_paths_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "paths": {
                    "datasets": {"libero_root": "/tmp/canonical_libero_root"},
                    "tests": {
                        "derived_checkpoint": "${paths.datasets.libero_heng_root}/derived/checkpoint_step_1/transformer"
                    }
                }
            },
            handle,
            sort_keys=False,
        )
    monkeypatch.setenv("OPEN_WAM_LOCAL_PATHS", str(local_paths_path))

    config_path = tmp_path / "alias_reference.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "name": "alias_reference_demo",
                "data": {
                    "dataset_name": "robotwin",
                    "local_root": "${paths.datasets.libero_heng_root}",
                },
                "backbone": {
                    "pretrained_model_name_or_path": "${paths.models.lingbot_va_base}",
                },
                "tests": {
                    "derived_checkpoint": "${paths.tests.derived_checkpoint}",
                },
            },
            handle,
            sort_keys=False,
        )

    resolved = read_yaml_with_local_paths(config_path)

    assert resolved["data"]["local_root"] == "/tmp/canonical_libero_root"
    assert resolved["backbone"]["pretrained_model_name_or_path"].endswith("lingbot-va-base")
    assert resolved["tests"]["derived_checkpoint"].endswith(
        "canonical_libero_root/derived/checkpoint_step_1/transformer"
    )


def test_missing_local_path_alias_raises_clear_error(monkeypatch, tmp_path: Path) -> None:
    local_paths_path = tmp_path / "local_paths.yaml"
    with local_paths_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"paths": {"tests": {"known_root": "/tmp/known"}}}, handle, sort_keys=False)
    monkeypatch.setenv("OPEN_WAM_LOCAL_PATHS", str(local_paths_path))

    config_path = tmp_path / "missing_alias.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "name": "missing_alias_demo",
                "data": {
                    "dataset_name": "robotwin",
                    "local_root": "${paths.tests.missing_root}",
                },
            },
            handle,
            sort_keys=False,
        )

    with pytest.raises(ValueError, match="configs/local_paths.sample.yaml"):
        load_experiment_config(config_path)


def test_loaded_enum_like_fields_are_real_enum_members() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )

    assert isinstance(config.backbone.train_attn_mode, AttentionMode)


@pytest.mark.parametrize(
    "removed_mode",
    ("random_subwindow", "contextual_subwindow", "aligned_subwindow"),
)
def test_removed_window_sampling_modes_are_rejected(tmp_path: Path, removed_mode: str) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["data"]["sample_construction"]["mode"] = removed_mode
    config_path = tmp_path / f"{removed_mode}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match=rf"{removed_mode}.*WindowSamplingMode"):
        load_experiment_config(config_path)


def test_deprecated_equal_bucket_latent_temporal_layout_is_rejected(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("data", {})
    raw["data"]["latent_temporal_layout"] = "equal_bucket_legacy"

    config_path = tmp_path / "deprecated_equal_bucket.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="equal_bucket_legacy.*deprecated and unsupported"):
        load_experiment_config(config_path)


def test_backbone_exported_runtime_action_init_mode_loads_as_enum(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("backbone", {})
    raw["backbone"]["exported_runtime_action_init_mode"] = "random"

    config_path = tmp_path / "joint_random_action_init.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.backbone.exported_runtime_action_init_mode == ExportedRuntimeActionInitMode.RANDOM


def test_parallel_stream_single_frame_context_flag_enables_condition_latents(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_decoupled_same_step.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"].pop("sequence_contract", None)
    raw["policy_variant"]["context_condition_latent_source"] = "single_frame_condition_latent"
    raw["policy_variant"]["history_stream_visibility"] = "video_only"
    raw.setdefault("data", {}).setdefault("sample_construction", {})["condition_source_frame_offset"] = -1

    config_path = tmp_path / "parallel_single_frame_context.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.condition_source_frame_offset == -1


def test_parallel_stream_single_frame_context_rejects_default_condition_offset(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_decoupled_same_step.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"].pop("sequence_contract", None)
    raw["policy_variant"]["context_condition_latent_source"] = "single_frame_condition_latent"
    raw.setdefault("data", {}).setdefault("sample_construction", {}).pop("condition_source_frame_offset", None)

    config_path = tmp_path / "parallel_single_frame_context_leaky_default.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="single_frame_condition_latent.*condition_source_frame_offset=-1"):
        load_experiment_config(config_path)


def test_parallel_stream_per_chunk_additive_proprio_flag_loads(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_decoupled_same_step.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"].pop("sequence_contract", None)
    raw["policy_variant"]["proprio_context_mode"] = "per_chunk_additive"
    raw["policy_variant"]["context_condition_latent_source"] = "single_frame_condition_latent"
    raw["policy_variant"]["history_stream_visibility"] = "video_only"
    raw.setdefault("data", {}).setdefault("sample_construction", {})["condition_source_frame_offset"] = -1

    config_path = tmp_path / "parallel_per_chunk_additive.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True


def test_dual_expert_per_chunk_single_frame_context_flags_load(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_decoupled_same_step.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"].pop("sequence_contract", None)
    raw["policy_variant"]["proprio_context_mode"] = "per_chunk_additive"
    raw["policy_variant"]["context_condition_latent_source"] = "single_frame_condition_latent"
    raw["policy_variant"]["history_stream_visibility"] = "video_only"
    raw.setdefault("data", {}).setdefault("sample_construction", {})["condition_source_frame_offset"] = -1

    config_path = tmp_path / "dual_expert_per_chunk_single_frame_context.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True


@pytest.mark.parametrize(
    ("config_name", "field_name"),
    [
        ("parallel_stream_libero_decoupled_same_step.yaml", "use_condition_latents"),
        ("parallel_stream_libero_decoupled_same_step.yaml", "require_condition_latents"),
        ("dual_expert_libero_decoupled_same_step.yaml", "use_condition_latents"),
        ("dual_expert_libero_decoupled_same_step.yaml", "require_condition_latents"),
    ],
)
def test_single_frame_condition_rejects_disabled_latent_flags(
    tmp_path: Path,
    config_name: str,
    field_name: str,
) -> None:
    source_path = REPO_ROOT / "configs/experiments" / config_name
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw["policy_variant"].pop("sequence_contract", None)
    raw["policy_variant"].update(
        {
            "context_condition_latent_source": "single_frame_condition_latent",
            "use_condition_latents": True,
            "require_condition_latents": True,
        }
    )
    raw["policy_variant"][field_name] = False
    raw["data"]["sample_construction"]["condition_source_frame_offset"] = -1
    config_path = tmp_path / f"single_frame_{field_name}_false.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="single_frame_condition_latent.*requires both",
    ):
        load_experiment_config(config_path)



def test_sequence_contract_expands_parallel_stream_rollout_parity_defaults(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_decoupled_same_step.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "rollout_parity_single_frame_perchunk_proprio"
    for key in (
        "proprio_context_mode",
        "context_condition_latent_source",
        "history_stream_visibility",
    ):
        raw["policy_variant"].pop(key, None)
    for key in (
        "condition_source_frame_offset",
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "parallel_contract.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.policy_variant.sequence_contract == (
        VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
    assert config.data.sample_construction.rollout_context_policy == RolloutContextPolicy.ONE_FRAME
    assert config.data.sample_construction.condition_source_frame_offset == -1
    assert config.data.sample_construction.context_prefix_policy == SegmentContextPolicy.NONE
    assert config.data.sample_construction.context_prefix_frames == 0


def test_sequence_contract_expands_dual_expert_rollout_parity_defaults(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_video_then_action.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "rollout_parity_single_frame_perchunk_proprio"
    for key in (
        "proprio_context_mode",
        "context_condition_latent_source",
        "history_stream_visibility",
    ):
        raw["policy_variant"].pop(key, None)
    for key in (
        "condition_source_frame_offset",
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "dual_expert_contract.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.sequence_contract == (
        VideoActionSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
    assert config.data.sample_construction.condition_source_frame_offset == -1


def test_sequence_contract_legacy_prefix_restores_target_only_sampling(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_video_then_action.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
    for key in (
        "proprio_context_mode",
        "context_condition_latent_source",
        "history_stream_visibility",
    ):
        raw["policy_variant"].pop(key, None)
    for key in (
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "parallel_legacy_prefix_contract.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.policy_variant.sequence_contract == (
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.LEGACY
    assert config.data.sample_construction.condition_source_frame_offset == -1
    assert config.data.sample_construction.start_padding_frames == 0


def test_sequence_contract_expands_dual_expert_legacy_prefix_defaults(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
    raw["policy_variant"]["joint_timestep_coupling"] = "shared_video_schedule"
    for key in (
        "proprio_context_mode",
        "context_condition_latent_source",
        "history_stream_visibility",
    ):
        raw["policy_variant"].pop(key, None)
    for key in (
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "dual_expert_legacy_prefix_contract.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.sequence_contract == (
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == HistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE
    assert config.policy_variant.noisy_video_condition_prob == pytest.approx(0.5)
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.LEGACY
    assert config.data.sample_construction.condition_source_frame_offset == -1
    assert config.data.sample_construction.start_padding_frames == 0


def test_sequence_contract_legacy_prefix_preserves_explicit_noisy_condition_prob(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_video_then_action.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
    raw["policy_variant"]["noisy_video_condition_prob"] = 0.0
    for key in (
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "dual_expert_legacy_prefix_explicit_noisy_prob.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.noisy_video_condition_prob == 0.0


def test_sequence_contract_legacy_prefix_preserves_explicit_joint_coupling(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
    raw["policy_variant"]["joint_timestep_coupling"] = "shared_video_schedule"
    for key in (
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "dual_expert_legacy_prefix_explicit_joint_coupling.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE


def test_sequence_contract_legacy_prefix_preserves_explicit_match_sigma_joint_coupling(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
    raw["policy_variant"]["joint_timestep_coupling"] = "match_sigma"
    for key in (
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "dual_expert_legacy_prefix_explicit_match_sigma_joint_coupling.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, DualExpertPolicyConfig)
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA


def test_sequence_contract_rejects_conflicting_explicit_values(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_decoupled_same_step.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "rollout_parity_single_frame_perchunk_proprio"
    raw["policy_variant"]["history_stream_visibility"] = "full"

    config_path = tmp_path / "parallel_contract_conflict.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="sequence_contract=.*history_stream_visibility=video_only"):
        load_experiment_config(config_path)


def test_shared_config_overrides_reject_sequence_contract_owned_fields() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )

    with pytest.raises(ValueError, match="sequence_contract=.*proprio_context_mode"):
        apply_config_overrides(
            config,
            {"policy_variant.proprio_context_mode": "none"},
        )


def test_shared_config_overrides_can_release_sequence_contract_fields() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_joint.yaml"
    )

    updated = apply_config_overrides(
        config,
        {
            "policy_variant.sequence_contract": "default",
            "policy_variant.proprio_context_mode": "none",
        },
    )

    assert updated.policy_variant.sequence_contract is VideoActionSequenceContract.DEFAULT
    assert updated.policy_variant.proprio_context_mode is ProprioContextMode.NONE


def test_parallel_stream_generalist_uses_planning_sequence_contract() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_generalist_joint_denoising.yaml"
    )

    assert config.policy_variant.sequence_contract == (
        VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.context_condition_latent_source == (
        ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.condition_source_frame_offset == -1


def test_sample_construction_defaults_to_replacement_order() -> None:
    assert (
        SampleConstructionConfig().sample_order_mode
        is SampleOrderMode.REPLACEMENT
    )


def test_sample_construction_yaml_strings_are_coerced_to_enum_members(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "default"
    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "uniform_segment",
        "anchor_policy": "random_valid",
        "num_frames": 5,
        "action_horizon": 20,
        "state_horizon": 2,
        "frame_stride": 1,
        "segment_min_frames": 8,
        "segment_max_frames": 32,
        "segment_length_stride": 4,
        "segment_locality_block_size": 3,
        "randomize_segment_length": True,
        "randomize_segment_start": True,
        "require_full_segment": True,
        "start_padding_frames": 3,
        "context_prefix_policy": "fixed",
        "context_prefix_frames": 5,
        "sample_weight_mode": "valid_action_steps_x_inverse_task_demo_count",
        "sample_order_mode": "replacement",
        "sample_weight_length_power": 0.5,
        "sample_weight_min": 0.25,
        "sample_weight_max": 4.0,
    }

    config_path = tmp_path / "sample_construction.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.data.sample_construction.mode == WindowSamplingMode.UNIFORM_SEGMENT
    assert config.data.sample_construction.anchor_policy == AnchorPolicy.RANDOM_VALID
    assert config.data.sample_construction.num_frames == 5
    assert config.data.sample_construction.action_horizon == 20
    assert config.data.sample_construction.state_horizon == 2
    assert config.data.sample_construction.segment_min_frames == 8
    assert config.data.sample_construction.segment_max_frames == 32
    assert config.data.sample_construction.segment_length_stride == 4
    assert config.data.sample_construction.segment_locality_block_size == 3
    assert config.data.sample_construction.randomize_segment_length is True
    assert config.data.sample_construction.randomize_segment_start is True
    assert config.data.sample_construction.require_full_segment is True
    assert config.data.sample_construction.start_padding_frames == 3
    assert config.data.sample_construction.context_prefix_policy == SegmentContextPolicy.FIXED
    assert config.data.sample_construction.context_prefix_frames == 5
    assert config.data.sample_construction.sample_weight_mode == SampleWeightMode.VALID_ACTION_STEPS_X_INVERSE_TASK_DEMO_COUNT
    assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
    assert config.data.sample_construction.sample_weight_length_power == pytest.approx(0.5)
    assert config.data.sample_construction.sample_weight_min == pytest.approx(0.25)
    assert config.data.sample_construction.sample_weight_max == pytest.approx(4.0)


def test_hierarchical_fixed_segment_sample_construction_loads_explicit_sampler_fields(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "default"
    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
        "sample_order_mode": "epoch_order",
        "segment_frames": 128,
        "start_padding_frames": 3,
        "context_prefix_policy": "rollout_history",
        "tail_padding_policy": "zero_order_hold",
        "padded_target_policy": "mask_loss",
        "task_start_power": 0.5,
        "demo_count_power": 0.0,
        "trajectory_start_power": 1.0,
    }

    config_path = tmp_path / "hierarchical_fixed_segment.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.data.sample_construction.mode == WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT
    assert config.data.sample_construction.segment_frames == 128
    assert config.data.sample_construction.start_padding_frames == 3
    assert config.data.sample_construction.context_prefix_policy == SegmentContextPolicy.ROLLOUT_HISTORY
    assert config.data.sample_construction.tail_padding_policy == TailPaddingPolicy.ZERO_ORDER_HOLD
    assert config.data.sample_construction.padded_target_policy == PaddedTargetPolicy.MASK_LOSS
    assert config.data.sample_construction.task_start_power == pytest.approx(0.5)
    assert config.data.sample_construction.demo_count_power == pytest.approx(0.0)
    assert config.data.sample_construction.trajectory_start_power == pytest.approx(1.0)


def test_hierarchical_fixed_segment_rejects_replacement_sample_order_typed_path(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "default"
    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
        "segment_frames": 128,
        "sample_order_mode": "replacement",
    }

    config_path = tmp_path / "bad_hierarchical_replacement_order.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="does not support replacement `sample_order_mode`"):
        load_experiment_config(config_path)


def test_hierarchical_fixed_segment_loads_strict_rollout_parity_fields(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "default"
    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
        "sample_order_mode": "epoch_order",
        "segment_frames": 128,
        "chunk_size": 4,
        "window_size": 30,
        "randomize_geometry": False,
        "start_padding_frames": 0,
        "condition_source_frame_offset": -1,
        "target_alignment": "next_after_context",
        "rollout_context_policy": "one_frame",
        "tail_padding_policy": "zero_order_hold",
        "padded_target_policy": "mask_loss",
        "task_start_power": 0.5,
        "demo_count_power": 0.0,
        "trajectory_start_power": 1.0,
    }
    raw.setdefault("training", {})["chunk_size"] = 4
    raw.setdefault("inference", {})["frame_chunk_size"] = 4

    config_path = tmp_path / "hierarchical_rollout_parity.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
    assert config.data.sample_construction.rollout_context_policy == RolloutContextPolicy.ONE_FRAME
    assert config.data.sample_construction.rollout_context_frames is None
    assert config.data.sample_construction.randomize_geometry is False
    assert config.data.sample_construction.start_padding_frames == 0
    assert config.data.sample_construction.condition_source_frame_offset == -1


def test_hierarchical_fixed_segment_rollout_parity_rejects_legacy_context_fields(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "default"
    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
        "sample_order_mode": "epoch_order",
        "segment_frames": 128,
        "chunk_size": 4,
        "randomize_geometry": False,
        "start_padding_frames": 0,
        "target_alignment": "next_after_context",
        "context_prefix_policy": "none",
    }

    config_path = tmp_path / "bad_hierarchical_rollout_parity.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="remove legacy context fields: `context_prefix_policy`"):
        load_experiment_config(config_path)


def test_sample_construction_rollout_parity_rejects_programmatic_legacy_context_fields() -> None:
    with pytest.raises(ValueError, match="remove legacy context fields"):
        SampleConstructionConfig(
            mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
            sample_order_mode=SampleOrderMode.EPOCH_ORDER,
            segment_frames=128,
            chunk_size=4,
            randomize_geometry=False,
            start_padding_frames=0,
            target_alignment=SampleTargetAlignment.NEXT_AFTER_CONTEXT,
            context_prefix_policy=SegmentContextPolicy.FIXED,
        )

    with pytest.raises(ValueError, match="remove legacy context fields"):
        SampleConstructionConfig(
            mode=WindowSamplingMode.HIERARCHICAL_FIXED_SEGMENT,
            sample_order_mode=SampleOrderMode.EPOCH_ORDER,
            segment_frames=128,
            chunk_size=4,
            randomize_geometry=False,
            start_padding_frames=0,
            target_alignment=SampleTargetAlignment.NEXT_AFTER_CONTEXT,
            context_prefix_frames=4,
        )


def test_hierarchical_fixed_segment_rollout_parity_rejects_malformed_chunk_size(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "default"
    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
        "sample_order_mode": "epoch_order",
        "segment_frames": 128,
        "chunk_size": 4,
        "randomize_geometry": False,
        "start_padding_frames": 0,
        "target_alignment": "next_after_context",
    }
    raw.setdefault("training", {})["chunk_size"] = "four"
    raw.setdefault("inference", {})["frame_chunk_size"] = 4

    config_path = tmp_path / "bad_strict_chunk.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="training\\.chunk_size='four'"):
        load_experiment_config(config_path)


def test_hierarchical_fixed_segment_rejects_legacy_full_segment_flag(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/dual_expert_libero_joint.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["sequence_contract"] = "default"
    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
        "sample_order_mode": "epoch_order",
        "segment_frames": 128,
        "require_full_segment": False,
    }

    config_path = tmp_path / "bad_hierarchical_fixed_segment.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="remove legacy fields: `require_full_segment`"):
        load_experiment_config(config_path)


def test_legacy_inference_aliases_still_map_to_generic_runtime_config(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["inference"].pop("video_cfg_mode", None)
    raw["inference"].pop("action_cfg_mode", None)
    raw["inference"].pop("joint_cache_initial_warmup_anchor", None)
    raw["inference"].pop("joint_cache_initial_warmup_frames", None)
    raw["inference"].pop("joint_cache_rollout_warmup_anchor", None)
    raw["inference"].pop("joint_cache_rollout_warmup_frames", None)
    raw["inference"]["joint_cfg_application"] = "joint"
    raw["inference"]["joint_cache_warmup_source"] = "dreamzero_reference_block"

    legacy_path = tmp_path / "legacy_inference_aliases.yaml"
    with legacy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(legacy_path)

    assert config.inference.video_cfg_mode == "guided"
    assert config.inference.action_cfg_mode == "guided"
    assert config.inference.joint_cache_warmup_source == "reference_video"
    assert config.inference.joint_cache_initial_warmup_anchor == "start"
    assert config.inference.joint_cache_initial_warmup_frames == 1
    assert config.inference.joint_cache_rollout_warmup_anchor == "end"
    assert config.inference.joint_cache_rollout_warmup_frames is None


def test_composable_runtime_fields_load(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["training"]["learning_rate"] = 2e-4
    raw["training"]["beta1"] = 0.8
    raw["training"]["beta2"] = 0.95
    raw["training"]["weight_decay"] = 0.1
    raw["training"]["warmup_steps"] = 7
    raw["training"]["gradient_accumulation_steps"] = 3
    raw["training"]["max_grad_norm"] = 1.5
    raw["training"]["num_steps"] = 11
    raw["training"]["text_condition_dropout_prob"] = 0.25
    raw["training"]["enabled_objectives"] = ["latent"]
    raw["training"]["latent_loss_weight"] = 0.75
    raw["training"]["action_loss_weight"] = 0.2
    raw["training"]["trainable_components"] = ["visual_tower.core", "policy_variant.action_expert", "action_decoder"]
    raw["training"]["frozen_components"] = ["visual_tower.frontend"]
    raw["trainer"]["runtime"] = "composable"
    raw["trainer"]["batch_adapter"] = "latents"
    raw["trainer"]["loop_policy"] = "steps"
    raw["trainer"]["strategy"] = "single_device"
    raw["trainer"]["default_root_dir"] = "/tmp/open-wam-test"
    raw["trainer"]["checkpoint_dir"] = "/tmp/open-wam-test/checkpoints"
    raw["trainer"]["save_interval"] = 5
    raw["trainer"]["checkpoint_mode"] = "model_only"
    raw["trainer"]["max_checkpoints_to_keep"] = 3
    raw["trainer"]["export_runtime_backbone"] = True
    raw["trainer"]["resume_from"] = "/tmp/open-wam-test/checkpoints/checkpoint_step_5"
    raw["trainer"]["enable_jsonl_logging"] = True
    raw["trainer"]["metrics_filename"] = "run.jsonl"
    raw["trainer"]["enable_wandb"] = True
    raw["trainer"]["wandb_project"] = "open-wam"
    raw["trainer"]["wandb_entity"] = "robotics"
    raw["trainer"]["wandb_mode"] = "offline"
    raw["trainer"]["run_name"] = "smoke-run"
    raw["data"]["latent_root"] = "/tmp/open-wam-test/latents"
    raw["data"]["latent_subdir"] = "custom_latents"
    raw["data"]["latent_camera_names"] = ["latent_cam_0", "latent_cam_1"]

    runtime_path = tmp_path / "runtime.yaml"
    with runtime_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(runtime_path)

    assert config.training.learning_rate == 2e-4
    assert config.training.beta1 == 0.8
    assert config.training.beta2 == 0.95
    assert config.training.weight_decay == 0.1
    assert config.training.warmup_steps == 7
    assert config.training.gradient_accumulation_steps == 3
    assert config.training.max_grad_norm == 1.5
    assert config.training.num_steps == 11
    assert config.training.text_condition_dropout_prob == 0.25
    assert config.training.enabled_objectives == ("latent",)
    assert config.training.latent_loss_weight == 0.75
    assert config.training.action_loss_weight == 0.2
    assert config.training.trainable_components == (
        "visual_tower.core",
        "policy_variant.action_expert",
        "action_decoder",
    )
    assert config.training.frozen_components == ("visual_tower.frontend",)
    assert config.trainer.runtime == "composable"
    assert config.trainer.batch_adapter == "latents"
    assert config.trainer.loop_policy == "steps"
    assert config.trainer.strategy == "single_device"
    assert config.trainer.default_root_dir == "/tmp/open-wam-test"
    assert config.trainer.checkpoint_dir == "/tmp/open-wam-test/checkpoints"
    assert config.trainer.save_interval == 5
    assert config.trainer.checkpoint_mode == "model_only"
    assert config.trainer.max_checkpoints_to_keep == 3
    assert config.trainer.export_runtime_backbone is True
    assert config.trainer.resume_from == "/tmp/open-wam-test/checkpoints/checkpoint_step_5"
    assert config.trainer.enable_jsonl_logging is True
    assert config.trainer.metrics_filename == "run.jsonl"
    assert config.trainer.enable_wandb is True
    assert config.trainer.wandb_project == "open-wam"
    assert config.trainer.wandb_entity == "robotics"
    assert config.trainer.wandb_mode == "offline"
    assert config.trainer.run_name == "smoke-run"
    assert config.data.latent_root == "/tmp/open-wam-test/latents"
    assert config.data.latent_subdir == "custom_latents"
    assert config.data.latent_camera_names == ("latent_cam_0", "latent_cam_1")


def test_causal_video_prediction_config_loads() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/causal_video_prediction_libero_latent_local.yaml"
    )

    assert isinstance(config.policy_variant, CausalVideoPredictionPolicyConfig)
    assert config.action_decoder.name == ActionDecoderName.VIDEO_ONLY
    assert config.data.sample_construction.mode == WindowSamplingMode.CAUSAL_PREFIX_SUFFIX
    assert config.training.enabled_objectives == ("latent",)
    assert config.training.trainable_components == (
        TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,
    )
    assert config.training.frozen_components == ()
    assert config.data.sample_construction.causal_prefix_suffix_buckets == (
        CausalPrefixSuffixBucketConfig(observed_frames=1, future_frames=3),
        CausalPrefixSuffixBucketConfig(observed_frames=2, future_frames=6),
        CausalPrefixSuffixBucketConfig(observed_frames=5, future_frames=10),
    )


def test_causal_video_prediction_mixed_video_config_loads() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/causal_video_prediction_mixed_video.yaml")

    assert isinstance(config.policy_variant, CausalVideoPredictionPolicyConfig)
    assert isinstance(config.data, MixedVideoDataConfig)
    assert config.backbone.load_wan_vae_frontend is True
    assert config.trainer.batch_adapter == BatchAdapterName.VIEWS
    assert config.data.sample_construction.causal_prefix_suffix_buckets[0] == CausalPrefixSuffixBucketConfig(
        observed_frames=1,
        future_frames=4,
    )
