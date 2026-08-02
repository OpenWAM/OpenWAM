from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pytest
import yaml

from open_wam.configs import (
    ActionChunkAnchorMode,
    ActionDecoderName,
    ActionNormalizationMode,
    ActionTargetRepresentation,
    AuxiliaryValidationSource,
    AttentionMode,
    BatchAdapterName,
    CausalPrefixSuffixBucketConfig,
    CausalVideoPredictionPolicyConfig,
    ExportedRuntimeActionInitMode,
    GeneralistTrainingParadigm,
    JointDenoiseTrainingMode,
    JointTimestepCoupling,
    LatentWindowProfile,
    MixedVideoDataConfig,
    MixedVideoDecodeSizeMode,
    MixedVideoFrameFitMode,
    MixedVideoLatentEncodingMode,
    MixedVideoMissingStreamPolicy,
    MixedVideoRandomMode,
    MixedVideoSourceFormat,
    MixedVideoWeightMode,
    MoTPolicyConfig,
    MoTRuntimeMode,
    MoTPreset,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ParallelRuntimeMode,
    ParallelSequenceContract,
    ParallelStreamPolicyConfig,
    ParallelStreamVariantProfile,
    PoolingMode,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    AnchorPolicy,
    PaddedTargetPolicy,
    ProprioContextMode,
    ReplayStatusPolicy,
    RolloutContextPolicy,
    SampleConstructionConfig,
    SampleOrderMode,
    SampleStateAnchorMode,
    SampleLossWeightMode,
    SampleTargetAlignment,
    SampleWeightMode,
    SegmentContextPolicy,
    TailPaddingPolicy,
    ReferenceCoreInitMode,
    TemporalPositionMode,
    WindowSamplingMode,
    StrategyName,
    TrainingComponentSelector,
    TrainingObjective,
    VideoConditionInputSpace,
    VideoConditionTrainMode,
)
from open_wam.models.policy_variants.parallel_stream.variant import ParallelStreamPolicyVariant
from open_wam.configs import load_experiment_config
from open_wam.configs import read_yaml_with_local_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize(
    "config_name",
    [
        "parallel_stream_libero_lingbot_exact_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_m1_action_then_video_heng_compatible.yaml",
        "parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml",
        "mot_libero_latent_local_video_then_action_heng_compatible.yaml",
    ],
)
def test_absolute_action_features_are_opt_in_for_legacy_libero_training_configs(config_name: str) -> None:
    config = load_experiment_config(REPO_ROOT / "configs" / "experiments" / config_name)

    assert config.data.action_schema.action_dim == 7
    assert config.data.action_target.representation == ActionTargetRepresentation.RAW
    assert config.data.action_target.source_key == "action"
    assert config.data.action_target.normalization.mode == ActionNormalizationMode.NONE
    assert config.data.action_target.joint_position_normalization.mode == ActionNormalizationMode.NONE
    assert getattr(config.policy_variant, "proprio_context_mode", ProprioContextMode.NONE) == ProprioContextMode.NONE
    assert getattr(config.action_decoder, "recovered_osc_loss_weight", 0.0) == 0.0


@pytest.mark.parametrize(
    ("config_name", "action_dim"),
    [
        ("contract_only_libero.yaml", 7),
        ("contract_only_libero_local.yaml", 7),
        ("contract_only_robotwin.yaml", 30),
    ],
)
def test_maintained_contract_only_configs_use_explicit_components(
    config_name: str,
    action_dim: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = REPO_ROOT / "configs" / "experiments" / config_name
    monkeypatch.setenv(
        "OPEN_WAM_LOCAL_PATHS",
        str(REPO_ROOT / "configs" / "local_paths.sample.yaml"),
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = load_experiment_config(config_path)

    assert "action_head" not in raw
    assert "policy_variant" in raw
    assert "action_decoder" in raw
    assert isinstance(config.policy_variant, PostLatentPolicyConfig)
    assert config.policy_variant.pooling_mode == PoolingMode.COMPAT_GLOBAL_MEAN
    assert config.policy_variant.compatibility_mode is True
    assert config.action_decoder.name == ActionDecoderName.MLP
    assert config.action_decoder.action_dim == action_dim
    assert config.action_decoder.action_horizon == 6
    assert not hasattr(config, "action_head")


def test_external_legacy_action_head_maps_to_explicit_components(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "legacy_action_head.yaml"
    config_path.write_text(
        """
name: legacy_action_head
data:
  dataset_name: synthetic
  dataset_type: synthetic_multiview
  num_frames: 4
  action_schema:
    action_dim: 4
    action_horizon: 2
    state_dim: 3
    state_horizon: 1
backbone:
  implementation: dummy
  hidden_size: 32
action_head:
  name: contract_only
  hidden_size: 32
  action_dim: 4
  action_horizon: 2
  state_dim: 3
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, PostLatentPolicyConfig)
    assert config.policy_variant.pooling_mode == PoolingMode.COMPAT_GLOBAL_MEAN
    assert config.policy_variant.compatibility_mode is True
    assert config.action_decoder.name == ActionDecoderName.MLP
    assert config.action_decoder.action_dim == 4
    assert config.action_decoder.action_horizon == 2


def test_new_variant_yaml_configs_load() -> None:
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_robotwin.yaml")
    post_latent_video_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml"
    )
    post_decoded_video_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_decoded_robotwin_video_conditioned.yaml"
    )
    mot = load_experiment_config(REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml")
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin.yaml")
    smoke_parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")

    assert isinstance(post_decoded.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(post_latent_video_conditioned.policy_variant, PostLatentPolicyConfig)
    assert isinstance(post_decoded_video_conditioned.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(mot.policy_variant, MoTPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(smoke_parallel.policy_variant, ParallelStreamPolicyConfig)
    assert post_latent_video_conditioned.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert post_decoded_video_conditioned.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert mot.policy_variant.preset == MoTPreset.FASTWAM
    assert mot.policy_variant.runtime_mode == MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE
    assert mot.policy_variant.condition_mode == "first_frame"
    assert mot.policy_variant.use_condition_latents is True
    assert mot.training.trainable_components == (TrainingComponentSelector.POLICY_VARIANT_ACTION_EXPERT,)
    assert parallel.backbone.implementation == "shared_transformer"
    assert smoke_parallel.backbone.implementation == "shared_transformer"
    assert parallel.backbone.train_attn_mode == "flex"
    assert parallel.backbone.infer_attn_mode == "torch"
    assert parallel.policy_variant.runtime_mode == "lingbot_exact"
    assert parallel.action_decoder.name == "lingbot_parallel_decoder"
    assert parallel.backbone.hidden_size == 3072
    assert smoke_parallel.policy_variant.runtime_mode == "lingbot_exact"
    assert smoke_parallel.action_decoder.name == "lingbot_parallel_decoder"
    assert smoke_parallel.backbone.reference_model_path is None
    assert post_latent_video_conditioned.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT
    assert post_decoded_video_conditioned.policy_variant.video_condition_input_space == VideoConditionInputSpace.RGB_VIDEO


def test_generalist_mixed_dynamics_knob_loads_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_dynamics.yaml"
    config_path.write_text(
        """
name: mixed_dynamics
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  generalist_dynamics_mixture:
    train_latent_root: /tmp/counterfactual_train/encoded_latents
    val_latent_root: /tmp/counterfactual_val/encoded_latents
    allow_train_latent_root_for_val: false
    real_joint_weight: 0.6
    real_action_conditioned_video_weight: 0.1
    real_video_conditioned_action_weight: 0.1
    counterfactual_action_conditioned_video_weight: 0.1
    counterfactual_video_conditioned_action_weight: 0.1
    conditional_history_frames: 8
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  current_block_coupling: joint
  video_condition_on_action: true
  generalist_training_paradigm: mixed_dynamics
action_decoder:
  name: lingbot_parallel_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: latents
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.policy_variant.generalist_training_paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT
    assert config.data.generalist_dynamics_mixture.train_latent_root == "/tmp/counterfactual_train/encoded_latents"
    assert config.data.generalist_dynamics_mixture.allow_train_latent_root_for_val is False
    assert config.data.generalist_dynamics_mixture.real_joint_weight == 0.6
    assert config.data.generalist_dynamics_mixture.conditional_history_frames == 8


def test_generalist_mixed_dynamics_accepts_replacement_sample_order(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_dynamics_replacement_order.yaml"
    config_path.write_text(
        """
name: mixed_dynamics_replacement_order
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
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
  generalist_dynamics_mixture:
    train_latent_root: /tmp/counterfactual_train/encoded_latents
    val_latent_root: /tmp/counterfactual_val/encoded_latents
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  current_block_coupling: joint
  video_condition_on_action: true
  generalist_training_paradigm: mixed_dynamics
action_decoder:
  name: lingbot_parallel_decoder
  action_dim: 7
  action_horizon: 16
trainer:
  accelerator: cpu
  batch_adapter: latents
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.policy_variant.generalist_training_paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS
    assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
    assert config.data.sample_construction.chunk_size == 4
    assert config.data.sample_construction.window_size == 64
    assert config.data.sample_construction.randomize_geometry is True
    assert config.data.sample_construction.sample_weight_mode == SampleWeightMode.UNIFORM
    assert config.data.generalist_dynamics_mixture.train_latent_root == "/tmp/counterfactual_train/encoded_latents"
    assert config.data.generalist_dynamics_mixture.val_latent_root == "/tmp/counterfactual_val/encoded_latents"
    assert config.data.generalist_dynamics_mixture.conditional_history_frames is None


def test_generalist_mixed_dynamics_rejects_views_batch_adapter(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_dynamics_views_adapter.yaml"
    config_path.write_text(
        """
name: mixed_dynamics_views_adapter
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  generalist_dynamics_mixture:
    train_latent_root: /tmp/counterfactual_train/encoded_latents
    val_latent_root: /tmp/counterfactual_val/encoded_latents
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  current_block_coupling: joint
  video_condition_on_action: true
  generalist_training_paradigm: mixed_dynamics
action_decoder:
  name: lingbot_parallel_decoder
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


def test_generalist_mixed_dynamics_rejects_non_uniform_sample_weight(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed_dynamics_weighted_source.yaml"
    config_path.write_text(
        """
name: mixed_dynamics_weighted_source
data:
  dataset_name: libero
  dataset_type: lerobot_v2_latent_local
  sample_construction:
    mode: uniform_segment
    sample_weight_mode: valid_action_steps
  action_schema:
    action_dim: 7
    action_horizon: 16
    state_dim: 8
    state_horizon: 1
  generalist_dynamics_mixture:
    train_latent_root: /tmp/counterfactual_train/encoded_latents
    val_latent_root: /tmp/counterfactual_val/encoded_latents
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
  variant_profile: generalist_joint_denoising
  current_block_coupling: joint
  video_condition_on_action: true
  generalist_training_paradigm: mixed_dynamics
action_decoder:
  name: lingbot_parallel_decoder
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
backbone:
  implementation: shared_transformer
policy_variant:
  name: parallel_stream
  attach_site: within_visual_core
  runtime_mode: lingbot_exact_action_conditioned
action_decoder:
  name: lingbot_parallel_decoder
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
      max_batches: 8
      report_prefix: val_idm
      drop_text_conditioning: false
trainer:
  accelerator: cpu
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    fdm, idm = config.validation.auxiliary_tasks
    assert fdm.name == "fdm_val"
    assert fdm.mode_override == JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO
    assert fdm.source == AuxiliaryValidationSource.COUNTERFACTUAL_DYNAMICS_IF_AVAILABLE
    assert fdm.phase == "val_fdm"
    assert fdm.should_drop_text is True
    assert idm.mode_override == JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION
    assert idm.max_batches == 8
    assert idm.should_drop_text is False


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
  runtime_mode: lingbot_exact_action_conditioned
action_decoder:
  name: lingbot_parallel_decoder
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
    post_latent = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_libero_smoke.yaml")
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_libero_smoke.yaml")
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_raw_smoke.yaml")
    parallel_action_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_action_conditioned_smoke.yaml"
    )

    assert isinstance(post_latent.policy_variant, PostLatentPolicyConfig)
    assert isinstance(post_decoded.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(parallel_action_conditioned.policy_variant, ParallelStreamPolicyConfig)

    assert post_latent.data.dataset_name == "libero"
    assert post_decoded.data.dataset_name == "libero"
    assert parallel.data.dataset_name == "libero"
    assert parallel_action_conditioned.data.dataset_name == "libero"

    assert parallel.data.action_schema.action_horizon == 16
    assert parallel.action_decoder.name == ActionDecoderName.LINGBOT_PARALLEL
    assert parallel_action_conditioned.policy_variant.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    assert parallel_action_conditioned.policy_variant.video_condition_on_action is True
    assert parallel_action_conditioned.policy_variant.video_action_condition_source == "noisy_action"
    assert parallel_action_conditioned.policy_variant.video_action_attention_scope == "block_local"


def test_latent_libero_local_training_yaml_configs_load() -> None:
    post_latent = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml")
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_libero_latent_local.yaml")
    post_latent_video_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_libero_latent_local_video_conditioned.yaml"
    )
    post_decoded_video_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_decoded_libero_latent_local_video_conditioned.yaml"
    )
    assert post_latent.data.dataset_type == "lerobot_v2_latent_local"
    assert post_decoded.data.dataset_type == "lerobot_v2_latent_local"
    assert post_latent_video_conditioned.data.dataset_type == "lerobot_v2_latent_local"
    assert post_decoded_video_conditioned.data.dataset_type == "lerobot_v2_latent_local"
    assert post_latent.data.local_root.endswith("/libero_heng/libero_10")
    assert post_decoded.data.local_root.endswith("/libero_heng/libero_10")
    assert post_latent_video_conditioned.data.local_root.endswith("/libero_heng/libero_10")
    assert post_decoded_video_conditioned.data.local_root.endswith("/libero_heng/libero_10")
    assert post_latent.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert post_decoded.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert post_latent_video_conditioned.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert post_decoded_video_conditioned.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert post_latent.trainer.strategy == StrategyName.FSDP
    assert post_decoded.trainer.strategy == StrategyName.FSDP
    assert post_latent_video_conditioned.trainer.strategy == StrategyName.FSDP
    assert post_decoded_video_conditioned.trainer.strategy == StrategyName.FSDP
    assert post_latent.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert post_decoded.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert post_latent_video_conditioned.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert post_decoded_video_conditioned.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert post_latent.backbone.load_reference_core_weights is True
    assert post_decoded.backbone.load_reference_core_weights is True
    assert post_latent_video_conditioned.backbone.load_reference_core_weights is True
    assert post_decoded_video_conditioned.backbone.load_reference_core_weights is True
    assert post_latent.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert post_decoded.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert post_latent_video_conditioned.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert post_decoded_video_conditioned.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert post_latent.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_decoded.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_latent_video_conditioned.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_decoded_video_conditioned.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_latent_video_conditioned.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert post_decoded_video_conditioned.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert post_latent_video_conditioned.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT
    assert post_decoded_video_conditioned.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT
def test_method4_current_frame_regression_yaml_configs_load() -> None:
    latent_direct = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_libero_latent_local_current_frame_regression.yaml"
    )
    rgb_direct = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_decoded_robotwin_current_frame_regression.yaml"
    )

    assert isinstance(latent_direct.policy_variant, PostLatentPolicyConfig)
    assert isinstance(rgb_direct.policy_variant, PostDecodedPolicyConfig)
    assert latent_direct.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert rgb_direct.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert latent_direct.action_decoder.train_mode == VideoConditionTrainMode.CURRENT_FRAME_REGRESSION
    assert rgb_direct.action_decoder.train_mode == VideoConditionTrainMode.CURRENT_FRAME_REGRESSION
    assert latent_direct.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT
    assert rgb_direct.policy_variant.video_condition_input_space == VideoConditionInputSpace.RGB_VIDEO


def test_method4_defaults_to_video_conditioned_decoder_when_action_decoder_is_omitted(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        post_latent_raw = yaml.safe_load(handle)
    post_latent_raw["name"] = "post_latent_default_video_conditioned"
    post_latent_raw.pop("action_decoder", None)

    post_latent_path = tmp_path / "post_latent_default_video_conditioned.yaml"
    with post_latent_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(post_latent_raw, handle, sort_keys=False)

    post_latent_config = load_experiment_config(post_latent_path)
    assert isinstance(post_latent_config.policy_variant, PostLatentPolicyConfig)
    assert post_latent_config.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert post_latent_config.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT
    assert post_latent_config.policy_variant.action_chunk_anchor_mode == ActionChunkAnchorMode.CURRENT_PLUS_FUTURE

    source_path = REPO_ROOT / "configs/experiments/post_decoded_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        post_decoded_raw = yaml.safe_load(handle)
    post_decoded_raw["name"] = "post_decoded_default_video_conditioned"
    post_decoded_raw.pop("action_decoder", None)

    post_decoded_path = tmp_path / "post_decoded_default_video_conditioned.yaml"
    with post_decoded_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(post_decoded_raw, handle, sort_keys=False)

    post_decoded_config = load_experiment_config(post_decoded_path)
    assert isinstance(post_decoded_config.policy_variant, PostDecodedPolicyConfig)
    assert post_decoded_config.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert post_decoded_config.policy_variant.video_condition_input_space == VideoConditionInputSpace.RGB_VIDEO
    assert post_decoded_config.policy_variant.action_chunk_anchor_mode == ActionChunkAnchorMode.CURRENT_PLUS_FUTURE


def test_mot_policy_yaml_config_loads(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "mot_robotwin"
    raw["policy_variant"]["name"] = "mot"
    raw["policy_variant"]["attach_site"] = "post_visual_core"
    raw["policy_variant"]["runtime_mode"] = "video_prefill_action_denoise"
    raw["policy_variant"]["video_prefix_frames"] = 1
    raw["policy_variant"]["num_action_layers"] = 4
    raw["policy_variant"]["action_hidden_size"] = 768
    raw["policy_variant"]["action_ffn_dim"] = 1024
    raw["policy_variant"]["use_state_conditioning"] = True
    raw["policy_variant"]["proprio_context_mode"] = "per_chunk_additive"
    raw["action_decoder"]["name"] = "mlp_decoder"

    config_path = tmp_path / "mot_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.name == "mot"
    assert config.policy_variant.runtime_mode == MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE
    assert config.policy_variant.video_prefix_frames == 1
    assert config.policy_variant.num_action_layers == 4
    assert config.policy_variant.action_hidden_size == 768
    assert config.policy_variant.action_ffn_dim == 1024
    assert config.policy_variant.use_state_conditioning is True
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert config.action_decoder.name == ActionDecoderName.MOT


def test_mot_policy_allows_shared_video_schedule(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "mot_shared_video_schedule_robotwin"
    raw["policy_variant"]["name"] = "mot"
    raw["policy_variant"]["attach_site"] = "post_visual_core"
    raw["policy_variant"]["runtime_mode"] = "video_prefill_action_denoise"
    raw["policy_variant"]["joint_timestep_coupling"] = "shared_video_schedule"
    raw["action_decoder"]["name"] = "mlp_decoder"

    config_path = tmp_path / "mot_shared_video_schedule_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE


def test_mot_policy_preset_applies_fastwam_joint_defaults(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "mot_fastwam_joint_robotwin"
    raw["policy_variant"]["name"] = "mot"
    raw["policy_variant"]["attach_site"] = "post_visual_core"
    raw["policy_variant"]["preset"] = "fastwam_joint"
    raw["action_decoder"]["name"] = "mlp_decoder"

    config_path = tmp_path / "mot_fastwam_joint_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.preset == MoTPreset.FASTWAM_JOINT
    assert config.policy_variant.condition_mode == "full_video"
    assert config.action_decoder.name == ActionDecoderName.MOT
    assert config.policy_variant.teacher_forcing_video_noise_prob == 0.0
    assert config.policy_variant.video_prefix_frames == 1


def test_mot_policy_preset_allows_explicit_overrides(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "mot_fastwam_override_robotwin"
    raw["policy_variant"]["name"] = "mot"
    raw["policy_variant"]["attach_site"] = "post_visual_core"
    raw["policy_variant"]["preset"] = "fastwam"
    raw["policy_variant"]["condition_mode"] = "teacher_forcing_cond_video"
    raw["policy_variant"]["teacher_forcing_video_noise_prob"] = 0.2
    raw["policy_variant"]["video_prefix_frames"] = 3
    raw["action_decoder"]["name"] = "mlp_decoder"

    config_path = tmp_path / "mot_fastwam_override_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.preset == MoTPreset.FASTWAM
    assert config.policy_variant.condition_mode == "teacher_forcing_cond_video"
    assert config.policy_variant.teacher_forcing_video_noise_prob == 0.2
    assert config.policy_variant.video_prefix_frames == 3


def test_local_libero_yaml_config_loads() -> None:
    local_libero = load_experiment_config(REPO_ROOT / "configs/experiments/contract_only_libero_local.yaml")
    assert local_libero.data.dataset_name == "libero"
    assert local_libero.data.dataset_type == "libero_hdf5"
    assert local_libero.data.repo_id is None
    assert "${" not in local_libero.data.local_root
    assert Path(local_libero.data.local_root).name == "libero_10"


def test_heng_compatible_libero_yaml_config_loads() -> None:
    heng_libero = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
    )

    assert isinstance(heng_libero.policy_variant, ParallelStreamPolicyConfig)
    assert heng_libero.data.dataset_type == "lerobot_v2_latent_local"
    assert heng_libero.data.local_root.endswith("libero_heng/libero_10")
    assert heng_libero.data.empty_text_embedding_path.endswith("empty_emb.pt")
    assert heng_libero.data.camera_names == (
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    )
    assert heng_libero.data.latent_camera_names == (
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    )
    assert heng_libero.data.action_target.source_key == "action"
    assert heng_libero.data.action_target.pose_source_key == "observation.state"
    assert heng_libero.data.latent_window_profile == LatentWindowProfile.EXACT_CHUNKED_WINDOW
    assert heng_libero.data.replay_status_policy == ReplayStatusPolicy.SUCCESSFUL_ONLY
    assert heng_libero.data.val_replay_status_policy == ReplayStatusPolicy.FAILURE_ONLY
    assert heng_libero.data.require_replay_status is True
    assert heng_libero.data.val_require_replay_status is True
    assert heng_libero.training.learning_rate == 1e-5
    assert heng_libero.training.gradient_accumulation_steps == 10
    assert heng_libero.training.num_steps == 5000
    assert heng_libero.training.enabled_objectives == ("latent", "action")
    assert heng_libero.training.action_loss_weight == 1.0
    assert heng_libero.training.trainable_components == ("visual_tower.runtime_backbone",)
    assert heng_libero.training.sample_loss_weight_mode == SampleLossWeightMode.VALID_ACTION_STEPS
    assert heng_libero.training.sample_loss_weight_min == 0.25
    assert heng_libero.training.sample_loss_weight_max == 4.0
    assert heng_libero.trainer.runtime == "composable"
    assert heng_libero.trainer.batch_adapter == "latents"
    assert heng_libero.trainer.loop_policy == "steps"
    assert heng_libero.trainer.strategy == "fsdp"
    assert heng_libero.trainer.save_interval == 100
    assert heng_libero.trainer.enable_wandb is True
    assert heng_libero.trainer.wandb_project == "openwam-method1-libero"


def test_current_frame_action_chunk_libero_yaml_config_loads() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_current_frame_action_chunk_heng_compatible.yaml"
    )

    assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
    assert config.policy_variant.runtime_mode == ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert config.policy_variant.temporal_position_mode == TemporalPositionMode.LOCAL_ZERO_BASED
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.backbone.train_attn_mode == AttentionMode.FLEX
    assert config.data.sample_construction.mode == "uniform_segment"
    assert config.data.sample_construction.segment_min_frames == 4
    assert config.data.sample_construction.segment_max_frames == 4
    assert config.data.action_schema.state_horizon == 1
    assert config.action_decoder.name == ActionDecoderName.LINGBOT_PARALLEL
    assert config.action_decoder.action_dim == 30
    assert config.training.enabled_objectives == (TrainingObjective.ACTION,)
    assert config.training.latent_loss_weight == 0.0
    assert config.training.action_loss_weight == 1.0
    assert config.inference.use_cache is False
    _instantiate_parallel_stream_variant(config)


def test_fastwam_first_frame_libero_yaml_config_loads() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_fastwam_first_frame_heng_compatible.yaml"
    )

    assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
    assert config.policy_variant.runtime_mode == ParallelRuntimeMode.FASTWAM_FIRST_FRAME
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert config.policy_variant.temporal_position_mode == TemporalPositionMode.LOCAL_ZERO_BASED
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.mode == "uniform_segment"
    assert config.data.sample_construction.state_anchor_mode == SampleStateAnchorMode.SAMPLE_START_FRAME
    assert config.data.sample_construction.segment_min_frames == 4
    assert config.data.sample_construction.segment_max_frames == 4
    assert config.data.action_schema.state_horizon == 1
    assert config.action_decoder.name == ActionDecoderName.LINGBOT_PARALLEL
    assert config.action_decoder.action_dim == 30
    assert config.training.enabled_objectives == (TrainingObjective.LATENT, TrainingObjective.ACTION)
    assert config.training.latent_loss_weight == 1.0
    assert config.training.action_loss_weight == 1.0
    assert config.inference.use_cache is False
    assert config.inference.guidance_scale == 1.0
    _instantiate_parallel_stream_variant(config)


def test_m1_non_generalist_heng_compatible_configs_instantiate_reference_variant() -> None:
    config_paths = sorted(
        (REPO_ROOT / "configs/experiments").glob("parallel_stream_libero_lingbot_m1_*_heng_compatible.yaml")
    )
    config_paths = [path for path in config_paths if "generalist_joint_denoising" not in path.name]
    assert len(config_paths) == 8

    for config_path in config_paths:
        config = load_experiment_config(config_path)
        assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        assert config.policy_variant.variant_profile != ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
        _instantiate_parallel_stream_variant(config)


def test_m1_generalist_joint_denoising_keeps_guidance_scale_strict() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
    )
    config = replace(config, inference=replace(config.inference, guidance_scale=1.0))

    with pytest.raises(ValueError, match="guidance_scale"):
        _instantiate_parallel_stream_variant(config)


def test_m1_generalist_joint_denoising_yaml_config_loads() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
    )

    assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
    assert config.policy_variant.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    assert config.policy_variant.variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
    assert config.policy_variant.current_block_coupling == "joint"
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT
    assert config.policy_variant.attn_window == 30
    assert config.policy_variant.parallel_sequence_contract == ParallelSequenceContract.DEFAULT
    assert config.policy_variant.generalist_training_paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS
    assert config.data.sample_construction.mode == WindowSamplingMode.UNIFORM_SEGMENT
    assert config.data.sample_construction.sample_order_mode == SampleOrderMode.REPLACEMENT
    assert config.data.sample_construction.window_size == 64
    assert config.data.sample_construction.segment_min_frames == 1000
    assert config.data.sample_construction.segment_max_frames == 1000
    assert config.data.sample_construction.require_full_segment is True
    assert config.data.generalist_dynamics_mixture.train_latent_root is not None
    assert config.data.generalist_dynamics_mixture.val_latent_root is not None
    assert config.training.window_size == 64
    assert config.training.sample_loss_weight_mode == SampleLossWeightMode.NONE
    probs = config.policy_variant.joint_denoise_training_mode_probs
    assert probs[JointDenoiseTrainingMode.JOINT] == pytest.approx(0.6)
    assert probs[JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO] == pytest.approx(0.2)
    assert probs[JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION] == pytest.approx(0.2)
    assert config.action_decoder.name == ActionDecoderName.LINGBOT_PARALLEL


def test_m5_generalist_joint_denoising_defaults_independent_joint_coupling(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"].pop("joint_timestep_coupling", None)

    config_path = tmp_path / "m5_generalist_default_coupling.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.INDEPENDENT


def test_m5_generalist_joint_denoising_rejects_multi_sample_batches(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["data"]["train_batch_size"] = 2

    config_path = tmp_path / "m5_generalist_bad_batch.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="mot_generalist_training_mode_probs.*train_batch_size"):
        load_experiment_config(config_path)


def test_m5_generalist_joint_denoising_mode_text_token_flag_loads(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_generalist_joint_denoising_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["generalist_mode_text_token"] = True

    config_path = tmp_path / "m5_generalist_mode_text_token.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.policy_variant.generalist_mode_text_token is True


def test_m5_generalist_mode_text_token_rejects_non_gjd_config(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_joint_heng_compatible.yaml"
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
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
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
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
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
        / "configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["generalist_mode_text_token"] = True

    config_path = tmp_path / "non_generalist_mode_text_token.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="generalist_mode_text_token"):
        load_experiment_config(config_path)


def test_m1_step3500_variant_yaml_configs_preserve_video_pretrain_history() -> None:
    config_paths = sorted(
        (REPO_ROOT / "configs/experiments").glob(
            "parallel_stream_libero_lingbot_m1_*_heng_compatible.yaml"
        )
    )
    config_paths = [
        path
        for path in config_paths
        if "current_frame_action_chunk" not in path.name and "fastwam_first_frame" not in path.name
    ]
    assert len(config_paths) == 7
    for config_path in config_paths:
        config = load_experiment_config(config_path)
        assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
        assert config.policy_variant.preserve_video_pretrain_history is True


def test_m1_generalist_joint_denoising_defaults_mode_probabilities(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"].pop("joint_denoise_training_mode_probs")

    config_path = tmp_path / "generalist_joint_default_probs.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    probs = config.policy_variant.joint_denoise_training_mode_probs
    assert probs[JointDenoiseTrainingMode.JOINT] == pytest.approx(0.6)
    assert probs[JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO] == pytest.approx(0.2)
    assert probs[JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION] == pytest.approx(0.2)


def test_m1_generalist_joint_denoising_rejects_invalid_probabilities(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["joint_denoise_training_mode_probs"] = {
        "joint": 0.0,
        "action_conditioned_video": 0.0,
        "video_conditioned_action": 0.0,
    }

    config_path = tmp_path / "generalist_joint_bad_probs.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="at least one positive probability"):
        load_experiment_config(config_path)


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf")])
def test_m1_generalist_joint_denoising_rejects_non_numeric_probabilities(
    tmp_path: Path,
    bad_value: object,
) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["policy_variant"]["joint_denoise_training_mode_probs"] = {
        "joint": 0.6,
        "action_conditioned_video": bad_value,
        "video_conditioned_action": 0.2,
    }

    config_path = tmp_path / "generalist_joint_non_numeric_probs.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="finite numeric probabilities"):
        load_experiment_config(config_path)


def test_local_path_registry_overrides_sample_aliases(monkeypatch, tmp_path: Path) -> None:
    local_paths_path = tmp_path / "local_paths.yaml"
    with local_paths_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "paths": {
                    "datasets": {"libero_heng_root": "/tmp/custom_libero_root"},
                    "models": {"lingbot_va_base": "/tmp/custom_model_root"},
                    "checkpoints": {"libero_videoonly_step_850": "/tmp/custom_videoonly_transformer"},
                }
            },
            handle,
            sort_keys=False,
        )
    monkeypatch.setenv("OPEN_WAM_LOCAL_PATHS", str(local_paths_path))

    config = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml")

    assert config.data.local_root == "/tmp/custom_libero_root"
    assert config.data.empty_text_embedding_path.endswith("empty_emb.pt")
    assert config.backbone.pretrained_model_name_or_path == "/tmp/custom_model_root"
    assert config.backbone.transformer_subdir == "/tmp/custom_videoonly_transformer"


def test_local_path_registry_override_can_reference_sample_aliases(monkeypatch, tmp_path: Path) -> None:
    local_paths_path = tmp_path / "local_paths.yaml"
    with local_paths_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "paths": {
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

    assert resolved["data"]["local_root"].endswith("libero_heng/libero_10")
    assert resolved["backbone"]["pretrained_model_name_or_path"].endswith("lingbot-va-base")
    assert resolved["tests"]["derived_checkpoint"].endswith(
        "libero_heng/libero_10/derived/checkpoint_step_1/transformer"
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
    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml")

    assert isinstance(config.backbone.train_attn_mode, AttentionMode)


@pytest.mark.parametrize(
    "removed_mode",
    ("random_subwindow", "contextual_subwindow", "aligned_subwindow"),
)
def test_removed_window_sampling_modes_are_rejected(tmp_path: Path, removed_mode: str) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_joint_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["data"]["sample_construction"]["mode"] = removed_mode
    config_path = tmp_path / f"{removed_mode}.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match=rf"{removed_mode}.*WindowSamplingMode"):
        load_experiment_config(config_path)


def test_deprecated_equal_bucket_latent_temporal_layout_is_rejected(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml"
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
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml"
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
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_decoupled_same_step_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["context_condition_latent_source"] = "single_frame_condition_latent"
    raw["policy_variant"]["history_stream_visibility"] = "video_only"
    raw.setdefault("data", {}).setdefault("sample_construction", {})["condition_source_frame_offset"] = -1

    config_path = tmp_path / "parallel_single_frame_context.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert (
        config.policy_variant.context_condition_latent_source
        == ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == ParallelHistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.condition_source_frame_offset == -1


def test_parallel_stream_single_frame_context_rejects_default_condition_offset(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_decoupled_same_step_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["context_condition_latent_source"] = "single_frame_condition_latent"
    raw.setdefault("data", {}).setdefault("sample_construction", {}).pop("condition_source_frame_offset", None)

    config_path = tmp_path / "parallel_single_frame_context_leaky_default.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="single_frame_condition_latent.*condition_source_frame_offset=-1"):
        load_experiment_config(config_path)


def test_parallel_stream_per_chunk_additive_proprio_flag_loads(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_decoupled_same_step_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

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


def test_mot_per_chunk_single_frame_context_flags_load(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_decoupled_same_step_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["proprio_context_mode"] = "per_chunk_additive"
    raw["policy_variant"]["context_condition_latent_source"] = "single_frame_condition_latent"
    raw["policy_variant"]["history_stream_visibility"] = "video_only"
    raw.setdefault("data", {}).setdefault("sample_construction", {})["condition_source_frame_offset"] = -1

    config_path = tmp_path / "mot_per_chunk_single_frame_context.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == ParallelHistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True



def test_parallel_sequence_contract_expands_parallel_stream_rollout_parity_defaults(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_decoupled_same_step_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "rollout_parity_single_frame_perchunk_proprio"
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

    assert config.policy_variant.parallel_sequence_contract == (
        ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == ParallelHistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
    assert config.data.sample_construction.rollout_context_policy == RolloutContextPolicy.ONE_FRAME
    assert config.data.sample_construction.condition_source_frame_offset == -1
    assert config.data.sample_construction.context_prefix_policy == SegmentContextPolicy.NONE
    assert config.data.sample_construction.context_prefix_frames == 0


def test_parallel_sequence_contract_expands_mot_rollout_parity_defaults(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "rollout_parity_single_frame_perchunk_proprio"
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

    config_path = tmp_path / "mot_contract.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.parallel_sequence_contract == (
        ParallelSequenceContract.ROLLOUT_PARITY_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == ParallelHistoryStreamVisibility.VIDEO_ONLY
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.NEXT_AFTER_CONTEXT
    assert config.data.sample_construction.condition_source_frame_offset == -1


def test_parallel_sequence_contract_legacy_prefix_restores_target_only_sampling(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_video_then_action_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
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

    assert config.policy_variant.parallel_sequence_contract == (
        ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == ParallelHistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.LEGACY
    assert config.data.sample_construction.condition_source_frame_offset == -1
    assert config.data.sample_construction.start_padding_frames == 0


def test_parallel_sequence_contract_expands_mot_legacy_prefix_defaults(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
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

    config_path = tmp_path / "mot_legacy_prefix_contract.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.parallel_sequence_contract == (
        ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )
    assert config.policy_variant.proprio_context_mode == ProprioContextMode.PER_CHUNK_ADDITIVE
    assert (
        config.policy_variant.context_condition_latent_source
        == ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
    )
    assert config.policy_variant.history_stream_visibility == ParallelHistoryStreamVisibility.VIDEO_ONLY
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE
    assert config.policy_variant.noisy_video_condition_prob == pytest.approx(0.5)
    assert config.policy_variant.use_condition_latents is True
    assert config.policy_variant.require_condition_latents is True
    assert config.data.sample_construction.target_alignment == SampleTargetAlignment.LEGACY
    assert config.data.sample_construction.condition_source_frame_offset == -1
    assert config.data.sample_construction.start_padding_frames == 0


def test_parallel_sequence_contract_legacy_prefix_rejects_fastwam_runtime(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_fastwam_first_frame_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"

    config_path = tmp_path / "fastwam_legacy_prefix_contract.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="legacy_prefix_single_frame_perchunk_proprio.*runtime_mode"):
        load_experiment_config(config_path)


def test_parallel_sequence_contract_legacy_prefix_preserves_explicit_noisy_condition_prob(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_video_then_action_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
    raw["policy_variant"]["noisy_video_condition_prob"] = 0.0
    for key in (
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "mot_legacy_prefix_explicit_noisy_prob.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.noisy_video_condition_prob == 0.0


def test_parallel_sequence_contract_legacy_prefix_preserves_explicit_joint_coupling(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_joint_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
    raw["policy_variant"]["joint_timestep_coupling"] = "shared_video_schedule"
    for key in (
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "mot_legacy_prefix_explicit_joint_coupling.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE


def test_parallel_sequence_contract_legacy_prefix_preserves_explicit_match_sigma_joint_coupling(
    tmp_path: Path,
) -> None:
    source_path = REPO_ROOT / "configs/experiments/mot_libero_latent_local_joint_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "legacy_prefix_single_frame_perchunk_proprio"
    raw["policy_variant"]["joint_timestep_coupling"] = "match_sigma"
    for key in (
        "target_alignment",
        "rollout_context_policy",
        "start_padding_frames",
        "context_prefix_policy",
        "context_prefix_frames",
    ):
        raw["data"]["sample_construction"].pop(key, None)

    config_path = tmp_path / "mot_legacy_prefix_explicit_match_sigma_joint_coupling.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, MoTPolicyConfig)
    assert config.policy_variant.joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA


def test_parallel_sequence_contract_rejects_conflicting_explicit_values(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_m1_decoupled_same_step_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["parallel_sequence_contract"] = "rollout_parity_single_frame_perchunk_proprio"
    raw["policy_variant"]["history_stream_visibility"] = "full"

    config_path = tmp_path / "parallel_contract_conflict.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="parallel_sequence_contract=.*history_stream_visibility=video_only"):
        load_experiment_config(config_path)


def test_parallel_stream_generalist_rejects_single_frame_context_source(tmp_path: Path) -> None:
    source_path = (
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
    )
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["policy_variant"]["context_condition_latent_source"] = "single_frame_condition_latent"

    config_path = tmp_path / "generalist_single_frame_context.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="single_frame_condition_latent.*generalist_joint_denoising"):
        load_experiment_config(config_path)


def test_sample_construction_yaml_strings_are_coerced_to_enum_members(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

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
    source_path = REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
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
    source_path = REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

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
    source_path = REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
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
    source_path = REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
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
            segment_frames=128,
            chunk_size=4,
            randomize_geometry=False,
            start_padding_frames=0,
            target_alignment=SampleTargetAlignment.NEXT_AFTER_CONTEXT,
            context_prefix_frames=4,
        )


def test_hierarchical_fixed_segment_rollout_parity_rejects_malformed_chunk_size(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
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
    source_path = REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
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


def test_causal_video_prediction_mixed_video_libero_only_config_loads() -> None:
    config = load_experiment_config(
        REPO_ROOT / "configs/experiments/causal_video_prediction_mixed_video_libero_only.yaml"
    )

    assert isinstance(config.policy_variant, CausalVideoPredictionPolicyConfig)
    assert isinstance(config.data, MixedVideoDataConfig)
    assert config.backbone.load_wan_vae_frontend is True
    assert config.trainer.batch_adapter == BatchAdapterName.VIEWS
    assert config.data.sample_construction.causal_prefix_suffix_buckets[0] == CausalPrefixSuffixBucketConfig(
        observed_frames=1,
        future_frames=4,
    )
