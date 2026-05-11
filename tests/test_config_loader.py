from __future__ import annotations

from pathlib import Path
import pytest
import yaml

from open_wam.configs import (
    ActionChunkAnchorMode,
    ActionDecoderName,
    AuxiliaryValidationSource,
    AttentionMode,
    BatchAdapterName,
    CacheWarmupSource,
    CausalPrefixSuffixBucketConfig,
    CFGMode,
    CausalVideoPredictionPolicyConfig,
    ExportedRuntimeActionInitMode,
    GeneralistTrainingParadigm,
    JointDenoiseTrainingMode,
    JointSampler,
    LatentWindowProfile,
    LoopPolicyName,
    MoTPolicyConfig,
    MoTRuntimeMode,
    MoTPreset,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    ParallelStreamVariantProfile,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    AnchorPolicy,
    PaddedTargetPolicy,
    SampleLossWeightMode,
    SampleWeightMode,
    SegmentContextPolicy,
    TailPaddingPolicy,
    ReferenceCoreInitMode,
    RegisterAttachedPolicyConfig,
    TemporalPositionMode,
    WindowSamplingMode,
    VideoSequencePolicyConfig,
    StrategyName,
    TrainerRuntimeName,
    TrainingComponentSelector,
    TrainingObjective,
    VideoConditionInputSpace,
    VideoConditionTrainMode,
    WarmupAnchor,
)
from open_wam.utils.config_loader import load_experiment_config
from open_wam.utils.local_paths import read_yaml_with_local_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_contract_only_maps_to_post_latent() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml")
    assert isinstance(config.policy_variant, PostLatentPolicyConfig)
    assert config.policy_variant.compatibility_mode is True
    assert config.action_decoder.name == "mlp_decoder"
    assert not hasattr(config, "action_head")


def test_new_variant_yaml_configs_load() -> None:
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_robotwin.yaml")
    post_latent_video_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_robotwin_video_conditioned.yaml"
    )
    post_decoded_video_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_decoded_robotwin_video_conditioned.yaml"
    )
    mot = load_experiment_config(REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml")
    register = load_experiment_config(REPO_ROOT / "configs/experiments/register_attached_robotwin.yaml")
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin.yaml")
    smoke_parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")

    assert isinstance(post_decoded.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(post_latent_video_conditioned.policy_variant, PostLatentPolicyConfig)
    assert isinstance(post_decoded_video_conditioned.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(mot.policy_variant, MoTPolicyConfig)
    assert isinstance(register.policy_variant, RegisterAttachedPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(smoke_parallel.policy_variant, ParallelStreamPolicyConfig)
    assert post_latent_video_conditioned.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert post_decoded_video_conditioned.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert mot.policy_variant.preset == MoTPreset.FASTWAM
    assert mot.policy_variant.runtime_mode == MoTRuntimeMode.VIDEO_PREFILL_ACTION_DENOISE
    assert mot.policy_variant.condition_mode == "first_frame"
    assert mot.training.trainable_components == (TrainingComponentSelector.POLICY_VARIANT_ACTION_EXPERT,)
    assert register.backbone.implementation == "shared_transformer"
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
    assert register.inference.video_cfg_mode == "guided"
    assert register.inference.action_cfg_mode == "conditioned"
    assert register.inference.joint_cache_warmup_source == "reference_video"
    assert register.inference.joint_cache_initial_warmup_anchor == "start"
    assert register.inference.joint_cache_rollout_warmup_anchor == "end"
    assert register.inference.joint_observed_video_prefix_frames == 1
    assert register.policy_variant.structured_block_mode == "register_explicit"
    assert register.policy_variant.structured_time_layout == "video_action_state"
    assert register.policy_variant.structured_frequency_mode == "stream_local"
    assert register.policy_variant.structured_teacher_forcing_layout == "clean_prefix"
    assert register.policy_variant.structured_attention_kernel == "branchwise_explicit"
    assert register.policy_variant.structured_cache_kernel == "branchwise_rollout_explicit"
    assert register.policy_variant.stream_input_adapter_family == "structured_register_streams"
    assert register.policy_variant.stream_output_head_family == "structured_joint_flow"


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
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.policy_variant.generalist_training_paradigm == GeneralistTrainingParadigm.MIXED_DYNAMICS
    assert config.data.generalist_dynamics_mixture.train_latent_root == "/tmp/counterfactual_train/encoded_latents"
    assert config.data.generalist_dynamics_mixture.allow_train_latent_root_for_val is False
    assert config.data.generalist_dynamics_mixture.real_joint_weight == 0.6
    assert config.data.generalist_dynamics_mixture.conditional_history_frames == 8


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


def test_raw_libero_smoke_variant_yaml_configs_load() -> None:
    post_latent = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_libero_smoke.yaml")
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_libero_smoke.yaml")
    video_sequence = load_experiment_config(REPO_ROOT / "configs/experiments/video_sequence_policy_libero_smoke.yaml")
    register = load_experiment_config(REPO_ROOT / "configs/experiments/register_attached_libero_smoke.yaml")
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_raw_smoke.yaml")
    parallel_action_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_libero_action_conditioned_smoke.yaml"
    )

    assert isinstance(post_latent.policy_variant, PostLatentPolicyConfig)
    assert isinstance(post_decoded.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(video_sequence.policy_variant, VideoSequencePolicyConfig)
    assert isinstance(register.policy_variant, RegisterAttachedPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(parallel_action_conditioned.policy_variant, ParallelStreamPolicyConfig)

    assert post_latent.data.dataset_name == "libero"
    assert post_decoded.data.dataset_name == "libero"
    assert video_sequence.data.dataset_name == "libero"
    assert register.data.dataset_name == "libero"
    assert parallel.data.dataset_name == "libero"
    assert parallel_action_conditioned.data.dataset_name == "libero"

    assert register.data.action_schema.state_horizon == 3
    assert parallel.data.action_schema.action_horizon == 16
    assert parallel.action_decoder.name == ActionDecoderName.LINGBOT_PARALLEL
    assert parallel_action_conditioned.policy_variant.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    assert parallel_action_conditioned.policy_variant.video_condition_on_action is True
    assert parallel_action_conditioned.policy_variant.video_action_condition_source == "noisy_action"
    assert parallel_action_conditioned.policy_variant.video_action_attention_scope == "block_local"
    assert video_sequence.action_decoder.name == ActionDecoderName.VPP


def test_latent_libero_local_training_yaml_configs_load() -> None:
    mot = load_experiment_config(REPO_ROOT / "configs/experiments/mot_libero_latent_local.yaml")
    mot_idm = load_experiment_config(REPO_ROOT / "configs/experiments/mot_libero_latent_local_idm.yaml")
    mot_joint = load_experiment_config(REPO_ROOT / "configs/experiments/mot_libero_latent_local_joint.yaml")
    post_latent = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_libero_latent_local.yaml")
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_libero_latent_local.yaml")
    post_latent_video_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_latent_libero_latent_local_video_conditioned.yaml"
    )
    post_decoded_video_conditioned = load_experiment_config(
        REPO_ROOT / "configs/experiments/post_decoded_libero_latent_local_video_conditioned.yaml"
    )
    video_sequence = load_experiment_config(
        REPO_ROOT / "configs/experiments/video_sequence_policy_libero_latent_local.yaml"
    )
    video_sequence_random = load_experiment_config(
        REPO_ROOT / "configs/experiments/video_sequence_policy_libero_latent_local_random_subwindow.yaml"
    )
    register = load_experiment_config(REPO_ROOT / "configs/experiments/register_attached_libero_latent_local.yaml")

    assert isinstance(mot.policy_variant, MoTPolicyConfig)
    assert isinstance(mot_idm.policy_variant, MoTPolicyConfig)
    assert isinstance(mot_joint.policy_variant, MoTPolicyConfig)
    assert mot.data.dataset_type == "lerobot_v2_latent_local"
    assert mot_idm.data.dataset_type == "lerobot_v2_latent_local"
    assert mot_joint.data.dataset_type == "lerobot_v2_latent_local"
    assert post_latent.data.dataset_type == "lerobot_v2_latent_local"
    assert post_decoded.data.dataset_type == "lerobot_v2_latent_local"
    assert post_latent_video_conditioned.data.dataset_type == "lerobot_v2_latent_local"
    assert post_decoded_video_conditioned.data.dataset_type == "lerobot_v2_latent_local"
    assert video_sequence.data.dataset_type == "lerobot_v2_latent_local"
    assert video_sequence_random.data.dataset_type == "lerobot_v2_latent_local"
    assert register.data.dataset_type == "lerobot_v2_latent_local"
    assert mot.data.local_root.endswith("/libero_heng/libero_10")
    assert mot_idm.data.local_root.endswith("/libero_heng/libero_10")
    assert mot_joint.data.local_root.endswith("/libero_heng/libero_10")
    assert post_latent.data.local_root.endswith("/libero_heng/libero_10")
    assert post_decoded.data.local_root.endswith("/libero_heng/libero_10")
    assert post_latent_video_conditioned.data.local_root.endswith("/libero_heng/libero_10")
    assert post_decoded_video_conditioned.data.local_root.endswith("/libero_heng/libero_10")
    assert video_sequence.data.local_root.endswith("/libero_heng/libero_10")
    assert register.data.local_root.endswith("/libero_heng/libero_10")
    assert mot.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert mot_idm.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert mot_joint.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert post_latent.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert post_decoded.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert post_latent_video_conditioned.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert post_decoded_video_conditioned.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert video_sequence.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert video_sequence_random.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert register.trainer.batch_adapter == BatchAdapterName.LATENTS
    assert mot.trainer.strategy == StrategyName.FSDP
    assert mot_idm.trainer.strategy == StrategyName.FSDP
    assert mot_joint.trainer.strategy == StrategyName.FSDP
    assert post_latent.trainer.strategy == StrategyName.FSDP
    assert post_decoded.trainer.strategy == StrategyName.FSDP
    assert post_latent_video_conditioned.trainer.strategy == StrategyName.FSDP
    assert post_decoded_video_conditioned.trainer.strategy == StrategyName.FSDP
    assert video_sequence.trainer.strategy == StrategyName.FSDP
    assert video_sequence_random.trainer.strategy == StrategyName.FSDP
    assert register.trainer.strategy == StrategyName.FSDP
    assert mot.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert mot_idm.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert mot_joint.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert post_latent.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert post_decoded.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert post_latent_video_conditioned.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert post_decoded_video_conditioned.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert video_sequence.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert video_sequence_random.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert register.backbone.reference_core_init_mode == ReferenceCoreInitMode.VIDEO_ONLY
    assert mot.backbone.load_reference_core_weights is True
    assert mot_idm.backbone.load_reference_core_weights is True
    assert mot_joint.backbone.load_reference_core_weights is True
    assert post_latent.backbone.load_reference_core_weights is True
    assert post_decoded.backbone.load_reference_core_weights is True
    assert post_latent_video_conditioned.backbone.load_reference_core_weights is True
    assert post_decoded_video_conditioned.backbone.load_reference_core_weights is True
    assert video_sequence.backbone.load_reference_core_weights is True
    assert video_sequence_random.backbone.load_reference_core_weights is True
    assert register.backbone.load_reference_core_weights is True
    assert mot.data.sample_construction.mode == WindowSamplingMode.ALIGNED_SUBWINDOW
    assert mot_idm.data.sample_construction.mode == WindowSamplingMode.ALIGNED_SUBWINDOW
    assert mot_joint.data.sample_construction.mode == WindowSamplingMode.ALIGNED_SUBWINDOW
    assert post_latent.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert post_decoded.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert post_latent_video_conditioned.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert post_decoded_video_conditioned.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert video_sequence.data.sample_construction.mode == WindowSamplingMode.FULL_SEGMENT
    assert video_sequence_random.data.sample_construction.mode == WindowSamplingMode.ALIGNED_SUBWINDOW
    assert register.data.sample_construction.mode == WindowSamplingMode.ALIGNED_SUBWINDOW
    assert mot.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert mot_idm.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert mot_joint.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_latent.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_decoded.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_latent_video_conditioned.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_decoded_video_conditioned.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert post_latent_video_conditioned.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert post_decoded_video_conditioned.action_decoder.name == ActionDecoderName.VIDEO_CONDITIONED
    assert post_latent_video_conditioned.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT
    assert post_decoded_video_conditioned.policy_variant.video_condition_input_space == VideoConditionInputSpace.VIDEO_LATENT
    assert video_sequence.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert video_sequence_random.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert register.data.latent_window_profile == LatentWindowProfile.STANDARD_POLICY_WINDOW
    assert mot.policy_variant.preset == MoTPreset.FASTWAM
    assert mot.action_decoder.name == ActionDecoderName.MOT
    assert mot.policy_variant.condition_mode == "first_frame"
    assert mot.policy_variant.action_hidden_size == 2048
    assert mot.policy_variant.action_ffn_dim == 8192
    assert mot.training.enabled_objectives == (TrainingObjective.ACTION,)
    assert mot.training.trainable_components == (TrainingComponentSelector.POLICY_VARIANT_ACTION_EXPERT,)
    assert mot_idm.policy_variant.preset == MoTPreset.FASTWAM_IDM
    assert mot_idm.action_decoder.name == ActionDecoderName.MOT
    assert mot_idm.policy_variant.condition_mode == "teacher_forcing_cond_video"
    assert mot_idm.policy_variant.action_hidden_size == 2048
    assert mot_idm.policy_variant.action_ffn_dim == 8192
    assert mot_idm.policy_variant.teacher_forcing_video_noise_prob == 0.5
    assert mot_idm.training.enabled_objectives == (TrainingObjective.ACTION,)
    assert mot_idm.training.trainable_components == (TrainingComponentSelector.POLICY_VARIANT_ACTION_EXPERT,)
    assert mot_joint.policy_variant.preset == MoTPreset.FASTWAM_JOINT
    assert mot_joint.action_decoder.name == ActionDecoderName.MOT
    assert mot_joint.policy_variant.condition_mode == "full_video"
    assert mot_joint.policy_variant.action_hidden_size == 2048
    assert mot_joint.policy_variant.action_ffn_dim == 8192
    assert mot_joint.training.enabled_objectives == (TrainingObjective.ACTION, TrainingObjective.LATENT)
    assert mot_joint.training.trainable_components == (
        TrainingComponentSelector.POLICY_VARIANT_ACTION_EXPERT,
        TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,
    )
    assert register.data.sample_construction.anchor_policy == AnchorPolicy.RANDOM_VALID
    assert register.data.sample_construction.num_frames == 4
    assert register.data.sample_construction.action_horizon == 6
    assert register.data.sample_construction.state_horizon == 3


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


def test_video_sequence_policy_yaml_config_loads(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "video_sequence_policy_robotwin"
    raw["policy_variant"]["name"] = "video_sequence_policy"
    raw["policy_variant"]["attach_site"] = "post_visual_core"
    raw["policy_variant"].pop("pooling_mode", None)
    raw["policy_variant"].pop("query_count", None)
    raw["policy_variant"].pop("use_state_projection", None)
    raw["action_decoder"]["name"] = "vpp_decoder"
    raw["action_decoder"]["rollout_chunk_steps"] = 2
    raw["action_decoder"]["num_sampling_steps"] = 2

    config_path = tmp_path / "video_sequence_policy_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, VideoSequencePolicyConfig)
    assert config.policy_variant.name == "video_sequence_policy"
    assert config.policy_variant.attach_site == "post_visual_core"
    assert config.action_decoder.name == ActionDecoderName.VPP
    assert config.action_decoder.temporal_compression_adapter_family == "temporal_latent_resampler_3d"
    assert config.action_decoder.sequence_denoiser_family == "generic_transformer"


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


def test_video_sequence_policy_exact_vpp_knobs_load(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["name"] = "video_sequence_policy_exact_robotwin"
    raw["policy_variant"]["name"] = "video_sequence_policy"
    raw["policy_variant"]["attach_site"] = "post_visual_core"
    raw["policy_variant"].pop("pooling_mode", None)
    raw["policy_variant"].pop("query_count", None)
    raw["policy_variant"].pop("use_state_projection", None)
    raw["action_decoder"]["name"] = "vpp_decoder"
    raw["action_decoder"]["temporal_compression_adapter_family"] = "video_former_3d"
    raw["action_decoder"]["sequence_denoiser_family"] = "film_diffusion_transformer"
    raw["action_decoder"]["rollout_chunk_steps"] = 2
    raw["action_decoder"]["num_sampling_steps"] = 2

    config_path = tmp_path / "video_sequence_policy_exact_robotwin.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, VideoSequencePolicyConfig)
    assert config.action_decoder.name == ActionDecoderName.VPP
    assert config.action_decoder.temporal_compression_adapter_family == "video_former_3d"
    assert config.action_decoder.sequence_denoiser_family == "film_diffusion_transformer"


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
    assert config.action_decoder.name == ActionDecoderName.MOT


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
    assert local_libero.data.local_root.endswith("libero/libero_10")


def test_exact_local_libero_yaml_config_loads() -> None:
    exact_libero = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_local.yaml")
    assert isinstance(exact_libero.policy_variant, ParallelStreamPolicyConfig)
    assert exact_libero.policy_variant.runtime_mode == "lingbot_exact"
    assert exact_libero.policy_variant.reference_profile == "libero"
    assert exact_libero.backbone.train_attn_mode == "flex"
    assert exact_libero.backbone.infer_attn_mode == "torch"
    assert exact_libero.backbone.max_text_tokens == 512
    assert exact_libero.data.canonical_height == 128
    assert exact_libero.data.canonical_width == 256
    assert [view.canonical_name for view in exact_libero.data.view_layout] == ["image", "wrist_image"]
    assert [(view.top, view.left, view.height, view.width) for view in exact_libero.data.view_layout] == [
        (0, 0, 128, 128),
        (0, 128, 128, 128),
    ]
    assert exact_libero.data.action_schema.action_dim == 7
    assert exact_libero.action_decoder.action_dim == 30
    assert exact_libero.action_decoder.action_horizon == 16


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


def test_m1_generalist_joint_denoising_yaml_config_loads() -> None:
    config = load_experiment_config(
        REPO_ROOT
        / "configs/experiments/parallel_stream_libero_lingbot_m1_generalist_joint_denoising_heng_compatible.yaml"
    )

    assert isinstance(config.policy_variant, ParallelStreamPolicyConfig)
    assert config.policy_variant.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
    assert config.policy_variant.variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
    assert config.policy_variant.current_block_coupling == "joint"
    probs = config.policy_variant.joint_denoise_training_mode_probs
    assert probs[JointDenoiseTrainingMode.JOINT] == pytest.approx(0.6)
    assert probs[JointDenoiseTrainingMode.ACTION_CONDITIONED_VIDEO] == pytest.approx(0.2)
    assert probs[JointDenoiseTrainingMode.VIDEO_CONDITIONED_ACTION] == pytest.approx(0.2)
    assert config.action_decoder.name == ActionDecoderName.LINGBOT_PARALLEL


def test_m1_step3500_variant_yaml_configs_preserve_video_pretrain_history() -> None:
    config_paths = sorted(
        (REPO_ROOT / "configs/experiments").glob(
            "parallel_stream_libero_lingbot_m1_*_heng_compatible.yaml"
        )
    )
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


def test_sample_construction_yaml_strings_are_coerced_to_enum_members(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/register_attached_libero_latent_local.yaml"
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
        "sample_weight_length_power": 0.5,
        "sample_weight_min": 0.25,
        "sample_weight_max": 4.0,
    }

    config_path = tmp_path / "register_attached_sample_construction.yaml"
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
    assert config.data.sample_construction.sample_weight_length_power == pytest.approx(0.5)
    assert config.data.sample_construction.sample_weight_min == pytest.approx(0.25)
    assert config.data.sample_construction.sample_weight_max == pytest.approx(4.0)


def test_hierarchical_fixed_segment_sample_construction_loads_explicit_sampler_fields(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/register_attached_libero_latent_local.yaml"
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

    config_path = tmp_path / "register_attached_hierarchical_fixed_segment.yaml"
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


def test_hierarchical_fixed_segment_rejects_legacy_full_segment_flag(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/register_attached_libero_latent_local.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "hierarchical_fixed_segment",
        "segment_frames": 128,
        "require_full_segment": False,
    }

    config_path = tmp_path / "register_attached_bad_hierarchical_fixed_segment.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    with pytest.raises(ValueError, match="remove legacy fields: `require_full_segment`"):
        load_experiment_config(config_path)


def test_contextual_sample_construction_and_temporal_position_mode_load(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_joint_denoise_heng_compatible.yaml"
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw.setdefault("data", {})
    raw["data"]["sample_construction"] = {
        "mode": "contextual_subwindow",
        "num_frames": 4,
        "action_horizon": 16,
        "state_horizon": 1,
        "frame_stride": 1,
        "chunk_size": 4,
        "window_size": 64,
        "predict_blocks_per_sample": 1,
        "randomize_geometry": False,
    }
    raw.setdefault("policy_variant", {})
    raw["policy_variant"]["temporal_position_mode"] = "global_shifted"

    config_path = tmp_path / "parallel_stream_contextual.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    config = load_experiment_config(config_path)

    assert config.data.sample_construction.mode == WindowSamplingMode.CONTEXTUAL_SUBWINDOW
    assert config.data.sample_construction.chunk_size == 4
    assert config.data.sample_construction.window_size == 64
    assert config.data.sample_construction.predict_blocks_per_sample == 1
    assert config.data.sample_construction.randomize_geometry is False
    assert config.policy_variant.temporal_position_mode == TemporalPositionMode.GLOBAL_SHIFTED


def test_legacy_method2_runtime_fields_still_map_to_generic_runtime_config(tmp_path: Path) -> None:
    source_path = REPO_ROOT / "configs/experiments/register_attached_robotwin_smoke.yaml"
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

    legacy_path = tmp_path / "legacy_register.yaml"
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
