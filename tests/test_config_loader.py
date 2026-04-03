from __future__ import annotations

from pathlib import Path
import yaml

from open_wam.configs import (
    ActionDecoderName,
    AttentionMode,
    BatchAdapterName,
    CacheWarmupSource,
    CFGMode,
    JointSampler,
    LoopPolicyName,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    RegisterAttachedPolicyConfig,
    VideoSequencePolicyConfig,
    StrategyName,
    TrainerRuntimeName,
    TrainingComponentSelector,
    TrainingObjective,
    WarmupAnchor,
)
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_contract_only_maps_to_post_latent() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml")
    assert isinstance(config.policy_variant, PostLatentPolicyConfig)
    assert config.policy_variant.compatibility_mode is True
    assert config.action_decoder.name == "mlp_decoder"
    assert not hasattr(config, "action_head")


def test_new_variant_yaml_configs_load() -> None:
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_robotwin.yaml")
    register = load_experiment_config(REPO_ROOT / "configs/experiments/register_attached_robotwin.yaml")
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin.yaml")
    smoke_parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")

    assert isinstance(post_decoded.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(register.policy_variant, RegisterAttachedPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(smoke_parallel.policy_variant, ParallelStreamPolicyConfig)
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


def test_raw_libero_smoke_variant_yaml_configs_load() -> None:
    post_latent = load_experiment_config(REPO_ROOT / "configs/experiments/post_latent_libero_smoke.yaml")
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_libero_smoke.yaml")
    video_sequence = load_experiment_config(REPO_ROOT / "configs/experiments/video_sequence_policy_libero_smoke.yaml")
    register = load_experiment_config(REPO_ROOT / "configs/experiments/register_attached_libero_smoke.yaml")
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_raw_smoke.yaml")

    assert isinstance(post_latent.policy_variant, PostLatentPolicyConfig)
    assert isinstance(post_decoded.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(video_sequence.policy_variant, VideoSequencePolicyConfig)
    assert isinstance(register.policy_variant, RegisterAttachedPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)

    assert post_latent.data.dataset_name == "libero"
    assert post_decoded.data.dataset_name == "libero"
    assert video_sequence.data.dataset_name == "libero"
    assert register.data.dataset_name == "libero"
    assert parallel.data.dataset_name == "libero"

    assert register.data.action_schema.state_horizon == 3
    assert parallel.data.action_schema.action_horizon == 16
    assert parallel.action_decoder.name == ActionDecoderName.LINGBOT_PARALLEL
    assert video_sequence.action_decoder.name == ActionDecoderName.VPP


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


def test_local_libero_yaml_config_loads() -> None:
    local_libero = load_experiment_config(REPO_ROOT / "configs/experiments/contract_only_libero_local.yaml")
    assert local_libero.data.dataset_name == "libero"
    assert local_libero.data.dataset_type == "libero_hdf5"
    assert local_libero.data.repo_id is None
    assert local_libero.data.local_root == "/path/to/datasets"


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
    assert heng_libero.data.local_root == "/path/to/private-resource"
    assert heng_libero.data.empty_text_embedding_path == (
        "/path/to/private-resource"
    )
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
    assert heng_libero.training.learning_rate == 1e-5
    assert heng_libero.training.gradient_accumulation_steps == 10
    assert heng_libero.training.num_steps == 5000
    assert heng_libero.training.enabled_objectives == ("latent", "action")
    assert heng_libero.training.action_loss_weight == 1.0
    assert heng_libero.training.trainable_components == ("visual_tower.runtime_backbone",)
    assert heng_libero.trainer.runtime == "composable"
    assert heng_libero.trainer.batch_adapter == "latents"
    assert heng_libero.trainer.loop_policy == "steps"
    assert heng_libero.trainer.strategy == "fsdp"
    assert heng_libero.trainer.save_interval == 100
    assert heng_libero.trainer.enable_wandb is True
    assert heng_libero.trainer.wandb_project == "lingbot-va-posttrain-libero_openwam"


def test_loaded_enum_like_fields_are_real_enum_members() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_libero_lingbot_exact_heng_compatible.yaml")

    assert isinstance(config.backbone.train_attn_mode, AttentionMode)
    assert isinstance(config.backbone.infer_attn_mode, AttentionMode)
    assert isinstance(config.policy_variant.runtime_mode, ParallelRuntimeMode)
    assert isinstance(config.action_decoder.name, ActionDecoderName)
    assert isinstance(config.inference.joint_sampler, JointSampler)
    assert isinstance(config.inference.video_cfg_mode, CFGMode)
    assert isinstance(config.inference.action_cfg_mode, CFGMode)
    assert isinstance(config.inference.joint_cache_warmup_source, CacheWarmupSource)
    assert isinstance(config.inference.joint_cache_initial_warmup_anchor, WarmupAnchor)
    assert isinstance(config.inference.joint_cache_rollout_warmup_anchor, WarmupAnchor)
    assert config.training.enabled_objectives == (
        TrainingObjective.LATENT,
        TrainingObjective.ACTION,
    )
    assert config.training.trainable_components == (TrainingComponentSelector.VISUAL_TOWER_RUNTIME_BACKBONE,)
    assert isinstance(config.trainer.runtime, TrainerRuntimeName)
    assert isinstance(config.trainer.batch_adapter, BatchAdapterName)
    assert isinstance(config.trainer.loop_policy, LoopPolicyName)
    assert isinstance(config.trainer.strategy, StrategyName)


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
    raw["training"]["trainable_components"] = ["visual_tower.core", "action_decoder"]
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
    assert config.training.trainable_components == ("visual_tower.core", "action_decoder")
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
