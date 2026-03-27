from __future__ import annotations

from pathlib import Path
import yaml

from open_wam.configs import (
    ParallelStreamPolicyConfig,
    PostDecodedPolicyConfig,
    PostLatentPolicyConfig,
    RegisterAttachedPolicyConfig,
)
from open_wam.utils.config_loader import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_contract_only_maps_to_post_latent() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/contract_only_robotwin.yaml")
    assert isinstance(config.policy_variant, PostLatentPolicyConfig)
    assert config.policy_variant.compatibility_mode is True
    assert config.action_decoder.name == "mlp_decoder"


def test_new_variant_yaml_configs_load() -> None:
    post_decoded = load_experiment_config(REPO_ROOT / "configs/experiments/post_decoded_robotwin.yaml")
    register = load_experiment_config(REPO_ROOT / "configs/experiments/register_attached_robotwin.yaml")
    parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin.yaml")
    smoke_parallel = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")

    assert isinstance(post_decoded.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(register.policy_variant, RegisterAttachedPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(smoke_parallel.policy_variant, ParallelStreamPolicyConfig)
    assert register.backbone.implementation == "lingbot_replica"
    assert parallel.backbone.implementation == "lingbot_replica"
    assert smoke_parallel.backbone.implementation == "lingbot_replica"
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
