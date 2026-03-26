from __future__ import annotations

from pathlib import Path

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
    replica_parallel = load_experiment_config(
        REPO_ROOT / "configs/experiments/parallel_stream_robotwin_lingbot_replica.yaml"
    )

    assert isinstance(post_decoded.policy_variant, PostDecodedPolicyConfig)
    assert isinstance(register.policy_variant, RegisterAttachedPolicyConfig)
    assert isinstance(parallel.policy_variant, ParallelStreamPolicyConfig)
    assert isinstance(replica_parallel.policy_variant, ParallelStreamPolicyConfig)
    assert replica_parallel.backbone.implementation == "lingbot_replica"
    assert replica_parallel.policy_variant.runtime_mode == "lingbot_exact"
    assert replica_parallel.action_decoder.name == "lingbot_parallel_decoder"
    assert replica_parallel.backbone.reference_model_path is None


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
