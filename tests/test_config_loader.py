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
