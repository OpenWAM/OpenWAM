from __future__ import annotations

from pathlib import Path

import pytest

from open_wam.configs import (
    expand_video_action_sequence_contract,
    load_experiment_config,
    parse_action_decoder_config,
    parse_data_config,
    parse_inference_config,
    parse_policy_variant_config,
    parse_shared_video_transformer_config,
    parse_trainer_config,
    parse_training_config,
    parse_validation_config,
    read_yaml_with_local_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGS = tuple(
    sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
)


@pytest.mark.parametrize(
    "config_path",
    EXPERIMENT_CONFIGS,
    ids=lambda path: path.stem,
)
def test_component_parsers_match_full_experiment_loading(config_path: Path) -> None:
    raw = expand_video_action_sequence_contract(read_yaml_with_local_paths(config_path))
    config = load_experiment_config(config_path)

    assert parse_data_config(raw.get("data")) == config.data
    assert parse_shared_video_transformer_config(raw.get("backbone")) == config.backbone
    assert parse_training_config(raw.get("training")) == config.training
    assert parse_inference_config(raw.get("inference")) == config.inference
    assert parse_trainer_config(raw.get("trainer")) == config.trainer
    assert parse_validation_config(raw.get("validation")) == config.validation
    parsed_policy = parse_policy_variant_config(
        policy_variant_raw=raw.get("policy_variant", {}),
        action_head_raw=raw.get("action_head", {}),
        data_config=config.data,
        backbone_config=config.backbone,
        training_config=config.training,
        inference_config=config.inference,
    )
    assert parsed_policy == config.policy_variant
    assert (
        parse_action_decoder_config(
            action_decoder_raw=raw.get("action_decoder", {}),
            policy_variant_config=parsed_policy,
            data_config=config.data,
            backbone_config=config.backbone,
        )
        == config.action_decoder
    )
