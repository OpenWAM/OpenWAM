from __future__ import annotations

from dataclasses import fields

from open_wam.configs import CheckpointMode, LoopPolicyName
from open_wam.sdk.config import (
    ExperimentConfig,
    ExtensionActionDecoderConfig,
    ExtensionPolicyConfig,
    load_experiment_config,
)
from open_wam.sdk.data import (
    DatasetArtifactKind,
    DatasetArtifactRequirement,
    WAMSample,
    preflight_dataset_artifacts,
    register_dataset_adapter,
)
from open_wam.sdk.policy import (
    ActionDecoder,
    PolicyVariant,
    PreparedAttentionProfile,
    RuntimeStepInput,
    VisualCoreInput,
    register_action_decoder,
    register_policy_variant,
)
from open_wam.sdk.results import build_result_envelope, write_result_json
from open_wam.sdk.simulator import (
    SimulatorBackend,
    SimulatorFactoryContext,
    register_simulator_adapter,
)


def test_role_specific_sdk_exposes_extension_contracts() -> None:
    assert ExperimentConfig.__module__ == "open_wam.configs.experiment"
    assert ExtensionPolicyConfig.__module__ == "open_wam.configs.policy_contracts"
    assert (
        ExtensionActionDecoderConfig.__module__
        == "open_wam.configs.action_decoder"
    )
    assert WAMSample.__module__ == "open_wam.data.contracts"
    assert DatasetArtifactKind.__module__ == "open_wam.data.artifacts"
    assert DatasetArtifactRequirement.__module__ == "open_wam.data.artifacts"
    assert PolicyVariant.__module__ == "open_wam.models.policy_variants.base"
    assert ActionDecoder.__module__ == "open_wam.models.action_decoders.base"
    assert PreparedAttentionProfile.__module__.endswith("attention_contracts")
    assert RuntimeStepInput.__module__.endswith("runtime_programs")
    assert VisualCoreInput.__module__.endswith("visual_tower.contracts")
    assert SimulatorBackend.__module__ == "open_wam.simulators.contracts"
    assert callable(register_dataset_adapter)
    assert callable(preflight_dataset_artifacts)
    assert callable(register_policy_variant)
    assert callable(register_action_decoder)
    assert callable(register_simulator_adapter)
    assert callable(build_result_envelope)
    assert callable(write_result_json)


def test_simulator_factory_context_does_not_expose_cli_arguments() -> None:
    assert tuple(field.name for field in fields(SimulatorFactoryContext)) == (
        "benchmark",
        "options",
        "local_paths",
    )


def test_packaged_extension_template_uses_typed_extension_envelopes() -> None:
    config = load_experiment_config("templates/extension_method/config.yaml")

    assert config.name == "extension_method_template"
    assert isinstance(config.policy_variant, ExtensionPolicyConfig)
    assert isinstance(config.action_decoder, ExtensionActionDecoderConfig)
    assert config.policy_variant.extension_type == "example.policy"
    assert config.action_decoder.extension_type == "example.decoder"


def test_public_tiny_config_owns_the_documented_checkpoint_lifecycle() -> None:
    config = load_experiment_config(
        "configs/examples/public_tiny_synthetic_contract.yaml"
    )

    assert config.training.num_steps == 1
    assert config.trainer.loop_policy is LoopPolicyName.STEPS
    assert config.trainer.enable_checkpointing is True
    assert config.trainer.save_interval == 1
    assert config.trainer.checkpoint_mode is CheckpointMode.FULL_TRAINING_STATE
