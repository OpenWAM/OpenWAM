from __future__ import annotations

from importlib import import_module
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from open_wam.configs import (
    ExtensionActionDecoderConfig,
    ExtensionPolicyConfig,
    GenericDataConfig,
)
from open_wam.data import build_synthetic_batch, build_train_val_datasets
from open_wam.extensions import (
    DEFAULT_EXTENSION_HOOK,
    load_extension_module,
    load_extension_modules,
    loaded_extensions,
)
from open_wam.pipelines import (
    build_action_decoder,
    build_policy_variant,
    build_variant_pipeline_from_config,
    registered_action_decoders,
    registered_policy_variants,
)
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch
from open_wam.configs import load_experiment_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_extension(tmp_path: Path, source: str) -> str:
    module_name = f"open_wam_test_extension_{uuid4().hex}"
    (tmp_path / f"{module_name}.py").write_text(source, encoding="utf-8")
    return module_name


@pytest.mark.unit
def test_default_extension_hook_runs_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = _write_extension(
        tmp_path,
        "calls = []\n"
        "def register_open_wam():\n"
        "    calls.append('registered')\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    first = load_extension_module(module_name)
    second = load_extension_module(module_name)

    assert first is second
    assert first.module_name == module_name
    assert first.hook_name == DEFAULT_EXTENSION_HOOK
    assert first.spec == f"{module_name}:{DEFAULT_EXTENSION_HOOK}"
    assert import_module(module_name).calls == ["registered"]
    assert first in loaded_extensions()


@pytest.mark.unit
def test_custom_extension_hook_and_operator_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_name = _write_extension(
        tmp_path,
        "calls = []\n"
        "def install():\n"
        "    calls.append('first')\n",
    )
    second_name = _write_extension(
        tmp_path,
        "calls = []\n"
        "def register_open_wam():\n"
        "    calls.append('second')\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    loaded = load_extension_modules((f"{first_name}:install", second_name))

    assert [item.module_name for item in loaded] == [first_name, second_name]
    assert import_module(first_name).calls == ["first"]
    assert import_module(second_name).calls == ["second"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "suffix", "error_type", "message"),
    [
        ("value = 1\n", "", AttributeError, "has no registration hook"),
        (
            "register_open_wam = 1\n",
            "",
            TypeError,
            "must be callable",
        ),
        (
            "def register_open_wam():\n    pass\n",
            ":",
            ValueError,
            "missing a hook name",
        ),
    ],
)
def test_invalid_extension_contracts_are_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    suffix: str,
    error_type: type[Exception],
    message: str,
) -> None:
    module_name = _write_extension(tmp_path, source)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(error_type, match=message):
        load_extension_module(f"{module_name}{suffix}")


@pytest.mark.unit
def test_out_of_tree_extension_registers_dataset_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_type = f"external_dataset_{uuid4().hex}"
    module_name = _write_extension(
        tmp_path,
        "from torch.utils.data import Dataset\n"
        "from open_wam.data import register_dataset_adapter\n"
        "\n"
        "class MarkerDataset(Dataset):\n"
        "    def __init__(self, marker):\n"
        "        self.marker = marker\n"
        "    def __len__(self):\n"
        "        return 1\n"
        "    def __getitem__(self, index):\n"
        "        raise IndexError(index)\n"
        "\n"
        "def build(config):\n"
        "    prefix = config.adapter_options['prefix']\n"
        "    return MarkerDataset(prefix + '_train'), MarkerDataset(prefix + '_val')\n"
        "\n"
        "def register_open_wam():\n"
        f"    register_dataset_adapter({dataset_type!r}, raw_builder=build)\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    load_extension_module(module_name)
    train, val = build_train_val_datasets(
        GenericDataConfig(
            dataset_name="external",
            dataset_type=dataset_type,
            adapter_options={"prefix": "custom"},
        )
    )

    assert train.marker == "custom_train"
    assert val.marker == "custom_val"


@pytest.mark.unit
def test_out_of_tree_extension_loads_typed_policy_and_decoder_from_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_type = f"external_policy_{uuid4().hex}"
    decoder_type = f"external_decoder_{uuid4().hex}"
    module_name = _write_extension(
        tmp_path,
        "from open_wam.pipelines import register_action_decoder, register_policy_variant\n"
        "\n"
        "def build_policy(config):\n"
        "    policy = config.policy_variant\n"
        "    return ('policy', policy.extension_type, dict(policy.options))\n"
        "\n"
        "def build_decoder(config):\n"
        "    decoder = config.action_decoder\n"
        "    return ('decoder', decoder.extension_type, dict(decoder.options))\n"
        "\n"
        "def register_open_wam():\n"
        f"    register_policy_variant({policy_type!r}, build_policy)\n"
        f"    register_action_decoder({decoder_type!r}, build_decoder)\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    raw = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml").read_text(encoding="utf-8")
    )
    raw["policy_variant"] = {
        "name": "extension",
        "extension_type": policy_type,
        "hidden_size": 256,
        "attach_site": "post_visual_core",
        "options": {"attention_profile": "acme.block_sparse", "width": 32},
    }
    raw["action_decoder"] = {
        "name": "extension",
        "extension_type": decoder_type,
        "hidden_size": 256,
        "action_dim": 30,
        "action_horizon": 8,
        "options": {"loss": "smooth_l1"},
    }
    config_path = tmp_path / "extension_experiment.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    load_extension_module(module_name)
    config = load_experiment_config(config_path)

    assert isinstance(config.policy_variant, ExtensionPolicyConfig)
    assert isinstance(config.action_decoder, ExtensionActionDecoderConfig)
    assert config.policy_variant.options == {
        "attention_profile": "acme.block_sparse",
        "width": 32,
    }
    assert config.action_decoder.options == {"loss": "smooth_l1"}
    assert policy_type in registered_policy_variants()
    assert decoder_type in registered_action_decoders()
    with pytest.raises(TypeError, match="expected an .*PolicyVariant"):
        build_policy_variant(config)
    with pytest.raises(TypeError, match="expected an .*ActionDecoder"):
        build_action_decoder(config)


@pytest.mark.unit
def test_extension_factory_error_explains_registration_order(tmp_path: Path) -> None:
    missing_type = f"missing_policy_{uuid4().hex}"
    raw = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/mot_robotwin_smoke.yaml").read_text(encoding="utf-8")
    )
    raw["policy_variant"] = {
        "name": "extension",
        "extension_type": missing_type,
        "hidden_size": 256,
        "attach_site": "post_visual_core",
    }
    config_path = tmp_path / "missing_extension.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_experiment_config(config_path)

    with pytest.raises(ValueError, match=r"--extension module\[:hook\]"):
        build_policy_variant(config)


@pytest.mark.smoke
def test_out_of_tree_policy_and_decoder_run_full_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_type = f"external_runtime_policy_{uuid4().hex}"
    decoder_type = f"external_runtime_decoder_{uuid4().hex}"
    module_name = _write_extension(
        tmp_path,
        "from open_wam.configs import PostLatentPolicyConfig\n"
        "from open_wam.models.action_decoders import MLPActionDecoder\n"
        "from open_wam.models.policy_variants import PostLatentPolicyVariant\n"
        "from open_wam.pipelines import register_action_decoder, register_policy_variant\n"
        "\n"
        "def build_policy(experiment):\n"
        "    extension = experiment.policy_variant\n"
        "    policy_config = PostLatentPolicyConfig(\n"
        "        hidden_size=extension.hidden_size,\n"
        "        attach_site=extension.attach_site,\n"
        "        use_state_projection=bool(extension.options['use_state_projection']),\n"
        "    )\n"
        "    return PostLatentPolicyVariant(\n"
        "        config=policy_config,\n"
        "        training_config=experiment.training,\n"
        "        inference_config=experiment.inference,\n"
        "        action_horizon=experiment.data.action_schema.action_horizon,\n"
        "        state_dim=experiment.data.action_schema.state_dim,\n"
        "    )\n"
        "\n"
        "def build_decoder(experiment):\n"
        "    config = experiment.action_decoder\n"
        "    return MLPActionDecoder(\n"
        "        hidden_size=config.hidden_size,\n"
        "        action_dim=config.action_dim,\n"
        "        action_horizon=config.action_horizon,\n"
        "        training_config=experiment.training,\n"
        "        inference_config=experiment.inference,\n"
        "        dropout=config.dropout,\n"
        "    )\n"
        "\n"
        "def register_open_wam():\n"
        f"    register_policy_variant({policy_type!r}, build_policy)\n"
        f"    register_action_decoder({decoder_type!r}, build_decoder)\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    raw = yaml.safe_load(
        (REPO_ROOT / "configs/experiments/post_latent_robotwin.yaml").read_text(encoding="utf-8")
    )
    raw["policy_variant"] = {
        "name": "extension",
        "extension_type": policy_type,
        "hidden_size": 256,
        "attach_site": "post_visual_core",
        "options": {"use_state_projection": True},
    }
    raw["action_decoder"] = {
        "name": "extension",
        "extension_type": decoder_type,
        "hidden_size": 256,
        "action_dim": 30,
        "action_horizon": 6,
    }
    raw["inference"]["video_num_inference_steps"] = 1
    raw["inference"]["action_num_inference_steps"] = 1
    config_path = tmp_path / "extension_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    load_extension_module(module_name)
    config = load_experiment_config(config_path)
    pipeline = build_variant_pipeline_from_config(config)
    batch = build_synthetic_batch(config.data, batch_size=1)
    train_batch = PolicyTrainBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
        extra={"task_text": batch.task_text},
    )

    train_output = pipeline.forward_train(batch.views, train_batch)
    train_output.decoder_output.loss.backward()
    infer_output = pipeline.forward_infer_step(
        batch.views,
        PolicyInferContext(state=batch.state, extra={"task_text": batch.task_text}),
    )

    assert train_output.decoder_output.action_pred.shape == (1, 6, 30)
    assert infer_output.decoder_output.action_pred.shape == (1, 6, 30)
    assert any(parameter.grad is not None for parameter in pipeline.policy_variant.parameters())
    assert any(parameter.grad is not None for parameter in pipeline.action_decoder.parameters())
