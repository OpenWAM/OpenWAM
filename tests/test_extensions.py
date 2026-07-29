from __future__ import annotations

from importlib import import_module
from pathlib import Path
from uuid import uuid4

import pytest

from open_wam.configs import GenericDataConfig
from open_wam.data import build_train_val_datasets
from open_wam.extensions import (
    DEFAULT_EXTENSION_HOOK,
    load_extension_module,
    load_extension_modules,
    loaded_extensions,
)


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
