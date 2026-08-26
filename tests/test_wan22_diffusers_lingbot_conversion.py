from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from open_wam.runtime.checkpoint_conversion import (
    SAFETENSORS_INDEX_FILENAME,
    checkpoint_safetensor_keys,
    parse_shard_size,
)


def _load_converter_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "convert_wan22_diffusers_to_lingbot_init.py"
    )
    spec = importlib.util.spec_from_file_location(
        "convert_wan22_diffusers_to_lingbot_init", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_converter_module()


def _write_checkpoint(root: Path, state: dict[str, torch.Tensor]) -> None:
    root.mkdir(parents=True)
    save_file(state, root / "diffusion_pytorch_model.safetensors")


def _tiny_lingbot_config() -> dict[str, object]:
    return {
        "patch_size": [1, 2, 2],
        "num_attention_heads": 1,
        "attention_head_dim": 2,
        "in_channels": 3,
        "out_channels": 3,
        "action_dim": 3,
        "text_dim": 2,
        "freq_dim": 2,
        "ffn_dim": 2,
        "num_layers": 1,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "rope_max_seq_len": 16,
        "pos_embed_seq_len": None,
        "attn_mode": "torch",
    }


def _states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    patch_weight = torch.arange(24, dtype=torch.float32).reshape(2, 3, 1, 2, 2)
    patch_bias = torch.tensor([1.0, 2.0])
    model = converter.load_internal_wan_transformer_class()(**_tiny_lingbot_config())
    template_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    template_state.update(
        {
            "patch_embedding.weight": torch.full_like(patch_weight, -1),
            "patch_embedding.bias": torch.full_like(patch_bias, -1),
        }
    )
    action_keys = {
        key for key in template_state if key.startswith(converter.ACTION_KEY_PREFIXES)
    }
    wan_state = {
        key: torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape)
        for key, value in template_state.items()
        if key not in action_keys and key not in converter.PATCH_MLP_SOURCE
    }
    wan_state.update(
        {
            "patch_embedding.weight": patch_weight,
            "patch_embedding.bias": patch_bias,
        }
    )
    return wan_state, template_state


def _read_output_state(root: Path) -> dict[str, torch.Tensor]:
    transformer = root / "transformer"
    state: dict[str, torch.Tensor] = {}
    for shard in transformer.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as handle:
            for key in list(handle.keys()):
                state[key] = handle.get_tensor(key)
    return state


def test_converter_preserves_every_tensor_provenance_and_index_entry(
    tmp_path: Path,
) -> None:
    wan_root = tmp_path / "wan"
    template_root = tmp_path / "template"
    output_root = tmp_path / "output"
    wan_state, template_state = _states()
    _write_checkpoint(wan_root, wan_state)
    _write_checkpoint(template_root, template_state)
    template_config = _tiny_lingbot_config()
    (template_root / "config.json").write_text(
        json.dumps(template_config), encoding="utf-8"
    )

    report = converter.convert_wan22_diffusers_to_lingbot_init(
        wan_diffusers_root=wan_root,
        lingbot_template_root=template_root,
        output_root=output_root,
        max_shard_size="64",
        dtype="float32",
    )

    output_state = _read_output_state(output_root)
    assert set(output_state) == set(template_state)
    assert torch.equal(
        output_state["blocks.0.attn1.to_q.weight"],
        wan_state["blocks.0.attn1.to_q.weight"],
    )
    assert torch.equal(
        output_state["patch_embedding_mlp.weight"],
        wan_state["patch_embedding.weight"].reshape(2, 12),
    )
    assert torch.equal(
        output_state["patch_embedding_mlp.bias"],
        wan_state["patch_embedding.bias"],
    )
    assert torch.equal(
        output_state["action_embedder.weight"],
        template_state["action_embedder.weight"],
    )
    assert report["counts"] == {
        "wan_common": len(wan_state),
        "patch_mlp": 2,
        "lingbot_only": len(
            [
                key
                for key in template_state
                if key.startswith(converter.ACTION_KEY_PREFIXES)
            ]
        ),
    }
    assert (
        json.loads(
            (output_root / "transformer" / "config.json").read_text(encoding="utf-8")
        )
        == template_config
    )

    index_path = output_root / "transformer" / SAFETENSORS_INDEX_FILENAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert set(index["weight_map"]) == set(output_state)
    for key, shard_name in index["weight_map"].items():
        with safe_open(
            output_root / "transformer" / shard_name, framework="pt"
        ) as handle:
            assert key in set(handle.keys())


def test_converter_rejects_unexpected_template_fallback(tmp_path: Path) -> None:
    wan_root = tmp_path / "wan"
    template_root = tmp_path / "template"
    wan_state, template_state = _states()
    template_state["blocks.1.weight"] = torch.ones(2, 2)
    _write_checkpoint(wan_root, wan_state)
    _write_checkpoint(template_root, template_state)
    (template_root / "config.json").write_text(
        json.dumps(_tiny_lingbot_config()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected keys=.*blocks.1.weight"):
        converter.convert_wan22_diffusers_to_lingbot_init(
            wan_diffusers_root=wan_root,
            lingbot_template_root=template_root,
            output_root=tmp_path / "output",
            dtype="float32",
        )


def test_converter_default_shard_size_materializes_derived_patch_tensors(
    tmp_path: Path,
) -> None:
    wan_root = tmp_path / "wan"
    template_root = tmp_path / "template"
    output_root = tmp_path / "output"
    wan_state, template_state = _states()
    _write_checkpoint(wan_root, wan_state)
    _write_checkpoint(template_root, template_state)
    (template_root / "config.json").write_text(
        json.dumps(_tiny_lingbot_config()),
        encoding="utf-8",
    )

    converter.convert_wan22_diffusers_to_lingbot_init(
        wan_diffusers_root=wan_root,
        lingbot_template_root=template_root,
        output_root=output_root,
        dtype="float32",
    )

    output_state = _read_output_state(output_root)
    assert torch.equal(
        output_state["patch_embedding_mlp.weight"],
        wan_state["patch_embedding.weight"].reshape(2, 12),
    )
    assert torch.equal(
        output_state["patch_embedding_mlp.bias"],
        wan_state["patch_embedding.bias"],
    )


def test_converter_rejects_matching_stale_wan_and_template_keys(
    tmp_path: Path,
) -> None:
    wan_root = tmp_path / "wan"
    template_root = tmp_path / "template"
    wan_state, template_state = _states()
    wan_state["stale.shared.weight"] = torch.ones(2, 2)
    template_state["stale.shared.weight"] = torch.ones(2, 2)
    _write_checkpoint(wan_root, wan_state)
    _write_checkpoint(template_root, template_state)
    (template_root / "config.json").write_text(
        json.dumps(_tiny_lingbot_config()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected keys=.*stale.shared.weight"):
        converter.convert_wan22_diffusers_to_lingbot_init(
            wan_diffusers_root=wan_root,
            lingbot_template_root=template_root,
            output_root=tmp_path / "output",
            dtype="float32",
        )


def test_converter_rejects_template_missing_model_state(tmp_path: Path) -> None:
    wan_root = tmp_path / "wan"
    template_root = tmp_path / "template"
    wan_state, template_state = _states()
    template_state.pop("action_proj_out.bias")
    _write_checkpoint(wan_root, wan_state)
    _write_checkpoint(template_root, template_state)
    (template_root / "config.json").write_text(
        json.dumps(_tiny_lingbot_config()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model schema.*action_proj_out.bias"):
        converter.convert_wan22_diffusers_to_lingbot_init(
            wan_diffusers_root=wan_root,
            lingbot_template_root=template_root,
            output_root=tmp_path / "output",
            dtype="float32",
        )


def test_converter_refuses_existing_output_without_touching_it(
    tmp_path: Path,
) -> None:
    wan_root = tmp_path / "wan"
    template_root = tmp_path / "template"
    output_root = tmp_path / "output"
    wan_state, template_state = _states()
    _write_checkpoint(wan_root, wan_state)
    _write_checkpoint(template_root, template_state)
    (template_root / "config.json").write_text("{}", encoding="utf-8")
    output_root.mkdir()
    sentinel = output_root / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        converter.convert_wan22_diffusers_to_lingbot_init(
            wan_diffusers_root=wan_root,
            lingbot_template_root=template_root,
            output_root=output_root,
            dtype="float32",
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(output_root.iterdir()) == [sentinel]


def test_checkpoint_index_must_match_shard_contents(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    save_file({"actual": torch.ones(1)}, root / "shard.safetensors")
    (root / SAFETENSORS_INDEX_FILENAME).write_text(
        json.dumps({"weight_map": {"claimed": "shard.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Checkpoint key/index mismatch"):
        checkpoint_safetensor_keys(root)


@pytest.mark.parametrize("size", ["0", "-1", "0GB"])
def test_shard_size_must_be_positive(size: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_shard_size(size)
