from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import torch
from safetensors.torch import save_file


def _load_converter_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "convert_wan22_to_lingbot_init.py"
    spec = importlib.util.spec_from_file_location("convert_wan22_to_lingbot_init", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_converter_module()


def test_wan22_to_lingbot_map_covers_expected_key_families() -> None:
    mapping = converter.build_wan22_to_lingbot_map(num_layers=2)

    assert len(mapping) == 15 + 2 * 27
    assert mapping["patch_embedding.weight"][0] == "patch_embedding_mlp.weight"
    assert mapping["head.head.weight"][0] == "proj_out.weight"
    assert mapping["head.modulation"][0] == "scale_shift_table"
    assert mapping["blocks.1.self_attn.q.weight"][0] == "blocks.1.attn1.to_q.weight"
    assert mapping["blocks.1.cross_attn.o.bias"][0] == "blocks.1.attn2.to_out.0.bias"
    assert mapping["blocks.1.ffn.0.weight"][0] == "blocks.1.ffn.net.0.proj.weight"
    assert mapping["blocks.1.norm3.bias"][0] == "blocks.1.norm2.bias"
    assert mapping["blocks.1.modulation"][0] == "blocks.1.scale_shift_table"


def test_lingbot_config_is_derived_from_raw_wan_config() -> None:
    lingbot_config = converter.build_lingbot_config_from_wan_config(
        {
            "dim": 3072,
            "num_heads": 24,
            "in_dim": 48,
            "out_dim": 48,
            "freq_dim": 256,
            "ffn_dim": 14336,
            "num_layers": 30,
            "eps": 1e-6,
        },
        action_dim=7,
        attn_mode="torch",
    )

    assert lingbot_config["_class_name"] == "WanTransformer3DModel"
    assert lingbot_config["attention_head_dim"] == 128
    assert lingbot_config["action_dim"] == 7
    assert lingbot_config["attn_mode"] == "torch"
    assert lingbot_config["patch_size"] == [1, 2, 2]


def _raw_tensor_for_key(key: str) -> torch.Tensor:
    if key == "patch_embedding.weight":
        return torch.arange(24, dtype=torch.float32).reshape(2, 3, 1, 2, 2)
    if key == "head.modulation":
        return torch.full((1, 2, 2), 3.0)
    if key.endswith("modulation"):
        return torch.full((1, 6, 2), 4.0)
    if key.endswith("bias") or key.endswith("norm_q.weight") or key.endswith("norm_k.weight"):
        return torch.arange(2, dtype=torch.float32)
    return torch.arange(4, dtype=torch.float32).reshape(2, 2)


def test_convert_wan22_to_lingbot_init_uses_remapped_weights_and_keeps_extra_keys_random(
    tmp_path: Path,
    monkeypatch,
) -> None:
    wan_root = tmp_path / "raw_wan"
    output_root = tmp_path / "wan22_as_lingbot"
    wan_root.mkdir()
    (wan_root / "config.json").write_text(
        json.dumps(
            {
                "dim": 2,
                "num_heads": 1,
                "in_dim": 3,
                "out_dim": 3,
                "freq_dim": 2,
                "ffn_dim": 2,
                "num_layers": 1,
                "eps": 1e-6,
            }
        ),
        encoding="utf-8",
    )
    mapping = converter.build_wan22_to_lingbot_map(num_layers=1)
    raw_state = {key: _raw_tensor_for_key(key) for key in mapping}
    save_file(raw_state, wan_root / "diffusion_pytorch_model.safetensors")

    class FakeWanTransformer3DModel:
        last_instance = None

        def __init__(self, **kwargs):
            del kwargs
            FakeWanTransformer3DModel.last_instance = self
            self.loaded_state = None
            self._state = {}
            for raw_key, (target_key, transform) in mapping.items():
                raw_value = raw_state[raw_key]
                if raw_key == "patch_embedding.weight":
                    self._state[target_key] = torch.zeros(raw_value.shape[0], raw_value.numel() // raw_value.shape[0])
                else:
                    self._state[target_key] = torch.zeros_like(raw_value)
            self._state["action_embedder.weight"] = torch.ones(2, 7)

        def to(self, **kwargs):
            del kwargs
            return self

        def state_dict(self):
            return dict(self._state)

        def load_state_dict(self, state, strict=True):
            assert strict is True
            self.loaded_state = dict(state)

        def save_pretrained(self, path, **kwargs):
            del kwargs
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "fake_model.txt").write_text("saved", encoding="utf-8")

    monkeypatch.setattr(converter, "load_internal_wan_transformer_class", lambda: FakeWanTransformer3DModel)

    report = converter.convert_wan22_to_lingbot_init(
        wan_root=wan_root,
        output_root=output_root,
        action_dim=7,
        attn_mode="torch",
    )

    instance = FakeWanTransformer3DModel.last_instance
    assert instance is not None
    assert torch.equal(
        instance.loaded_state["patch_embedding_mlp.weight"],
        raw_state["patch_embedding.weight"].reshape(2, -1),
    )
    assert torch.equal(instance.loaded_state["proj_out.weight"], raw_state["head.head.weight"])
    assert torch.equal(instance.loaded_state["blocks.0.attn1.to_q.weight"], raw_state["blocks.0.self_attn.q.weight"])
    assert torch.equal(instance.loaded_state["action_embedder.weight"], torch.ones(2, 7))
    assert report["mapped_raw_keys"] == len(mapping)
    assert report["left_random_lingbot_keys"] == ["action_embedder.weight"]
    assert (output_root / "transformer" / "fake_model.txt").exists()
    assert (output_root / "wan22_to_lingbot_init_report.json").exists()
