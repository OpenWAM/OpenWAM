#!/usr/bin/env python
"""Build a LingBot-format init from the *Diffusers* release of Wan2.2 plus a LingBot template.

This is the sibling of ``convert_wan22_to_lingbot_init.py`` and exists because the
two Wan2.2 releases need different work:

* The **raw** ``Wan2.2-TI2V-5B`` checkpoint uses upstream module names
  (``blocks.i.self_attn.*``, ``blocks.i.ffn.0/2.*``, ``head.*``), so it needs the
  full rename table in the sibling script, and it has nothing to fill the
  LingBot-only action stream with -- that script initialises those randomly.
* The **Diffusers** conversion of the same model already uses LingBot's own
  names: all 825 of its keys match LingBot keys verbatim, with identical shapes.
  No rename table is needed. What it still lacks is the action stream, and for
  that a LingBot template checkpoint supplies real weights instead of noise.

So the output is a hybrid, and for a 5B TI2V model the three groups are:

    825 keys  Wan2.2-Diffusers tensor, cast to the target dtype
     14 keys  copied from the LingBot template (action embedder / projection /
              condition_embedder_action.*)
      2 keys  patch_embedding_mlp, reshaped from Wan's Conv3d patch_embedding
              (weight ``(3072, 48, 1, 2, 2) -> (3072, 192)``, bias verbatim)

The **template's** key set defines the output, not a model instance: LingBot
checkpoints carry ``patch_embedding.{weight,bias}`` alongside the flattened
``patch_embedding_mlp.*``, while the transformer class only declares the latter.
Deriving keys from the model would therefore drop two keys that released
artifacts contain.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from open_wam.models.visual_tower.reference_loader import (
    load_internal_wan_transformer_class,
)
from open_wam.runtime.checkpoint_conversion import (
    conv3d_to_linear,
    load_safetensors_state,
    model_state_shapes,
    parse_shard_size,
    parse_torch_dtype,
    save_sharded_safetensors,
)
from open_wam.runtime.publication import staged_output_directory

# Flattened views of the Conv3d patch embedding; both live in a LingBot checkpoint.
PATCH_MLP_SOURCE = {
    "patch_embedding_mlp.weight": "patch_embedding.weight",
    "patch_embedding_mlp.bias": "patch_embedding.bias",
}
ACTION_KEY_PREFIXES = (
    "action_embedder.",
    "action_proj_out.",
    "condition_embedder_action.",
)


def _validate_lingbot_template_schema(
    config: dict[str, object],
    state: dict[str, torch.Tensor],
) -> None:
    required_shapes = model_state_shapes(
        load_internal_wan_transformer_class(),
        config,
    )
    expected_keys = set(required_shapes) | set(PATCH_MLP_SOURCE.values())
    missing = sorted(expected_keys - set(state))
    unexpected = sorted(set(state) - expected_keys)
    shape_mismatches = sorted(
        key
        for key, expected_shape in required_shapes.items()
        if key in state and tuple(state[key].shape) != expected_shape
    )
    if missing or unexpected or shape_mismatches:
        mismatch_preview = [
            (
                key,
                tuple(state[key].shape),
                required_shapes[key],
            )
            for key in shape_mismatches[:10]
        ]
        raise ValueError(
            "LingBot template does not satisfy the model schema declared by config.json: "
            f"missing keys={missing[:10]}, unexpected keys={unexpected[:10]}, "
            f"shape mismatches={mismatch_preview}."
        )


def convert_wan22_diffusers_to_lingbot_init(
    *,
    wan_diffusers_root: Path,
    lingbot_template_root: Path,
    output_root: Path,
    max_shard_size: str = "5GB",
    dtype: str = "bfloat16",
) -> dict[str, object]:
    with staged_output_directory(output_root) as staging_root:
        return _convert_wan22_diffusers_to_lingbot_init(
            wan_diffusers_root=wan_diffusers_root,
            lingbot_template_root=lingbot_template_root,
            output_root=output_root,
            staging_root=staging_root,
            max_shard_size=max_shard_size,
            dtype=dtype,
        )


def _convert_wan22_diffusers_to_lingbot_init(
    *,
    wan_diffusers_root: Path,
    lingbot_template_root: Path,
    output_root: Path,
    staging_root: Path,
    max_shard_size: str,
    dtype: str,
) -> dict[str, object]:
    target_dtype = parse_torch_dtype(dtype)
    max_shard_bytes = parse_shard_size(max_shard_size)

    template_config = lingbot_template_root / "config.json"
    if not template_config.exists():
        raise FileNotFoundError(f"Missing LingBot template config: {template_config}")

    wan_state = load_safetensors_state(wan_diffusers_root)
    template_state = load_safetensors_state(lingbot_template_root)
    lingbot_config = json.loads(template_config.read_text(encoding="utf-8"))
    if not isinstance(lingbot_config, dict):
        raise TypeError(
            f"LingBot template config must be a JSON object: {template_config}"
        )
    _validate_lingbot_template_schema(lingbot_config, template_state)

    unexpected = sorted(set(wan_state) - set(template_state))
    if unexpected:
        raise ValueError(
            f"Wan2.2-Diffusers checkpoint has {len(unexpected)} keys with no LingBot counterpart, so this "
            f"is not the Diffusers release this script expects. First keys: {unexpected[:10]}"
        )

    template_only = set(template_state) - set(wan_state)
    patch_mlp_keys = set(PATCH_MLP_SOURCE)
    missing_patch_mlp = sorted(patch_mlp_keys - template_only)
    action_keys = {key for key in template_only if key.startswith(ACTION_KEY_PREFIXES)}
    unsupported_template_only = sorted(template_only - patch_mlp_keys - action_keys)
    if missing_patch_mlp or unsupported_template_only:
        raise ValueError(
            "LingBot/Wan key composition does not match the expected hybrid: "
            f"missing patch MLP keys={missing_patch_mlp}, "
            f"unsupported LingBot-only keys={unsupported_template_only[:10]}"
        )

    from_wan: list[str] = []
    from_patch_mlp: list[str] = []
    from_template: list[str] = []
    out_state: dict[str, torch.Tensor] = {}

    for key, reference in template_state.items():
        if key in wan_state:
            value = wan_state[key]
            from_wan.append(key)
        elif key in PATCH_MLP_SOURCE:
            source_key = PATCH_MLP_SOURCE[key]
            if source_key not in wan_state:
                raise KeyError(
                    f"Cannot build {key}: Wan checkpoint has no {source_key}."
                )
            source = wan_state[source_key]
            value = conv3d_to_linear(source, reference) if source.ndim > 1 else source
            from_patch_mlp.append(key)
        else:
            value = reference
            from_template.append(key)

        value = value.to(dtype=target_dtype).contiguous()
        if value.shape != reference.shape:
            raise ValueError(
                f"Shape mismatch for {key}: got {tuple(value.shape)}, expected {tuple(reference.shape)}."
            )
        # The derived linear tensors must not share storage with the archival
        # Conv3d tensors when both land in one safetensors shard.
        out_state[key] = value.clone() if key in PATCH_MLP_SOURCE else value

    def _numel(keys: list[str]) -> int:
        return int(sum(out_state[key].numel() for key in keys))

    transformer_dir = output_root / "transformer"
    report = {
        "lingbot_template": str(lingbot_template_root),
        "wan_diffusers": str(wan_diffusers_root),
        "output_root": str(output_root),
        "transformer_dir": str(transformer_dir),
        "target_dtype": dtype,
        "wan_keys": len(wan_state),
        "lingbot_keys": len(out_state),
        "common_keys_from_wan": len(from_wan),
        "wan_only_keys": unexpected,
        "lingbot_only_keys": sorted(from_template + from_patch_mlp),
        "counts": {
            "wan_common": len(from_wan),
            "patch_mlp": len(from_patch_mlp),
            "lingbot_only": len(from_template),
        },
        "param_counts": {
            "wan_common": _numel(from_wan),
            "patch_mlp": _numel(from_patch_mlp),
            "lingbot_only": _numel(from_template),
        },
        "num_layers": int(lingbot_config.get("num_layers", 0)),
    }

    staging_transformer_dir = staging_root / "transformer"
    staging_transformer_dir.mkdir()
    save_sharded_safetensors(
        out_state,
        staging_transformer_dir,
        max_shard_bytes=max_shard_bytes,
    )
    shutil.copyfile(template_config, staging_transformer_dir / "config.json")
    (staging_root / "wan22_diffusers_to_lingbot_init_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Wan2.2 Diffusers release into a LingBot-format init, "
        "taking the action stream from a LingBot template."
    )
    parser.add_argument(
        "--wan-diffusers-root",
        required=True,
        type=Path,
        help="Wan2.2-TI2V-5B-Diffusers transformer directory (the one holding config.json).",
    )
    parser.add_argument(
        "--lingbot-template-root",
        required=True,
        type=Path,
        help="LingBot transformer directory supplying the action stream, e.g. lingbot-va-base/transformer.",
    )
    parser.add_argument(
        "--output-root", required=True, type=Path, help="Output model root to create."
    )
    parser.add_argument(
        "--dtype", default="bfloat16", help="dtype of the saved checkpoint."
    )
    parser.add_argument(
        "--max-shard-size", default="5GB", help="Approximate cap per output shard."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = convert_wan22_diffusers_to_lingbot_init(
        wan_diffusers_root=args.wan_diffusers_root,
        lingbot_template_root=args.lingbot_template_root,
        output_root=args.output_root,
        max_shard_size=args.max_shard_size,
        dtype=args.dtype,
    )
    print(json.dumps(report["counts"], indent=2))
    print(f"wrote {report['transformer_dir']}")


if __name__ == "__main__":
    main()
