"""Reusable safetensor loading, schema, and publication helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SAFETENSORS_INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"
SAFETENSORS_FILENAME = "diffusion_pytorch_model.safetensors"


def conv3d_to_linear(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Flatten one spatial Conv3d kernel into a shape-checked linear weight."""

    reshaped = value.reshape(value.shape[0], -1)
    if reshaped.shape != target.shape:
        raise ValueError(
            "Unable to reshape patch embedding into a linear weight: "
            f"raw {tuple(value.shape)} -> {tuple(reshaped.shape)}, "
            f"expected {tuple(target.shape)}."
        )
    return reshaped


def checkpoint_safetensor_keys(root: Path) -> set[str]:
    """Validate one sharded/single-file checkpoint and return its keys."""

    index_path = root / SAFETENSORS_INDEX_FILENAME
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"Checkpoint index has no non-empty weight_map: {index_path}"
            )

        indexed_files = set(weight_map.values())
        actual_files = {path.name for path in root.glob("*.safetensors")}
        missing_files = sorted(indexed_files - actual_files)
        unindexed_files = sorted(actual_files - indexed_files)
        if missing_files or unindexed_files:
            raise ValueError(
                f"Checkpoint shard/index mismatch under {root}: "
                f"missing files={missing_files}, unindexed files={unindexed_files}"
            )

        actual_locations: dict[str, str] = {}
        duplicate_keys: list[str] = []
        for shard_name in sorted(indexed_files):
            with safe_open(root / shard_name, framework="pt") as handle:
                # safetensors.safe_open exposes keys() but is not iterable.
                for key in handle.keys():  # noqa: SIM118
                    if key in actual_locations:
                        duplicate_keys.append(key)
                    actual_locations[key] = shard_name
        if duplicate_keys:
            raise ValueError(
                "Checkpoint shards contain duplicate keys: "
                f"{sorted(duplicate_keys)[:10]}"
            )

        indexed_keys = set(weight_map)
        actual_keys = set(actual_locations)
        missing_keys = sorted(indexed_keys - actual_keys)
        unindexed_keys = sorted(actual_keys - indexed_keys)
        mismapped_keys = sorted(
            key
            for key, shard_name in weight_map.items()
            if actual_locations.get(key) != shard_name
        )
        if missing_keys or unindexed_keys or mismapped_keys:
            raise ValueError(
                f"Checkpoint key/index mismatch under {root}: "
                f"missing keys={missing_keys[:10]}, "
                f"unindexed keys={unindexed_keys[:10]}, "
                f"mismapped keys={mismapped_keys[:10]}"
            )
        return indexed_keys

    single_path = root / SAFETENSORS_FILENAME
    shard_files = sorted(root.glob("*.safetensors"))
    if shard_files != [single_path]:
        raise FileNotFoundError(
            f"Expected {single_path} or a valid {index_path}; "
            f"found {[path.name for path in shard_files]}"
        )
    with safe_open(single_path, framework="pt") as handle:
        return set(handle.keys())


def load_selected_safetensors(
    root: Path,
    keys: set[str],
) -> dict[str, torch.Tensor]:
    """Load selected tensors from a validated single or sharded checkpoint."""

    available = checkpoint_safetensor_keys(root)
    selected = keys & available
    index_path = root / SAFETENSORS_INDEX_FILENAME
    state: dict[str, torch.Tensor] = {}
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shard_to_keys: dict[str, list[str]] = {}
        for key in sorted(selected):
            shard_to_keys.setdefault(weight_map[key], []).append(key)
        for shard_name, shard_keys in shard_to_keys.items():
            with safe_open(root / shard_name, framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    state[key] = handle.get_tensor(key)
        return state
    with safe_open(root / SAFETENSORS_FILENAME, framework="pt", device="cpu") as handle:
        for key in sorted(selected):
            state[key] = handle.get_tensor(key)
    return state


def load_safetensors_state(root: Path) -> dict[str, torch.Tensor]:
    keys = checkpoint_safetensor_keys(root)
    return load_selected_safetensors(root, keys)


def save_sharded_safetensors(
    state: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    max_shard_bytes: int,
) -> None:
    """Write deterministic safetensor shards and a matching index."""

    if max_shard_bytes <= 0:
        raise ValueError(f"max_shard_bytes must be positive, got {max_shard_bytes}.")
    if not state:
        raise ValueError("Cannot publish an empty safetensor state.")

    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    for key in sorted(state):
        tensor = state[key]
        nbytes = tensor.numel() * tensor.element_size()
        if sizes[-1] and sizes[-1] + nbytes > max_shard_bytes:
            shards.append({})
            sizes.append(0)
        shards[-1][key] = tensor
        sizes[-1] += nbytes

    total = len(shards)
    weight_map: dict[str, str] = {}
    for index, shard in enumerate(shards, start=1):
        name = (
            SAFETENSORS_FILENAME
            if total == 1
            else f"diffusion_pytorch_model-{index:05d}-of-{total:05d}.safetensors"
        )
        save_file(shard, str(out_dir / name), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = name

    if total > 1:
        payload = {"metadata": {"total_size": sum(sizes)}, "weight_map": weight_map}
        (out_dir / SAFETENSORS_INDEX_FILENAME).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_shard_size(text: str) -> int:
    units = {"KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}
    upper = text.strip().upper()
    for suffix, scale in units.items():
        if upper.endswith(suffix):
            size = int(float(upper[: -len(suffix)]) * scale)
            break
    else:
        size = int(upper)
    if size <= 0:
        raise ValueError(f"Shard size must be positive, got {text!r}.")
    return size


def parse_torch_dtype(name: str) -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return choices[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype {name!r}; expected one of {sorted(choices)}."
        ) from exc


def model_state_shapes(
    model_class: type,
    config: dict[str, Any],
) -> dict[str, tuple[int, ...]]:
    """Build a model on the meta device and return its required state schema."""

    with torch.device("meta"):
        model = model_class.from_config(config)
    return {
        key: tuple(int(value) for value in tensor.shape)
        for key, tensor in model.state_dict().items()
    }


__all__ = [
    "SAFETENSORS_FILENAME",
    "SAFETENSORS_INDEX_FILENAME",
    "checkpoint_safetensor_keys",
    "conv3d_to_linear",
    "load_safetensors_state",
    "load_selected_safetensors",
    "model_state_shapes",
    "parse_shard_size",
    "parse_torch_dtype",
    "save_sharded_safetensors",
]
