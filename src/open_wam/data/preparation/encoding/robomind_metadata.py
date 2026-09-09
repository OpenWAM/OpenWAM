"""Native RoboMind HDF5 field names and language provenance."""

import hashlib

import h5py
import numpy as np

RGB_GROUP = "observations/rgb_images"


def native_strings(value) -> list[str]:
    """Preserve decoded strings exactly; never index a string as a sequence."""
    if isinstance(value, np.ndarray):
        if value.ndim > 1:
            raise ValueError(f"Native text has unsupported rank {value.ndim}")
        value = value.item() if value.ndim == 0 else value.tolist()
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    texts = []
    for item in values:
        if isinstance(item, (bytes, np.bytes_)):
            item = bytes(item).decode("utf-8", errors="strict")
        if not isinstance(item, str):
            raise TypeError(
                f"Native text must contain strings, got {type(item).__name__}"
            )
        if "\x00" in item:
            raise ValueError("Native text contains embedded NUL")
        if item.strip() and item not in texts:
            texts.append(item)
    return texts


def read_native_text(handle: h5py.File) -> dict:
    fields = {}
    for name in ("language_instruction", "language_raw"):
        if name not in handle:
            continue
        dataset = handle[name]
        if not isinstance(dataset, h5py.Dataset) or dataset.size > 16384:
            raise ValueError(f"Invalid or oversized text dataset {name}")
        texts = native_strings(dataset[()])
        if sum(len(s.encode("utf-8")) for s in texts) > 1024 * 1024:
            raise ValueError(f"Text dataset exceeds metadata size bound: {name}")
        fields[name] = {
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
            "texts": texts,
        }
    distinct = list(
        dict.fromkeys(text for field in fields.values() for text in field["texts"])
    )
    if len(distinct) == 1:
        task = distinct[0]
        status = "native_hdf5_text"
    elif len(distinct) > 1:
        task, status = "", "ambiguous_native_hdf5_text"
    else:
        task, status = "", "missing_native_hdf5_text"
    used_fields = [
        name for name, field in fields.items() if task and task in field["texts"]
    ]
    return {
        "task": task,
        "text_status": status,
        "native_fields": fields,
        "text_fields": used_fields,
        "text_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
    }
