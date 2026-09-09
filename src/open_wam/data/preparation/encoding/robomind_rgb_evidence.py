"""Canonical source-RGB and encoder-contract evidence for exact source aliases."""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

RGB_HASH_SCHEMA = "rgb_uint8_thwc_v1"
EVIDENCE_FIELDS = (
    "source_rgb_sha256",
    "source_rgb_shape",
    "source_rgb_dtype",
    "source_rgb_hash_schema",
    "encode_contract_sha256",
    "encode_contract",
)


def canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rgb_evidence(frames: np.ndarray) -> dict:
    require(
        isinstance(frames, np.ndarray)
        and frames.dtype == np.uint8
        and frames.ndim == 4
        and frames.shape[-1] == 3
        and all(frames.shape),
        "Expected complete uint8 THWC RGB frames",
    )
    contiguous = np.ascontiguousarray(frames)
    header = {"dtype": "uint8", "shape": list(contiguous.shape)}
    digest = hashlib.sha256(
        RGB_HASH_SCHEMA.encode() + b"\n" + canonical_json(header) + b"\n"
    )
    digest.update(memoryview(contiguous).cast("B"))
    return dict(
        source_rgb_sha256=digest.hexdigest(),
        source_rgb_shape=header["shape"],
        source_rgb_dtype="uint8",
        source_rgb_hash_schema=RGB_HASH_SCHEMA,
    )


def attach_contract(evidence: dict, contract: dict) -> dict:
    result = dict(
        evidence,
        encode_contract=contract,
        encode_contract_sha256=hashlib.sha256(canonical_json(contract)).hexdigest(),
    )
    return validate_evidence(result)


def validate_evidence(holder: dict) -> dict:
    require(
        isinstance(holder, dict) and all(field in holder for field in EVIDENCE_FIELDS),
        "Missing complete RGB/encode evidence",
    )
    evidence = {field: holder[field] for field in EVIDENCE_FIELDS}
    require(
        evidence["source_rgb_hash_schema"] == RGB_HASH_SCHEMA
        and evidence["source_rgb_dtype"] == "uint8",
        "Unknown RGB hash schema/dtype",
    )
    shape = evidence["source_rgb_shape"]
    require(
        isinstance(shape, list)
        and len(shape) == 4
        and all(type(x) is int and x > 0 for x in shape)
        and shape[-1] == 3,
        "Invalid full source RGB shape",
    )
    for field in ("source_rgb_sha256", "encode_contract_sha256"):
        require(
            isinstance(evidence[field], str)
            and re.fullmatch("[0-9a-f]{64}", evidence[field]),
            "Invalid evidence SHA256",
        )
    contract = evidence["encode_contract"]
    require(
        isinstance(contract, dict) and contract.get("format_version") == 1,
        "Invalid encode contract",
    )
    require(
        evidence["encode_contract_sha256"]
        == hashlib.sha256(canonical_json(contract)).hexdigest(),
        "Encode contract digest mismatch",
    )
    require(
        contract.get("source_rgb_shape") == shape, "RGB/contract source shape mismatch"
    )
    for field in ("source_fps", "fps"):
        value = contract.get(field)
        require(
            type(value) in (int, float) and math.isfinite(value) and value > 0,
            "Invalid contract FPS",
        )
    ids = contract.get("frame_ids")
    require(
        isinstance(ids, list)
        and bool(ids)
        and all(type(i) is int and 0 <= i < shape[0] for i in ids)
        and all(a < b for a, b in zip(ids, ids[1:])),
        "Invalid contract frame IDs",
    )
    require(
        contract.get("temporal_stride") == 4 and contract.get("spatial_stride") == 16,
        "Unsupported VAE stride contract",
    )
    require(
        type(contract.get("latents_normalized")) is bool
        and contract.get("store_dtype") in ("fp16", "bf16", "fp32"),
        "Invalid normalization/storage contract",
    )
    require(
        contract.get("encode_dtype") == "bf16"
        and contract.get("fit_mode") in ("letterbox_pad", "center_crop", "stretch"),
        "Invalid encode dtype/fit contract",
    )
    require(
        contract.get("size_mode") in ("aspect_bins", "fixed")
        and isinstance(contract.get("bins"), list),
        "Missing resize policy contract",
    )
    selected = contract.get("selected_bin", {})
    require(
        all(
            type(selected.get(k)) is int and selected[k] > 0
            for k in ("target_height", "target_width")
        ),
        "Invalid selected resize bin",
    )
    fingerprint = contract.get("vae_fingerprint", {})
    require(
        re.fullmatch("[0-9a-f]{64}", str(fingerprint.get("config_sha256", "")))
        is not None,
        "Missing VAE config fingerprint",
    )
    weights = fingerprint.get("weights_sha256")
    require(
        isinstance(weights, dict)
        and bool(weights)
        and all(re.fullmatch("[0-9a-f]{64}", str(value)) for value in weights.values()),
        "Missing VAE weight fingerprints",
    )
    require(
        re.fullmatch(
            "[0-9a-f]{64}", str(contract.get("encoder_implementation_sha256", ""))
        )
        is not None,
        "Missing encoder implementation fingerprint",
    )
    require(
        isinstance(contract.get("runtime_versions"), dict)
        and all(
            contract["runtime_versions"].get(name) for name in ("torch", "diffusers")
        ),
        "Missing encode runtime versions",
    )
    return evidence


def validate_payload_evidence(payload: dict) -> dict:
    evidence = validate_evidence(payload)
    contract = evidence["encode_contract"]
    shape = evidence["source_rgb_shape"]
    require(
        payload.get("start_frame") == 0 and payload.get("end_frame") == shape[0],
        "RGB evidence source span mismatch",
    )
    require(
        contract.get("color_policy") == payload.get("color_policy")
        and contract.get("embodiment") == payload.get("embodiment"),
        "Color contract mismatch",
    )
    for actual, expected in (
        ("ori_fps", "source_fps"),
        ("fps", "fps"),
        ("frame_ids", "frame_ids"),
        ("latents_normalized", "latents_normalized"),
    ):
        require(
            payload.get(actual) == contract[expected],
            f"Payload/contract {actual} mismatch",
        )
    require(
        payload.get("video_num_frames") == len(contract["frame_ids"]),
        "Contract encoded-frame count mismatch",
    )
    require(
        payload.get("video_height") == contract["selected_bin"]["target_height"]
        and payload.get("video_width") == contract["selected_bin"]["target_width"],
        "Contract spatial target mismatch",
    )
    require(
        payload.get("fit_mode") == contract["fit_mode"]
        and payload.get("resize_bin") == contract["selected_bin"]["name"],
        "Payload resize contract mismatch",
    )
    dtype = {
        "fp16": "torch.float16",
        "bf16": "torch.bfloat16",
        "fp32": "torch.float32",
    }[contract["store_dtype"]]
    require(
        str(getattr(payload.get("latent"), "dtype", None)) == dtype,
        "Payload storage dtype differs from contract",
    )
    require(
        sum(payload.get("source_frame_encoding_counts", {}).values()) == shape[0],
        "Encoding counts differ from RGB evidence",
    )
    return evidence
