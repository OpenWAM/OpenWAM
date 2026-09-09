"""RoboMIND's official decoder and embodiment-specific RGB conversion.

Reference: https://github.com/x-humanoid-robomind/
x-humanoid-robomind.github.io/blob/main/static/quick_start.ipynb
JPEG decoding and raw byte reshaping must precede the embodiment conversion.
In particular, a PIL RGB decoder is not interchangeable with cv2.imdecode here.
"""

from __future__ import annotations

from collections import Counter
from urllib.parse import urlsplit

import cv2
import h5py
import numpy as np

from .robomind_metadata import RGB_GROUP

COLOR_POLICY = "robomind_official_rgb_v1"
BGR_EMBODIMENTS = frozenset(
    {
        "h5_franka_3rgb",
        "h5_franka_1rgb",
        "h5_ur_1rgb",
        "h5_franka_fr3_dual",
    }
)
RGB_EMBODIMENTS = frozenset(
    {
        "h5_agilex_3rgb",
        "h5_simulation",
        "h5_sim_franka_3rgb",
        "h5_sim_tienkung_1rgb",
        "h5_tienkung_gello_1rgb",
        "h5_tienkung_xsens_1rgb",
        "h5_tienkung_prod1_gello_1rgb",
    }
)
KNOWN_EMBODIMENTS = BGR_EMBODIMENTS | RGB_EMBODIMENTS
FRAME_ENCODINGS = frozenset({"jpeg", "png", "encoded_image", "raw"})

# Resolutions seen in this corpus, plus their transposes, all of which are
# checked against the byte count before any of them is believed.
RAW_SHAPES = (
    (480, 640),
    (240, 320),
    (360, 640),
    (720, 1280),
    (1080, 1920),
    (256, 256),
    (480, 848),
    (128, 128),
)


def raw_shape(raw: bytes) -> tuple[int, int]:
    """Resolve (H, W) for an uncompressed RGB buffer, and prove the choice.

    A byte count does not name a shape: 921600 is 480x640x3 and equally
    640x480x3. Choosing wrong does not fail -- it shears the picture into
    diagonal stripes that still encode to a perfectly plausible latent, and
    nothing downstream would ever flag it. So each candidate with the right
    byte count is scored by how smooth the image is down its columns (a sheared
    frame is violently discontinuous there: measured 1.48 for the right shape
    against 37.87 for its transpose on real data), and the winner must beat the
    runner-up by a wide margin. If it does not, this raises rather than guesses.
    """
    n = len(raw)
    cands = []
    for h, w in RAW_SHAPES:
        if h * w * 3 == n:
            if (h, w) not in cands:
                cands.append((h, w))
            if h != w and (w, h) not in cands:
                cands.append((w, h))
    if not cands:
        raise ValueError(f"raw buffer of {n} bytes matches no known resolution")
    a = np.frombuffer(raw, np.uint8)
    scored = []
    for h, w in cands:
        img = a.reshape(h, w, 3).astype(np.int16)
        scored.append((float(np.abs(np.diff(img, axis=0)).mean()), (h, w)))
    scored.sort()
    best_score, best = scored[0]
    if len(scored) > 1:
        runner = scored[1][0]
        if runner <= best_score * 3.0:
            raise ValueError(
                f"cannot tell {best} from {scored[1][1]} for a {n}-byte buffer "
                f"(column roughness {best_score:.2f} vs {runner:.2f}); refusing "
                f"to guess an orientation"
            )
    return best


def resolve_embodiment(archive_source: str) -> str:
    """Use an exact archive path component, never a camera or task name."""
    if not isinstance(archive_source, str):
        raise ValueError(
            "archive_source must be a URI containing a known h5_* embodiment"
        )
    parsed = urlsplit(archive_source)
    if parsed.scheme not in ("s3", "http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid archive_source URI: {archive_source!r}")
    components = [part for part in parsed.path.split("/") if part.startswith("h5_")]
    if len(components) != 1 or components[0] not in KNOWN_EMBODIMENTS:
        raise ValueError(
            f"Unknown or ambiguous RoboMIND embodiment in archive_source: {archive_source!r}"
        )
    return components[0]


def decode_rgb_frame(
    raw: bytes, embodiment: str, *, source_shape=None
) -> tuple[np.ndarray, str]:
    """Return contiguous uint8 HWC RGB and the observed byte encoding."""
    if embodiment not in KNOWN_EMBODIMENTS:
        raise ValueError(f"Unsupported RoboMIND embodiment: {embodiment!r}")
    if not raw:
        raise ValueError("Empty camera frame")
    if source_shape is not None:
        # A typed HDF5 HWC array already records the actual geometry. Flattening
        # it and re-inferring the dimensions fails on uniform/low-texture frames.
        if (
            len(source_shape) != 3
            or source_shape[2] != 3
            or any(type(n) is not int or n <= 0 for n in source_shape)
            or len(raw) != int(np.prod(source_shape))
        ):
            raise ValueError(f"Invalid explicit uint8 HWC geometry: {source_shape}")
        decoded = np.frombuffer(raw, dtype=np.uint8).reshape(source_shape)
        encoding = "raw"
    else:
        try:
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            decoded = None
        if decoded is None:
            # Byte-only frames retain the existing strict geometry check.
            height, width = raw_shape(raw)
            decoded = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            encoding = "raw"
        else:
            encoding = (
                "jpeg"
                if raw.startswith(b"\xff\xd8")
                else "png"
                if raw.startswith(b"\x89PNG\r\n\x1a\n")
                else "encoded_image"
            )
    if decoded.ndim != 3 or decoded.shape[2] != 3 or decoded.dtype != np.uint8:
        raise ValueError(
            f"Unexpected decoded camera image: {decoded.shape}, {decoded.dtype}"
        )
    if embodiment in BGR_EMBODIMENTS:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(decoded), encoding


def color_metadata(embodiment: str, counts: dict[str, int]) -> dict:
    if embodiment not in KNOWN_EMBODIMENTS:
        raise ValueError(f"Unsupported RoboMIND embodiment: {embodiment!r}")
    if not counts or any(
        key not in FRAME_ENCODINGS or type(count) is not int or count < 1
        for key, count in counts.items()
    ):
        raise ValueError(f"Invalid source frame encoding counts: {counts!r}")
    counts = dict(sorted(counts.items()))
    return {
        "color_policy": COLOR_POLICY,
        "embodiment": embodiment,
        "source_frame_encoding": next(iter(counts)) if len(counts) == 1 else "mixed",
        "source_frame_encoding_counts": counts,
    }


def validate_color_metadata(payload: dict, archive_source: str) -> dict:
    """Only certified outputs from this exact policy can be reused."""
    embodiment = resolve_embodiment(archive_source)
    if not isinstance(payload, dict) or payload.get("color_policy") != COLOR_POLICY:
        raise ValueError(
            "Existing latent lacks the required RGB color policy; use a new output root"
        )
    if payload.get("embodiment") != embodiment:
        raise ValueError("Existing latent embodiment conflicts with archive_source")
    counts = payload.get("source_frame_encoding_counts")
    if not isinstance(counts, dict):
        raise ValueError("Existing latent lacks source frame encoding counts")
    metadata = color_metadata(embodiment, counts)
    if payload.get("source_frame_encoding") != metadata["source_frame_encoding"]:
        raise ValueError(
            "Existing latent has inconsistent source frame encoding metadata"
        )
    if sum(counts.values()) != payload.get("end_frame"):
        raise ValueError(
            "Existing latent color metadata does not cover every source frame"
        )
    return metadata


def read_rgb_frames(
    path: str, camera: str, archive_source: str
) -> tuple[np.ndarray, dict]:
    embodiment = resolve_embodiment(archive_source)
    frames, counts, expected_shape = [], Counter(), None
    with h5py.File(path, "r") as handle:
        dataset = handle[f"{RGB_GROUP}/{camera}"]
        for index in range(len(dataset)):
            try:
                native = dataset[index]
                source_shape = None
                if isinstance(native, np.ndarray) and native.ndim > 1:
                    if (
                        native.ndim != 3
                        or native.shape[2] != 3
                        or native.dtype != np.uint8
                    ):
                        raise ValueError(
                            f"Expected typed uint8 HWC RGB, got {native.shape}, {native.dtype}"
                        )
                    source_shape = tuple(int(n) for n in native.shape)
                frame, encoding = decode_rgb_frame(
                    bytes(native), embodiment, source_shape=source_shape
                )
                if expected_shape is not None and frame.shape != expected_shape:
                    raise ValueError(
                        f"Camera geometry changed from {expected_shape} to {frame.shape}"
                    )
                expected_shape = frame.shape
            except Exception as error:
                raise ValueError(f"{camera} frame {index}: {error}") from error
            frames.append(frame)
            counts[encoding] += 1
    if not frames:
        raise ValueError(f"{camera}: no frames")
    return np.stack(frames), color_metadata(embodiment, dict(counts))
