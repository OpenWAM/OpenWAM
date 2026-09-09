"""Characterize shared payload and native-text behavior before moving ownership."""

import hashlib

import h5py
import numpy as np
import pytest
import torch

from open_wam.configs.data_mixed_video import MixedVideoResizeBinConfig
from open_wam.data.preparation.encoding.robomind_metadata import (
    native_strings,
    read_native_text,
)
from open_wam.data.preparation.encoding.payload import build_payload


@pytest.mark.parametrize(
    "dtype_name,dtype",
    [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)],
)
@pytest.mark.parametrize("normalize", [False, True])
def test_payload_exact_tensor_and_metadata(dtype_name, dtype, normalize):
    latent = torch.linspace(-1.01, 2.03, 3 * 8 * 16 * 48).reshape(3, 8, 16, 48)
    original = latent.clone()
    result = build_payload(
        latent=latent,
        camera="wrist",
        episode_index=7,
        indices=list(range(0, 18, 2)),
        source_frames=18,
        source_fps=30.0,
        bin_config=MixedVideoResizeBinConfig("wide", 1, 2, 128, 256),
        fps=15.0,
        store_dtype=dtype_name,
        fit_mode="letterbox_pad",
        normalize_latents=normalize,
        vae_path="/any/location/vae/",
    )
    tensor = result.pop("latent")
    torch.testing.assert_close(tensor, original.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(latent, original, rtol=0, atol=0)
    assert result == dict(
        latent_layout="THWC",
        latent_num_frames=3,
        latent_height=8,
        latent_width=16,
        frame_ids=list(range(0, 18, 2)),
        start_frame=0,
        end_frame=18,
        video_num_frames=9,
        fps=15.0,
        ori_fps=30.0,
        video_height=128,
        video_width=256,
        resize_bin="wide",
        camera="wrist",
        episode_index=7,
        fit_mode="letterbox_pad",
        latents_normalized=normalize,
        vae_id="vae",
    )


@pytest.mark.parametrize(
    "value",
    [
        b" move left ",
        np.array(b" move left "),
        np.array([b" move left ", b"", b" move left "]),
    ],
)
def test_native_text_preserves_whitespace_and_deduplicates(value):
    assert native_strings(value) == [" move left "]


@pytest.mark.parametrize(
    "value,error",
    [
        (np.zeros((1, 1)), ValueError),
        (3, TypeError),
        (b"\xff", UnicodeDecodeError),
        ("left\x00right", ValueError),
    ],
)
def test_native_text_rejects_invalid_metadata(value, error):
    with pytest.raises(error):
        native_strings(value)


@pytest.mark.parametrize(
    "texts,status,task",
    [
        ([], "missing_native_hdf5_text", ""),
        ([" move left "], "native_hdf5_text", " move left "),
        (["left", "right"], "ambiguous_native_hdf5_text", ""),
    ],
)
def test_hdf5_text_provenance(tmp_path, texts, status, task):
    with h5py.File(tmp_path / "native.h5", "w") as handle:
        for name, text in zip(("language_instruction", "language_raw"), texts):
            handle.create_dataset(name, data=np.bytes_(text))
        result = read_native_text(handle)
    assert result == dict(
        task=task,
        text_status=status,
        native_fields={
            name: dict(shape=[], dtype=f"|S{len(text)}", texts=[text])
            for name, text in zip(("language_instruction", "language_raw"), texts)
        },
        text_fields=["language_instruction"] if task else [],
        text_sha256=hashlib.sha256(task.encode()).hexdigest(),
    )
