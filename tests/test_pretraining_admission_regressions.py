"""Read-only PR #65 audit reproductions using temporary local fixtures."""

import io
import json
import sys

import av
import numpy as np
import pytest
import torch

from open_wam.data.preparation import build_manifest, prepare_dataset
from open_wam.data.preparation.frames import decode_video
from open_wam.data.preparation.video_sources import Clip


def test_reviewed_override_can_resolve_ambiguous_tasks(tmp_path, monkeypatch):
    meta = tmp_path / "raw" / "demo" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            dict(
                codebase_version="v2.1",
                fps=30,
                total_episodes=1,
                features={"front": {"dtype": "video"}},
            )
        )
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            dict(
                episode_index=0,
                length=18,
                tasks=["move left", "move right"],
            )
        )
        + "\n"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps(dict(task_index=0, task="move left")) + "\n"
    )
    override = tmp_path / "reviewed.jsonl"
    override.write_text(
        json.dumps(
            dict(
                repo_id="demo",
                episode_index=0,
                task="move left",
                text_provenance="reviewed native annotation",
            )
        )
        + "\n"
    )
    output = tmp_path / "episodes.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dataset",
            "--source",
            "VPT-01",
            "--format",
            "lerobot",
            "--raw-root",
            str(meta.parent),
            "--out",
            str(output),
            "--text-overrides",
            str(override),
        ],
    )
    prepare_dataset.main()
    assert json.loads(output.read_text())["task"] == "move left"


def test_manifest_rejects_temporally_truncated_payload(tmp_path):
    episode = prepare_dataset.make_row(
        "VPT-01",
        "demo",
        0,
        ["front"],
        {"front": 18},
        dict(type="lerobot", root="/unused", fps=30),
        "move left",
        "native",
    )
    index = tmp_path / "episodes.jsonl"
    index.write_text(json.dumps(episode) + "\n")
    root = tmp_path / "latents"
    tensor = root / "demo/latents/chunk-000/front/episode_000000_0_18.pth"
    tensor.parent.mkdir(parents=True)
    # 18 source frames at 30 Hz need 9 sampled RGB / 3 latent frames at 15 Hz.
    torch.save(
        dict(
            latent=torch.zeros(2, 8, 8, 48, dtype=torch.float16),
            latent_layout="THWC",
            latents_normalized=True,
            video_num_frames=5,
            video_height=128,
            video_width=128,
            fps=15.0,
            ori_fps=30.0,
        ),
        tensor,
    )
    from types import SimpleNamespace

    with pytest.raises(ValueError):
        list(
            build_manifest.single_rows(
                SimpleNamespace(
                    source="VPT-01", episodes=str(index), latent_root=str(root)
                )
            )
        )


def test_float32_packed_timestamp_keeps_boundary_frame():
    payload = io.BytesIO()
    with av.open(payload, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i in range(410):
            color = 40 if i <= 404 else 200
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), color, np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    rounded_start = float(np.float32(40.4))
    result = decode_video(
        payload.getvalue(),
        Clip(path="unused.mp4", from_timestamp=rounded_start, to_timestamp=41.0),
        declared_fps=10,
    )
    assert result.frames[0].mean() < 100, (rounded_start, result.frames[0].mean())


def test_skip_inventory_uses_cheap_native_episode_length():
    from types import SimpleNamespace
    from open_wam.data.preparation.encoding.encode_latents import drop_completed

    repo = SimpleNamespace(name='demo', cameras=['front'], total_episodes=1,
                           chunk_of=lambda _: 0, length_of=lambda _: 18)
    paths = {'chunk-000/front/episode_000000_0_12.pth'}
    store = SimpleNamespace(list_keys=lambda _: paths)
    options = dict(out_store=store, out_root='/unused', latent_subdir='latents',
                   cameras_filter=None, start_episode=0, episodes=None, overwrite=False)
    assert drop_completed([repo], **options) == [repo]
    paths.clear()
    paths.add('chunk-000/front/episode_000000_0_18.pth')
    assert drop_completed([repo], **options) == []
