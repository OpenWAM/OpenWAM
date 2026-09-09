"""Exercise portable raw-data adapters using tiny local media, without a VAE."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

av = pytest.importorskip("av")
pytest.importorskip("boto3")
ROOT = Path(__file__).resolve().parents[1]
import open_wam.data.preparation.build_manifest as build_manifest
import open_wam.data.preparation.convert_agibot as convert_agibot
import open_wam.data.preparation.encoding.encode_failure as encode_failure
import open_wam.data.preparation.prepare_dataset as prepare_dataset
from open_wam.data.preparation.multiview.archive_worker import JoinedParts
from open_wam.data.preparation.multiview.prepare import digest
from open_wam.data.preparation.multiview.sources import (
    HDFCursor,
    MetadataStore,
    RangeReader,
    VideoCursor,
    open_cursors,
)


def write_video(path, color=(230, 20, 10), *, frames=9, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv444p"
        stream.options = {"crf": "0"}
        for _ in range(frames):
            image = np.full((64, 64, 3), color, dtype=np.uint8)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def assert_rgb(image, expected):
    np.testing.assert_allclose(image[32, 32], expected, atol=3, rtol=0)


def close_cursors(cursors, owned):
    for item in list(cursors.values()) + owned:
        item.close()


def forbid_cloud_client(monkeypatch):
    def unexpected_client(_):
        raise AssertionError("Local media must not initialize cloud credentials")

    monkeypatch.setattr(MetadataStore, "client", property(unexpected_client))


def write_jpeg_hdf(path, *, frames=9, color=(230, 20, 10)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.full((64, 64, 3), color, dtype=np.uint8)
    memory = io.BytesIO()
    Image.fromarray(rgb).save(memory, format="JPEG", quality=100, subsampling=0)
    jpeg = memory.getvalue()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("language_instruction", data="Move the red object")
        group = handle.create_dataset(
            "observations/rgb_images/camera_left",
            (frames,),
            dtype=h5py.vlen_dtype(np.dtype("uint8")),
        )
        for i in range(frames):
            group[i] = np.frombuffer(jpeg, dtype=np.uint8)
    return jpeg


def test_video_cursor_uses_rgb_and_checks_declared_fps(tmp_path):
    path = tmp_path / "camera.mp4"
    write_video(path)
    cursor = VideoCursor(RangeReader(None, str(path)), 30, 9)
    try:
        assert_rgb(cursor.at(0), (230, 20, 10))
        assert_rgb(cursor.at(8 / 30), (230, 20, 10))
        assert cursor.duration == pytest.approx(8 / 30)
    finally:
        cursor.close()
    reader = RangeReader(None, str(path))
    try:
        with pytest.raises(ValueError, match="Container FPS"):
            VideoCursor(reader, 24, 9)
    finally:
        reader.close()


def make_agibot_archive(tmp_path, *, missing_camera=False):
    colors = {
        "head_color": (230, 20, 10),
        "hand_left_color": (10, 230, 20),
        "hand_right_color": (20, 10, 230),
    }
    archive = tmp_path / "selected.tar"
    with tarfile.open(archive, "w") as target:
        for native in ("901", "42"):
            for camera, color in colors.items():
                if missing_camera and native == "42" and camera == "hand_right_color":
                    continue
                video = tmp_path / "native" / native / "videos" / (camera + ".mp4")
                write_video(video, color)
                target.add(video, arcname=f"{native}/videos/{camera}.mp4")
        ignored = tarfile.TarInfo("42/depth/unused.bin")
        ignored.size = 4
        target.addfile(ignored, io.BytesIO(b"junk"))
    metadata = tmp_path / "task_327.json"
    # List ordering deliberately disagrees with sorted archive episode IDs.
    metadata.write_text(
        json.dumps(
            [
                dict(episode_id=901, task_id=327, task_name="Put cup on tray"),
                dict(episode_id=42, task_id=327, task_name="Take cup from shelf"),
                dict(episode_id=999, task_id=327, task_name="Unselected episode"),
            ]
        )
    )
    return archive, metadata, colors


def test_agibot_exact_id_join_prepares_reopenable_camera_paths(tmp_path, monkeypatch):
    archive, task_info, colors = make_agibot_archive(tmp_path)
    converted = tmp_path / "converted" / "alpha_task327"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_agibot",
            "--archive",
            str(archive),
            "--task-info",
            str(task_info),
            "--out",
            str(converted),
        ],
    )
    convert_agibot.main()
    assert json.loads((converted / "episode_names.json").read_text()) == {
        "0": "42",
        "1": "901",
    }
    assert not list(converted.rglob("*.bin"))
    rows = list(
        prepare_dataset.lerobot(
            SimpleNamespace(source="VPT-05", raw_root=str(converted))
        )
    )
    assert [row["task"] for row in rows] == ["Take cup from shelf", "Put cup on tray"]
    assert rows[0]["text_provenance"].endswith("#episode_id=42")
    assert rows[0]["native_end_frames"] == {camera: 9 for camera in colors}
    assert rows[0]["raw_source"]["fps"] == 30
    forbid_cloud_client(monkeypatch)
    cursors, owned = open_cursors(rows[0], MetadataStore(tmp_path / "metadata"))
    try:
        for camera, color in colors.items():
            assert_rgb(cursors[camera].at(4 / 30), color)
    finally:
        close_cursors(cursors, owned)


def test_agibot_missing_camera_cannot_publish_native_metadata(tmp_path, monkeypatch):
    archive, task_info, _ = make_agibot_archive(tmp_path, missing_camera=True)
    converted = tmp_path / "converted"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_agibot",
            "--archive",
            str(archive),
            "--task-info",
            str(task_info),
            "--out",
            str(converted),
        ],
    )
    with pytest.raises(KeyError):
        convert_agibot.main()
    assert not (converted / "meta/info.json").exists()


@pytest.mark.parametrize("take", ["kitchen_take_a", "kitchen__take_a"])
def test_ego_exact_take_identity_and_local_camera_mapping(tmp_path, monkeypatch, take):
    raw = tmp_path / "takes"
    write_video(raw / take / "frame_aligned_videos" / "aria01_rgb.mp4")
    takes = tmp_path / "takes.json"
    takes.write_text(
        json.dumps([dict(take_name=take, task_name="Arrange ingredients")])
    )
    rows = list(
        prepare_dataset.video_tree(
            SimpleNamespace(
                source="VPT-10R",
                raw_root=str(raw),
                takes=str(takes),
                pattern="*/frame_aligned_videos/*.mp4",
            )
        )
    )
    assert len(rows) == 1
    assert rows[0]["repo_id"] == take
    assert rows[0]["task"] == "Arrange ingredients"
    assert rows[0]["cameras"] == ["aria01_rgb"]
    assert rows[0]["native_end_frames"] == {"aria01_rgb": 9}
    forbid_cloud_client(monkeypatch)
    cursors, owned = open_cursors(rows[0], MetadataStore(tmp_path / "metadata"))
    try:
        assert_rgb(cursors["aria01_rgb"].at(0), (230, 20, 10))
    finally:
        close_cursors(cursors, owned)


@pytest.mark.parametrize(
    "embodiment,expected",
    [
        ("h5_franka_3rgb", (223, 91, 17)),
        ("h5_agilex_3rgb", (17, 91, 223)),
    ],
)
def test_official_hdf_uses_source_uri_not_local_archive_name(
    tmp_path, embodiment, expected
):
    path = tmp_path / "typed.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "observations/rgb_images/camera",
            data=np.full((9, 64, 64, 3), (17, 91, 223), dtype=np.uint8),
        )
    raw = dict(
        type="robomind_official_archive",
        uri=str(tmp_path / "download.tar.gz"),
        fps=30,
        source_uri=f"https://example.invalid/datasets/{embodiment}/task.tar.gz",
    )
    with h5py.File(path, "r") as handle:
        cursor = HDFCursor(handle, "camera", raw)
        np.testing.assert_array_equal(cursor.at(1 / 30)[32, 32], expected)


def test_failure_jpeg_rgb_sampling_and_manifest_without_model(tmp_path, monkeypatch):
    raw = tmp_path / "failure"
    write_jpeg_hdf(raw / "task/data/trajectory.hdf5")
    captured = []
    model_token = object()
    monkeypatch.setattr(encode_failure, "load_vae", lambda *a, **k: model_token)

    def fake_encode(model, video, *, normalize):
        assert model is model_token
        assert normalize
        captured.append(video.clone())
        t, c, h, w = video.shape
        assert c == 3
        return torch.zeros(1 + (t - 1) // 4, h // 16, w // 16, 48)

    monkeypatch.setattr(encode_failure, "encode_clip", fake_encode)
    output, episodes = tmp_path / "latents", tmp_path / "episodes.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "encode_failure",
            "--dataset",
            str(raw),
            "--vae",
            str(tmp_path / "unused"),
            "--source-fps",
            "30",
            "--out-root",
            str(output),
            "--episodes-out",
            str(episodes),
        ],
    )
    encode_failure.main()
    assert len(captured) == 1
    assert captured[0].shape == (5, 3, 128, 128)
    np.testing.assert_allclose(
        captured[0][0, :, 64, 64].numpy() * 255, (230, 20, 10), atol=3
    )
    payload = torch.load(next(output.rglob("*.pth")), weights_only=True)
    assert payload["frame_ids"] == [0, 2, 4, 6, 8]
    assert payload["color_policy"] == "robomind_failure_standard_jpeg_v1"
    rows = list(
        build_manifest.single_rows(
            SimpleNamespace(
                source="VPT-06", episodes=str(episodes), latent_root=str(output)
            )
        )
    )
    assert len(rows) == 1
    assert rows[0]["task"] == "Move the red object"
    assert rows[0]["native_fps"] == 30
    assert rows[0]["observation_fps"] == 15


def test_local_joined_archive_checks_complete_gzip_stream(tmp_path):
    original = tmp_path / "sample.hdf5"
    write_jpeg_hdf(original)
    archive = tmp_path / "task.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(original, arcname="task/data/trajectory.hdf5")
    with (
        JoinedParts(None, str(archive)) as joined,
        gzip.GzipFile(fileobj=joined) as decoded,
    ):
        with tarfile.open(fileobj=decoded, mode="r|") as members:
            member = next(iter(members))
            assert members.extractfile(member).read() == original.read_bytes()
        while decoded.read(1024):
            pass
    damaged = bytearray(archive.read_bytes())
    damaged[-8] ^= 1  # Corrupt gzip CRC while retaining a valid tar body.
    corrupt = tmp_path / "bad.tar.gz"
    corrupt.write_bytes(damaged)
    with pytest.raises(gzip.BadGzipFile):
        with (
            JoinedParts(None, str(corrupt)) as joined,
            gzip.GzipFile(fileobj=joined) as decoded,
        ):
            while decoded.read(1024):
                pass


def test_official_multi_view_requires_successful_matching_archive_receipt(tmp_path):
    output = tmp_path / "output"
    receipt_path = output / "receipts/VPT-06/plan.json"
    receipt_path.parent.mkdir(parents=True)
    tensor = output / "clip.pth"
    tensor.write_bytes(b"verified clip fixture")
    uri = str(tmp_path / "source.tar.gz")
    receipt = dict(
        status="complete",
        source_id="VPT-06",
        plan_id="plan-id",
        repo_id="repo",
        episode_index=0,
        contract_sha256="contract",
        physical_episode_key="episode",
        raw_source=dict(type="robomind_official_archive", uri=uri),
        outputs=[
            dict(
                latent_path=str(tensor),
                latent_bytes=tensor.stat().st_size,
                latent_sha256=hashlib.sha256(tensor.read_bytes()).hexdigest(),
                repo_id="repo",
            )
        ],
    )
    receipt_path.write_text(json.dumps(receipt))
    args = SimpleNamespace(source="VPT-06", receipts=str(output / "receipts"))
    gate = output / "archive_units" / (digest(uri) + ".json")
    gate.parent.mkdir()
    good = dict(
        status="complete",
        gzip_eof_verified=True,
        contract_sha256="contract",
        plan_ids=["plan-id"],
    )
    for bad in [
        None,
        dict(good, status="failed"),
        dict(good, gzip_eof_verified=False),
        dict(good, contract_sha256="other"),
        dict(good, plan_ids=["different-plan"]),
    ]:
        if bad is not None:
            gate.write_text(json.dumps(bad))
        assert list(build_manifest.multi_view_rows(args)) == []
    gate.write_text(json.dumps(good))
    assert len(list(build_manifest.multi_view_rows(args))) == 1


def test_official_report_keeps_download_path_separate_from_color_provenance(tmp_path):
    extracted = tmp_path / "extracted"
    member = "task/data/trajectory.hdf5"
    write_jpeg_hdf(extracted / member)
    latent = (
        tmp_path
        / "latents"
        / "repo"
        / "latents/chunk-000/camera_left/episode_000000_0_9.pth"
    )
    sidecar = latent.parents[3] / "text_metadata.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            dict(
                hdf5_relative_path=member,
                source_fps=30,
                color_policy="robomind_official_rgb_v1",
                task="Move the red object",
                text_provenance="native-hdf#language_instruction",
            )
        )
    )
    report = tmp_path / "encode_report.json"
    report.write_text(
        json.dumps(
            dict(
                status="complete",
                episodes=[
                    dict(
                        repo_id="repo",
                        outputs=[dict(path=str(latent), camera="camera_left")],
                    )
                ],
            )
        )
    )
    local_archive = tmp_path / "downloaded.tar.gz"
    official_uri = "https://example.invalid/datasets/h5_franka_3rgb/task.tar.gz"
    rows = list(
        prepare_dataset.robomind_report(
            SimpleNamespace(
                source="VPT-06",
                raw_root=str(extracted),
                report=str(report),
                archive_path=str(local_archive),
                archive_source=official_uri,
            )
        )
    )
    assert rows[0]["raw_source"]["uri"] == str(local_archive)
    assert rows[0]["raw_source"]["source_uri"] == official_uri
    assert rows[0]["raw_source"]["member"] == member
    assert rows[0]["native_end_frames"] == {"camera_left": 9}
    cursors, owned = open_cursors(
        rows[0], MetadataStore(tmp_path / "metadata"), extracted / member
    )
    try:
        assert_rgb(cursors["camera_left"].at(0), (230, 20, 10))
    finally:
        close_cursors(cursors, owned)
