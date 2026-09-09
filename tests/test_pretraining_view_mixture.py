"""Mixed pretraining must use actual single-view and RGB multi view manifest rows."""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
import open_wam.cli.video_pretraining_make_config as make_config
import open_wam.data.preparation.publish_snapshot as publish_snapshot
from open_wam.data.preparation.build_manifest import write_csv
from open_wam.data.preparation.view_mixture import validate_view_mixture


def publish(tmp_path, monkeypatch, augmentations):
    rows = [
        dict(
            source_id="VPT-01",
            clip_id=f"clip-{index}",
            physical_episode_key="same-physical-demonstration",
            latent_sha256=str(index) * 64,
            latent_path=f"/logical/clip-{index}.pth",
            augmentation=augmentation,
            video_num_frames=17,
            length_frames=5,
            observation_fps=15.0,
            task="move the block",
        )
        for index, augmentation in enumerate(augmentations)
    ]
    manifest = tmp_path / "input.csv"
    write_csv(manifest, rows)
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(
        sys, "argv", ["publish", "--manifests", str(manifest), "--out", str(snapshot)]
    )
    publish_snapshot.main()
    return snapshot, json.loads((snapshot / "snapshot.json").read_text())


def generate_config(tmp_path, monkeypatch, snapshot, *, require_multiview):
    path = tmp_path / "config.yaml"
    args = [
        "config",
        "--snapshot",
        str(snapshot),
        "--model-assets",
        str(tmp_path / "models"),
        "--run-root",
        str(tmp_path / "runs"),
        "--out",
        str(path),
        "--allow-subset",
        "--online-text",
    ]
    if require_multiview:
        args.append("--require-multiview")
    monkeypatch.setattr(sys, "argv", args)
    make_config.main()
    return path


def train_entrypoint():
    import open_wam.cli.video_pretraining_train as module

    return module


def test_mixed_manifest_generates_named_config_and_passes_startup(
    tmp_path, monkeypatch, capsys
):
    snapshot, record = publish(tmp_path, monkeypatch, ["single_view", "multi_view"])
    cfgpath = generate_config(tmp_path, monkeypatch, snapshot, require_multiview=True)
    cfg = yaml.safe_load(cfgpath.read_text())
    assert cfg["name"] == "openwam_single_and_multiview_pretraining"
    assert cfg["data"]["video_sources"][0]["manifest_csv"] == str(
        snapshot / "VPT-01.csv"
    )
    assert record["sources"]["VPT-01"]["physical_episodes"] == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--snapshot",
            str(snapshot),
            "--cfg",
            str(cfgpath),
            "--require-multiview",
            "--check-only",
        ],
    )
    train_entrypoint().main()
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report["view_mixture"] == dict(
        single_clips=1,
        multi_view_clips=1,
        multiview_sources=["VPT-01"],
        mixed_pretraining_required=True,
        manifest_view_counts_verified=True,
    )


@pytest.mark.parametrize("augmentation", ["single_view", "multi_view"])
def test_mixed_config_refuses_one_representation_before_writing(
    tmp_path, monkeypatch, augmentation
):
    snapshot, _ = publish(tmp_path, monkeypatch, [augmentation])
    with pytest.raises(
        ValueError, match="requires both single-view and RGB multi view"
    ):
        generate_config(tmp_path, monkeypatch, snapshot, require_multiview=True)
    assert not (tmp_path / "config.yaml").exists()


def test_startup_flag_rejects_an_existing_single_view_config(tmp_path, monkeypatch):
    snapshot, _ = publish(tmp_path, monkeypatch, ["single_view"])
    cfgpath = generate_config(tmp_path, monkeypatch, snapshot, require_multiview=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--snapshot",
            str(snapshot),
            "--cfg",
            str(cfgpath),
            "--require-multiview",
            "--check-only",
        ],
    )
    with pytest.raises(
        ValueError, match="requires both single-view and RGB multi view"
    ):
        train_entrypoint().main()


def test_mixed_gate_checks_actual_rows_not_only_summary(tmp_path, monkeypatch):
    snapshot, record = publish(tmp_path, monkeypatch, ["single_view", "single_view"])
    record["sources"]["VPT-01"].update(single_clips=1, multi_view_clips=1)
    with pytest.raises(ValueError, match="Actual manifest view counts differ"):
        validate_view_mixture(snapshot, record, require_multiview=True)


def test_mixed_gate_rejects_an_unknown_representation(tmp_path, monkeypatch):
    snapshot, record = publish(tmp_path, monkeypatch, ["single_view", "multi_view"])
    path = snapshot / "VPT-01.csv"
    path.write_text(path.read_text().replace("multi_view", "unverified_composite"))
    with pytest.raises(ValueError, match="Unknown pretraining view representation"):
        validate_view_mixture(snapshot, record, require_multiview=True)


def test_mixed_gate_rejects_a_multi_view_summary_without_a_manifest(
    tmp_path, monkeypatch
):
    snapshot, record = publish(tmp_path, monkeypatch, ["single_view"])
    record["sources"]["VPT-09"] = dict(single_clips=0, multi_view_clips=1, clips=1)
    with pytest.raises(
        ValueError, match="source summaries differ from actual manifest sources"
    ):
        validate_view_mixture(snapshot, record, require_multiview=True)
