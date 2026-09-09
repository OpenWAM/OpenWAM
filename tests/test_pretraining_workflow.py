"""Portable metadata/geometry/manifest workflow checks with small local fixtures."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
import open_wam.data.preparation.build_manifest as build_manifest
import open_wam.data.preparation.prepare_dataset as prepare_dataset
import open_wam.data.preparation.publish_snapshot as publish_snapshot
from open_wam.data.preparation.camera_recipes import camera_recipe, select_cameras
from open_wam.data.preparation.multiview.inner_roi import inner_box
from open_wam.data.preparation.multiview.layout import make_layout, render_frame
from open_wam.data.preparation.multiview.prepare import make_plan


@pytest.mark.parametrize("source", ["VPT-01", "custom_source"])
@pytest.mark.parametrize("mode", ["strict", "padded", "bucket", "packed"])
def test_native_metadata_through_snapshot_and_training(
    tmp_path, monkeypatch, source, mode
):
    raw = tmp_path / "raw" / "demo"
    meta = raw / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            dict(
                codebase_version="v2.1",
                fps=30,
                total_episodes=1,
                features={"left": {"dtype": "video"}, "right": {"dtype": "video"}},
            )
        )
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps(dict(episode_index=0, length=18, tasks=["place object"])) + "\n"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps(dict(task_index=0, task="place object")) + "\n"
    )
    args = SimpleNamespace(source=source, raw_root=str(raw))
    episodes = list(prepare_dataset.lerobot(args))
    index = tmp_path / "episodes.jsonl"
    index.write_text(json.dumps(episodes[0]) + "\n")
    latent_root = tmp_path / "latents"
    for camera in ("left", "right"):
        path = (
            latent_root
            / "demo"
            / "latents/chunk-000"
            / camera
            / "episode_000000_0_18.pth"
        )
        path.parent.mkdir(parents=True)
        torch.save(
            dict(
                latent=torch.arange(3 * 8 * 8 * 48)
                .reshape(3, 8, 8, 48)
                .div(4096)
                .half(),
                latent_layout="THWC",
                latents_normalized=True,
                video_num_frames=9,
                video_height=128,
                video_width=128,
                fps=15.0,
                ori_fps=30.0,
            ),
            path,
        )
    rows = list(
        build_manifest.single_rows(
            SimpleNamespace(
                source=source, episodes=str(index), latent_root=str(latent_root)
            )
        )
    )
    assert len(rows) == 2
    assert len({r["clip_id"] for r in rows}) == 2
    assert len({r["physical_episode_key"] for r in rows}) == 1
    assert {r["length_frames"] for r in rows} == {3}  # latent units, not 18 raw frames
    plan, _ = make_plan(episodes[0], **camera_recipe(episodes[0]))
    assert plan["physical_episode_key"] == rows[0]["physical_episode_key"]
    assert plan["pair_orientation"] == ("horizontal" if source == "VPT-01" else "auto")
    manifest = tmp_path / "single.csv"
    build_manifest.write_csv(manifest, rows)
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(
        sys, "argv", ["publish", "--manifests", str(manifest), "--out", str(snapshot)]
    )
    publish_snapshot.main()
    import open_wam.cli.video_pretraining_make_config as make_config
    from tests.test_offline_prompt_cache import _cache

    cache_root = tmp_path / "text"
    _cache(cache_root, extra_prompt=True)

    cfgpath = tmp_path / "config.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_config",
            "--snapshot",
            str(snapshot),
            "--model-assets",
            str(tmp_path / "models"),
            "--run-root",
            str(tmp_path / "runs"),
            "--out",
            str(cfgpath),
            "--allow-subset",
            "--text-cache",
            str(cache_root),
            "--batch-size",
            "1",
            "--batching",
            mode,
        ],
    )
    make_config.main()
    from open_wam.configs import load_experiment_config

    cfg = load_experiment_config(cfgpath)
    assert cfg.data.batching.mode.value == mode
    assert cfg.data.target_observation_fps is None
    assert cfg.trainer.save_interval == 1000
    from open_wam.cli.video_pretraining_train import validate_snapshot

    assert set(validate_snapshot(snapshot, cfg)["sources"]) == {source}
    from open_wam.data.mixed_video import MixedVideoLatentWindowDataset
    from open_wam.data.mixed_video_catalog_assembly import load_mixed_video_catalog
    from open_wam.evals.video_prediction import rollout_causal_video_prediction
    from open_wam.pipelines import build_variant_pipeline_from_config
    from open_wam.training import LatentBatchAdapter, PipelineTrainStepExecutor
    from open_wam.training.data_loading import _build_variable_length_loader
    from tests.test_causal_video_prediction import (
        _tiny_causal_video_pipeline,
        _deterministic_cpu_math,
    )

    data = replace(cfg.data, num_workers=0)
    catalog = load_mixed_video_catalog(data)
    dataset = MixedVideoLatentWindowDataset(
        data,
        catalog=catalog,
        split="train",
        episode_keys=tuple(ep.key for ep in catalog.episodes),
    )
    loader = _build_variable_length_loader(
        replace(cfg, data=data),
        dataset,
        None,
        batch_size=1,
        shuffle=False,
        train=True,
    )
    batch = next(iter(loader))
    assert batch.video_latents.shape[1:] == (48, 64 if mode == "strict" else 3, 8, 8)
    stored = torch.load(rows[0]["latent_path"], weights_only=True)["latent"]
    torch.testing.assert_close(
        batch.video_latents[0, :, :3],
        stored.permute(3, 0, 1, 2).float(),
        rtol=0,
        atol=0,
    )
    assert batch.metadata[0]["observed_prefix_frames"] == 1
    assert batch.metadata[0]["future_suffix_frames"] == 2
    with _deterministic_cpu_math():
        tiny, _, _ = _tiny_causal_video_pipeline(text_condition_dropout_prob=0.1)
        tiny = replace(
            tiny,
            backbone=replace(
                tiny.backbone,
                prompt_cache=cfg.backbone.prompt_cache,
                text_dim=4,
            ),
        )
        pipeline = build_variant_pipeline_from_config(tiny)
        executor = PipelineTrainStepExecutor(
            pipeline=pipeline,
            batch_adapter=LatentBatchAdapter(),
            training_config=tiny.training,
        )
        result = executor.forward_train(batch)
        result.loss.backward()
        assert torch.isfinite(result.loss)
        gradients = [p.grad for p in pipeline.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
        assert any(grad.count_nonzero() for grad in gradients)
        pipeline.eval()
        rollout = rollout_causal_video_prediction(pipeline, batch)
        assert rollout.predicted_latents.shape == (1, 48, 3, 8, 8)
        assert torch.isfinite(rollout.predicted_latents).all()
        torch.testing.assert_close(
            rollout.predicted_latents[:, :, :1], batch.video_latents[:, :, :1],
            rtol=0, atol=0,
        )
    (snapshot / f"{source}.csv").write_text("changed bytes")
    with pytest.raises(ValueError, match="manifest changed"):
        validate_snapshot(snapshot, cfg)
    with pytest.raises(ValueError, match="manifest changed"):
        make_config.main()


@pytest.mark.parametrize("source", ["VPT-01", "custom_source"])
@pytest.mark.parametrize("change", ["rewrite", "remove", "add", "tamper"])
def test_snapshot_rejects_rewriting_previous_sample(tmp_path, monkeypatch, source, change):
    row = dict(
        source_id=source,
        clip_id="a",
        physical_episode_key="physical",
        latent_sha256="a" * 64,
        latent_path="/logical/a",
        augmentation="single_view",
        video_num_frames=5,
        observation_fps=15.0,
        task="move",
    )
    first = tmp_path / "a.csv"
    build_manifest.write_csv(first, [row])
    out = tmp_path / "first"
    monkeypatch.setattr(
        sys, "argv", ["publish", "--manifests", str(first), "--out", str(out)]
    )
    publish_snapshot.main()
    second = tmp_path / "b.csv"
    replacement = dict(row, latent_sha256="b" * 64)
    if change == "remove":
        replacement["clip_id"] = "new_clip"
    next_rows = [replacement]
    if change == "add":
        next_rows = [row, dict(row, clip_id="new_clip")]
    if change == "tamper":
        (out / f"{source}.csv").write_text("changed bytes")
    build_manifest.write_csv(second, next_rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish",
            "--manifests",
            str(second),
            "--out",
            str(tmp_path / "next"),
            "--previous",
            str(out),
        ],
    )
    if change == "add":
        publish_snapshot.main()
        from open_wam.data.preparation.snapshot import load_verified_snapshot

        record = load_verified_snapshot(tmp_path / "next")
        assert record["sources"][source]["clips"] == 2
    else:
        match = "manifest changed" if change == "tamper" else "changed or removed"
        with pytest.raises(ValueError, match=match):
            publish_snapshot.main()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"status": "incomplete"}, "complete"),
        ({"format_version": 2}, "format_version"),
        ({"sources": {}}, "inventory"),
        ({"manifests_sha256": {}}, "inventory"),
        ({"snapshot_sha256": "a" * 64}, "checksum"),
    ],
)
def test_snapshot_rejects_invalid_inventory(tmp_path, change, match):
    import hashlib

    from open_wam.data.preparation.snapshot import load_verified_snapshot

    manifests = {"custom.csv": "b" * 64}
    record = dict(
        format_version=1, status="complete", sources={"custom": {}},
        manifests_sha256=manifests,
        snapshot_sha256=hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest(),
    )
    record.update(change)
    (tmp_path / "snapshot.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match=match):
        load_verified_snapshot(tmp_path)


@pytest.mark.parametrize(
    "views",
    [
        [("a", 640, 480), ("b", 640, 480)],
        [("a", 720, 1280), ("b", 720, 1280)],
        [("head", 1920, 1080), ("left", 640, 480), ("right", 640, 480)],
    ],
)
def test_multi_view_preserves_aspect_and_pixel_budget(views):
    layout = make_layout(views)
    assert layout.width % 32 == layout.height % 32 == 0
    assert 0.8 * 65536 <= layout.width * layout.height <= 1.2 * 65536
    frames = {name: np.full((h, w, 3), (12, 70, 230), np.uint8) for name, w, h in views}
    image = render_frame(frames, layout)
    assert image.shape == (layout.height, layout.width, 3)
    for placement in layout.placements:
        assert placement.scale > 0


def test_current_fastumi_crop_preserves_full_height():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 240:1040] = 200
    box, facts = inner_box([frame] * 5)
    assert box[1:4:2] == [0, 720]
    assert box[0] <= 192 and box[2] >= 1088
    assert facts["retained_native_area_fraction"] >= 0.7
    assert inner_box([np.zeros_like(frame)])[0] == [0, 0, 1280, 720]


def test_robomind_top_and_ego_exclusion():
    cameras = [
        "camera_top",
        "camera_front",
        "camera_wrist_left",
        "camera_wrist_right",
        "other_left",
        "other_right",
    ]
    selected, _ = select_cameras("VPT-06", cameras)
    assert selected == ["camera_top", "camera_wrist_left", "camera_wrist_right"]
    row = prepare_dataset.make_row(
        "VPT-06",
        "demo",
        0,
        cameras,
        {c: 30 for c in cameras},
        {"type": "robomind_failure_hdf5"},
        "move",
        "native",
    )
    plan, _ = make_plan(row, **camera_recipe(row))
    assert plan["camera_rotations_degrees"] == {"camera_top": 180}
    for source in ("VPT-10R", "VPT-10S"):
        assert not select_cameras(source, cameras)[0]


def test_ego_rgb_and_slam_share_physical_take_identity():
    assert prepare_dataset.physical_key(
        "VPT-10R", "take__native", 0
    ) == prepare_dataset.physical_key("VPT-10S", "take__native", 0)


@pytest.mark.parametrize("new_data_phase", [False, True])
def test_pretraining_resume_resets_all_loader_cursors_only_for_new_data(
    monkeypatch, new_data_phase
):
    import open_wam.cli.video_pretraining_train as entrypoint
    from open_wam import training
    from open_wam.training import launch
    from open_wam.training.state import TrainState

    state = TrainState(
        global_step=24000,
        optimizer_step=12000,
        epoch_index=7,
        next_batch_index=19,
        seen_batches=24000,
        resume_source="checkpoint/full_training_state.pt",
    )
    original_state = state.state_dict()
    optimizer, scheduler = object(), object()
    events, observed = [], []
    runtime = SimpleNamespace(
        train_state=state,
        optimizer=optimizer,
        scheduler=scheduler,
        log_sink=SimpleNamespace(log_event=lambda **event: events.append(event)),
        run=lambda: observed.append(state.state_dict()),
    )
    config = SimpleNamespace(
        trainer=SimpleNamespace(resume_from="checkpoint/full_training_state.pt")
    )
    monkeypatch.setattr(training, "load_training_cli_config", lambda overrides: config)
    monkeypatch.setattr(
        training,
        "TrainingRuntime",
        SimpleNamespace(from_config=lambda *args, **kwargs: runtime),
    )
    monkeypatch.setattr(
        launch, "validate_training_launch", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        entrypoint, "validate_snapshot", lambda *args: {"snapshot_sha256": "a" * 64}
    )
    argv = [
        "train",
        "--snapshot",
        "new-snapshot",
        "--cfg",
        "config.yaml",
        "--expected-world-size",
        "1",
    ]
    if new_data_phase:
        argv.append("--new-data-phase")
    monkeypatch.setattr(sys, "argv", argv)

    entrypoint.main()

    expected = dict(original_state)
    if new_data_phase:
        expected.update(epoch_index=0, next_batch_index=0, seen_batches=0)
    assert observed == [expected]
    assert state.global_step == original_state["global_step"]
    assert state.optimizer_step == original_state["optimizer_step"]
    assert runtime.optimizer is optimizer
    assert runtime.scheduler is scheduler
    assert len(events) == 1 + int(new_data_phase)
    assert events[0]["name"] == "pretraining_view_mixture"
    if new_data_phase:
        assert events[-1]["name"] == "pretraining_new_data_phase"
        assert events[-1]["payload"]["optimizer_step"] == state.optimizer_step
