from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from open_wam.configs.enums import RolloutArtifactProfile
from open_wam.evals import libero_rollout_artifacts as artifacts


def _observation(index: int) -> dict[str, np.ndarray]:
    frame = np.full((2, 2, 3), index, dtype=np.uint8)
    return {
        artifacts.LIBERO_OBS_KEYS[0]: frame,
        artifacts.LIBERO_OBS_KEYS[1]: frame + 1,
    }


def test_realtime_artifact_policy_uses_typed_profile_semantics() -> None:
    lean = artifacts.RolloutArtifactPolicy.from_value("lean")
    standard = artifacts.RolloutArtifactPolicy.from_value(
        RolloutArtifactProfile.STANDARD
    )
    forced_timeline = artifacts.RolloutArtifactPolicy.from_value(
        RolloutArtifactProfile.LEAN,
        write_fallback_timeline_video=True,
    )
    directly_coerced = artifacts.RolloutArtifactPolicy(profile="debug")  # type: ignore[arg-type]

    assert lean.profile is RolloutArtifactProfile.LEAN
    assert not lean.writes_rollout_video
    assert not lean.writes_debug_artifacts
    assert not lean.collects_video_records
    assert standard.writes_rollout_video
    assert standard.writes_debug_artifacts
    assert not standard.writes_fallback_timeline_video
    assert forced_timeline.writes_fallback_timeline_video
    assert forced_timeline.collects_video_records
    assert directly_coerced.profile is RolloutArtifactProfile.DEBUG


def test_build_realtime_output_stem_sanitizes_prompt_and_suffix(
    tmp_path: Path,
) -> None:
    output_stem = artifacts.build_libero_realtime_output_stem(
        root=tmp_path,
        identity=artifacts.LiberoRealtimeArtifactIdentity(
            benchmark="libero_10",
            task_id=1,
            prompt="put / both: things? in <basket>",
            episode_idx=7,
            suffix="step600/unsafe",
        ),
    )

    assert output_stem.parent.name == "1_put_both_things_in_basket"
    assert output_stem.name == "7_step600_unsafe"


def test_append_predicted_latent_chunk_honors_frame_cap() -> None:
    chunks: list[torch.Tensor] = []

    artifacts.append_predicted_latent_chunk(
        chunks,
        torch.ones(1, 2, 3, 4, 4),
        max_imagined_latent_frames=5,
    )
    artifacts.append_predicted_latent_chunk(
        chunks,
        torch.ones(1, 2, 4, 4, 4) * 2.0,
        max_imagined_latent_frames=5,
    )

    assert [int(chunk.shape[2]) for chunk in chunks] == [3, 2]
    assert all(chunk.device.type == "cpu" for chunk in chunks)
    assert torch.all(chunks[1] == 2.0)


def test_append_predicted_latent_chunk_zero_cap_disables_collection() -> None:
    chunks: list[torch.Tensor] = []

    artifacts.append_predicted_latent_chunk(
        chunks,
        torch.ones(1, 2, 3, 4, 4),
        max_imagined_latent_frames=0,
    )

    assert chunks == []


def test_comparison_frames_resample_imagined_video_lazily() -> None:
    real_observations = [_observation(1), _observation(2), _observation(3)]
    imagined_video = np.stack(
        [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 127, dtype=np.uint8),
            np.full((2, 2, 3), 255, dtype=np.uint8),
            np.full((2, 2, 3), 64, dtype=np.uint8),
            np.full((2, 2, 3), 32, dtype=np.uint8),
        ],
        axis=0,
    )

    frames = list(
        artifacts.iter_comparison_video_frames(
            real_observations=real_observations,
            imagined_video=imagined_video,
        )
    )

    assert len(frames) == len(real_observations)
    assert all(frame.flags["C_CONTIGUOUS"] for frame in frames)


def test_write_video_frames_streams_to_imageio_writer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    written: list[np.ndarray] = []

    class _Writer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def append_data(self, frame):
            written.append(np.array(frame, copy=True))

    def _fake_get_writer(path, *, fps):
        assert path == tmp_path / "out.mp4"
        assert fps == 7.0
        return _Writer()

    monkeypatch.setattr(artifacts.imageio, "get_writer", _fake_get_writer)

    artifacts.write_video_frames(
        tmp_path / "out.mp4",
        [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
        ],
        fps=7.0,
    )

    assert len(written) == 2


def test_persist_rollout_artifacts_preserves_legacy_schema_and_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    written_videos: list[tuple[Path, float, list[np.ndarray]]] = []

    def _write_video(path: Path, frames, *, fps: float) -> None:
        materialized = list(frames)
        path.write_bytes(b"video")
        written_videos.append((path, fps, materialized))

    monkeypatch.setattr(artifacts, "write_video_frames", _write_video)
    summary = {
        "benchmark": "libero_10",
        "task_id": 2,
        "prompt": "put object in basket",
        "episode_idx": 3,
        "success": False,
        "terminal": False,
        "chunk_count": 1,
        "env_timestep": 21,
        "seed": 3,
        "video_path": None,
        "comparison_video_path": None,
        "rollout_video_path": None,
        "pipeline": "open_wam_mot",
        "runtime_mode": "non_joint_two_stream",
        "condition_mode": "teacher_forcing_cond_video",
        "startup_model_obs_frames": 1,
        "startup_env_init_steps": 5,
        "startup_env_steps_executed": 5,
        "execute_action_steps": None,
        "execute_frame_chunk_size": None,
        "action_count": 2,
        "checkpoint_file": "/checkpoint/model_state.pt",
        "mot_gjd_action_route": "joint",
    }
    output = artifacts.persist_libero_rollout_artifacts(
        pipeline=SimpleNamespace(),  # type: ignore[arg-type]
        identity=artifacts.LiberoRolloutArtifactIdentity(
            benchmark="libero_10",
            task_id=2,
            prompt="put object in basket",
            episode_idx=3,
            success=False,
            suffix="test",
        ),
        options=artifacts.LiberoRolloutArtifactOptions(
            output_root=tmp_path,
            video_fps=15.0,
            save_rollout_video=True,
        ),
        payload=artifacts.LiberoRolloutArtifactPayload(
            real_observations=(_observation(1), _observation(2)),
            predicted_latent_chunks=(),
            action_trace=(
                np.asarray([0.1, 0.2], dtype=np.float64),
                np.asarray([-0.3, 0.4], dtype=np.float32),
            ),
            chunk_events=({"phase": "infer", "value": Path("debug")},),
            component_report={"loaded_keys": 848},
        ),
        summary=summary,
        decode_device=torch.device("cpu"),
    )

    expected_root = (
        tmp_path
        / "libero_10"
        / "2_put_object_in_basket"
    )
    expected_video = expected_root / "3_False_test.mp4"
    assert output.comparison_video_path == expected_video.resolve()
    assert output.rollout_video_path == (
        expected_root / "3_False_test_rollout.mp4"
    ).resolve()
    assert [path for path, _, _ in written_videos] == [
        expected_video,
        expected_root / "3_False_test_rollout.mp4",
    ]
    assert all(fps == 15.0 for _, fps, _ in written_videos)
    assert all(len(frames) == 2 for _, _, frames in written_videos)

    persisted_summary = json.loads(output.summary_path.read_text(encoding="utf-8"))
    assert persisted_summary == output.summary
    assert persisted_summary["video_path"] == str(expected_video.resolve())
    assert persisted_summary["action_trace_path"] == str(
        output.action_trace_path.resolve()
    )
    action_rows = [
        json.loads(line)
        for line in output.action_trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert action_rows == [
        {
            "action_index": 0,
            "action": np.asarray([0.1, 0.2], dtype=np.float32).tolist(),
        },
        {
            "action_index": 1,
            "action": np.asarray([-0.3, 0.4], dtype=np.float32).tolist(),
        },
    ]
    assert json.loads(output.chunk_events_path.read_text(encoding="utf-8")) == [
        {"phase": "infer", "value": "debug"}
    ]
    assert json.loads(
        output.component_report_path.read_text(encoding="utf-8")
    ) == {"loaded_keys": 848}


def test_persist_realtime_artifacts_preserves_schema_and_render_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    written_videos: list[tuple[Path, float, list[np.ndarray]]] = []

    def _mimsave(path: Path, frames, *, fps: float) -> None:
        materialized = [np.array(frame, copy=True) for frame in frames]
        path.write_bytes(b"video")
        written_videos.append((path, fps, materialized))

    monkeypatch.setattr(artifacts.imageio, "mimsave", _mimsave)
    action_video_records = (
        {
            "obs": _observation(1),
            "action_index": 0,
            "absolute_frame_index": 1,
            "action_offset": 0,
            "source": "startup_plan",
            "generation_lag_frames": 0,
            "lateness_s": 0.001,
            "env_step_s": 0.002,
            "frame_history_decision": "history",
            "frame_contains_fallback_action": False,
            "generation_frame_start": 1,
            "plan_ready_delay_s": 0.003,
            "action": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0],
        },
        {
            "obs": _observation(2),
            "action_index": 1,
            "absolute_frame_index": 1,
            "action_offset": 1,
            "source": "fallback_hold_state",
            "generation_lag_frames": None,
            "lateness_s": 0.01,
            "env_step_s": 0.02,
            "frame_history_decision": "washout",
            "frame_contains_fallback_action": True,
            "generation_frame_start": None,
            "plan_ready_delay_s": None,
            "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        },
    )
    summary = {"target_action_hz": 10.0, "success": False}

    output = artifacts.persist_libero_realtime_artifacts(
        identity=artifacts.LiberoRealtimeArtifactIdentity(
            benchmark="libero_10",
            task_id=2,
            prompt="put object in basket",
            episode_idx=3,
            suffix="debug",
        ),
        options=artifacts.LiberoRealtimeArtifactOptions(
            output_root=tmp_path,
            video_fps=7.5,
            action_per_frame=4,
            policy=artifacts.RolloutArtifactPolicy.from_value("debug"),
        ),
        payload=artifacts.LiberoRealtimeArtifactPayload(
            action_records=({"action_index": 0, "action": [0.1, -0.2]},),
            action_video_records=action_video_records,
            replan_records=({"event": "replan", "frame": 1},),
            extension_records=({"event": "extension", "frame": 5},),
            component_report={"loaded_keys": 848},
            startup_debug_report={"startup": "one_observation"},
        ),
        summary=summary,
    )

    expected_stem = (
        tmp_path
        / "libero_10"
        / "2_put_object_in_basket"
        / "3_debug"
    )
    assert output.summary is summary
    assert output.summary_path == expected_stem.with_suffix(".json")
    assert output.video_path == expected_stem.with_suffix(".mp4")
    assert output.fallback_timeline_video_path == expected_stem.with_name(
        "3_debug_fallback_timeline.mp4"
    )
    assert output.action_trace_path == expected_stem.with_name(
        "3_debug_actions.jsonl"
    )
    assert output.replan_trace_path == expected_stem.with_name(
        "3_debug_replans.jsonl"
    )
    assert output.extension_trace_path == expected_stem.with_name(
        "3_debug_extensions.jsonl"
    )
    assert output.load_report_path == expected_stem.with_name(
        "3_debug_load_report.json"
    )
    assert output.startup_debug_path == expected_stem.with_name(
        "3_debug_startup_debug.json"
    )
    assert output.summary["artifact_profile"] == "debug"
    assert [path for path, _, _ in written_videos] == [
        output.video_path,
        output.fallback_timeline_video_path,
    ]
    assert all(fps == 7.5 for _, fps, _ in written_videos)
    assert [len(frames) for _, _, frames in written_videos] == [2, 2]
    assert [frames[0].shape for _, _, frames in written_videos] == [
        (146, 4, 3),
        (256, 16, 3),
    ]
    assert all(
        frame.flags["C_CONTIGUOUS"]
        for _, _, frames in written_videos
        for frame in frames
    )
    assert output.action_trace_path.read_text(encoding="utf-8") == (
        '{"action": [0.1, -0.2], "action_index": 0}\n'
    )
    assert json.loads(output.summary_path.read_text(encoding="utf-8")) == summary
    assert json.loads(output.load_report_path.read_text(encoding="utf-8")) == {
        "loaded_keys": 848
    }
    assert json.loads(output.startup_debug_path.read_text(encoding="utf-8")) == {
        "startup": "one_observation"
    }
