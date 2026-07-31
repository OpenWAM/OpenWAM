from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from open_wam.evals import libero_rollout_artifacts as artifacts


def _observation(index: int) -> dict[str, np.ndarray]:
    frame = np.full((2, 2, 3), index, dtype=np.uint8)
    return {
        artifacts.LIBERO_OBS_KEYS[0]: frame,
        artifacts.LIBERO_OBS_KEYS[1]: frame + 1,
    }


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
