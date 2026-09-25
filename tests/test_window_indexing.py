from __future__ import annotations

from types import SimpleNamespace

import pytest

from open_wam.data.window_indexing import observation_window_starts


@pytest.mark.parametrize("num_frames,frame_stride,action_horizon,sample_stride", [
    (1, 1, 1, 1), (4, 1, 16, 1), (3, 2, 4, 3), (4, 3, 1, 2),
])
def test_window_starts_match_last_required_index(num_frames, frame_stride, action_horizon, sample_stride) -> None:
    for episode_length in range(40):
        starts = observation_window_starts(
            episode_length=episode_length, num_frames=num_frames,
            frame_stride=frame_stride, action_horizon=action_horizon,
            sample_stride=sample_stride,
        )
        expected = [
            start for start in range(0, episode_length, sample_stride)
            if start + (num_frames - 1) * frame_stride + action_horizon - 1 < episode_length
        ]
        assert isinstance(starts, range)
        assert list(starts) == expected


def test_window_starts_preserve_empty_episode_and_stride_behavior() -> None:
    kwargs = dict(num_frames=4, frame_stride=1, action_horizon=4, sample_stride=0)
    assert list(observation_window_starts(episode_length=6, **kwargs)) == []
    with pytest.raises(ValueError, match="range"):
        observation_window_starts(episode_length=7, **kwargs)


@pytest.mark.parametrize("adapter_name", ["libero_hdf5", "lerobot_v2", "lerobot_video"])
@pytest.mark.parametrize("sample_stride", [1, 3, 20])
def test_adapter_window_order_type_and_short_episode_filtering(adapter_name, sample_stride) -> None:
    from open_wam.data import libero_hdf5, lerobot_v2, lerobot_video

    module, dataset_type = {
        "libero_hdf5": (libero_hdf5, libero_hdf5.LiberoOfflineWindowDataset),
        "lerobot_v2": (lerobot_v2, lerobot_v2.LeRobotV2WindowDataset),
        "lerobot_video": (lerobot_video, lerobot_video.LeRobotV2VideoWindowDataset),
    }[adapter_name]
    dataset = object.__new__(dataset_type)
    dataset.data_config = SimpleNamespace(
        num_frames=3, frame_stride=2, sample_stride=sample_stride,
        action_schema=SimpleNamespace(action_horizon=4),
    )
    dataset.episodes = (9, 2, 17, 4, 9)
    dataset.episode_records = {
        index: SimpleNamespace(length=length) for index, length in ((9, 15), (2, 0), (17, 7), (4, 8))
    }
    expected = [
        module.EpisodeWindow(episode_index=index, observation_start=start)
        for index in dataset.episodes
        for start in range(0, max(0, dataset.episode_records[index].length - 8 + 1), sample_stride)
    ]

    actual = dataset._build_sample_index()

    assert actual == expected
    assert all(type(window) is module.EpisodeWindow for window in actual)
