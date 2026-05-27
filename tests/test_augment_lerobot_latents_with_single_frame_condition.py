from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "augment_lerobot_latents_with_single_frame_condition.py"
    spec = importlib.util.spec_from_file_location("augment_lerobot_latents_with_single_frame_condition", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_condition_source_frame_indices_follow_next_wan_source_boundaries() -> None:
    module = _load_script_module()

    indices = module._condition_source_frame_indices(
        frame_ids=list(range(10, 25)),
        latent_num_frames=4,
    )

    assert indices == [11, 15, 19, 23]


def test_condition_source_frame_indices_support_previous_frame_offset() -> None:
    module = _load_script_module()

    indices = module._condition_source_frame_indices(
        frame_ids=list(range(10, 25)),
        latent_num_frames=4,
        source_frame_offset=-1,
    )

    assert indices == [10, 14, 18, 22]


def test_condition_source_frame_indices_clamp_explicit_frame_ids() -> None:
    module = _load_script_module()

    indices = module._condition_source_frame_indices(
        frame_ids=[3, 7, 11],
        latent_num_frames=5,
    )

    assert indices == [7, 11, 11, 11, 11]


def test_libero_canonical_video_from_single_view_duplicates_width_slots() -> None:
    module = _load_script_module()
    single_view = torch.randn(2, 3, 1, 128, 128)

    canonical = module._libero_canonical_video_from_single_view(single_view)

    assert canonical.shape == (2, 3, 1, 128, 256)
    torch.testing.assert_close(canonical[..., :128], single_view)
    torch.testing.assert_close(canonical[..., 128:], single_view)


def test_libero_camera_slot_maps_known_camera_names() -> None:
    module = _load_script_module()

    assert module._libero_camera_slot("observation.images.agentview_rgb") == 0
    assert module._libero_camera_slot("observation.images.eye_in_hand_rgb") == 1
    assert module._libero_camera_slot("observation.images.wrist_image") == 1
    with pytest.raises(ValueError, match="Could not map LIBERO camera name"):
        module._libero_camera_slot("observation.images.side_rgb")


def test_batch_encoding_max_diff_skips_batch_check_for_size_one() -> None:
    module = _load_script_module()

    class FakeAssets:
        def __init__(self) -> None:
            self.calls = 0

        def encode_video(self, video, *, placements=None, reset_cache=True):
            self.calls += 1
            return torch.zeros(video.shape[0], 48, 1, 8, 8)

    assets = FakeAssets()
    diff = module._batch_encoding_max_diff(
        torch.zeros(1, 3, 1, 128, 128),
        placements=None,
        assets=assets,
        batch_size=1,
    )

    assert diff == 0.0
    assert assets.calls == 0


def test_build_payload_tasks_groups_libero_camera_pair(tmp_path: Path) -> None:
    module = _load_script_module()
    root = tmp_path / "latents" / "chunk-000"
    agent = root / "observation.images.agentview_rgb" / "episode_000000_0_272.pth"
    wrist = root / "observation.images.eye_in_hand_rgb" / "episode_000000_0_272.pth"
    side = root / "observation.images.side_rgb" / "episode_000001_0_272.pth"
    for path in (agent, wrist, side):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    tasks = module._build_payload_tasks([side, wrist, agent])

    assert len(tasks) == 2
    assert set(tasks[0]) == {agent, wrist}
    assert tasks[1] == (side,)


def test_save_payload_atomic_replaces_payload_without_tmp_leftover(tmp_path: Path) -> None:
    module = _load_script_module()
    payload_path = tmp_path / "episode_000000.pth"
    torch.save({"old": torch.tensor([1])}, payload_path)

    module._save_payload_atomic({"new": torch.tensor([2])}, payload_path)

    loaded = torch.load(payload_path, map_location="cpu", weights_only=False)
    assert loaded["new"].item() == 2
    assert not payload_path.with_name(f"{payload_path.name}.tmp").exists()
