from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from open_wam.integrations.image_ops import resize_nearest_to_height


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("shape,target_h", [((3, 5, 3), 7), ((7, 5, 3), 3), ((6, 5, 3), 3), ((8, 1, 3), 1)])
def test_nearest_resize_preserves_pixels_and_width_rounding(dtype, shape, target_h) -> None:
    frame = np.arange(np.prod(shape)).reshape(shape).astype(dtype)[::-1]
    scale = target_h / shape[0]
    target_w = max(1, round(shape[1] * scale))
    expected = np.empty((target_h, target_w, shape[2]), dtype=dtype)
    for y in range(target_h):
        for x in range(target_w):
            expected[y, x] = frame[min(int(y / scale), shape[0] - 1), min(int(x / scale), shape[1] - 1)]

    actual = resize_nearest_to_height(frame, target_h)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == frame.dtype


def test_nearest_resize_retains_noop_alias_and_does_not_mutate_input() -> None:
    frame = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    before = frame.copy()
    assert resize_nearest_to_height(frame, 4) is frame
    resized = resize_nearest_to_height(frame, 7)
    resized[:] = 0
    np.testing.assert_array_equal(frame, before)


@pytest.mark.parametrize("benchmark", ["calvin", "robotwin"])
def test_render_composes_unequal_camera_sizes_in_order(benchmark) -> None:
    from open_wam.integrations.calvin_env import CalvinBenchmarkAdapter
    from open_wam.integrations.robotwin_env import RobotwinBenchmarkAdapter

    large = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    small = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    upsampled = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    if benchmark == "calvin":
        render = CalvinBenchmarkAdapter.render_frame
        views = {"rgb_static": large, "rgb_gripper": small}
        expected = np.concatenate([large, upsampled], axis=1)
    else:
        render = RobotwinBenchmarkAdapter.render_frame
        views = {
            "observation.images.cam_high": large,
            "observation.images.cam_left_wrist": small,
            "observation.images.cam_right_wrist": small + 20,
        }
        expected = np.concatenate([large, upsampled, upsampled + 20], axis=1)

    adapter = SimpleNamespace(extract_views=lambda observation: observation)
    np.testing.assert_array_equal(render(adapter, views), expected)
