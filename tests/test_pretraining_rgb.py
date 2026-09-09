import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

# Preserve initialization of the geometry helper's module search path.
import open_wam.data.preparation.encoding.robomind_rgb as rgb


class OfficialRGBTests(unittest.TestCase):
    def test_typed_hwc_geometry_and_color_are_preserved_on_uniform_frames(self):
        for height, width in ((480, 640), (640, 480)):
            for embodiment in ("h5_simulation", "h5_franka_3rgb"):
                with (
                    self.subTest(shape=(height, width), embodiment=embodiment),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    native = np.full(
                        (2, height, width, 3), [17, 91, 223], dtype=np.uint8
                    )
                    path = Path(directory) / "trajectory.hdf5"
                    with h5py.File(path, "w") as handle:
                        handle.create_dataset(
                            "observations/rgb_images/camera", data=native
                        )
                    actual, meta = rgb.read_rgb_frames(
                        str(path), "camera", f"s3://source/{embodiment}/task.tar.gz"
                    )
                    expected = (
                        native
                        if embodiment == "h5_simulation"
                        else native[..., [2, 1, 0]]
                    )
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(meta["source_frame_encoding_counts"], {"raw": 2})

    def test_flat_ambiguous_frames_still_fail_without_explicit_geometry(self):
        with self.assertRaisesRegex(ValueError, "cannot tell"):
            rgb.decode_rgb_frame(bytes(480 * 640 * 3), "h5_simulation")

    def test_shaped_non_hwc_or_non_uint8_data_is_rejected(self):
        for native in (
            np.zeros((1, 3, 480, 640), dtype=np.uint8),
            np.zeros((1, 480, 640, 3), dtype=np.float32),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trajectory.hdf5"
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("observations/rgb_images/camera", data=native)
                with self.assertRaisesRegex(ValueError, "typed uint8 HWC"):
                    rgb.read_rgb_frames(
                        str(path), "camera", "s3://source/h5_simulation/task.tar.gz"
                    )

    def setUp(self):
        self.native = np.zeros((128, 128, 3), dtype=np.uint8)
        self.native[:] = [17, 91, 223]
        self.native[32:80, 20:65] = [202, 63, 9]
        ok, encoded = cv2.imencode(".jpg", self.native)
        assert ok
        self.jpeg = encoded.tobytes()

    def test_four_official_embodiment_and_storage_combinations(self):
        # This independently spells out the official notebook's output contract.
        decoded_jpeg = cv2.imdecode(
            np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR
        )
        for embodiment in ("h5_franka_3rgb", "h5_agilex_3rgb"):
            for storage in ("jpeg", "raw"):
                with self.subTest(embodiment=embodiment, storage=storage):
                    raw = self.jpeg if storage == "jpeg" else self.native.tobytes()
                    expected = decoded_jpeg if storage == "jpeg" else self.native
                    if embodiment == "h5_franka_3rgb":
                        expected = expected[:, :, [2, 1, 0]]
                    actual, observed = rgb.decode_rgb_frame(raw, embodiment)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(observed, storage)
                    self.assertTrue(actual.flags.c_contiguous)

    def test_mixed_storage_processes_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.hdf5"
            with h5py.File(path, "w") as handle:
                dataset = handle.create_dataset(
                    "observations/rgb_images/camera",
                    (2,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                )
                dataset[0] = np.frombuffer(self.jpeg, np.uint8)
                dataset[1] = np.frombuffer(self.native.tobytes(), np.uint8)
            frames, meta = rgb.read_rgb_frames(
                str(path), "camera", "s3://source/h5_franka_3rgb/task.tar.gz"
            )
            self.assertEqual(frames.shape, (2, 128, 128, 3))
            self.assertEqual(meta["source_frame_encoding"], "mixed")
            self.assertEqual(
                meta["source_frame_encoding_counts"], {"jpeg": 1, "raw": 1}
            )
            np.testing.assert_array_equal(frames[1], self.native[:, :, [2, 1, 0]])

    def test_archive_source_must_identify_one_known_embodiment(self):
        self.assertEqual(
            rgb.resolve_embodiment(
                "s3://source/benchmark1_0_compressed/h5_agilex_3rgb/18_makebread.tar.gz"
            ),
            "h5_agilex_3rgb",
        )
        for uri in (
            "s3://source/task.tar.gz",
            "s3://source/h5_new_robot/task.tar.gz",
            "s3://source/h5_franka_3rgb/h5_agilex_3rgb/task.tar.gz",
            "s3://source/task_h5_franka_3rgb.tar.gz",
            "h5_franka_3rgb",
        ):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                rgb.resolve_embodiment(uri)

    def test_existing_output_requires_matching_complete_color_provenance(self):
        uri = "s3://source/h5_franka_3rgb/task.tar.gz"
        good = dict(rgb.color_metadata("h5_franka_3rgb", {"jpeg": 4}), end_frame=4)
        self.assertEqual(
            rgb.validate_color_metadata(good, uri)["source_frame_encoding"], "jpeg"
        )
        for overrides in (
            {"color_policy": None},
            {"embodiment": "h5_agilex_3rgb"},
            {"source_frame_encoding": "raw"},
            {"end_frame": 9},
            {"source_frame_encoding_counts": {"jpeg": 0}},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                rgb.validate_color_metadata(dict(good, **overrides), uri)

    def test_invalid_frame_and_geometry_change_are_loud(self):
        for raw in (b"", b"not an image"):
            with self.assertRaises(ValueError):
                rgb.decode_rgb_frame(raw, "h5_franka_3rgb")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.hdf5"
            with h5py.File(path, "w") as handle:
                dataset = handle.create_dataset(
                    "observations/rgb_images/camera",
                    (2,),
                    dtype=h5py.vlen_dtype(np.dtype("uint8")),
                )
                dataset[0] = np.frombuffer(self.jpeg, np.uint8)
                _, different = cv2.imencode(".jpg", self.native[:64])
                dataset[1] = different
            with self.assertRaisesRegex(ValueError, "geometry changed"):
                rgb.read_rgb_frames(
                    str(path), "camera", "s3://source/h5_franka_3rgb/task.tar.gz"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
