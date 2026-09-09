"""Fixed, conservative per-video FastUMI border removal before multi view rendering."""

import numpy as np

from open_wam.data.preparation.multiview.inner_roi import inner_box
from open_wam.data.preparation.multiview.sources import open_cursors as open_native


class CroppedCursor:
    def __init__(self, cursor, spec):
        self.cursor, self.crop_spec = cursor, spec
        self.box = tuple(spec["box"])
        x0, y0, x1, y1 = self.box
        if [cursor.width, cursor.height] != spec["native_size"]:
            raise ValueError("Native geometry differs from crop preflight")
        self.width, self.height = x1 - x0, y1 - y0

    def __getattr__(self, key):
        return getattr(self.cursor, key)

    def at(self, timestamp):
        frame = self.cursor.at(timestamp)
        x0, y0, x1, y1 = self.box
        if frame.shape != (self.cursor.height, self.cursor.width, 3):
            raise ValueError("Source geometry changed within a video")
        return np.ascontiguousarray(frame[y0:y1, x0:x1])

    def close(self):
        self.cursor.close()


def close_all(cursors, owned):
    for item in list(cursors.values()) + list(reversed(owned)):
        try:
            item.close()
        except Exception:
            pass


def open_cursors(plan, store, hdf_path=None):
    if plan["source_id"] != "VPT-09":
        return open_native(plan, store, hdf_path)
    if plan["pair_orientation"] != "horizontal":
        raise ValueError("FastUMI requires a horizontal pair")
    cursors, owned = open_native(plan, store, hdf_path)
    crops = {}
    try:
        for camera, cursor in cursors.items():
            times = sorted(
                set(
                    max(0, cursor.duration - 1 / cursor.fps) * q
                    for q in (0, 0.25, 0.5, 0.75, 1)
                )
            )
            box, facts = inner_box([cursor.at(t) for t in times])
            crops[camera] = dict(
                native_size=[cursor.width, cursor.height],
                box=box,
                sample_times_seconds=times,
                source_etag=cursor.reader.etag,
                **facts,
            )
    finally:
        close_all(cursors, owned)
    cursors, owned = open_native(plan, store, hdf_path)
    try:
        for camera, cursor in cursors.items():
            if cursor.reader.etag != crops[camera]["source_etag"]:
                raise ValueError("Raw source changed during crop preflight")
        return {c: CroppedCursor(cur, crops[c]) for c, cur in cursors.items()}, owned
    except BaseException:
        close_all(cursors, owned)
        raise
