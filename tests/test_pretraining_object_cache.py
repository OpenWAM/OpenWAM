import hashlib
import io
import multiprocessing
import tempfile
import threading
import time
import unittest
from pathlib import Path

from open_wam.artifacts.object_cache import CacheFull, ObjectCache


class FakeClient:
    def __init__(self, source):
        self.source = Path(source)

    def get_object(self, **kw):
        with (self.source / "gets").open("a") as f:
            f.write(kw["Key"] + "\n")
        return {"Body": io.BytesIO((self.source / kw["Key"]).read_bytes())}


def spec(source, data):
    sha = hashlib.sha256(data).hexdigest()
    (Path(source) / sha).write_bytes(data)
    return {"sha256": sha, "key": sha, "bucket": "test", "bytes": len(data)}


def worker(root, source, item, barrier, deadline):
    cache = ObjectCache(root, 100, 0, lambda: FakeClient(source))
    barrier.wait(timeout=max(0, deadline - time.monotonic()))
    with cache.acquire(item) as p:
        assert p.read_bytes() == (Path(source) / item["key"]).read_bytes()
        time.sleep(0.05)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "cache"
        self.source = Path(self.tmp.name) / "source"
        self.source.mkdir()
        self.cache = ObjectCache(
            self.root, 100, 0, lambda: FakeClient(self.source), wait_seconds=0.05
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_singleflight_across_processes(self):
        item = spec(self.source, b"a" * 40)
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(8)
        deadline = time.monotonic() + 120
        ps = [
            ctx.Process(
                target=worker,
                args=(str(self.root), str(self.source), item, barrier, deadline),
            )
            for _ in range(8)
        ]
        try:
            for p in ps:
                p.start()
            for p in ps:
                p.join(max(0, deadline - time.monotonic()))
                self.assertEqual(p.exitcode, 0)
        finally:
            for p in ps:
                if p.is_alive():
                    p.terminate()
            for p in ps:
                if p.pid is not None:
                    p.join()
        self.assertEqual(len((self.source / "gets").read_text().splitlines()), 1)

    def test_eviction_protects_reader_and_budget(self):
        a = spec(self.source, b"a" * 60)
        b = spec(self.source, b"b" * 60)
        with self.cache.acquire(a) as p:
            with self.assertRaises(CacheFull):
                with self.cache.acquire(b):
                    pass
            self.assertTrue(p.exists())
            self.assertLessEqual(self.cache.stats()["reserved_and_ready_bytes"], 100)
        with self.cache.acquire(b):
            pass
        self.assertFalse(self.cache._path(a["sha256"]).exists())
        self.assertEqual(self.cache.stats()["ready_bytes"], 60)

    def test_corruption_redownloaded(self):
        a = spec(self.source, b"a" * 40)
        with self.cache.acquire(a) as p:
            pass
        p.write_bytes(b"x" * 40)
        with self.cache.acquire(a) as p:
            self.assertEqual(p.read_bytes(), b"a" * 40)
        self.assertEqual(len((self.source / "gets").read_text().splitlines()), 2)

    def test_bad_remote_response_never_published(self):
        a = spec(self.source, b"a" * 40)
        (self.source / a["key"]).write_bytes(b"x" * 40)
        with self.assertRaises(ValueError):
            with self.cache.acquire(a):
                pass
        self.assertEqual(self.cache.stats()["reserved_and_ready_bytes"], 0)
        self.assertFalse(self.cache._path(a["sha256"]).exists())

    def test_download_in_progress_counts_against_budget(self):
        a = spec(self.source, b"a" * 60)
        b = spec(self.source, b"b" * 60)
        entered = threading.Event()
        release = threading.Event()
        errors = []
        source = self.source

        class Slow(FakeClient):
            def get_object(self, **kw):
                entered.set()
                release.wait(5)
                return super().get_object(**kw)

        slow = ObjectCache(self.root, 100, 0, lambda: Slow(source), wait_seconds=0.05)

        def run():
            try:
                with slow.acquire(a):
                    pass
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(entered.wait(2))
        try:
            self.assertEqual(self.cache.stats()["downloading_bytes"], 60)
            with self.assertRaises(CacheFull):
                with self.cache.acquire(b):
                    pass
            self.assertEqual(self.cache.stats()["occupied_bytes"], 60)
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(errors)

    def test_restart_recovers_abandoned_reservation(self):
        a = spec(self.source, b"a" * 60)
        b = spec(self.source, b"b" * 60)
        with self.cache._db() as db:
            db.execute(
                "INSERT INTO entries VALUES (?,?,?,?)",
                (a["sha256"], 60, "downloading", 0),
            )
            self.cache._bump(db, "occupied_bytes", 60)
        tmp = self.root / "tmp" / (a["sha256"] + ".part")
        tmp.write_bytes(b"partial")
        with self.cache.acquire(b):
            pass
        self.assertFalse(tmp.exists())
        self.assertEqual(self.cache.stats()["reserved_and_ready_bytes"], 60)


if __name__ == "__main__":
    unittest.main()
