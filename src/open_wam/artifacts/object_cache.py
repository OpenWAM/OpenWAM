"""Process-safe, byte-bounded node-local cache for immutable S3 tensor files.

Payload bytes, including downloads in progress, are reserved before a GET.
Readers hold shared object leases; eviction requires an exclusive lease.
Only this cache's own content-addressed files are eligible for eviction.
"""

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

GiB = 1024**3


class CacheFull(RuntimeError):
    pass


class ObjectCache:
    def __init__(
        self,
        root,
        max_bytes=200 * GiB,
        min_free_bytes=8 * GiB,
        client_factory=None,
        wait_seconds=60,
    ):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.wait_seconds = wait_seconds
        self.client_factory = client_factory
        self._client = None
        self._pid = None
        for name in ("objects", "locks", "tmp"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS entries (sha TEXT PRIMARY KEY,size INTEGER NOT NULL,state TEXT NOT NULL,atime REAL NOT NULL)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS entries_lru ON entries(atime)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS stats (name TEXT PRIMARY KEY,value INTEGER NOT NULL)"
            )
            if (
                db.execute(
                    "SELECT value FROM stats WHERE name='occupied_bytes'"
                ).fetchone()
                is None
            ):
                db.execute(
                    "INSERT INTO stats SELECT 'occupied_bytes',COALESCE(SUM(size),0) FROM entries"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY,budget INTEGER NOT NULL)"
            )
            old = db.execute("SELECT budget FROM config WHERE id=1").fetchone()
            if old and old[0] != self.max_bytes:
                raise ValueError(
                    "All users of a shared cache must specify the same byte budget"
                )
            db.execute("INSERT OR IGNORE INTO config VALUES (1,?)", (self.max_bytes,))

    @contextmanager
    def _db(self):
        with (self.root / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = sqlite3.connect(self.root / "state.sqlite", timeout=60)
            try:
                with db:
                    yield db
            finally:
                db.close()

    def _path(self, sha):
        return self.root / "objects" / sha[:2] / sha

    def _lock(self, sha):
        return (self.root / "locks" / sha).open("a")

    def _bump(self, db, name, n=1):
        db.execute(
            "INSERT INTO stats VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value",
            (name, n),
        )

    def _s3(self):
        if self._client is None or self._pid != os.getpid():
            if self.client_factory:
                self._client = self.client_factory()
            else:
                import boto3
                from botocore.config import Config

                self._client = boto3.client(
                    "s3",
                    endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                    config=Config(read_timeout=120, retries={"max_attempts": 4}),
                )
            self._pid = os.getpid()
        return self._client

    def _valid(self, spec):
        path = self._path(spec["sha256"])
        try:
            if path.stat().st_size != spec["bytes"]:
                return False
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024**2), b""):
                    digest.update(block)
            if digest.hexdigest() != spec["sha256"]:
                return False
            with self._db() as db:
                row = db.execute(
                    "SELECT state FROM entries WHERE sha=?", (spec["sha256"],)
                ).fetchone()
                if row is None or row[0] != "ready":
                    return False
                db.execute(
                    "UPDATE entries SET atime=? WHERE sha=?",
                    (time.time(), spec["sha256"]),
                )
            return True
        except FileNotFoundError:
            return False

    def _remove(self, db, sha):
        row = db.execute("SELECT size FROM entries WHERE sha=?", (sha,)).fetchone()
        self._path(sha).unlink(missing_ok=True)
        (self.root / "tmp" / (sha + ".part")).unlink(missing_ok=True)
        db.execute("DELETE FROM entries WHERE sha=?", (sha,))
        if row:
            self._bump(db, "occupied_bytes", -row[0])

    def _reserve(self, spec):
        sha, size = spec["sha256"], spec["bytes"]
        if size > self.max_bytes:
            raise CacheFull(
                f"Object {sha} ({size} bytes) exceeds cache budget {self.max_bytes}"
            )
        deadline = time.monotonic() + self.wait_seconds
        while True:
            with self._db() as db:
                # The caller owns this object's exclusive lease, including crash recovery.
                self._remove(db, sha)
                used = db.execute(
                    "SELECT value FROM stats WHERE name='occupied_bytes'"
                ).fetchone()[0]

                def room():
                    return (
                        used + size <= self.max_bytes
                        and shutil.disk_usage(self.root).free - size
                        >= self.min_free_bytes
                    )

                if not room():
                    cursor = db.execute(
                        "SELECT sha,size,state FROM entries ORDER BY atime"
                    )
                    for victim, n, state in cursor:
                        if room():
                            break
                        with self._lock(victim) as lock:
                            try:
                                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                continue
                            self._remove(db, victim)
                            used -= n
                            self._bump(
                                db,
                                "evictions"
                                if state == "ready"
                                else "recovered_downloads",
                            )
                if room():
                    db.execute(
                        "INSERT INTO entries VALUES (?,?,?,?)",
                        (sha, size, "downloading", time.time()),
                    )
                    self._bump(db, "occupied_bytes", size)
                    self._bump(db, "misses")
                    return
            if time.monotonic() >= deadline:
                raise CacheFull(
                    "Cache has no unleased space; all active readers and download reservations are protected"
                )
            time.sleep(0.1)

    def _download(self, spec):
        sha = spec["sha256"]
        tmp = self.root / "tmp" / (sha + ".part")
        target = self._path(sha)
        self._reserve(spec)
        try:
            args = {"Bucket": spec["bucket"], "Key": spec["key"]}
            if spec.get("etag"):
                args["IfMatch"] = spec["etag"]
            response = self._s3().get_object(**args)
            body = response["Body"]
            digest = hashlib.sha256()
            size = 0
            try:
                with tmp.open("wb") as f:
                    for block in iter(lambda: body.read(8 * 1024**2), b""):
                        size += len(block)
                        if size > spec["bytes"]:
                            raise ValueError("S3 response exceeds catalog size")
                        digest.update(block)
                        f.write(block)
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                body.close()
            if size != spec["bytes"] or digest.hexdigest() != sha:
                raise ValueError("S3 download differs from catalog SHA256/size")
            target.parent.mkdir(exist_ok=True)
            tmp.replace(target)
            with self._db() as db:
                db.execute(
                    "UPDATE entries SET state=?,atime=? WHERE sha=?",
                    ("ready", time.time(), sha),
                )
                self._bump(db, "download_bytes", size)
        except BaseException:
            with self._db() as db:
                self._remove(db, sha)
                self._bump(db, "download_errors")
            raise

    @contextmanager
    def acquire(self, spec):
        sha = spec["sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid SHA256 identity")
        with self._lock(sha) as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            if self._valid(spec):
                with self._db() as db:
                    self._bump(db, "hits")
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self._valid(spec):
                    with self._db() as db:
                        self._bump(db, "hits")
                else:
                    self._download(spec)
                fcntl.flock(lock, fcntl.LOCK_SH)
            yield self._path(sha)

    def stats(self):
        with self._db() as db:
            result = dict(db.execute("SELECT name,value FROM stats"))
            result.update(
                {
                    state + "_bytes": size
                    for state, size in db.execute(
                        "SELECT state,SUM(size) FROM entries GROUP BY state"
                    )
                }
            )
            result["reserved_and_ready_bytes"] = db.execute(
                "SELECT COALESCE(SUM(size),0) FROM entries"
            ).fetchone()[0]
            result["max_bytes"] = self.max_bytes
            return result


class Catalog:
    def __init__(self, path):
        self.path = str(path)
        self.db = None
        self.pid = None

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def get(self, path):
        if self.db is None or self.pid != os.getpid():
            if self.db:
                self.db.close()
            self.db = sqlite3.connect(
                "file:" + self.path + "?mode=ro&immutable=1",
                uri=True,
                check_same_thread=False,
            )
            self.pid = os.getpid()
        row = self.db.execute(
            "SELECT spec FROM paths WHERE path=?", (str(path),)
        ).fetchone()
        return json.loads(row[0]) if row else None
