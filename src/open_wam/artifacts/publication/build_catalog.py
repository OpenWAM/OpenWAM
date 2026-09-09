"""Seal a read-only path-to-verified-object catalog from upload receipts."""

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from open_wam.artifacts.files import atomic_json, sha256_file


def build(target, sources, base=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    roots = {}
    processed = 0
    started = time.time()
    # Build SQLite on node-local storage; atomically publish one closed file.
    with tempfile.TemporaryDirectory(prefix="openwam-latent-catalog-") as work:
        local = Path(work) / "catalog.sqlite"
        if base:
            shutil.copyfile(base, local)
        db = sqlite3.connect(local)
        try:
            db.execute("PRAGMA cache_size=-262144")
            db.execute(
                "CREATE TABLE IF NOT EXISTS paths (path TEXT PRIMARY KEY,spec TEXT NOT NULL) WITHOUT ROWID"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS objects (sha TEXT PRIMARY KEY,bytes INTEGER NOT NULL) WITHOUT ROWID"
            )
            for source in sources:
                with open(source) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Malformed receipt in {source}") from exc
                        if r.get("status") != "verified":
                            continue
                        spec = {
                            k: r[k]
                            for k in (
                                "sha256",
                                "bucket",
                                "key",
                                "bytes",
                                "etag",
                                "verification",
                            )
                        }
                        value = json.dumps(spec, separators=(",", ":"))
                        for path in set((r["path"], r.get("real_path", r["path"]))):
                            old = db.execute(
                                "SELECT spec FROM paths WHERE path=?", (path,)
                            ).fetchone()
                            if old and old[0] != value:
                                oldspec = json.loads(old[0])
                                if oldspec["sha256"] != spec["sha256"]:
                                    raise ValueError(
                                        "Conflicting immutable path: " + path
                                    )
                            db.execute(
                                "INSERT OR REPLACE INTO paths VALUES (?,?)",
                                (path, value),
                            )
                        db.execute(
                            "INSERT OR IGNORE INTO objects VALUES (?,?)",
                            (r["sha256"], r["bytes"]),
                        )
                        roots[r["root_id"]] = roots.get(r["root_id"], 0) + 1
                        processed += 1
                        if processed % 100000 == 0:
                            print(
                                json.dumps(
                                    {
                                        "phase": "building",
                                        "verified_receipts": processed,
                                        "source": str(source),
                                        "elapsed_seconds": round(
                                            time.time() - started, 1
                                        ),
                                    }
                                ),
                                flush=True,
                            )
                db.commit()
            count = db.execute("SELECT COUNT(*) FROM paths").fetchone()[0]
            n, size = db.execute(
                "SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM objects"
            ).fetchone()
            if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Catalog integrity check failed")
        finally:
            db.close()
        digest = sha256_file(local)
        fd, staged_name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".partial", dir=target.parent
        )
        staged = Path(staged_name)
        try:
            with os.fdopen(fd, "wb") as out, local.open("rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024**2)
                out.flush()
                os.fsync(out.fileno())
            if sha256_file(staged) != digest:
                raise ValueError("Catalog publication copy checksum mismatch")
            staged.replace(target)
        finally:
            staged.unlink(missing_ok=True)
    report = {
        "catalog": str(target),
        "sha256": digest,
        "paths": count,
        "unique_objects": n,
        "unique_bytes": size,
        "roots_receipts": roots,
        "created_unix": time.time(),
        "build_seconds": round(time.time() - started, 1),
    }
    atomic_json(target.with_suffix(".json"), report)
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--base")
    p.add_argument("receipts", nargs="+")
    a = p.parse_args()
    build(a.output, a.receipts, a.base)


if __name__ == "__main__":
    main()
