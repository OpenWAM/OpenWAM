"""Back up and relocate a tiny snapshot without retaining local tensor files."""

import base64
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import open_wam.artifacts.publication.inventory as inventory
import open_wam.artifacts.publication.restore as restore
import open_wam.artifacts.publication.seal as seal
import open_wam.artifacts.publication.upload as upload
from open_wam.artifacts.files import sha256_file
from open_wam.artifacts.object_cache import Catalog, ObjectCache


class Missing(Exception):
    response = {"Error": {"Code": "404"}}


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.reads = []

    def head_object(self, *, Bucket, Key, **kwargs):
        if (Bucket, Key) not in self.objects:
            raise Missing()
        data = self.objects[Bucket, Key]
        return dict(
            ContentLength=len(data),
            ETag='"immutable"',
            ChecksumSHA256=base64.b64encode(hashlib.sha256(data).digest()).decode(),
        )

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        assert kwargs["IfNoneMatch"] == "*"
        self.objects[Bucket, Key] = Body.read()

    def get_object(self, *, Bucket, Key, **kwargs):
        self.reads.append((Bucket, Key))
        return dict(Body=io.BytesIO(self.objects[Bucket, Key]))


def test_seal_restore_and_read_only_fetch_used_tensor(tmp_path, monkeypatch):
    tensor = tmp_path / "original.pth"
    tensor.write_bytes(b"tiny immutable tensor payload")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = snapshot / "VPT-01.csv"
    row = dict(
        latent_path=str(tensor),
        latent_sha256=sha256_file(tensor),
        latent_bytes=tensor.stat().st_size,
    )
    with manifest.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    sums = {manifest.name: sha256_file(manifest)}
    (snapshot / "snapshot.json").write_text(
        json.dumps(
            dict(
                status="complete",
                manifests_sha256=sums,
                snapshot_sha256="source-identity",
            )
        )
    )
    inv = tmp_path / "inventory.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["inventory", "--snapshot", str(snapshot), "--out", str(inv)]
    )
    inventory.main()
    remote = MemoryS3()
    monkeypatch.setattr(seal, "client", lambda: remote)
    monkeypatch.setattr(restore, "client", lambda: remote)
    receipts = tmp_path / "receipts.jsonl"
    rows = [
        upload.upload(json.loads(line), client=remote, bucket="test", prefix="prefix")
        for line in inv.read_text().splitlines()
    ]
    assert all(row["status"] == "verified" for row in rows)
    receipts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sealed = tmp_path / "sealed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seal",
            "--inventory",
            str(inv),
            "--receipts",
            str(receipts),
            "--out",
            str(sealed),
            "--bucket",
            "test",
            "--prefix",
            "prefix",
        ],
    )
    seal.main()
    tensor.unlink()

    def cache(root, max_bytes):
        return ObjectCache(
            root, max_bytes=max_bytes, min_free_bytes=0, client_factory=lambda: remote
        )

    monkeypatch.setattr(restore, "ObjectCache", cache)
    new = tmp_path / "relocated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restore",
            "--restore-spec",
            str(sealed / "restore_spec.json"),
            "--out",
            str(new),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--cache-gib",
            "0.01",
            "--bucket",
            "test",
            "--prefix",
            "prefix",
        ],
    )
    restore.main()
    tensor_key = next(row["key"] for row in rows if row["kind"] == "tensor")
    assert ("test", tensor_key) not in remote.reads
    with (new / "snapshot/VPT-01.csv").open() as stream:
        restored = next(csv.DictReader(stream))
    logical = Path(restored["latent_path"])
    assert not logical.exists()
    spec = Catalog(new / "catalog.sqlite").get(logical)
    with cache(tmp_path / "cache", int(0.01 * 1024**3)).acquire(spec) as path:
        assert path.read_bytes() == b"tiny immutable tensor payload"
