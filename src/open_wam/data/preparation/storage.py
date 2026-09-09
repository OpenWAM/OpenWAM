"""Storage."""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading


def is_s3(uri: str) -> bool:
    return str(uri).startswith("s3://")


def join(root: str, *parts: str) -> str:
    """Join that works for both local paths and s3:// URIs."""

    tail = "/".join(str(p).strip("/") for p in parts if str(p))
    return f"{str(root).rstrip('/')}/{tail}" if tail else str(root)


class LocalStore:
    def find_repo_roots(self, root: str) -> list[str]:
        from pathlib import Path

        base = Path(root).expanduser()
        if (base / "meta" / "info.json").is_file():
            return [str(base)]
        found = sorted({str(p.parent.parent) for p in base.glob("*/meta/info.json")})
        return found or sorted(
            {str(p.parent.parent) for p in base.rglob("meta/info.json")}
        )

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        # Write to a sibling temp file and rename. A plain write leaves a
        # half-finished .pth under its FINAL name if the process dies mid-write,
        # and the resume check would then treat that episode as encoded and never
        # revisit it. Not hypothetical: the EgoExo4D pass was SIGKILLed 51 times
        # by the cgroup memory cap, every one of them a chance to strand a
        # truncated file. os.replace is atomic within a filesystem, and the temp
        # file sits in the destination directory to guarantee that.
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def list_keys(self, prefix: str, suffix: str = ".pth") -> set[str]:
        from pathlib import Path

        base = Path(prefix)
        if not base.is_dir():
            return set()
        return {
            str(p.relative_to(base)) for p in base.rglob("*") if p.name.endswith(suffix)
        }


class S3Store:
    """Thread-safe enough for our use: one client per thread, created lazily."""

    def __init__(self, endpoint: str | None) -> None:
        import boto3

        self._boto3 = boto3
        self._endpoint = endpoint
        self._local = threading.local()

    @property
    def client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            kwargs = {"endpoint_url": self._endpoint} if self._endpoint else {}
            client = self._boto3.client("s3", **kwargs)
            self._local.client = client
        return client

    @staticmethod
    def split(uri: str) -> tuple[str, str]:
        rest = str(uri)[len("s3://") :]
        bucket, _, key = rest.partition("/")
        return bucket, key

    def _iter_keys(self, uri: str):
        bucket, prefix = self.split(uri)
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                yield item["Key"]

    def find_repo_roots(self, root: str) -> list[str]:
        SUFFIX = "/meta/info.json"
        bucket, _ = self.split(root)
        roots = set()
        for key in self._iter_keys(root):
            if key.endswith(SUFFIX):
                roots.add(f"s3://{bucket}/{key[: -len(SUFFIX)]}")
        return sorted(roots)

    def read_bytes(self, path: str) -> bytes:
        bucket, key = self.split(path)
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()

    def write_bytes(self, path: str, data: bytes) -> None:
        bucket, key = self.split(path)
        self.client.put_object(Bucket=bucket, Key=key, Body=data)

    def list_keys(self, prefix: str, suffix: str = ".pth") -> set[str]:
        bucket, base = self.split(prefix)
        base = base.rstrip("/") + "/"
        return {k[len(base) :] for k in self._iter_keys(prefix) if k.endswith(suffix)}


def make_store(uri: str, endpoint: str | None):
    return S3Store(endpoint) if is_s3(uri) else LocalStore()
