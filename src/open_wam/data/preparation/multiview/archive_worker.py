"""Process each RoboMind compressed archive once, with one bounded HDF staging file."""

import argparse
import collections
import gzip
import io
import json
import os
import shutil
import signal
import tarfile
import time
from pathlib import Path

from open_wam.artifacts.files import atomic_json, sha256_file
from open_wam.data.preparation.multiview.prepare import digest
from open_wam.data.preparation.multiview.sources import RangeReader, split_s3
from open_wam.data.preparation.multiview.worker import Encoder


class JoinedParts(io.RawIOBase):
    def __init__(self, client, uri):
        self.client = client
        self.uri = uri
        self.current = None
        self.part_index = 0
        if not uri.startswith("s3://"):
            st = Path(uri).stat()
            self.parts = [
                {
                    "uri": uri,
                    "size": st.st_size,
                    "etag": str((st.st_size, st.st_mtime_ns, st.st_ino)),
                }
            ]
            return
        bucket, key = split_s3(uri)
        parts = []
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=key + ".part-"
        ):
            parts.extend(page.get("Contents", []))
        parts.sort(key=lambda x: x["Key"])
        if not parts:
            head = client.head_object(Bucket=bucket, Key=key)
            parts = [{"Key": key, "Size": head["ContentLength"], "ETag": head["ETag"]}]
        self.parts = [
            {
                "uri": f"s3://{bucket}/{p['Key']}",
                "size": p["Size"],
                "etag": p["ETag"].strip('"'),
            }
            for p in parts
        ]

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded archive read prohibited")
        output = []
        while size and self.part_index < len(self.parts):
            part = self.parts[self.part_index]
            if self.current is None:
                self.current = RangeReader(
                    self.client, part["uri"], block_size=8 * 1024**2
                )
                if (
                    self.current.object_size != part["size"]
                    or self.current.etag.strip('"') != part["etag"]
                ):
                    raise ValueError("Archive part changed after listing")
            block = self.current.read(size)
            if block:
                output.append(block)
                size -= len(block)
            else:
                self.current.close()
                self.current = None
                self.part_index += 1
        return b"".join(output)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=82800)
    args = parser.parse_args()
    shard, count = map(int, args.shard.split("/"))

    def interrupted(signum, frame):
        raise SystemExit(75)

    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    encoder = Encoder(args)
    groups = collections.defaultdict(list)
    for entry in json.loads(Path(args.index).read_text())["plans"]:
        if entry["raw_type"] != "robomind_official_archive":
            continue
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("Plan changed")
        for plan in json.loads(path.read_text())["plans"]:
            groups[plan["raw_source"]["uri"]].append(plan)
    done = 0
    for uri, plans in sorted(groups.items()):
        if time.monotonic() - started > args.max_seconds:
            return 75
        unit = digest(uri)
        if int(unit, 16) % count != shard:
            continue
        receipt = Path(args.out_root) / "archive_units" / (unit + ".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if (
                saved.get("status") == "complete"
                and saved.get("contract_sha256") == encoder.contract_sha
                and all(encoder.is_complete(p) for p in plans)
            ):
                continue
        expected = {p["raw_source"]["member"].removeprefix("./"): p for p in plans}
        if len(expected) != len(plans):
            raise ValueError("Duplicate episode member in archive")
        seen = set()
        joined = JoinedParts(
            encoder.store.client if uri.startswith("s3://") else None, uri
        )
        stage = Path(args.out_root) / "raw_stage" / f"archive-{unit}-{os.getpid()}"
        stage.mkdir(parents=True, exist_ok=False)
        archive_code_sha256 = sha256_file(Path(__file__))
        atomic_json(
            receipt,
            {
                "status": "processing",
                "uri": uri,
                "plans": len(plans),
                "contract_sha256": encoder.contract_sha,
                "archive_code_sha256": archive_code_sha256,
                "started_unix": time.time(),
            },
        )
        try:
            with gzip.GzipFile(fileobj=joined) as uncompressed:
                with tarfile.open(fileobj=uncompressed, mode="r|") as archive:
                    for member in archive:
                        if time.monotonic() - started > args.max_seconds:
                            return 75
                        name = member.name.removeprefix("./")
                        if name not in expected:
                            continue
                        if name in seen or not member.isfile():
                            raise ValueError("Invalid or duplicate expected HDF member")
                        plan = expected[name]
                        seen.add(name)
                        if encoder.is_complete(plan):
                            continue
                        if member.size > 16 * 1024**3:
                            raise ValueError("HDF member exceeds 16 GiB staging bound")
                        if shutil.disk_usage(stage).free < member.size + 32 * 1024**3:
                            raise RuntimeError(
                                "Insufficient free space for bounded HDF staging"
                            )
                        path = stage / (plan["id"] + ".hdf5")
                        try:
                            source = archive.extractfile(member)
                            with path.open("xb") as out:
                                shutil.copyfileobj(source, out, 4 * 1024**2)
                            if path.stat().st_size != member.size:
                                raise IOError("Short HDF extraction")
                            encoder.process(plan, path)
                        finally:
                            if path.exists():
                                path.unlink()
                # Reach gzip EOF to verify its checksum before publishing this
                # archive unit, including runs whose episodes were resumed.
                while uncompressed.read(8 * 1024**2):
                    pass
            if seen != set(expected):
                raise ValueError(
                    f"Missing expected HDF members: {len(set(expected) - seen)}"
                )
            atomic_json(
                receipt,
                {
                    "status": "complete",
                    "uri": uri,
                    "parts": joined.parts,
                    "plans": len(plans),
                    "contract_sha256": encoder.contract_sha,
                    "archive_code_sha256": archive_code_sha256,
                    "gzip_eof_verified": True,
                    "plan_ids": sorted(p["id"] for p in plans),
                    "finished_at_unix": time.time(),
                },
            )
            done += 1
            print(
                json.dumps(
                    {"event": "archive_complete", "uri": uri, "plans": len(plans)}
                ),
                flush=True,
            )
        except Exception as error:
            atomic_json(
                receipt,
                {
                    "status": "failed",
                    "uri": uri,
                    "error": f"{type(error).__name__}: {error}",
                    "contract_sha256": encoder.contract_sha,
                    "time": time.time(),
                },
            )
            raise
        finally:
            joined.close()
            stage.rmdir()  # Only this task's empty staging directory is removed.
        if args.max_archives and done >= args.max_archives:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
