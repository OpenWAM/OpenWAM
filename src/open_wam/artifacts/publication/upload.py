"""Immutable, resumable latent backup with independently verified S3 checksums."""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

from open_wam.artifacts.files import atomic_json, sha256_file
from open_wam.artifacts.publication.common import client, object_key


def same(path, row):
    try:
        s = Path(path).stat()
    except FileNotFoundError:
        return False
    # st_dev identifies a mount on one host and differs across cluster nodes.
    return (s.st_size, s.st_mtime_ns, s.st_ino) == (
        row["bytes"],
        row["mtime_ns"],
        row["inode"],
    )


def verify(key, size, sha, *, client, bucket):
    c = client
    h = c.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    if h["ContentLength"] != size:
        raise ValueError("S3 length mismatch: " + key)
    expected = base64.b64encode(bytes.fromhex(sha)).decode()
    if h.get("ChecksumSHA256") == expected:
        method = "s3_full_object_sha256"
    else:
        body = c.get_object(Bucket=bucket, Key=key, IfMatch=h["ETag"])["Body"]
        digest = hashlib.sha256()
        count = 0
        try:
            for block in iter(lambda: body.read(8 * 1024**2), b""):
                digest.update(block)
                count += len(block)
        finally:
            body.close()
        if count != size or digest.hexdigest() != sha:
            raise ValueError("S3 download SHA256 mismatch: " + key)
        method = "full_download_sha256"
    return {"etag": h["ETag"], "verification": method, "verified_unix": time.time()}


def put(path, key, size, sha, *, client, bucket):
    c = client
    checksum = base64.b64encode(bytes.fromhex(sha)).decode()
    if size < 4 * 1024**3:
        try:
            with open(path, "rb") as f:
                c.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=f,
                    ContentLength=size,
                    ChecksumSHA256=checksum,
                    Metadata={"sha256": sha},
                    IfNoneMatch="*",
                )
        except Exception as e:
            if getattr(e, "response", {}).get("Error", {}).get("Code") not in (
                "PreconditionFailed",
                "412",
                "KeyAlreadyExists",
            ):
                raise
        return
    upload = c.create_multipart_upload(
        Bucket=bucket, Key=key, Metadata={"sha256": sha}
    )["UploadId"]
    try:
        parts = []
        with open(path, "rb") as f:
            for number in range(1, 10001):
                block = f.read(128 * 1024**2)
                if not block:
                    break
                r = c.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload,
                    PartNumber=number,
                    Body=block,
                )
                parts.append({"PartNumber": number, "ETag": r["ETag"]})
        c.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload,
            MultipartUpload={"Parts": parts},
            IfNoneMatch="*",
        )
    except Exception as e:
        c.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload)
        if getattr(e, "response", {}).get("Error", {}).get("Code") not in (
            "PreconditionFailed",
            "412",
            "KeyAlreadyExists",
        ):
            raise


def upload(row, *, client, bucket, prefix):
    path = row["path"]
    try:
        if not same(path, row):
            return {**row, "status": "changed_before_upload"}
        sha = sha256_file(path)
        key = object_key(sha, prefix)
        if not same(path, row):
            return {**row, "status": "changed_while_hashing"}
        try:
            verification = verify(key, row["bytes"], sha, client=client, bucket=bucket)
        except Exception as e:
            if getattr(e, "response", {}).get("Error", {}).get("Code") not in (
                "404",
                "NoSuchKey",
                "NotFound",
            ):
                raise
            put(path, key, row["bytes"], sha, client=client, bucket=bucket)
            verification = verify(key, row["bytes"], sha, client=client, bucket=bucket)
        if not same(path, row):
            return {**row, "status": "changed_during_upload"}
        return {
            **row,
            "status": "verified",
            "sha256": sha,
            "bucket": bucket,
            "key": key,
            "uri": f"s3://{bucket}/{key}",
            **verification,
        }
    except Exception as e:
        return {**row, "status": "error", "error": type(e).__name__ + ": " + str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inventory")
    p.add_argument("--name")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--seed-receipts")
    p.add_argument("--out-root", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="openwam/latents/v1")
    a = p.parse_args()
    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    bucket = a.bucket
    prefix = a.prefix
    source = Path(a.inventory)
    name = a.name or source.stem
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    output = receipts / (name + ".jsonl")
    progress = receipts / (name + ".progress.json")
    import fcntl

    lock = open(str(output) + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    done = {}
    for prior in ([Path(a.seed_receipts)] if a.seed_receipts else []) + [output]:
        if not prior.exists():
            continue
        with prior.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    r.get("status") == "verified"
                    and r.get("bucket") == bucket
                    and r.get("key") == object_key(r["sha256"], prefix)
                ):
                    done[r["path"]] = (r["bytes"], r["mtime_ns"], r["inode"])
    counts = {
        "status": "running",
        "inventory": str(source),
        "inventory_sha256": sha256_file(source),
        "started_unix": time.time(),
        "verified": 0,
        "verified_bytes": 0,
        "skipped_verified": 0,
        "already_verified_bytes": 0,
        "errors": 0,
        "changed": 0,
        "changed_tensors": 0,
        "changed_metadata": 0,
        "workers": a.workers,
    }

    def rows():
        with source.open() as f:
            for line in f:
                r = json.loads(line)
                if done.get(r["path"]) == (r["bytes"], r["mtime_ns"], r["inode"]):
                    counts["skipped_verified"] += 1
                    counts["already_verified_bytes"] += r["bytes"]
                    continue
                yield r

    s3_client = client()
    it = iter(rows())
    last = 0
    with (
        output.open("a", buffering=1) as out,
        concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool,
    ):
        pending = {}

        def refill():
            while len(pending) < a.workers * 2:
                r = next(it, None)
                if r is None:
                    break
                pending[
                    pool.submit(
                        upload, r, client=s3_client, bucket=bucket, prefix=prefix
                    )
                ] = r

        refill()
        while pending:
            ready, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in ready:
                pending.pop(future)
                r = future.result()
                out.write(json.dumps(r, separators=(",", ":")) + "\n")
                if r["status"] == "verified":
                    counts["verified"] += 1
                    counts["verified_bytes"] += r["bytes"]
                elif r["status"] == "error":
                    counts["errors"] += 1
                    print(json.dumps(r), flush=True)
                else:
                    counts["changed"] += 1
                    counts[
                        "changed_tensors"
                        if r["kind"] == "tensor"
                        else "changed_metadata"
                    ] += 1
            refill()
            if time.time() - last >= 30:
                out.flush()
                os.fsync(out.fileno())
                counts["updated_unix"] = time.time()
                atomic_json(progress, counts)
                print(json.dumps(counts), flush=True)
                last = time.time()
        out.flush()
        os.fsync(out.fileno())
    counts["status"] = (
        "complete" if not counts["errors"] and not counts["changed"] else "review"
    )
    counts["finished_unix"] = time.time()
    atomic_json(progress, counts)
    print(json.dumps(counts), flush=True)
    if counts["errors"] or counts["changed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
