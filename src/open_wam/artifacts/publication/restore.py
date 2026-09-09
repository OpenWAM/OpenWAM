"""Restore small snapshot metadata and rebase catalog aliases without downloading latent tensors."""

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import open_wam.artifacts.publication.upload as upload
from open_wam.artifacts.files import atomic_json, sha256_file
from open_wam.artifacts.object_cache import ObjectCache
from open_wam.artifacts.publication.common import client
from open_wam.artifacts.publication.inventory import entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-gib", type=float, default=200)
    parser.add_argument(
        "--bucket", required=True, help="Destination for the rebased immutable catalog"
    )
    parser.add_argument("--prefix", default="openwam/latents/v1")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new restore directory")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True)
    cache = ObjectCache(args.cache_dir, max_bytes=int(args.cache_gib * 1024**3))
    with cache.acquire(json.loads(args.restore_spec.read_text())) as path:
        release = json.loads(path.read_text())
    if release["status"] != "complete":
        raise ValueError("Incomplete backup")
    aliases = {}
    catalog = args.out / "catalog.sqlite"
    with sqlite3.connect(catalog) as db:
        db.execute(
            "CREATE TABLE paths (path TEXT PRIMARY KEY,spec TEXT NOT NULL) WITHOUT ROWID"
        )
        db.execute(
            "CREATE TABLE objects (sha TEXT PRIMARY KEY,bytes INTEGER NOT NULL) WITHOUT ROWID"
        )
        for item in release["objects"]:
            spec = item["spec"]
            root = item["root_id"]
            if root == "latents":
                target = args.out / "latent_objects" / (spec["sha256"] + ".pth")
            elif root == "text_embeddings":
                target = (
                    args.out / "text_cache" / "embeddings" / Path(item["path"]).name
                )
            elif root == "text_metadata":
                target = args.out / "text_cache" / Path(item["path"]).name
            elif root == "snapshot":
                target = args.out / "snapshot" / Path(item["path"]).name
            else:
                raise ValueError("Unknown backup artifact role")
            aliases[item["path"]] = str(target)
            db.execute(
                "INSERT OR IGNORE INTO paths VALUES (?,?)",
                (str(target), json.dumps(spec)),
            )
            db.execute(
                "INSERT OR IGNORE INTO objects VALUES (?,?)",
                (spec["sha256"], spec["bytes"]),
            )
            if item["kind"] == "metadata":
                target.parent.mkdir(parents=True, exist_ok=True)
                with cache.acquire(spec) as path:
                    target.write_bytes(path.read_bytes())
        db.commit()
    snapshot = args.out / "snapshot"
    record = json.loads((snapshot / "snapshot.json").read_text())
    for name, expected in record["manifests_sha256"].items():
        path = snapshot / name
        if sha256_file(path) != expected:
            raise ValueError("Snapshot metadata hash mismatch")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        for row in rows:
            row["latent_path"] = aliases[row["latent_path"]]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        record["manifests_sha256"][name] = sha256_file(path)
    record["restored_from_snapshot_sha256"] = record["snapshot_sha256"]
    record["snapshot_sha256"] = hashlib.sha256(
        json.dumps(record["manifests_sha256"], sort_keys=True).encode()
    ).hexdigest()
    atomic_json(snapshot / "snapshot.json", record)
    s3_client = client()
    receipt = upload.upload(
        entry(catalog, "catalog", "metadata"),
        client=s3_client,
        bucket=args.bucket,
        prefix=args.prefix,
    )
    if receipt["status"] != "verified":
        raise RuntimeError(receipt)
    atomic_json(
        args.out / "catalog_spec.json",
        {
            key: receipt[key]
            for key in ("bucket", "key", "bytes", "sha256", "etag", "verification")
        },
    )
    print(
        json.dumps(
            dict(
                snapshot=str(snapshot),
                text_cache=str(args.out / "text_cache"),
                catalog_spec=str(args.out / "catalog_spec.json"),
                latent_downloads=0,
            )
        )
    )


if __name__ == "__main__":
    main()
