"""Verify snapshot coverage, build/upload its immutable catalog and publish restore metadata."""

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import open_wam.artifacts.publication.upload as upload
from open_wam.artifacts.files import atomic_json, sha256_file
from open_wam.artifacts.publication.build_catalog import build
from open_wam.artifacts.publication.common import client
from open_wam.artifacts.publication.inventory import entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--receipts", required=True, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="openwam/latents/v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "restore.json").exists():
        raise FileExistsError("Already sealed; use a new release output")
    catalog = args.out / "catalog.sqlite"
    build(catalog, args.receipts)
    objects = []
    with sqlite3.connect(catalog) as db, args.inventory.open() as stream:
        for line in stream:
            item = json.loads(line)
            found = db.execute(
                "SELECT spec FROM paths WHERE path=?", (item["path"],)
            ).fetchone()
            if not found:
                raise ValueError("Unverified inventory object: " + item["path"])
            spec = json.loads(found[0])
            if spec["bytes"] != item["bytes"]:
                raise ValueError("Receipt size differs from inventory")
            objects.append(
                dict(
                    path=item["path"],
                    kind=item["kind"],
                    root_id=item["root_id"],
                    spec=spec,
                )
            )
        by_path = {item["path"]: item["spec"] for item in objects}
        for item in objects:
            if item["root_id"] == "snapshot" and item["path"].endswith(".csv"):
                if sha256_file(item["path"]) != item["spec"]["sha256"]:
                    raise ValueError("Manifest changed since upload")
                with open(item["path"]) as manifest:
                    for row in csv.DictReader(manifest):
                        path = Path(row["latent_path"])
                        if not path.is_absolute():
                            path = Path(item["path"]).parent / path
                        spec = by_path.get(str(path))
                        if (
                            not spec
                            or spec["sha256"] != row["latent_sha256"]
                            or spec["bytes"] != int(row["latent_bytes"])
                        ):
                            raise ValueError(
                                "Manifest tensor checksum differs from verified backup"
                            )
    s3_client = client()
    result = upload.upload(
        entry(catalog, "catalog", "metadata"),
        client=s3_client,
        bucket=args.bucket,
        prefix=args.prefix,
    )
    if result["status"] != "verified":
        raise RuntimeError(result)
    spec = {
        key: result[key]
        for key in ("bucket", "key", "sha256", "bytes", "etag", "verification")
    }
    atomic_json(args.out / "catalog_spec.json", spec)
    restore = dict(
        format_version=1, status="complete", catalog_spec=spec, objects=objects
    )
    atomic_json(args.out / "restore.json", restore)
    receipt = upload.upload(
        entry(args.out / "restore.json", "restore", "metadata"),
        client=s3_client,
        bucket=args.bucket,
        prefix=args.prefix,
    )
    if receipt["status"] != "verified":
        raise RuntimeError(receipt)
    atomic_json(args.out / "restore_spec.json", {key: receipt[key] for key in spec})
    print(
        json.dumps(
            dict(
                catalog_spec=str(args.out / "catalog_spec.json"),
                restore_uri=receipt["uri"],
                restore_sha256=receipt["sha256"],
                verified_objects=len(objects),
            )
        )
    )


if __name__ == "__main__":
    main()
