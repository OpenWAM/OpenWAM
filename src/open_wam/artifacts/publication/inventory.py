"""Inventory only explicit snapshot references and text artifacts; never scan raw video roots."""

import argparse
import csv
import json
import os
from pathlib import Path

from open_wam.artifacts.files import sha256_file


def entry(path, root_id, kind):
    path = Path(path).expanduser().absolute()
    info = path.stat()
    if not path.is_file() or info.st_size == 0:
        raise ValueError("Expected a nonempty regular artifact: " + str(path))
    return dict(
        path=str(path),
        real_path=str(path.resolve()),
        root_id=root_id,
        kind=kind,
        bytes=info.st_size,
        mtime_ns=info.st_mtime_ns,
        inode=info.st_ino,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--text-cache", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    snapshot = json.loads((args.snapshot / "snapshot.json").read_text())
    if snapshot["status"] != "complete":
        raise ValueError("Snapshot is incomplete")
    paths = {}
    for name in snapshot["manifests_sha256"]:
        manifest = args.snapshot / name
        if sha256_file(manifest) != snapshot["manifests_sha256"][name]:
            raise ValueError("Snapshot manifest changed since publication")
        paths[str(manifest.absolute())] = entry(manifest, "snapshot", "metadata")
        with manifest.open(newline="") as stream:
            for row in csv.DictReader(stream):
                path = Path(row["latent_path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                paths[str(path.absolute())] = entry(path, "latents", "tensor")
    for path in (args.snapshot / "snapshot.json",):
        paths[str(path.absolute())] = entry(path, "snapshot", "metadata")
    if args.text_cache:
        index = json.loads((args.text_cache / "index.json").read_text())
        if not index["complete"]:
            raise ValueError("Text cache is incomplete")
        for path in (args.text_cache / "index.json", args.text_cache / "prompts.jsonl"):
            paths[str(path.absolute())] = entry(path, "text_metadata", "metadata")
        for line in (args.text_cache / "prompts.jsonl").read_text().splitlines():
            prompt = json.loads(line)
            path = args.text_cache / "embeddings" / (prompt["sha256"] + ".pt")
            paths[str(path.absolute())] = entry(path, "text_embeddings", "tensor")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        for key in sorted(paths):
            stream.write(json.dumps(paths[key]) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            dict(files=len(paths), bytes=sum(row["bytes"] for row in paths.values()))
        )
    )


if __name__ == "__main__":
    main()
