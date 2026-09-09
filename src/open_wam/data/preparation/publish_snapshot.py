"""Merge validated single/multi view CSVs into an immutable, optionally additive snapshot."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from open_wam.artifacts.files import sha256_file
from open_wam.data.preparation.build_manifest import write_csv
from open_wam.data.preparation.snapshot import load_verified_snapshot


def load(paths):
    rows = {}
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("physical_episode_key") or not row.get("latent_sha256"):
                    raise ValueError(
                        "Every row needs physical identity and verified tensor checksum"
                    )
                if row.get("augmentation") not in (
                    "single_view",
                    "multi_view",
                ):
                    raise ValueError("Unknown sample construction")
                source = row.get("source_id") or Path(path).stem
                row["source_id"] = source
                key = source, row["clip_id"]
                if key in rows and rows[key] != row:
                    raise ValueError("Conflicting immutable clip identity: " + str(key))
                rows[key] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--previous",
        help="Previous snapshot directory; every previous sample must be retained",
    )
    args = parser.parse_args()
    target = Path(args.out).resolve()
    if target.exists():
        raise FileExistsError("Snapshot names are immutable; choose a new output")
    rows = load(args.manifests)
    if args.previous:
        record = load_verified_snapshot(args.previous)
        previous = load(
            Path(args.previous) / name for name in sorted(record["manifests_sha256"])
        )
        for key, row in previous.items():
            if key not in rows or rows[key] != row:
                raise ValueError(
                    "Additive snapshot changed or removed an existing sample"
                )
    groups = collections.defaultdict(list)
    for (source, _), row in sorted(rows.items()):
        groups[source].append(row)
    target.mkdir(parents=True)
    report = dict(format_version=1, status="complete", sources={}, manifests_sha256={})
    for source, group in groups.items():
        path = target / (source + ".csv")
        write_csv(path, group)
        report["manifests_sha256"][path.name] = sha256_file(path)
        report["sources"][source] = dict(
            clips=len(group),
            physical_episodes=len({r["physical_episode_key"] for r in group}),
            labelled=sum(bool(r.get("task")) for r in group),
            single_clips=sum(r["augmentation"] == "single_view" for r in group),
            multi_view_clips=sum(
                r["augmentation"] == "multi_view" for r in group
            ),
            encoded_view_hours=sum(
                int(r["video_num_frames"]) / float(r["observation_fps"]) for r in group
            )
            / 3600,
        )
    report["snapshot_sha256"] = hashlib.sha256(
        json.dumps(report["manifests_sha256"], sort_keys=True).encode()
    ).hexdigest()
    (target / "snapshot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
