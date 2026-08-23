#!/usr/bin/env python3
"""Canonicalize an encoded-dynamics manifest without rewriting payloads."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from open_wam.data import migrate_encoded_dynamics_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoded_root", help="Encoded-dynamics artifact root")
    parser.add_argument(
        "--raw-root",
        help=(
            "Raw payload root. Required when legacy provenance paths no longer "
            "exist; stored as an artifact-relative path."
        ),
    )
    parser.add_argument(
        "--reference-branch",
        default="gt",
        help="Unperturbed branch label to record (default: gt)",
    )
    args = parser.parse_args(argv)

    artifact = migrate_encoded_dynamics_artifact(
        args.encoded_root,
        raw_root=args.raw_root,
        reference_branch=args.reference_branch,
    )
    print(
        json.dumps(
            {
                "artifact_schema": artifact.manifest["artifact_schema"],
                "reference_branch": artifact.reference_branch,
                "raw_payload_root": artifact.manifest["raw_payload_root"],
                "resolved_raw_root": str(artifact.raw_root),
                "root": str(artifact.root),
                "transition_rows": len(artifact.transition_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
