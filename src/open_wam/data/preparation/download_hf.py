"""Download a selected Hugging Face revision and record its resolved immutable commit."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", choices=("dataset", "model"), default="dataset")
    parser.add_argument(
        "--revision",
        required=True,
        help="Commit SHA or ref, resolved once before transfer",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--include",
        nargs="+",
        required=True,
        help="Explicit path patterns, for example meta/** videos/**",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().repo_info(
        args.repo, repo_type=args.repo_type, revision=args.revision, files_metadata=True
    )
    import fnmatch

    selected = [
        f
        for f in info.siblings
        if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in args.include)
    ]
    if not selected:
        raise ValueError("No paths match the requested patterns")
    receipt = dict(
        repo_id=args.repo,
        repo_type=args.repo_type,
        requested_revision=args.revision,
        resolved_commit=info.sha,
        include=args.include,
        files=len(selected),
        known_bytes=sum(f.size or 0 for f in selected),
        files_with_unknown_size=sum(f.size is None for f in selected),
    )
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if args.plan:
        return
    previous = args.out / "OPENWAM_SOURCE.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if (old["repo_id"], old["resolved_commit"]) != (args.repo, info.sha):
            raise ValueError(
                "Destination contains another dataset/revision; use a new output directory"
            )
    snapshot_download(
        args.repo,
        repo_type=args.repo_type,
        revision=info.sha,
        local_dir=args.out,
        allow_patterns=args.include,
        max_workers=args.workers,
    )
    receipt.update(
        status="download_complete", recorded_at=datetime.now(timezone.utc).isoformat()
    )
    previous.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
