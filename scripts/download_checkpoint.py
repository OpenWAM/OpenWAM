#!/usr/bin/env python3
"""Download the OpenWAM LIBERO+OXE pretrained checkpoint from HuggingFace.

Usage:
    # Inference weights only (~10 GB)
    python scripts/download_checkpoint.py --mode inference

    # Full model bundle (~30 GB; no optimizer/scheduler training state)
    python scripts/download_checkpoint.py --mode full

Requires HF_TOKEN environment variable with read access to the private repo.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# WHY HF_TOKEN: repo is private — unauthenticated requests return 401.
# Users must request access from Yao Feng before downloading.
REPO_ID = "openwam-data/libero-oxe-pretrain-5k"
DEFAULT_DIR = Path("checkpoints/libero-oxe-pretrain-5k")

# WHY two modes: inference only needs the exported safetensors (~10 GB),
# but fine-tuning needs model_state.pt (~20 GB) + config/metadata. The
# optimizer/scheduler training state is not part of the released HF bundle.
ALLOW_PATTERNS_BY_MODE = {
    "inference": [
        "transformer/*",
        "resolved_config.yaml",
        "README.md",
    ],
    "full": None,  # WHY None: download every released file in the repo
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=list(ALLOW_PATTERNS_BY_MODE),
        default="inference",
        help="Download mode: inference for safetensors only (~10 GB), full for the released model bundle (~30 GB).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Local directory for the downloaded checkpoint (default: {DEFAULT_DIR}).",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN environment variable not set.\n"
            "\n"
            f"The checkpoint repo ({REPO_ID}) is private.\n"
            "To download it:\n"
            "  1. Request access from Yao Feng (yaofeng1995@gmail.com)\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens\n"
            "  3. Export it:  export HF_TOKEN=hf_...\n"
            "  4. Re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    allow_patterns = ALLOW_PATTERNS_BY_MODE[args.mode]
    print(f"Downloading {REPO_ID} (mode={args.mode}) to {args.output_dir} ...")

    try:
        local_path = snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(args.output_dir),
            allow_patterns=allow_patterns,
            token=token,
        )
    except Exception as exc:
        # WHY catch broadly: huggingface_hub raises different exceptions for
        # 401 (bad token), 403 (no access), 404 (repo not found). A clear
        # message helps users distinguish auth vs access issues.
        msg = str(exc)
        if "401" in msg or "403" in msg:
            print(
                f"ERROR: Access denied to {REPO_ID}.\n"
                "Your HF_TOKEN may be invalid, or you have not been granted access.\n"
                "Request access from Yao Feng (yaofeng1995@gmail.com).",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloaded to: {local_path}")
    print("Done.")


if __name__ == "__main__":
    main()
