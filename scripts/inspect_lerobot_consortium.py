from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import LeRobotConsortiumDataConfig
from open_wam.data import (
    build_lerobot_consortium_report,
    build_lerobot_consortium_train_val_datasets,
    format_lerobot_consortium_report,
)
from open_wam.utils.config_loader import load_experiment_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a lerobot_consortium config and preview its resolved member/split/sample surface.",
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML path.")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=3,
        help="Number of sample previews to load per split.",
    )
    parser.add_argument(
        "--sampler-preview-count",
        type=int,
        default=16,
        help="Number of epoch-0 train sampler indices to include in the report.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--write-audit-dir",
        type=str,
        default=None,
        help="Optional output directory for the train/val audit JSON files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    if not isinstance(config.data, LeRobotConsortiumDataConfig):
        raise ValueError(
            "inspect_lerobot_consortium.py requires `data.dataset_type: lerobot_consortium`."
        )

    report = build_lerobot_consortium_report(
        config.data,
        preview_count=args.preview_count,
        sampler_preview_count=args.sampler_preview_count,
    )
    if args.write_audit_dir is not None:
        train_dataset, val_dataset = build_lerobot_consortium_train_val_datasets(config.data)
        train_dataset.write_audit_artifacts(args.write_audit_dir)
        val_dataset.write_audit_artifacts(args.write_audit_dir)

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_lerobot_consortium_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
