from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import (
    build_lerobot_consortium_contract_catalog_from_inventory_rows,
    build_lerobot_consortium_inventory,
    load_lerobot_consortium_repo_targets,
    write_lerobot_consortium_contract_catalog,
    write_lerobot_consortium_inventory_csv,
    write_lerobot_consortium_inventory_json,
    write_lerobot_consortium_inventory_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenWAM's LeRobot consortium inventory and contract catalog from a repo-id list.",
    )
    parser.add_argument(
        "--repo-list",
        type=str,
        required=True,
        help="Path to a plain-text or CSV repo target list.",
    )
    parser.add_argument(
        "--default-source-group",
        type=str,
        default="manual",
        help="Fallback source_group when the repo list does not specify one explicitly.",
    )
    parser.add_argument(
        "--token-file",
        type=str,
        default=None,
        help="Optional file containing a Hugging Face token.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out-inventory-csv",
        type=str,
        default="notes/index/lerobot_consortium_hf_dataset_inventory.csv",
    )
    parser.add_argument(
        "--out-inventory-json",
        type=str,
        default="notes/index/lerobot_consortium_hf_dataset_inventory.json",
    )
    parser.add_argument(
        "--out-inventory-md",
        type=str,
        default="notes/index/lerobot_consortium_hf_dataset_inventory.md",
    )
    parser.add_argument(
        "--out-contracts-json",
        type=str,
        default="notes/index/lerobot_consortium_hf_dataset_contracts.json",
    )
    return parser.parse_args(argv)


def _load_token(path_text: str | None) -> str | None:
    if path_text is None:
        return None
    path = Path(path_text).expanduser()
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = load_lerobot_consortium_repo_targets(
        Path(args.repo_list),
        default_source_group=args.default_source_group,
    )
    token = _load_token(args.token_file)
    inventory_rows = build_lerobot_consortium_inventory(
        targets,
        token=token,
        workers=args.workers,
    )
    contracts = build_lerobot_consortium_contract_catalog_from_inventory_rows(inventory_rows)

    write_lerobot_consortium_inventory_csv(Path(args.out_inventory_csv), inventory_rows)
    write_lerobot_consortium_inventory_json(Path(args.out_inventory_json), inventory_rows)
    write_lerobot_consortium_inventory_markdown(Path(args.out_inventory_md), inventory_rows)
    write_lerobot_consortium_contract_catalog(Path(args.out_contracts_json), contracts)

    print(
        json.dumps(
            {
                "target_count": len(targets),
                "inventory_row_count": len(inventory_rows),
                "contract_dataset_count": contracts["dataset_count"],
                "out_inventory_csv": args.out_inventory_csv,
                "out_inventory_json": args.out_inventory_json,
                "out_inventory_md": args.out_inventory_md,
                "out_contracts_json": args.out_contracts_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
