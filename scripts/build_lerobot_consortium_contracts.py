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
    load_lerobot_consortium_inventory_rows,
    write_lerobot_consortium_contract_catalog,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and auto-parse a consortium inventory CSV and emit the contract JSON catalog.",
    )
    parser.add_argument(
        "--inventory-csv",
        type=str,
        default="notes/index/lerobot_consortium_hf_dataset_inventory.csv",
    )
    parser.add_argument(
        "--out-contracts-json",
        type=str,
        default="notes/index/lerobot_consortium_hf_dataset_contracts.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory_path = Path(args.inventory_csv)
    rows = load_lerobot_consortium_inventory_rows(inventory_path)
    contracts = build_lerobot_consortium_contract_catalog_from_inventory_rows(rows)
    write_lerobot_consortium_contract_catalog(Path(args.out_contracts_json), contracts)
    print(
        json.dumps(
            {
                "inventory_row_count": len(rows),
                "contract_dataset_count": contracts["dataset_count"],
                "inventory_csv": str(inventory_path),
                "out_contracts_json": args.out_contracts_json,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
