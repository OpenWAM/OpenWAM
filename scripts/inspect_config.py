from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.utils import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    pprint(load_experiment_config(args.config))


if __name__ == "__main__":
    main()
