from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.utils.libero_paradigm import require_current_libero_script  # noqa: E402


def main() -> None:
    try:
        require_current_libero_script("scripts/run_libero_realtime_ablation.py")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    raise AssertionError("Removed LIBERO entrypoint unexpectedly passed its guard.")


if __name__ == "__main__":
    main()
