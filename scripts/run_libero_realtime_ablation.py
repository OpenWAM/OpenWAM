from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.utils.libero_paradigm import require_current_libero_script  # noqa: E402


def main() -> None:
    allow_deprecated = "--allow-deprecated-libero-config" in sys.argv[1:]
    try:
        require_current_libero_script(
            "scripts/run_libero_realtime_ablation.py",
            allow_deprecated=allow_deprecated,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc

    target = REPO_ROOT / "scripts" / "deprecated" / "run_libero_realtime_ablation.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
