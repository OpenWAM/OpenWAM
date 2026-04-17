from __future__ import annotations

import argparse
from pathlib import Path
import sys

from open_wam.configs.static_schema import format_report, reports_to_exit_code, validate_config_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run static Open-WAM config validation without model imports.")
    parser.add_argument("paths", nargs="+", help="YAML files or directories to validate.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Only print failing reports.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = tuple(_iter_yaml_paths(args.paths))
    if not paths:
        raise SystemExit("No YAML files matched the requested paths.")
    reports = validate_config_files(paths, repo_root=args.repo_root)
    for report in reports:
        if args.quiet and report.ok:
            continue
        print(format_report(report, repo_root=args.repo_root))
    raise SystemExit(reports_to_exit_code(reports))


def _iter_yaml_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("*.yaml")))
            paths.extend(sorted(path.glob("*.yml")))
            continue
        paths.append(path)
    return paths


if __name__ == "__main__":
    main(sys.argv[1:])
