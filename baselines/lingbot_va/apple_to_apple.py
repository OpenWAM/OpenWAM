from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArmResult:
    key: str
    label: str
    source: str
    success: int
    total: int

    @property
    def success_rate(self) -> float:
        return float(self.success) / float(self.total) if self.total else 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "source": self.source,
            "success": self.success,
            "total": self.total,
            "success_rate": self.success_rate,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine vanilla LingBot-VA and Open-WAM LIBERO-10 rollout summaries."
    )
    parser.add_argument("--lingbot-summary", required=True, type=Path)
    parser.add_argument("--openwam-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="LingBot-VA LIBERO-10 Apple-to-Apple")
    args = parser.parse_args()

    summary = build_comparison_summary(
        lingbot_summary=load_json(args.lingbot_summary),
        openwam_summary=load_json(args.openwam_summary),
        title=args.title,
        lingbot_summary_path=args.lingbot_summary,
        openwam_summary_path=args.openwam_summary,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "combined_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_dir / "combined_summary.md"
    markdown_path.write_text(format_markdown(summary), encoding="utf-8")
    print(f"summary: {summary_path}")
    print(f"markdown: {markdown_path}")


def build_comparison_summary(
    *,
    lingbot_summary: Mapping[str, Any],
    openwam_summary: Mapping[str, Any],
    title: str,
    lingbot_summary_path: Path | None = None,
    openwam_summary_path: Path | None = None,
) -> dict[str, Any]:
    lingbot_rows = _lingbot_rows_by_episode(lingbot_summary)
    openwam_rows = _openwam_rows_by_episode(openwam_summary)
    episode_keys = sorted(set(lingbot_rows) | set(openwam_rows))
    row_set_warnings = build_row_set_warnings(
        lingbot_keys=set(lingbot_rows),
        openwam_keys=set(openwam_rows),
    )
    arm_results = [
        *summarize_lingbot_arms(lingbot_summary),
        *summarize_openwam_arms(openwam_summary),
    ]
    target_labels = openwam_target_labels(openwam_summary)
    per_episode: list[dict[str, Any]] = []
    for key in episode_keys:
        benchmark, task_id, episode_idx = key
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "task_id": task_id,
            "episode_idx": episode_idx,
        }
        for checkpoint_name, lingbot_row in sorted(lingbot_rows.get(key, {}).items()):
            prefix = f"lingbot:{checkpoint_name}"
            row[f"{prefix}:success"] = bool(lingbot_row.get("success"))
            row[f"{prefix}:env_timestep"] = lingbot_row.get("env_timestep")
        openwam_row = openwam_rows.get(key)
        if openwam_row is not None:
            for target_key in openwam_summary.get("target_keys", []):
                prefix = f"openwam:{target_key}"
                row[f"{prefix}:label"] = target_labels.get(str(target_key), str(target_key))
                row[f"{prefix}:success"] = openwam_row.get(f"{target_key}_success")
                row[f"{prefix}:executed_actions"] = openwam_row.get(f"{target_key}_executed_actions")
                row[f"{prefix}:summary_path"] = openwam_row.get(f"{target_key}_summary_path")
        per_episode.append(row)

    return {
        "title": title,
        "lingbot_summary_path": str(lingbot_summary_path) if lingbot_summary_path is not None else None,
        "openwam_summary_path": str(openwam_summary_path) if openwam_summary_path is not None else None,
        "episodes": [
            {"benchmark": key[0], "task_id": key[1], "episode_idx": key[2]} for key in episode_keys
        ],
        "arms": [arm.to_json_dict() for arm in arm_results],
        "row_set_warnings": row_set_warnings,
        "per_episode": per_episode,
        "openwam_output_root": openwam_summary.get("output_root"),
        "lingbot_output_dir": (lingbot_summary.get("suite") or {}).get("output_dir"),
    }


def summarize_lingbot_arms(summary: Mapping[str, Any]) -> list[ArmResult]:
    arms: list[ArmResult] = []
    for checkpoint in summary.get("checkpoints", []):
        name = str(checkpoint.get("checkpoint_name"))
        total = int(checkpoint.get("episodes", 0))
        success = int(checkpoint.get("successes", 0))
        arms.append(
            ArmResult(
                key=f"lingbot:{name}",
                label=f"LingBot-VA native {name}",
                source="lingbot_va_native",
                success=success,
                total=total,
            )
        )
    return arms


def summarize_openwam_arms(summary: Mapping[str, Any]) -> list[ArmResult]:
    labels = openwam_target_labels(summary)
    arms: list[ArmResult] = []
    by_checkpoint = summary.get("by_checkpoint") or {}
    ordered_keys = [str(key) for key in summary.get("target_keys", []) if str(key) in by_checkpoint]
    ordered_keys.extend(sorted(str(key) for key in by_checkpoint if str(key) not in set(ordered_keys)))
    for key in ordered_keys:
        payload = by_checkpoint[key]
        total = int(payload.get("total", payload.get("finished", 0)))
        arms.append(
            ArmResult(
                key=f"openwam:{key}",
                label=labels.get(str(key), str(payload.get("label") or key)),
                source="open_wam",
                success=int(payload.get("success", 0)),
                total=total,
            )
        )
    return arms


def openwam_target_labels(summary: Mapping[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for spec in summary.get("checkpoint_specs", []):
        if not isinstance(spec, Mapping) or spec.get("key") is None:
            continue
        labels[str(spec["key"])] = str(spec.get("label") or spec["key"])
    return labels


def _lingbot_rows_by_episode(
    summary: Mapping[str, Any],
) -> dict[tuple[str, int, int], dict[str, dict[str, Any]]]:
    rows: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in summary.get("results", []):
        if not isinstance(row, Mapping):
            continue
        key = (
            str(row.get("benchmark", "libero_10")),
            int(row["task_id"]),
            int(row["episode_idx"]),
        )
        rows.setdefault(key, {})[str(row.get("checkpoint_name"))] = dict(row)
    return rows


def _openwam_rows_by_episode(summary: Mapping[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    benchmark = str(summary.get("benchmark", "libero_10"))
    for row in summary.get("paired_rows", []):
        if not isinstance(row, Mapping):
            continue
        # Native LingBot names the env init id `episode_idx`; Open-WAM keeps both
        # task-local `episode_idx` and actual rollout `init_id`, so match on the
        # actual env init used by the Open-WAM rollout.
        key = (
            benchmark,
            int(row["task_id"]),
            int(row.get("init_id", row.get("episode_idx"))),
        )
        rows[key] = dict(row)
    return rows


def build_row_set_warnings(
    *,
    lingbot_keys: set[tuple[str, int, int]],
    openwam_keys: set[tuple[str, int, int]],
) -> list[str]:
    warnings: list[str] = []
    missing_openwam = sorted(lingbot_keys - openwam_keys)
    missing_lingbot = sorted(openwam_keys - lingbot_keys)
    if missing_openwam:
        warnings.append(
            "Open-WAM summary is missing native LingBot rows: "
            + ", ".join(format_episode_key(key) for key in missing_openwam[:10])
            + ("" if len(missing_openwam) <= 10 else f", ... ({len(missing_openwam)} total)")
        )
    if missing_lingbot:
        warnings.append(
            "Native LingBot summary is missing Open-WAM rows: "
            + ", ".join(format_episode_key(key) for key in missing_lingbot[:10])
            + ("" if len(missing_lingbot) <= 10 else f", ... ({len(missing_lingbot)} total)")
        )
    return warnings


def format_episode_key(key: tuple[str, int, int]) -> str:
    benchmark, task_id, episode_idx = key
    return f"{benchmark}:task{task_id}:init{episode_idx}"


def format_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary['title']}",
        "",
        f"LingBot summary: `{summary.get('lingbot_summary_path')}`",
        f"Open-WAM summary: `{summary.get('openwam_summary_path')}`",
        "",
        "## Aggregate",
        "",
        "| Arm | Source | Success | Total | Rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for arm in summary.get("arms", []):
        rate = 100.0 * float(arm.get("success_rate", 0.0))
        lines.append(
            f"| {md_escape(str(arm['label']))} | `{arm['source']}` | "
            f"{arm['success']} | {arm['total']} | {rate:.1f}% |"
        )
    row_set_warnings = summary.get("row_set_warnings") or []
    if row_set_warnings:
        lines.extend(["", "## Row Set Warnings", ""])
        lines.extend([f"- {md_escape(str(warning))}" for warning in row_set_warnings])
    lines.extend(["", "## Per Episode", ""])
    if not summary.get("per_episode"):
        lines.append("No episode rows found.")
        return "\n".join(lines) + "\n"
    discovered_columns = {
        key
        for row in summary["per_episode"]
        for key in row
        if key.endswith(":success")
    }
    success_columns = [
        f"{arm['key']}:success"
        for arm in summary.get("arms", [])
        if f"{arm['key']}:success" in discovered_columns
    ]
    success_columns.extend(sorted(discovered_columns - set(success_columns)))
    headers = ["Task", "Episode", *success_columns]
    lines.append("| " + " | ".join(md_escape(header) for header in headers) + " |")
    lines.append("| " + " | ".join(["---:"] * len(headers)) + " |")
    for row in summary["per_episode"]:
        cells = [str(row["task_id"]), str(row["episode_idx"])]
        for column in success_columns:
            value = row.get(column)
            cells.append("missing" if value is None else ("yes" if bool(value) else "no"))
        lines.append("| " + " | ".join(md_escape(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
