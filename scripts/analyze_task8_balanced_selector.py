#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from open_wam.planning.contracts import CandidateTrajectory
from open_wam.planning.evaluators import GoalDeltaAlignmentEvaluator, GoalImageL2Evaluator


def main() -> None:
    args = _parse_args()
    rng = random.Random(int(args.seed))
    records: list[dict[str, Any]] = []
    for records_path in args.records:
        path = Path(records_path)
        for record in _load_jsonl(path):
            record["_records_base_dir"] = str(path.parent)
            records.append(record)
    if bool(args.derive_task8_strict_labels):
        records = [_derive_task8_strict_labels(record) for record in records]
    if args.rescore_evaluator != "none":
        score_key = str(args.rescore_score_key or f"{args.rescore_evaluator}_cached_score")
        records = [
            _rescore_record_from_cached_artifacts(
                record,
                evaluator_name=str(args.rescore_evaluator),
                score_key=score_key,
            )
            for record in records
        ]
        args.score_key = score_key
        if args.rescore_output_records is not None:
            output_path = Path(args.rescore_output_records)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    row = {key: value for key, value in record.items() if key != "_records_base_dir"}
                    handle.write(json.dumps(row) + "\n")
    balanced_records: list[dict[str, Any]] = []
    skipped = Counter()
    generation_records: list[dict[str, Any]] = []
    for record in records:
        if not bool(record.get("state_collected", False)):
            skipped["state_not_collected"] += 1
            continue
        candidates = list(record.get("candidates") or [])
        positives = [item for item in candidates if bool(item.get(args.label_key, False))]
        negatives = [item for item in candidates if not bool(item.get(args.label_key, False))]
        policy_samples = [item for item in candidates if item.get("candidate_role") == "policy_sample"]
        policy_prior = next((item for item in candidates if item.get("candidate_role") == "policy_prior"), None)
        selected_id = str(record.get("vlm_top1_candidate_id"))
        selected = next((item for item in candidates if str(item.get("candidate_id")) == selected_id), None)
        generation_records.append(
            {
                "target_stage": str(record.get("target_stage", "unknown")),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "selected_success": bool(selected and selected.get(args.label_key, False)),
                "prior_success": bool(policy_prior and policy_prior.get(args.label_key, False)),
                "oracle_success": bool(positives),
                "policy_sample_successes": sum(bool(item.get(args.label_key, False)) for item in policy_samples),
                "policy_sample_count": len(policy_samples),
            }
        )
        if len(positives) < int(args.positives):
            skipped["not_enough_positives"] += 1
            continue
        if len(negatives) < int(args.negatives):
            skipped["not_enough_negatives"] += 1
            continue

        # Candidate scores are already computed independently by the planner
        # run. Re-ranking a balanced subset by score does not require
        # rerunning Open-WAM.
        selected_positive = _choose_items(
            positives,
            int(args.positives),
            rng=rng,
            mode=args.positive_choice,
            score_key=str(args.score_key),
        )
        selected_negative = _choose_items(
            negatives,
            int(args.negatives),
            rng=rng,
            mode=args.negative_choice,
            score_key=str(args.score_key),
        )
        subset = [*selected_positive, *selected_negative]
        if bool(args.shuffle_candidates):
            rng.shuffle(subset)
        subset = sorted(subset, key=lambda item: float(item.get(args.score_key, float("-inf"))), reverse=True)
        selected = subset[0]
        balanced_records.append(
            {
                "episode_idx": int(record.get("episode_idx", -1)),
                "target_stage": str(record.get("target_stage", "unknown")),
                "selected_candidate_id": str(selected.get("candidate_id")),
                "selected_success": bool(selected.get(args.label_key, False)),
                "label_key": str(args.label_key),
                "balanced_candidates": subset,
            }
        )

    summary = _summarize_balanced(
        balanced_records,
        generation_records=generation_records,
        positives_per_state=int(args.positives),
        negatives_per_state=int(args.negatives),
        skipped=skipped,
    )
    posthoc_summary = _summarize_posthoc(records)
    if posthoc_summary is not None:
        summary["posthoc_balanced_rerank"] = posthoc_summary
    if args.output_json is not None:
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if bool(args.print_markdown):
        _print_markdown(summary)
    else:
        print(json.dumps(summary, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", help="Path(s) to Task-8 offline reranking candidate_sets.jsonl.")
    parser.add_argument("--score-key", default="vlm_score")
    parser.add_argument("--label-key", default="stage_completed")
    parser.add_argument(
        "--derive-task8-strict-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Populate strict_stage_completed from cached Task-8 metrics_after. "
            "Use with --label-key strict_stage_completed for physical subtask labels."
        ),
    )
    parser.add_argument(
        "--rescore-evaluator",
        choices=("none", "goal_delta_alignment", "goal_image_l2"),
        default="none",
        help=(
            "Recompute candidate scores from cached predicted_video_path and planner_goal_artifacts "
            "before building the balanced matrix. This does not touch the simulator."
        ),
    )
    parser.add_argument("--rescore-score-key", default=None)
    parser.add_argument("--rescore-output-records", default=None)
    parser.add_argument("--goal-delta-alignment-weight", type=float, default=1.0)
    parser.add_argument("--goal-delta-background-penalty-weight", type=float, default=0.0)
    parser.add_argument("--goal-delta-change-threshold", type=float, default=8.0)
    parser.add_argument("--goal-image-weight", type=float, default=1.0)
    parser.add_argument("--positives", type=int, default=1)
    parser.add_argument("--negatives", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument(
        "--positive-choice",
        choices=("random", "best_score", "worst_score"),
        default="random",
        help="How to choose positives when a state has more than the requested count.",
    )
    parser.add_argument(
        "--negative-choice",
        choices=("random", "best_score", "worst_score"),
        default="random",
        help="How to choose negatives when a state has more than the requested count.",
    )
    parser.add_argument("--shuffle-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-markdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _derive_task8_strict_labels(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    stage_name = str(updated.get("target_stage") or updated.get("stage_name") or "")
    active_entity = str(updated.get("active_entity") or "")
    updated_candidates = []
    for raw_candidate in list(updated.get("candidates") or []):
        candidate = dict(raw_candidate)
        metrics_after = dict(candidate.get("metrics_after") or {})
        entity_metrics = metrics_after.get(active_entity)
        strict = False
        if bool(candidate.get("task_success", False)):
            strict = True
        elif isinstance(entity_metrics, dict):
            if stage_name.startswith("pick_up_pot"):
                strict = bool(entity_metrics.get("lifted", False))
            elif stage_name.startswith("put_down_pot"):
                strict = bool(entity_metrics.get("on_stove", False))
        candidate["strict_stage_completed"] = bool(strict)
        candidate["legacy_stage_completed"] = bool(candidate.get("stage_completed", False))
        updated_candidates.append(candidate)
    updated["candidates"] = updated_candidates
    selected_id = str(updated.get("vlm_top1_candidate_id"))
    selected = next((item for item in updated_candidates if str(item.get("candidate_id")) == selected_id), None)
    updated["selected_strict_stage_completed"] = bool(selected and selected.get("strict_stage_completed", False))
    updated["oracle_strict_stage_completed"] = any(bool(item.get("strict_stage_completed", False)) for item in updated_candidates)
    return updated


def _rescore_record_from_cached_artifacts(
    record: dict[str, Any],
    *,
    evaluator_name: str,
    score_key: str,
) -> dict[str, Any]:
    if not bool(record.get("state_collected", False)):
        return record
    goal = _load_cached_goal(record, evaluator_name=evaluator_name)
    if evaluator_name == "goal_delta_alignment":
        evaluator = GoalDeltaAlignmentEvaluator()
    elif evaluator_name == "goal_image_l2":
        evaluator = GoalImageL2Evaluator()
    else:
        raise ValueError(f"Unsupported cached rescoring evaluator: {evaluator_name!r}.")

    updated = dict(record)
    updated_candidates = []
    for candidate_record in list(record.get("candidates") or []):
        candidate = dict(candidate_record)
        predicted_video_path = candidate.get("predicted_video_path")
        if not predicted_video_path:
            raise ValueError(
                "Cached rescoring requires every candidate to have predicted_video_path; "
                f"missing for episode={record.get('episode_idx')} candidate={candidate.get('candidate_id')}."
            )
        predicted_video = _read_video(_resolve_path(record, str(predicted_video_path)))
        trajectory = CandidateTrajectory(
            context=None,  # type: ignore[arg-type]
            predicted_videos=(predicted_video,),
        )
        candidate[score_key] = float(evaluator.score(trajectory, goal=goal))
        updated_candidates.append(candidate)
    updated["candidates"] = updated_candidates
    selected = max(updated_candidates, key=lambda item: float(item.get(score_key, float("-inf"))), default=None)
    if selected is not None:
        updated["vlm_top1_candidate_id"] = str(selected.get("candidate_id"))
        updated["selected_stage_completed"] = bool(selected.get("stage_completed", False))
        updated["selected_task_success"] = bool(selected.get("task_success", False))
        updated["selected_oracle_score"] = float(selected.get("oracle_score", 0.0))
        updated["cached_rescore_evaluator"] = evaluator_name
        updated["cached_rescore_score_key"] = score_key
    return updated


def _load_cached_goal(record: dict[str, Any], *, evaluator_name: str) -> Any:
    artifacts = dict(record.get("planner_goal_artifacts") or {})
    target_path = artifacts.get("target_path") or (record.get("goal_metadata") or {}).get("goal_path")
    if not target_path:
        raise ValueError(
            "Cached rescoring requires planner_goal_artifacts.target_path or goal_metadata.goal_path; "
            f"missing for episode={record.get('episode_idx')}."
        )
    target = np.asarray(imageio.imread(_resolve_path(record, str(target_path))), dtype=np.uint8)
    if evaluator_name == "goal_image_l2":
        return target
    current_path = artifacts.get("current_path")
    if not current_path:
        raise ValueError(
            "Goal-delta cached rescoring requires planner_goal_artifacts.current_path; "
            f"missing for episode={record.get('episode_idx')}."
        )
    current = np.asarray(imageio.imread(_resolve_path(record, str(current_path))), dtype=np.uint8)
    return {"current": current, "target": target}


def _resolve_path(record: dict[str, Any], raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    base = Path(str(record.get("_records_base_dir", ".")))
    return base / path


def _read_video(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    frames = [np.asarray(frame, dtype=np.uint8) for frame in imageio.get_reader(path)]
    if not frames:
        raise ValueError(f"Cached predicted video has no frames: {path}")
    return np.stack(frames, axis=0)


def _choose_items(
    items: list[dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
    mode: str,
    score_key: str,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if len(items) < count:
        raise ValueError(f"Cannot choose {count} items from {len(items)} candidates.")
    if mode == "random":
        return rng.sample(items, count)
    reverse = mode == "best_score"
    return sorted(items, key=lambda item: float(item.get(score_key, float("-inf"))), reverse=reverse)[:count]


def _summarize_balanced(
    records: list[dict[str, Any]],
    *,
    generation_records: list[dict[str, Any]],
    positives_per_state: int,
    negatives_per_state: int,
    skipped: Counter[str],
) -> dict[str, Any]:
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    for record in records:
        stage = str(record["target_stage"])
        label_key = str(record.get("label_key", "stage_completed"))
        for candidate in record["balanced_candidates"]:
            predicted = str(candidate.get("candidate_id")) == str(record["selected_candidate_id"])
            actual = bool(candidate.get(label_key, False))
            if actual and predicted:
                key = "tp"
            elif not actual and predicted:
                key = "fp"
            elif actual and not predicted:
                key = "fn"
            else:
                key = "tn"
            by_stage[stage][key] += 1
            overall[key] += 1
        by_stage[stage]["states"] += 1
        overall["states"] += 1
        if bool(record["selected_success"]):
            by_stage[stage]["selected_success_states"] += 1
            overall["selected_success_states"] += 1

    return {
        "states": int(overall["states"]),
        "positives_per_state": int(positives_per_state),
        "negatives_per_state": int(negatives_per_state),
        "skipped": dict(skipped),
        "candidate_generation": _summarize_generation(generation_records),
        "overall": _with_rates(overall),
        "by_stage": {stage: _with_rates(counts) for stage, counts in sorted(by_stage.items())},
    }


def _summarize_generation(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    for record in records:
        stage = str(record["target_stage"])
        for counts in (overall, by_stage[stage]):
            counts["states"] += 1
            counts["selected_success_states"] += int(bool(record["selected_success"]))
            counts["prior_success_states"] += int(bool(record["prior_success"]))
            counts["oracle_success_states"] += int(bool(record["oracle_success"]))
            counts["policy_sample_successes"] += int(record["policy_sample_successes"])
            counts["policy_sample_count"] += int(record["policy_sample_count"])
            counts[f"positive_count_{int(record['positive_count'])}"] += 1
    return {
        "overall": _generation_rates(overall),
        "by_stage": {stage: _generation_rates(counts) for stage, counts in sorted(by_stage.items())},
    }


def _summarize_posthoc(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        record["posthoc_balanced_rerank"]
        for record in records
        if bool(record.get("state_collected", False)) and "posthoc_balanced_rerank" in record
    ]
    if not rows:
        return None
    scored = [row for row in rows if not bool(row.get("skipped", False))]
    skipped = Counter(str(row.get("skip_reason", "unknown")) for row in rows if bool(row.get("skipped", False)))
    counts: Counter[str] = Counter()
    for row in scored:
        selected_id = str(row.get("selected_candidate_id"))
        for candidate in row.get("balanced_candidates", []):
            predicted = str(candidate.get("candidate_id")) == selected_id
            actual = bool(candidate.get("stage_completed", False))
            if actual and predicted:
                key = "tp"
            elif not actual and predicted:
                key = "fp"
            elif actual and not predicted:
                key = "fn"
            else:
                key = "tn"
            counts[key] += 1
        counts["states"] += 1
        counts["selected_success_states"] += int(bool(row.get("selected_stage_completed", False)))
    return {
        "states_with_posthoc": len(rows),
        "scored_states": len(scored),
        "skipped": dict(skipped),
        "overall": _with_rates(counts),
    }


def _generation_rates(counts: Counter[str]) -> dict[str, Any]:
    states = int(counts["states"])
    sample_count = int(counts["policy_sample_count"])
    hist = {
        key.removeprefix("positive_count_"): int(value)
        for key, value in sorted(counts.items())
        if key.startswith("positive_count_")
    }
    return {
        "states": states,
        "selected_success_rate": None if states == 0 else int(counts["selected_success_states"]) / states,
        "prior_success_rate": None if states == 0 else int(counts["prior_success_states"]) / states,
        "oracle_success_rate": None if states == 0 else int(counts["oracle_success_states"]) / states,
        "random_policy_sample_success_rate": None
        if sample_count == 0
        else int(counts["policy_sample_successes"]) / sample_count,
        "selected_success_states": int(counts["selected_success_states"]),
        "prior_success_states": int(counts["prior_success_states"]),
        "oracle_success_states": int(counts["oracle_success_states"]),
        "policy_sample_successes": int(counts["policy_sample_successes"]),
        "policy_sample_count": sample_count,
        "positive_count_histogram": hist,
    }


def _with_rates(counts: Counter[str]) -> dict[str, Any]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tn = int(counts["tn"])
    total = tp + fp + fn + tn
    return {
        "states": int(counts["states"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "accuracy": None if total == 0 else (tp + tn) / total,
        "selected_success_rate": None
        if int(counts["states"]) == 0
        else int(counts["selected_success_states"]) / int(counts["states"]),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{100.0 * value:.1f}%"
    return str(value)


def _print_markdown(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    print("# Balanced selector matrix")
    print()
    print(
        f"Balanced states: {summary['states']} "
        f"({summary['positives_per_state']} positive + {summary['negatives_per_state']} negatives per state)"
    )
    print(f"Skipped: {summary['skipped']}")
    print()
    generation = summary["candidate_generation"]["overall"]
    print("## Candidate generation")
    print()
    print("| Split | States | Open-WAM selected | Prior | Oracle ceiling | Random policy samples | Positive-count histogram |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    _print_generation_row("Overall", generation)
    for stage, row in summary["candidate_generation"]["by_stage"].items():
        _print_generation_row(f"`{stage}`", row)
    print()
    print("## Balanced selector")
    print()
    print("| Split | States | TP | FP | FN | TN | Precision | Recall | Accuracy | Selected success |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        "| Overall | "
        f"{overall['states']} | {overall['tp']} | {overall['fp']} | {overall['fn']} | {overall['tn']} | "
        f"{_fmt(overall['precision'])} | {_fmt(overall['recall'])} | {_fmt(overall['accuracy'])} | "
        f"{_fmt(overall['selected_success_rate'])} |"
    )
    for stage, row in summary["by_stage"].items():
        print(
            f"| `{stage}` | {row['states']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
            f"{_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['selected_success_rate'])} |"
        )
    posthoc = summary.get("posthoc_balanced_rerank")
    if isinstance(posthoc, dict):
        row = posthoc.get("overall", {})
        print()
        print("## Post-hoc balanced rerank")
        print()
        print(f"Scored states: {posthoc.get('scored_states', 0)}/{posthoc.get('states_with_posthoc', 0)}")
        if posthoc.get("skipped"):
            print(f"Skipped: {posthoc['skipped']}")
        print()
        print("| Split | States | TP | FP | FN | TN | Precision | Recall | Accuracy | Selected success |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        print(
            "| Post-hoc | "
            f"{row.get('states', 0)} | {row.get('tp', 0)} | {row.get('fp', 0)} | "
            f"{row.get('fn', 0)} | {row.get('tn', 0)} | {_fmt(row.get('precision'))} | "
            f"{_fmt(row.get('recall'))} | {_fmt(row.get('accuracy'))} | "
            f"{_fmt(row.get('selected_success_rate'))} |"
        )


def _print_generation_row(label: str, row: dict[str, Any]) -> None:
    sample_text = (
        "n/a"
        if row["random_policy_sample_success_rate"] is None
        else f"{row['policy_sample_successes']}/{row['policy_sample_count']} ({_fmt(row['random_policy_sample_success_rate'])})"
    )
    print(
        f"| {label} | {row['states']} | "
        f"{row['selected_success_states']}/{row['states']} ({_fmt(row['selected_success_rate'])}) | "
        f"{row['prior_success_states']}/{row['states']} ({_fmt(row['prior_success_rate'])}) | "
        f"{row['oracle_success_states']}/{row['states']} ({_fmt(row['oracle_success_rate'])}) | "
        f"{sample_text} | {row['positive_count_histogram']} |"
    )


if __name__ == "__main__":
    main()
