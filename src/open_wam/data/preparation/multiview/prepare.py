"""Build immutable RGB multi view plans from the portable episode index."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from open_wam.artifacts.files import atomic_json
from open_wam.data.preparation.camera_recipes import camera_recipe
from open_wam.data.preparation.multiview.layout import POLICY


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def make_plan(
    row,
    *,
    cameras,
    selection_reason="explicit camera order",
    camera_rotations=None,
    pair_orientation="auto",
):
    source = row["source_id"]
    selected, reason = list(cameras), selection_reason
    if not selected:
        return None, reason
    if (
        len(selected) not in (2, 3)
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(row["cameras"])
    ):
        raise ValueError("Select two or three distinct cameras present in the episode")
    rotations = dict(camera_rotations or {})
    plan = dict(
        policy=POLICY,
        source_id=source,
        repo_id=row["repo_id"],
        episode_index=int(row["episode_index"]),
        native_start=0,
        physical_episode_key=row["physical_episode_key"],
        cameras=selected,
        selection_reason=reason,
        original_camera_count=len(row["cameras"]),
        task=row["task"],
        text_provenance=row["text_provenance"],
        text_status=row["text_status"],
        single_view_paths=row.get("single_view_paths", []),
        native_end_frames={c: int(row["native_end_frames"][c]) for c in selected},
        raw_source=row["raw_source"],
        target_fps=15.0,
        max_clip_frames=257,
        camera_rotations_degrees=rotations,
        pair_orientation=pair_orientation,
    )
    plan["id"] = digest(plan)
    return plan, reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", required=True, nargs="+", help="Episode JSONL indexes"
    )
    parser.add_argument("--out", required=True, help="New immutable plan directory")
    args = parser.parse_args()
    root = Path(args.out).resolve()
    if root.exists():
        raise FileExistsError("Use a new plan directory; keep old plans immutable")
    groups, counts, seen = collections.defaultdict(list), collections.Counter(), set()
    for path in args.episodes:
        with open(path) as stream:
            for line in stream:
                row = json.loads(line)
                plan, reason = make_plan(row, **camera_recipe(row))
                counts[row["source_id"] + "/" + ("planned" if plan else reason)] += 1
                if plan:
                    identity = row["physical_episode_key"]
                    if identity in seen:
                        raise ValueError("Duplicate physical episode: " + identity)
                    seen.add(identity)
                    groups[(row["source_id"], row["repo_id"])].append(plan)
    index = []
    for (source, repo), plans in sorted(groups.items()):
        path = root / "plans" / source / (digest([source, repo]) + ".json")
        atomic_json(path, dict(source_id=source, repo_id=repo, plans=plans))
        index.append(
            dict(
                source_id=source,
                repo_id=repo,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                episodes=len(plans),
                raw_type=plans[0]["raw_source"]["type"],
            )
        )
    atomic_json(
        root / "plan_index.json", dict(policy=POLICY, plans=index, counts=dict(counts))
    )
    print(
        json.dumps(
            dict(
                groups=len(index),
                episodes=sum(len(v) for v in groups.values()),
                counts=dict(counts),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
