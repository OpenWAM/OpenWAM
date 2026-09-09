"""Convert one AgiBot RGB tar and native task_info JSON into a LeRobot video-only view."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import av

CAMERAS = ("head_color", "hand_left_color", "hand_right_color")


def probe(path):
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        frames = stream.frames or sum(1 for _ in video.decode(stream))
        if abs(fps - 30) > 0.01 or frames <= 0:
            raise ValueError("Expected a nonempty 30 FPS RGB camera video")
        return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--task-info", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    tasks = json.loads(args.task_info.read_text())
    if not isinstance(tasks, list):
        raise ValueError("Expected official task_info list[dict]")
    by_id = {}
    for task in tasks:
        key = str(task["episode_id"])
        if key in by_id:
            raise ValueError("Duplicate native episode_id")
        by_id[key] = task
    if args.out.exists():
        raise FileExistsError("Use a new output directory")
    args.out.mkdir(parents=True)
    with tarfile.open(args.archive, "r:*") as archive:
        members = {}
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 3 or parts[-2] != "videos":
                continue
            camera = PurePosixPath(parts[-1]).stem
            if not parts[-1].endswith(".mp4") or camera not in CAMERAS:
                continue
            native = parts[-3]
            if (native, camera) in members:
                raise ValueError("Duplicate native camera member")
            members[native, camera] = member
        ids = sorted({native for native, _ in members})
        if not ids:
            raise ValueError("No selected AgiBot RGB videos")
        episodes = []
        for index, native in enumerate(ids):
            task = by_id[native]
            lengths = []
            for camera in CAMERAS:
                member = members[native, camera]
                target = (
                    args.out / f"episode_{index:06d}" / "videos" / (camera + ".mp4")
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 4 * 1024**2)
                if target.stat().st_size != member.size:
                    raise IOError("Short tar extraction")
                lengths.append(probe(target))
            if len(set(lengths)) != 1:
                raise ValueError(
                    "AgiBot camera lengths disagree; inspect native synchronization"
                )
            episodes.append(
                dict(
                    episode_index=index,
                    length=lengths[0],
                    agibot_id=native,
                    tasks=[task["task_name"]],
                    text_provenance=str(args.task_info.resolve())
                    + "#episode_id="
                    + native,
                )
            )
    meta = args.out / "meta"
    meta.mkdir()
    info = dict(
        codebase_version="v2.1",
        fps=30,
        chunk_size=1000,
        total_episodes=len(episodes),
        video_path="episode_{episode_index:06d}/videos/{video_key}.mp4",
        features={camera: dict(dtype="video") for camera in CAMERAS},
    )
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes)
    )
    (meta / "tasks.jsonl").write_text(
        "".join(
            json.dumps(dict(task_index=i, task=t), ensure_ascii=False) + "\n"
            for i, t in enumerate(sorted({r["tasks"][0] for r in episodes}))
        )
    )
    (args.out / "episode_names.json").write_text(
        json.dumps({str(r["episode_index"]): r["agibot_id"] for r in episodes}) + "\n"
    )
    print(
        json.dumps(dict(episodes=len(episodes), cameras=CAMERAS, output=str(args.out)))
    )


if __name__ == "__main__":
    main()
