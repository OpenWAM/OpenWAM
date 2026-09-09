"""Export native episode identities, camera timelines and task text for pretraining."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path


def physical_key(source, repo, episode):
    # No mount point, output location, camera, crop or segment in this identity.
    identity_source = "EgoExo4D" if source in ("VPT-10R", "VPT-10S") else source
    return hashlib.sha256(
        json.dumps([identity_source, repo, int(episode)], ensure_ascii=False).encode()
    ).hexdigest()


def make_row(source, repo, episode, cameras, lengths, raw, task, provenance):
    if not cameras or any(int(lengths[c]) <= 0 for c in cameras):
        raise ValueError(f"Missing native camera length: {repo}/{episode}")
    return dict(
        source_id=source,
        repo_id=repo,
        episode_index=int(episode),
        physical_episode_key=physical_key(source, repo, episode),
        cameras=list(cameras),
        native_end_frames=lengths,
        raw_source=raw,
        task=task,
        text_provenance=provenance,
        text_status="native" if task else "missing_native_text",
    )


def metadata_rows(store, root, family):
    from open_wam.data.preparation.storage import join

    try:
        raw = store.read_bytes(join(root, f"meta/{family}.jsonl"))
    except (FileNotFoundError, KeyError):
        raw = None
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") not in (
            "404",
            "NoSuchKey",
            "NotFound",
        ):
            raise
        raw = None
    if raw is not None:
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    import pyarrow.parquet as pq

    keys = sorted(store.list_keys(join(root, f"meta/{family}"), suffix=".parquet"))
    if not keys:
        try:
            raw = store.read_bytes(join(root, f"meta/{family}.parquet"))
            return pq.read_table(io.BytesIO(raw)).to_pylist()
        except FileNotFoundError:
            return []
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in (
                "404",
                "NoSuchKey",
                "NotFound",
            ):
                return []
            raise
    rows = []
    for key in keys:
        rows.extend(
            pq.read_table(
                io.BytesIO(store.read_bytes(join(root, f"meta/{family}", key)))
            ).to_pylist()
        )
    return rows


def native_task(row, tasks):
    value = row.get("tasks", row.get("task", []))
    if isinstance(value, str):
        value = [value]
    if value:
        texts = list(dict.fromkeys(str(v) for v in value if str(v).strip()))
        if len(texts) == 1:
            return texts[0]
        raise ValueError(
            "Episode has multiple native tasks; provide a reviewed text mapping"
        )
    if row.get("task_index") is not None:
        return tasks[int(row["task_index"])]
    if len(tasks) == 1:
        return next(iter(tasks.values()))
    return ""


def lerobot(args, *, text_overrides=None):
    from open_wam.data.preparation.storage import LocalStore, S3Store
    from open_wam.data.preparation.video_sources import discover_repos

    store = (
        S3Store(os.environ.get("AWS_ENDPOINT_URL"))
        if args.raw_root.startswith("s3://")
        else LocalStore()
    )
    diagnostics = io.StringIO()
    with contextlib.redirect_stderr(diagnostics):
        repos = discover_repos(store, args.raw_root)
    if "disagrees" in diagnostics.getvalue():
        raise ValueError(
            "Conflicting native episode timestamps; repair metadata into a new input root"
        )
    for repo in repos:
        tasks = {
            int(r.get("task_index", r.get("index", i))): str(r.get("task", ""))
            for i, r in enumerate(metadata_rows(store, repo.root, "tasks"))
        }
        episodes = {}
        for row in metadata_rows(store, repo.root, "episodes"):
            index = int(row["episode_index"])
            if index in episodes and episodes[index] != row:
                raise ValueError(f"Conflicting episode metadata: {repo.name}/{index}")
            episodes[index] = row
        if not episodes:
            raise ValueError(f"No native episode metadata: {repo.root}")
        for index, meta in sorted(episodes.items()):
            length = meta.get("length") or repo.length_of(index)
            if not length:
                raise ValueError("Native episode length must be explicit")
            override = (text_overrides or {}).get((repo.name, index))
            task = (
                override["task"] if override is not None else native_task(meta, tasks)
            )
            yield make_row(
                args.source,
                repo.name,
                index,
                repo.cameras,
                {c: int(length) for c in repo.cameras},
                dict(type="lerobot", root=repo.root, fps=repo.fps),
                task,
                meta.get("text_provenance")
                or repo.root + "/meta/episodes;" + repo.root + "/meta/tasks",
            )


def video_tree(args, *, text_overrides=None):
    import av

    from open_wam.data.preparation.storage import LocalStore, S3Store
    from open_wam.data.preparation.video_sources import discover_video_tree_repos

    store = (
        S3Store(os.environ.get("AWS_ENDPOINT_URL"))
        if args.raw_root.startswith("s3://")
        else LocalStore()
    )
    if not args.takes:
        raise ValueError("--takes takes.json is required for EgoExo4D native text")
    data = json.loads(Path(args.takes).read_text())
    takes = data.get("takes", data) if isinstance(data, dict) else data
    if isinstance(takes, dict):
        takes = list(takes.values())
    tasks = {row["take_name"]: row["task_name"] for row in takes}
    for repo in discover_video_tree_repos(store, args.raw_root, pattern=args.pattern):
        name = repo.name
        take = name
        if take not in tasks:
            raise ValueError(f"No exact takes.json match: {take}")
        cameras, lengths, rates = [], {}, set()
        for camera in repo.cameras:
            clip = repo.clip_of(0, camera)
            payload = (
                io.BytesIO(store.read_bytes(clip.path))
                if clip.path.startswith("s3://")
                else clip.path
            )
            with av.open(payload) as video:
                stream = video.streams.video[0]
                fps = float(stream.average_rate or stream.guessed_rate or 0)
                if fps <= 0:
                    raise ValueError("Missing native FPS")
                frames = stream.frames or sum(1 for _ in video.decode(stream))
            cameras.append(camera)
            lengths[camera] = frames
            rates.add(fps)
        if len(rates) != 1:
            raise ValueError("Ego aligned camera FPS disagree")
        yield make_row(
            args.source,
            name,
            0,
            cameras,
            lengths,
            dict(
                type="egoexo_aligned",
                root=repo.root + "/frame_aligned_videos",
                fps=rates.pop(),
            ),
            tasks[take],
            str(Path(args.takes).resolve()) + "#" + take,
        )


def robomind_report(args, *, text_overrides=None):
    report = json.loads(Path(args.report).read_text())
    if report["status"] not in ("complete", "planned", "metadata_complete"):
        raise ValueError("RoboMind report is not complete; do not admit failed work")
    import h5py

    for episode in report["episodes"]:
        outputs = episode["outputs"]
        repo = episode.get("repo_id", episode.get("repo"))
        if not repo:
            repo = Path(outputs[0]["path"]).parents[3].name
        member = episode.get("hdf5_relative_path")
        sidecar = Path(outputs[0]["path"]).parents[3] / "text_metadata.json"
        metadata = json.loads(sidecar.read_text())
        member = member or metadata["hdf5_relative_path"]
        raw_path = Path(args.raw_root) / member
        cameras = [output["camera"] for output in outputs]
        with h5py.File(raw_path, "r") as handle:
            lengths = {c: len(handle["observations/rgb_images/" + c]) for c in cameras}
        # Local extracted HDF input uses the same official decoding policy.
        raw = dict(
            type="robomind_official_archive",
            uri=args.archive_path or args.archive_source,
            source_uri=args.archive_source,
            member=member,
            fps=float(metadata["source_fps"]),
            color_policy=metadata["color_policy"],
        )
        row = make_row(
            args.source,
            repo,
            0,
            cameras,
            lengths,
            raw,
            metadata["task"],
            metadata["text_provenance"],
        )
        row["single_view_paths"] = [output["path"] for output in outputs]
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--format", choices=("lerobot", "video_tree", "robomind_report"), required=True
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--takes")
    parser.add_argument("--pattern", default="**/frame_aligned_videos/*.mp4")
    parser.add_argument("--report")
    parser.add_argument("--archive-source")
    parser.add_argument(
        "--archive-path",
        help="Downloaded local gzip archive; provenance stays in --archive-source",
    )
    parser.add_argument(
        "--text-overrides",
        help="JSONL with repo_id, episode_index, task, text_provenance",
    )
    args = parser.parse_args()
    overrides = {}
    if args.text_overrides:
        for line in Path(args.text_overrides).read_text().splitlines():
            row = json.loads(line)
            if not row["text_provenance"]:
                raise ValueError("Text overrides require native provenance")
            key = row["repo_id"], int(row["episode_index"])
            if key in overrides:
                raise ValueError("Duplicate text override")
            overrides[key] = row
    target = Path(args.out).resolve()
    if target.exists():
        raise FileExistsError("Use a new episode index path")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pending.jsonl")
    count = labelled = 0
    seen = set()
    with temporary.open("w") as handle:
        adapter = {
            "lerobot": lerobot,
            "video_tree": video_tree,
            "robomind_report": robomind_report,
        }[args.format]
        for row in adapter(args, text_overrides=overrides):
            override = overrides.pop((row["repo_id"], row["episode_index"]), None)
            if override:
                row.update(
                    task=override["task"],
                    text_provenance=override["text_provenance"],
                    text_status="native_reviewed",
                )
            if row["physical_episode_key"] in seen:
                raise ValueError("Duplicate physical episode identity")
            seen.add(row["physical_episode_key"])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            labelled += bool(row["task"])
    if overrides:
        raise ValueError("Text overrides contain unmatched episodes")
    if not count:
        raise ValueError("No episodes found")
    temporary.replace(target)
    print(
        json.dumps(
            dict(
                episodes=count,
                labelled=labelled,
                missing_text=count - labelled,
                output=str(target),
            )
        )
    )


if __name__ == "__main__":
    main()
