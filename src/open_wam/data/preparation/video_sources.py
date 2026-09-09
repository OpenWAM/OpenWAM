"""Video sources."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

from open_wam.data.preparation.storage import join

DEFAULT_VIDEO_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


@dataclass(frozen=True)
class Clip:
    """Where one episode's footage lives for one camera.

    v2.1 gives every episode its own file, so the range covers the whole thing.
    v3.0 concatenates many episodes into one file and identifies each by a
    timestamp range, which is why the range has to be carried around at all.
    """

    path: str
    from_timestamp: float | None = None
    to_timestamp: float | None = None


@dataclass(frozen=True)
class Repo:
    """One LeRobot repo root plus the bits of meta/info.json we rely on."""

    root: str
    fps: float
    chunk_size: int
    total_episodes: int
    cameras: tuple[str, ...]
    video_template: str
    version: str = "v2.1"
    clips: dict[tuple[int, str], Clip] | None = None
    # Declared row count per episode, keyed by episode index. The only
    # independent statement of how long an episode really is, and so the only
    # way to notice a decode that came back short.
    lengths: dict[int, int] | None = None
    # The prefix the repo was discovered under. `name` is taken relative to this,
    # because the last path segment alone is not unique once a corpus nests its
    # repos: VPT-07 puts 1,488 repos under 290 distinct last segments, and the
    # 194 collisions silently overwrote one another in a shared output directory.
    scan_root: str | None = None

    def length_of(self, episode: int) -> int | None:
        return None if self.lengths is None else self.lengths.get(episode)

    @property
    def name(self) -> str:
        root = str(self.root).rstrip("/")
        if self.scan_root:
            base = str(self.scan_root).rstrip("/")
            if root.startswith(base + "/"):
                # Flatten to one directory level so the layout below `out_root`
                # is unchanged for every consumer -- make_manifest reads repo_id
                # from the path, and a deeper tree would change what it reads.
                return root[len(base) + 1 :].replace("/", "__")
        return PurePosixPath(root).name

    def chunk_of(self, episode: int) -> int:
        return episode // self.chunk_size

    def clip_of(self, episode: int, camera: str) -> Clip:
        if self.clips is not None:
            try:
                return self.clips[(episode, camera)]
            except KeyError:
                raise KeyError(
                    f"{self.name}: no clip for episode {episode} camera {camera}"
                ) from None
        return Clip(
            join(
                self.root,
                self.video_template.format(
                    episode_chunk=self.chunk_of(episode),
                    video_key=camera,
                    episode_index=episode,
                ),
            )
        )


def load_repo(store, root: str, scan_root: str | None = None) -> Repo:
    info = json.loads(store.read_bytes(join(root, "meta/info.json")).decode())
    features = info.get("features", {})
    cameras = tuple(
        key
        for key, spec in features.items()
        if spec.get("dtype") in {"video", "image"} or "image" in key.lower()
    )
    if not cameras:
        raise ValueError(f"{root}: no video/image features in meta/info.json")
    version = str(info.get("codebase_version") or "v2.1")
    template = info.get("video_path") or DEFAULT_VIDEO_TEMPLATE
    clips = None
    if "{chunk_index}" in template or version.startswith("v3"):
        clips = load_v3_clips(store, str(root), template, cameras)

    return Repo(
        lengths=load_episode_lengths(store, str(root)),
        scan_root=scan_root,
        root=str(root),
        fps=float(info["fps"]),
        chunk_size=int(info.get("chunks_size") or info.get("chunk_size") or 1000),
        total_episodes=int(info["total_episodes"]),
        cameras=cameras,
        video_template=template,
        version=version,
        clips=clips,
    )


def load_episode_lengths(store, root: str) -> dict[int, int] | None:
    """Read the declared length of every episode, if the repo states them.

    One small object per repo, read once at discovery. It is what makes the
    length check at write time possible: without an independent count there is
    nothing to compare a decode against, and a clip that came back short would
    be written as though it were whole.
    """

    try:
        raw = store.read_bytes(join(root, "meta/episodes.jsonl")).decode()
    except Exception:
        return None
    lengths: dict[int, int] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
        except (ValueError, KeyError, TypeError):
            continue
    return lengths or None


def load_v3_clips(
    store, root: str, template: str, cameras: tuple[str, ...]
) -> dict[tuple[int, str], Clip]:
    """Index a v3.0 repo's episodes to the file and time range that hold them.

    v3.0 packs many episodes into one mp4 per camera, so the only way to find an
    episode is `meta/episodes/`, which records the file and the timestamps it
    spans for each camera separately.
    """

    import pyarrow.parquet as pq

    keys = sorted(store.list_keys(join(root, "meta/episodes"), suffix=".parquet"))
    if not keys:
        raise FileNotFoundError(f"{root}: v3.0 repo without meta/episodes/*.parquet")

    clips: dict[tuple[int, str], Clip] = {}
    conflicts = 0
    for key in keys:
        table = pq.read_table(
            io.BytesIO(store.read_bytes(join(root, "meta/episodes", key)))
        )
        wanted = ["episode_index"]
        for camera in cameras:
            wanted += [
                f"videos/{camera}/chunk_index",
                f"videos/{camera}/file_index",
                f"videos/{camera}/from_timestamp",
                f"videos/{camera}/to_timestamp",
            ]
        present = [c for c in wanted if c in table.schema.names]
        for row in table.select(present).to_pylist():
            episode = int(row["episode_index"])
            for camera in cameras:
                chunk = row.get(f"videos/{camera}/chunk_index")
                if chunk is None:
                    continue
                candidate = Clip(
                    path=join(
                        root,
                        template.format(
                            video_key=camera,
                            chunk_index=int(chunk),
                            file_index=int(row[f"videos/{camera}/file_index"]),
                        ),
                    ),
                    from_timestamp=float(row[f"videos/{camera}/from_timestamp"]),
                    to_timestamp=float(row[f"videos/{camera}/to_timestamp"]),
                )
                # These files are meant to partition the episodes between them.
                # Where they overlap instead, one of the two is stale, and taking
                # the later one silently encoded the wrong footage. Keep the
                # first and say so, rather than trusting whichever sorted last.
                existing = clips.get((episode, camera))
                if existing is None:
                    clips[(episode, camera)] = candidate
                elif existing != candidate:
                    conflicts += 1

    if conflicts:
        raise ValueError(
            f"{root}: metadata disagrees for {conflicts} episode/camera pairs; repair the native index before encoding"
        )
    return clips


def discover_video_tree_repos(store, root: str, *, pattern: str) -> list[Repo]:
    """Treat a directory of videos as one episode, each file in it a camera.

    Corpora that ship no LeRobot metadata still have the structure the encoder
    needs — a per-episode folder whose files are synchronised camera streams —
    it just has to be read off the paths. Frame rate comes from the container
    rather than a manifest, since there is no manifest to read.

    `pattern` is a glob relative to `root`; everything before the last directory
    separator identifies the episode, the file stem names the camera.
    """

    import fnmatch

    keys = sorted(store.list_keys(root, suffix=".mp4"))
    matched = [k for k in keys if fnmatch.fnmatch(k, pattern)]
    if not matched:
        raise FileNotFoundError(
            f"No videos under {root} matching {pattern!r} ({len(keys)} mp4 seen)"
        )

    grouped: dict[str, list[str]] = {}
    for key in matched:
        grouped.setdefault(PurePosixPath(key).parent.as_posix(), []).append(key)

    repos = []
    for index, (folder, members) in enumerate(sorted(grouped.items())):
        cameras = tuple(sorted(PurePosixPath(m).stem for m in members))
        clips = {(0, PurePosixPath(m).stem): Clip(join(root, m)) for m in members}
        # `Repo.name` is the last path component, which for a nested layout is a
        # shared directory like "448" — every episode would collide on it. The
        # first component is the one that identifies the take.
        episode_name = PurePosixPath(folder).parts[0]
        repos.append(
            Repo(
                root=join(root, episode_name),
                # The container is the only source of truth here; the decoder
                # reads the real rate and resampling corrects for it.
                fps=0.0,
                chunk_size=1000,
                total_episodes=1,
                cameras=cameras,
                video_template="",
                version="video_tree",
                clips=clips,
            )
        )
    return repos


def discover_repos(store, root: str, keep: set[str] | None = None) -> list[Repo]:
    """Find the LeRobot repos under `root`, optionally only the named ones.

    Naming the repos skips the search entirely, which matters more than it
    looks: finding them means listing every key under `root`, and for a corpus
    this size that is hundreds of LIST requests per process. A run that restarts
    every few minutes, against a prefix something else is still uploading to,
    should not spend the bucket's request budget rediscovering what it was told.
    """

    if keep is None:
        roots = store.find_repo_roots(root)
        if not roots:
            raise FileNotFoundError(
                f"No LeRobot repo (meta/info.json) found under {root}"
            )
        return [load_repo(store, r, scan_root=root) for r in roots]

    # Named repos need no search. Finding them means listing every key under
    # `root` -- for a corpus of this size, hundreds of LIST requests per process
    # per pass, against the same bucket an upload is still writing to. The
    # caller already knows the names, so address the repos directly and let a
    # name that turns out not to exist fail as the one repo it is.
    found = []
    for name in sorted(keep):
        candidate = join(root, name)
        try:
            found.append(load_repo(store, candidate, scan_root=root))
        except Exception as error:
            print(f"    ! {name}: {error}", file=sys.stderr)
    if not found:
        raise FileNotFoundError(
            f"None of the {len(keep)} named repos loaded under {root}"
        )
    return found
