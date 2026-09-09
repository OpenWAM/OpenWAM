"""Bounded S3 reads and synchronized RGB cursors; no S3 mutations."""

from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import av
import boto3
import numpy as np
from botocore.config import Config

from open_wam.data.mixed_video_decode_backends import (
    _TIMESTAMP_BOUNDARY_EPSILON_SECONDS,
)


def split_s3(uri):
    if not uri.startswith("s3://"):
        raise ValueError("Expected an S3 URI")
    bucket, sep, key = uri[5:].partition("/")
    if not sep or not key:
        raise ValueError("Missing S3 object key")
    return bucket, key


class RangeReader(io.RawIOBase):
    """Seekable, ETag-pinned object/member with a small block LRU cache."""

    def __init__(self, client, uri, offset=0, length=None, block_size=1024 * 1024):
        super().__init__()
        self.client = client
        self.uri = uri
        self.local_path = (
            Path(uri).expanduser() if not uri.startswith("s3://") else None
        )
        if self.local_path is not None:
            st = self.local_path.stat()
            self.local_identity = (st.st_size, st.st_mtime_ns, st.st_ino)
            head = {"ETag": str(self.local_identity), "ContentLength": st.st_size}
            self.bucket = self.key = None
        else:
            self.bucket, self.key = split_s3(uri)
            head = client.head_object(Bucket=self.bucket, Key=self.key)
        self.etag = head["ETag"]
        self.object_size = int(head["ContentLength"])
        self.offset = offset
        self.length = self.object_size - offset if length is None else length
        if offset < 0 or self.length < 0 or offset + self.length > self.object_size:
            raise ValueError("Invalid member span")
        self.position = 0
        self.block_size = block_size
        self.blocks = collections.OrderedDict()
        self.fetched_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.length + offset
        )
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        if size <= 0:
            return b""
        output = []
        while size:
            block = self.position // self.block_size
            if block not in self.blocks:
                start = self.offset + block * self.block_size
                end = min(self.offset + self.length, start + self.block_size) - 1
                if self.local_path is not None:
                    st = self.local_path.stat()
                    if (st.st_size, st.st_mtime_ns, st.st_ino) != self.local_identity:
                        raise ValueError("Local raw source changed")
                    with self.local_path.open("rb") as handle:
                        handle.seek(start)
                        data = handle.read(end - start + 1)
                else:
                    response = self.client.get_object(
                        Bucket=self.bucket,
                        Key=self.key,
                        Range=f"bytes={start}-{end}",
                        IfMatch=self.etag,
                    )
                    try:
                        data = response["Body"].read()
                    finally:
                        response["Body"].close()
                if len(data) != end - start + 1:
                    raise IOError("Short S3 range read")
                self.fetched_bytes += len(data)
                self.blocks[block] = data
                while len(self.blocks) > 8:
                    self.blocks.popitem(last=False)
            self.blocks.move_to_end(block)
            data = self.blocks[block]
            start = self.position % self.block_size
            take = min(size, len(data) - start)
            if take <= 0:
                raise IOError("Invalid cached range")
            output.append(data[start : start + take])
            self.position += take
            size -= take
        return b"".join(output)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self):
        self.blocks.clear()
        super().close()


class MetadataStore:
    """Compatible with the existing LeRobot metadata resolver, with provenance."""

    def __init__(self, root):
        self._client = None
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.repos = {}
        self.receipts = {}
        self.tar_indexes = {}

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                config=Config(
                    read_timeout=120,
                    retries={"max_attempts": 4},
                    max_pool_connections=16,
                ),
            )
        return self._client

    def read_bytes(self, uri):
        if not uri.startswith("s3://"):
            path = Path(uri).expanduser()
            if path.stat().st_size > 256 * 1024**2:
                raise ValueError("Oversized metadata object")
            return path.read_bytes()
        key = hashlib.sha256(uri.encode()).hexdigest()
        path = self.root / (key + ".bin")
        receipt = self.root / (key + ".json")
        if path.exists() and receipt.exists():
            data = path.read_bytes()
            meta = json.loads(receipt.read_text())
            if hashlib.sha256(data).hexdigest() != meta["sha256"]:
                raise ValueError("Metadata cache checksum mismatch")
            self.receipts[uri] = meta
            return data
        bucket, obj = split_s3(uri)
        response = self.client.get_object(Bucket=bucket, Key=obj)
        if response["ContentLength"] > 256 * 1024**2:
            response["Body"].close()
            raise ValueError("Oversized metadata object")
        try:
            data = response["Body"].read()
        finally:
            response["Body"].close()
        if len(data) != response["ContentLength"]:
            raise IOError("Short metadata read")
        meta = {
            "uri": uri,
            "bytes": len(data),
            "etag": response["ETag"],
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        tmp = receipt.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(receipt)
        self.receipts[uri] = meta
        return data

    def list_keys(self, prefix, suffix=".parquet"):
        if not prefix.startswith("s3://"):
            root = Path(prefix).expanduser()
            return {str(p.relative_to(root)) for p in root.rglob("*" + suffix)}
        bucket, key = split_s3(prefix)
        key = key.rstrip("/") + "/"
        found = []
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=key
        ):
            found.extend(
                x["Key"][len(key) :]
                for x in page.get("Contents", [])
                if x["Key"].endswith(suffix)
            )
        return set(found)

    def repo(self, root):
        if root not in self.repos:
            # The old v3 parser merely logs conflicting metadata. Treat that
            # ambiguity as a hard failure in this new synchronization path.
            import contextlib

            from open_wam.data.preparation.video_sources import load_repo

            log = io.StringIO()
            with contextlib.redirect_stderr(log):
                repo = load_repo(self, root)
            if "disagrees" in log.getvalue():
                raise ValueError(log.getvalue())
            self.repos[root] = repo
        return self.repos[root]

    def tar_index(self, uri):
        if uri in self.tar_indexes:
            return self.tar_indexes[uri]
        # AgiBot tars contain hundreds of thousands of ~100 KiB depth PNGs.
        # A tiny range per tar header is latency-bound. Large cached blocks
        # amortize those requests while bounding RAM to 64 MiB per indexer.
        reader = RangeReader(
            self.client if uri.startswith("s3://") else None,
            uri,
            block_size=8 * 1024**2,
        )
        key = hashlib.sha256((uri + "|" + reader.etag).encode()).hexdigest()
        cached = self.root / (key + ".tar-index.json")
        if cached.exists():
            index = json.loads(cached.read_text())
        else:
            progress = self.root / (key + ".partial-index.json")
            previous = json.loads(progress.read_text()) if progress.exists() else {}
            index = previous.get("index", {})
            scanned = int(previous.get("scanned", 0))
            reader.seek(int(previous.get("next_offset", 0)))
            with tarfile.open(
                fileobj=reader, mode="r:", pax_headers=previous.get("pax_headers", {})
            ) as archive:
                for member in archive:
                    scanned += 1
                    if member.isfile() and member.name.endswith(".mp4"):
                        name = member.name.removeprefix("./")
                        if name in index:
                            raise ValueError("Duplicate tar member")
                        index[name] = {
                            "offset": member.offset_data,
                            "size": member.size,
                            "etag": reader.etag,
                        }
                    if scanned % 10000 == 0:
                        state = {
                            "index": index,
                            "scanned": scanned,
                            "next_offset": archive.offset,
                            "pax_headers": archive.pax_headers,
                        }
                        tmp = progress.with_suffix(f".{os.getpid()}.tmp")
                        tmp.write_text(json.dumps(state))
                        tmp.replace(progress)
                        print(
                            json.dumps(
                                {
                                    "event": "tar_index_progress",
                                    "uri": uri,
                                    "members": scanned,
                                    "videos": len(index),
                                    "offset": archive.offset,
                                    "total_bytes": reader.object_size,
                                }
                            ),
                            flush=True,
                        )
            tmp = cached.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(index))
            tmp.replace(cached)
        reader.close()
        self.tar_indexes[uri] = index
        return index


class VideoCursor:
    def __init__(self, reader, fps, expected_frames, start=0.0, stop=None):
        self.reader = reader
        self.container = av.open(reader)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.stream.codec_context.thread_count = 2
        self.fps = float(fps)
        self.start = float(start or 0.0)
        self.stop = stop
        container_rate = float(
            self.stream.average_rate or self.stream.guessed_rate or 0
        )
        if container_rate and abs(container_rate - self.fps) > max(
            0.1, self.fps * 0.01
        ):
            raise ValueError(f"Container FPS {container_rate} != declared {self.fps}")
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height
        self.duration = (expected_frames - 1) / self.fps
        if stop is not None:
            self.duration = min(
                self.duration, max(0.0, stop - self.start - 1 / self.fps)
            )
        if self.start:
            self.container.seek(
                int(self.start / self.stream.time_base),
                stream=self.stream,
                backward=True,
            )
        self.iterator = self._frames()
        self.previous = None
        self.next = next(self.iterator, None)
        if self.next is None:
            raise ValueError("No frames in clip")
        if abs(self.next[0]) > 1.1 / self.fps:
            raise ValueError("Clip start is not aligned to declared timestamp")
        self.last_t = -1.0
        self.max_time_error = 0.0

    def _frames(self):
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                raise ValueError("Video frame has no timestamp")
            t = float(frame.pts * self.stream.time_base)
            if t < self.start - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS:
                continue
            if (
                self.stop is not None
                and t >= self.stop - _TIMESTAMP_BOUNDARY_EPSILON_SECONDS
            ):
                break
            yield t - self.start, frame

    def at(self, t):
        if t < self.last_t - 1e-6:
            raise ValueError("Nonmonotonic target timeline")
        self.last_t = t
        while self.next is not None and self.next[0] < t:
            self.previous = self.next
            self.next = next(self.iterator, None)
        choices = [x for x in (self.previous, self.next) if x is not None]
        if not choices:
            raise ValueError("No frame at timestamp")
        moment, frame = min(choices, key=lambda x: abs(x[0] - t))
        error = abs(moment - t)
        self.max_time_error = max(self.max_time_error, error)
        if error > 1.1 / self.fps:
            raise ValueError(f"Missing/truncated frame at {t}: nearest {moment}")
        return frame.to_ndarray(format="rgb24")

    def close(self):
        self.container.close()
        self.reader.close()


class HDFCursor:
    def __init__(self, handle, camera, raw):
        self.dataset = handle[f"observations/rgb_images/{camera}"]
        self.raw = raw
        self.fps = float(raw["fps"])
        self.duration = (len(self.dataset) - 1) / self.fps
        self.counts = collections.Counter()
        frame = self._decode(0)
        self.height, self.width = frame.shape[:2]
        self.max_time_error = 0.0

    def _decode(self, index):
        native = self.dataset[index]
        source_shape = (
            tuple(native.shape)
            if isinstance(native, np.ndarray) and native.ndim > 1
            else None
        )
        if source_shape is not None and (
            native.ndim != 3 or source_shape[-1] != 3 or native.dtype != np.uint8
        ):
            raise ValueError("Expected explicit uint8 HWC source geometry")
        raw = bytes(native)
        if self.raw["type"] == "robomind_official_archive":
            from open_wam.data.preparation.encoding.robomind_rgb import (
                decode_rgb_frame,
                resolve_embodiment,
            )

            frame, encoding = decode_rgb_frame(
                raw,
                resolve_embodiment(self.raw.get("source_uri", self.raw["uri"])),
                source_shape=source_shape,
            )
        else:
            if self.raw[
                "color_policy"
            ] != "robomind_failure_standard_jpeg_v1" or not raw.startswith(b"\xff\xd8"):
                raise ValueError("Uncertified failure image encoding")
            from PIL import Image

            frame = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            encoding = "jpeg"
        self.counts[encoding] += 1
        return frame

    def at(self, t):
        idx = round(t * self.fps)
        if idx >= len(self.dataset):
            raise ValueError("HDF camera ended before target time")
        self.max_time_error = max(self.max_time_error, abs(idx / self.fps - t))
        return self._decode(idx)

    def close(self):
        pass


def open_cursors(plan, store, hdf_path=None):
    raw = plan["raw_source"]
    cameras = plan["cameras"]
    cursors = {}
    owned = []
    try:
        if raw["type"].startswith("robomind_"):
            import h5py

            source = (
                hdf_path
                if hdf_path
                else RangeReader(
                    store.client if raw["uri"].startswith("s3://") else None, raw["uri"]
                )
            )
            if hdf_path is None:
                owned.append(source)
            evidence = plan.get("orientation_evidence")
            if evidence and (hdf_path is not None or source.etag != evidence["etag"]):
                raise ValueError("Audited orientation source changed; review required")
            handle = h5py.File(source, "r")
            owned.append(handle)
            for camera in cameras:
                cursors[camera] = HDFCursor(handle, camera, raw)
        else:
            repo = store.repo(raw["root"]) if raw["type"] == "lerobot" else None
            for camera in cameras:
                start = 0.0
                stop = None
                fps = float(raw.get("fps") or repo.fps)
                expected = plan["native_end_frames"][camera]
                if repo:
                    clip = repo.clip_of(plan["episode_index"], camera)
                    reader = RangeReader(
                        store.client if clip.path.startswith("s3://") else None,
                        clip.path,
                    )
                    start = clip.from_timestamp or 0.0
                    stop = clip.to_timestamp
                    declared = repo.length_of(plan["episode_index"])
                    if declared is not None and abs(declared - expected) > 1:
                        raise ValueError(
                            f"Source metadata length {declared} differs from frozen span {expected}"
                        )
                elif raw["type"] == "egoexo_aligned":
                    reader = RangeReader(
                        store.client if raw["root"].startswith("s3://") else None,
                        raw["root"] + "/" + camera + ".mp4",
                    )
                elif raw["type"] == "agibot_tar":
                    index = store.tar_index(raw["uri"])
                    suffix = raw["member_episode"] + "/videos/" + camera + ".mp4"
                    matches = [
                        v
                        for k, v in index.items()
                        if k == suffix or k.endswith("/" + suffix)
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"No unique AgiBot camera member: {suffix}")
                    member = matches[0]
                    reader = RangeReader(
                        store.client if raw["uri"].startswith("s3://") else None,
                        raw["uri"],
                        member["offset"],
                        member["size"],
                    )
                    if reader.etag != member["etag"]:
                        raise ValueError("AgiBot archive changed since indexing")
                else:
                    raise ValueError("Unknown raw adapter")
                owned.append(reader)
                cursors[camera] = VideoCursor(reader, fps, expected, start, stop)
        return cursors, owned
    except BaseException:
        for item in list(cursors.values()) + owned:
            try:
                item.close()
            except Exception:
                pass
        raise
