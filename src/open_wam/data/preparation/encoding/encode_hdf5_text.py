#!/usr/bin/env python3
"""Encode RoboMind with native text provenance and an explicit archive contract.

Video transforms and payload geometry reuse the existing HDF5/video encoders.
Only outputs certified with the current RGB policy may be reused.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import torch

from open_wam.artifacts.files import atomic_json, sha256_file
from open_wam.configs.data_mixed_video import MixedVideoResizeBinConfig
from open_wam.data.preparation import frames
from open_wam.data.preparation.encoding.payload import build_payload
from open_wam.data.preparation.encoding.robomind_metadata import (
    RGB_GROUP,
    read_native_text,
)
from open_wam.data.preparation.encoding.robomind_rgb import (
    COLOR_POLICY,
    read_rgb_frames,
    resolve_embodiment,
    validate_color_metadata,
)
from open_wam.data.preparation.encoding.robomind_rgb_evidence import (
    EVIDENCE_FIELDS,
    attach_contract,
    rgb_evidence,
    validate_payload_evidence,
)
from open_wam.data.mixed_video_decode_frames import select_mixed_video_resize_bin
from open_wam.data.preparation.frames import (
    DEFAULT_BINS,
    fit_frames,
    resample_indices,
    truncate_to_temporal_stride,
)
from open_wam.data.preparation.storage import LocalStore
from open_wam.models.visual_tower import vae_encoding
from open_wam.models.visual_tower.vae_encoding import encode_clip, load_vae


def model_fingerprint(vae_path: str) -> dict:
    """Hash actual local VAE assets once; a changed file identity invalidates cache."""
    root = Path(vae_path).resolve()
    config = root / "config.json"
    weights = sorted(root.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise ValueError(
            "A local VAE config and safetensor weights are required for alias evidence"
        )
    paths = [config, *weights]

    def signature():
        return [
            {
                "path": str(p),
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
                "ctime_ns": p.stat().st_ctime_ns,
                "inode": p.stat().st_ino,
            }
            for p in paths
        ]

    cache = Path(
        os.environ.get(
            "OPENWAM_FINGERPRINT_CACHE", "~/.cache/openwam/model_fingerprints"
        )
    ).expanduser() / (hashlib.sha256(str(root).encode()).hexdigest() + ".json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = signature()
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get("files") == before:
                return saved["fingerprint"]
        result = {
            "config_sha256": sha256_file(config),
            "weights_sha256": {p.name: sha256_file(p) for p in weights},
        }
        if signature() != before:
            raise ValueError("VAE assets changed during fingerprinting")
        atomic_json(cache, {"files": before, "fingerprint": result})
        return result


def sampling_contract(args, source_shape):
    import diffusers

    indices = resample_indices(source_shape[0], args.source_fps, args.fps)
    keep = truncate_to_temporal_stride(len(indices), 4)
    if keep < 1:
        raise ValueError("Episode too short after original temporal resampling")
    indices = indices[:keep]
    bin_config = (
        select_mixed_video_resize_bin(
            DEFAULT_BINS, source_height=source_shape[1], source_width=source_shape[2]
        )
        if args.size_mode == "aspect_bins"
        else MixedVideoResizeBinConfig("fixed", 1, 1, args.resolution, args.resolution)
    )
    if not hasattr(args, "_model_fingerprint"):
        args._model_fingerprint = model_fingerprint(args.vae)
    if not hasattr(args, "_implementation_fingerprint"):
        implementation = {
            module.__name__: sha256_file(module.__file__)
            for module in (frames, vae_encoding)
        }
        args._implementation_fingerprint = hashlib.sha256(
            json.dumps(implementation, sort_keys=True).encode()
        ).hexdigest()
    contract = dict(
        format_version=1,
        color_policy=COLOR_POLICY,
        embodiment=args.embodiment,
        source_rgb_shape=list(source_shape),
        source_fps=args.source_fps,
        fps=args.fps,
        frame_ids=[int(i) for i in indices],
        temporal_stride=4,
        spatial_stride=16,
        size_mode=args.size_mode,
        bins=[asdict(b) for b in DEFAULT_BINS],
        selected_bin=asdict(bin_config),
        fit_mode=args.fit_mode,
        latents_normalized=not args.no_normalize,
        store_dtype=args.store_dtype,
        encode_dtype="bf16",
        vae_fingerprint=args._model_fingerprint,
        encoder_implementation_sha256=args._implementation_fingerprint,
        runtime_versions={
            "torch": str(torch.__version__),
            "diffusers": str(diffusers.__version__),
        },
    )
    return contract, indices, bin_config


def record_evidence(text, camera, evidence):
    text["camera_rgb_sha256"][camera] = evidence["source_rgb_sha256"]
    text["camera_encode_evidence"][camera] = evidence


def prove_alias(args, path, incoming, canonical, sidecar, outputs):
    """Prove complete decoded videos and all encode inputs; never mutate canonical files."""
    if args.cameras.strip().lower() != "auto":
        raise ValueError(
            "Archive aliases require the full automatic live-camera inventory"
        )
    identity_fields = (
        "archive_id",
        "archive_source",
        "hdf5_relative_path",
        "repo_id",
        "embodiment",
        "source_frames",
        "source_fps",
        "cameras",
        "task",
        "native_fields",
    )
    equivalent_fields = set(identity_fields) - {
        "archive_id",
        "archive_source",
        "hdf5_relative_path",
    }
    for field in identity_fields:
        if field not in canonical:
            raise ValueError(f"Canonical sidecar lacks alias evidence: {field}")
    for field in equivalent_fields:
        if canonical[field] != incoming[field]:
            raise ValueError(f"Archive alias differs on {field}")
    if resolve_embodiment(canonical["archive_source"]) != args.embodiment:
        raise ValueError("Alias embodiment differs")
    if canonical.get("live_cameras") != incoming["live_cameras"] or canonical.get(
        "cameras"
    ) != canonical.get("live_cameras"):
        raise ValueError(
            "Canonical alias lacks an identical full live-camera inventory"
        )
    sidecar_hash = sha256_file(sidecar)
    evidence = {
        "format_version": 1,
        "incoming": {k: incoming[k] for k in identity_fields},
        "canonical": {k: canonical[k] for k in identity_fields},
        "canonical_sidecar_sha256": sidecar_hash,
        "canonical_payload_sha256": {},
        "camera_evidence": {},
    }
    for output in outputs:
        camera, target = output["camera"], Path(output["path"])
        payload_hash = sha256_file(target)
        payload = torch.load(target, map_location="cpu", weights_only=True, mmap=True)
        color = validate_color_metadata(payload, canonical["archive_source"])
        old = validate_payload_evidence(payload)
        if old != canonical.get("camera_encode_evidence", {}).get(camera) or old[
            "source_rgb_sha256"
        ] != canonical.get("camera_rgb_sha256", {}).get(camera):
            raise ValueError(
                f"Canonical RGB evidence is missing or inconsistent: {camera}"
            )
        if color != canonical.get("camera_color_metadata", {}).get(camera):
            raise ValueError(f"Canonical color metadata is inconsistent: {camera}")
        for field in ("task", "text_status", "text_provenance"):
            if payload.get(field) != canonical.get(field):
                raise ValueError(f"Canonical payload text differs on {field}")
        if payload.get("text_metadata_path") != str(sidecar):
            raise ValueError("Canonical payload sidecar path differs")
        frames, _ = read_rgb_frames(str(path), camera, args.archive_source)
        contract, _, _ = sampling_contract(args, frames.shape)
        new = attach_contract(rgb_evidence(frames), contract)
        del frames
        if new != old:
            raise ValueError(f"Archive alias RGB or encode contract differs: {camera}")
        if sha256_file(target) != payload_hash:
            raise ValueError("Canonical payload changed during alias verification")
        evidence["canonical_payload_sha256"][camera] = payload_hash
        evidence["camera_evidence"][camera] = {"incoming": new, "canonical": old}
        output.update(**old, **color)
    if sha256_file(sidecar) != sidecar_hash:
        raise ValueError("Canonical sidecar changed during alias verification")
    return evidence


@dataclass
class Episode:
    path: Path
    repo: str
    frames: int
    cameras: tuple[str, ...]
    text: dict
    sidecar: Path
    outputs: list[dict]
    alias_evidence: dict | None = None


def discover(args) -> tuple[list[Episode], list[dict], int]:
    root = Path(args.dataset)
    paths = sorted(root.rglob("trajectory.hdf5"))
    if args.limit:
        paths = paths[: args.limit]
    wanted = (
        None
        if args.cameras.strip().lower() == "auto"
        else set(c.strip() for c in args.cameras.split(",") if c.strip())
    )
    episodes, errors, repos = [], [], set()
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
            parts = [
                p for p in path.parent.relative_to(root).parts if p not in (".", "data")
            ]
            repo = "__".join(parts).replace(" ", "_")
            if not repo or repo in repos:
                raise ValueError(
                    f"Duplicate or empty original encoder repo key: {repo}"
                )
            repos.add(repo)
            with h5py.File(path, "r") as handle:
                text = read_native_text(handle)
                if RGB_GROUP not in handle:
                    raise ValueError("no_rgb_group")
                live, unavailable, frames = [], [], 0
                camera_lengths = {}
                for camera, dataset in handle[RGB_GROUP].items():
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError(f"Camera entry is not a dataset: {camera}")
                    if not len(dataset) or not len(bytes(dataset[0])):
                        unavailable.append(camera)
                        continue
                    live.append(camera)
                    camera_lengths[camera] = len(dataset)
                    frames = max(frames, len(dataset))
                if not live:
                    raise ValueError("no_live_camera")
            cameras = tuple(c for c in live if wanted is None or c in wanted)
            if not cameras:
                raise ValueError("no_requested_live_camera")
            provenance = f"{args.archive_source}#hdf5={relative};field={'|'.join(text['text_fields'])}"
            text.update(
                format_version=1,
                archive_id=args.archive_id,
                archive_source=args.archive_source,
                hdf5_relative_path=relative,
                repo_id=repo,
                episode_index=0,
                text_provenance=provenance,
                cameras=list(cameras),
                empty_placeholder_cameras=unavailable,
                source_frames=frames,
                source_fps=args.source_fps,
                live_cameras=list(live),
                camera_rgb_sha256={},
                camera_encode_evidence={},
                color_policy=COLOR_POLICY,
                embodiment=args.embodiment,
                camera_color_metadata={},
            )
            sidecar = Path(args.out_root) / repo / "text_metadata.json"
            outputs = []
            for camera in cameras:
                stem = f"episode_000000_0_{camera_lengths[camera]}"
                target = (
                    Path(args.out_root)
                    / repo
                    / "latents/chunk-000"
                    / camera
                    / (stem + ".pth")
                )
                outputs.append(
                    {"path": str(target), "camera": camera, "status": "pending"}
                )
            alias = None
            if sidecar.exists():
                previous = json.loads(sidecar.read_text())
                if previous.get("archive_source") != text["archive_source"]:
                    alias = prove_alias(args, path, text, previous, sidecar, outputs)
                    text = previous
                else:
                    for field in (
                        "archive_source",
                        "hdf5_relative_path",
                        "repo_id",
                        "task",
                        "native_fields",
                        "color_policy",
                        "embodiment",
                        "cameras",
                        "source_frames",
                    ):
                        if previous.get(field) != text[field]:
                            raise ValueError(
                                f"Existing text sidecar conflicts on {field}: {sidecar}"
                            )
                    if (
                        "source_fps" in previous
                        and previous["source_fps"] != args.source_fps
                    ):
                        raise ValueError("Existing sidecar source FPS differs")
                    for field in (
                        "camera_color_metadata",
                        "camera_rgb_sha256",
                        "camera_encode_evidence",
                    ):
                        text[field].update(previous.get(field, {}))
            else:
                atomic_json(sidecar, text)
            episodes.append(
                Episode(path, repo, frames, cameras, text, sidecar, outputs, alias)
            )
        except Exception as error:
            errors.append(
                {"hdf5": str(path), "error": f"{type(error).__name__}: {error}"}
            )
    return episodes, errors, len(paths)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--archive-id", required=True)
    parser.add_argument("--archive-source", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dataset-id", default="VPT-06")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--cameras", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--store-dtype", default="fp16")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--size-mode", default="aspect_bins")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fit-mode", default="letterbox_pad")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.embodiment = resolve_embodiment(args.archive_source)
    except ValueError as error:
        parser.error(str(error))
    if args.limit and not (args.metadata_only or args.plan):
        parser.error(
            "--limit is only permitted with --metadata-only or --plan; archive completion requires all episodes"
        )
    started = time.monotonic()
    episodes, errors, discovered = discover(args)
    expected = sum(len(e.outputs) for e in episodes)
    report = {
        "format_version": 1,
        "status": "prepared",
        "archive_id": args.archive_id,
        "color_policy": COLOR_POLICY,
        "embodiment": args.embodiment,
        "resize_policy_new_outputs": "shared_default_bins_20260905",
        "resize_policy_skipped_outputs": "preserve_existing_payload_geometry",
        "archive_source": args.archive_source,
        "dataset": args.dataset,
        "source_hdf5_count": discovered,
        "usable_episodes": len(episodes),
        "metadata_failed": len(errors),
        "metadata_errors": errors,
        "expected_clips": expected,
        "wrote": 0,
        "skipped": 0,
        "failed": 0,
        "text_labeled_episodes": sum(bool(e.text["task"]) for e in episodes),
        "text_missing_episodes": sum(not bool(e.text["task"]) for e in episodes),
        "episodes": [
            {
                "repo_id": e.repo,
                "text_sidecar": str(e.sidecar),
                "task": e.text["task"],
                "text_status": e.text["text_status"],
                "text_provenance": e.text["text_provenance"],
                "color_policy": COLOR_POLICY,
                "embodiment": args.embodiment,
                "camera_color_metadata": e.text["camera_color_metadata"],
                "source_fps": e.text.get("source_fps"),
                "camera_rgb_sha256": e.text.get("camera_rgb_sha256", {}),
                "camera_encode_evidence": e.text.get("camera_encode_evidence", {}),
                **({"alias_evidence": e.alias_evidence} if e.alias_evidence else {}),
                "outputs": e.outputs,
            }
            for e in episodes
        ],
    }
    if errors or not expected:
        report["status"] = "failed_metadata"
        atomic_json(args.report, report)
        print(
            json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True
        )
        return 2
    atomic_json(args.report, report)
    if args.metadata_only or args.plan:
        print(
            json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True
        )
        return 0
    store, vae = LocalStore(), None
    for e in episodes:
        for output in e.outputs:
            target, camera = Path(output["path"]), output["camera"]
            try:
                if e.alias_evidence:
                    # Discovery already verified every incoming frame against immutable canonical evidence.
                    output.update(status="skipped", bytes=target.stat().st_size)
                    report["skipped"] += 1
                    continue
                if target.exists():
                    if not target.is_file() or target.stat().st_size == 0:
                        raise ValueError("Existing latent is not a nonempty file")
                    # Never certify or reuse pre-fix tensors by filename alone.
                    existing = torch.load(
                        target, map_location="cpu", weights_only=True, mmap=True
                    )
                    color = validate_color_metadata(existing, args.archive_source)
                    evidence = {}
                    if any(field in existing for field in EVIDENCE_FIELDS):
                        evidence = validate_payload_evidence(existing)
                        contract, _, _ = sampling_contract(
                            args, evidence["source_rgb_shape"]
                        )
                        if contract != evidence["encode_contract"]:
                            raise ValueError(
                                "Existing latent encode contract differs from requested encoding"
                            )
                        record_evidence(e.text, camera, evidence)
                    elif (
                        camera in e.text["camera_encode_evidence"]
                        or camera in e.text["camera_rgb_sha256"]
                    ):
                        raise ValueError(
                            "Existing latent lost RGB evidence recorded in its sidecar"
                        )
                    e.text["camera_color_metadata"][camera] = color
                    atomic_json(e.sidecar, e.text)
                    report["skipped"] += 1
                    output.update(
                        status="skipped",
                        bytes=target.stat().st_size,
                        **color,
                        **evidence,
                    )
                    continue
                frames, color = read_rgb_frames(
                    str(e.path), camera, args.archive_source
                )
                contract, indices, bin_config = sampling_contract(args, frames.shape)
                evidence = attach_contract(rgb_evidence(frames), contract)
                if vae is None:
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    vae = load_vae(args.vae, device, torch.bfloat16)
                video = fit_frames(
                    frames[indices],
                    bin_config.target_height,
                    bin_config.target_width,
                    args.fit_mode,
                )
                latent = encode_clip(vae, video, normalize=not args.no_normalize)
                payload = build_payload(
                    latent=latent.cpu(),
                    camera=camera,
                    episode_index=0,
                    indices=indices,
                    source_frames=len(frames),
                    source_fps=args.source_fps,
                    bin_config=bin_config,
                    fps=args.fps,
                    store_dtype=args.store_dtype,
                    fit_mode=args.fit_mode,
                    normalize_latents=not args.no_normalize,
                    vae_path=args.vae,
                )
                payload.update(
                    task=e.text["task"],
                    text_provenance=e.text["text_provenance"],
                    text_status=e.text["text_status"],
                    text_metadata_path=str(e.sidecar),
                    **color,
                    **evidence,
                )
                validate_payload_evidence(payload)
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                store.write_bytes(str(target), buffer.getvalue())
                e.text["camera_color_metadata"][camera] = color
                record_evidence(e.text, camera, evidence)
                atomic_json(e.sidecar, e.text)
                report["wrote"] += 1
                output.update(
                    status="wrote",
                    bytes=target.stat().st_size,
                    latent_shape=list(payload["latent"].shape),
                    latent_num_frames=payload["latent_num_frames"],
                    **color,
                    **evidence,
                )
                if report["wrote"] % 25 == 0:
                    print(
                        f"encoded {report['wrote']}/{expected}: {e.repo} {camera}",
                        flush=True,
                    )
            except Exception as error:
                report["failed"] += 1
                output.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(f"FAILED {e.repo} {camera}: {error}", file=sys.stderr, flush=True)
        if (report["wrote"] + report["skipped"] + report["failed"]) % 25 == 0:
            atomic_json(args.report, report)
    complete = (
        report["failed"] == 0
        and report["metadata_failed"] == 0
        and report["wrote"] + report["skipped"] == expected
    )
    report.update(
        status="complete" if complete else "failed", seconds=time.monotonic() - started
    )
    atomic_json(args.report, report)
    print(
        f"wrote {report['wrote']}, skipped {report['skipped']}, failed {report['failed']} expected {expected}",
        flush=True,
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
