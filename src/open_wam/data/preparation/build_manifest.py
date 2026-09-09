"""Validate complete single-view tensors or multi view receipts into a latent manifest."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from open_wam.artifacts.files import sha256_file

from .frames import resample_indices, truncate_to_temporal_stride


def write_csv(path, rows):
    if not rows:
        raise ValueError("No validated tensors; refusing an empty manifest")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Use a new immutable manifest path")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(".pending.csv")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _validate_source_rgb(payload: dict, raw_source: dict) -> None:
    """Apply the declared adapter's RGB policy, independent of source labels."""
    source_type = raw_source.get("type")
    if source_type == "robomind_official_archive":
        from .encoding.robomind_rgb import validate_color_metadata
        from .encoding.robomind_rgb_evidence import validate_payload_evidence

        validate_color_metadata(
            payload, raw_source.get("source_uri") or raw_source["uri"]
        )
        validate_payload_evidence(payload)
    elif source_type == "robomind_failure_hdf5":
        if (
            payload.get("color_policy") != "robomind_failure_standard_jpeg_v1"
            or not re.fullmatch("[0-9a-f]{64}", payload.get("source_rgb_sha256", ""))
        ):
            raise ValueError("Failure tensor lacks standard-JPEG RGB evidence")
    elif source_type not in ("lerobot", "egoexo_aligned"):
        raise ValueError(f"Missing or unsupported raw_source.type: {source_type!r}")


def single_rows(args):
    import torch

    episodes = {}
    for line in Path(args.episodes).read_text().splitlines():
        row = json.loads(line)
        key = row["repo_id"], int(row["episode_index"])
        if key in episodes:
            raise ValueError("Duplicate episode index entry")
        if row["source_id"] != args.source:
            raise ValueError("Episode index mixes sources")
        episodes[key] = row
    for path in sorted(Path(args.latent_root).resolve().rglob("episode_*.pth")):
        match = re.fullmatch(r"episode_(\d+)_(\d+)_(\d+)", path.stem)
        if not match:
            raise ValueError("Unrecognized tensor filename: " + str(path))
        episode, start, end = map(int, match.groups())
        camera, repo = path.parent.name, path.parents[3].name
        meta = episodes[(repo, episode)]
        if camera not in meta["cameras"]:
            raise ValueError("Tensor camera absent from native metadata")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        _validate_source_rgb(payload, meta.get("raw_source", {}))
        z = payload["latent"]
        if z.ndim != 4 or z.shape[-1] != 48 or not torch.isfinite(z).all():
            raise ValueError("Expected finite THWC 48-channel latents: " + str(path))
        if payload.get("latent_layout") != "THWC" or not payload.get(
            "latents_normalized"
        ):
            raise ValueError(
                "Missing explicit THWC/normalization contract: " + str(path)
            )
        if z.dtype != torch.float16:
            raise ValueError("Expected fp16 stored latents")
        if start != 0 or end != int(meta["native_end_frames"][camera]):
            raise ValueError(
                "Single-view latent does not cover its declared native episode"
            )
        n = int(payload["video_num_frames"])
        if n != 1 + 4 * (z.shape[0] - 1) or n < 5:
            raise ValueError("Incomplete temporal VAE contract")
        source_fps, target_fps = float(payload["ori_fps"]), float(payload["fps"])
        if not all(
            math.isfinite(rate) and rate > 0 for rate in (source_fps, target_fps)
        ):
            raise ValueError("Source and encoded FPS must be finite and positive")
        if not math.isclose(source_fps, float(meta["raw_source"]["fps"]), rel_tol=1e-5):
            raise ValueError("Encoded source FPS disagrees with native metadata")
        # The two encoders retain either nearest source anchors (arrays) or
        # threshold-crossing anchors (streaming decode). Both must cover the
        # complete native timeline before discarding the final partial VAE group.
        counts = {len(resample_indices(end, source_fps, target_fps))}
        if target_fps < source_fps:
            counts.add(math.ceil(end * target_fps / source_fps))
        if n not in {truncate_to_temporal_stride(count, 4) for count in counts}:
            raise ValueError("Encoded frames do not cover the complete native timeline")
        if (int(payload["video_height"]), int(payload["video_width"])) != (
            16 * z.shape[1],
            16 * z.shape[2],
        ):
            raise ValueError("Spatial VAE contract mismatch")
        identifier = f"{repo}/{episode}/{camera}/{start}:{end}"
        yield dict(
            dataset_id=repo,
            repo_id=repo,
            episode_index=episode,
            source_id=args.source,
            clip_id=identifier,
            stream_key=identifier,
            target_slot="observation.images.slot0",
            latent_path=str(path),
            latent_key="latent",
            latent_layout="THWC",
            channels=48,
            height=z.shape[1],
            width=z.shape[2],
            length_frames=z.shape[0],
            latent_length_frames=z.shape[0],
            video_num_frames=n,
            observation_fps=float(payload["fps"]),
            task=meta["task"],
            text_provenance=meta["text_provenance"],
            text_status=meta["text_status"],
            physical_episode_key=meta["physical_episode_key"],
            augmentation="single_view",
            latent_bytes=path.stat().st_size,
            latent_sha256=sha256_file(path),
            native_start_frame=start,
            native_end_frame=end,
            native_fps=float(payload["ori_fps"]),
        )


def multi_view_rows(args):
    for path in sorted(Path(args.receipts).rglob("*.json")):
        receipt = json.loads(path.read_text())
        if (
            receipt.get("source_id") != args.source
            or receipt.get("status") != "complete"
        ):
            continue
        if not receipt.get("contract_sha256") or not receipt.get(
            "physical_episode_key"
        ):
            raise ValueError("Uncertified multi view receipt")
        if receipt.get("raw_source", {}).get("type") == "robomind_official_archive":
            raw_uri = receipt["raw_source"]["uri"]
            unit = hashlib.sha256(
                json.dumps(
                    raw_uri, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            unit_path = Path(args.receipts).parent / "archive_units" / (unit + ".json")
            if not unit_path.exists():
                continue
            archive = json.loads(unit_path.read_text())
            if (
                archive.get("status") != "complete"
                or not archive.get("gzip_eof_verified")
                or archive.get("contract_sha256") != receipt["contract_sha256"]
                or receipt["plan_id"] not in archive.get("plan_ids", [])
            ):
                continue
        for row in receipt["outputs"]:
            tensor = Path(row["latent_path"])
            if (
                tensor.stat().st_size != row["latent_bytes"]
                or sha256_file(tensor) != row["latent_sha256"]
            ):
                raise ValueError("Multi view tensor no longer matches completion receipt")
            row = dict(row)
            row.update(
                dataset_id=row["repo_id"],
                episode_index=receipt["episode_index"],
                source_id=args.source,
            )
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--episodes")
    parser.add_argument("--latent-root")
    parser.add_argument("--receipts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.receipts) == bool(args.latent_root) or (
        args.latent_root and not args.episodes
    ):
        parser.error("Choose --receipts OR --latent-root plus --episodes")
    rows = list(multi_view_rows(args) if args.receipts else single_rows(args))
    identities = [row["clip_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate clip identity")
    write_csv(args.out, rows)
    print(
        json.dumps(
            dict(
                clips=len(rows),
                physical_episodes=len({r["physical_episode_key"] for r in rows}),
                encoded_view_hours=sum(
                    r["video_num_frames"] / r["observation_fps"] for r in rows
                )
                / 3600,
                manifest_sha256=sha256_file(args.out),
            )
        )
    )


if __name__ == "__main__":
    main()
