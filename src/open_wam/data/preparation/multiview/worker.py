"""Encode synchronized RGB multi view videos into additional VPM latent samples."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from open_wam.artifacts.files import atomic_json, sha256_file
from open_wam.data.preparation.multiview.cropping import open_cursors
from open_wam.data.preparation.multiview.layout import POLICY, make_layout, render_frame
from open_wam.data.preparation.multiview.prepare import digest
from open_wam.data.preparation.multiview.sources import MetadataStore

ENCODING_ROOT = Path(__file__).resolve().parents[1] / "encoding"
from open_wam.data.preparation import frames
from open_wam.models.visual_tower import vae_encoding
from open_wam.models.visual_tower.vae_encoding import encode_clip, load_vae


def write_video(path, frames, fps=15):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for rgb in frames:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class Encoder:
    def __init__(self, args):
        self.args = args
        self.vae = None
        self.out = Path(args.out_root).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.store = MetadataStore(self.out / "metadata_cache")
        self.impl = {
            p.name: sha256_file(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("sources.py"),
                Path(__file__).with_name("layout.py"),
                Path(__file__).with_name("cropping.py"),
                Path(__file__).with_name("inner_roi.py"),
            ]
        }
        for namespace, path in [
            ("vae_encoding", vae_encoding.__file__),
            ("frames", frames.__file__),
            ("robomind_rgb", ENCODING_ROOT / "robomind_rgb.py"),
            ("robomind_metadata", ENCODING_ROOT / "robomind_metadata.py"),
        ]:
            self.impl[namespace] = sha256_file(Path(path))
        import diffusers

        self.contract = {
            "policy": POLICY,
            "implementation": self.impl,
            "vae_config_sha256": sha256_file(Path(args.vae) / "config.json"),
            "vae_weights": {
                p.name: sha256_file(p)
                for p in sorted(Path(args.vae).glob("*.safetensors"))
            },
            "normalized": True,
            "dtype": "fp16",
            "encode_dtype": "bf16",
            "spatial_stride": 16,
            "temporal_stride": 4,
            "runtime_versions": {
                "torch": str(torch.__version__),
                "diffusers": str(diffusers.__version__),
            },
        }
        self.contract_sha = digest(self.contract)

    def complete_path(self, plan):
        return self.out / "receipts" / plan["source_id"] / (plan["id"] + ".json")

    def is_complete(self, plan):
        path = self.complete_path(plan)
        if not path.exists():
            return False
        saved = json.loads(path.read_text())
        if (
            saved["contract_sha256"] != self.contract_sha
            or saved["plan_id"] != plan["id"]
        ):
            raise ValueError(
                "Existing output has a different immutable encoding contract"
            )
        expected = "smoke_complete" if self.args.smoke else "complete"
        if saved["status"] == "excluded_too_short":
            return True
        if saved["status"] != expected:
            return False
        return all(
            Path(r["latent_path"]).is_file()
            and Path(r["latent_path"]).stat().st_size == r["latent_bytes"]
            for r in saved["outputs"]
        )

    def model(self):
        if self.vae is None:
            if not torch.cuda.is_available():
                raise RuntimeError("A scheduled GPU is required")
            torch.set_num_threads(4)
            self.vae = load_vae(self.args.vae, torch.device("cuda:0"), torch.bfloat16)
        return self.vae

    def process(self, plan, hdf_path=None):
        lock_path = self.out / "locks" / plan["id"][:2] / (plan["id"] + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._process_locked(plan, hdf_path)

    def _process_locked(self, plan, hdf_path=None):
        if self.is_complete(plan):
            return (
                "excluded"
                if json.loads(self.complete_path(plan).read_text())["status"]
                == "excluded_too_short"
                else "skipped"
            )
        started = time.time()
        cursors, owned = open_cursors(plan, self.store, hdf_path)
        try:
            geometry = [
                (c, cursors[c].width, cursors[c].height) for c in plan["cameras"]
            ]
            layout = make_layout(
                geometry,
                pair_orientation=plan.get("pair_orientation", "auto"),
                camera_rotations=plan.get("camera_rotations_degrees"),
            )
            duration = min(c.duration for c in cursors.values())
            fps = plan["target_fps"]
            total = 1 + math.floor(duration * fps + 1e-6)
            full_frames = total
            if self.args.smoke:
                total = min(total, 65)
            total = 1 + ((total - 1) // 4) * 4
            if total < 5:
                atomic_json(
                    self.complete_path(plan),
                    {
                        "status": "excluded_too_short",
                        "plan_id": plan["id"],
                        "contract_sha256": self.contract_sha,
                        "source_id": plan["source_id"],
                        "repo_id": plan["repo_id"],
                        "episode_index": plan["episode_index"],
                        "raw_source": plan["raw_source"],
                        "outputs": [],
                        "source_timeline_frames": full_frames,
                        "reason": "Fewer than five frames at 15 FPS cannot form observed and future latent frames",
                    },
                )
                return "excluded"
            self.active_source_objects = {}
            for camera, cursor in cursors.items():
                reader = getattr(cursor, "reader", None)
                obj = {"source_fps": cursor.fps}
                if hasattr(cursor, "crop_spec"):
                    obj["native_crop"] = cursor.crop_spec
                if reader is not None:
                    obj.update(
                        uri=reader.uri,
                        etag=reader.etag,
                        object_bytes=reader.object_size,
                        member_offset=reader.offset,
                        member_bytes=reader.length,
                        clip_start=getattr(cursor, "start", None),
                        clip_stop=getattr(cursor, "stop", None),
                    )
                else:
                    obj.update(
                        uri=plan["raw_source"]["uri"],
                        hdf_camera=camera,
                        member=plan["raw_source"].get("member"),
                    )
                    remote = next((x for x in owned if hasattr(x, "etag")), None)
                    if remote is not None:
                        obj.update(etag=remote.etag, object_bytes=remote.object_size)
                self.active_source_objects[camera] = obj
            output = []
            pending = []
            chunk_start = 0
            # Non-overlapping clips preserve the source timeline. Each clip is
            # 1+4k frames; residual 0-3 frames are explicitly recorded below.
            for index in range(total):
                frames = {c: cursor.at(index / fps) for c, cursor in cursors.items()}
                pending.append(render_frame(frames, layout))
                if len(pending) == plan["max_clip_frames"] or index == total - 1:
                    keep = 1 + ((len(pending) - 1) // 4) * 4
                    if keep >= 5:
                        rgb = np.stack(pending[:keep])
                        result = self.encode_chunk(plan, layout, rgb, chunk_start)
                        output.append(result)
                    pending = []
                    chunk_start = index + 1
            if not output:
                raise ValueError("No usable composite clips")
            receipt = {
                "status": "smoke_complete" if self.args.smoke else "complete",
                "plan_id": plan["id"],
                "contract_sha256": self.contract_sha,
                "contract": self.contract,
                "source_id": plan["source_id"],
                "repo_id": plan["repo_id"],
                "episode_index": plan["episode_index"],
                "raw_source": plan["raw_source"],
                "physical_episode_key": plan["physical_episode_key"],
                "single_view_paths": plan["single_view_paths"],
                "source_objects": self.active_source_objects,
                "layout": layout.to_dict(),
                "source_timeline_frames": full_frames,
                "requested_frames": total,
                "encoded_frames": sum(r["video_num_frames"] for r in output),
                "max_sampling_time_error_seconds": {
                    c: x.max_time_error for c, x in cursors.items()
                },
                "outputs": output,
                "finished_at_unix": time.time(),
                "seconds": time.time() - started,
            }
            atomic_json(self.complete_path(plan), receipt)
            print(
                json.dumps(
                    {
                        "event": "encoded",
                        "source": plan["source_id"],
                        "plan": plan["id"],
                        "clips": len(output),
                        "shape": [layout.height, layout.width],
                        "seconds": receipt["seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return "encoded"
        finally:
            for item in list(cursors.values()) + list(reversed(owned)):
                try:
                    item.close()
                except Exception:
                    pass

    def encode_chunk(self, plan, layout, rgb, start):
        vae = self.model()
        video = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
        latent = encode_clip(vae, video, normalize=True).cpu().to(torch.float16)
        expected = (
            1 + (len(rgb) - 1) // 4,
            layout.height // 16,
            layout.width // 16,
            48,
        )
        if tuple(latent.shape) != expected or not torch.isfinite(latent).all():
            raise ValueError(
                f"Invalid encoded latent {tuple(latent.shape)} expected {expected}"
            )
        clip_id = f"multi_view/{plan['id']}/{start:08d}"
        path = (
            self.out
            / "latents"
            / plan["source_id"]
            / plan["id"]
            / f"clip_{start:08d}_{start + len(rgb):08d}.pth"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "stream_key": clip_id,
            "clip_id": clip_id,
            "dataset_id": plan["source_id"],
            "repo_id": plan["repo_id"],
            "source_group": "pretrain",
            "target_slot": "observation.images.slot0",
            "latent_path": str(path),
            "latent_key": "latent",
            "length_frames": int(latent.shape[0]),
            "latent_length_frames": int(latent.shape[0]),
            "latent_layout": "THWC",
            "height": int(latent.shape[1]),
            "width": int(latent.shape[2]),
            "channels": 48,
            "observation_fps": 15.0,
            "task": plan["task"],
            "text_provenance": plan["text_provenance"],
            "text_status": plan["text_status"],
            "augmentation": "multi_view",
            "physical_episode_key": plan["physical_episode_key"],
            "source_episode_index": plan["episode_index"],
            "view_count": len(plan["cameras"]),
            "view_names": json.dumps(plan["cameras"]),
            "rgb_layout": json.dumps(layout.to_dict()),
            "source_timeline_start_seconds": start / 15.0,
            "source_timeline_end_seconds": (start + len(rgb) - 1) / 15.0,
        }
        payload = {
            "latent": latent,
            "latent_num_frames": expected[0],
            "latent_height": expected[1],
            "latent_width": expected[2],
            "latent_layout": "THWC",
            "latents_normalized": True,
            "fps": 15.0,
            "video_num_frames": len(rgb),
            "video_width": layout.width,
            "video_height": layout.height,
            "start_frame": start,
            "end_frame": start + len(rgb),
            "frame_ids": list(range(start, start + len(rgb))),
            "frame_id_timebase_fps": 15.0,
            "task": plan["task"],
            "text_status": plan["text_status"],
            "text_provenance": plan["text_provenance"],
            "layout": layout.to_dict(),
            "plan_id": plan["id"],
            "physical_episode_key": plan["physical_episode_key"],
            "source_cameras": plan["cameras"],
            "single_view_paths": plan["single_view_paths"],
            "raw_source": plan["raw_source"],
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "source_objects": self.active_source_objects,
            "encoding_contract": self.contract,
            "encoding_contract_sha256": self.contract_sha,
        }
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(payload, tmp)
        # Verify the persisted payload before the completion receipt is visible.
        verified = torch.load(tmp, map_location="cpu", weights_only=True)
        if not torch.equal(verified["latent"], latent):
            raise IOError("Saved latent verification failed")
        tmp.replace(path)
        row.update(
            latent_bytes=path.stat().st_size,
            latent_sha256=sha256_file(path),
            video_num_frames=len(rgb),
        )
        if self.args.smoke:
            self.preview(plan, layout, rgb, latent, start)
        return row

    @torch.no_grad()
    def preview(self, plan, layout, rgb, latent, start):
        from PIL import Image

        directory = self.out / "previews" / plan["source_id"]
        directory.mkdir(parents=True, exist_ok=True)
        name = plan["id"][:12] + f"-{start}"
        write_video(directory / (name + "-rgb.mp4"), rgb)
        Image.fromarray(rgb[len(rgb) // 2]).save(directory / (name + "-rgb.png"))
        vae = self.model()
        device = next(vae.parameters()).device
        z = latent.float().permute(3, 0, 1, 2).unsqueeze(0).to(device)
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        decoded = vae.decode((z * std + mean).to(torch.bfloat16)).sample
        recon = (
            ((decoded[0].float().permute(1, 2, 3, 0).cpu() + 1) / 2)
            .clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .numpy()
        )
        write_video(directory / (name + "-vae.mp4"), recon)
        Image.fromarray(recon[len(recon) // 2]).save(directory / (name + "-vae.png"))
        atomic_json(
            directory / (name + ".json"),
            {
                "plan": plan,
                "layout": layout.to_dict(),
                "latent_shape": list(latent.shape),
                "rgb_video": name + "-rgb.mp4",
                "vae_video": name + "-vae.mp4",
                "reconstruction_mse": float(
                    np.mean(
                        (recon.astype(np.float32) / 255 - rgb.astype(np.float32) / 255)
                        ** 2
                    )
                ),
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--source")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-plans", type=int)
    parser.add_argument("--max-seconds", type=float, default=82800)
    args = parser.parse_args()
    shard, count = map(int, args.shard.split("/"))
    if not 0 <= shard < count:
        parser.error("Invalid shard")
    encoder = Encoder(args)
    index = json.loads(Path(args.index).read_text())
    encoded = failed = skipped = excluded = 0
    seen_sources = set()
    started = time.monotonic()
    timed_out = False
    # CPU archive index preparation can overlap useful GPU work on all other
    # sources. Every AgiBot archive is still included in the same fixed shard.
    entries = sorted(
        index["plans"],
        key=lambda e: (e["raw_type"] == "agibot_tar", e["source_id"], e["repo_id"]),
    )
    for entry in entries:
        if args.source and entry["source_id"] != args.source:
            continue
        if (
            int(hashlib.sha256(entry["repo_id"].encode()).hexdigest(), 16) % count
            != shard
        ):
            continue
        if args.smoke and entry["source_id"] in seen_sources:
            continue
        # Official compressed archives are processed by archive_worker.py so
        # a large archive is fetched once for all its constituent episodes.
        if entry["raw_type"] == "robomind_official_archive":
            continue
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("Plan changed after publication")
        plans = json.loads(path.read_text())["plans"]
        for plan in plans:
            if time.monotonic() - started > args.max_seconds:
                timed_out = True
                break
            if args.smoke and plan["source_id"] in seen_sources:
                break
            try:
                result = encoder.process(plan)
                if result == "encoded":
                    encoded += 1
                elif result == "excluded":
                    excluded += 1
                    continue
                else:
                    skipped += 1
                seen_sources.add(plan["source_id"])
            except Exception as error:
                failed += 1
                failure = {
                    "plan_id": plan["id"],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "time": time.time(),
                    "source": plan["source_id"],
                    "plan": plan,
                }
                atomic_json(
                    Path(args.out_root)
                    / "failures"
                    / plan["source_id"]
                    / (plan["id"] + ".json"),
                    failure,
                )
                print(
                    json.dumps(
                        {
                            "event": "failed",
                            "source": plan["source_id"],
                            "plan": plan["id"],
                            "error": failure["error"],
                        }
                    ),
                    flush=True,
                )
                if args.smoke:
                    break
                if failed >= 20:
                    raise RuntimeError(
                        "Stopping after 20 failures; inspect recorded source errors before retrying"
                    )
            if args.max_plans and encoded + skipped >= args.max_plans:
                break
        if args.max_plans and encoded + skipped >= args.max_plans:
            break
        if timed_out:
            break
    summary = {
        "encoded": encoded,
        "skipped": skipped,
        "excluded_too_short": excluded,
        "failed": failed,
        "seen_sources": sorted(seen_sources),
        "smoke": args.smoke,
        "contract_sha256": encoder.contract_sha,
        "time_budget_exhausted": timed_out,
    }
    atomic_json(Path(args.out_root) / f"summary-{shard}-of-{count}.json", summary)
    print(json.dumps(summary), flush=True)
    return 1 if failed else 75 if timed_out else 0


if __name__ == "__main__":
    raise SystemExit(main())
