"""Check a latent tree against the contract the training loader relies on.

Every assertion here is something a silently wrong encode would violate: a shape
that disagrees with its own metadata, a temporal count the causal VAE could not
have produced, a spatial grid that is not a 16x reduction of its canvas.
"""

import argparse
import collections
import glob
import os

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Latent tree to check")
    parser.add_argument(
        "--limit", type=int, default=None, help="Check only the first N files"
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the summary")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "*.pth"), recursive=True))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"No .pth files under {args.root}")
        return 1

    bins: collections.Counter = collections.Counter()
    grids: collections.Counter = collections.Counter()
    problems = []

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        latent = payload["latent"]
        name = os.path.relpath(path, args.root)

        def check(condition, message):
            if not condition:
                problems.append(f"{name}: {message}")

        check(
            tuple(latent.shape[:3])
            == (
                payload["latent_num_frames"],
                payload["latent_height"],
                payload["latent_width"],
            ),
            f"shape {tuple(latent.shape)} disagrees with its own metadata",
        )
        check(
            payload["latent_num_frames"] == 1 + (payload["video_num_frames"] - 1) // 4,
            "temporal count is not 1 + (frames - 1) / 4",
        )
        check(
            payload["latent_height"] * 16 == payload["video_height"],
            "height is not a 16x reduction",
        )
        check(
            payload["latent_width"] * 16 == payload["video_width"],
            "width is not a 16x reduction",
        )
        check(
            len(payload["frame_ids"]) == payload["video_num_frames"],
            "frame_ids count disagrees",
        )
        check(
            max(payload["frame_ids"]) < payload["end_frame"],
            "frame_ids run past the source timeline",
        )
        check(latent.float().isfinite().all().item(), "latent holds NaN or Inf")

        bins[payload.get("resize_bin", "multi_view")] += 1
        grids[f"{payload['latent_height']}x{payload['latent_width']}"] += 1

        if not args.quiet and len(paths) <= 20:
            print(
                f"  {name[:64]:<64} {str(tuple(latent.shape)):<20} "
                f"{payload.get('resize_bin', 'multi_view'):<20} "
                f"{payload['end_frame']}@{payload.get('ori_fps', payload['fps']):.0f} -> "
                f"{payload['video_num_frames']}@{payload['fps']:.0f}"
            )

    print(f"\n{len(paths)} files checked")
    for name, count in bins.most_common():
        print(f"  bin {name:<24} {count}")
    for name, count in grids.most_common():
        print(f"  grid {name:<23} {count}")

    if problems:
        print(f"\n{len(problems)} problems:")
        for problem in problems[:20]:
            print(f"  {problem}")
        return 1
    print("\nAll structural assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
