from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.models.video_backbone import LingbotCompatibleVideoBackboneConfig
from open_wam.pipelines import BackboneOnlyPipeline


def main() -> None:
    batch_size = 2
    num_frames = 4

    views = {
        "cam_high": torch.randint(0, 255, (batch_size, num_frames, 300, 400, 3), dtype=torch.uint8),
        "cam_left_wrist": torch.randint(0, 255, (batch_size, num_frames, 160, 200, 3), dtype=torch.uint8),
        "cam_right_wrist": torch.randint(0, 255, (batch_size, num_frames, 160, 200, 3), dtype=torch.uint8),
    }

    # Use a smaller hidden size here so the smoke test stays lightweight while
    # exercising the same RGB -> latent -> token geometry contract.
    pipeline = BackboneOnlyPipeline(
        backbone_config=LingbotCompatibleVideoBackboneConfig(
            hidden_size=256,
            num_layers=1,
            num_heads=8,
        )
    )
    output = pipeline(views)

    print("canonical_video", tuple(output.canonical_video.shape))
    print("video_latents", tuple(output.video_latents.shape))
    print("video_tokens", tuple(output.video_tokens.shape))
    print("token_grid", output.token_grid)


if __name__ == "__main__":
    main()
