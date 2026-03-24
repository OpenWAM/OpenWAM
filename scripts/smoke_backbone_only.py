from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import ActionSchemaConfig, RobotWinDataConfig
from open_wam.data import build_canonical_video_preprocessor, build_synthetic_views
from open_wam.models.video_backbone import LingbotCompatibleVideoBackboneConfig
from open_wam.pipelines import BackboneOnlyPipeline


def main() -> None:
    batch_size = 2
    data_config = RobotWinDataConfig(
        num_frames=4,
        action_schema=ActionSchemaConfig(
            action_dim=30,
            action_horizon=6,
            state_dim=30,
            state_horizon=1,
        ),
    )
    views = build_synthetic_views(data_config, batch_size=batch_size)

    # Use a smaller hidden size here so the smoke test stays lightweight while
    # exercising the same RGB -> latent -> token geometry contract.
    pipeline = BackboneOnlyPipeline(
        backbone_config=LingbotCompatibleVideoBackboneConfig(
            hidden_size=256,
            num_layers=1,
            num_heads=8,
        ),
        preprocessor=build_canonical_video_preprocessor(data_config),
    )
    output = pipeline(views)

    print("canonical_video", tuple(output.canonical_video.shape))
    print("video_latents", tuple(output.video_latents.shape))
    print("video_tokens", tuple(output.video_tokens.shape))
    print("token_grid", output.token_grid)


if __name__ == "__main__":
    main()
