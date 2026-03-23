from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.models.action_heads import (  # noqa: E402
    ActionHeadInferContext,
    ActionHeadTrainingBatch,
    ContractOnlyActionHead,
    ContractOnlyActionHeadConfig,
)
from open_wam.models.video_backbone import LingbotCompatibleVideoBackboneConfig  # noqa: E402
from open_wam.pipelines import UnifiedWAMPipeline  # noqa: E402


def build_views(batch_size: int, num_frames: int) -> dict[str, torch.Tensor]:
    return {
        "cam_high": torch.randint(0, 255, (batch_size, num_frames, 300, 400, 3), dtype=torch.uint8),
        "cam_left_wrist": torch.randint(0, 255, (batch_size, num_frames, 160, 200, 3), dtype=torch.uint8),
        "cam_right_wrist": torch.randint(0, 255, (batch_size, num_frames, 160, 200, 3), dtype=torch.uint8),
    }


def main() -> None:
    batch_size = 2
    num_frames = 4
    action_horizon = 6
    action_dim = 30
    state_dim = 30

    pipeline = UnifiedWAMPipeline(
        action_head=ContractOnlyActionHead(
            ContractOnlyActionHeadConfig(
                action_dim=action_dim,
                action_horizon=action_horizon,
                state_dim=state_dim,
                hidden_size=256,
            )
        ),
        backbone_config=LingbotCompatibleVideoBackboneConfig(
            hidden_size=256,
            num_layers=1,
            num_heads=8,
        ),
    )

    views = build_views(batch_size=batch_size, num_frames=num_frames)
    train_batch = ActionHeadTrainingBatch(
        actions=torch.randn(batch_size, action_horizon, action_dim),
        action_mask=torch.ones(batch_size, action_horizon, action_dim),
        state=torch.randn(batch_size, 1, state_dim),
    )
    train_output = pipeline.forward_train(views=views, batch=train_batch)
    print("train.video_tokens", tuple(train_output.backbone_output.video_tokens.shape))
    print("train.action_pred", tuple(train_output.head_output.action_pred.shape))
    print("train.loss", float(train_output.head_output.loss))

    infer_context = ActionHeadInferContext(state=torch.randn(batch_size, 1, state_dim))
    infer_output = pipeline.forward_infer_step(views=views, context=infer_context)
    print("infer.action_pred", tuple(infer_output.head_output.action_pred.shape))
    print("infer.step_index", infer_output.head_output.next_state.step_index)


if __name__ == "__main__":
    main()
