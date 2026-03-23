from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.models.action_heads import ActionHeadInferContext, ContractOnlyActionHead, ContractOnlyActionHeadConfig
from open_wam.pipelines import UnifiedWAMPipeline
from open_wam.utils import load_experiment_config


def _build_views(batch_size: int, num_frames: int) -> dict[str, torch.Tensor]:
    return {
        "cam_high": torch.randint(0, 255, (batch_size, num_frames, 300, 400, 3), dtype=torch.uint8),
        "cam_left_wrist": torch.randint(0, 255, (batch_size, num_frames, 160, 200, 3), dtype=torch.uint8),
        "cam_right_wrist": torch.randint(0, 255, (batch_size, num_frames, 160, 200, 3), dtype=torch.uint8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "--config", dest="config", type=str, required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.action_head.name != "contract_only":
        raise ValueError(
            f"Phase-2 eval currently supports only 'contract_only', got '{config.action_head.name}'."
        )

    pipeline = UnifiedWAMPipeline(
        action_head=ContractOnlyActionHead(
            ContractOnlyActionHeadConfig(
                action_dim=config.action_head.action_dim,
                action_horizon=config.action_head.action_horizon,
                state_dim=config.action_head.state_dim,
                hidden_size=config.action_head.hidden_size,
            )
        ),
        backbone_config=config.backbone,
    )

    views = _build_views(batch_size=2, num_frames=config.data.num_frames)
    infer_context = ActionHeadInferContext(
        state=torch.randn(2, config.data.action_schema.state_horizon, config.data.action_schema.state_dim)
    )
    output = pipeline.forward_infer_step(views=views, context=infer_context)
    print("eval.video_tokens", tuple(output.backbone_output.video_tokens.shape))
    print("eval.action_pred", tuple(output.head_output.action_pred.shape))
    print("eval.step_index", output.head_output.next_state.step_index)


if __name__ == "__main__":
    main()

