from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import ActionSchemaConfig, RobotWinDataConfig  # noqa: E402
from open_wam.data import build_canonical_video_preprocessor, build_synthetic_batch  # noqa: E402
from open_wam.models.action_heads import (  # noqa: E402
    ActionHeadInferContext,
    ActionHeadTrainingBatch,
    ContractOnlyActionHead,
    ContractOnlyActionHeadConfig,
)
from open_wam.models.video_backbone import LingbotCompatibleVideoBackboneConfig  # noqa: E402
from open_wam.pipelines import UnifiedWAMPipeline  # noqa: E402


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
    action_horizon = data_config.action_schema.action_horizon
    action_dim = data_config.action_schema.action_dim
    state_dim = data_config.action_schema.state_dim
    batch = build_synthetic_batch(data_config, batch_size=batch_size)

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
        preprocessor=build_canonical_video_preprocessor(data_config),
    )

    train_batch = ActionHeadTrainingBatch(
        actions=batch.actions,
        action_mask=batch.action_mask,
        state=batch.state,
    )
    train_output = pipeline.forward_train(views=batch.views, batch=train_batch)
    print("train.video_tokens", tuple(train_output.backbone_output.video_tokens.shape))
    print("train.action_pred", tuple(train_output.head_output.action_pred.shape))
    print("train.loss", float(train_output.head_output.loss))

    infer_context = ActionHeadInferContext(state=batch.state)
    infer_output = pipeline.forward_infer_step(views=batch.views, context=infer_context)
    print("infer.action_pred", tuple(infer_output.head_output.action_pred.shape))
    print("infer.step_index", infer_output.head_output.next_state.step_index)


if __name__ == "__main__":
    main()
