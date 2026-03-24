from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import build_canonical_video_preprocessor, build_synthetic_batch
from open_wam.models.action_heads import ActionHeadInferContext, ContractOnlyActionHead, ContractOnlyActionHeadConfig
from open_wam.pipelines import UnifiedWAMPipeline
from open_wam.utils import load_experiment_config


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
        preprocessor=build_canonical_video_preprocessor(config.data),
    )

    batch = build_synthetic_batch(config.data, batch_size=2)
    infer_context = ActionHeadInferContext(
        state=batch.state,
    )
    output = pipeline.forward_infer_step(views=batch.views, context=infer_context)
    print("eval.video_tokens", tuple(output.backbone_output.video_tokens.shape))
    print("eval.action_pred", tuple(output.head_output.action_pred.shape))
    print("eval.step_index", output.head_output.next_state.step_index)


if __name__ == "__main__":
    main()
