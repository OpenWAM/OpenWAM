from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import build_synthetic_batch  # noqa: E402
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch  # noqa: E402
from open_wam.pipelines import build_variant_pipeline_from_config  # noqa: E402
from open_wam.utils.config_loader import load_experiment_config  # noqa: E402


def main() -> None:
    config = load_experiment_config("configs/experiments/parallel_stream_robotwin_lingbot_replica.yaml")
    batch = build_synthetic_batch(config.data, batch_size=2)
    pipeline = build_variant_pipeline_from_config(config)

    train_output = pipeline.forward_train(
        views=batch.views,
        batch=PolicyTrainBatch(actions=batch.actions, action_mask=batch.action_mask, state=batch.state),
    )
    print("train.backbone_impl", config.backbone.implementation)
    print("train.policy_features", tuple(train_output.policy_output.policy_features.shape))
    print("train.action_pred", tuple(train_output.decoder_output.action_pred.shape))
    print("train.loss", float(train_output.decoder_output.loss))

    infer_output = pipeline.forward_infer_step(views=batch.views, context=PolicyInferContext(state=batch.state))
    print("infer.action_pred", tuple(infer_output.decoder_output.action_pred.shape))
    print("infer.step_index", infer_output.policy_output.next_state.step_index)


if __name__ == "__main__":
    main()
