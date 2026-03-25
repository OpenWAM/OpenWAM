from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.configs import ActionSchemaConfig, PostDecodedPolicyConfig, RobotWinDataConfig  # noqa: E402
from open_wam.configs.inference import InferenceConfig  # noqa: E402
from open_wam.configs.training import TrainingConfig  # noqa: E402
from open_wam.data import build_canonical_video_preprocessor, build_synthetic_batch  # noqa: E402
from open_wam.models.action_decoders import DecodedFeatureActionDecoder  # noqa: E402
from open_wam.models.policy_variants import PolicyInferContext, PolicyTrainBatch, PostDecodedPolicyVariant  # noqa: E402
from open_wam.models.video_backbone import LingbotCompatibleVideoBackboneConfig  # noqa: E402
from open_wam.models.visual_tower import VisualTower  # noqa: E402
from open_wam.pipelines import VariantPipeline  # noqa: E402


def main() -> None:
    batch_size = 2
    data_config = RobotWinDataConfig(
        num_frames=4,
        action_schema=ActionSchemaConfig(action_dim=30, action_horizon=6, state_dim=30, state_horizon=1),
    )
    batch = build_synthetic_batch(data_config, batch_size=batch_size)
    pipeline = VariantPipeline(
        visual_tower=VisualTower(LingbotCompatibleVideoBackboneConfig(hidden_size=256, num_layers=1, num_heads=8)),
        policy_variant=PostDecodedPolicyVariant(
            config=PostDecodedPolicyConfig(hidden_size=256),
            training_config=TrainingConfig(),
            inference_config=InferenceConfig(),
            action_horizon=data_config.action_schema.action_horizon,
            state_dim=data_config.action_schema.state_dim,
        ),
        action_decoder=DecodedFeatureActionDecoder(hidden_size=256, action_dim=30, action_horizon=6),
        preprocessor=build_canonical_video_preprocessor(data_config),
    )
    train_batch = PolicyTrainBatch(actions=batch.actions, action_mask=batch.action_mask, state=batch.state)
    train_output = pipeline.forward_train(views=batch.views, batch=train_batch)
    print("train.decoded_features", tuple(train_output.visual_outputs.decode.decoded_features.shape))  # type: ignore[union-attr]
    print("train.action_pred", tuple(train_output.decoder_output.action_pred.shape))
    print("train.loss", float(train_output.decoder_output.loss))

    infer_output = pipeline.forward_infer_step(views=batch.views, context=PolicyInferContext(state=batch.state))
    print("infer.action_pred", tuple(infer_output.decoder_output.action_pred.shape))
    print("infer.step_index", infer_output.policy_output.next_state.step_index)


if __name__ == "__main__":
    main()
