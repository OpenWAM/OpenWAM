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
from open_wam.utils import load_experiment_config  # noqa: E402


def _summarize_cache(cache) -> dict[str, object]:
    return {
        "supported": cache.supported,
        "current_start_frame": cache.current_start_frame,
        "cached_frames": cache.cached_frames,
        "chunk_size": cache.chunk_size,
        "backend_name": cache.backend_name,
        "capability": cache.capability,
    }


def main() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/register_attached_robotwin_smoke.yaml")
    batch = build_synthetic_batch(config.data, batch_size=2)
    pipeline = build_variant_pipeline_from_config(config)
    train_batch = PolicyTrainBatch(actions=batch.actions, action_mask=batch.action_mask, state=batch.state)
    train_output = pipeline.forward_train(views=batch.views, batch=train_batch)
    print("train.policy_features", tuple(train_output.policy_output.policy_features.shape))
    print("train.action_pred", tuple(train_output.decoder_output.action_pred.shape))
    print("train.loss", float(train_output.decoder_output.loss.detach()))
    layout = train_output.policy_output.aux["layout"]
    print("train.num_image_blocks", layout.num_image_blocks)

    infer_output = pipeline.forward_infer_step(
        views=batch.views,
        context=PolicyInferContext(state=batch.state),
    )
    print("infer.action_pred", tuple(infer_output.decoder_output.action_pred.shape))
    print("infer.cache", _summarize_cache(infer_output.policy_output.next_state.cache))


if __name__ == "__main__":
    main()
