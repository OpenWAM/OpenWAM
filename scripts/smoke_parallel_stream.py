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
<<<<<<< feat/exact-visualization-config-alignment
=======


def _summarize_cache(cache: dict[str, object]) -> dict[str, object]:
    backbone_cache = cache.get("backbone_cache")
    backbone_summary = None
    if backbone_cache is not None:
        backbone_summary = {
            "supported": backbone_cache.supported,
            "current_start_frame": backbone_cache.current_start_frame,
            "cached_frames": backbone_cache.cached_frames,
            "chunk_size": backbone_cache.chunk_size,
            "backend_name": backbone_cache.backend_name,
            "capability": backbone_cache.capability,
        }
    return {
        "runtime_mode": cache.get("runtime_mode"),
        "cache_name": cache.get("cache_name"),
        "cache_initialized": cache.get("cache_initialized"),
        "frame_start": cache.get("frame_start"),
        "step_index": cache.get("step_index"),
        "backbone_cache": backbone_summary,
    }
>>>>>>> main


def main() -> None:
    config = load_experiment_config(REPO_ROOT / "configs/experiments/parallel_stream_robotwin_smoke.yaml")
    batch = build_synthetic_batch(config.data, batch_size=2)
    pipeline = build_variant_pipeline_from_config(config)

    train_batch = PolicyTrainBatch(actions=batch.actions, action_mask=batch.action_mask, state=batch.state)
    train_output = pipeline.forward_train(views=batch.views, batch=train_batch)
    print("train.policy_features", tuple(train_output.policy_output.policy_features.shape))
    print("train.action_pred", tuple(train_output.decoder_output.action_pred.shape))
    print("train.loss", float(train_output.decoder_output.loss.detach()))
    aux_keys = sorted(train_output.policy_output.aux.keys())
    print("train.aux_keys", aux_keys)
    layout = train_output.policy_output.aux.get("layout")
    if layout is not None:
        print("train.packed_sequence_length", int(layout.frame_ids.numel()))

    infer_output = pipeline.forward_infer_step(views=batch.views, context=PolicyInferContext(state=batch.state))
    print("infer.action_pred", tuple(infer_output.decoder_output.action_pred.shape))
    print("infer.cache", _summarize_cache(infer_output.policy_output.next_state.cache))


if __name__ == "__main__":
    main()
