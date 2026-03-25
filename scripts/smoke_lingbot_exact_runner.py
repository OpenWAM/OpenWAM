from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from open_wam.data import build_synthetic_batch  # noqa: E402
from open_wam.pipelines import build_lingbot_exact_runner_from_config  # noqa: E402
from open_wam.utils.config_loader import load_experiment_config  # noqa: E402


def main() -> None:
    config = load_experiment_config("configs/experiments/parallel_stream_robotwin_lingbot_replica.yaml")
    batch = build_synthetic_batch(config.data, batch_size=2)
    runner = build_lingbot_exact_runner_from_config(config)

    session = runner.reset(task_text=batch.task_text)
    warmup = runner.warmup_cache(
        session=session,
        views=batch.views,
        action_history=batch.actions,
        action_space="model",
    )
    chunk = runner.infer_chunk(session=warmup.session)

    print("warmup.frame_start", warmup.session.policy_state.cache["frame_start"])
    print("chunk.model_action_pred", tuple(chunk.chunk_action_pred.shape))
    print("chunk.aligned_action_pred", tuple(chunk.decoder_output.action_pred.shape))
    print("chunk.predicted_latents", tuple(chunk.predicted_latents.shape))
    print("chunk.step_index", chunk.session.policy_state.step_index)


if __name__ == "__main__":
    main()
