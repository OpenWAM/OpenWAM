from __future__ import annotations

from pathlib import Path

import torch

from open_wam.pipelines import (
    LingbotExactArtifactBundle,
    load_lingbot_exact_artifact_bundle,
    save_lingbot_exact_artifact_bundle,
)


def test_lingbot_exact_artifact_bundle_round_trip(tmp_path: Path) -> None:
    bundle = LingbotExactArtifactBundle(
        video_latents=torch.randn(2, 48, 2, 24, 20),
        action_history=torch.randn(2, 4, 30),
        task_text=("pick up block", None),
        text_context=torch.randn(2, 512, 16),
        metadata={"episode_id": "demo-0"},
    )
    path = tmp_path / "artifact_bundle.pt"
    save_lingbot_exact_artifact_bundle(path, bundle)
    loaded = load_lingbot_exact_artifact_bundle(path)

    assert torch.equal(loaded.video_latents, bundle.video_latents)
    assert torch.equal(loaded.action_history, bundle.action_history)
    assert loaded.task_text == bundle.task_text
    assert torch.equal(loaded.text_context, bundle.text_context)
    assert loaded.metadata == bundle.metadata
