from __future__ import annotations

import torch

from open_wam.configs import InferenceConfig, TrainingConfig
from open_wam.models.common import (
    build_action_flow_match_inference_scheduler,
    build_action_flow_match_train_artifacts,
)


def test_action_flow_match_artifacts_stay_on_input_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions = torch.randn(2, 6, 7, device=device)
    action_mask = torch.ones_like(actions)

    artifacts = build_action_flow_match_train_artifacts(
        actions,
        action_mask,
        training_config=TrainingConfig(),
    )

    assert artifacts.timesteps.device == device
    assert artifacts.noisy_actions.device == device
    assert artifacts.targets.device == device


def test_action_flow_match_scheduler_step_preserves_sample_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = build_action_flow_match_inference_scheduler(
        training_config=TrainingConfig(),
        inference_config=InferenceConfig(action_num_inference_steps=4),
    )
    sample = torch.randn(2, 6, 7, device=device)
    model_output = torch.randn_like(sample)

    next_sample = scheduler.step(model_output, scheduler.timesteps[0], sample)

    assert next_sample.device == device
