from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_wam.configs import PolicyVariantName, VideoConditionSource
from open_wam.runtime.rollout import (
    apply_rollout_chunk_steps_override,
    build_sequence_rollout_infer_extra,
    prepare_rollout_observation_inputs,
    resolve_initial_generation_action_start,
    resolve_runtime_devices,
    uses_zero_based_generation_start,
)


class _ReferenceAssets:
    has_vae = True

    def __init__(self) -> None:
        self.video_calls: list[dict[str, object]] = []

    def encode_video(self, video, *, placements, reset_cache):
        self.video_calls.append(
            {
                "video": video,
                "placements": placements,
                "reset_cache": reset_cache,
            }
        )
        return torch.ones(1, 2, 1, 1, 1)

    def encode_text(self, task_text, *, device, dtype):
        del task_text
        return torch.ones(1, 1, 2, device=device, dtype=dtype)

    def encode_blank_text(self, *, batch_size, device, dtype):
        return torch.zeros(batch_size, 1, 2, device=device, dtype=dtype)


def test_prepare_rollout_observation_inputs_uses_reset_cache_encoding() -> None:
    assets = _ReferenceAssets()
    canonical_video = torch.zeros(1, 3, 1, 2, 2)
    placements = ("agentview",)
    pipeline = SimpleNamespace(
        canonicalize=lambda views: SimpleNamespace(
            video=canonical_video,
            placements=placements,
        ),
        visual_tower=SimpleNamespace(
            frontend=SimpleNamespace(reference_assets=assets),
        ),
    )

    result = prepare_rollout_observation_inputs(
        pipeline,
        views={"agentview": torch.zeros(1, 2, 2, 3)},
        task_text=("task",),
        frontend_device=torch.device("cpu"),
        runtime_device=torch.device("cpu"),
    )

    assert result["video_latents"].shape == (1, 2, 1, 1, 1)
    assert result["text_context"].shape == (1, 1, 2)
    assert result["negative_text_context"].shape == (1, 1, 2)
    assert assets.video_calls == [
        {
            "video": canonical_video,
            "placements": placements,
            "reset_cache": True,
        }
    ]


def test_sequence_rollout_metadata_uses_typed_policy_choices() -> None:
    method4 = SimpleNamespace(
        policy_variant=SimpleNamespace(
            name=PolicyVariantName.POST_LATENT,
            train_video_condition_source=VideoConditionSource.GENERATED_FUTURE,
        )
    )
    dual_expert = SimpleNamespace(
        policy_variant=SimpleNamespace(name=PolicyVariantName.DUAL_EXPERT)
    )

    method4_extra = build_sequence_rollout_infer_extra(
        config=method4,
        prompt="task",
        generation_action_start=9,
        task_id=1,
        episode_idx=2,
    )
    dual_expert_extra = build_sequence_rollout_infer_extra(
        config=dual_expert,
        prompt="task",
        generation_action_start=0,
        runtime_device=torch.device("cpu"),
    )

    assert method4_extra["video_condition_frame_start"] == 9
    assert method4_extra["video_condition_observed_prefix_anchor"] == "end"
    assert "video_condition_sample_seed" in method4_extra
    assert dual_expert_extra == {
        "task_text": ("task",),
        "action_device": "cpu",
    }
    assert uses_zero_based_generation_start(method4) is True
    assert uses_zero_based_generation_start(dual_expert) is True


def test_initial_generation_start_defaults_to_observation_count() -> None:
    observations = [{"frame": index} for index in range(5)]

    assert (
        resolve_initial_generation_action_start(
            observations,
            initial_generation_action_start=None,
        )
        == 5
    )
    assert (
        resolve_initial_generation_action_start(
            observations,
            initial_generation_action_start=None,
            rollout_starts_at_action_zero=True,
        )
        == 0
    )


def test_runtime_device_and_chunk_overrides_validate_operator_input() -> None:
    fallback = torch.device("cpu")
    assert resolve_runtime_devices(None, fallback=fallback) == (fallback,)
    assert resolve_runtime_devices(" cpu, cuda:1 ", fallback=fallback) == (
        torch.device("cpu"),
        torch.device("cuda:1"),
    )

    config = SimpleNamespace(
        action_decoder=SimpleNamespace(rollout_chunk_steps=4)
    )
    apply_rollout_chunk_steps_override(config, 2)
    assert config.action_decoder.rollout_chunk_steps == 2

    with pytest.raises(ValueError, match="has no rollout chunk steps"):
        apply_rollout_chunk_steps_override(
            SimpleNamespace(action_decoder=SimpleNamespace()),
            2,
        )
