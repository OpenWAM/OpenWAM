from __future__ import annotations

import torch

from open_wam.configs import (
    JointDenoiseTrainingMode,
    ParallelStreamPolicyConfig,
)
from open_wam.models.policy_variants.parallel_stream import reference_runtime
from open_wam.models.policy_variants.parallel_stream.inference_conditioning import (
    append_generalist_mode_text_context,
    repeat_parallel_exact_input_for_cfg,
)


def test_reference_runtime_inference_conditioning_names_alias_owner() -> None:
    assert (
        reference_runtime._inject_generalist_mode_text_context
        is append_generalist_mode_text_context
    )
    assert (
        reference_runtime._repeat_joint_input_for_cfg
        is repeat_parallel_exact_input_for_cfg
    )


def test_disabled_mode_context_preserves_text_object_identity() -> None:
    text = torch.randn(1, 3, 4)
    negative = torch.randn(1, 3, 4)

    actual_text, actual_negative = append_generalist_mode_text_context(
        object(),
        policy_config=ParallelStreamPolicyConfig(
            hidden_size=8,
            generalist_mode_text_token=False,
        ),
        text_emb=text,
        negative_text_emb=negative,
        mode=JointDenoiseTrainingMode.JOINT,
    )

    assert actual_text is text
    assert actual_negative is negative


def test_parallel_cfg_repeat_preserves_metadata_values_and_gradients() -> None:
    latent_noisy = torch.randn(1, 2, 2, 1, 1, requires_grad=True)
    action_noisy = torch.randn(1, 3, 2, 2, 1, requires_grad=True)
    text = torch.randn(1, 4, 5, requires_grad=True)
    negative = torch.randn(1, 4, 5, requires_grad=True)
    proprio = torch.randn(1, 2, 6, requires_grad=True)
    marker = object()
    input_dict = {
        "latent_dict": {
            "noisy_latents": latent_noisy,
            "latent": torch.ones_like(latent_noisy),
            "grid_id": torch.zeros(1, 2, 3),
            "timesteps": torch.zeros(1, 2),
            "cond_timesteps": torch.zeros(1, 2),
            "text_emb": text,
            "loss_mask": torch.ones_like(latent_noisy),
        },
        "action_dict": {
            "noisy_latents": action_noisy,
            "latent": torch.ones_like(action_noisy),
            "grid_id": torch.zeros(1, 4, 3),
            "timesteps": torch.zeros(1, 2),
            "cond_timesteps": torch.zeros(1, 2),
            "text_emb": text,
            "actions_mask": torch.ones_like(action_noisy),
            "loss_mask": torch.ones_like(action_noisy),
        },
        "per_chunk_proprio_state": proprio,
        "marker": marker,
    }

    repeated = repeat_parallel_exact_input_for_cfg(
        input_dict,
        negative_text_emb=negative,
    )

    assert repeated is not input_dict
    assert repeated["marker"] is marker
    assert repeated["latent_dict"]["noisy_latents"].shape[0] == 2
    assert repeated["action_dict"]["noisy_latents"].shape[0] == 2
    assert repeated["per_chunk_proprio_state"].shape[0] == 2
    torch.testing.assert_close(
        repeated["latent_dict"]["text_emb"],
        torch.cat([text, negative], dim=0),
    )
    assert input_dict["latent_dict"]["noisy_latents"] is latent_noisy
    assert input_dict["action_dict"]["noisy_latents"] is action_noisy

    loss = (
        repeated["latent_dict"]["noisy_latents"].sum()
        + repeated["action_dict"]["noisy_latents"].sum()
        + repeated["latent_dict"]["text_emb"].sum()
        + repeated["action_dict"]["text_emb"].sum()
        + repeated["per_chunk_proprio_state"].sum()
    )
    loss.backward()

    torch.testing.assert_close(latent_noisy.grad, torch.full_like(latent_noisy, 2))
    torch.testing.assert_close(action_noisy.grad, torch.full_like(action_noisy, 2))
    torch.testing.assert_close(text.grad, torch.full_like(text, 2))
    torch.testing.assert_close(negative.grad, torch.full_like(negative, 2))
    torch.testing.assert_close(proprio.grad, torch.full_like(proprio, 2))
