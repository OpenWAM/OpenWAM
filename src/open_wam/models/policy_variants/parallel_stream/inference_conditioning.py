"""Prepare parallel-stream inference conditioning without model execution."""

from __future__ import annotations

import torch

from open_wam.configs.enums import DynamicsObjective
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.common.dynamics_conditioning import (
    append_dynamics_mode_context_token,
)


def append_generalist_mode_text_context(
    transformer: torch.nn.Module,
    *,
    policy_config: ParallelStreamPolicyConfig,
    text_emb: torch.Tensor,
    negative_text_emb: torch.Tensor | None,
    mode: DynamicsObjective | str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not bool(policy_config.generalist_mode_text_token):
        return text_emb, negative_text_emb
    text_emb, token_count = append_dynamics_mode_context_token(
        transformer,
        text_emb,
        mode,
    )
    if negative_text_emb is not None:
        negative_text_emb, negative_token_count = append_dynamics_mode_context_token(
            transformer,
            negative_text_emb,
            mode,
        )
        if negative_token_count != token_count:
            raise ValueError(
                "Generalist mode text-token ablation expects conditioned and CFG-negative "
                "branches to append the same number of tokens, "
                f"got conditioned={token_count} and negative={negative_token_count}."
            )
    return text_emb, negative_text_emb


def repeat_parallel_exact_input_for_cfg(
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    *,
    negative_text_emb: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    latent_dict = dict(input_dict["latent_dict"])  # type: ignore[index]
    action_dict = dict(input_dict["action_dict"])  # type: ignore[index]
    repeated_latent_dict = {
        **latent_dict,
        "noisy_latents": latent_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "latent": latent_dict["latent"].repeat(2, 1, 1, 1, 1),
        "grid_id": latent_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": latent_dict["timesteps"].repeat(2, 1),
        "cond_timesteps": latent_dict["cond_timesteps"].repeat(2, 1),
        "text_emb": torch.cat([latent_dict["text_emb"], negative_text_emb], dim=0),
    }
    repeated_action_dict = {
        **action_dict,
        "noisy_latents": action_dict["noisy_latents"].repeat(2, 1, 1, 1, 1),
        "latent": action_dict["latent"].repeat(2, 1, 1, 1, 1),
        "grid_id": action_dict["grid_id"].repeat(2, 1, 1),
        "timesteps": action_dict["timesteps"].repeat(2, 1),
        "cond_timesteps": action_dict["cond_timesteps"].repeat(2, 1),
        "text_emb": torch.cat([action_dict["text_emb"], negative_text_emb], dim=0),
    }
    if "actions_mask" in action_dict:
        repeated_action_dict["actions_mask"] = action_dict["actions_mask"].repeat(2, 1, 1, 1, 1)
    if "loss_mask" in latent_dict:
        repeated_latent_dict["loss_mask"] = latent_dict["loss_mask"].repeat(2, 1, 1, 1, 1)
    if "loss_mask" in action_dict:
        repeated_action_dict["loss_mask"] = action_dict["loss_mask"].repeat(2, 1, 1, 1, 1)
    repeated_input = {
        **input_dict,
        "latent_dict": repeated_latent_dict,
        "action_dict": repeated_action_dict,
    }
    proprio_state = input_dict.get("per_chunk_proprio_state")
    if isinstance(proprio_state, torch.Tensor):
        repeated_input["per_chunk_proprio_state"] = proprio_state.repeat(2, 1, 1)
    return repeated_input


__all__ = [
    "append_generalist_mode_text_context",
    "repeat_parallel_exact_input_for_cfg",
]
