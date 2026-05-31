from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import ReferenceCoreInitMode
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig

from .reference_transformer import build_reference_transformer


@dataclass(frozen=True)
class BackboneLoadReport:
    loaded_keys: tuple[str, ...]
    missing_reference_keys: tuple[str, ...]


ReferenceCoreLoadReport = BackboneLoadReport
_OPTIONAL_RUNTIME_TARGET_PREFIXES = (
    "proprio_context_encoder.",
    "proprio_hidden_context_encoder.",
    "generalist_mode_context_encoder.",
)


def _copy_if_present(
    target_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
    *,
    target_key: str,
    reference_key: str,
    loaded_keys: list[str],
    missing_reference_keys: list[str],
) -> None:
    if reference_key not in reference_state:
        missing_reference_keys.append(reference_key)
        return
    target_state[target_key] = reference_state[reference_key].detach().clone().to(dtype=target_state[target_key].dtype)
    loaded_keys.append(target_key)


def load_reference_weights_into_replica_core(
    replica_core: torch.nn.Module,
    *,
    backbone_config: SharedVideoTransformerConfig,
    action_dim: int,
) -> BackboneLoadReport:
    reference_transformer = build_reference_transformer(backbone_config, action_dim=action_dim)
    reference_state = reference_transformer.state_dict()
    target_state = replica_core.state_dict()
    loaded_keys: list[str] = []
    missing_reference_keys: list[str] = []
    init_mode = getattr(backbone_config, "reference_core_init_mode", ReferenceCoreInitMode.FULL)

    direct_pairs = [
        ("scale_shift_table", "scale_shift_table"),
        ("patch_embedding_mlp.weight", "patch_embedding_mlp.weight"),
        ("patch_embedding_mlp.bias", "patch_embedding_mlp.bias"),
        ("proj_out.weight", "proj_out.weight"),
        ("proj_out.bias", "proj_out.bias"),
        ("time_conditioner.time_embedder.linear_1.weight", "condition_embedder.time_embedder.linear_1.weight"),
        ("time_conditioner.time_embedder.linear_1.bias", "condition_embedder.time_embedder.linear_1.bias"),
        ("time_conditioner.time_embedder.linear_2.weight", "condition_embedder.time_embedder.linear_2.weight"),
        ("time_conditioner.time_embedder.linear_2.bias", "condition_embedder.time_embedder.linear_2.bias"),
        ("time_conditioner.time_proj.weight", "condition_embedder.time_proj.weight"),
        ("time_conditioner.time_proj.bias", "condition_embedder.time_proj.bias"),
        ("text_proj.linear_1.weight", "condition_embedder.text_embedder.linear_1.weight"),
        ("text_proj.linear_1.bias", "condition_embedder.text_embedder.linear_1.bias"),
        ("text_proj.linear_2.weight", "condition_embedder.text_embedder.linear_2.weight"),
        ("text_proj.linear_2.bias", "condition_embedder.text_embedder.linear_2.bias"),
    ]
    if init_mode == ReferenceCoreInitMode.FULL:
        direct_pairs.extend(
            [
                ("action_embedder.weight", "action_embedder.weight"),
                ("action_embedder.bias", "action_embedder.bias"),
                ("action_proj_out.weight", "action_proj_out.weight"),
                ("action_proj_out.bias", "action_proj_out.bias"),
                (
                    "action_time_conditioner.time_embedder.linear_1.weight",
                    "condition_embedder_action.time_embedder.linear_1.weight",
                ),
                (
                    "action_time_conditioner.time_embedder.linear_1.bias",
                    "condition_embedder_action.time_embedder.linear_1.bias",
                ),
                (
                    "action_time_conditioner.time_embedder.linear_2.weight",
                    "condition_embedder_action.time_embedder.linear_2.weight",
                ),
                (
                    "action_time_conditioner.time_embedder.linear_2.bias",
                    "condition_embedder_action.time_embedder.linear_2.bias",
                ),
                ("action_time_conditioner.time_proj.weight", "condition_embedder_action.time_proj.weight"),
                ("action_time_conditioner.time_proj.bias", "condition_embedder_action.time_proj.bias"),
                ("action_text_proj.linear_1.weight", "condition_embedder_action.text_embedder.linear_1.weight"),
                ("action_text_proj.linear_1.bias", "condition_embedder_action.text_embedder.linear_1.bias"),
                ("action_text_proj.linear_2.weight", "condition_embedder_action.text_embedder.linear_2.weight"),
                ("action_text_proj.linear_2.bias", "condition_embedder_action.text_embedder.linear_2.bias"),
            ]
        )
    for layer_index in range(backbone_config.num_layers):
        block_prefix = f"blocks.{layer_index}"
        direct_pairs.extend(
            [
                (f"{block_prefix}.scale_shift_table", f"{block_prefix}.scale_shift_table"),
                (f"{block_prefix}.attn1.to_q.weight", f"{block_prefix}.attn1.to_q.weight"),
                (f"{block_prefix}.attn1.to_q.bias", f"{block_prefix}.attn1.to_q.bias"),
                (f"{block_prefix}.attn1.to_k.weight", f"{block_prefix}.attn1.to_k.weight"),
                (f"{block_prefix}.attn1.to_k.bias", f"{block_prefix}.attn1.to_k.bias"),
                (f"{block_prefix}.attn1.to_v.weight", f"{block_prefix}.attn1.to_v.weight"),
                (f"{block_prefix}.attn1.to_v.bias", f"{block_prefix}.attn1.to_v.bias"),
                (f"{block_prefix}.attn1.to_out.0.weight", f"{block_prefix}.attn1.to_out.0.weight"),
                (f"{block_prefix}.attn1.to_out.0.bias", f"{block_prefix}.attn1.to_out.0.bias"),
                (f"{block_prefix}.attn1.norm_q.weight", f"{block_prefix}.attn1.norm_q.weight"),
                (f"{block_prefix}.attn1.norm_k.weight", f"{block_prefix}.attn1.norm_k.weight"),
                (f"{block_prefix}.attn2.to_q.weight", f"{block_prefix}.attn2.to_q.weight"),
                (f"{block_prefix}.attn2.to_q.bias", f"{block_prefix}.attn2.to_q.bias"),
                (f"{block_prefix}.attn2.to_k.weight", f"{block_prefix}.attn2.to_k.weight"),
                (f"{block_prefix}.attn2.to_k.bias", f"{block_prefix}.attn2.to_k.bias"),
                (f"{block_prefix}.attn2.to_v.weight", f"{block_prefix}.attn2.to_v.weight"),
                (f"{block_prefix}.attn2.to_v.bias", f"{block_prefix}.attn2.to_v.bias"),
                (f"{block_prefix}.attn2.to_out.0.weight", f"{block_prefix}.attn2.to_out.0.weight"),
                (f"{block_prefix}.attn2.to_out.0.bias", f"{block_prefix}.attn2.to_out.0.bias"),
                (f"{block_prefix}.attn2.norm_q.weight", f"{block_prefix}.attn2.norm_q.weight"),
                (f"{block_prefix}.attn2.norm_k.weight", f"{block_prefix}.attn2.norm_k.weight"),
                (f"{block_prefix}.norm2.weight", f"{block_prefix}.norm2.weight"),
                (f"{block_prefix}.norm2.bias", f"{block_prefix}.norm2.bias"),
                (f"{block_prefix}.ffn.net.0.proj.weight", f"{block_prefix}.ffn.net.0.proj.weight"),
                (f"{block_prefix}.ffn.net.0.proj.bias", f"{block_prefix}.ffn.net.0.proj.bias"),
                (f"{block_prefix}.ffn.net.2.weight", f"{block_prefix}.ffn.net.2.weight"),
                (f"{block_prefix}.ffn.net.2.bias", f"{block_prefix}.ffn.net.2.bias"),
            ]
        )

    for target_key, reference_key in direct_pairs:
        _copy_if_present(
            target_state,
            reference_state,
            target_key=target_key,
            reference_key=reference_key,
            loaded_keys=loaded_keys,
            missing_reference_keys=missing_reference_keys,
        )

    loaded_key_set = set(loaded_keys)
    for target_key in target_state:
        if target_key.startswith(_OPTIONAL_RUNTIME_TARGET_PREFIXES) and target_key not in loaded_key_set:
            missing_reference_keys.append(target_key)

    replica_core.load_state_dict(target_state, strict=False)
    return BackboneLoadReport(
        loaded_keys=tuple(loaded_keys),
        missing_reference_keys=tuple(missing_reference_keys),
    )
