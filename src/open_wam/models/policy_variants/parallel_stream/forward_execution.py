"""Parallel-stream transformer execution kernels and train dispatch.

This module owns tensor execution against the shared VisualTower transformer.
Recurrent rollout state, scheduler loops, and cache lifecycle remain separate.
"""

from __future__ import annotations

import torch
from einops import rearrange

from open_wam.models.common import (
    PreparedAttentionProfile,
    build_chunked_temporal_exact_attention_profile,
)
from open_wam.models.common.cache_backend_contracts import cache_backend_uses_slot_pool
from open_wam.models.common.cache_backend_lifecycle import (
    materialize_cache_backend_entries,
)
from open_wam.models.video_backbone.contracts import CacheState
from open_wam.models.visual_tower import (
    RuntimeStepInput,
    build_parallel_stream_exact_train_program,
)
from open_wam.models.visual_tower.exact_runtime import resolve_runtime_module_dtype
from open_wam.models.visual_tower.sequence_adapters import (
    prepare_exact_dual_stream_train_sequence,
)

from .exact_cache import build_dual_stream_cache_stream_ids
from .inference_conditioning import repeat_parallel_exact_input_for_cfg
from .proprio_conditioning import apply_parallel_chunk_proprio_context


def run_parallel_exact_dual_stream_forward(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    *,
    update_cache: int = 0,
    cache_name: str = "open_wam_exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = prepare_exact_dual_stream_train_sequence(
        input_dict,
        config=transformer.config,
        patch_size=transformer.patch_size,
        model_dtype=resolve_runtime_module_dtype(transformer),
        input_embed=lambda tensor, input_type: transformer._input_embed(tensor, input_type=input_type),
        exact_text_hidden_states=lambda text_emb: transformer._exact_text_hidden_states(
            text_emb,
            dtype=resolve_runtime_module_dtype(transformer),
        ),
        time_embed=lambda timesteps, height, width, dtype, action_mode: transformer._time_embed(
            timesteps,
            height,
            width,
            dtype=dtype,
            action_mode=action_mode,
        ),
        rope=transformer.rope,
    )
    hidden_states = prepared.hidden_states
    text_hidden_states = prepared.text_hidden_states
    rotary_emb = prepared.rotary_emb
    temb = prepared.temb
    timestep_proj = prepared.timestep_proj
    split_list = prepared.split_list
    exact_attention_profile = prepared.attention_profile
    hidden_states = apply_parallel_chunk_proprio_context(
        transformer,
        hidden_states=hidden_states,
        split_list=split_list,
        input_dict=input_dict,
    )
    cache_stream_ids = build_dual_stream_cache_stream_ids(
        split_list,
        device=hidden_states.device,
    )
    cache_state = transformer._resolve_exact_cache_state(cache_name)
    cache_backend_name = cache_state.backend_name if cache_state is not None else None
    cache_backend_payload = cache_state.backend_payload if cache_state is not None else None
    if cache_backend_uses_slot_pool(cache_backend_name):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        assert isinstance(latent_dict, dict)
        assert isinstance(action_dict, dict)
        attention_profile_name = input_dict.get("attention_profile_name")
        rebuilt_dense_profile = build_chunked_temporal_exact_attention_profile(
            latent_shape=tuple(int(dim) for dim in latent_dict["noisy_latents"].shape),
            action_shape=tuple(int(dim) for dim in action_dict["noisy_latents"].shape),
            padded_length=int(hidden_states.shape[1] - sum(int(length) for length in split_list[:4])),
            chunk_size=int(input_dict["chunk_size"]),
            window_size=int(input_dict["window_size"]),
            patch_size=transformer.patch_size,
            text_token_count=int(latent_dict["text_emb"].shape[1]),
            base_text_token_count=(
                None
                if input_dict.get("base_text_token_count") is None
                else int(input_dict["base_text_token_count"])
            ),
            proprio_context_token_count=int(input_dict.get("proprio_context_token_count", 0) or 0),
            chunk_origin_frame=int(input_dict.get("chunk_origin_frame", 0) or 0),
            prefix_condition_frames=int(input_dict.get("prefix_condition_frames", 0) or 0),
            singleton_chunk_frame=(
                None
                if input_dict.get("singleton_chunk_frame") is None
                else int(input_dict["singleton_chunk_frame"])
            ),
            action_context_mask=(
                action_dict.get("actions_mask")
                if torch.is_tensor(action_dict.get("actions_mask"))
                else None
            ),
            device=hidden_states.device,
            build_dense_masks=True,
            build_flex_masks=False,
            current_block_coupling=(
                str(attention_profile_name)
                if attention_profile_name not in (None, "none")
                else None
            ),
            preserve_video_pretrain_history=bool(
                input_dict.get("preserve_video_pretrain_history", False)
            ),
            history_stream_visibility=input_dict.get("history_stream_visibility"),
            conditional_history_policy=input_dict.get("conditional_history_policy"),
        )
        exact_attention_profile = PreparedAttentionProfile(
            spec=rebuilt_dense_profile.spec,
            self_attention_mask=rebuilt_dense_profile.self_attention_mask,
            cross_attention_mask=rebuilt_dense_profile.cross_attention_mask,
            self_attention_block_mask=None,
            cross_attention_block_mask=None,
            metadata=dict(rebuilt_dense_profile.metadata),
        )

    for layer_index, block in enumerate(transformer.blocks):
        hidden_states, _, _ = block(
            hidden_states,
            encoder_hidden_states=text_hidden_states,
            temb=timestep_proj,
            rotary_emb=rotary_emb,
            attention_profile=exact_attention_profile,
            self_attention_cache_backend_name=cache_backend_name,
            self_attention_cache_backend_state=(
                cache_backend_payload.layer_states[layer_index]
                if cache_backend_uses_slot_pool(cache_backend_name)
                and cache_backend_payload is not None
                and layer_index < len(cache_backend_payload.layer_states)
                else None
            ),
            self_attention_cache_update_mode=update_cache,
            self_attention_cache_stream_ids=cache_stream_ids,
        )

    temb_scale_shift_table = transformer.scale_shift_table[None] + temb[:, :, None, ...]
    shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
    shift = shift.to(hidden_states.device).squeeze(1)
    scale = scale.to(hidden_states.device).squeeze(1)
    hidden_states = (transformer.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)
    if cache_state is not None and cache_backend_uses_slot_pool(cache_backend_name):
        materialized_entries = materialize_cache_backend_entries(cache_backend_payload)
        transformer._exact_runtime_caches[cache_name] = CacheState(
            supported=cache_state.supported,
            current_start_frame=cache_state.current_start_frame,
            cached_frames=cache_state.cached_frames,
            chunk_size=cache_state.chunk_size,
            capability=cache_state.capability,
            backend_name=cache_state.backend_name,
            backend_payload=cache_backend_payload,
            payload=dict(cache_state.payload),
            self_attention_kv=materialized_entries,
            cross_attention_kv=cache_state.cross_attention_kv,
            update_metadata=cache_state.update_metadata,
        )
    latent_hidden_states, _, action_hidden_states, _, _ = torch.split(
        hidden_states,
        tuple(int(length) for length in split_list),
        dim=1,
    )
    effective_batch_size = int(input_dict["latent_dict"]["noisy_latents"].shape[0])  # type: ignore[index]
    latent_hidden_states = transformer.proj_out(latent_hidden_states)
    if latent_hidden_states.shape[0] == 1:
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "1 (b l) c -> b l c",
            b=effective_batch_size,
        )
    elif latent_hidden_states.shape[0] == effective_batch_size:
        latent_hidden_states = latent_hidden_states.contiguous()
    else:
        raise ValueError(
            "Unexpected exact joint latent output layout: expected leading dimension to be 1 "
            f"or effective_batch_size={effective_batch_size}, got {latent_hidden_states.shape[0]}."
        )
    action_hidden_states = transformer.action_proj_out(action_hidden_states)
    if action_hidden_states.shape[0] == 1:
        action_hidden_states = rearrange(
            action_hidden_states,
            "1 (b l) c -> b l c",
            b=effective_batch_size,
        )
    elif action_hidden_states.shape[0] != effective_batch_size:
        raise ValueError(
            "Unexpected exact joint action output layout: expected leading dimension to be 1 "
            f"or effective_batch_size={effective_batch_size}, got {action_hidden_states.shape[0]}."
        )
    return latent_hidden_states, action_hidden_states


def run_parallel_action_conditioned_forward(
    transformer: torch.nn.Module,
    *,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    video_guidance_scale: float,
    action_guidance_scale: float,
    negative_text_emb: torch.Tensor | None,
    update_cache: int = 0,
    cache_name: str = "open_wam_exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    def _split_cfg_prediction(
        prediction: torch.Tensor,
        *,
        logical_batch_size: int,
        expected_tokens: int,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.ndim != 3:
            raise ValueError(f"Expected {name} prediction rank 3, got shape {tuple(prediction.shape)}.")
        if prediction.shape[0] == logical_batch_size * 2 and prediction.shape[1] == expected_tokens:
            return prediction[:logical_batch_size], prediction[logical_batch_size:]
        if prediction.shape[0] == logical_batch_size * 2 and prediction.shape[1] == logical_batch_size * 2 * expected_tokens:
            packed = rearrange(
                prediction,
                "(g b_row) (h b_seq l) c -> g b_row h b_seq l c",
                g=2,
                h=2,
                b_row=logical_batch_size,
                b_seq=logical_batch_size,
                l=expected_tokens,
            )
            batch_index = torch.arange(logical_batch_size, device=prediction.device)
            cond = packed[0, batch_index, 0, batch_index]
            uncond = packed[1, batch_index, 1, batch_index]
            return cond.contiguous(), uncond.contiguous()
        if prediction.shape[0] == logical_batch_size and prediction.shape[1] == expected_tokens * 2:
            return prediction[:, :expected_tokens], prediction[:, expected_tokens:]
        if prediction.shape[0] == 1 and prediction.shape[1] == logical_batch_size * expected_tokens * 2:
            unpacked = rearrange(
                prediction,
                "1 (g b l) c -> (g b) l c",
                g=2,
                b=logical_batch_size,
                l=expected_tokens,
            )
            return unpacked[:logical_batch_size], unpacked[logical_batch_size:]
        raise ValueError(
            f"Unable to split CFG {name} prediction with shape {tuple(prediction.shape)}; "
            f"expected logical_batch_size={logical_batch_size}, expected_tokens={expected_tokens}."
        )

    batch_size = input_dict["latent_dict"]["noisy_latents"].shape[0]  # type: ignore[index]
    latent_noisy = input_dict["latent_dict"]["noisy_latents"]  # type: ignore[index]
    action_noisy = input_dict["action_dict"]["noisy_latents"]  # type: ignore[index]
    expected_video_tokens = (
        int(latent_noisy.shape[2]) // transformer.patch_size[0]
    ) * (
        int(latent_noisy.shape[3]) // transformer.patch_size[1]
    ) * (
        int(latent_noisy.shape[4]) // transformer.patch_size[2]
    )
    expected_action_tokens = int(action_noisy.shape[2]) * int(action_noisy.shape[3])
    use_cfg = negative_text_emb is not None and (video_guidance_scale > 1.0 or action_guidance_scale > 1.0)
    effective_input = input_dict
    if use_cfg:
        effective_input = repeat_parallel_exact_input_for_cfg(input_dict, negative_text_emb=negative_text_emb)
    with torch.inference_mode():
        video_pred, action_pred = run_parallel_exact_dual_stream_forward(
            transformer,
            effective_input,
            update_cache=update_cache,
            cache_name=cache_name,
        )
    if not use_cfg:
        return video_pred, action_pred
    cond_video_pred, uncond_video_pred = _split_cfg_prediction(
        video_pred,
        logical_batch_size=batch_size,
        expected_tokens=expected_video_tokens,
        name="video",
    )
    cond_action_pred, uncond_action_pred = _split_cfg_prediction(
        action_pred,
        logical_batch_size=batch_size,
        expected_tokens=expected_action_tokens,
        name="action",
    )
    combined_video_pred = uncond_video_pred + video_guidance_scale * (cond_video_pred - uncond_video_pred)
    combined_action_pred = uncond_action_pred + action_guidance_scale * (cond_action_pred - uncond_action_pred)
    return combined_video_pred, combined_action_pred


def run_parallel_exact_train(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_dict.get("per_chunk_proprio_state") is not None:
        return run_parallel_exact_dual_stream_forward(transformer, input_dict)
    if hasattr(transformer, "execute_runtime_step"):
        step_output = transformer.execute_runtime_step(
            RuntimeStepInput(
                program=build_parallel_stream_exact_train_program(
                    attention_profile_name=input_dict.get("attention_profile_name"),  # type: ignore[arg-type]
                    cache_backend_name="slot_pool_exact",
                ),
                payload=input_dict,
            )
        )
        try:
            return (
                step_output.projected_outputs["video_prediction"],
                step_output.projected_outputs["action_prediction"],
            )
        except KeyError as exc:
            raise ValueError("Exact dual-stream runtime step did not return both video/action predictions.") from exc

    forward_train = getattr(transformer, "forward_train", None)
    if callable(forward_train):
        return forward_train(input_dict)
    return run_parallel_exact_dual_stream_forward(transformer, input_dict)


def run_parallel_action_conditioned_train(
    transformer: torch.nn.Module,
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the LingBot exact train-time packed layout and shared runtime
    # execution path; the joint-denoise variant changes inference rollout
    # semantics, not the backbone's train-time sequence contract.
    return run_parallel_exact_train(transformer, input_dict)
