from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.models.common import (
    PreparedAttentionProfile,
    build_chunked_temporal_exact_attention_profile,
    build_register_attention_mask,
    build_register_position_context,
    chunked_temporal_exact_coupling_from_profile_name,
    normalize_attention_profile_name,
)
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig, resolve_stage_attention_mode

from .contracts import (
    StructuredAttentionContext,
    StructuredBlockSemantics,
    StructuredFrequencyBundle,
    VisualCoreInput,
)
from .grid_ids import build_block_register_grid_ids, build_video_grid_ids
from .runtime_programs import RuntimeStepInput


@dataclass(frozen=True)
class PreparedExactTrainSequence:
    """Prepared exact dual-stream train inputs for the shared backbone."""

    hidden_states: torch.Tensor
    text_hidden_states: torch.Tensor
    rotary_emb: torch.Tensor
    temb: torch.Tensor
    timestep_proj: torch.Tensor
    split_list: list[int]
    batch_size: int
    attention_profile: PreparedAttentionProfile | None


@dataclass(frozen=True)
class PreparedRuntimeSequence:
    """Backbone-ready step payload resolved from a runtime program."""

    mode: str
    core_input: VisualCoreInput | None = None
    payload: dict[str, Any] | None = None
    exact_train: PreparedExactTrainSequence | None = None
    exact_inference: PreparedExactTrainSequence | None = None
    update_cache: int = 0
    cache_name: str = "open_wam_exact"
    action_mode: bool = False


def _materialize_register_core_input(
    core_input: VisualCoreInput,
    *,
    hidden_size: int,
) -> VisualCoreInput:
    register_components = core_input.register_components
    if register_components is None:
        return core_input

    layout = register_components.layout
    semantics = register_components.semantics
    if semantics.sequence_family != "register_sequence":
        raise ValueError(
            "Structured runtime adapter only supports `register_sequence`, "
            f"got {semantics.sequence_family!r}."
        )
    if semantics.attention_style != "blockwise_causal":
        raise ValueError(
            "Structured runtime adapter only supports `blockwise_causal`, "
            f"got {semantics.attention_style!r}."
        )

    batch_size = register_components.noisy_video_tokens.shape[0]
    device = register_components.noisy_video_tokens.device

    clean_prefix_grid_ids = build_video_grid_ids(
        register_components.token_grid,
        device=device,
        frame_shift=float(register_components.current_start_frame),
    )
    noisy_video_grid_ids = build_video_grid_ids(
        register_components.token_grid,
        device=device,
        frame_shift=float(register_components.current_start_frame),
    )
    packed_token_chunks: list[torch.Tensor] = []
    packed_grid_chunks: list[torch.Tensor] = []
    if register_components.clean_video_prefix_tokens is not None:
        packed_token_chunks.append(register_components.clean_video_prefix_tokens)
        packed_grid_chunks.append(clean_prefix_grid_ids)
    packed_token_chunks.extend(
        [
            register_components.noisy_video_tokens,
            register_components.action_register_tokens,
            register_components.state_register_tokens,
        ]
    )
    observed_prefix_frames = 1
    register_frame_shift = float(register_components.current_start_frame + observed_prefix_frames)
    action_tokens_per_block = (
        register_components.action_register_tokens.shape[1] // max(layout.num_action_blocks, 1)
        if layout.num_action_blocks > 0
        else 0
    )
    state_tokens_per_block = (
        register_components.state_register_tokens.shape[1] // max(layout.num_state_blocks, 1)
        if layout.num_state_blocks > 0
        else 0
    )
    action_grid_ids = build_block_register_grid_ids(
        num_blocks=layout.num_action_blocks,
        tokens_per_block=action_tokens_per_block,
        device=device,
        frame_shift=register_frame_shift,
        stream_marker=-1.0,
    )
    state_grid_ids = build_block_register_grid_ids(
        num_blocks=layout.num_state_blocks,
        tokens_per_block=state_tokens_per_block,
        device=device,
        frame_shift=register_frame_shift,
        stream_marker=-2.0,
    )
    packed_grid_chunks.extend(
        [
            noisy_video_grid_ids,
            action_grid_ids,
            state_grid_ids,
        ]
    )
    packed_tokens = torch.cat(packed_token_chunks, dim=1)
    packed_grid_ids = torch.cat(packed_grid_chunks, dim=1)
    position_context = build_register_position_context(
        layout=layout,
        token_grid=register_components.token_grid,
        hidden_size=hidden_size,
        device=device,
        current_start_frame=register_components.current_start_frame,
    )[None, :, :].expand(batch_size, -1, -1)

    clean_video_values = torch.zeros(
        batch_size,
        layout.clean_video_sequence_length,
        device=device,
        dtype=torch.float32,
    )
    noisy_video_values = register_components.video_timesteps.repeat_interleave(
        register_components.token_grid.tokens_per_frame,
        dim=1,
    )
    timestep_chunks: list[torch.Tensor] = []
    if layout.has_clean_video_prefix:
        timestep_chunks.append(clean_video_values)
    timestep_chunks.append(noisy_video_values)
    if register_components.action_register_tokens.shape[1] > 0:
        timestep_chunks.append(register_components.action_timesteps)
    if register_components.state_register_tokens.shape[1] > 0:
        timestep_chunks.append(register_components.state_timesteps)
    timestep_values = torch.cat(timestep_chunks, dim=1)
    attention_mask = build_register_attention_mask(layout, batch_size=batch_size, device=device)

    stream_id_chunks: list[torch.Tensor] = []
    if layout.has_clean_video_prefix:
        stream_id_chunks.append(
            torch.zeros(batch_size, layout.clean_video_sequence_length, device=device, dtype=torch.long)
        )
    stream_id_chunks.extend(
        [
            torch.zeros(batch_size, layout.noisy_video_sequence_length, device=device, dtype=torch.long),
            torch.ones(batch_size, register_components.action_register_tokens.shape[1], device=device, dtype=torch.long),
            torch.ones(batch_size, register_components.state_register_tokens.shape[1], device=device, dtype=torch.long),
        ]
    )
    stream_ids = torch.cat(stream_id_chunks, dim=1)

    action_span = (
        layout.action_block_spans[0][0],
        layout.action_block_spans[-1][1],
    ) if layout.action_block_spans else (layout.noisy_video_span[1], layout.noisy_video_span[1])
    state_span = (
        layout.state_block_spans[0][0],
        layout.state_block_spans[-1][1],
    ) if layout.state_block_spans else (action_span[1], action_span[1])
    structured_block_semantics = None
    structured_frequency_bundle = None
    structured_attention_context = None
    if register_components.semantics.structured_block_mode != "none":
        structured_block_semantics = StructuredBlockSemantics(
            mode=register_components.semantics.structured_block_mode,
            teacher_forcing_enabled=register_components.semantics.teacher_forcing,
            clean_prefix_span=layout.clean_video_span,
            video_span=layout.noisy_video_span,
            action_span=action_span,
            state_span=state_span,
            clean_prefix_length=layout.clean_video_sequence_length,
            video_token_length=layout.noisy_video_sequence_length,
            action_register_length=register_components.action_register_tokens.shape[1],
            state_register_length=register_components.state_register_tokens.shape[1],
            current_start_frame=register_components.current_start_frame,
            observed_prefix_frames=observed_prefix_frames,
            time_layout=register_components.semantics.structured_time_layout,
            position_layout=register_components.semantics.structured_teacher_forcing_layout,
            frequency_mode=register_components.semantics.structured_frequency_mode,
            metadata={
                "sequence_family": register_components.semantics.sequence_family,
                "attention_style": register_components.semantics.attention_style,
                "teacher_forcing_layout": register_components.semantics.teacher_forcing_layout,
                "timestep_layout": register_components.semantics.timestep_layout,
                "attention_kernel": register_components.semantics.structured_attention_kernel,
                "cache_kernel": register_components.semantics.structured_cache_kernel,
                "rollout_phase": (
                    "teacher_forcing"
                    if register_components.semantics.teacher_forcing
                    else "cached_rollout"
                ),
                "action_state_index": max(
                    (register_components.current_start_frame - observed_prefix_frames)
                    // max(layout.tokens_per_image_block // max(layout.tokens_per_frame, 1), 1),
                    0,
                ),
                "register_frame_shift": register_frame_shift,
            },
        )
        structured_frequency_bundle = StructuredFrequencyBundle(
            layout=register_components.semantics.structured_frequency_mode,
            clean_prefix_grid_ids=(
                clean_prefix_grid_ids if register_components.clean_video_prefix_tokens is not None else None
            ),
            video_grid_ids=noisy_video_grid_ids,
            action_grid_ids=action_grid_ids,
            state_grid_ids=state_grid_ids,
            shared_grid_ids=packed_grid_ids,
            metadata={
                "current_start_frame": register_components.current_start_frame,
                "register_frame_shift": register_frame_shift,
                "video_tokens": layout.noisy_video_sequence_length,
                "action_tokens": register_components.action_register_tokens.shape[1],
                "state_tokens": register_components.state_register_tokens.shape[1],
                "num_action_blocks": layout.num_action_blocks,
                "num_state_blocks": layout.num_state_blocks,
                "action_tokens_per_block": action_tokens_per_block,
                "state_tokens_per_block": state_tokens_per_block,
            },
        )
        structured_attention_context = StructuredAttentionContext(
            mode=register_components.semantics.structured_block_mode,
            teacher_forcing_enabled=register_components.semantics.teacher_forcing,
            clean_prefix_length=layout.clean_video_sequence_length,
            video_token_length=layout.noisy_video_sequence_length,
            action_register_length=register_components.action_register_tokens.shape[1],
            state_register_length=register_components.state_register_tokens.shape[1],
            current_start_frame=register_components.current_start_frame,
            observed_prefix_frames=observed_prefix_frames,
            num_frame_per_block=max(layout.tokens_per_image_block // max(layout.tokens_per_frame, 1), 1),
            num_action_per_block=action_tokens_per_block,
            num_state_per_block=state_tokens_per_block,
            num_video_blocks=layout.num_image_blocks,
            num_action_blocks=layout.num_action_blocks,
            num_state_blocks=layout.num_state_blocks,
            tokens_per_frame=layout.tokens_per_frame,
            tokens_per_video_block=layout.tokens_per_image_block,
            frequency_mode=register_components.semantics.structured_frequency_mode,
            attention_kernel=register_components.semantics.structured_attention_kernel,
            cache_kernel=register_components.semantics.structured_cache_kernel,
            rollout_phase=(
                "teacher_forcing"
                if register_components.semantics.teacher_forcing
                else "cached_rollout"
            ),
            action_state_index=max(
                (register_components.current_start_frame - observed_prefix_frames)
                // max(
                    max(layout.tokens_per_image_block // max(layout.tokens_per_frame, 1), 1),
                    1,
                ),
                0,
            ),
            clean_prefix_grid_ids=(
                clean_prefix_grid_ids if register_components.clean_video_prefix_tokens is not None else None
            ),
            video_grid_ids=noisy_video_grid_ids,
            action_grid_ids=action_grid_ids,
            state_grid_ids=state_grid_ids,
            metadata={
                "sequence_family": register_components.semantics.sequence_family,
                "attention_style": register_components.semantics.attention_style,
                "teacher_forcing_layout": register_components.semantics.teacher_forcing_layout,
                "timestep_layout": register_components.semantics.timestep_layout,
                "register_frame_shift": register_frame_shift,
            },
        )

    return VisualCoreInput(
        tokens=packed_tokens,
        token_layout=layout,
        position_context=position_context,
        timestep_context=None,
        grid_ids=packed_grid_ids,
        timestep_values=timestep_values,
        stream_ids=stream_ids,
        text_context=core_input.text_context,
        attention_mask=attention_mask,
        attention_profile=core_input.attention_profile,
        cache_state=core_input.cache_state,
        cache_update_metadata=core_input.cache_update_metadata,
        conditioning=core_input.conditioning,
        sequence_metadata=core_input.sequence_metadata,
        register_components=None,
        structured_block_semantics=structured_block_semantics,
        structured_frequency_bundle=structured_frequency_bundle,
        structured_attention_context=structured_attention_context,
    )


def prepare_exact_dual_stream_train_sequence(
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    *,
    config: SharedVideoTransformerConfig,
    patch_size: tuple[int, int, int],
    model_dtype: torch.dtype,
    input_embed: Callable[[torch.Tensor, str], torch.Tensor],
    exact_text_hidden_states: Callable[[torch.Tensor], torch.Tensor],
    time_embed: Callable[[torch.Tensor, int, int, torch.dtype, bool], tuple[torch.Tensor, torch.Tensor]],
    rope: Callable[[torch.Tensor], torch.Tensor],
) -> PreparedExactTrainSequence:
    latent_dict = input_dict["latent_dict"]
    action_dict = input_dict["action_dict"]
    assert isinstance(latent_dict, dict)
    assert isinstance(action_dict, dict)

    latent_dict = {
        key: value.to(model_dtype) if torch.is_tensor(value) and torch.is_floating_point(value) else value
        for key, value in latent_dict.items()
    }
    action_dict = {
        key: value.to(model_dtype) if torch.is_tensor(value) and torch.is_floating_point(value) else value
        for key, value in action_dict.items()
    }

    batch_size = int(latent_dict["noisy_latents"].shape[0])
    latent_hidden_states = input_embed(latent_dict["noisy_latents"], "latent").flatten(0, 1).contiguous()[None].clone()
    action_hidden_states = input_embed(action_dict["noisy_latents"], "action").flatten(0, 1).contiguous()[None].clone()
    text_hidden_states = exact_text_hidden_states(latent_dict["text_emb"]).flatten(0, 1).contiguous()[None].clone()
    condition_latent_hidden_states = input_embed(latent_dict["latent"], "latent").flatten(0, 1).contiguous()[None].clone()
    condition_action_hidden_states = input_embed(action_dict["latent"], "action").flatten(0, 1).contiguous()[None].clone()

    hidden_states = torch.cat(
        [
            latent_hidden_states,
            condition_latent_hidden_states,
            action_hidden_states,
            condition_action_hidden_states,
        ],
        dim=1,
    )
    latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1).contiguous()[None].clone()
    action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1).contiguous()[None].clone()
    full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)
    rotary_emb = rope(full_grid_id)[:, :, None]

    latent_time_steps = torch.cat(
        [latent_dict["timesteps"].flatten(0, 1), latent_dict["cond_timesteps"].flatten(0, 1)],
        dim=0,
    ).contiguous()[None].clone()
    action_time_steps = torch.cat(
        [action_dict["timesteps"].flatten(0, 1), action_dict["cond_timesteps"].flatten(0, 1)],
        dim=0,
    ).contiguous()[None].clone()
    latent_temb, latent_timestep_proj = time_embed(
        latent_time_steps,
        int(latent_dict["noisy_latents"].shape[-2]),
        int(latent_dict["noisy_latents"].shape[-1]),
        hidden_states.dtype,
        False,
    )
    action_temb, action_timestep_proj = time_embed(
        action_time_steps,
        int(action_dict["noisy_latents"].shape[-2]),
        int(action_dict["noisy_latents"].shape[-1]),
        hidden_states.dtype,
        True,
    )
    temb = torch.cat([latent_temb, action_temb], dim=1)
    timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

    total_length = int(hidden_states.shape[1])
    padded_length = (128 - total_length % 128) % 128
    if padded_length > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

    attention_profile_name = normalize_attention_profile_name(input_dict.get("attention_profile_name"))
    if attention_profile_name is None and resolve_stage_attention_mode(
        config,
        stage="train",
        exact_runtime=True,
    ) == "flex":
        attention_profile_name = "chunked_temporal_exact"
    base_text_token_count = input_dict.get("base_text_token_count")
    proprio_context_token_count = int(input_dict.get("proprio_context_token_count", 0) or 0)

    exact_attention_profile = None
    if attention_profile_name in {
        "chunked_temporal_exact",
        "chunked_temporal_exact_joint",
        "chunked_temporal_exact_action_then_video",
        "chunked_temporal_exact_decoupled_same_step",
        "chunked_temporal_exact_video_noisy_to_action",
        "chunked_temporal_exact_action_noisy_to_video",
    }:
        exact_attention_profile = build_chunked_temporal_exact_attention_profile(
            latent_shape=tuple(int(dim) for dim in latent_dict["noisy_latents"].shape),
            action_shape=tuple(int(dim) for dim in action_dict["noisy_latents"].shape),
            padded_length=int(padded_length),
            chunk_size=int(input_dict["chunk_size"]),
            window_size=int(input_dict["window_size"]),
            patch_size=patch_size,
            text_token_count=int(latent_dict["text_emb"].shape[1]),
            base_text_token_count=(
                None if base_text_token_count is None else int(base_text_token_count)
            ),
            proprio_context_token_count=proprio_context_token_count,
            chunk_origin_frame=int(input_dict.get("chunk_origin_frame", 0) or 0),
            prefix_condition_frames=int(input_dict.get("prefix_condition_frames", 0) or 0),
            action_context_mask=(
                action_dict.get("actions_mask")
                if torch.is_tensor(action_dict.get("actions_mask"))
                else None
            ),
            device=hidden_states.device,
            build_dense_masks=hidden_states.device.type != "cuda",
            build_flex_masks=hidden_states.device.type == "cuda",
            current_block_coupling=chunked_temporal_exact_coupling_from_profile_name(attention_profile_name),
            preserve_video_pretrain_history=bool(
                input_dict.get("preserve_video_pretrain_history", False)
            ),
            history_stream_visibility=input_dict.get("history_stream_visibility"),
        )
    elif attention_profile_name not in (None, "none"):
        raise ValueError(
            "Exact dual-stream adapter only supports `attention_profile_name` of "
            "`None`, `none`, or a `chunked_temporal_exact*` profile, "
            f"got {attention_profile_name!r}."
        )
    elif _action_mask_has_invalid_tokens(action_dict.get("actions_mask")):
        raise ValueError(
            "Exact dual-stream training received invalid action tokens without a visibility profile. "
            "This unsafe legacy mode is deprecated because zero/invalid action tokens could be attended; "
            "use a `chunked_temporal_exact*` attention profile so `actions_mask` is applied as "
            "an action-context visibility mask."
        )

    return PreparedExactTrainSequence(
        hidden_states=hidden_states,
        text_hidden_states=text_hidden_states,
        rotary_emb=rotary_emb,
        temb=temb,
        timestep_proj=timestep_proj,
        split_list=[
            latent_hidden_states.shape[1],
            condition_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            condition_action_hidden_states.shape[1],
            padded_length,
        ],
        batch_size=batch_size,
        attention_profile=exact_attention_profile,
    )


def _action_mask_has_invalid_tokens(mask: Any) -> bool:
    if not torch.is_tensor(mask):
        return False
    if mask.numel() == 0:
        return False
    if mask.ndim == 5:
        token_valid = mask.float().amax(dim=1) > 0
    elif mask.ndim == 4:
        token_valid = mask.float() > 0
    elif mask.ndim == 3:
        token_valid = mask.float().amax(dim=-1) > 0
    elif mask.ndim == 2:
        token_valid = mask.float() > 0
    else:
        raise ValueError(f"Unsupported action visibility mask shape {tuple(mask.shape)}.")
    return bool((~token_valid).any().item())


def prepare_runtime_sequence(
    step_input: RuntimeStepInput,
    *,
    hidden_size: int | None = None,
    exact_train_preparer: Callable[[dict[str, torch.Tensor | dict[str, torch.Tensor]]], PreparedExactTrainSequence] | None = None,
) -> PreparedRuntimeSequence:
    """Resolve one runtime step into an executable backbone payload."""

    family = step_input.program.sequence_family
    if family in {"dense_default", "register_sequence"}:
        if step_input.core_input is None:
            raise ValueError(
                f"Runtime program {step_input.program.name!r} requires `core_input`."
            )
        core_input = step_input.core_input
        if family == "register_sequence":
            if hidden_size is None:
                raise ValueError("Register-sequence runtime preparation requires `hidden_size`.")
            core_input = _materialize_register_core_input(core_input, hidden_size=hidden_size)
        if (
            step_input.structured_block_semantics is not None
            or step_input.structured_frequency_bundle is not None
            or step_input.structured_attention_context is not None
        ):
            core_input = VisualCoreInput(
                tokens=core_input.tokens,
                token_layout=core_input.token_layout,
                position_context=core_input.position_context,
                timestep_context=core_input.timestep_context,
                grid_ids=core_input.grid_ids,
                timestep_values=core_input.timestep_values,
                stream_ids=core_input.stream_ids,
                text_context=core_input.text_context,
                attention_mask=core_input.attention_mask,
                attention_profile=core_input.attention_profile,
                cache_state=core_input.cache_state,
                cache_update_metadata=core_input.cache_update_metadata,
                conditioning=core_input.conditioning,
                readout_request=core_input.readout_request,
                sequence_metadata=core_input.sequence_metadata,
                register_components=core_input.register_components,
                structured_block_semantics=(
                    step_input.structured_block_semantics
                    if step_input.structured_block_semantics is not None
                    else core_input.structured_block_semantics
                ),
                structured_frequency_bundle=(
                    step_input.structured_frequency_bundle
                    if step_input.structured_frequency_bundle is not None
                    else core_input.structured_frequency_bundle
                ),
                structured_attention_context=(
                    step_input.structured_attention_context
                    if step_input.structured_attention_context is not None
                    else core_input.structured_attention_context
                ),
            )
        return PreparedRuntimeSequence(
            mode="core_input",
            core_input=core_input,
        )
    if family in {"chunked_dual_stream_exact", "chunked_dual_stream_exact_inference"}:
        if step_input.payload is None:
            raise ValueError(
                f"Runtime program {step_input.program.name!r} requires exact-train `payload`."
            )
        if exact_train_preparer is None:
            raise ValueError("Exact train runtime preparation requires `exact_train_preparer`.")
        payload = dict(step_input.payload)
        if (
            step_input.program.attention_profile_name is not None
            and payload.get("attention_profile_name") is None
        ):
            payload["attention_profile_name"] = step_input.program.attention_profile_name
        if family == "chunked_dual_stream_exact_inference":
            return PreparedRuntimeSequence(
                mode="exact_inference",
                payload=payload,
                exact_inference=exact_train_preparer(payload),
                update_cache=step_input.update_cache,
                cache_name=step_input.cache_name,
            )
        return PreparedRuntimeSequence(
            mode="exact_train",
            payload=payload,
            exact_train=exact_train_preparer(payload),
        )
    if family == "single_stream_exact":
        if step_input.payload is None:
            raise ValueError(
                f"Runtime program {step_input.program.name!r} requires exact-stream `payload`."
            )
        return PreparedRuntimeSequence(
            mode="exact_single_stream",
            payload=step_input.payload,
            update_cache=step_input.update_cache,
            cache_name=step_input.cache_name,
            action_mode=step_input.action_mode,
        )
    raise ValueError(
        f"Unsupported runtime sequence family {family!r} for program {step_input.program.name!r}."
    )
