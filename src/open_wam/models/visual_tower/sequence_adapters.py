from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from open_wam.configs.backbone import (
    SharedVideoTransformerConfig,
    resolve_stage_attention_mode,
)
from open_wam.models.common import (
    PreparedAttentionProfile,
    build_chunked_conditioned_video_attention_profile,
    build_chunked_temporal_exact_attention_profile,
    chunked_temporal_exact_coupling_from_profile_name,
    normalize_attention_profile_name,
)
from open_wam.models.common.proprio_conditioning import (
    HiddenProprioContext,
    ProprioContextGranularity,
    project_hidden_proprio_context_to_frames,
)

from .contracts import VisualCoreInput
from .runtime_programs import RuntimeSequenceFamily, RuntimeStepInput


@dataclass(frozen=True)
class PreparedExactTrainSequence:
    """Prepared packed exact inputs for the shared video backbone."""

    hidden_states: torch.Tensor
    text_hidden_states: torch.Tensor
    rotary_emb: torch.Tensor
    temb: torch.Tensor
    timestep_proj: torch.Tensor
    split_list: list[int]
    batch_size: int
    attention_profile: PreparedAttentionProfile | None
    use_activation_checkpointing: bool = False


@dataclass(frozen=True)
class _PreparedExactVideoStreams:
    payload: dict[str, Any]
    noisy_hidden_states: torch.Tensor
    condition_hidden_states: torch.Tensor
    text_hidden_states: torch.Tensor
    grid_id: torch.Tensor
    temb: torch.Tensor
    timestep_proj: torch.Tensor
    batch_size: int


@dataclass(frozen=True)
class PreparedRuntimeSequence:
    """Backbone-ready step payload resolved from a runtime program."""

    mode: str
    core_input: VisualCoreInput | None = None
    payload: dict[str, Any] | None = None
    exact_train: PreparedExactTrainSequence | None = None
    exact_inference: PreparedExactTrainSequence | None = None
    exact_conditioned_video: PreparedExactTrainSequence | None = None
    update_cache: int = 0
    cache_name: str = "open_wam_exact"
    action_mode: bool = False


def _cast_floating_payload(
    payload: Mapping[str, Any], *, dtype: torch.dtype
) -> dict[str, Any]:
    return {
        key: value.to(dtype)
        if torch.is_tensor(value) and torch.is_floating_point(value)
        else value
        for key, value in payload.items()
    }


def _prepare_exact_video_streams(
    latent_payload: Mapping[str, Any],
    *,
    model_dtype: torch.dtype,
    input_embed: Callable[[torch.Tensor, str], torch.Tensor],
    exact_text_hidden_states: Callable[[torch.Tensor], torch.Tensor],
    time_embed: Callable[
        [torch.Tensor, int, int, torch.dtype, bool], tuple[torch.Tensor, torch.Tensor]
    ],
) -> _PreparedExactVideoStreams:
    """Embed the two video streams shared by exact VTA and video-only runtime."""

    latent_dict = _cast_floating_payload(latent_payload, dtype=model_dtype)
    noisy_latents = latent_dict.get("noisy_latents")
    condition_latents = latent_dict.get("latent")
    text_emb = latent_dict.get("text_emb")
    grid_id = latent_dict.get("grid_id")
    timesteps = latent_dict.get("timesteps")
    condition_timesteps = latent_dict.get("cond_timesteps")
    required = {
        "noisy_latents": noisy_latents,
        "latent": condition_latents,
        "text_emb": text_emb,
        "grid_id": grid_id,
        "timesteps": timesteps,
        "cond_timesteps": condition_timesteps,
    }
    missing = [name for name, value in required.items() if not torch.is_tensor(value)]
    if missing:
        raise TypeError(
            "Exact video streams require tensor payload fields: "
            + ", ".join(sorted(missing))
            + "."
        )
    assert isinstance(noisy_latents, torch.Tensor)
    assert isinstance(condition_latents, torch.Tensor)
    assert isinstance(text_emb, torch.Tensor)
    assert isinstance(grid_id, torch.Tensor)
    assert isinstance(timesteps, torch.Tensor)
    assert isinstance(condition_timesteps, torch.Tensor)
    if tuple(noisy_latents.shape) != tuple(condition_latents.shape):
        raise ValueError(
            "Exact video noisy and condition streams must have identical shapes, "
            f"got noisy={tuple(noisy_latents.shape)}, "
            f"condition={tuple(condition_latents.shape)}."
        )

    noisy_hidden_states = (
        input_embed(noisy_latents, "latent")
        .flatten(0, 1)
        .contiguous()[None]
        .clone()
    )
    condition_hidden_states = (
        input_embed(condition_latents, "latent")
        .flatten(0, 1)
        .contiguous()[None]
        .clone()
    )
    text_hidden_states = (
        exact_text_hidden_states(text_emb)
        .flatten(0, 1)
        .contiguous()[None]
        .clone()
    )
    packed_grid_id = (
        grid_id.permute(1, 0, 2).flatten(1).contiguous()[None].clone()
    )
    packed_timesteps = (
        torch.cat(
            [timesteps.flatten(0, 1), condition_timesteps.flatten(0, 1)],
            dim=0,
        )
        .contiguous()[None]
        .clone()
    )
    temb, timestep_proj = time_embed(
        packed_timesteps,
        int(noisy_latents.shape[-2]),
        int(noisy_latents.shape[-1]),
        noisy_hidden_states.dtype,
        False,
    )
    return _PreparedExactVideoStreams(
        payload=latent_dict,
        noisy_hidden_states=noisy_hidden_states,
        condition_hidden_states=condition_hidden_states,
        text_hidden_states=text_hidden_states,
        grid_id=packed_grid_id,
        temb=temb,
        timestep_proj=timestep_proj,
        batch_size=int(noisy_latents.shape[0]),
    )


def _pad_exact_sequence(
    *,
    hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    temb: torch.Tensor,
    timestep_proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    padded_length = (128 - int(hidden_states.shape[1]) % 128) % 128
    if padded_length > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))
    return hidden_states, rotary_emb, temb, timestep_proj, padded_length


def _apply_packed_video_action_proprio_context(
    *,
    hidden_states: torch.Tensor,
    stream_lengths: Sequence[int],
    payload: Mapping[str, Any],
    patch_size: tuple[int, int, int],
    encode_context: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Add boundary-aligned state context to a packed video/action sequence."""

    proprio_state = payload.get("per_chunk_proprio_state")
    if proprio_state is None:
        return hidden_states
    if not isinstance(proprio_state, torch.Tensor):
        raise TypeError("`per_chunk_proprio_state` must be a tensor.")
    latent_payload = payload.get("latent_dict")
    action_payload = payload.get("action_dict")
    if not isinstance(latent_payload, Mapping) or not isinstance(
        action_payload, Mapping
    ):
        raise TypeError(
            "Packed proprio context requires `latent_dict` and `action_dict` payloads."
        )
    noisy_video = latent_payload.get("noisy_latents")
    noisy_action = action_payload.get("noisy_latents")
    if not isinstance(noisy_video, torch.Tensor) or not isinstance(
        noisy_action, torch.Tensor
    ):
        raise TypeError(
            "Packed proprio context requires tensor `noisy_latents` for both streams."
        )
    if len(stream_lengths) < 4:
        raise ValueError(
            "Packed proprio context requires video-noisy, video-condition, "
            "action-noisy, and action-condition stream lengths."
        )

    latent_shape = tuple(int(dim) for dim in noisy_video.shape)
    action_shape = tuple(int(dim) for dim in noisy_action.shape)
    batch_size, _, latent_frames, latent_height, latent_width = latent_shape
    action_batch, _, action_frames, action_height, action_width = action_shape
    if batch_size != action_batch:
        raise ValueError(
            "Packed proprio context expects matching video/action batches, "
            f"got {batch_size} and {action_batch}."
        )
    if proprio_state.ndim != 3 or int(proprio_state.shape[0]) != batch_size:
        raise ValueError(
            "Packed proprio context expects state shape "
            "[B, frames_or_chunks, state_dim], "
            f"got {tuple(proprio_state.shape)} for batch_size={batch_size}."
        )

    chunk_size = max(1, int(payload["chunk_size"]))
    chunk_origin_frame = int(payload.get("chunk_origin_frame", 0) or 0)
    granularity_value = payload.get(
        "per_chunk_proprio_state_granularity",
        ProprioContextGranularity.CHUNK,
    )
    try:
        granularity = ProprioContextGranularity(granularity_value)
    except ValueError as exc:
        raise ValueError(
            "`per_chunk_proprio_state_granularity` must be `frame` or `chunk`, "
            f"got {granularity_value!r}."
        ) from exc
    prefix_frames = max(0, int(payload.get("prefix_condition_frames", 0) or 0))
    boundary_state = project_hidden_proprio_context_to_frames(
        HiddenProprioContext(values=proprio_state, granularity=granularity),
        num_frames=latent_frames,
        chunk_size=chunk_size,
        chunk_origin_frame=chunk_origin_frame,
        prefix_frames=prefix_frames,
    )

    chunk_context = encode_context(
        boundary_state,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    patch_t, patch_h, patch_w = (int(value) for value in patch_size)
    video_frames = latent_frames // patch_t
    if patch_t != 1:
        chunk_context = chunk_context[:, ::patch_t, :]
    video_tokens_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    action_tokens_per_frame = action_height * action_width
    if video_frames != action_frames + prefix_frames:
        raise ValueError(
            "Packed proprio context expects patchified video frames to equal "
            "action frames plus prefix frames, "
            f"got video_frames={video_frames}, action_frames={action_frames}, "
            f"prefix_frames={prefix_frames}."
        )
    video_context = chunk_context.repeat_interleave(
        video_tokens_per_frame,
        dim=1,
    )
    action_context = (
        chunk_context[:, prefix_frames:, :] if prefix_frames > 0 else chunk_context
    ).repeat_interleave(action_tokens_per_frame, dim=1)
    if hidden_states.shape[0] == 1:
        video_context = rearrange(video_context, "b l c -> 1 (b l) c")
        action_context = rearrange(action_context, "b l c -> 1 (b l) c")
    elif hidden_states.shape[0] != batch_size:
        raise ValueError(
            "Unexpected packed hidden-state batch layout: expected leading "
            f"dimension 1 or {batch_size}, got {hidden_states.shape[0]}."
        )

    video_noisy_len, video_condition_len, action_noisy_len, action_condition_len = (
        int(stream_lengths[index]) for index in range(4)
    )
    if (
        int(video_context.shape[1]) != video_noisy_len
        or int(action_context.shape[1]) != action_noisy_len
    ):
        raise ValueError(
            "Packed proprio context does not match stream token lengths: "
            f"video_context={tuple(video_context.shape)}, "
            f"action_context={tuple(action_context.shape)}, "
            f"stream_lengths={tuple(int(value) for value in stream_lengths)}."
        )

    output = hidden_states.clone()
    if bool(payload.get("per_chunk_proprio_apply_to_video", True)):
        output[:, :video_noisy_len, :] += video_context
        if video_condition_len > 0:
            if video_condition_len != video_noisy_len:
                raise ValueError(
                    "Packed proprio video streams must have equal token lengths, "
                    f"got noisy={video_noisy_len}, condition={video_condition_len}."
                )
            output[:, video_noisy_len : video_noisy_len + video_condition_len, :] += (
                video_context
            )
    action_start = video_noisy_len + video_condition_len
    output[:, action_start : action_start + action_noisy_len, :] += action_context
    if action_condition_len > 0:
        if action_condition_len != action_noisy_len:
            raise ValueError(
                "Packed proprio action streams must have equal token lengths, "
                f"got noisy={action_noisy_len}, condition={action_condition_len}."
            )
        condition_start = action_start + action_noisy_len
        output[:, condition_start : condition_start + action_condition_len, :] += (
            action_context
        )
    return output


def prepare_exact_dual_stream_train_sequence(
    input_dict: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    *,
    config: SharedVideoTransformerConfig,
    patch_size: tuple[int, int, int],
    model_dtype: torch.dtype,
    input_embed: Callable[[torch.Tensor, str], torch.Tensor],
    exact_text_hidden_states: Callable[[torch.Tensor], torch.Tensor],
    time_embed: Callable[
        [torch.Tensor, int, int, torch.dtype, bool], tuple[torch.Tensor, torch.Tensor]
    ],
    rope: Callable[[torch.Tensor], torch.Tensor],
    encode_proprio_context: Callable[..., torch.Tensor] | None = None,
) -> PreparedExactTrainSequence:
    latent_dict = input_dict["latent_dict"]
    action_dict = input_dict["action_dict"]
    assert isinstance(latent_dict, dict)
    assert isinstance(action_dict, dict)

    video = _prepare_exact_video_streams(
        latent_dict,
        model_dtype=model_dtype,
        input_embed=input_embed,
        exact_text_hidden_states=exact_text_hidden_states,
        time_embed=time_embed,
    )
    latent_dict = video.payload
    action_dict = _cast_floating_payload(action_dict, dtype=model_dtype)
    batch_size = video.batch_size
    action_hidden_states = (
        input_embed(action_dict["noisy_latents"], "action")
        .flatten(0, 1)
        .contiguous()[None]
        .clone()
    )
    condition_action_hidden_states = (
        input_embed(action_dict["latent"], "action")
        .flatten(0, 1)
        .contiguous()[None]
        .clone()
    )

    hidden_states = torch.cat(
        [
            video.noisy_hidden_states,
            video.condition_hidden_states,
            action_hidden_states,
            condition_action_hidden_states,
        ],
        dim=1,
    )
    action_grid_id = (
        action_dict["grid_id"].permute(1, 0, 2).flatten(1).contiguous()[None].clone()
    )
    full_grid_id = torch.cat([video.grid_id] * 2 + [action_grid_id] * 2, dim=2)
    rotary_emb = rope(full_grid_id)[:, :, None]

    action_time_steps = (
        torch.cat(
            [
                action_dict["timesteps"].flatten(0, 1),
                action_dict["cond_timesteps"].flatten(0, 1),
            ],
            dim=0,
        )
        .contiguous()[None]
        .clone()
    )
    action_temb, action_timestep_proj = time_embed(
        action_time_steps,
        int(action_dict["noisy_latents"].shape[-2]),
        int(action_dict["noisy_latents"].shape[-1]),
        hidden_states.dtype,
        True,
    )
    temb = torch.cat([video.temb, action_temb], dim=1)
    timestep_proj = torch.cat([video.timestep_proj, action_timestep_proj], dim=1)

    hidden_states, rotary_emb, temb, timestep_proj, padded_length = (
        _pad_exact_sequence(
            hidden_states=hidden_states,
            rotary_emb=rotary_emb,
            temb=temb,
            timestep_proj=timestep_proj,
        )
    )
    stream_lengths = [
        int(video.noisy_hidden_states.shape[1]),
        int(video.condition_hidden_states.shape[1]),
        int(action_hidden_states.shape[1]),
        int(condition_action_hidden_states.shape[1]),
        int(padded_length),
    ]

    if input_dict.get("per_chunk_proprio_state") is not None:
        if encode_proprio_context is None:
            raise ValueError(
                "Packed proprio conditioning requires a visual-core state encoder."
            )
        hidden_states = _apply_packed_video_action_proprio_context(
            hidden_states=hidden_states,
            stream_lengths=stream_lengths,
            payload=input_dict,
            patch_size=patch_size,
            encode_context=encode_proprio_context,
        )

    attention_profile_name = normalize_attention_profile_name(
        input_dict.get("attention_profile_name")
    )
    if (
        attention_profile_name is None
        and resolve_stage_attention_mode(
            config,
            stage="train",
            exact_runtime=True,
        )
        == "flex"
    ):
        attention_profile_name = "chunked_temporal_exact"
    base_text_token_count = input_dict.get("base_text_token_count")
    proprio_context_token_count = int(
        input_dict.get("proprio_context_token_count", 0) or 0
    )

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
            prefix_condition_frames=int(
                input_dict.get("prefix_condition_frames", 0) or 0
            ),
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
            build_dense_masks=hidden_states.device.type != "cuda",
            build_flex_masks=hidden_states.device.type == "cuda",
            current_block_coupling=chunked_temporal_exact_coupling_from_profile_name(
                attention_profile_name
            ),
            history_stream_visibility=input_dict.get("history_stream_visibility"),
            conditional_history_policy=input_dict.get("conditional_history_policy"),
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
        text_hidden_states=video.text_hidden_states,
        rotary_emb=rotary_emb,
        temb=temb,
        timestep_proj=timestep_proj,
        split_list=stream_lengths,
        batch_size=batch_size,
        attention_profile=exact_attention_profile,
    )


def prepare_exact_conditioned_video_sequence(
    input_dict: dict[str, Any],
    *,
    config: SharedVideoTransformerConfig,
    patch_size: tuple[int, int, int],
    model_dtype: torch.dtype,
    input_embed: Callable[[torch.Tensor, str], torch.Tensor],
    exact_text_hidden_states: Callable[[torch.Tensor], torch.Tensor],
    time_embed: Callable[
        [torch.Tensor, int, int, torch.dtype, bool], tuple[torch.Tensor, torch.Tensor]
    ],
    rope: Callable[[torch.Tensor], torch.Tensor],
) -> PreparedExactTrainSequence:
    """Prepare native conditioned-video execution without an action stream."""

    if "action_dict" in input_dict:
        raise ValueError(
            "Conditioned-video runtime does not accept action stream payloads."
        )
    latent_payload = input_dict.get("latent_dict")
    if not isinstance(latent_payload, Mapping):
        raise TypeError(
            "Conditioned-video runtime requires a mapping `latent_dict` payload."
        )
    video = _prepare_exact_video_streams(
        latent_payload,
        model_dtype=model_dtype,
        input_embed=input_embed,
        exact_text_hidden_states=exact_text_hidden_states,
        time_embed=time_embed,
    )
    hidden_states = torch.cat(
        [video.noisy_hidden_states, video.condition_hidden_states], dim=1
    )
    rotary_emb = rope(torch.cat([video.grid_id] * 2, dim=2))[:, :, None]
    hidden_states, rotary_emb, temb, timestep_proj, padded_length = (
        _pad_exact_sequence(
            hidden_states=hidden_states,
            rotary_emb=rotary_emb,
            temb=video.temb,
            timestep_proj=video.timestep_proj,
        )
    )
    stream_lengths = [
        int(video.noisy_hidden_states.shape[1]),
        int(video.condition_hidden_states.shape[1]),
        int(padded_length),
    ]
    runtime_stage = str(input_dict.get("stage", "train"))
    if runtime_stage not in {"train", "infer"}:
        raise ValueError(
            "Conditioned-video runtime stage must be `train` or `infer`, "
            f"got {runtime_stage!r}."
        )
    attention_mode = resolve_stage_attention_mode(
        config,
        stage=runtime_stage,
        exact_runtime=True,
    )
    profile = build_chunked_conditioned_video_attention_profile(
        latent_shape=tuple(
            int(dim) for dim in video.payload["noisy_latents"].shape
        ),
        padded_length=int(padded_length),
        chunk_size=int(input_dict["chunk_size"]),
        window_size=int(input_dict["window_size"]),
        patch_size=patch_size,
        text_token_count=int(video.payload["text_emb"].shape[1]),
        chunk_origin_frame=int(input_dict.get("chunk_origin_frame", 0) or 0),
        prefix_condition_frames=int(
            input_dict.get("prefix_condition_frames", 0) or 0
        ),
        singleton_chunk_frame=(
            None
            if input_dict.get("singleton_chunk_frame") is None
            else int(input_dict["singleton_chunk_frame"])
        ),
        conditional_history_policy=input_dict.get("conditional_history_policy"),
        device=hidden_states.device,
        build_dense_masks=attention_mode != "flex" or hidden_states.device.type != "cuda",
        build_flex_masks=(
            attention_mode == "flex" and hidden_states.device.type == "cuda"
        ),
    )
    return PreparedExactTrainSequence(
        hidden_states=hidden_states,
        text_hidden_states=video.text_hidden_states,
        rotary_emb=rotary_emb,
        temb=temb,
        timestep_proj=timestep_proj,
        split_list=stream_lengths,
        batch_size=video.batch_size,
        attention_profile=profile,
        use_activation_checkpointing=(
            runtime_stage == "train"
            and bool(input_dict.get("use_activation_checkpointing", False))
        ),
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
        raise ValueError(
            f"Unsupported action visibility mask shape {tuple(mask.shape)}."
        )
    return bool((~token_valid).any().item())


def prepare_runtime_sequence(
    step_input: RuntimeStepInput,
    *,
    exact_train_preparer: Callable[
        [dict[str, torch.Tensor | dict[str, torch.Tensor]]], PreparedExactTrainSequence
    ]
    | None = None,
    conditioned_video_preparer: Callable[
        [dict[str, Any]], PreparedExactTrainSequence
    ]
    | None = None,
) -> PreparedRuntimeSequence:
    """Resolve one runtime step into an executable backbone payload."""

    family = step_input.program.sequence_family
    if family is RuntimeSequenceFamily.DENSE:
        if step_input.core_input is None:
            raise ValueError(
                f"Runtime program {step_input.program.name!r} requires `core_input`."
            )
        return PreparedRuntimeSequence(
            mode="core_input",
            core_input=step_input.core_input,
        )
    if family in {
        RuntimeSequenceFamily.CHUNKED_DUAL_STREAM_TRAIN,
        RuntimeSequenceFamily.CHUNKED_DUAL_STREAM_INFERENCE,
    }:
        if step_input.payload is None:
            raise ValueError(
                f"Runtime program {step_input.program.name!r} requires exact-train `payload`."
            )
        if exact_train_preparer is None:
            raise ValueError(
                "Exact train runtime preparation requires `exact_train_preparer`."
            )
        payload = dict(step_input.payload)
        if (
            step_input.program.attention_profile_name is not None
            and payload.get("attention_profile_name") is None
        ):
            payload["attention_profile_name"] = (
                step_input.program.attention_profile_name
            )
        if family is RuntimeSequenceFamily.CHUNKED_DUAL_STREAM_INFERENCE:
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
    if family is RuntimeSequenceFamily.CHUNKED_CONDITIONED_VIDEO:
        if step_input.payload is None:
            raise ValueError(
                f"Runtime program {step_input.program.name!r} requires a video payload."
            )
        if conditioned_video_preparer is None:
            raise ValueError(
                "Conditioned-video runtime preparation requires "
                "`conditioned_video_preparer`."
            )
        payload = dict(step_input.payload)
        return PreparedRuntimeSequence(
            mode="exact_conditioned_video",
            payload=payload,
            exact_conditioned_video=conditioned_video_preparer(payload),
        )
    if family is RuntimeSequenceFamily.SINGLE_STREAM:
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
