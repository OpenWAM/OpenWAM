from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contracts import StructuredAttentionContext


@dataclass(frozen=True)
class StructuredAttentionExecutionPlan:
    """Resolved block-local execution plan for structured attention modes.

    Structured variants still execute on the shared backbone blocks. This plan
    records the extra role-aware attention semantics those blocks should follow
    after the runtime-program layer has already resolved layout, frequencies,
    and cache visibility.
    """

    mode: str
    attention_kernel: str
    cache_kernel: str
    attention_mask: torch.Tensor | None
    rotary_freqs: torch.Tensor | None
    cached_prefix_visibility: torch.Tensor | None
    use_full_cached_prefix: bool
    cached_prefix_len: int
    cached_segment_lengths: tuple[int, ...]
    clean_prefix_span: tuple[int, int]
    video_span: tuple[int, int]
    action_span: tuple[int, int]
    state_span: tuple[int, int]


def _compose_structured_attention_freqs(
    context: StructuredAttentionContext | None,
) -> torch.Tensor | None:
    if context is None or context.mode == "none":
        return None
    frequency_chunks: list[torch.Tensor] = []
    if context.clean_prefix_length > 0:
        if context.clean_prefix_freqs is None:
            raise ValueError(
                "Structured attention context requires `clean_prefix_freqs` when a clean prefix is present."
            )
        if context.clean_prefix_freqs.shape[1] != context.clean_prefix_length:
            raise ValueError(
                "Structured clean-prefix frequency length mismatch: expected "
                f"{context.clean_prefix_length}, got {context.clean_prefix_freqs.shape[1]}."
            )
        frequency_chunks.append(context.clean_prefix_freqs)
    if context.video_token_length > 0:
        if context.video_freqs is None:
            raise ValueError(
                "Structured attention context requires `video_freqs` when video tokens are present."
            )
        if context.video_freqs.shape[1] != context.video_token_length:
            raise ValueError(
                "Structured video frequency length mismatch: expected "
                f"{context.video_token_length}, got {context.video_freqs.shape[1]}."
            )
        frequency_chunks.append(context.video_freqs)
    if context.action_register_length > 0:
        if context.action_freqs is None:
            raise ValueError(
                "Structured attention context requires `action_freqs` when action registers are present."
            )
        if context.action_freqs.shape[1] != context.action_register_length:
            raise ValueError(
                "Structured action-register frequency length mismatch: expected "
                f"{context.action_register_length}, got {context.action_freqs.shape[1]}."
            )
        frequency_chunks.append(context.action_freqs)
    if context.state_register_length > 0:
        if context.state_freqs is None:
            raise ValueError(
                "Structured attention context requires `state_freqs` when state registers are present."
            )
        if context.state_freqs.shape[1] != context.state_register_length:
            raise ValueError(
                "Structured state-register frequency length mismatch: expected "
                f"{context.state_register_length}, got {context.state_freqs.shape[1]}."
            )
        frequency_chunks.append(context.state_freqs)
    if not frequency_chunks:
        return None
    return torch.cat(frequency_chunks, dim=1)


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.shape[1] == 0:
        return query.new_zeros(query.shape)
    query_t = query.transpose(1, 2)
    key_t = key.transpose(1, 2)
    value_t = value.transpose(1, 2)
    mask = None
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            mask = attention_mask[None, None, :, :]
        elif attention_mask.ndim == 3:
            mask = attention_mask[:, None, :, :]
        else:
            mask = attention_mask
        mask = mask.to(device=query.device)
    output = F.scaled_dot_product_attention(query_t, key_t, value_t, attn_mask=mask)
    return output.transpose(1, 2)


def _build_causal_rectangular_mask(
    query_len: int,
    key_len: int,
    *,
    device: torch.device,
    query_offset: int = 0,
) -> torch.Tensor:
    query_positions = torch.arange(query_len, device=device) + int(query_offset)
    key_positions = torch.arange(key_len, device=device)
    return key_positions[None, :] <= query_positions[:, None]


def _execute_branchwise_clean_image_attention(
    clean_query: torch.Tensor,
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
) -> torch.Tensor:
    # Clean prefix tokens act as a stable causal image stream. They provide the
    # trusted teacher-forcing context that later noisy blocks can attend back to.
    clean_len = clean_query.shape[1]
    if clean_len == 0:
        return clean_query.new_zeros(clean_query.shape)
    causal_mask = _build_causal_rectangular_mask(
        clean_len,
        clean_key.shape[1],
        device=clean_query.device,
    )
    return _scaled_dot_product_attention(
        clean_query,
        clean_key,
        clean_value,
        attention_mask=causal_mask,
    )


def _execute_branchwise_state_attention(
    state_query: torch.Tensor,
    state_key: torch.Tensor,
    state_value: torch.Tensor,
    *,
    tokens_per_block: int,
) -> torch.Tensor:
    # State registers stay local to their aligned block instead of sharing the
    # broader image/action context.
    if state_query.shape[1] == 0:
        return state_query.new_zeros(state_query.shape)
    if tokens_per_block <= 0:
        return _scaled_dot_product_attention(state_query, state_key, state_value)
    output = torch.empty_like(state_query)
    num_blocks = state_query.shape[1] // tokens_per_block
    for block_index in range(num_blocks):
        start = block_index * tokens_per_block
        end = start + tokens_per_block
        output[:, start:end] = _scaled_dot_product_attention(
            state_query[:, start:end],
            state_key[:, start:end],
            state_value[:, start:end],
        )
    return output


def _execute_branchwise_noisy_image_attention(
    noisy_query: torch.Tensor,
    noisy_key: torch.Tensor,
    noisy_value: torch.Tensor,
    *,
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    action_key: torch.Tensor,
    action_value: torch.Tensor,
    state_key: torch.Tensor,
    state_value: torch.Tensor,
    tokens_per_frame: int,
    tokens_per_video_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    num_video_blocks: int,
) -> torch.Tensor:
    # Each noisy image block sees accumulated clean-image context plus only its
    # aligned noisy-image/action/state block, mirroring the explicit branchwise
    # decomposition from DreamZero-style structured attention.
    output = torch.empty_like(noisy_query)
    first_frame_len = min(tokens_per_frame, noisy_query.shape[1])
    if first_frame_len > 0:
        output[:, :first_frame_len] = _scaled_dot_product_attention(
            noisy_query[:, :first_frame_len],
            noisy_key[:, :first_frame_len],
            noisy_value[:, :first_frame_len],
        )
    for block_index in range(num_video_blocks):
        block_start = first_frame_len + block_index * tokens_per_video_block
        block_end = min(block_start + tokens_per_video_block, noisy_query.shape[1])
        if block_end <= block_start:
            continue
        clean_end = min(tokens_per_frame + block_index * tokens_per_video_block, clean_key.shape[1])
        action_start = block_index * num_action_per_block
        action_end = min(action_start + num_action_per_block, action_key.shape[1])
        state_start = block_index * num_state_per_block
        state_end = min(state_start + num_state_per_block, state_key.shape[1])
        context_key = torch.cat(
            [
                clean_key[:, :clean_end],
                noisy_key[:, block_start:block_end],
                action_key[:, action_start:action_end],
                state_key[:, state_start:state_end],
            ],
            dim=1,
        )
        context_value = torch.cat(
            [
                clean_value[:, :clean_end],
                noisy_value[:, block_start:block_end],
                action_value[:, action_start:action_end],
                state_value[:, state_start:state_end],
            ],
            dim=1,
        )
        output[:, block_start:block_end] = _scaled_dot_product_attention(
            noisy_query[:, block_start:block_end],
            context_key,
            context_value,
        )
    return output


def _execute_branchwise_noisy_action_attention(
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    action_value: torch.Tensor,
    *,
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    noisy_video_key: torch.Tensor,
    noisy_video_value: torch.Tensor,
    state_key: torch.Tensor,
    state_value: torch.Tensor,
    tokens_per_frame: int,
    tokens_per_video_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    num_video_blocks: int,
) -> torch.Tensor:
    # Action blocks are processed separately from image blocks so they can use
    # their own context mix instead of behaving like generic tail tokens.
    if action_query.shape[1] == 0:
        return action_query.new_zeros(action_query.shape)
    output = torch.empty_like(action_query)
    for block_index in range(num_video_blocks):
        action_start = block_index * num_action_per_block
        action_end = min(action_start + num_action_per_block, action_query.shape[1])
        if action_end <= action_start:
            continue
        clean_end = min(tokens_per_frame + block_index * tokens_per_video_block, clean_key.shape[1])
        noisy_start = min(tokens_per_frame + block_index * tokens_per_video_block, noisy_video_key.shape[1])
        noisy_end = min(noisy_start + tokens_per_video_block, noisy_video_key.shape[1])
        state_start = block_index * num_state_per_block
        state_end = min(state_start + num_state_per_block, state_key.shape[1])
        context_key = torch.cat(
            [
                clean_key[:, :clean_end],
                noisy_video_key[:, noisy_start:noisy_end],
                action_key[:, action_start:action_end],
                state_key[:, state_start:state_end],
            ],
            dim=1,
        )
        context_value = torch.cat(
            [
                clean_value[:, :clean_end],
                noisy_video_value[:, noisy_start:noisy_end],
                action_value[:, action_start:action_end],
                state_value[:, state_start:state_end],
            ],
            dim=1,
        )
        output[:, action_start:action_end] = _scaled_dot_product_attention(
            action_query[:, action_start:action_end],
            context_key,
            context_value,
        )
    return output


def _execute_branchwise_rollout_video_attention(
    video_query: torch.Tensor,
    video_key: torch.Tensor,
    video_value: torch.Tensor,
    *,
    cached_video_key: torch.Tensor,
    cached_video_value: torch.Tensor,
    action_key: torch.Tensor,
    action_value: torch.Tensor,
    state_key: torch.Tensor,
    state_value: torch.Tensor,
    tokens_per_video_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    num_video_blocks: int,
) -> torch.Tensor:
    if video_query.shape[1] == 0:
        return video_query.new_zeros(video_query.shape)
    output = torch.empty_like(video_query)
    for block_index in range(max(num_video_blocks, 1)):
        video_start = block_index * tokens_per_video_block
        video_end = min(video_start + tokens_per_video_block, video_query.shape[1])
        if video_end <= video_start:
            continue
        action_start = block_index * num_action_per_block
        action_end = min(action_start + num_action_per_block, action_key.shape[1])
        state_start = block_index * num_state_per_block
        state_end = min(state_start + num_state_per_block, state_key.shape[1])
        context_key = torch.cat(
            [
                cached_video_key,
                video_key,
                action_key[:, action_start:action_end],
                state_key[:, state_start:state_end],
            ],
            dim=1,
        )
        context_value = torch.cat(
            [
                cached_video_value,
                video_value,
                action_value[:, action_start:action_end],
                state_value[:, state_start:state_end],
            ],
            dim=1,
        )
        output[:, video_start:video_end] = _scaled_dot_product_attention(
            video_query[:, video_start:video_end],
            context_key,
            context_value,
        )
    return output


def _execute_branchwise_rollout_action_attention(
    action_query: torch.Tensor,
    action_key: torch.Tensor,
    action_value: torch.Tensor,
    *,
    cached_video_key: torch.Tensor,
    cached_video_value: torch.Tensor,
    video_key: torch.Tensor,
    video_value: torch.Tensor,
    state_key: torch.Tensor,
    state_value: torch.Tensor,
    num_action_per_block: int,
    num_state_per_block: int,
    num_video_blocks: int,
    num_action_blocks: int,
) -> torch.Tensor:
    if action_query.shape[1] == 0:
        return action_query.new_zeros(action_query.shape)
    output = torch.empty_like(action_query)
    for block_index in range(max(num_action_blocks, 1)):
        action_start = block_index * num_action_per_block
        action_end = min(action_start + num_action_per_block, action_query.shape[1])
        if action_end <= action_start:
            continue
        state_start = block_index * num_state_per_block
        state_end = min(state_start + num_state_per_block, state_key.shape[1])
        context_key = torch.cat(
            [
                cached_video_key,
                video_key,
                action_key[:, action_start:action_end],
                state_key[:, state_start:state_end],
            ],
            dim=1,
        )
        context_value = torch.cat(
            [
                cached_video_value,
                video_value,
                action_value[:, action_start:action_end],
                state_value[:, state_start:state_end],
            ],
            dim=1,
        )
        output[:, action_start:action_end] = _scaled_dot_product_attention(
            action_query[:, action_start:action_end],
            context_key,
            context_value,
        )
    return output


def execute_structured_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    context: StructuredAttentionContext | None,
    plan: StructuredAttentionExecutionPlan | None,
    cached_key_value=None,
) -> torch.Tensor | None:
    if context is None or plan is None or context.mode != "register_explicit":
        return None

    clean_start, clean_end = plan.clean_prefix_span
    video_start, video_end = plan.video_span
    action_start, action_end = plan.action_span
    state_start, state_end = plan.state_span

    clean_query = query[:, clean_start:clean_end]
    clean_key = key[:, clean_start:clean_end]
    clean_value = value[:, clean_start:clean_end]
    noisy_video_query = query[:, video_start:video_end]
    noisy_video_key = key[:, video_start:video_end]
    noisy_video_value = value[:, video_start:video_end]
    action_query = query[:, action_start:action_end]
    action_key = key[:, action_start:action_end]
    action_value = value[:, action_start:action_end]
    state_query = query[:, state_start:state_end]
    state_key = key[:, state_start:state_end]
    state_value = value[:, state_start:state_end]

    if (
        context.attention_kernel == "branchwise_explicit"
        and context.teacher_forcing_enabled
        and context.clean_prefix_length > 0
    ):
        clean_output = _execute_branchwise_clean_image_attention(clean_query, clean_key, clean_value)
        noisy_video_output = _execute_branchwise_noisy_image_attention(
            noisy_video_query,
            noisy_video_key,
            noisy_video_value,
            clean_key=clean_key,
            clean_value=clean_value,
            action_key=action_key,
            action_value=action_value,
            state_key=state_key,
            state_value=state_value,
            tokens_per_frame=context.tokens_per_frame,
            tokens_per_video_block=context.tokens_per_video_block,
            num_action_per_block=context.num_action_per_block,
            num_state_per_block=context.num_state_per_block,
            num_video_blocks=context.num_video_blocks,
        )
        action_output = _execute_branchwise_noisy_action_attention(
            action_query,
            action_key,
            action_value,
            clean_key=clean_key,
            clean_value=clean_value,
            noisy_video_key=noisy_video_key,
            noisy_video_value=noisy_video_value,
            state_key=state_key,
            state_value=state_value,
            tokens_per_frame=context.tokens_per_frame,
            tokens_per_video_block=context.tokens_per_video_block,
            num_action_per_block=context.num_action_per_block,
            num_state_per_block=context.num_state_per_block,
            num_video_blocks=context.num_video_blocks,
        )
        state_output = _execute_branchwise_state_attention(
            state_query,
            state_key,
            state_value,
            tokens_per_block=context.num_state_per_block,
        )
        return torch.cat([clean_output, noisy_video_output, action_output, state_output], dim=1)

    if (
        context.cache_kernel != "branchwise_rollout_explicit"
        or context.teacher_forcing_enabled
        or cached_key_value is None
        or cached_key_value.key is None
        or cached_key_value.value is None
    ):
        return None

    cached_video_key = cached_key_value.key.to(device=query.device, dtype=query.dtype).transpose(1, 2)
    cached_video_value = cached_key_value.value.to(device=query.device, dtype=value.dtype).transpose(1, 2)
    video_output = _execute_branchwise_rollout_video_attention(
        noisy_video_query,
        noisy_video_key,
        noisy_video_value,
        cached_video_key=cached_video_key,
        cached_video_value=cached_video_value,
        action_key=action_key,
        action_value=action_value,
        state_key=state_key,
        state_value=state_value,
        tokens_per_video_block=context.tokens_per_video_block,
        num_action_per_block=context.num_action_per_block,
        num_state_per_block=context.num_state_per_block,
        num_video_blocks=context.num_video_blocks,
    )
    action_output = _execute_branchwise_rollout_action_attention(
        action_query,
        action_key,
        action_value,
        cached_video_key=cached_video_key,
        cached_video_value=cached_video_value,
        video_key=noisy_video_key,
        video_value=noisy_video_value,
        state_key=state_key,
        state_value=state_value,
        num_action_per_block=context.num_action_per_block,
        num_state_per_block=context.num_state_per_block,
        num_video_blocks=context.num_video_blocks,
        num_action_blocks=context.num_action_blocks,
    )
    state_output = _execute_branchwise_state_attention(
        state_query,
        state_key,
        state_value,
        tokens_per_block=context.num_state_per_block,
    )
    if clean_end > clean_start:
        clean_output = clean_query.new_zeros(clean_query.shape)
        return torch.cat([clean_output, video_output, action_output, state_output], dim=1)
    return torch.cat([video_output, action_output, state_output], dim=1)


def _build_structured_register_attention_mask(
    context: StructuredAttentionContext | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if context is None or context.mode != "register_explicit":
        return None

    seq_len = (
        context.clean_prefix_length
        + context.video_token_length
        + context.action_register_length
        + context.state_register_length
    )
    if seq_len <= 0:
        return None

    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)

    clean_start = 0
    clean_end = context.clean_prefix_length
    video_start = clean_end
    first_noisy_frame_end = video_start + min(context.tokens_per_frame, context.video_token_length)

    if context.clean_prefix_length > 0:
        clean_len = clean_end - clean_start
        mask[clean_start:clean_end, clean_start:clean_end] = torch.tril(
            torch.ones(clean_len, clean_len, device=device, dtype=torch.bool)
        )
        if first_noisy_frame_end > video_start:
            mask[video_start:first_noisy_frame_end, video_start:first_noisy_frame_end] = True
    elif first_noisy_frame_end > video_start:
        mask[video_start:first_noisy_frame_end, video_start:first_noisy_frame_end] = True

    action_start = video_start + context.video_token_length
    state_start = action_start + context.action_register_length

    video_block_len = context.tokens_per_video_block
    action_block_len = context.num_action_per_block
    state_block_len = context.num_state_per_block

    for block_index in range(context.num_video_blocks):
        row_start = first_noisy_frame_end + block_index * video_block_len
        row_end = min(row_start + video_block_len, video_start + context.video_token_length)
        if row_end <= row_start:
            continue
        if context.clean_prefix_length > 0:
            clean_context_end = clean_start + context.tokens_per_frame + block_index * video_block_len
            clean_context_end = min(clean_context_end, clean_end)
            mask[row_start:row_end, clean_start:clean_context_end] = True
        else:
            if first_noisy_frame_end > video_start:
                mask[row_start:row_end, video_start:first_noisy_frame_end] = True
            for previous_index in range(block_index):
                prev_start = first_noisy_frame_end + previous_index * video_block_len
                prev_end = min(prev_start + video_block_len, video_start + context.video_token_length)
                mask[row_start:row_end, prev_start:prev_end] = True
        mask[row_start:row_end, row_start:row_end] = True
        if action_block_len > 0:
            action_block_start = action_start + block_index * action_block_len
            action_block_end = min(action_block_start + action_block_len, action_start + context.action_register_length)
            mask[row_start:row_end, action_block_start:action_block_end] = True
        if state_block_len > 0:
            state_block_start = state_start + block_index * state_block_len
            state_block_end = min(state_block_start + state_block_len, state_start + context.state_register_length)
            mask[row_start:row_end, state_block_start:state_block_end] = True

    for block_index in range(context.num_action_blocks):
        row_start = action_start + block_index * action_block_len
        row_end = min(row_start + action_block_len, action_start + context.action_register_length)
        if row_end <= row_start:
            continue
        if context.clean_prefix_length > 0:
            clean_context_end = clean_start + context.tokens_per_frame + block_index * video_block_len
            clean_context_end = min(clean_context_end, clean_end)
            mask[row_start:row_end, clean_start:clean_context_end] = True
        else:
            if first_noisy_frame_end > video_start:
                mask[row_start:row_end, video_start:first_noisy_frame_end] = True
            for previous_index in range(block_index):
                prev_start = first_noisy_frame_end + previous_index * video_block_len
                prev_end = min(prev_start + video_block_len, video_start + context.video_token_length)
                mask[row_start:row_end, prev_start:prev_end] = True
        if video_block_len > 0:
            video_block_start = first_noisy_frame_end + block_index * video_block_len
            video_block_end = min(video_block_start + video_block_len, video_start + context.video_token_length)
            mask[row_start:row_end, video_block_start:video_block_end] = True
        mask[row_start:row_end, row_start:row_end] = True
        if state_block_len > 0:
            state_block_start = state_start + block_index * state_block_len
            state_block_end = min(state_block_start + state_block_len, state_start + context.state_register_length)
            mask[row_start:row_end, state_block_start:state_block_end] = True

    for block_index in range(context.num_state_blocks):
        row_start = state_start + block_index * state_block_len
        row_end = min(row_start + state_block_len, state_start + context.state_register_length)
        if row_end <= row_start:
            continue
        mask[row_start:row_end, row_start:row_end] = True

    return mask[None, :, :].expand(batch_size, -1, -1)


def build_structured_attention_execution_plan(
    context: StructuredAttentionContext | None,
    *,
    batch_size: int,
    device: torch.device,
    cached_prefix_len: int = 0,
    cached_segment_lengths: tuple[int, ...] = (),
) -> StructuredAttentionExecutionPlan | None:
    if context is None or context.mode == "none":
        return None

    clean_prefix_start = 0
    clean_prefix_end = context.clean_prefix_length
    video_start = clean_prefix_end
    video_end = video_start + context.video_token_length
    action_start = video_end
    action_end = action_start + context.action_register_length
    state_start = action_end
    state_end = state_start + context.state_register_length

    cached_prefix_visibility = None
    use_full_cached_prefix = False
    if cached_prefix_len > 0 and context.mode == "register_explicit":
        cached_prefix_visibility = torch.ones(
            batch_size,
            state_end,
            cached_prefix_len,
            device=device,
            dtype=torch.bool,
        )
        use_full_cached_prefix = True

    return StructuredAttentionExecutionPlan(
        mode=context.mode,
        attention_kernel=context.attention_kernel,
        cache_kernel=context.cache_kernel,
        attention_mask=_build_structured_register_attention_mask(
            context,
            batch_size=batch_size,
            device=device,
        ),
        rotary_freqs=_compose_structured_attention_freqs(context),
        cached_prefix_visibility=cached_prefix_visibility,
        use_full_cached_prefix=use_full_cached_prefix,
        cached_prefix_len=cached_prefix_len,
        cached_segment_lengths=tuple(cached_segment_lengths or context.cached_segment_lengths),
        clean_prefix_span=(clean_prefix_start, clean_prefix_end),
        video_span=(video_start, video_end),
        action_span=(action_start, action_end),
        state_span=(state_start, state_end),
    )
