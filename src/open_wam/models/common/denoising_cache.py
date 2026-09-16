"""Call-local reuse of attention-closed, invariant transformer tokens.

The caller owns immutable conditioning for one denoising stage. This cache is
discarded before a new stage or history revision, so it cannot advance a
rollout cursor or retain dependencies from an evicted observation window.
"""

from dataclasses import dataclass, replace

import torch

from open_wam.models.common.attention_backends import create_block_mask
from open_wam.models.common.attention_contracts import PreparedAttentionProfile


@dataclass
class InvariantTokenCache:
    """One layer's K/V and completed hidden states in original token order."""

    token_count: int
    invariant_indices: torch.Tensor
    active_indices: torch.Tensor
    computed_indices: torch.Tensor
    active_positions: torch.Tensor
    _output: torch.Tensor | None = None
    _key: torch.Tensor | None = None
    _value: torch.Tensor | None = None

    @property
    def ready(self) -> bool:
        return self._output is not None

    def select(
        self, tensor: torch.Tensor | None, *, dim: int = 1
    ) -> torch.Tensor | None:
        if tensor is None:
            return tensor
        if tensor.shape[dim] == 1 and self.token_count != 1:
            return tensor
        if tensor.shape[dim] != self.token_count:
            raise ValueError("Cached execution received a different token geometry.")
        indices = self.active_indices if self.ready else self.computed_indices
        return tensor.index_select(dim, indices)

    def key_value(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine fresh active projections with immutable conditioning K/V."""
        if not self.ready:
            self._key = key.detach().clone()
            self._value = value.detach().clone()
            return key, value
        if self._key is None or self._value is None:
            raise RuntimeError("Layer features require a completed K/V prefill.")
        return (
            self._key.index_copy(2, self.active_positions, key),
            self._value.index_copy(2, self.active_positions, value),
        )

    def output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.ready:
            shape = (hidden_states.shape[0], self.token_count, *hidden_states.shape[2:])
            result = hidden_states.new_zeros(shape).index_copy(
                1, self.computed_indices, hidden_states
            )
            self._output = result.detach().clone()
            return result
        return self._output.index_copy(1, self.active_indices, hidden_states)


def select_block_mask(
    block_mask, query_indices: torch.Tensor, key_indices: torch.Tensor | None = None
):
    if block_mask is None:
        return None
    if create_block_mask is None:
        raise RuntimeError("Cached FlexAttention requires create_block_mask.")
    count = query_indices.numel()
    if not count:
        return None
    original = block_mask.mask_mod

    def visibility(batch, head, query, key):
        mapped_query = query_indices[query.clamp_max(count - 1)]
        if key_indices is None:
            return (query < count) & original(batch, head, mapped_query, key)
        mapped_key = key_indices[key.clamp_max(key_indices.numel() - 1)]
        return (
            (query < count)
            & (key < key_indices.numel())
            & original(batch, head, mapped_query, mapped_key)
        )

    return create_block_mask(
        visibility,
        B=block_mask.shape[0],
        H=block_mask.shape[1],
        Q_LEN=count,
        KV_LEN=block_mask.shape[-1] if key_indices is None else key_indices.numel(),
        device=query_indices.device,
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
    )


def attention_dependency_closure(
    profile: PreparedAttentionProfile, queries: torch.Tensor
) -> torch.Tensor:
    """Find the tokens needed to evaluate these queries through causal layers."""
    if queries.ndim != 1 or queries.dtype != torch.bool or not queries.any():
        raise ValueError(
            "Dependency roots must be a nonempty one-dimensional boolean selection."
        )
    if (
        profile.self_attention_visibility is None
        and profile.self_attention_mask is None
    ):
        raise ValueError("Dependency analysis requires explicit attention visibility.")
    required = queries.clone()
    frontier = required.nonzero().flatten()
    keys = torch.arange(required.numel(), device=required.device)
    while frontier.numel():
        dependencies = torch.zeros_like(required)
        for rows in frontier.split(128):
            visibility = (
                profile.self_attention_visibility(rows[:, None], keys[None, :])
                if profile.self_attention_visibility is not None
                else profile.self_attention_mask.index_select(0, rows)
            )
            dependencies |= visibility.any(dim=0)
        frontier = (dependencies & ~required).nonzero().flatten()
        required |= dependencies
    return required


def select_attention_profile(profile, query_indices, key_indices):
    """Index one canonical law without inventing cache-specific visibility."""
    return replace(
        profile,
        self_attention_mask=None
        if profile.self_attention_mask is None
        else profile.self_attention_mask.index_select(0, query_indices).index_select(
            1, key_indices
        ),
        self_attention_block_mask=select_block_mask(
            profile.self_attention_block_mask, query_indices, key_indices
        ),
        cross_attention_mask=None
        if profile.cross_attention_mask is None
        else profile.cross_attention_mask.index_select(-2, query_indices),
        cross_attention_block_mask=select_block_mask(
            profile.cross_attention_block_mask, query_indices
        ),
        token_layout=None,
        self_attention_visibility=(
            None
            if profile.self_attention_visibility is None
            else lambda query, key: profile.self_attention_visibility(
                query_indices[query], key_indices[key]
            )
        ),
    )


class DenoisingCache:
    """Derived feature storage for one fixed conditioning/attention scope.

    Stream partitions describe tensor layout, not a research method. Invariant
    input tokens must form an attention-closed set at every transformer layer.
    Do not reuse an instance across calls, conditioning changes, or stages that
    promote a newly generated modality into conditioning.
    """

    def __init__(self) -> None:
        self.layers: tuple[tuple[InvariantTokenCache, ...], ...] = ()

    @property
    def bound(self) -> bool:
        return bool(self.layers)

    def bind(
        self,
        *,
        profile: PreparedAttentionProfile,
        invariant_tokens: torch.Tensor,
        stream_lengths: tuple[int, ...],
        num_layers: int,
        required_tokens: torch.Tensor | None = None,
    ) -> None:
        if self.bound:
            raise ValueError(
                "A denoising cache is bound once; create a new scope for changed conditioning."
            )
        if invariant_tokens.ndim != 1 or invariant_tokens.dtype != torch.bool:
            raise ValueError(
                "Invariant token flags must be a one-dimensional boolean tensor."
            )
        if (
            num_layers <= 0
            or not stream_lengths
            or any(length < 0 for length in stream_lengths)
            or sum(stream_lengths) == 0
        ):
            raise ValueError(
                "Cached execution requires positive layers and nonempty stream extent."
            )
        if sum(stream_lengths) != invariant_tokens.numel():
            raise ValueError(
                "Cache stream lengths must cover the original token layout exactly."
            )
        ids = torch.arange(invariant_tokens.numel(), device=invariant_tokens.device)
        required = (
            torch.ones_like(invariant_tokens)
            if required_tokens is None
            else required_tokens
        )
        if (
            required.shape != invariant_tokens.shape
            or required.dtype != torch.bool
            or required.device != invariant_tokens.device
        ):
            raise ValueError(
                "Required token flags must match the invariant token layout."
            )
        if required_tokens is not None and not torch.equal(
            attention_dependency_closure(profile, required), required
        ):
            raise ValueError("Required tokens omit attention dependencies.")
        invariant, active = (
            ids[invariant_tokens & required],
            ids[~invariant_tokens & required],
        )
        if not active.numel():
            raise ValueError("A denoising stage must have at least one active token.")
        visibility = profile.self_attention_visibility
        dense = profile.self_attention_mask
        if visibility is None and dense is None:
            raise ValueError("Cached execution requires explicit attention visibility.")
        for queries in invariant.split(128):
            dependencies = (
                visibility(queries[:, None], active[None, :])
                if visibility is not None
                else dense[queries[:, None], active[None, :]]
            )
            if dependencies.any().item():
                raise ValueError(
                    "Invariant tokens attend to changing tokens; their features cannot be cached."
                )
        self.query_indices = active
        self.key_indices = ids[required]
        self.prefill_profile = select_attention_profile(
            profile, self.key_indices, self.key_indices
        )
        self.query_profile = select_attention_profile(profile, active, self.key_indices)
        partitions = []
        offset = 0
        for count in stream_lengths:
            stream_ids = torch.arange(count, device=invariant_tokens.device)
            constant = invariant_tokens[offset : offset + count]
            needed = required[offset : offset + count]
            computed = stream_ids[needed]
            active_positions = (~constant[needed]).nonzero().flatten()
            partitions.append(
                (
                    count,
                    stream_ids[constant & needed],
                    stream_ids[~constant & needed],
                    computed,
                    active_positions,
                )
            )
            offset += count
        self.layers = tuple(
            tuple(InvariantTokenCache(*partition) for partition in partitions)
            for _ in range(num_layers)
        )
