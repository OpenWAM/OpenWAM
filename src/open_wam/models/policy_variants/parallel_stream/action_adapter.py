from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange

from open_wam.configs import ActionNormMethod, ActionSpace
from open_wam.configs.enums import coerce_enum_value
from open_wam.configs.policy_variant import ParallelStreamPolicyConfig

from .reference_profile import LingbotReferenceProfile, load_reference_profile


@dataclass(frozen=True)
class LingbotActionAdapterSpec:
    """Model-space vs raw-space action alignment for the exact parallel-stream runtime."""

    model_action_dim: int
    raw_action_dim: int
    action_norm_method: ActionNormMethod
    used_action_channel_ids: tuple[int, ...]
    inverse_used_action_channel_ids: tuple[int, ...]
    norm_q01: tuple[float, ...]
    norm_q99: tuple[float, ...]
    reference_profile: LingbotReferenceProfile | None = None


def build_action_adapter_spec(
    config: ParallelStreamPolicyConfig,
    *,
    model_action_dim: int,
) -> LingbotActionAdapterSpec | None:
    profile = load_reference_profile(config.reference_profile)
    used_action_channel_ids = config.used_action_channel_ids or (
        tuple(profile.used_action_channel_ids) if profile is not None else tuple()
    )
    inverse_used_action_channel_ids = config.inverse_used_action_channel_ids or (
        tuple(profile.inverse_used_action_channel_ids) if profile is not None else tuple()
    )
    if not used_action_channel_ids:
        return None
    action_norm_method = coerce_enum_value(ActionNormMethod, config.action_norm_method)
    if action_norm_method == ActionNormMethod.PROFILE:
        if profile is None:
            raise ValueError("Exact parallel-stream action_norm_method='profile' requires a reference_profile.")
        action_norm_method = profile.action_norm_method
    action_norm_method = coerce_enum_value(ActionNormMethod, action_norm_method)
    norm_q01 = config.norm_q01 or (tuple(profile.norm_q01) if profile is not None else tuple())
    norm_q99 = config.norm_q99 or (tuple(profile.norm_q99) if profile is not None else tuple())

    if len(inverse_used_action_channel_ids) != model_action_dim:
        raise ValueError(
            "Exact parallel-stream inverse channel ids must have length equal to the model action dim, "
            f"got {len(inverse_used_action_channel_ids)} and model_action_dim={model_action_dim}."
        )
    if action_norm_method not in {ActionNormMethod.NONE, ActionNormMethod.QUANTILES}:
        raise ValueError(f"Unsupported exact parallel-stream action_norm_method '{action_norm_method}'.")
    if action_norm_method == ActionNormMethod.QUANTILES and (
        len(norm_q01) != model_action_dim or len(norm_q99) != model_action_dim
    ):
        raise ValueError(
            "Quantile-normalized exact parallel-stream actions require q01/q99 values for every model action channel, "
            f"got len(q01)={len(norm_q01)}, len(q99)={len(norm_q99)}, model_action_dim={model_action_dim}."
        )
    return LingbotActionAdapterSpec(
        model_action_dim=model_action_dim,
        raw_action_dim=len(used_action_channel_ids),
        action_norm_method=action_norm_method,
        used_action_channel_ids=tuple(used_action_channel_ids),
        inverse_used_action_channel_ids=tuple(inverse_used_action_channel_ids),
        norm_q01=tuple(norm_q01),
        norm_q99=tuple(norm_q99),
        reference_profile=profile,
    )


class LingbotActionAdapter:
    """Convert exact LingBot parallel-stream actions between raw-space and model-space."""

    def __init__(self, spec: LingbotActionAdapterSpec | None) -> None:
        self.spec = spec

    @property
    def supports_raw_actions(self) -> bool:
        return self.spec is not None

    def infer_action_space(self, action: torch.Tensor) -> ActionSpace:
        if self.spec is None:
            return ActionSpace.MODEL
        feature_dim = self._flatten_to_sequence(action).shape[-1]
        if feature_dim == self.spec.model_action_dim and feature_dim != self.spec.raw_action_dim:
            return ActionSpace.MODEL
        if feature_dim == self.spec.raw_action_dim and feature_dim != self.spec.model_action_dim:
            return ActionSpace.RAW
        if feature_dim == self.spec.model_action_dim == self.spec.raw_action_dim:
            raise ValueError(
                "Automatic exact action-space inference is ambiguous because raw and model action dims are equal. "
                "Pass `action_space='model'` or `action_space='raw'` explicitly."
            )
        raise ValueError(
            "Unable to infer exact action space from the trailing dimension "
            f"{feature_dim}; expected raw={self.spec.raw_action_dim} or model={self.spec.model_action_dim}."
        )

    def to_model_action_sequence(
        self,
        action: torch.Tensor,
        *,
        action_space: ActionSpace | str = ActionSpace.AUTO,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        sequence = self._flatten_to_sequence(action)
        if self.spec is None:
            if device is not None or dtype is not None:
                sequence = sequence.to(device=device or sequence.device, dtype=dtype or sequence.dtype)
            return sequence
        resolved_space = self.infer_action_space(sequence) if action_space == ActionSpace.AUTO else ActionSpace(action_space)
        target_device = device or sequence.device
        target_dtype = dtype or sequence.dtype
        if resolved_space == ActionSpace.MODEL:
            if sequence.shape[-1] != self.spec.model_action_dim:
                raise ValueError(
                    f"Expected model-space action dim {self.spec.model_action_dim}, got {sequence.shape[-1]}."
                )
            return sequence.to(device=target_device, dtype=target_dtype)
        if resolved_space != ActionSpace.RAW:
            raise ValueError(f"Unsupported exact action_space '{action_space}'.")
        if sequence.shape[-1] != self.spec.raw_action_dim:
            raise ValueError(f"Expected raw-space action dim {self.spec.raw_action_dim}, got {sequence.shape[-1]}.")
        padded = torch.cat(
            [sequence.to(device=target_device, dtype=torch.float32), sequence.new_zeros(sequence.shape[0], sequence.shape[1], 1, device=target_device, dtype=torch.float32)],
            dim=-1,
        )
        gather_ids = torch.tensor(
            self.spec.inverse_used_action_channel_ids,
            device=target_device,
            dtype=torch.long,
        )
        aligned = padded.index_select(dim=-1, index=gather_ids)
        if self.spec.action_norm_method == ActionNormMethod.QUANTILES:
            q01 = torch.tensor(self.spec.norm_q01, device=target_device, dtype=torch.float32)
            q99 = torch.tensor(self.spec.norm_q99, device=target_device, dtype=torch.float32)
            aligned = (aligned - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return aligned.to(dtype=target_dtype)

    def to_model_action_mask_sequence(
        self,
        action_mask: torch.Tensor,
        *,
        action_space: ActionSpace | str = ActionSpace.AUTO,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        sequence = self._flatten_to_sequence(action_mask)
        if device is not None or dtype is not None:
            sequence = sequence.to(device=device or sequence.device, dtype=dtype or sequence.dtype)
        if self.spec is None:
            return sequence
        resolved_space = self.infer_action_space(sequence) if action_space == ActionSpace.AUTO else ActionSpace(action_space)
        if resolved_space == ActionSpace.MODEL:
            if sequence.shape[-1] != self.spec.model_action_dim:
                raise ValueError(
                    f"Expected model-space action mask dim {self.spec.model_action_dim}, got {sequence.shape[-1]}."
                )
            return sequence
        if resolved_space != ActionSpace.RAW:
            raise ValueError(f"Unsupported exact action_space '{action_space}'.")
        if sequence.shape[-1] != self.spec.raw_action_dim:
            raise ValueError(
                f"Expected raw-space action mask dim {self.spec.raw_action_dim}, got {sequence.shape[-1]}."
            )
        padded = torch.cat(
            [sequence, sequence.new_zeros(sequence.shape[0], sequence.shape[1], 1)],
            dim=-1,
        )
        gather_ids = torch.tensor(
            self.spec.inverse_used_action_channel_ids,
            device=sequence.device,
            dtype=torch.long,
        )
        return padded.index_select(dim=-1, index=gather_ids)

    def to_model_action_latents(
        self,
        action: torch.Tensor,
        *,
        action_per_frame: int,
        action_space: ActionSpace | str = ActionSpace.AUTO,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        sequence = self.to_model_action_sequence(
            action,
            action_space=action_space,
            device=device,
            dtype=dtype,
        )
        if sequence.shape[1] % action_per_frame != 0:
            raise ValueError(
                "Exact parallel-stream action sequence length must be divisible by action_per_frame, "
                f"got sequence_length={sequence.shape[1]} and action_per_frame={action_per_frame}."
            )
        return rearrange(
            sequence,
            "b (f a) c -> b c f a 1",
            a=action_per_frame,
        )

    def to_raw_action_sequence(
        self,
        model_action_sequence: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.spec is None:
            return None
        sequence = self._flatten_to_sequence(model_action_sequence)
        if sequence.shape[-1] != self.spec.model_action_dim:
            raise ValueError(
                f"Expected model-space action dim {self.spec.model_action_dim}, got {sequence.shape[-1]}."
            )
        sequence = sequence.float()
        if self.spec.action_norm_method == ActionNormMethod.QUANTILES:
            q01 = torch.tensor(self.spec.norm_q01, device=sequence.device, dtype=torch.float32)
            q99 = torch.tensor(self.spec.norm_q99, device=sequence.device, dtype=torch.float32)
            sequence = (sequence + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        gather_ids = torch.tensor(self.spec.used_action_channel_ids, device=sequence.device, dtype=torch.long)
        return sequence.index_select(dim=-1, index=gather_ids)

    def _flatten_to_sequence(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 5:
            return rearrange(action, "b c f n 1 -> b (f n) c")
        if action.ndim == 4:
            return rearrange(action, "b f n c -> b (f n) c")
        if action.ndim == 3:
            return action
        if action.ndim == 2:
            return action.unsqueeze(0)
        raise ValueError(
            "Unsupported exact action tensor shape. Expected [B,T,C], [B,F,A,C], or [B,C,F,A,1], "
            f"got {tuple(action.shape)}."
        )
