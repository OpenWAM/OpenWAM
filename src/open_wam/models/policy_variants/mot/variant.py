from __future__ import annotations

import os
from dataclasses import replace as _dataclass_replace

import torch
import torch.nn.functional as F

from open_wam.models.common.flow_matching import (
    VideoFlowMatchTrainArtifacts,
    build_video_flow_match_train_artifacts,
    build_video_flow_match_inference_scheduler,
    build_action_flow_match_inference_scheduler,
    build_action_flow_match_train_artifacts,
    build_frame_aligned_action_flow_match_train_artifacts,
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
    sample_timestep_id,
    timesteps_matching_sigmas,
)
from open_wam.models.common.flow_noise_plan import frame_sigmas_for_timesteps
from open_wam.models.common.attention_profiles import build_chunked_text_context_cross_attention_mask
from open_wam.models.common.joint_conditioning import (
    resolve_generalist_joint_conditioning_semantics,
    sample_conditioning_mode,
)
from open_wam.models.common.modality_slots import clean_noisy_slot_tensor, zero_loss_mask_like
from open_wam.models.common.rollout_startup import (
    build_strict_action_context_mask,
    resolve_strict_startup_plan,
)
from open_wam.configs import (
    CurrentBlockCoupling,
    InferenceConfig,
    JointTimestepCoupling,
    MoTGeneralistTrainingMode,
    MoTPolicyConfig,
    MoTRuntimeMode,
    ParallelSequenceContract,
    ParallelContextConditionLatentSource,
    ParallelHistoryStreamVisibility,
    ProprioContextMode,
    TrainingConfig,
)
from open_wam.data.sample_metadata import SampleConstructionMetadata
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower
from open_wam.models.visual_tower.grid_ids import build_action_grid_ids
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig

from ..base import PolicyVariant
from ..common.layouts import expand_previous_action
from ..contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)
from .contracts import (
    MoTActionCache,
    MoTActionLayerCache,
    MoTActionTrainArtifacts,
    MoTInferArtifacts,
    MoTRuntimeState,
    MoTTrainArtifacts,
    MoTVideoCache,
    MoTVideoLayerCache,
    MoTVideoTrainArtifacts,
)
from .modules import MoTActionExpert, init_action_expert_from_video_core
from .packed_block import MoTPackedBlock, MoTPackedBlockStack
from .runtime import (
    append_mot_action_cache,
    append_mot_video_cache,
    build_chunk_causal_video_mask,
    build_mot_attention_mask,
    build_mot_inference_action_attention_mask,
    build_mot_packed_coupling_attention_profile,
    forward_joint_video_action_denoise,
    forward_mot_packed_coupling_denoise,
    forward_action_with_video_and_action_cache,
    forward_action_with_video_cache,
    move_mot_action_cache,
    move_mot_video_cache,
    prefill_video_kv_cache,
    trim_mot_action_cache_prefix,
    trim_mot_action_cache_tail,
    trim_mot_video_cache_tail,
    resolve_mot_condition_latents,
)
from .runtime_routing import (
    MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS,
    ensure_mot_policy_variant_inference_backend,
    resolve_mot_rollout_cache_window_frames,
)

# Default LingBot-reference slot-pool window used by both `_initialize_reference_cache`
# and the Method-1-aligned video-cache trim. Rollout callers may override this
# through `PolicyInferContext.extra["mot_inference_window_size"]`. Method 1's
# per-stream effective lookback is `(attn_window // 2) * frame_chunk_size`
# integer frames (60 at attn_window=30, frame_chunk_size=4).
_MOT_SLOT_POOL_ATTN_WINDOW = 30
_MOT_ACTION_ONLY_ROLLOUT_COUPLINGS = frozenset(
    {
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
    }
)


def _resolve_mot_inference_window_size(
    context: PolicyInferContext,
    *,
    default_window_size: int,
) -> int:
    raw_override = context.extra.get("mot_inference_window_size")
    if raw_override is None:
        resolved = int(default_window_size)
    else:
        resolved = int(raw_override)
    if resolved <= 0:
        raise ValueError(
            "MoT inference window size must be positive, "
            f"got {resolved}."
        )
    return resolved


def _resolve_mot_action_only_rollout(
    context: PolicyInferContext,
    *,
    current_block_coupling: CurrentBlockCoupling,
) -> bool:
    requested = bool(context.extra.get("mot_action_only_rollout", False))
    if requested and current_block_coupling not in _MOT_ACTION_ONLY_ROLLOUT_COUPLINGS:
        supported = ", ".join(
            (
                CurrentBlockCoupling.ACTION_THEN_VIDEO.value,
                CurrentBlockCoupling.DECOUPLED_SAME_STEP.value,
            )
        )
        raise ValueError(
            "`mot_action_only_rollout` is only supported for M5 action-only-safe "
            f"couplings ({supported}); got current_block_coupling={current_block_coupling.value!r}."
        )
    return requested


def resolve_mot_current_block_coupling(config: MoTPolicyConfig) -> CurrentBlockCoupling:
    """Resolve Method-5 current-block coupling, defaulting to current behavior."""

    if config.current_block_coupling is None:
        if config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            return CurrentBlockCoupling.JOINT
        return CurrentBlockCoupling.VIDEO_THEN_ACTION
    return CurrentBlockCoupling(config.current_block_coupling)


def _is_mot_same_step_coupling(coupling: CurrentBlockCoupling) -> bool:
    return coupling in {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.DECOUPLED_SAME_STEP,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }


def _should_couple_mot_action_to_video_sigmas(
    config: MoTPolicyConfig,
    coupling: CurrentBlockCoupling,
) -> bool:
    """Return whether M5 rollout should integrate action on the video sigma clock."""

    return _resolve_mot_joint_timestep_coupling(config, coupling) in {
        JointTimestepCoupling.MATCH_SIGMA,
        JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
    }


def _resolve_mot_joint_timestep_coupling(
    config: MoTPolicyConfig,
    coupling: CurrentBlockCoupling,
) -> JointTimestepCoupling:
    """Resolve M5 joint-like action/video timestep coupling."""

    if coupling not in {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }:
        return JointTimestepCoupling.INDEPENDENT
    return JointTimestepCoupling(config.joint_timestep_coupling)


def _uses_mot_legacy_prefix_contract(config: MoTPolicyConfig) -> bool:
    return (
        ParallelSequenceContract(config.parallel_sequence_contract)
        == ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
    )


def _slice_current_noisy_action_flow(
    packed_action_flow: torch.Tensor,
    *,
    history_action_tokens: int,
    action_horizon: int,
) -> torch.Tensor:
    """Select current noisy-action tokens from a packed M5 action stream."""

    start = int(history_action_tokens)
    end = start + int(action_horizon)
    if start < 0 or action_horizon <= 0:
        raise ValueError(
            "M5 packed action flow slicing requires non-negative history tokens "
            f"and positive action_horizon, got history_action_tokens={history_action_tokens}, "
            f"action_horizon={action_horizon}."
        )
    if packed_action_flow.ndim < 2 or packed_action_flow.shape[1] < end:
        raise ValueError(
            "M5 packed action flow is too short to contain the current noisy-action window, "
            f"got shape={tuple(packed_action_flow.shape)}, history_action_tokens={history_action_tokens}, "
            f"action_horizon={action_horizon}."
        )
    return packed_action_flow[:, start:end].contiguous()


def _scheduler_next_sigma(scheduler, step_index: int) -> torch.Tensor:
    if int(step_index) + 1 >= len(scheduler.sigmas):
        return scheduler.sigmas.new_tensor(0.0)
    return scheduler.sigmas[int(step_index) + 1]


def _flow_step_with_sigmas(
    sample: torch.Tensor,
    flow_pred: torch.Tensor,
    *,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    return sample + flow_pred * (
        sigma_next.to(device=sample.device, dtype=sample.dtype)
        - sigma.to(device=sample.device, dtype=sample.dtype)
    )


def _expand_scalar_timestep(
    value: torch.Tensor | float,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar timestep value, got shape {tuple(value.shape)}.")
        return value.to(device=device, dtype=torch.float32).reshape(()).expand(shape).clone()
    return torch.full(shape, float(value), device=device, dtype=torch.float32)


def _mot_packed_cache_inference_couplings() -> set[CurrentBlockCoupling]:
    return {
        CurrentBlockCoupling.JOINT,
        CurrentBlockCoupling.ACTION_THEN_VIDEO,
        CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
        CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
    }


def _sample_mot_generalist_training_mode(
    probs: dict[MoTGeneralistTrainingMode, float],
    *,
    device: torch.device,
) -> MoTGeneralistTrainingMode:
    """Sample one M5 generalist regime per segment.

    Mirrors PR #95's ``_sample_joint_denoise_training_mode``: builds a
    categorical from the (already-normalized) probs dict and draws a single
    mode. Sampling runs on the same device as the training segment so it
    stays deterministic under a seeded RNG state.
    """

    return sample_conditioning_mode(
        probs,
        enum_cls=MoTGeneralistTrainingMode,
        device=device,
        error_label="M5 generalist training mode",
    )


def _resolve_mot_generalist_training_metadata(
    batch: PolicyTrainBatch,
) -> tuple[MoTGeneralistTrainingMode | None, bool | None, str | None]:
    sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
    if sample_metadata is None:
        return None, None, None
    raw_mode = sample_metadata.generalist.mode_override
    mode = None if raw_mode is None else MoTGeneralistTrainingMode(raw_mode)
    return mode, sample_metadata.generalist.drop_text_conditioning, sample_metadata.generalist.source


def _apply_mot_generalist_training_mode(
    *,
    sampled_mode: MoTGeneralistTrainingMode,
    video_artifacts: VideoFlowMatchTrainArtifacts,
    noisy_actions: torch.Tensor,
    clean_actions: torch.Tensor,
    noisy_slot_timesteps: torch.Tensor,
    future_loss_mask: torch.Tensor,
    effective_action_mask: torch.Tensor | None,
    clean_action_condition_mask: torch.Tensor | None = None,
) -> tuple[
    VideoFlowMatchTrainArtifacts,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Apply M5 generalist denoising mode semantics to packed train tensors.

    Realizes the conditional sub-modes by placing the clean modality into its
    noisy slot, preserving real clean condition slots for history/context,
    forcing per-frame timesteps to 0 on the conditioned side, and masking that
    side's loss. The ``JOINT`` bucket intentionally preserves the clean
    condition slots so its training contract matches plain M5 packed-joint
    training and rollout: noisy current tokens can use past clean video/action
    context through the same Method-1-style packed mask.

    ``effective_action_mask`` is the supervised action-loss mask. It may be
    narrower than the raw valid-action mask under fixed-segment sampling, so it
    must not be reused to hide clean action conditions from FDM/IDM context.
    """

    semantics = resolve_generalist_joint_conditioning_semantics(
        sampled_mode,
        joint_mode=MoTGeneralistTrainingMode.JOINT,
        action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    )

    if semantics.is_joint:
        return (
            video_artifacts,
            noisy_actions,
            clean_actions,
            noisy_slot_timesteps,
            future_loss_mask,
            effective_action_mask,
        )

    if semantics.clean_action_noisy_slot:
        # Clean action overwrites the A_noisy slot at timestep 0; A_clean
        # remains real clean action history/context; action loss is masked out
        # so video-only gradients drive this segment.
        new_noisy_actions = clean_noisy_slot_tensor(
            clean_actions.clone(),
            action_mask=clean_action_condition_mask,
        )
        new_noisy_slot_timesteps = torch.zeros_like(noisy_slot_timesteps)
        new_action_mask = zero_loss_mask_like(effective_action_mask, fallback_like=noisy_actions)
        return (
            video_artifacts,
            new_noisy_actions,
            clean_actions,
            new_noisy_slot_timesteps,
            future_loss_mask,
            new_action_mask,
        )

    if semantics.clean_video_noisy_slot:
        # Clean video overwrites the V_noisy slot at timestep 0; V_clean
        # remains available as past clean context under the packed attention
        # mask; video loss is masked out so action-only gradients drive this
        # segment.
        new_video_artifacts = _dataclass_replace(
            video_artifacts,
            noisy_latents=video_artifacts.condition_latents.clone(),
            timesteps=torch.zeros_like(video_artifacts.timesteps),
        )
        new_future_loss_mask = torch.zeros_like(future_loss_mask)
        return (
            new_video_artifacts,
            noisy_actions,
            clean_actions,
            noisy_slot_timesteps,
            new_future_loss_mask,
            effective_action_mask,
        )

    raise ValueError(f"Unsupported MoTGeneralistTrainingMode {sampled_mode!r}.")


def _mot_generalist_forces_clean_video_condition(
    sampled_mode: MoTGeneralistTrainingMode | None,
) -> bool:
    if sampled_mode is None:
        return False
    semantics = resolve_generalist_joint_conditioning_semantics(
        sampled_mode,
        joint_mode=MoTGeneralistTrainingMode.JOINT,
        action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
        video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
    )
    return semantics.force_clean_video_condition


def _rewind_runtime_action_cache_to_frame(
    runtime_state: MoTRuntimeState,
    *,
    absolute_frame_start: int,
    action_tokens_per_frame: int,
) -> None:
    if action_tokens_per_frame <= 0:
        raise ValueError(
            "MoT action-cache rewind requires positive action_tokens_per_frame, "
            f"got {action_tokens_per_frame}."
        )
    target_frame = int(absolute_frame_start)
    action_cache = runtime_state.action_cache
    if action_cache is None:
        runtime_state.action_cache_start_frame = target_frame
        return
    if action_cache.action_seq_len % action_tokens_per_frame != 0:
        raise ValueError(
            "MoT action cache length must be frame-aligned before rewind, "
            f"got action_seq_len={action_cache.action_seq_len}, "
            f"action_tokens_per_frame={action_tokens_per_frame}."
        )
    cache_start_frame = int(runtime_state.action_cache_start_frame)
    keep_frames = target_frame - cache_start_frame
    if keep_frames <= 0:
        runtime_state.action_cache = None
        runtime_state.action_cache_start_frame = target_frame
        return
    cached_frames = action_cache.action_seq_len // action_tokens_per_frame
    if keep_frames >= cached_frames:
        return
    runtime_state.action_cache = trim_mot_action_cache_prefix(
        action_cache,
        max_action_seq_len=int(keep_frames * action_tokens_per_frame),
    )


class MoTPolicyVariant(PolicyVariant):
    """Method-5 scaffold for the future FastWAM-style MoT runtime.

    This class intentionally only wires the config/build surface in the first
    landing. The actual action expert and mixed-attention runtime are added in
    follow-up changes instead of silently degrading into another policy family.
    """

    def __init__(
        self,
        config: MoTPolicyConfig,
        backbone_config: SharedVideoTransformerConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_dim: int,
        action_horizon: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone_config = backbone_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_dim = state_dim
        action_hidden_size = (
            int(config.action_hidden_size)
            if config.action_hidden_size is not None
            else int(backbone_config.hidden_size)
        )
        self.action_expert = MoTActionExpert(
            hidden_size=action_hidden_size,
            action_dim=action_dim,
            num_layers=config.num_action_layers,
            num_heads=backbone_config.num_heads,
            attention_head_dim=backbone_config.attention_head_dim,
            ffn_dim=(
                int(config.action_ffn_dim)
                if config.action_ffn_dim is not None
                else (backbone_config.ffn_dim or (backbone_config.hidden_size * backbone_config.mlp_ratio))
            ),
            text_dim=backbone_config.text_dim,
            hidden_context_dim=backbone_config.hidden_size,
            freq_dim=backbone_config.freq_dim,
            cross_attn_norm=backbone_config.cross_attn_norm,
            eps=backbone_config.latent_norm_eps,
        )
        self._action_expert_initialized = False
        self._train_video_cache_detach_by_core_id: dict[int, bool] = {}
        # Lazy-initialized at pipeline assembly time when current_block_coupling
        # is set. Owns video_block + action_block pairs after ownership transfer
        # so FSDP can wrap the packed unit cleanly without aliasing.
        self.packed_block_stack: MoTPackedBlockStack | None = None
        self._packed_block_stack_attached = False
        self._legacy_inference_blocks_restored = False

    def _uses_proprio_context(self) -> bool:
        return ProprioContextMode(self.config.proprio_context_mode) != ProprioContextMode.NONE

    def _uses_text_proprio_context(self) -> bool:
        # Deprecated compatibility path; new proprio runs use per-chunk additive context.
        return ProprioContextMode(self.config.proprio_context_mode) == ProprioContextMode.TEXT_CONTEXT_TOKEN

    def _uses_per_chunk_proprio_context(self) -> bool:
        return ProprioContextMode(self.config.proprio_context_mode) == ProprioContextMode.PER_CHUNK_ADDITIVE

    @staticmethod
    def _select_anchor_state(state: torch.Tensor | None) -> torch.Tensor | None:
        if state is None:
            return None
        if state.ndim == 2:
            return state
        if state.ndim == 3:
            return state[:, -1, :]
        raise ValueError(
            "M5 proprio context expects state with shape [B, state_dim] or [B, H, state_dim], "
            f"got {tuple(state.shape)}."
        )

    def _resolve_proprio_state(
        self,
        state: torch.Tensor | None,
        *,
        label: str,
        fallback_state: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self._uses_text_proprio_context():
            return None
        selected = self._select_anchor_state(state)
        if selected is None:
            selected = self._select_anchor_state(fallback_state)
        if selected is None:
            raise ValueError(f"Proprio context mode is enabled but no state was provided for {label}.")
        return selected

    def _resolve_train_proprio_context(self, batch: PolicyTrainBatch) -> torch.Tensor | None:
        if not self._uses_text_proprio_context():
            return None
        proprio_context_state = batch.extra.get("proprio_context_state")
        if isinstance(proprio_context_state, torch.Tensor):
            if proprio_context_state.ndim != 3:
                raise ValueError(
                    "Per-chunk proprio context expects shape [B, chunks, state_dim], "
                    f"got {tuple(proprio_context_state.shape)}."
                )
            proprio_context_state_mask = batch.extra.get("proprio_context_state_mask")
            if isinstance(proprio_context_state_mask, torch.Tensor):
                if tuple(proprio_context_state_mask.shape) != tuple(proprio_context_state.shape):
                    raise ValueError(
                        "Per-chunk proprio context mask must match proprio_context_state shape, "
                        f"got mask={tuple(proprio_context_state_mask.shape)}, "
                        f"state={tuple(proprio_context_state.shape)}."
                    )
                proprio_context_state = proprio_context_state * proprio_context_state_mask.to(
                    device=proprio_context_state.device,
                    dtype=proprio_context_state.dtype,
                )
            return proprio_context_state
        return self._resolve_proprio_state(
            batch.state,
            label="M5 training",
        )

    def _resolve_train_hidden_proprio_context(self, batch: PolicyTrainBatch) -> torch.Tensor | None:
        if not self._uses_per_chunk_proprio_context():
            return None
        value = batch.extra.get("proprio_context_frames")
        mask = batch.extra.get("proprio_context_frames_mask")
        if not isinstance(value, torch.Tensor):
            fallback_value = batch.extra.get("proprio_context_state")
            if _uses_mot_legacy_prefix_contract(self.config) and isinstance(fallback_value, torch.Tensor):
                raise ValueError(
                    "M5 legacy-prefix per-chunk additive proprio requires frame-level "
                    "`proprio_context_frames`; chunk-level `proprio_context_state` cannot be "
                    "safely aligned to prefix and causal chunk-boundary states."
                )
            value = fallback_value
            mask = batch.extra.get("proprio_context_state_mask")
        if not isinstance(value, torch.Tensor):
            raise ValueError("proprio_context_mode=per_chunk_additive requires M5 per-frame or per-chunk proprio context.")
        if value.ndim != 3:
            raise ValueError(
                "M5 per-chunk additive proprio expects state with shape [B, frames, state_dim], "
                f"got {tuple(value.shape)}."
            )
        if isinstance(mask, torch.Tensor):
            if tuple(mask.shape) != tuple(value.shape):
                raise ValueError(
                    "M5 per-chunk additive proprio mask must match state shape, "
                    f"got mask={tuple(mask.shape)}, state={tuple(value.shape)}."
                )
            value = value * mask.to(device=value.device, dtype=value.dtype)
        return value

    def _resolve_infer_hidden_proprio_context(
        self,
        state: torch.Tensor | None,
        *,
        fallback_state: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self._uses_per_chunk_proprio_context():
            return None
        selected = self._select_anchor_state(state)
        if selected is None:
            selected = self._select_anchor_state(fallback_state)
        if selected is None:
            raise ValueError("proprio_context_mode=per_chunk_additive requires M5 inference state.")
        return selected

    def _resolve_text_context_with_proprio(
        self,
        visual_tower: VisualTower,
        text_context: torch.Tensor | None,
        proprio_state: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        materialize_if_missing: bool,
    ) -> torch.Tensor | None:
        if text_context is None:
            if not materialize_if_missing:
                return None
            text_context = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=device,
                dtype=dtype,
            )
        else:
            text_context = text_context.to(device=device, dtype=dtype)
        if proprio_state is None:
            return text_context
        if not self._uses_text_proprio_context():
            return text_context
        append = getattr(visual_tower.core, "append_proprio_context_tokens", None)
        if not callable(append):
            raise ValueError(
                "Deprecated text-space proprio token mode requires the visual tower core "
                "to support proprio appending."
            )
        return append(text_context, proprio_state)

    def _encode_hidden_proprio_context(
        self,
        visual_tower: VisualTower,
        proprio_state: torch.Tensor | None,
        *,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        chunk_size_frames: int | None = None,
    ) -> torch.Tensor | None:
        if proprio_state is None:
            return None
        encode = getattr(visual_tower.core, "encode_proprio_hidden_context", None)
        if not callable(encode):
            raise ValueError("proprio_context_mode=per_chunk_additive requires a core hidden proprio encoder hook.")
        if proprio_state.ndim == 2:
            frame_state = proprio_state[:, None, :].expand(-1, int(num_frames), -1)
        elif proprio_state.ndim == 3:
            if int(proprio_state.shape[1]) == int(num_frames):
                frame_state = proprio_state
            elif int(proprio_state.shape[1]) == 1:
                frame_state = proprio_state.expand(-1, int(num_frames), -1)
            elif chunk_size_frames is not None and int(chunk_size_frames) > 0:
                expanded = proprio_state.repeat_interleave(int(chunk_size_frames), dim=1)
                if int(expanded.shape[1]) < int(num_frames):
                    raise ValueError(
                        "M5 chunk-level hidden proprio context is too short for requested frames, "
                        f"got state={tuple(proprio_state.shape)}, chunk_size_frames={chunk_size_frames}, "
                        f"num_frames={num_frames}."
                    )
                frame_state = expanded[:, : int(num_frames), :]
            else:
                raise ValueError(
                    "M5 hidden proprio context frame count must match requested frames, be singleton, "
                    "or be chunk-level with `chunk_size_frames`, "
                    f"got state={tuple(proprio_state.shape)}, num_frames={num_frames}."
                )
        else:
            raise ValueError(
                "M5 hidden proprio context expects shape [B, state_dim] or [B, frames, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        return encode(frame_state, device=device, dtype=dtype)

    def _video_hidden_context_for_tokens(
        self,
        visual_tower: VisualTower,
        proprio_state: torch.Tensor | None,
        *,
        video_latents: torch.Tensor,
        copies: int = 1,
        chunk_size_frames: int | None = None,
    ) -> torch.Tensor | None:
        frame_context = self._encode_hidden_proprio_context(
            visual_tower,
            proprio_state,
            num_frames=int(video_latents.shape[2]),
            device=video_latents.device,
            dtype=video_latents.dtype,
            chunk_size_frames=chunk_size_frames,
        )
        if frame_context is None:
            return None
        patch_t, patch_h, patch_w = visual_tower.core.patch_size
        frame_context = frame_context[:, :: int(patch_t), :]
        tokens_per_frame = (int(video_latents.shape[3]) // int(patch_h)) * (
            int(video_latents.shape[4]) // int(patch_w)
        )
        token_context = frame_context.repeat_interleave(tokens_per_frame, dim=1)
        return token_context.repeat(1, int(copies), 1)

    def _action_hidden_context_for_tokens(
        self,
        visual_tower: VisualTower,
        proprio_state: torch.Tensor | None,
        *,
        action_tokens: torch.Tensor,
        action_tokens_per_frame: int,
        copies: int = 1,
        chunk_size_frames: int | None = None,
    ) -> torch.Tensor | None:
        if action_tokens_per_frame <= 0 or int(action_tokens.shape[1]) % int(action_tokens_per_frame) != 0:
            raise ValueError(
                "M5 action hidden proprio context requires action length divisible by action_tokens_per_frame, "
                f"got action_shape={tuple(action_tokens.shape)}, action_tokens_per_frame={action_tokens_per_frame}."
            )
        num_frames = int(action_tokens.shape[1]) // int(action_tokens_per_frame)
        frame_context = self._encode_hidden_proprio_context(
            visual_tower,
            proprio_state,
            num_frames=num_frames,
            device=action_tokens.device,
            dtype=action_tokens.dtype,
            chunk_size_frames=chunk_size_frames,
        )
        if frame_context is None:
            return None
        token_context = frame_context.repeat_interleave(int(action_tokens_per_frame), dim=1)
        return token_context.repeat(1, int(copies), 1)

    @staticmethod
    def _proprio_context_token_count(proprio_state: torch.Tensor | None) -> int:
        if proprio_state is None:
            return 0
        if proprio_state.ndim == 2:
            return 1
        if proprio_state.ndim == 3:
            return int(proprio_state.shape[1])
        raise ValueError(
            "Proprio context expects state with shape [B, state_dim] or [B, chunks, state_dim], "
            f"got {tuple(proprio_state.shape)}."
        )

    def _build_proprio_cross_attention_mask(
        self,
        *,
        resolved_text_context: torch.Tensor,
        proprio_state: torch.Tensor | None,
        query_frames_per_copy: int,
        tokens_per_frame: int,
        chunk_size_frames: int,
        chunk_origin_frame: int = 0,
        repeat_copies: int = 1,
        global_suffix_token_count: int = 0,
    ) -> torch.Tensor | None:
        proprio_token_count = self._proprio_context_token_count(proprio_state)
        suffix_token_count = int(global_suffix_token_count)
        if proprio_token_count <= 1 and suffix_token_count <= 0:
            return None
        gated_proprio_token_count = int(proprio_token_count) if proprio_token_count > 1 else 0
        if query_frames_per_copy <= 0 or tokens_per_frame <= 0:
            raise ValueError(
                "Proprio cross-attention masking requires positive query geometry, "
                f"got frames={query_frames_per_copy}, tokens_per_frame={tokens_per_frame}."
            )
        chunk_size = max(1, int(chunk_size_frames))
        frame_ids = torch.arange(
            int(query_frames_per_copy),
            device=resolved_text_context.device,
            dtype=torch.long,
        ).repeat_interleave(int(tokens_per_frame))
        query_chunk_ids = torch.div(
            frame_ids - int(chunk_origin_frame),
            chunk_size,
            rounding_mode="floor",
        ).repeat(int(repeat_copies))
        base_text_token_count = int(resolved_text_context.shape[1]) - gated_proprio_token_count - suffix_token_count
        return build_chunked_text_context_cross_attention_mask(
            query_chunk_ids=query_chunk_ids,
            batch_size=int(resolved_text_context.shape[0]),
            text_token_count=int(resolved_text_context.shape[1]),
            base_text_token_count=base_text_token_count,
            proprio_context_token_count=gated_proprio_token_count,
            global_suffix_token_count=suffix_token_count,
            device=resolved_text_context.device,
        )

    def _append_generalist_mode_text_token(
        self,
        visual_tower: VisualTower,
        text_context: torch.Tensor,
        mode: MoTGeneralistTrainingMode,
    ) -> tuple[torch.Tensor, int]:
        if not bool(getattr(self.config, "generalist_mode_text_token", False)):
            return text_context, 0
        append = getattr(visual_tower.core, "append_generalist_mode_context_token", None)
        if not callable(append):
            raise ValueError("MoT `generalist_mode_text_token=true` requires a visual core mode-token hook.")
        before_tokens = int(text_context.shape[1])
        resolved = append(text_context, mode.value)
        token_count = int(resolved.shape[1]) - before_tokens
        if token_count != 1:
            raise ValueError(
                "MoT generalist mode token appending must add exactly one token, "
                f"got token_count={token_count}."
            )
        return resolved, token_count

    def attach_visual_tower(self, visual_tower: VisualTower) -> None:
        """Pipeline-time hook: build the packed-coupling block stack.

        Must run AFTER both ``visual_tower`` and ``self.action_expert`` exist
        but BEFORE FSDP sharding. Transfers ownership of video core blocks and
        action expert blocks into ``self.packed_block_stack`` so FSDP only
        sees a single owner per nn.Parameter (no shared-module aliasing).
        Non-packed runtime modes are no-ops.

        ``_maybe_initialize_action_expert`` runs BEFORE the transfer because
        the init helper reads from ``visual_tower.core.blocks`` and writes to
        ``self.action_expert.blocks``; after transfer both ModuleLists are
        empty.
        """
        if self._uses_text_proprio_context():
            configure = getattr(visual_tower.core, "configure_proprio_context_encoder", None)
            if not callable(configure):
                raise ValueError(
                    "Deprecated proprio_context_mode=text_context_token requires a core proprio encoder hook."
                )
            configure(enabled=True, state_dim=int(self.state_dim))
        elif self._uses_per_chunk_proprio_context():
            configure = getattr(visual_tower.core, "configure_proprio_hidden_context_encoder", None)
            if not callable(configure):
                raise ValueError("proprio_context_mode=per_chunk_additive requires a core proprio hidden encoder hook.")
            configure(enabled=True, state_dim=int(self.state_dim))
        if self._packed_block_stack_attached:
            return
        self._packed_block_stack_attached = True
        if self.config.current_block_coupling is None:
            return
        # Run lazy action-expert init now, while blocks still live under
        # visual_tower.core / self.action_expert.
        self._maybe_initialize_action_expert(visual_tower)
        video_blocks = list(visual_tower.core.blocks)
        action_blocks = list(self.action_expert.blocks)
        # Build the stack first so it owns the children; then drop them from
        # the original ModuleList containers. Param identity is preserved
        # across the move (same nn.Parameter objects, just under a new parent),
        # so any optimizer built from `model.parameters()` after this hook runs
        # sees the same set.
        self.packed_block_stack = MoTPackedBlockStack(video_blocks, action_blocks)
        visual_tower.core.blocks = torch.nn.ModuleList()
        self.action_expert.blocks = torch.nn.ModuleList()

    def restore_packed_blocks_for_legacy_inference(self, visual_tower: VisualTower) -> bool:
        """Move packed-owned blocks back for inference-only legacy cache rollout.

        Packed training transfers block ownership into ``packed_block_stack`` so
        FSDP can shard paired video/action blocks cleanly. Legacy split-cache
        inference needs the pre-packed module lists, so this performs a
        one-way ownership transfer back to ``visual_tower.core.blocks`` and
        ``action_expert.blocks``. ``packed_block_stack`` is cleared afterward
        so the module tree has a single owner for each block.
        """

        if self.packed_block_stack is None:
            return False
        video_blocks = [packed_block.video_block for packed_block in self.packed_block_stack.packed_blocks]
        action_blocks = [packed_block.action_block for packed_block in self.packed_block_stack.packed_blocks]
        if not video_blocks or not action_blocks:
            return False
        visual_tower.core.blocks = torch.nn.ModuleList(video_blocks)
        self.action_expert.blocks = torch.nn.ModuleList(action_blocks)
        self.packed_block_stack = None
        self._legacy_inference_blocks_restored = True
        return True

    def _should_detach_train_video_cache(self, visual_tower: VisualTower) -> bool:
        core_id = id(visual_tower.core)
        detach_cache = self._train_video_cache_detach_by_core_id.get(core_id)
        if detach_cache is None:
            detach_cache = not any(parameter.requires_grad for parameter in visual_tower.core.parameters())
            self._train_video_cache_detach_by_core_id[core_id] = detach_cache
        return bool(detach_cache)

    def _resolve_train_loss_frame_range(
        self,
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
        start_key: str = "loss_frame_start",
        end_key: str = "loss_frame_end",
        fallback_to_generic: bool = True,
    ) -> tuple[int, int] | None:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        if sample_metadata is None:
            return None
        return sample_metadata.optional_frame_range(
            observed_num_frames=observed_num_frames,
            start_key=start_key,
            end_key=end_key,
            fallback_to_generic=fallback_to_generic,
            error_label="MoT train loss-frame metadata",
        )

    def _resolve_train_history_frames(
        self,
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        resolved_history_frames: int | None = None
        if sample_metadata is not None:
            resolved_history_frames = sample_metadata.history_frames
        loss_frame_range = self._resolve_train_loss_frame_range(
            batch=batch,
            observed_num_frames=observed_num_frames,
        )
        if loss_frame_range is not None:
            resolved_history_frames = int(loss_frame_range[0])
        if resolved_history_frames is None:
            resolved_history_frames = int(self.config.video_prefix_frames)
        if resolved_history_frames <= 0 or resolved_history_frames >= observed_num_frames:
            raise ValueError(
                "MoT training requires at least one history frame and one current frame, "
                f"got resolved_history_frames={resolved_history_frames}, observed_num_frames={observed_num_frames}."
            )
        return resolved_history_frames

    def _build_effective_action_mask(
        self,
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> torch.Tensor | None:
        base_mask = batch.action_mask
        loss_frame_range = self._resolve_train_loss_frame_range(
            batch=batch,
            observed_num_frames=observed_num_frames,
            start_key="action_loss_frame_start",
            end_key="action_loss_frame_end",
        )
        if loss_frame_range is None:
            return base_mask
        if batch.actions.shape[1] % max(1, observed_num_frames) != 0:
            return base_mask
        action_per_frame = batch.actions.shape[1] // max(1, observed_num_frames)
        if action_per_frame <= 0:
            return base_mask
        loss_frame_start, loss_frame_end = loss_frame_range
        effective_mask = (
            torch.ones_like(batch.actions, dtype=torch.float32)
            if base_mask is None
            else base_mask.to(dtype=torch.float32)
        )
        frame_mask = torch.zeros_like(effective_mask)
        frame_mask[:, loss_frame_start * action_per_frame : loss_frame_end * action_per_frame] = 1.0
        return effective_mask * frame_mask

    def _build_effective_video_loss_mask(
        self,
        *,
        video_latents: torch.Tensor,
        batch: PolicyTrainBatch,
        default_history_frames: int,
    ) -> torch.Tensor:
        future_loss_mask = torch.zeros(
            video_latents.shape[0],
            1,
            video_latents.shape[2],
            1,
            1,
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        loss_frame_range = self._resolve_train_loss_frame_range(
            batch=batch,
            observed_num_frames=int(video_latents.shape[2]),
            start_key="latent_loss_frame_start",
            end_key="latent_loss_frame_end",
        )
        if loss_frame_range is None:
            future_loss_mask[:, :, default_history_frames:] = 1.0
            return future_loss_mask
        loss_frame_start, loss_frame_end = loss_frame_range
        future_loss_mask[:, :, loss_frame_start:loss_frame_end] = 1.0
        return future_loss_mask

    def _resolve_train_action_tokens_per_frame(
        self,
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int | None:
        if observed_num_frames <= 0:
            return None
        if batch.actions.shape[1] % observed_num_frames != 0:
            return None
        action_tokens_per_frame = batch.actions.shape[1] // observed_num_frames
        return action_tokens_per_frame if action_tokens_per_frame > 0 else None

    def _resolve_train_sampled_chunk_size(
        self,
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int | None:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        if sample_metadata is None:
            return None
        return sample_metadata.sampled_chunk_size_for(observed_num_frames)

    def _resolve_train_sampled_window_size(
        self,
        *,
        batch: PolicyTrainBatch,
    ) -> int | None:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        return None if sample_metadata is None else sample_metadata.sampled_window_size

    def _resolve_train_frame_shift(
        self,
        *,
        batch: PolicyTrainBatch,
    ) -> int:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        if sample_metadata is None or sample_metadata.frame_shift is None:
            return 0
        return int(sample_metadata.frame_shift)

    @staticmethod
    def _resolve_train_chunk_origin_frame(
        *,
        batch: PolicyTrainBatch,
        observed_num_frames: int,
    ) -> int:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        if sample_metadata is None:
            return 0
        if str(sample_metadata.raw.get("target_alignment", "")) != "next_after_context":
            return 0
        loss_frame_start, _ = sample_metadata.frame_range_or_default(
            observed_num_frames=observed_num_frames,
            error_label="M5 train chunk-origin metadata",
        )
        return int(loss_frame_start)

    def _sample_full_segment_train_geometry(
        self,
        *,
        observed_num_frames: int,
        device: torch.device,
    ) -> tuple[int, int, int]:
        # FULL_SEGMENT data path: data adapter does not pre-sample chunk/window
        # geometry, so the variant draws it per-step the same way method-1 does
        # in `prepare_lingbot_parallel_train_artifacts` — chunk_size in
        # [1, training_config.chunk_size] and window_size in
        # [4, training_config.window_size]. history_frames is then drawn
        # uniformly over chunk-aligned positions inside the episode so the
        # action expert sees every (history_len, current_chunk) pair.
        cs_max = max(1, int(self.training_config.chunk_size))
        sampled_chunk_size = int(torch.randint(1, cs_max + 1, (1,), device=device).item())
        if int(self.training_config.window_size) >= 4:
            sampled_window_size = int(
                torch.randint(4, int(self.training_config.window_size) + 1, (1,), device=device).item()
            )
        else:
            sampled_window_size = max(1, int(self.training_config.window_size))
        max_history_chunks = max(1, observed_num_frames // sampled_chunk_size - 1)
        history_chunks = int(torch.randint(1, max_history_chunks + 1, (1,), device=device).item())
        history_frames = max(1, min(history_chunks * sampled_chunk_size, observed_num_frames - sampled_chunk_size))
        return sampled_chunk_size, sampled_window_size, history_frames

    def _build_action_grid_ids_for_sequence(
        self,
        *,
        batch_size: int,
        seq_len: int,
        action_tokens_per_frame: int,
        device: torch.device,
        frame_shift: int,
    ) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError(f"Expected positive action seq_len, got {seq_len}.")
        if action_tokens_per_frame <= 0 or seq_len % action_tokens_per_frame != 0:
            raise ValueError(
                "MoT action grid ids require `seq_len` to divide by `action_tokens_per_frame`, "
                f"got seq_len={seq_len}, action_tokens_per_frame={action_tokens_per_frame}."
            )
        num_frames = seq_len // action_tokens_per_frame
        return build_action_grid_ids(
            num_frames=num_frames,
            action_per_frame=action_tokens_per_frame,
            device=device,
            frame_shift=float(frame_shift),
        )[None].expand(batch_size, -1, -1)

    def _apply_train_history_action_condition(
        self,
        *,
        train_artifacts,
        actions: torch.Tensor,
        observed_num_frames: int,
        history_frames: int,
    ):
        if observed_num_frames <= 0 or actions.shape[1] % observed_num_frames != 0:
            return train_artifacts
        action_tokens_per_frame = actions.shape[1] // observed_num_frames
        if action_tokens_per_frame <= 0:
            return train_artifacts
        history_action_tokens = int(history_frames * action_tokens_per_frame)
        if history_action_tokens <= 0:
            return train_artifacts
        history_action_tokens = min(history_action_tokens, int(actions.shape[1]))
        train_artifacts.noisy_actions[:, :history_action_tokens] = actions[:, :history_action_tokens]
        train_artifacts.timesteps[:, :history_action_tokens] = 0.0
        return train_artifacts

    def _build_video_train_rollout(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
        history_frames: int,
        condition_latents: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> MoTVideoTrainArtifacts:
        video_latents = visual_outputs.frontend.video_latents
        clean_condition_latents, _ = self._train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )
        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_condition_latents,
        )
        noisy_latents = video_artifacts.noisy_latents.clone()
        timesteps = video_artifacts.timesteps.clone()
        history_condition_latents = clean_condition_latents if clean_condition_latents is not None else video_latents
        noisy_latents[:, :, :history_frames] = history_condition_latents[:, :, :history_frames]
        timesteps[:, :history_frames] = 0.0
        future_loss_mask = self._build_effective_video_loss_mask(
            video_latents=video_latents,
            batch=batch,
            default_history_frames=history_frames,
        )
        flow_pred = visual_tower.predict_video_flow(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            text_context=visual_outputs.frontend.conditioning.text_context,
            frame_start=0,
            attention_mask=attention_mask,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=noisy_latents,
            flow_pred=flow_pred,
            timesteps=timesteps,
            scheduler=video_artifacts.scheduler,
        )
        return MoTVideoTrainArtifacts(
            flow_pred=flow_pred,
            targets=video_artifacts.targets,
            timesteps=timesteps,
            scheduler=video_artifacts.scheduler,
            predicted_latents=predicted_latents,
            target_latents=video_latents,
            future_loss_mask=future_loss_mask,
        )

    def _maybe_initialize_action_expert(self, visual_tower: VisualTower) -> None:
        if self._action_expert_initialized:
            return
        init_action_expert_from_video_core(
            action_expert=self.action_expert,
            video_core=visual_tower.core,
            mode=str(self.config.action_expert_init_mode),
        )
        self._action_expert_initialized = True

    def initialize_for_training(self, visual_tower: VisualTower) -> None:
        self._maybe_initialize_action_expert(visual_tower)

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend",)

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        condition_latents = self._resolve_train_condition_latents(
            batch,
            video_latents=visual_outputs.frontend.video_latents,
        )
        proprio_state = self._resolve_train_proprio_context(batch)
        hidden_proprio_state = self._resolve_train_hidden_proprio_context(batch)
        return PolicyPreparedInputs(
            batch=batch,
            variant_inputs={
                "video_latents": visual_outputs.frontend.video_latents,
                "condition_latents": condition_latents,
                "proprio_state": proprio_state,
                "hidden_proprio_state": hidden_proprio_state,
                "text_context": visual_outputs.frontend.conditioning.text_context,
                "video_tokens_per_frame": visual_outputs.frontend.token_grid.tokens_per_frame,
            },
        )

    def _resolve_train_condition_latents(
        self,
        batch: PolicyTrainBatch,
        *,
        video_latents: torch.Tensor,
    ) -> torch.Tensor | None:
        if not bool(self.config.use_condition_latents):
            return None
        condition_latents = batch.extra.get("condition_latents")
        if condition_latents is None:
            if bool(self.config.require_condition_latents):
                raise ValueError(
                    "M5 training was configured with `require_condition_latents=true`, "
                    "but the latent batch did not provide `condition_latents`."
                )
            return None
        if not isinstance(condition_latents, torch.Tensor):
            raise ValueError(
                "M5 `condition_latents` must be a tensor when provided, "
                f"got {type(condition_latents).__name__}."
            )
        if condition_latents.ndim != 5 or tuple(condition_latents.shape) != tuple(video_latents.shape):
            raise ValueError(
                "M5 `condition_latents` must match video_latents exactly for train-time video conditioning, "
                f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
            )
        return condition_latents.to(device=video_latents.device, dtype=video_latents.dtype)

    @staticmethod
    def _video_condition_source(condition_latents: torch.Tensor | None) -> str:
        return "condition_latents" if condition_latents is not None else "video_latents"

    def _context_condition_latent_source(self) -> ParallelContextConditionLatentSource:
        return ParallelContextConditionLatentSource(self.config.context_condition_latent_source)

    def _train_clean_video_condition_latents(
        self,
        *,
        video_latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        history_frames: int,
    ) -> tuple[torch.Tensor | None, str]:
        if self._context_condition_latent_source() != ParallelContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT:
            return condition_latents, self._video_condition_source(condition_latents)
        if condition_latents is None:
            raise ValueError(
                "M5 `context_condition_latent_source=single_frame_condition_latent` requires `condition_latents`."
            )
        if history_frames <= 0:
            raise ValueError(
                "M5 single-frame condition latents require at least one history/context frame, "
                f"got history_frames={history_frames}."
            )
        clean_condition = video_latents.clone()
        clean_condition[:, :, : int(history_frames)] = condition_latents[:, :, : int(history_frames)].to(
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        return clean_condition, "context_condition_latents"

    def _prepend_legacy_prefix_video_latents(
        self,
        *,
        video_latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        hidden_proprio_state: torch.Tensor | None,
        batch: PolicyTrainBatch,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, str]:
        if not _uses_mot_legacy_prefix_contract(self.config):
            return video_latents, hidden_proprio_state, 0, self._video_condition_source(condition_latents)
        if condition_latents is None:
            raise ValueError(
                "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` requires "
                "precomputed single-frame condition_latents for M5. "
                "Run scripts/augment_lerobot_latents_with_single_frame_condition.py with --source-frame-offset -1."
            )
        if condition_latents.ndim != 5 or int(condition_latents.shape[2]) < 1:
            raise ValueError(
                "M5 legacy-prefix condition_latents must have shape [B, C, T>=1, H, W], "
                f"got {tuple(condition_latents.shape)}."
            )
        prefix_latents = condition_latents[:, :, :1].to(device=video_latents.device, dtype=video_latents.dtype)
        model_video_latents = torch.cat([prefix_latents, video_latents], dim=2)
        if hidden_proprio_state is not None:
            if hidden_proprio_state.ndim != 3:
                raise ValueError(
                    "M5 legacy-prefix per-chunk proprio expects target frame states with shape "
                    "[B, target_frames, state_dim], "
                    f"got {tuple(hidden_proprio_state.shape)}."
                )
            prefix_state = self._select_anchor_state(batch.state)
            if prefix_state is None:
                raise ValueError(
                    "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` requires "
                    "batch.state for the prefix/current proprio frame."
                )
            target_frames = int(video_latents.shape[2])
            if int(hidden_proprio_state.shape[1]) < target_frames:
                raise ValueError(
                    "M5 legacy-prefix per-chunk proprio expects at least one state per target frame, "
                    f"got {tuple(hidden_proprio_state.shape)} for target_frames={target_frames}."
                )
            hidden_proprio_state = torch.cat(
                [
                    prefix_state[:, None, :].to(
                        device=hidden_proprio_state.device,
                        dtype=hidden_proprio_state.dtype,
                    ),
                    hidden_proprio_state[:, :target_frames, :],
                ],
                dim=1,
            )
        return model_video_latents, hidden_proprio_state, 1, "condition_latents_prefix"

    @staticmethod
    def _legacy_prefix_action_hidden_proprio_state(
        hidden_proprio_state: torch.Tensor | None,
        *,
        prefix_condition_frames: int,
        target_num_frames: int,
        chunk_size_frames: int,
    ) -> torch.Tensor | None:
        if hidden_proprio_state is None or int(prefix_condition_frames) <= 0:
            return hidden_proprio_state
        if hidden_proprio_state.ndim != 3:
            raise ValueError(
                "M5 legacy-prefix per-chunk proprio expects frame state shape "
                "[B, prefix_plus_target_frames, state_dim], "
                f"got {tuple(hidden_proprio_state.shape)}."
            )
        required_frames = int(prefix_condition_frames) + int(target_num_frames)
        if int(hidden_proprio_state.shape[1]) < required_frames:
            raise ValueError(
                "M5 legacy-prefix per-chunk proprio expects prefix plus target frame states, "
                f"got {tuple(hidden_proprio_state.shape)} for required_frames={required_frames}."
            )
        chunk_size = max(1, int(chunk_size_frames))
        target_frame_ids = torch.arange(
            int(target_num_frames),
            device=hidden_proprio_state.device,
            dtype=torch.long,
        )
        target_boundary_ids = torch.div(target_frame_ids, chunk_size, rounding_mode="floor") * chunk_size
        target_boundary_state = hidden_proprio_state.index_select(dim=1, index=target_boundary_ids)
        return target_boundary_state

    def _resolve_history_stream_visibility(self) -> ParallelHistoryStreamVisibility:
        return ParallelHistoryStreamVisibility(self.config.history_stream_visibility)

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        self._maybe_initialize_action_expert(visual_tower)
        if self.config.current_block_coupling is not None:
            return self._forward_train_packed_coupling(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                prepared_inputs=prepared_inputs,
            )
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            return self._forward_train_joint_denoise(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                prepared_inputs=prepared_inputs,
            )
        if self.config.runtime_mode == MoTRuntimeMode.NON_JOINT_TWO_STREAM:
            return self._forward_train_non_joint_two_stream(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                prepared_inputs=prepared_inputs,
            )
        return self._forward_train_prefill_action_denoise(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
        )

    def _forward_train_prefill_action_denoise(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        self._maybe_initialize_action_expert(visual_tower)
        video_latents = prepared_inputs.variant_inputs["video_latents"]
        condition_latents = prepared_inputs.variant_inputs.get("condition_latents")
        text_context = prepared_inputs.variant_inputs["text_context"]
        proprio_state = prepared_inputs.variant_inputs.get("proprio_state")
        hidden_proprio_state = prepared_inputs.variant_inputs.get("hidden_proprio_state")
        history_frames = self._resolve_train_history_frames(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        clean_video_condition_latents, video_condition_source = self._train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )
        video_train_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
        )
        effective_action_mask = self._build_effective_action_mask(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        action_tokens_per_frame = self._resolve_train_action_tokens_per_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        video_tokens_per_frame = int(prepared_inputs.variant_inputs["video_tokens_per_frame"])
        sampled_chunk_size = self._resolve_train_sampled_chunk_size(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        if sampled_chunk_size is None:
            sampled_chunk_size = max(
                1,
                min(int(self.training_config.chunk_size), int(video_latents.shape[2])),
            )
        sampled_window_size = self._resolve_train_sampled_window_size(
            batch=prepared_inputs.batch,
        )
        if sampled_window_size is None:
            sampled_window_size = max(1, int(self.training_config.window_size))
        frame_shift = self._resolve_train_frame_shift(batch=prepared_inputs.batch)
        chunk_origin_frame = self._resolve_train_chunk_origin_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        chunk_causal_video_mask = build_chunk_causal_video_mask(
            video_seq_len=video_tokens_per_frame * int(video_latents.shape[2]),
            video_tokens_per_frame=video_tokens_per_frame,
            action_chunk_size_frames=sampled_chunk_size,
            device=video_latents.device,
            attention_window_size=sampled_window_size,
            chunk_origin_frame=chunk_origin_frame,
        )

        # Method-5 video-prefill action denoise is aligned to method-1 full-seg
        # semantics: the action expert conditions on the full clean video
        # sample, but the video K/V prefill itself stays chunk-causal so future
        # chunks do not leak through the shared video backbone.
        action_condition_latents = (
            clean_video_condition_latents if clean_video_condition_latents is not None else video_latents
        )
        video_text_context = self._resolve_text_context_with_proprio(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(action_condition_latents.shape[0]),
            device=action_condition_latents.device,
            dtype=action_condition_latents.dtype,
            materialize_if_missing=self._uses_proprio_context(),
        )
        video_cross_attention_mask = self._build_proprio_cross_attention_mask(
            resolved_text_context=video_text_context,
            proprio_state=proprio_state,
            query_frames_per_copy=int(action_condition_latents.shape[2]),
            tokens_per_frame=video_tokens_per_frame,
            chunk_size_frames=sampled_chunk_size,
            chunk_origin_frame=chunk_origin_frame,
        ) if video_text_context is not None else None
        video_cache = prefill_video_kv_cache(
            visual_tower=visual_tower,
            observed_prefix=action_condition_latents,
            text_context=video_text_context,
            frame_start=0,
            attention_mask=chunk_causal_video_mask,
            cross_attention_mask=video_cross_attention_mask,
            detach_cache=self._should_detach_train_video_cache(visual_tower),
        )
        train_artifacts = build_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            effective_action_mask,
            training_config=self.training_config,
        )
        train_artifacts = self._apply_train_history_action_condition(
            train_artifacts=train_artifacts,
            actions=prepared_inputs.batch.actions,
            observed_num_frames=int(video_latents.shape[2]),
            history_frames=history_frames,
        )
        resolved_text = self._resolve_text_context_with_proprio(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(action_condition_latents.shape[0]),
            device=action_condition_latents.device,
            dtype=action_condition_latents.dtype,
            materialize_if_missing=True,
        )
        if resolved_text is None:  # pragma: no cover - materialized above
            raise RuntimeError("M5 action text context unexpectedly resolved to None.")
        action_cross_attention_mask = (
            None
            if action_tokens_per_frame is None
            else self._build_proprio_cross_attention_mask(
                resolved_text_context=resolved_text,
                proprio_state=proprio_state,
                query_frames_per_copy=int(video_latents.shape[2]),
                tokens_per_frame=int(action_tokens_per_frame),
                chunk_size_frames=sampled_chunk_size,
                chunk_origin_frame=chunk_origin_frame,
            )
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=train_artifacts.noisy_actions,
            timestep=train_artifacts.timesteps,
            context=resolved_text,
            cross_attention_mask=action_cross_attention_mask,
            action_grid_ids=self._build_action_grid_ids_for_sequence(
                batch_size=train_artifacts.noisy_actions.shape[0],
                seq_len=train_artifacts.noisy_actions.shape[1],
                action_tokens_per_frame=action_tokens_per_frame,
                device=train_artifacts.noisy_actions.device,
                frame_shift=frame_shift,
            ) if action_tokens_per_frame is not None else None,
            hidden_context=(
                None
                if action_tokens_per_frame is None
                else self._action_hidden_context_for_tokens(
                    visual_tower,
                    hidden_proprio_state,
                    action_tokens=train_artifacts.noisy_actions,
                    action_tokens_per_frame=int(action_tokens_per_frame),
                    chunk_size_frames=sampled_chunk_size,
                )
            ),
        )
        action_hidden_states = forward_action_with_video_cache(
            action_expert=self.action_expert,
            action_pre=action_pre,
            video_cache=video_cache,
            attention_mask=build_mot_attention_mask(
                video_seq_len=video_cache.video_seq_len,
                action_seq_len=train_artifacts.noisy_actions.shape[1],
                device=train_artifacts.noisy_actions.device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=prepared_inputs.variant_inputs["video_tokens_per_frame"],
                action_tokens_per_frame=action_tokens_per_frame,
                action_chunk_size_frames=sampled_chunk_size,
                clean_video_frames=int(video_latents.shape[2]),
                clean_action_frames=history_frames,
                attention_window_size=sampled_window_size,
            ),
        )
        flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=train_artifacts.noisy_actions,
            flow_pred=flow_pred,
            timesteps=train_artifacts.timesteps,
            scheduler=train_artifacts.scheduler,
        )
        video_rollout: MoTVideoTrainArtifacts | None = None
        if self.training_config.objective_enabled("latent"):
            video_rollout = self._build_video_train_rollout(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                batch=prepared_inputs.batch,
                history_frames=history_frames,
                condition_latents=condition_latents,
                attention_mask=chunk_causal_video_mask,
            )
        batch_size = action_condition_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=action_condition_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_history_frames": action_condition_latents.new_tensor(float(history_frames)),
                "mot_video_prefix_frames": action_condition_latents.new_tensor(float(history_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "video_cache_seq_len": video_cache.video_seq_len,
                "runtime_mode": str(self.config.runtime_mode),
                "sampled_chunk_size": sampled_chunk_size,
                "sampled_window_size": sampled_window_size,
                "video_condition_source": video_condition_source,
                "mot_train_artifacts": MoTTrainArtifacts(
                    action=MoTActionTrainArtifacts(
                        flow_pred=flow_pred,
                        targets=train_artifacts.targets,
                        timesteps=train_artifacts.timesteps,
                        scheduler=train_artifacts.scheduler,
                        denoised_actions=denoised_actions,
                        action_mask=train_artifacts.action_mask,
                    ),
                    video=video_rollout,
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                    history_frames=int(history_frames),
                    video_cache_seq_len=video_cache.video_seq_len,
                ),
            },
        )

    def _forward_train_joint_denoise(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        video_latents = prepared_inputs.variant_inputs["video_latents"]
        condition_latents = prepared_inputs.variant_inputs.get("condition_latents")
        text_context = prepared_inputs.variant_inputs["text_context"]
        proprio_state = prepared_inputs.variant_inputs.get("proprio_state")
        hidden_proprio_state = prepared_inputs.variant_inputs.get("hidden_proprio_state")
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        if not _is_mot_same_step_coupling(current_block_coupling):
            raise NotImplementedError(
                "M5 joint_denoise train supports same-step couplings only; "
                f"got current_block_coupling={current_block_coupling.value!r}. "
                "Use runtime_mode='non_joint_two_stream' for staged video_then_action."
            )
        history_frames = self._resolve_train_history_frames(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        clean_video_condition_latents, video_condition_source = self._train_clean_video_condition_latents(
            video_latents=video_latents,
            condition_latents=condition_latents,
            history_frames=history_frames,
        )

        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
        )
        noisy_video_latents = video_artifacts.noisy_latents.clone()
        video_timesteps = video_artifacts.timesteps.clone()
        history_condition_latents = clean_video_condition_latents if clean_video_condition_latents is not None else video_latents
        noisy_video_latents[:, :, :history_frames] = history_condition_latents[:, :, :history_frames]
        video_timesteps[:, :history_frames] = 0.0
        future_loss_mask = self._build_effective_video_loss_mask(
            video_latents=video_latents,
            batch=prepared_inputs.batch,
            default_history_frames=history_frames,
        )
        effective_action_mask = self._build_effective_action_mask(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        action_tokens_per_frame = self._resolve_train_action_tokens_per_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        sampled_chunk_size = self._resolve_train_sampled_chunk_size(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )
        sampled_window_size = self._resolve_train_sampled_window_size(
            batch=prepared_inputs.batch,
        )
        if sampled_chunk_size is None:
            sampled_chunk_size = max(1, int(self.training_config.chunk_size))
        if sampled_window_size is None:
            sampled_window_size = max(1, int(self.training_config.window_size))
        frame_shift = self._resolve_train_frame_shift(batch=prepared_inputs.batch)
        chunk_origin_frame = self._resolve_train_chunk_origin_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=int(video_latents.shape[2]),
        )

        train_artifacts = build_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            effective_action_mask,
            training_config=self.training_config,
        )
        resolved_text = self._resolve_text_context_with_proprio(
            visual_tower,
            text_context,
            proprio_state,
            batch_size=int(video_latents.shape[0]),
            device=video_latents.device,
            dtype=video_latents.dtype,
            materialize_if_missing=True,
        )
        if resolved_text is None:  # pragma: no cover - materialized above
            raise RuntimeError("M5 joint action text context unexpectedly resolved to None.")
        action_cross_attention_mask = (
            None
            if action_tokens_per_frame is None
            else self._build_proprio_cross_attention_mask(
                resolved_text_context=resolved_text,
                proprio_state=proprio_state,
                query_frames_per_copy=int(video_latents.shape[2]),
                tokens_per_frame=int(action_tokens_per_frame),
                chunk_size_frames=sampled_chunk_size,
            )
        )
        video_cross_attention_mask = self._build_proprio_cross_attention_mask(
            resolved_text_context=resolved_text,
            proprio_state=proprio_state,
            query_frames_per_copy=int(video_latents.shape[2]),
            tokens_per_frame=int(prepared_inputs.variant_inputs["video_tokens_per_frame"]),
            chunk_size_frames=sampled_chunk_size,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=train_artifacts.noisy_actions,
            timestep=train_artifacts.timesteps,
            context=resolved_text,
            cross_attention_mask=action_cross_attention_mask,
            action_grid_ids=self._build_action_grid_ids_for_sequence(
                batch_size=train_artifacts.noisy_actions.shape[0],
                seq_len=train_artifacts.noisy_actions.shape[1],
                action_tokens_per_frame=action_tokens_per_frame,
                device=train_artifacts.noisy_actions.device,
                frame_shift=frame_shift,
            ) if action_tokens_per_frame is not None else None,
            hidden_context=(
                None
                if action_tokens_per_frame is None
                else self._action_hidden_context_for_tokens(
                    visual_tower,
                    hidden_proprio_state,
                    action_tokens=train_artifacts.noisy_actions,
                    action_tokens_per_frame=int(action_tokens_per_frame),
                    chunk_size_frames=sampled_chunk_size,
                )
            ),
        )
        video_flow_pred, action_hidden_states = forward_joint_video_action_denoise(
            visual_tower=visual_tower,
            noisy_video_latents=noisy_video_latents,
            video_timesteps=video_timesteps,
            action_expert=self.action_expert,
            action_pre=action_pre,
            text_context=resolved_text,
            attention_mask=build_mot_attention_mask(
                video_seq_len=prepared_inputs.variant_inputs["video_tokens_per_frame"] * video_latents.shape[2],
                action_seq_len=train_artifacts.noisy_actions.shape[1],
                device=train_artifacts.noisy_actions.device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=prepared_inputs.variant_inputs["video_tokens_per_frame"],
                video_can_attend_action=self.config.video_can_attend_action,
                action_tokens_per_frame=action_tokens_per_frame,
                action_chunk_size_frames=sampled_chunk_size,
                clean_video_frames=history_frames,
                attention_window_size=sampled_window_size,
                current_block_coupling=current_block_coupling,
            ),
            use_activation_checkpointing=self.config.use_activation_checkpointing,
            video_cross_attention_mask=video_cross_attention_mask,
            video_hidden_context=self._video_hidden_context_for_tokens(
                visual_tower,
                hidden_proprio_state,
                video_latents=video_latents,
                chunk_size_frames=sampled_chunk_size,
            ),
        )
        flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=train_artifacts.noisy_actions,
            flow_pred=flow_pred,
            timesteps=train_artifacts.timesteps,
            scheduler=train_artifacts.scheduler,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=noisy_video_latents,
            flow_pred=video_flow_pred,
            timesteps=video_timesteps,
            scheduler=video_artifacts.scheduler,
        )
        batch_size = video_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=video_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_history_frames": video_latents.new_tensor(float(history_frames)),
                "mot_video_prefix_frames": video_latents.new_tensor(float(history_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "runtime_mode": str(self.config.runtime_mode),
                "current_block_coupling": current_block_coupling.value,
                "sampled_chunk_size": sampled_chunk_size,
                "sampled_window_size": sampled_window_size,
                "video_condition_source": video_condition_source,
                "mot_train_artifacts": MoTTrainArtifacts(
                    action=MoTActionTrainArtifacts(
                        flow_pred=flow_pred,
                        targets=train_artifacts.targets,
                        timesteps=train_artifacts.timesteps,
                        scheduler=train_artifacts.scheduler,
                        denoised_actions=denoised_actions,
                        action_mask=train_artifacts.action_mask,
                    ),
                    video=MoTVideoTrainArtifacts(
                        flow_pred=video_flow_pred,
                        targets=video_artifacts.targets,
                        timesteps=video_timesteps,
                        scheduler=video_artifacts.scheduler,
                        predicted_latents=predicted_latents,
                        target_latents=video_latents,
                        future_loss_mask=future_loss_mask,
                    ),
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                    history_frames=int(history_frames),
                ),
            },
        )

    def _forward_train_packed_coupling(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        # Method-1-style four-branch packed training for M5's two-expert
        # architecture. Query/key layout is [V_noisy, V_clean, A_noisy,
        # A_clean]; the coupling mask determines current-chunk visibility for
        # all six modes while both experts remain separate transformer stacks.
        self._maybe_initialize_action_expert(visual_tower)
        video_latents = prepared_inputs.variant_inputs["video_latents"]
        condition_latents = prepared_inputs.variant_inputs.get("condition_latents")
        text_context = prepared_inputs.variant_inputs["text_context"]
        proprio_state = prepared_inputs.variant_inputs.get("proprio_state")
        hidden_proprio_state = prepared_inputs.variant_inputs.get("hidden_proprio_state")
        video_tokens_per_frame = int(prepared_inputs.variant_inputs["video_tokens_per_frame"])
        target_video_latents = video_latents
        target_num_video_frames = int(target_video_latents.shape[2])
        num_video_frames = target_num_video_frames
        # Geometry resolution: contextual_subwindow data path stamps
        # `sampled_chunk_size` etc. into per-sample metadata; FULL_SEGMENT
        # data path leaves it unset, so we draw it per-step here the same way
        # method-1's `prepare_lingbot_parallel_train_artifacts` does.
        metadata_for_geometry = prepared_inputs.batch.extra.get("metadata")
        metadata_has_geometry = (
            isinstance(metadata_for_geometry, tuple)
            and len(metadata_for_geometry) > 0
            and metadata_for_geometry[0].get("sampled_chunk_size") is not None
        )
        if metadata_has_geometry:
            history_frames = self._resolve_train_history_frames(
                batch=prepared_inputs.batch,
                observed_num_frames=target_num_video_frames,
            )
            sampled_chunk_size = self._resolve_train_sampled_chunk_size(
                batch=prepared_inputs.batch,
                observed_num_frames=target_num_video_frames,
            )
            sampled_window_size = self._resolve_train_sampled_window_size(
                batch=prepared_inputs.batch,
            )
        else:
            sampled_chunk_size, sampled_window_size, history_frames = (
                self._sample_full_segment_train_geometry(
                    observed_num_frames=target_num_video_frames,
                    device=video_latents.device,
                )
            )
        video_latents, hidden_proprio_state, prefix_condition_frames, legacy_video_condition_source = (
            self._prepend_legacy_prefix_video_latents(
                video_latents=target_video_latents,
                condition_latents=condition_latents,
                hidden_proprio_state=hidden_proprio_state,
                batch=prepared_inputs.batch,
            )
        )
        num_video_frames = int(video_latents.shape[2])
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        effective_action_mask = self._build_effective_action_mask(
            batch=prepared_inputs.batch,
            observed_num_frames=target_num_video_frames,
        )
        clean_action_condition_mask = prepared_inputs.batch.action_mask
        action_tokens_per_frame = self._resolve_train_action_tokens_per_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=target_num_video_frames,
        )
        frame_shift = self._resolve_train_frame_shift(batch=prepared_inputs.batch)
        chunk_origin_frame = self._resolve_train_chunk_origin_frame(
            batch=prepared_inputs.batch,
            observed_num_frames=target_num_video_frames,
        )

        if action_tokens_per_frame is None:
            raise ValueError(
                "MoT non_joint_two_stream packed training requires "
                "`action_tokens_per_frame` resolvable from the batch, got None."
            )
        if sampled_chunk_size is None:
            raise ValueError(
                "MoT non_joint_two_stream packed training requires "
                "`sampled_chunk_size` resolvable from the batch metadata or full-segment fallback, got None."
            )

        sampled_generalist_mode: MoTGeneralistTrainingMode | None = None
        forced_generalist_mode, metadata_drop_text, generalist_source = _resolve_mot_generalist_training_metadata(
            prepared_inputs.batch
        )
        generalist_probs = self.config.mot_generalist_training_mode_probs
        if forced_generalist_mode is not None:
            sampled_generalist_mode = forced_generalist_mode
        elif generalist_probs is not None:
            sampled_generalist_mode = _sample_mot_generalist_training_mode(
                generalist_probs,
                device=video_latents.device,
            )
        if sampled_generalist_mode is not None and int(video_latents.shape[0]) != 1:
            raise ValueError(
                "M5 generalist joint denoising currently requires rank-local train_batch_size=1 because "
                "one GJD mode is sampled/applied per segment forward and per-sample forced modes are only "
                f"unambiguous for batch size 1; got batch_size={int(video_latents.shape[0])}."
            )

        joint_timestep_coupling = _resolve_mot_joint_timestep_coupling(
            self.config,
            current_block_coupling,
        )
        shared_timestep_ids = None
        if joint_timestep_coupling in {
            JointTimestepCoupling.MATCH_INDEX,
            JointTimestepCoupling.SHARED_VIDEO_SCHEDULE,
        }:
            if int(self.training_config.video_num_train_timesteps) != int(self.training_config.action_num_train_timesteps):
                if joint_timestep_coupling == JointTimestepCoupling.MATCH_INDEX:
                    raise ValueError(
                        "M5 index-matched joint denoising requires equal video/action train timestep counts, "
                        f"got video={self.training_config.video_num_train_timesteps}, "
                        f"action={self.training_config.action_num_train_timesteps}."
                    )
            shared_timestep_ids = sample_timestep_id(
                batch_size=int(video_latents.shape[0]),
                sample_shape=(num_video_frames,),
                num_train_timesteps=int(self.training_config.video_num_train_timesteps),
                device=video_latents.device,
            )
        if prefix_condition_frames > 0:
            clean_video_condition_latents = video_latents
            video_condition_source = legacy_video_condition_source
        else:
            clean_video_condition_latents, video_condition_source = self._train_clean_video_condition_latents(
                video_latents=video_latents,
                condition_latents=condition_latents,
                history_frames=history_frames,
            )

        video_artifacts = build_video_flow_match_train_artifacts(
            video_latents,
            training_config=self.training_config,
            condition_latents=clean_video_condition_latents,
            timestep_ids=shared_timestep_ids,
            noisy_condition_prob=0.0
            if _mot_generalist_forces_clean_video_condition(sampled_generalist_mode)
            else float(self.config.noisy_video_condition_prob),
        )
        if prefix_condition_frames > 0:
            prefix_latents = video_latents[:, :, :prefix_condition_frames]
            video_artifacts.noisy_latents[:, :, :prefix_condition_frames] = prefix_latents
            video_artifacts.condition_latents[:, :, :prefix_condition_frames] = prefix_latents
            video_artifacts.targets[:, :, :prefix_condition_frames] = 0
            video_artifacts.timesteps[:, :prefix_condition_frames] = 0.0
            video_artifacts.condition_timesteps[:, :prefix_condition_frames] = 0.0
        coupled_action_sigma_values = (
            frame_sigmas_for_timesteps(video_artifacts.scheduler, video_artifacts.timesteps[:, prefix_condition_frames:])
            if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA
            else None
        )
        action_scheduler_override = (
            video_artifacts.scheduler
            if joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE
            else None
        )
        future_loss_mask = self._build_effective_video_loss_mask(
            video_latents=video_latents,
            batch=prepared_inputs.batch,
            default_history_frames=history_frames,
        )
        if prefix_condition_frames > 0:
            future_loss_mask.zero_()
            future_loss_mask[:, :, prefix_condition_frames:] = 1.0
        action_artifacts = build_frame_aligned_action_flow_match_train_artifacts(
            prepared_inputs.batch.actions,
            effective_action_mask,
            training_config=self.training_config,
            num_frames=target_num_video_frames,
            action_per_frame=int(action_tokens_per_frame),
            frame_sigma_values=coupled_action_sigma_values,
            frame_timestep_ids=(
                shared_timestep_ids[:, prefix_condition_frames:]
                if shared_timestep_ids is not None and prefix_condition_frames > 0
                else shared_timestep_ids
            ),
            scheduler_override=action_scheduler_override,
        )
        noisy_actions = action_artifacts.noisy_actions
        clean_actions = action_artifacts.condition_actions.to(
            device=noisy_actions.device, dtype=noisy_actions.dtype
        )
        if clean_actions.shape != noisy_actions.shape:
            raise ValueError(
                "Packed action training requires noisy/clean actions to share shape, "
                f"got noisy={tuple(noisy_actions.shape)}, clean={tuple(clean_actions.shape)}."
            )
        action_seq_len = int(noisy_actions.shape[1])
        num_action_frames = action_seq_len // int(action_tokens_per_frame)

        # Per-token timesteps broadcast from the per-frame sample (matches
        # Method 1's `_time_embed` repeat-interleave of per-frame timesteps).
        noisy_slot_timesteps = action_artifacts.slot_timesteps

        # ---- A1 generalist mode sampling (strict M1 PR #95 parity) ----
        # When ``mot_generalist_training_mode_probs`` is set, sample one
        # regime per segment. Sampling lives at the segment top so the same
        # mode flows through every layer / block of this forward; it must
        # NOT be re-sampled at block granularity (would break attention
        # profile cache + cause same-step layers to disagree).
        if sampled_generalist_mode is not None:
            generalist_semantics = resolve_generalist_joint_conditioning_semantics(
                sampled_generalist_mode,
                joint_mode=MoTGeneralistTrainingMode.JOINT,
                action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
                video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
                drop_text_conditioning=metadata_drop_text,
            )
            (
                video_artifacts,
                noisy_actions,
                clean_actions,
                noisy_slot_timesteps,
                future_loss_mask,
                effective_action_mask,
            ) = _apply_mot_generalist_training_mode(
                sampled_mode=sampled_generalist_mode,
                video_artifacts=video_artifacts,
                noisy_actions=noisy_actions,
                clean_actions=clean_actions,
                noisy_slot_timesteps=noisy_slot_timesteps,
                future_loss_mask=future_loss_mask,
                effective_action_mask=effective_action_mask,
                clean_action_condition_mask=clean_action_condition_mask,
            )
            if generalist_semantics.is_conditional:
                # Match the M1 GJD conditional contract: FDM/IDM are local
                # dynamics probes. Keep real tokens intact, but restrict K/V
                # visibility to one immediate history chunk through the packed
                # attention window.
                sampled_window_size = generalist_semantics.attention_window_size(
                    fallback_window_size=sampled_window_size,
                )

        packed_action_tokens = torch.cat([noisy_actions, clean_actions], dim=1)
        action_hidden_proprio_state = self._legacy_prefix_action_hidden_proprio_state(
            hidden_proprio_state,
            prefix_condition_frames=prefix_condition_frames,
            target_num_frames=target_num_video_frames,
            chunk_size_frames=sampled_chunk_size,
        )
        packed_action_hidden_context = self._action_hidden_context_for_tokens(
            visual_tower,
            action_hidden_proprio_state,
            action_tokens=noisy_actions,
            action_tokens_per_frame=int(action_tokens_per_frame),
            copies=2,
            chunk_size_frames=sampled_chunk_size,
        )
        clean_slot_timesteps = torch.zeros_like(noisy_slot_timesteps)
        packed_action_timesteps = torch.cat(
            [noisy_slot_timesteps, clean_slot_timesteps], dim=1
        )

        text_dropped = False
        if sampled_generalist_mode is not None:
            text_dropped = resolve_generalist_joint_conditioning_semantics(
                sampled_generalist_mode,
                joint_mode=MoTGeneralistTrainingMode.JOINT,
                action_conditioned_video_mode=MoTGeneralistTrainingMode.ACTION_CONDITIONED_VIDEO,
                video_conditioned_action_mode=MoTGeneralistTrainingMode.VIDEO_CONDITIONED_ACTION,
                drop_text_conditioning=metadata_drop_text,
            ).drop_text_conditioning
        resolved_text = text_context
        if resolved_text is None:
            resolved_text = video_latents.new_zeros(
                video_latents.shape[0],
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
            )
        elif text_dropped:
            resolved_text = torch.zeros_like(resolved_text)
        resolved_text = self._resolve_text_context_with_proprio(
            visual_tower,
            resolved_text,
            proprio_state,
            batch_size=int(video_latents.shape[0]),
            device=video_latents.device,
            dtype=video_latents.dtype,
            materialize_if_missing=True,
        )
        if resolved_text is None:  # pragma: no cover - materialized above
            raise RuntimeError("M5 packed text context unexpectedly resolved to None.")
        generalist_mode_text_token_count = 0
        if bool(getattr(self.config, "generalist_mode_text_token", False)):
            if sampled_generalist_mode is None:
                raise ValueError(
                    "MoT `generalist_mode_text_token=true` requires an active sampled or forced GJD mode."
                )
            resolved_text, generalist_mode_text_token_count = self._append_generalist_mode_text_token(
                visual_tower,
                resolved_text,
                sampled_generalist_mode,
            )
        packed_video_cross_attention_mask = self._build_proprio_cross_attention_mask(
            resolved_text_context=resolved_text,
            proprio_state=proprio_state,
            query_frames_per_copy=num_video_frames,
            tokens_per_frame=video_tokens_per_frame,
            chunk_size_frames=sampled_chunk_size,
            chunk_origin_frame=chunk_origin_frame,
            repeat_copies=2,
            global_suffix_token_count=generalist_mode_text_token_count,
        )

        single_action_grid = self._build_action_grid_ids_for_sequence(
            batch_size=noisy_actions.shape[0],
            seq_len=action_seq_len,
            action_tokens_per_frame=action_tokens_per_frame,
            device=noisy_actions.device,
            frame_shift=frame_shift,
        )  # [B, 4, T_a*ppF_a]
        packed_action_grid = torch.cat([single_action_grid, single_action_grid], dim=-1)
        packed_action_cross_attention_mask = self._build_proprio_cross_attention_mask(
            resolved_text_context=resolved_text,
            proprio_state=proprio_state,
            query_frames_per_copy=num_action_frames,
            tokens_per_frame=int(action_tokens_per_frame),
            chunk_size_frames=sampled_chunk_size,
            chunk_origin_frame=chunk_origin_frame,
            repeat_copies=2,
            global_suffix_token_count=generalist_mode_text_token_count,
        )

        packed_action_pre = self.action_expert.pre_dit(
            action_tokens=packed_action_tokens,
            timestep=packed_action_timesteps,
            context=resolved_text,
            cross_attention_mask=packed_action_cross_attention_mask,
            action_grid_ids=packed_action_grid,
            hidden_context=packed_action_hidden_context,
        )
        packed_attention_profile = build_mot_packed_coupling_attention_profile(
            num_video_frames=num_video_frames,
            video_tokens_per_frame=video_tokens_per_frame,
            num_action_frames=num_action_frames,
            action_tokens_per_frame=int(action_tokens_per_frame),
            chunk_size_frames=sampled_chunk_size,
            device=noisy_actions.device,
            attention_window_size=sampled_window_size,
            current_block_coupling=current_block_coupling,
            chunk_origin_frame=chunk_origin_frame,
            action_context_mask=clean_action_condition_mask,
            history_stream_visibility=self._resolve_history_stream_visibility().value,
            prefix_condition_frames=prefix_condition_frames,
        )
        packed_video_hidden_context = (
            None
            if prefix_condition_frames > 0
            else self._video_hidden_context_for_tokens(
                visual_tower,
                hidden_proprio_state,
                video_latents=video_latents,
                copies=2,
                chunk_size_frames=sampled_chunk_size,
            )
        )
        video_flow_pred, packed_action_hidden = forward_mot_packed_coupling_denoise(
            visual_tower=visual_tower,
            noisy_video_latents=video_artifacts.noisy_latents,
            clean_video_latents=video_artifacts.condition_latents,
            noisy_video_timesteps=video_artifacts.timesteps,
            clean_video_timesteps=video_artifacts.condition_timesteps,
            action_expert=self.action_expert,
            packed_action_pre=packed_action_pre,
            attention_profile=packed_attention_profile,
            text_context=resolved_text,
            frame_start=frame_shift - prefix_condition_frames,
            use_activation_checkpointing=bool(self.config.use_activation_checkpointing),
            packed_block_stack=self.packed_block_stack,
            video_cross_attention_mask=packed_video_cross_attention_mask,
            video_hidden_context=packed_video_hidden_context,
        )
        predicted_latents = denoised_video_latents_from_flow(
            noisy_latents=video_artifacts.noisy_latents,
            flow_pred=video_flow_pred,
            timesteps=video_artifacts.timesteps,
            scheduler=video_artifacts.scheduler,
        )
        packed_action_flow = self.action_expert.post_dit(packed_action_hidden, packed_action_pre)
        # Loss from the A_noisy half only (first action_seq_len tokens).
        action_flow_pred = packed_action_flow[:, :action_seq_len]
        denoised_actions = denoised_actions_from_flow(
            noisy_actions=noisy_actions,
            flow_pred=action_flow_pred,
            timesteps=noisy_slot_timesteps,
            scheduler=action_artifacts.scheduler,
        )

        # ---- Assemble training artifacts ----
        video_rollout: MoTVideoTrainArtifacts | None = None
        if self.training_config.objective_enabled("latent"):
            video_rollout = MoTVideoTrainArtifacts(
                flow_pred=video_flow_pred,
                targets=video_artifacts.targets,
                timesteps=video_artifacts.timesteps,
                scheduler=video_artifacts.scheduler,
                predicted_latents=predicted_latents,
                target_latents=video_latents,
                future_loss_mask=future_loss_mask,
            )

        batch_size = video_latents.shape[0]
        return PolicyTrainOutput(
            policy_features=video_latents.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            metrics={
                "mot_history_frames": video_latents.new_tensor(float(history_frames)),
                "mot_video_prefix_frames": video_latents.new_tensor(float(history_frames)),
            },
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "runtime_mode": str(self.config.runtime_mode),
                "current_block_coupling": current_block_coupling.value,
                "sampled_chunk_size": sampled_chunk_size,
                "sampled_window_size": sampled_window_size,
                "generalist_training_paradigm": self.config.generalist_training_paradigm.value,
                "generalist_training_source": generalist_source,
                "video_condition_source": video_condition_source,
                "mot_generalist_training_mode_override": (
                    forced_generalist_mode.value if forced_generalist_mode is not None else None
                ),
                "mot_generalist_text_dropped": bool(text_dropped),
                "mot_generalist_training_mode": (
                    sampled_generalist_mode.value if sampled_generalist_mode is not None else None
                ),
                "mot_generalist_mode_text_token": (
                    sampled_generalist_mode.value
                    if generalist_mode_text_token_count > 0 and sampled_generalist_mode is not None
                    else None
                ),
                "mot_generalist_mode_text_token_count": int(generalist_mode_text_token_count),
                "mot_train_artifacts": MoTTrainArtifacts(
                    action=MoTActionTrainArtifacts(
                        flow_pred=action_flow_pred,
                        targets=action_artifacts.targets,
                        timesteps=noisy_slot_timesteps,
                        scheduler=action_artifacts.scheduler,
                        denoised_actions=denoised_actions,
                        # Use the post-A1 mask, not the dataset-derived one
                        # baked into `action_artifacts.action_mask`. When the
                        # generalist sampler picks ACTION_CONDITIONED_VIDEO,
                        # `effective_action_mask` was zeroed by
                        # `_apply_mot_generalist_training_mode` to actually
                        # mask the action loss — but `action_artifacts` still
                        # holds the pre-A1 mask reference (the builder just
                        # stores-and-returns the input tensor at
                        # `flow_matching.py` line 417), so threading the
                        # post-A1 mask here is the only place the masking
                        # actually takes effect downstream.
                        action_mask=effective_action_mask,
                    ),
                    video=video_rollout,
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                    history_frames=int(history_frames),
                ),
            },
        )

    def _forward_train_non_joint_two_stream(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        return self._forward_train_packed_coupling(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            prepared_inputs=prepared_inputs,
        )

    def _forward_infer_packed_coupling(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
        runtime_state: MoTRuntimeState,
    ) -> PolicyInferOutput:
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        action_only_rollout = _resolve_mot_action_only_rollout(
            context,
            current_block_coupling=current_block_coupling,
        )
        if (
            action_only_rollout
            and current_block_coupling != CurrentBlockCoupling.ACTION_THEN_VIDEO
        ):
            raise ValueError(
                "M5 packed action-only rollout is only used for action_then_video; "
                "decoupled_same_step action-only rollout uses the legacy split-cache route."
            )
        inference_window_size = _resolve_mot_inference_window_size(
            context,
            default_window_size=int(self.training_config.window_size),
        )
        device = next(visual_tower.core.parameters()).device
        action_device = next(self.action_expert.parameters()).device
        if action_device != device:
            raise ValueError(
                "M5 packed coupling inference currently requires visual tower and action expert on the same device, "
                f"got visual_device={device}, action_device={action_device}."
            )
        dtype = next(self.action_expert.parameters()).dtype
        batch_size = int(visual_outputs.frontend.video_latents.shape[0])
        frame_chunk_size = max(1, int(self.inference_config.frame_chunk_size))
        if self.action_horizon % frame_chunk_size != 0:
            raise ValueError(
                "M5 packed coupling inference expects `action_horizon` to divide by `inference.frame_chunk_size`, "
                f"got action_horizon={self.action_horizon}, frame_chunk_size={frame_chunk_size}."
            )
        action_tokens_per_frame = self.action_horizon // frame_chunk_size
        video_latents = visual_outputs.frontend.video_latents.to(device=device, dtype=dtype)
        latent_height = int(video_latents.shape[-2])
        latent_width = int(video_latents.shape[-1])
        video_tokens_per_frame = int(visual_outputs.frontend.token_grid.tokens_per_frame)
        current_start_frame = int(infer_state.cursor.current_start_frame)
        startup_plan = resolve_strict_startup_plan(
            step_index=int(infer_state.step_index),
            current_start_frame=current_start_frame,
            frame_chunk_size=frame_chunk_size,
            action_tokens_per_frame=action_tokens_per_frame,
            action_horizon=self.action_horizon,
        )
        first_step_bootstrap = startup_plan.is_startup

        past_clean_latents = runtime_state.past_clean_latents
        if past_clean_latents is not None:
            past_clean_latents = past_clean_latents.to(device=device, dtype=dtype)
            if past_clean_latents.shape[0] != batch_size or past_clean_latents.shape[1] != video_latents.shape[1]:
                raise ValueError(
                    "M5 packed video history shape does not match current video latents, "
                    f"got past={tuple(past_clean_latents.shape)}, current={tuple(video_latents.shape)}."
                )
            if past_clean_latents.shape[-2:] != video_latents.shape[-2:]:
                raise ValueError(
                    "M5 packed video history spatial shape does not match current video latents, "
                    f"got past={tuple(past_clean_latents.shape)}, current={tuple(video_latents.shape)}."
                )
        past_clean_actions = runtime_state.past_clean_actions
        if past_clean_actions is not None:
            past_clean_actions = past_clean_actions.to(device=device, dtype=dtype)
            if past_clean_actions.shape[0] != batch_size or past_clean_actions.shape[-1] != self.action_dim:
                raise ValueError(
                    "M5 packed action history shape does not match current action shape, "
                    f"got past_actions={tuple(past_clean_actions.shape)}, batch_size={batch_size}, action_dim={self.action_dim}."
                )
            if past_clean_actions.shape[1] % action_tokens_per_frame != 0:
                raise ValueError(
                    "M5 packed action history length must be divisible by action_tokens_per_frame, "
                    f"got past_action_tokens={past_clean_actions.shape[1]}, action_tokens_per_frame={action_tokens_per_frame}."
                )

        current_video_prefix_frames = startup_plan.video_prefix_frames
        generation_frame_start = startup_plan.generation_frame_start
        if first_step_bootstrap:
            observed_prefix = video_latents[:, :, -1:].contiguous()
            current_generated_video = torch.randn(
                batch_size,
                video_latents.shape[1],
                frame_chunk_size,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
            current_noisy_video = torch.cat([observed_prefix.to(dtype=dtype), current_generated_video], dim=2)
            current_clean_video = torch.zeros_like(current_noisy_video)
            current_clean_video[:, :, :1] = observed_prefix.to(dtype=dtype)
        else:
            current_video_observation = video_latents
            current_video_frames = int(current_video_observation.shape[2])
            if current_video_frames >= frame_chunk_size:
                current_video_condition = current_video_observation[:, :, -frame_chunk_size:].contiguous()
            else:
                pad_frames = frame_chunk_size - current_video_frames
                current_video_condition = torch.cat(
                    [
                        current_video_observation,
                        current_video_observation[:, :, -1:].expand(-1, -1, pad_frames, -1, -1),
                    ],
                    dim=2,
                ).contiguous()
            current_noisy_video = torch.randn_like(current_video_condition, device=device, dtype=dtype)
            current_clean_video = torch.zeros_like(current_noisy_video)
        current_video_sequence_frames = int(current_noisy_video.shape[2])
        current_action_sample = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device, dtype=dtype)
        current_action_prefix_tokens = startup_plan.action_prefix_tokens
        current_action_sequence_tokens = startup_plan.current_action_sequence_tokens

        history_window_frames = resolve_mot_rollout_cache_window_frames(
            window_size=inference_window_size,
            frame_chunk_size=frame_chunk_size,
        )
        history_video_frames = 0 if past_clean_latents is None else int(past_clean_latents.shape[2])
        history_action_tokens = 0 if past_clean_actions is None else int(past_clean_actions.shape[1])
        history_action_frames = history_action_tokens // action_tokens_per_frame
        shared_history_frames = min(history_video_frames, history_action_frames)
        max_history_frames = max(0, history_window_frames - frame_chunk_size)
        if max_history_frames > 0:
            shared_history_frames = min(shared_history_frames, max_history_frames)
        else:
            shared_history_frames = 0
        if shared_history_frames > 0:
            history_video = past_clean_latents[:, :, -shared_history_frames:].contiguous()
            history_action_tokens = shared_history_frames * action_tokens_per_frame
            history_actions = past_clean_actions[:, -history_action_tokens:].contiguous()
        else:
            history_video = None
            history_actions = None
            history_action_tokens = 0
        hidden_proprio_state = runtime_state.hidden_proprio_state
        history_hidden_proprio = runtime_state.past_hidden_proprio_states
        if history_hidden_proprio is not None and shared_history_frames > 0:
            history_hidden_proprio = history_hidden_proprio.to(device=device, dtype=dtype)
            if int(history_hidden_proprio.shape[0]) != batch_size:
                raise ValueError(
                    "M5 packed hidden proprio history batch size does not match current batch, "
                    f"got history={tuple(history_hidden_proprio.shape)}, batch_size={batch_size}."
                )
            history_hidden_proprio = history_hidden_proprio[:, -shared_history_frames:].contiguous()
        else:
            history_hidden_proprio = None
        if shared_history_frames > 0 and self._uses_per_chunk_proprio_context() and history_hidden_proprio is None:
            raise ValueError("M5 per-chunk additive proprio inference is missing hidden proprio history.")
        current_hidden_proprio_frames = None
        if hidden_proprio_state is not None:
            current_hidden_proprio_frames = hidden_proprio_state.to(device=device, dtype=dtype)[:, None, :].expand(
                -1,
                current_video_sequence_frames,
                -1,
            )
        if history_hidden_proprio is None:
            video_hidden_proprio_sequence = current_hidden_proprio_frames
        elif current_hidden_proprio_frames is None:
            video_hidden_proprio_sequence = history_hidden_proprio
        else:
            video_hidden_proprio_sequence = torch.cat([history_hidden_proprio, current_hidden_proprio_frames], dim=1)

        if history_video is None:
            noisy_video_sequence = current_noisy_video
            clean_video_sequence = current_clean_video
        else:
            noisy_video_sequence = torch.cat([history_video, current_noisy_video], dim=2)
            clean_video_sequence = torch.cat([history_video, current_clean_video], dim=2)
        history_video_timesteps = torch.zeros(batch_size, shared_history_frames, device=device, dtype=torch.float32)
        current_zero_video_timesteps = torch.zeros(
            batch_size,
            current_video_sequence_frames,
            device=device,
            dtype=torch.float32,
        )
        zero_action_current_timesteps = torch.zeros(
            batch_size,
            current_action_sequence_tokens,
            device=device,
            dtype=torch.float32,
        )
        zero_current_action_condition = current_action_sample.new_zeros(
            batch_size,
            current_action_sequence_tokens,
            self.action_dim,
        )

        text_context = runtime_state.text_context
        if text_context is None:
            text_context = visual_outputs.frontend.conditioning.text_context
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=device,
                dtype=dtype,
            )
        else:
            text_context = text_context.to(device=device, dtype=dtype)
        if (
            bool(getattr(self.config, "generalist_mode_text_token", False))
            and int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) <= 0
        ):
            text_context, token_count = self._append_generalist_mode_text_token(
                visual_tower,
                text_context,
                MoTGeneralistTrainingMode.JOINT,
            )
            runtime_state.text_context = text_context
            runtime_state.generalist_mode_text_token_count = int(token_count)

        video_scheduler = build_video_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        action_scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        if len(video_scheduler.timesteps) != len(action_scheduler.timesteps):
            raise ValueError(
                "M5 packed coupling inference expects matched video/action denoise step counts, "
                f"got video_steps={len(video_scheduler.timesteps)}, action_steps={len(action_scheduler.timesteps)}."
            )
        couple_action_video_sigmas = _should_couple_mot_action_to_video_sigmas(
            self.config,
            current_block_coupling,
        )
        joint_timestep_coupling = _resolve_mot_joint_timestep_coupling(
            self.config,
            current_block_coupling,
        )
        action_timestep_lookup_scheduler = None
        if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
            action_timestep_lookup_scheduler = build_action_flow_match_inference_scheduler(
                training_config=self.training_config,
                inference_config=self.inference_config,
                num_inference_steps_override=self.training_config.action_num_train_timesteps,
            )
        sequence_frame_start = current_start_frame - shared_history_frames
        packed_chunk_origin_frame = startup_plan.chunk_origin_frame(shared_history_frames)
        packed_action_context_mask = build_strict_action_context_mask(
            batch_size=batch_size,
            history_action_tokens=history_action_tokens,
            current_action_sequence_tokens=current_action_sequence_tokens,
            invalid_current_prefix_tokens=current_action_prefix_tokens,
            device=device,
            dtype=torch.float32,
        )
        attention_profile = build_mot_packed_coupling_attention_profile(
            num_video_frames=shared_history_frames + current_video_sequence_frames,
            video_tokens_per_frame=video_tokens_per_frame,
            num_action_frames=shared_history_frames + current_video_prefix_frames + frame_chunk_size,
            action_tokens_per_frame=action_tokens_per_frame,
            chunk_size_frames=frame_chunk_size,
            device=device,
            attention_window_size=inference_window_size,
            current_block_coupling=current_block_coupling,
            chunk_origin_frame=packed_chunk_origin_frame,
            action_context_mask=packed_action_context_mask,
            build_dense_masks=True,
            build_flex_masks=False,
            history_stream_visibility=self._resolve_history_stream_visibility().value,
        )
        action_grid_ids = self._build_action_grid_ids_for_sequence(
            batch_size=batch_size,
            seq_len=current_action_sequence_tokens,
            action_tokens_per_frame=action_tokens_per_frame,
            device=device,
            frame_shift=current_start_frame,
        )
        if shared_history_frames > 0:
            history_action_grid_ids = self._build_action_grid_ids_for_sequence(
                batch_size=batch_size,
                seq_len=history_action_tokens,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
                frame_shift=int(sequence_frame_start),
            )
            action_sequence_grid_ids = torch.cat([history_action_grid_ids, action_grid_ids], dim=2)
        else:
            action_sequence_grid_ids = action_grid_ids
        packed_action_grid_ids = torch.cat([action_sequence_grid_ids, action_sequence_grid_ids], dim=2)

        predicted_video_sequence = noisy_video_sequence
        action_sample = current_action_sample
        apply_video_hidden_proprio = not _uses_mot_legacy_prefix_contract(self.config)
        zero_current_video_timestep = torch.zeros(
            batch_size,
            current_video_sequence_frames,
            device=device,
            dtype=torch.float32,
        )
        zero_current_action_timestep = torch.zeros(
            batch_size,
            current_action_sequence_tokens,
            device=device,
            dtype=torch.float32,
        )

        def _compose_clean_video_sequence(current_clean_video_for_step: torch.Tensor) -> torch.Tensor:
            if history_video is None:
                return current_clean_video_for_step
            return torch.cat([history_video, current_clean_video_for_step], dim=2)

        def _compose_current_action_sequence(action_tokens: torch.Tensor) -> torch.Tensor:
            if current_action_prefix_tokens <= 0:
                return action_tokens
            invalid_prefix = action_tokens.new_zeros(
                action_tokens.shape[0],
                current_action_prefix_tokens,
                action_tokens.shape[-1],
            )
            return torch.cat([invalid_prefix, action_tokens], dim=1)

        def _build_packed_action_pre(
            *,
            action_tokens: torch.Tensor,
            action_timestep: torch.Tensor,
            current_clean_action_for_step: torch.Tensor,
        ):
            if history_actions is None:
                noisy_action_sequence = action_tokens
                noisy_action_timesteps = action_timestep
                clean_action_sequence = current_clean_action_for_step
            else:
                noisy_action_sequence = torch.cat([history_actions, action_tokens], dim=1)
                noisy_action_timesteps = torch.cat(
                    [
                        torch.zeros(batch_size, history_action_tokens, device=device, dtype=torch.float32),
                        action_timestep,
                    ],
                    dim=1,
                )
                clean_action_sequence = torch.cat([history_actions, current_clean_action_for_step], dim=1)
            packed_action_tokens = torch.cat([noisy_action_sequence, clean_action_sequence], dim=1)
            packed_action_hidden_context = self._action_hidden_context_for_tokens(
                visual_tower,
                video_hidden_proprio_sequence,
                action_tokens=noisy_action_sequence,
                action_tokens_per_frame=int(action_tokens_per_frame),
                copies=2,
            )
            packed_action_timesteps = torch.cat(
                [
                    noisy_action_timesteps,
                    torch.zeros(
                        batch_size,
                        history_action_tokens + current_action_sequence_tokens,
                        device=device,
                        dtype=torch.float32,
                    ),
                ],
                dim=1,
            )
            return self.action_expert.pre_dit(
                action_tokens=packed_action_tokens,
                timestep=packed_action_timesteps,
                context=text_context,
                action_grid_ids=packed_action_grid_ids,
                hidden_context=packed_action_hidden_context,
            )

        def _run_packed_step(
            *,
            video_timestep: torch.Tensor,
            action_timestep: torch.Tensor,
            current_clean_video_for_step: torch.Tensor,
            current_clean_action_for_step: torch.Tensor,
        ):
            dense_video_timestep = torch.cat([history_video_timesteps, video_timestep], dim=1)
            packed_video_hidden_context = (
                self._video_hidden_context_for_tokens(
                    visual_tower,
                    video_hidden_proprio_sequence,
                    video_latents=predicted_video_sequence,
                    copies=2,
                )
                if apply_video_hidden_proprio
                else None
            )
            packed_action_pre = _build_packed_action_pre(
                action_tokens=_compose_current_action_sequence(action_sample),
                action_timestep=action_timestep,
                current_clean_action_for_step=current_clean_action_for_step,
            )
            return forward_mot_packed_coupling_denoise(
                visual_tower=visual_tower,
                noisy_video_latents=predicted_video_sequence,
                clean_video_latents=_compose_clean_video_sequence(current_clean_video_for_step),
                noisy_video_timesteps=dense_video_timestep,
                clean_video_timesteps=torch.zeros_like(dense_video_timestep),
                action_expert=self.action_expert,
                packed_action_pre=packed_action_pre,
                attention_profile=attention_profile,
                text_context=text_context,
                frame_start=int(sequence_frame_start),
                use_activation_checkpointing=False,
                packed_block_stack=self.packed_block_stack,
                prefer_flex_attention=False,
                video_hidden_context=packed_video_hidden_context,
            ) + (packed_action_pre,)

        def _video_timestep(value: torch.Tensor) -> torch.Tensor:
            timestep = _expand_scalar_timestep(
                value,
                shape=(batch_size, current_video_sequence_frames),
                device=device,
            )
            if current_video_prefix_frames > 0:
                timestep[:, :current_video_prefix_frames] = 0.0
                predicted_video_sequence[
                    :,
                    :,
                    shared_history_frames : shared_history_frames + current_video_prefix_frames,
                ] = current_clean_video[:, :, :current_video_prefix_frames]
            return timestep

        def _action_timestep(value: torch.Tensor) -> torch.Tensor:
            timestep = _expand_scalar_timestep(
                value,
                shape=(batch_size, current_action_sequence_tokens),
                device=device,
            )
            if current_action_prefix_tokens > 0:
                timestep[:, :current_action_prefix_tokens] = 0.0
            return timestep

        def _update_video(
            video_flow_pred: torch.Tensor,
            video_timestep: torch.Tensor,
            *,
            sigma: torch.Tensor | None = None,
            sigma_next: torch.Tensor | None = None,
        ) -> None:
            nonlocal predicted_video_sequence
            generated_start = shared_history_frames + current_video_prefix_frames
            current_video_flow = video_flow_pred[:, :, generated_start : generated_start + frame_chunk_size].contiguous()
            current_predicted_video = predicted_video_sequence[
                :,
                :,
                generated_start : generated_start + frame_chunk_size,
            ].contiguous()
            if sigma is None or sigma_next is None:
                current_predicted_video = video_scheduler.step(current_video_flow, video_timestep, current_predicted_video)
            else:
                current_predicted_video = _flow_step_with_sigmas(
                    current_predicted_video,
                    current_video_flow,
                    sigma=sigma,
                    sigma_next=sigma_next,
                )
            predicted_video_sequence = torch.cat(
                [predicted_video_sequence[:, :, :generated_start], current_predicted_video],
                dim=2,
            )

        def _update_action(
            packed_action_hidden: torch.Tensor,
            packed_action_pre,
            action_timestep: torch.Tensor,
            *,
            sigma: torch.Tensor | None = None,
            sigma_next: torch.Tensor | None = None,
        ) -> None:
            nonlocal action_sample
            packed_action_flow = self.action_expert.post_dit(packed_action_hidden, packed_action_pre)
            flow_start = history_action_tokens + current_action_prefix_tokens
            action_flow_pred = packed_action_flow[:, flow_start : flow_start + self.action_horizon].contiguous()
            generated_action_timestep = action_timestep[:, current_action_prefix_tokens:].contiguous()
            scheduler_timestep = generated_action_timestep.reshape(-1)[0]
            if sigma is None or sigma_next is None:
                action_sample = action_scheduler.step(action_flow_pred, scheduler_timestep, action_sample)
            else:
                action_sample = _flow_step_with_sigmas(
                    action_sample,
                    action_flow_pred,
                    sigma=sigma,
                    sigma_next=sigma_next,
                )

        if current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION:
            for video_timestep in video_scheduler.timesteps:
                current_video_timestep = _video_timestep(video_timestep)
                video_flow_pred, _, _ = _run_packed_step(
                    video_timestep=current_video_timestep,
                    action_timestep=zero_current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_video(video_flow_pred, video_timestep)
            current_clean_video = predicted_video_sequence[:, :, shared_history_frames:].contiguous()
            for action_timestep in action_scheduler.timesteps:
                current_action_timestep = _action_timestep(action_timestep)
                _, packed_action_hidden, packed_action_pre = _run_packed_step(
                    video_timestep=zero_current_video_timestep,
                    action_timestep=current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_action(packed_action_hidden, packed_action_pre, current_action_timestep)
        elif current_block_coupling == CurrentBlockCoupling.ACTION_THEN_VIDEO:
            for action_timestep in action_scheduler.timesteps:
                current_action_timestep = _action_timestep(action_timestep)
                _, packed_action_hidden, packed_action_pre = _run_packed_step(
                    video_timestep=zero_current_video_timestep,
                    action_timestep=current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_action(packed_action_hidden, packed_action_pre, current_action_timestep)
            if not action_only_rollout:
                current_clean_action = _compose_current_action_sequence(action_sample)
                for video_timestep in video_scheduler.timesteps:
                    current_video_timestep = _video_timestep(video_timestep)
                    video_flow_pred, _, _ = _run_packed_step(
                        video_timestep=current_video_timestep,
                        action_timestep=zero_current_action_timestep,
                        current_clean_video_for_step=current_clean_video,
                        current_clean_action_for_step=current_clean_action,
                    )
                    _update_video(video_flow_pred, video_timestep)
        else:
            for step_index, video_timestep in enumerate(video_scheduler.timesteps):
                action_timestep = action_scheduler.timesteps[step_index]
                shared_sigma = None
                shared_sigma_next = None
                if couple_action_video_sigmas:
                    shared_sigma = video_scheduler.sigmas[step_index].to(device=device, dtype=torch.float32)
                    shared_sigma_next = _scheduler_next_sigma(video_scheduler, step_index).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
                        if action_timestep_lookup_scheduler is None:  # pragma: no cover - defensive guard
                            raise RuntimeError(
                                "M5 match-sigma same-step inference requires an action timestep lookup scheduler."
                            )
                        action_timestep = timesteps_matching_sigmas(
                            action_timestep_lookup_scheduler,
                            shared_sigma.reshape(1),
                        )[0].to(device=device, dtype=torch.float32)
                    elif joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
                        action_timestep = video_timestep.to(device=device, dtype=torch.float32)
                current_video_timestep = _video_timestep(video_timestep)
                current_action_timestep = _action_timestep(action_timestep)
                video_flow_pred, packed_action_hidden, packed_action_pre = _run_packed_step(
                    video_timestep=current_video_timestep,
                    action_timestep=current_action_timestep,
                    current_clean_video_for_step=current_clean_video,
                    current_clean_action_for_step=zero_current_action_condition,
                )
                _update_video(
                    video_flow_pred,
                    video_timestep,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                )
                _update_action(
                    packed_action_hidden,
                    packed_action_pre,
                    current_action_timestep,
                    sigma=shared_sigma,
                    sigma_next=shared_sigma_next,
                )

        clean_video_prefix_frames = shared_history_frames + current_video_prefix_frames
        if action_only_rollout:
            predicted_chunk_latents = predicted_video_sequence.new_empty(
                batch_size,
                predicted_video_sequence.shape[1],
                0,
                latent_height,
                latent_width,
            )
            next_clean_context = clean_video_sequence[:, :, :clean_video_prefix_frames].contiguous()
            pending_predicted_video_frames = 0
        else:
            predicted_chunk_latents = predicted_video_sequence[:, :, -frame_chunk_size:].contiguous()
            next_clean_context = torch.cat(
                [clean_video_sequence[:, :, :clean_video_prefix_frames], predicted_chunk_latents],
                dim=2,
            )
            pending_predicted_video_frames = frame_chunk_size
        runtime_state.past_clean_latents = next_clean_context[:, :, -history_window_frames:].detach()
        if video_hidden_proprio_sequence is not None:
            next_hidden_context = video_hidden_proprio_sequence[
                :,
                : clean_video_prefix_frames + pending_predicted_video_frames,
            ].contiguous()
            runtime_state.past_hidden_proprio_states = next_hidden_context[:, -history_window_frames:].detach()
        else:
            runtime_state.past_hidden_proprio_states = None
        if history_actions is None:
            next_clean_actions = action_sample
        else:
            next_clean_actions = torch.cat([history_actions, action_sample], dim=1)
        max_action_history_tokens = history_window_frames * action_tokens_per_frame
        runtime_state.past_clean_actions = next_clean_actions[:, -max_action_history_tokens:].detach()
        runtime_state.next_condition_frame_start = int(generation_frame_start + frame_chunk_size)
        runtime_state.pending_predicted_video_frames = int(pending_predicted_video_frames)
        next_state = infer_state
        next_state.step_index += 1
        next_state.cursor.current_start_frame = int(generation_frame_start + frame_chunk_size)
        next_state.variant_state = runtime_state
        return PolicyInferOutput(
            policy_features=action_sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            next_state=next_state,
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "current_block_coupling": current_block_coupling.value,
                "generation_frame_start": int(generation_frame_start),
                "mot_action_only_rollout": bool(action_only_rollout),
                "predicted_latents": predicted_chunk_latents.detach(),
                "predicted_video_latents": predicted_chunk_latents.detach(),
                "mot_first_step_bootstrap": first_step_bootstrap,
                "mot_action_cond_tokens": 0,
                "mot_invalid_startup_action_tokens": int(current_action_prefix_tokens),
                "mot_action_context_invalid_tokens": int(
                    attention_profile.metadata.get("invalid_action_context_tokens", 0)
                ),
                "mot_generalist_mode_text_token": (
                    MoTGeneralistTrainingMode.JOINT.value
                    if int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) > 0
                    else None
                ),
                "mot_generalist_mode_text_token_count": int(
                    getattr(runtime_state, "generalist_mode_text_token_count", 0)
                ),
                "mot_history_anchor_frames": int(shared_history_frames),
                "mot_packed_history_debug": {
                    "past_clean_latent_frames": 0 if past_clean_latents is None else int(past_clean_latents.shape[2]),
                    "past_clean_action_frames": 0 if past_clean_actions is None else int(past_clean_actions.shape[1] // action_tokens_per_frame),
                    "shared_history_frames": int(shared_history_frames),
                    "current_observed_latent_frames": int(video_latents.shape[2]),
                    "current_clean_condition_frames": int(current_clean_video.shape[2]),
                    "packed_video_frames": int(shared_history_frames + current_video_sequence_frames),
                    "packed_action_frames": int(shared_history_frames + current_video_prefix_frames + frame_chunk_size),
                    "current_action_flow_start": int(history_action_tokens + current_action_prefix_tokens),
                    "current_action_flow_end": int(history_action_tokens + current_action_prefix_tokens + self.action_horizon),
                    "history_window_frames": int(history_window_frames),
                    "inference_window_size": int(inference_window_size),
                    "max_history_frames": int(max_history_frames),
                    "next_past_clean_latent_frames": int(runtime_state.past_clean_latents.shape[2]),
                    "next_past_clean_action_frames": int(runtime_state.past_clean_actions.shape[1] // action_tokens_per_frame),
                    "pending_predicted_video_frames": int(runtime_state.pending_predicted_video_frames),
                    "sequence_frame_start": int(sequence_frame_start),
                    "current_frame_start": int(current_start_frame),
                    "current_video_prefix_frames": int(current_video_prefix_frames),
                    "current_action_prefix_tokens": int(current_action_prefix_tokens),
                    "mode_uses_packed_cache": True,
                    "joint_timestep_coupling": joint_timestep_coupling.value,
                    "coupled_action_video_sigmas": bool(couple_action_video_sigmas),
                },
                "mot_infer_artifacts": MoTInferArtifacts(
                    action_pred=action_sample,
                    predicted_latents=predicted_chunk_latents.detach(),
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                ),
            },
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        self._maybe_initialize_action_expert(visual_tower)
        state = previous_state or PolicyInferState()
        runtime_state = state.variant_state if isinstance(state.variant_state, MoTRuntimeState) else MoTRuntimeState()
        action_device_raw = context.extra.get("action_device")
        action_device = (
            next(self.action_expert.parameters()).device
            if action_device_raw is None
            else torch.device(str(action_device_raw))
        )
        action_dtype = next(self.action_expert.parameters()).dtype
        proprio_state = self._resolve_proprio_state(
            context.state,
            label="M5 inference",
            fallback_state=runtime_state.proprio_state,
        )
        if proprio_state is not None:
            runtime_state.proprio_state = proprio_state.detach().clone()
        hidden_proprio_state = self._resolve_infer_hidden_proprio_context(
            context.state,
            fallback_state=runtime_state.hidden_proprio_state,
        )
        if hidden_proprio_state is not None:
            runtime_state.hidden_proprio_state = hidden_proprio_state.detach().clone()
        resolved_text_context = self._resolve_text_context_with_proprio(
            visual_tower,
            visual_outputs.frontend.conditioning.text_context,
            proprio_state,
            batch_size=int(visual_outputs.frontend.video_latents.shape[0]),
            device=action_device,
            dtype=action_dtype,
            materialize_if_missing=(
                self._uses_proprio_context()
                or bool(getattr(self.config, "generalist_mode_text_token", False))
            ),
        )
        generalist_mode_text_token_count = 0
        if bool(getattr(self.config, "generalist_mode_text_token", False)):
            if resolved_text_context is None:  # pragma: no cover - materialized above
                raise RuntimeError("M5 mode-token rollout expected materialized text context.")
            resolved_text_context, generalist_mode_text_token_count = self._append_generalist_mode_text_token(
                visual_tower,
                resolved_text_context,
                MoTGeneralistTrainingMode.JOINT,
            )
        runtime_state.generalist_mode_text_token_count = int(generalist_mode_text_token_count)
        # Only `joint_denoise` stays on the simultaneous video+action denoise
        # path. `non_joint_two_stream` falls through to the method-1-aligned
        # default path below (video fully denoised first, then action attends
        # clean video K/V via `forward_action_with_video_cache`).
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            runtime_device = next(visual_tower.core.parameters()).device
            if action_device != runtime_device:
                raise ValueError(
                    "MoT joint_denoise inference currently requires video and action to run on the same device, "
                    f"got runtime_device={runtime_device}, action_device={action_device}, "
                    f"runtime_mode={self.config.runtime_mode!r}."
                )
            runtime_state.text_context = resolved_text_context
            runtime_state.action_device = str(action_device)
            state.variant_state = runtime_state
            del context
            return state
        condition_latents = visual_outputs.frontend.video_latents
        current_condition_frame_start = int(state.cursor.current_start_frame)
        # Note: `runtime_state.video_cache` is populated inside
        # `forward_infer_step` after the slot-pool warmup + video denoise
        # last-step write, so we don't prefill it here.
        runtime_state.text_context = resolved_text_context
        runtime_state.video_tokens_per_frame = int(visual_outputs.frontend.token_grid.tokens_per_frame)
        runtime_state.chunk_advance_frames = max(1, int(self.inference_config.frame_chunk_size))
        # Only initialize `next_condition_frame_start` on the first chunk of a
        # session. After that, `forward_infer_step` at the end of each chunk
        # sets it to the current chunk's `generation_frame_start` so the NEXT
        # chunk's observation write lands on the same rotary positions as the
        # current chunk's pred entries (overwriting them, keeping the cache
        # contiguous). Without this guard, advancing here by
        # `condition_latents.shape[2]` double-advances alongside
        # `cursor.current_start_frame` and leaves a `chunk_frames`-wide gap
        # of empty rotary slots at every chunk boundary, which desynchronizes
        # the training-time contiguous rotary assumption from the inference
        # cache layout (Method 1 avoids this by using `advance_frame_start=
        # False` inside its denoise rollout and a separate post-rollout
        # `warmup_cache` that writes observations at the same frame_start
        # where the pred just landed).
        if runtime_state.past_clean_latents is None:
            runtime_state.next_condition_frame_start = int(
                current_condition_frame_start + int(condition_latents.shape[2])
            )
        runtime_state.action_device = str(action_device)
        state.variant_state = runtime_state
        del context
        return state

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        runtime_state = (
            infer_state.variant_state if isinstance(infer_state.variant_state, MoTRuntimeState) else MoTRuntimeState()
        )
        self._maybe_initialize_action_expert(visual_tower)
        current_block_coupling_for_infer = resolve_mot_current_block_coupling(self.config)
        mot_inference_backend = ensure_mot_policy_variant_inference_backend(
            policy_variant=self,
            visual_tower=visual_tower,
            policy_config=self.config,
            allow_module_mutation=bool(context.extra.get("allow_mot_legacy_backend_restore", True)),
        )
        use_legacy_cache_infer = (
            self.config.current_block_coupling is not None
            and mot_inference_backend["backend"] == "legacy_split_cache"
            and current_block_coupling_for_infer in MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS
        )
        if self.config.current_block_coupling is not None and not use_legacy_cache_infer:
            return self._forward_infer_packed_coupling(
                visual_tower=visual_tower,
                visual_outputs=visual_outputs,
                context=context,
                infer_state=infer_state,
                runtime_state=runtime_state,
            )
        # Only `joint_denoise` uses the simultaneous video+action denoise
        # branch below. `non_joint_two_stream` falls through to the
        # method-1-aligned default path at the bottom of this function, which
        # denoises video to completion first via
        # `visual_tower.generate_conditioned_future_latents` and then runs the
        # action expert against the resulting all-clean video K/V cache.
        if self.config.runtime_mode == MoTRuntimeMode.JOINT_DENOISE:
            current_block_coupling = resolve_mot_current_block_coupling(self.config)
            if not _is_mot_same_step_coupling(current_block_coupling):
                raise NotImplementedError(
                    "M5 joint_denoise inference supports same-step couplings only; "
                    f"got current_block_coupling={current_block_coupling.value!r}."
                )
            device = next(visual_tower.core.parameters()).device
            action_device = next(self.action_expert.parameters()).device
            if action_device != device:
                raise ValueError(
                    "MoT joint_denoise inference currently requires visual tower and action expert on the same device, "
                    f"got visual_device={device}, action_device={action_device}, "
                    f"runtime_mode={self.config.runtime_mode!r}."
                )
            dtype = next(self.action_expert.parameters()).dtype
            batch_size = visual_outputs.frontend.video_latents.shape[0]
            observed_prefix_frames = int(self.config.video_prefix_frames)
            video_latents = visual_outputs.frontend.video_latents.to(device=device, dtype=dtype)
            if video_latents.shape[2] <= observed_prefix_frames:
                raise ValueError(
                    "MoT two-stream inference requires at least one future frame after the observed prefix, "
                    f"got video_latents.shape={tuple(video_latents.shape)}, video_prefix_frames={observed_prefix_frames}, "
                    f"runtime_mode={self.config.runtime_mode!r}."
                )
            if self.inference_config.video_num_inference_steps != self.inference_config.action_num_inference_steps:
                raise ValueError(
                    "MoT two-stream inference currently requires matching video/action inference step counts, "
                    f"got video_num_inference_steps={self.inference_config.video_num_inference_steps}, "
                    f"action_num_inference_steps={self.inference_config.action_num_inference_steps}, "
                    f"runtime_mode={self.config.runtime_mode!r}."
                )
            frame_chunk_size = max(1, int(self.inference_config.frame_chunk_size))
            if self.action_horizon % frame_chunk_size != 0:
                raise ValueError(
                    "MoT joint_denoise inference expects `action_horizon` to divide by `inference.frame_chunk_size`, "
                    f"got action_horizon={self.action_horizon}, frame_chunk_size={frame_chunk_size}."
                )
            observed_prefix = video_latents[:, :, :observed_prefix_frames]
            future_template = video_latents[:, :, observed_prefix_frames:]
            noisy_video_latents = torch.cat(
                [
                    observed_prefix,
                    torch.randn_like(future_template, device=device, dtype=dtype),
                ],
                dim=2,
            )
            action_scheduler = build_action_flow_match_inference_scheduler(
                training_config=self.training_config,
                inference_config=self.inference_config,
            )
            video_scheduler = build_video_flow_match_inference_scheduler(
                training_config=self.training_config,
                inference_config=self.inference_config,
            )
            couple_action_video_sigmas = _should_couple_mot_action_to_video_sigmas(
                self.config,
                current_block_coupling,
            )
            joint_timestep_coupling = _resolve_mot_joint_timestep_coupling(
                self.config,
                current_block_coupling,
            )
            action_timestep_lookup_scheduler = None
            if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
                action_timestep_lookup_scheduler = build_action_flow_match_inference_scheduler(
                    training_config=self.training_config,
                    inference_config=self.inference_config,
                    num_inference_steps_override=self.training_config.action_num_train_timesteps,
                )
            sample = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
            text_context = runtime_state.text_context
            if text_context is None:
                text_context = torch.zeros(
                    batch_size,
                    visual_tower.config.max_text_tokens,
                    visual_tower.config.text_dim,
                    device=device,
                    dtype=dtype,
                )
            else:
                text_context = text_context.to(device=device, dtype=dtype)
            hidden_proprio_state = runtime_state.hidden_proprio_state
            hidden_proprio_sequence = None
            if hidden_proprio_state is not None:
                hidden_proprio_sequence = hidden_proprio_state.to(device=device, dtype=dtype)[:, None, :].expand(
                    -1,
                    int(video_latents.shape[2]),
                    -1,
                )
            action_tokens_per_frame = self.action_horizon // frame_chunk_size
            attention_mask = build_mot_attention_mask(
                video_seq_len=visual_outputs.frontend.token_grid.tokens_per_frame * video_latents.shape[2],
                action_seq_len=self.action_horizon,
                device=device,
                condition_mode=self.config.condition_mode,
                video_tokens_per_frame=visual_outputs.frontend.token_grid.tokens_per_frame,
                video_can_attend_action=self.config.video_can_attend_action,
                action_tokens_per_frame=action_tokens_per_frame,
                action_chunk_size_frames=frame_chunk_size,
                clean_video_frames=observed_prefix_frames,
                current_block_coupling=current_block_coupling,
            )
            for step_index, video_timestep in enumerate(video_scheduler.timesteps):
                action_timestep = action_scheduler.timesteps[step_index]
                shared_sigma = None
                shared_sigma_next = None
                if couple_action_video_sigmas:
                    shared_sigma = video_scheduler.sigmas[step_index].to(device=device, dtype=torch.float32)
                    shared_sigma_next = _scheduler_next_sigma(video_scheduler, step_index).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    if joint_timestep_coupling == JointTimestepCoupling.MATCH_SIGMA:
                        if action_timestep_lookup_scheduler is None:  # pragma: no cover - defensive guard
                            raise RuntimeError("M5 match-sigma joint denoise requires an action timestep lookup scheduler.")
                        action_timestep = timesteps_matching_sigmas(
                            action_timestep_lookup_scheduler,
                            shared_sigma.reshape(1),
                        )[0].to(device=device, dtype=torch.float32)
                    elif joint_timestep_coupling == JointTimestepCoupling.SHARED_VIDEO_SCHEDULE:
                        action_timestep = video_timestep.to(device=device, dtype=torch.float32)
                dense_video_timestep = _expand_scalar_timestep(
                    video_timestep,
                    shape=(batch_size, video_latents.shape[2]),
                    device=device,
                )
                dense_video_timestep[:, :observed_prefix_frames] = 0.0
                dense_action_timestep = _expand_scalar_timestep(
                    action_timestep,
                    shape=(batch_size, self.action_horizon),
                    device=device,
                )
                action_pre = self.action_expert.pre_dit(
                    action_tokens=sample,
                    timestep=dense_action_timestep,
                    context=text_context,
                    hidden_context=self._action_hidden_context_for_tokens(
                        visual_tower,
                        hidden_proprio_sequence,
                        action_tokens=sample,
                        action_tokens_per_frame=action_tokens_per_frame,
                    ),
                )
                video_flow_pred, action_hidden_states = forward_joint_video_action_denoise(
                    visual_tower=visual_tower,
                    noisy_video_latents=noisy_video_latents,
                    video_timesteps=dense_video_timestep,
                    action_expert=self.action_expert,
                    action_pre=action_pre,
                    text_context=text_context,
                    attention_mask=attention_mask,
                    frame_start=int(infer_state.cursor.current_start_frame),
                    video_hidden_context=self._video_hidden_context_for_tokens(
                        visual_tower,
                        hidden_proprio_sequence,
                        video_latents=noisy_video_latents,
                    ),
                )
                flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
                if shared_sigma is None or shared_sigma_next is None:
                    noisy_video_latents = video_scheduler.step(video_flow_pred, video_timestep, noisy_video_latents)
                else:
                    noisy_video_latents = _flow_step_with_sigmas(
                        noisy_video_latents,
                        video_flow_pred,
                        sigma=shared_sigma,
                        sigma_next=shared_sigma_next,
                    )
                noisy_video_latents[:, :, :observed_prefix_frames] = observed_prefix
                if shared_sigma is None or shared_sigma_next is None:
                    sample = action_scheduler.step(flow_pred, action_timestep, sample)
                else:
                    sample = _flow_step_with_sigmas(
                        sample,
                        flow_pred,
                        sigma=shared_sigma,
                        sigma_next=shared_sigma_next,
                    )
            predicted_latents = noisy_video_latents[:, :, observed_prefix_frames:].detach()
            next_state = infer_state
            next_state.step_index += 1
            next_state.variant_state = runtime_state
            return PolicyInferOutput(
                policy_features=sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
                next_state=next_state,
                aux={
                    "variant": self.config.name,
                    "method_family": "mot",
                    "condition_mode": str(self.config.condition_mode),
                    "current_block_coupling": current_block_coupling.value,
                    "predicted_latents": predicted_latents,
                    "predicted_video_latents": predicted_latents,
                    "mot_infer_artifacts": MoTInferArtifacts(
                        action_pred=sample,
                        predicted_latents=predicted_latents,
                        condition_mode=str(self.config.condition_mode),
                        runtime_mode=str(self.config.runtime_mode),
                    ),
                    "joint_timestep_coupling": joint_timestep_coupling.value,
                    "coupled_action_video_sigmas": bool(couple_action_video_sigmas),
                    "mot_generalist_mode_text_token": (
                        MoTGeneralistTrainingMode.JOINT.value
                        if int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) > 0
                        else None
                    ),
                    "mot_generalist_mode_text_token_count": int(
                        getattr(runtime_state, "generalist_mode_text_token_count", 0)
                    ),
                },
            )
        # True Method-1-aligned NON_JOINT_TWO_STREAM rollout with persistent
        # KV cache on the shared video core. Each chunk:
        #   1) First chunk only -- bootstrap the shared transformer's
        #      `_exact_runtime_caches[cache_name]` with observed env latents
        #      at frame_start=0, all frames clean (timestep=0), update_cache=2.
        #      Matches Method 1's `_write_exact_cache_chunk` bootstrap.
        #   2) Every chunk -- run video denoise with ONLY the
        #      `frame_chunk_size` noisy current-chunk latents as Q. Past
        #      context comes from cache. `update_cache=0` during denoise
        #      steps, `update_cache=1` on the last step writes the new
        #      chunk's clean K/V back into cache, matching Method 1 exactly.
        #   3) Extract a MoTVideoCache view of the updated cache (taking the
        #      cond half when CFG is doubled) so the action expert can
        #      cross-attend the full rollout history.
        #   4) Run action denoise against that MoTVideoCache.
        from open_wam.models.policy_variants.parallel_stream.reference_runtime import (
            FlowMatchScheduler as _VideoFlowMatchScheduler,
            _clear_exact_prediction_cache as _clear_pred_cache,
            data_seq_to_patch as _data_seq_to_patch,
            initialize_reference_cache as _initialize_reference_cache,
            prepare_reference_single_stream_input as _prepare_single_stream_input,
            reference_runtime_dtype as _reference_runtime_dtype,
            run_reference_single_stream_forward as _run_single_stream_forward,
        )

        video_latents = visual_outputs.frontend.video_latents
        batch_size = int(video_latents.shape[0])
        device = next(self.action_expert.parameters()).device
        dtype = next(self.action_expert.parameters()).dtype
        video_device = next(visual_tower.core.parameters()).device
        video_dtype = _reference_runtime_dtype(visual_tower.core)

        chunk_frames = max(1, int(self.inference_config.frame_chunk_size))
        if self.action_horizon % chunk_frames != 0:
            raise ValueError(
                "MoT non-joint inference expects `action_horizon` to divide by `inference.frame_chunk_size`, "
                f"got action_horizon={self.action_horizon}, frame_chunk_size={chunk_frames}."
            )
        action_tokens_per_frame = self.action_horizon // chunk_frames
        current_block_coupling = resolve_mot_current_block_coupling(self.config)
        action_only_rollout = _resolve_mot_action_only_rollout(
            context,
            current_block_coupling=current_block_coupling,
        )
        if current_block_coupling not in MOT_LEGACY_SPLIT_CACHE_INFERENCE_COUPLINGS:
            raise NotImplementedError(
                "M5 legacy split-cache inference only supports staged video_then_action and decoupled_same_step; "
                f"got current_block_coupling={current_block_coupling.value!r}."
            )
        video_commit_before_action = current_block_coupling == CurrentBlockCoupling.VIDEO_THEN_ACTION

        text_context_for_video = runtime_state.text_context
        if text_context_for_video is None:
            text_context_for_video = visual_outputs.frontend.conditioning.text_context
        if text_context_for_video is None:
            text_context_for_video = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=video_device,
                dtype=video_dtype,
            )
        else:
            text_context_for_video = text_context_for_video.to(
                device=video_device, dtype=video_dtype
            )
        if (
            bool(getattr(self.config, "generalist_mode_text_token", False))
            and int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) <= 0
        ):
            text_context_for_video, token_count = self._append_generalist_mode_text_token(
                visual_tower,
                text_context_for_video,
                MoTGeneralistTrainingMode.JOINT,
            )
            runtime_state.text_context = text_context_for_video
            runtime_state.generalist_mode_text_token_count = int(token_count)
        # Full Method-1 alignment: cache at 2B with CFG throughout the
        # video path. Bootstrap uses `force_cfg_batch=True` so every
        # subsequent denoise step (with `guidance_scale>1` and
        # `negative_text_emb`) can do CFG batching consistently. The
        # action expert runs at batch=B, so when we extract the
        # MoTVideoCache for action we slice the cond half `[:B]`.
        negative_text_context = visual_outputs.frontend.conditioning.negative_text_context
        negative_text_context = self._resolve_text_context_with_proprio(
            visual_tower,
            negative_text_context,
            runtime_state.proprio_state,
            batch_size=batch_size,
            device=video_device,
            dtype=video_dtype,
            materialize_if_missing=False,
        )
        if bool(getattr(self.config, "generalist_mode_text_token", False)) and negative_text_context is not None:
            negative_text_context, _ = self._append_generalist_mode_text_token(
                visual_tower,
                negative_text_context,
                MoTGeneralistTrainingMode.JOINT,
            )
        use_cfg = (
            negative_text_context is not None
            and bool(self.inference_config.use_cache)
        )

        cache_name = "mot_non_joint_two_stream_cache"
        latent_channels = int(visual_tower.config.latent_channels)
        latent_height = int(video_latents.shape[-2])
        latent_width = int(video_latents.shape[-1])
        is_first_chunk = runtime_state.past_clean_latents is None
        skip_observation_update = bool(context.extra.get("mot_skip_observation_update", False))
        if skip_observation_update and is_first_chunk:
            raise ValueError("MoT open-loop extension requires an initialized non-joint rollout cache.")
        condition_frame_start_override_raw = context.extra.get("mot_condition_frame_start")
        if skip_observation_update and condition_frame_start_override_raw is not None:
            raise ValueError("MoT condition-frame rewind is only valid for observation-conditioned replans.")
        inference_window_size = _resolve_mot_inference_window_size(
            context,
            default_window_size=_MOT_SLOT_POOL_ATTN_WINDOW,
        )
        # Method-1-aligned per-chunk warmup. On chunk 0 we allocate the
        # slot-pool backend via `initialize_reference_cache` and write the
        # bootstrap obs latents at frame_start=0. On subsequent chunks the
        # driver passes a fresh window of real env observations (encoded
        # into `video_latents`). We:
        #   1) clear the prediction cache (last chunk's denoise last-step
        #      pred K/V),
        #   2) write the real-env observation as a NEW stable chunk at
        #      `frame_start = runtime_state.next_condition_frame_start`.
        # `observed_prefix.shape[2]` is allowed to vary between chunks --
        # chunk 0 may use a 1-frame bootstrap (matching Method 1) while
        # subsequent chunks pass `chunk_frames` real env-observation latents
        # that overwrite the previous chunk's pred slots. The Route-A
        # inference mask removes the old chunk_frames-aligned bootstrap
        # constraint by treating past KV as always-visible.
        current_obs_frame_start = int(runtime_state.next_condition_frame_start)
        if is_first_chunk:
            _initialize_reference_cache(
                visual_tower.core,
                cache_name=cache_name,
                attn_window=inference_window_size,
                batch_size=batch_size,
                frame_chunk_size=chunk_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                device=video_device,
                action_per_frame=action_tokens_per_frame,
                use_cfg=use_cfg,
            )
            current_obs_frame_start = 0
        elif condition_frame_start_override_raw is not None:
            current_obs_frame_start = int(condition_frame_start_override_raw)
        if self.inference_config.use_cache and not skip_observation_update:
            _clear_pred_cache(visual_tower.core, cache_name=cache_name)
        observed_prefix = video_latents.to(device=video_device, dtype=video_dtype)
        if not skip_observation_update:
            boot_video_input = _prepare_single_stream_input(
                latents=observed_prefix,
                timestep=0.0,
                text_emb=text_context_for_video,
                frame_st_id=current_obs_frame_start,
                backbone_config=visual_tower.config,
                action_mode=False,
            )
            _run_single_stream_forward(
                visual_tower.core,
                input_dict=boot_video_input,
                update_cache=2,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=1.0,
                negative_text_emb=negative_text_context,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )
            runtime_state.past_clean_latents = observed_prefix.detach()
        # Observation-conditioned replans write real observations into the
        # next slots. Async open-loop extensions intentionally skip this
        # write so planning ahead does not leak too-early real frames into a
        # future chunk; they extend from the already generated cache instead.
        generation_frame_start = (
            current_obs_frame_start + chunk_frames
            if skip_observation_update
            else current_obs_frame_start + int(observed_prefix.shape[2])
        )
        if is_first_chunk:
            runtime_state.chunk_origin_frame = int(generation_frame_start) % int(chunk_frames)
        runtime_state.next_condition_frame_start = (
            generation_frame_start + chunk_frames if skip_observation_update else generation_frame_start
        )

        if action_only_rollout:
            predicted_latents = observed_prefix.new_empty(
                batch_size,
                latent_channels,
                0,
                latent_height,
                latent_width,
            )
        else:
            # Cache-aware video denoise on the current noisy chunk only.
            latents = torch.randn(
                batch_size,
                latent_channels,
                chunk_frames,
                latent_height,
                latent_width,
                device=video_device,
                dtype=video_dtype,
            )
            video_scheduler = _VideoFlowMatchScheduler(
                shift=self.training_config.video_sigma_shift,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=self.training_config.video_num_train_timesteps,
            )
            video_scheduler.set_timesteps(self.inference_config.video_num_inference_steps)
            video_timesteps = F.pad(
                video_scheduler.timesteps.to(device=video_device),
                (0, 1),
                mode="constant",
                value=0,
            )
            for index, timestep in enumerate(video_timesteps):
                last_step = index == len(video_timesteps) - 1
                video_input = _prepare_single_stream_input(
                    latents=latents,
                    timestep=timestep,
                    text_emb=text_context_for_video,
                    frame_st_id=generation_frame_start,
                    backbone_config=visual_tower.config,
                    action_mode=False,
                )
                video_noise_pred = _run_single_stream_forward(
                    visual_tower.core,
                    input_dict=video_input,
                    update_cache=1 if (last_step and video_commit_before_action and self.inference_config.use_cache) else 0,
                    cache_name=cache_name,
                    action_mode=False,
                    guidance_scale=self.inference_config.guidance_scale,
                    negative_text_emb=negative_text_context,
                    force_cfg_batch=use_cfg,
                )
                if not last_step:
                    video_noise_pred = _data_seq_to_patch(
                        visual_tower.core.patch_size,
                        video_noise_pred,
                        chunk_frames,
                        latent_height,
                        latent_width,
                        batch_size=batch_size,
                    ).to(dtype=video_dtype)
                    latents = video_scheduler.step(video_noise_pred, timestep, latents)
            predicted_latents = latents

        # Don't advance `next_condition_frame_start` past the observation
        # write position. Method 1 with `advance_frame_start=False` keeps
        # frame_start at the value warmup set it to, so that the NEXT
        # chunk's warmup writes its real observations at the same rotary
        # positions that the current chunk's pred entries just landed on
        # (teacher-forcing the pred positions with real obs). The pred
        # entries (chunk_frames tokens at rotary [gen_start..gen_start+4))
        # will be cleared + overwritten by the next chunk's stable obs
        # write via `_clear_pred_cache` + `_run_single_stream_forward(
        # update_cache=2)`. The TOTAL number of clean video frames the
        # action expert sees this chunk is observation frames +
        # current-chunk pred frames.
        total_clean_video_frames = generation_frame_start + (0 if action_only_rollout else chunk_frames)
        action_visible_video_end_frame = (
            total_clean_video_frames
            if video_commit_before_action
            else generation_frame_start
        )

        def extract_mot_video_cache_from_exact_cache() -> MoTVideoCache:
            # With CFG active the cache is doubled `[cond, uncond]` on the
            # batch dim; slice the cond half for the action expert (batch=B).
            cache_state = visual_tower.core._resolve_exact_cache_state(cache_name)
            if cache_state is None:
                raise RuntimeError(
                    f"MoT non_joint_two_stream expected cache state at `{cache_name}` "
                    "but the shared transformer returned None."
                )
            extracted_layers: list[MoTVideoLayerCache] = []
            for entry in cache_state.self_attention_kv:
                if entry.key is None or entry.value is None:
                    raise RuntimeError(
                        "MoT non_joint_two_stream cache extraction found an empty layer entry."
                    )
                key = entry.key
                value = entry.value
                if key.shape[0] == 2 * batch_size:
                    key = key[:batch_size]
                    value = value[:batch_size]
                elif key.shape[0] != batch_size:
                    raise RuntimeError(
                        "MoT non_joint_two_stream cache batch dimension must match the current batch "
                        f"(or 2x for CFG), got cache_batch={key.shape[0]}, batch_size={batch_size}."
                    )
                extracted_layers.append(
                    MoTVideoLayerCache(key=key.detach(), value=value.detach())
                )
            return MoTVideoCache(
                layers=tuple(extracted_layers),
                video_seq_len=int(extracted_layers[0].key.shape[2]),
            )

        action_video_cache = extract_mot_video_cache_from_exact_cache()
        def prepare_action_video_cache(cache: MoTVideoCache) -> MoTVideoCache:
            # Method-1 alignment: Method 1's slot pool stores both video and
            # action so video occupies `(attn_window // 2) * latent_token_per_chunk`
            # tokens, which is integer-frame-aligned. Method 5 only writes video so the
            # slot pool fills with `(attn_window // 2) * (latent + action)` tokens
            # (= 67.5 frames here), leaving a partial leading frame after eviction.
            # Trim to Method 1's per-stream cap so the action expert sees the
            # same frame-aligned video lookback Method 1 does.
            method1_video_lookback_frames = (
                (inference_window_size // 2) * int(chunk_frames)
            )
            max_video_tokens_for_action = int(method1_video_lookback_frames) * int(
                runtime_state.video_tokens_per_frame
            ) if runtime_state.video_tokens_per_frame else None
            if (
                max_video_tokens_for_action is not None
                and max_video_tokens_for_action > 0
                and cache.video_seq_len > max_video_tokens_for_action
            ):
                cache = trim_mot_video_cache_tail(
                    cache,
                    max_video_seq_len=max_video_tokens_for_action,
                )
            return move_mot_video_cache(cache, device=device, dtype=dtype)

        action_video_cache = prepare_action_video_cache(action_video_cache)
        runtime_state.video_cache = action_video_cache
        cached_batch_size = int(action_video_cache.layers[0].key.shape[0])
        if cached_batch_size != batch_size:
            raise ValueError(
                "MoT cached-action inference requires the current observation batch to match the cached video batch, "
                f"got current_batch_size={batch_size}, cached_batch_size={cached_batch_size}."
            )
        if self.action_horizon % chunk_frames != 0:
            raise ValueError(
                "MoT non-joint inference expects `action_horizon` to divide by `inference.frame_chunk_size`, "
                f"got action_horizon={self.action_horizon}, frame_chunk_size={chunk_frames}."
            )
        action_tokens_per_frame = self.action_horizon // chunk_frames
        scheduler = build_action_flow_match_inference_scheduler(
            training_config=self.training_config,
            inference_config=self.inference_config,
        )
        sample = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        text_context = runtime_state.text_context
        if text_context is None:
            text_context = torch.zeros(
                batch_size,
                visual_tower.config.max_text_tokens,
                visual_tower.config.text_dim,
                device=device,
                dtype=dtype,
            )
        # Method-1-aligned action denoise with persistent action K/V
        # cache. Past action chunks' clean K/V live in
        # `runtime_state.action_cache`; fresh `action_horizon` tokens are
        # the only Q this forward recomputes every step. On the final
        # (padded timestep=0) step we capture the fresh per-layer K/V
        # and append to `runtime_state.action_cache`, mirroring Method 1's
        # `update_cache=1` at the last action step.
        action_cache_rewind_frame_start_raw = context.extra.get("mot_action_cache_rewind_frame_start")
        if action_cache_rewind_frame_start_raw is None:
            action_cache_rewind_frame_start_raw = context.extra.get("mot_action_cache_prefix_frames")
        if action_cache_rewind_frame_start_raw is not None:
            _rewind_runtime_action_cache_to_frame(
                runtime_state,
                absolute_frame_start=int(action_cache_rewind_frame_start_raw),
                action_tokens_per_frame=action_tokens_per_frame,
            )
        past_action_cache = runtime_state.action_cache
        # Diagnostic: setting OPEN_WAM_MOT_DISABLE_PAST_ACTION_CACHE=1 forces
        # the action expert to see only video + current noisy action per
        # chunk (no past action history). Useful for isolating whether
        # autoregressive drift in the past_action K/V chain is the source
        # of chunk-to-chunk instability.
        if os.environ.get("OPEN_WAM_MOT_DISABLE_PAST_ACTION_CACHE", "0") == "1":
            past_action_cache = None
        past_action_seq_len = int(past_action_cache.action_seq_len) if past_action_cache is not None else 0
        if past_action_seq_len % action_tokens_per_frame != 0:
            raise ValueError(
                "MoT non_joint_two_stream action cache length must be a multiple of action_tokens_per_frame, "
                f"got past_action_seq_len={past_action_seq_len}, action_tokens_per_frame={action_tokens_per_frame}."
            )
        past_action_frames = past_action_seq_len // action_tokens_per_frame
        total_action_seq_len = past_action_seq_len + self.action_horizon
        # Method-1 byte-aligned mask: replicates
        # `build_chunked_temporal_exact_attention_profile` for the inference
        # `[video_cache; past_action_cache; current_action]` layout. Block
        # ids are video=chunk*2 / action=chunk*2+1, the within-window check
        # uses `training_config.window_size` (same value Method 1 passes as
        # `input_dict["window_size"]` at inference), and clean/noise causal
        # rules match Method 1's chunked_temporal_exact profile.
        if runtime_state.video_tokens_per_frame is None or runtime_state.video_tokens_per_frame <= 0:
            raise RuntimeError(
                "MoT inference mask requires `runtime_state.video_tokens_per_frame` to be set, "
                f"got {runtime_state.video_tokens_per_frame!r}."
            )
        video_lookback_frames_for_mask = int(action_video_cache.video_seq_len) // int(
            runtime_state.video_tokens_per_frame
        )
        current_action_frame_start = int(generation_frame_start)
        video_frame_start = int(action_visible_video_end_frame - video_lookback_frames_for_mask)
        past_action_frame_start = (
            int(runtime_state.action_cache_start_frame)
            if past_action_cache is not None
            else int(current_action_frame_start)
        )
        if past_action_cache is not None:
            cached_action_end_frame = int(past_action_frame_start + past_action_frames)
            if cached_action_end_frame != current_action_frame_start:
                raise RuntimeError(
                    "MoT action cache frame span is not contiguous with the current chunk, "
                    f"cache_span=[{past_action_frame_start}, {cached_action_end_frame}), "
                    f"current_action_frame_start={current_action_frame_start}."
                )
        attention_mask = build_mot_inference_action_attention_mask(
            video_seq_len=action_video_cache.video_seq_len,
            past_action_seq_len=past_action_seq_len,
            current_action_seq_len=self.action_horizon,
            video_tokens_per_frame=int(runtime_state.video_tokens_per_frame),
            action_tokens_per_frame=action_tokens_per_frame,
            chunk_size_frames=max(1, int(self.training_config.chunk_size)),
            window_size_frames=max(1, int(self.training_config.window_size)),
            device=device,
            video_can_attend_action=False,
            video_frame_start=video_frame_start,
            past_action_frame_start=past_action_frame_start,
            current_action_frame_start=current_action_frame_start,
            chunk_origin_frame=int(runtime_state.chunk_origin_frame),
            current_block_coupling=current_block_coupling,
        )
        # Method-1-aligned cache write: run the denoise loop without
        # capturing K/V, then issue a SEPARATE fresh forward at timestep=0
        # with the final denoised sample to capture cache-bound K/V. Mirrors
        # `_write_exact_cache_chunk(update_cache=1)` in
        # `run_parallel_action_conditioned_inference_rollout`, which calls a
        # fresh single-stream forward after all denoise steps complete
        # rather than reusing the loop's last-step K/V.
        fresh_action_kv: MoTActionCache | None = None
        for timestep in scheduler.timesteps.to(device=device):
            dense_timestep = torch.full(
                (batch_size, self.action_horizon),
                float(timestep),
                device=device,
                dtype=torch.float32,
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=sample,
                timestep=dense_timestep,
                context=text_context.to(device=device, dtype=dtype),
                action_grid_ids=self._build_action_grid_ids_for_sequence(
                    batch_size=batch_size,
                    seq_len=self.action_horizon,
                    action_tokens_per_frame=action_tokens_per_frame,
                    device=device,
                    frame_shift=int(current_action_frame_start),
                ),
                hidden_context=self._action_hidden_context_for_tokens(
                    visual_tower,
                    runtime_state.hidden_proprio_state,
                    action_tokens=sample,
                    action_tokens_per_frame=action_tokens_per_frame,
                    chunk_size_frames=chunk_frames,
                ),
            )
            action_hidden_states, _ = forward_action_with_video_and_action_cache(
                action_expert=self.action_expert,
                action_pre=action_pre,
                video_cache=action_video_cache,
                action_cache=past_action_cache,
                attention_mask=attention_mask,
            )
            flow_pred = self.action_expert.post_dit(action_hidden_states, action_pre)
            sample = scheduler.step(flow_pred, timestep, sample)
        # Separate cache-write forward at timestep=0 with the final denoised
        # sample. This is the Method-1 parity step.
        cache_write_timestep = torch.zeros(
            (batch_size, self.action_horizon),
            device=device,
            dtype=torch.float32,
        )
        cache_write_action_pre = self.action_expert.pre_dit(
            action_tokens=sample,
            timestep=cache_write_timestep,
            context=text_context.to(device=device, dtype=dtype),
            action_grid_ids=self._build_action_grid_ids_for_sequence(
                batch_size=batch_size,
                seq_len=self.action_horizon,
                action_tokens_per_frame=action_tokens_per_frame,
                device=device,
                frame_shift=int(current_action_frame_start),
            ),
            hidden_context=self._action_hidden_context_for_tokens(
                visual_tower,
                runtime_state.hidden_proprio_state,
                action_tokens=sample,
                action_tokens_per_frame=action_tokens_per_frame,
                chunk_size_frames=chunk_frames,
            ),
        )
        _, fresh_action_kv = forward_action_with_video_and_action_cache(
            action_expert=self.action_expert,
            action_pre=cache_write_action_pre,
            video_cache=action_video_cache,
            action_cache=past_action_cache,
            attention_mask=attention_mask,
        )
        if fresh_action_kv is None:
            raise RuntimeError(
                "MoT non_joint_two_stream cache-write forward did not produce fresh K/V."
            )
        # Append fresh action K/V to the persistent action cache for next chunk.
        fresh_action_kv_moved = move_mot_action_cache(
            fresh_action_kv, device=device, dtype=dtype
        )
        if past_action_cache is None:
            runtime_state.action_cache = fresh_action_kv_moved
            runtime_state.action_cache_start_frame = int(current_action_frame_start)
        else:
            runtime_state.action_cache = append_mot_action_cache(
                past_action_cache, fresh_action_kv_moved
            )
        if (
            current_block_coupling == CurrentBlockCoupling.DECOUPLED_SAME_STEP
            and self.inference_config.use_cache
            and not action_only_rollout
        ):
            deferred_video_input = _prepare_single_stream_input(
                latents=predicted_latents.to(device=video_device, dtype=video_dtype),
                timestep=0.0,
                text_emb=text_context_for_video,
                frame_st_id=generation_frame_start,
                backbone_config=visual_tower.config,
                action_mode=False,
            )
            _run_single_stream_forward(
                visual_tower.core,
                input_dict=deferred_video_input,
                update_cache=1,
                cache_name=cache_name,
                action_mode=False,
                guidance_scale=1.0,
                negative_text_emb=negative_text_context,
                combine_cfg=False,
                force_cfg_batch=use_cfg,
            )
            action_video_cache = prepare_action_video_cache(
                extract_mot_video_cache_from_exact_cache()
            )
            runtime_state.video_cache = action_video_cache
        # Method-1 alignment: video and action share the same effective
        # lookback. Method 1 stores both streams in the slot pool and
        # `attn_window` evicts them together; to mirror that with Method 5's
        # split caches we trim the action cache to exactly the video cache's
        # current frame count. Asymmetric lookback (action shorter or longer
        # than video) is OOD: training always saw matched lengths, so the
        # action expert hallucinates when past action covers a different
        # frame span than past video.
        video_tokens_per_frame_for_trim = runtime_state.video_tokens_per_frame
        if video_tokens_per_frame_for_trim is not None and video_tokens_per_frame_for_trim > 0:
            video_lookback_frames = int(action_video_cache.video_seq_len) // int(
                video_tokens_per_frame_for_trim
            )
            max_action_seq_len = max(
                action_tokens_per_frame,
                video_lookback_frames * action_tokens_per_frame,
            )
            action_cache_before_trim = runtime_state.action_cache
            if action_cache_before_trim.action_seq_len > max_action_seq_len:
                dropped_action_tokens = int(action_cache_before_trim.action_seq_len - max_action_seq_len)
                runtime_state.action_cache_start_frame += int(dropped_action_tokens // action_tokens_per_frame)
            runtime_state.action_cache = trim_mot_action_cache_tail(
                action_cache_before_trim,
                max_action_seq_len=max_action_seq_len,
            )
        next_state = infer_state
        next_state.step_index += 1
        next_state.cursor.current_start_frame = int(
            infer_state.cursor.current_start_frame + max(1, runtime_state.chunk_advance_frames)
        )
        next_state.variant_state = runtime_state
        return PolicyInferOutput(
            policy_features=sample.new_zeros(batch_size, 0, self.action_expert.hidden_size),
            next_state=next_state,
            aux={
                "variant": self.config.name,
                "method_family": "mot",
                "condition_mode": str(self.config.condition_mode),
                "current_block_coupling": current_block_coupling.value,
                "generation_frame_start": int(current_action_frame_start),
                "mot_action_only_rollout": bool(action_only_rollout),
                "mot_generalist_mode_text_token": (
                    MoTGeneralistTrainingMode.JOINT.value
                    if int(getattr(runtime_state, "generalist_mode_text_token_count", 0)) > 0
                    else None
                ),
                "mot_generalist_mode_text_token_count": int(
                    getattr(runtime_state, "generalist_mode_text_token_count", 0)
                ),
                "mot_cache_debug": {
                    "video_cache_seq_len": int(runtime_state.video_cache.video_seq_len) if runtime_state.video_cache is not None else 0,
                    "action_video_cache_seq_len": int(action_video_cache.video_seq_len),
                    "action_cache_seq_len": int(runtime_state.action_cache.action_seq_len) if runtime_state.action_cache is not None else 0,
                    "action_cache_start_frame": int(runtime_state.action_cache_start_frame),
                    "action_cache_frames_before_chunk": int(past_action_frames),
                    "total_clean_video_frames": int(total_clean_video_frames),
                    "is_first_chunk": bool(is_first_chunk),
                    "use_cfg": bool(use_cfg),
                    "current_start_frame": int(next_state.cursor.current_start_frame),
                    "next_condition_frame_start": int(runtime_state.next_condition_frame_start),
                    "chunk_advance_frames": int(runtime_state.chunk_advance_frames),
                    "video_frame_start": int(video_frame_start),
                    "past_action_frame_start": int(past_action_frame_start),
                    "current_action_frame_start": int(current_action_frame_start),
                    "chunk_origin_frame": int(runtime_state.chunk_origin_frame),
                    "skip_observation_update": bool(skip_observation_update),
                    "condition_frame_start_override": (
                        None
                        if condition_frame_start_override_raw is None
                        else int(condition_frame_start_override_raw)
                    ),
                    "current_block_coupling": current_block_coupling.value,
                    "mot_action_only_rollout": bool(action_only_rollout),
                    "video_commit_before_action": bool(video_commit_before_action),
                    "action_visible_video_end_frame": int(action_visible_video_end_frame),
                    "inference_window_size": int(inference_window_size),
                    "action_cache_rewind_frame_start": (
                        None
                        if action_cache_rewind_frame_start_raw is None
                        else int(action_cache_rewind_frame_start_raw)
                    ),
                },
                **(
                    {"predicted_latents": predicted_latents.detach(), "predicted_video_latents": predicted_latents.detach()}
                    if isinstance(predicted_latents, torch.Tensor)
                    else {}
                ),
                "mot_infer_artifacts": MoTInferArtifacts(
                    action_pred=sample,
                    predicted_latents=predicted_latents.detach() if isinstance(predicted_latents, torch.Tensor) else None,
                    condition_mode=str(self.config.condition_mode),
                    runtime_mode=str(self.config.runtime_mode),
                ),
            },
        )
