from __future__ import annotations

from dataclasses import dataclass

import torch

from open_wam.configs import (
    ContextConditionLatentSource,
    DualExpertConditionMode,
    GeneralistDenoisingMode,
    ProprioContextMode,
    VideoActionSequenceContract,
)
from open_wam.configs.policy_dual_expert import DualExpertPolicyConfig
from open_wam.models.common.chunked_attention import (
    build_chunked_text_context_cross_attention_mask,
)
from open_wam.models.common.packed_token_layout import frame_chunk_ids_for_origin
from open_wam.models.visual_tower import VisualTower

from ..contracts import PolicyTrainBatch


def resolve_dual_expert_condition_latents(
    *,
    video_latents: torch.Tensor,
    condition_mode: DualExpertConditionMode | str,
    video_prefix_frames: int,
    teacher_forcing_video_noise_prob: float,
    training: bool,
    scheduler=None,
) -> torch.Tensor:
    """Select the video branch used to condition the DualExpert action expert."""

    resolved_mode = DualExpertConditionMode(condition_mode)
    if resolved_mode == DualExpertConditionMode.FIRST_FRAME:
        return video_latents[:, :, :1]
    if resolved_mode == DualExpertConditionMode.FULL_VIDEO:
        return video_latents
    if resolved_mode == DualExpertConditionMode.TEACHER_FORCING_COND_VIDEO:
        cond_latents = video_latents[:, :, : max(1, video_prefix_frames)].clone()
        if (
            training
            and scheduler is not None
            and teacher_forcing_video_noise_prob > 0.0
            and torch.rand(1, device=video_latents.device).item() < teacher_forcing_video_noise_prob
        ):
            batch_size = cond_latents.shape[0]
            timestep_ids = torch.randint(
                low=0,
                high=len(scheduler.timesteps),
                size=(batch_size, cond_latents.shape[2]),
                device=video_latents.device,
            )
            timesteps = scheduler.timesteps.to(device=video_latents.device)[timestep_ids]
            noise = torch.randn_like(cond_latents)
            cond_latents = scheduler.add_noise(cond_latents, noise, timesteps, t_dim=2)
        return cond_latents
    raise ValueError(f"Unsupported DualExpert condition mode {resolved_mode!r}.")


@dataclass(frozen=True, slots=True)
class DualExpertConditioning:
    """Prepare DualExpert video, text, proprio, and mode conditioning tensors."""

    config: DualExpertPolicyConfig

    def uses_proprio_context(self) -> bool:
        return ProprioContextMode(self.config.proprio_context_mode) != ProprioContextMode.NONE

    def uses_text_proprio_context(self) -> bool:
        # Deprecated compatibility path; new proprio runs use per-chunk additive context.
        return ProprioContextMode(self.config.proprio_context_mode) == ProprioContextMode.TEXT_CONTEXT_TOKEN

    def uses_per_chunk_proprio_context(self) -> bool:
        return ProprioContextMode(self.config.proprio_context_mode) == ProprioContextMode.PER_CHUNK_ADDITIVE

    def uses_legacy_prefix_contract(self) -> bool:
        return (
            VideoActionSequenceContract(self.config.sequence_contract)
            == VideoActionSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
        )

    @staticmethod
    def select_anchor_state(state: torch.Tensor | None) -> torch.Tensor | None:
        if state is None:
            return None
        if state.ndim == 2:
            return state
        if state.ndim == 3:
            return state[:, -1, :]
        raise ValueError(
            "dual-expert proprio context expects state with shape [B, state_dim] or [B, H, state_dim], "
            f"got {tuple(state.shape)}."
        )

    def resolve_proprio_state(
        self,
        state: torch.Tensor | None,
        *,
        label: str,
        fallback_state: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self.uses_text_proprio_context():
            return None
        selected = self.select_anchor_state(state)
        if selected is None:
            selected = self.select_anchor_state(fallback_state)
        if selected is None:
            raise ValueError(f"Proprio context mode is enabled but no state was provided for {label}.")
        return selected

    def resolve_train_proprio_context(self, batch: PolicyTrainBatch) -> torch.Tensor | None:
        if not self.uses_text_proprio_context():
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
        return self.resolve_proprio_state(
            batch.state,
            label="dual-expert training",
        )

    def resolve_train_hidden_proprio_context(self, batch: PolicyTrainBatch) -> torch.Tensor | None:
        if not self.uses_per_chunk_proprio_context():
            return None
        value = batch.extra.get("proprio_context_frames")
        mask = batch.extra.get("proprio_context_frames_mask")
        if not isinstance(value, torch.Tensor):
            fallback_value = batch.extra.get("proprio_context_state")
            if self.uses_legacy_prefix_contract() and isinstance(fallback_value, torch.Tensor):
                raise ValueError(
                    "dual-expert legacy-prefix per-chunk additive proprio requires frame-level "
                    "`proprio_context_frames`; chunk-level `proprio_context_state` cannot be "
                    "safely aligned to prefix and causal chunk-boundary states."
                )
            value = fallback_value
            mask = batch.extra.get("proprio_context_state_mask")
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "proprio_context_mode=per_chunk_additive requires dual-expert per-frame "
                "or per-chunk proprio context."
            )
        if value.ndim != 3:
            raise ValueError(
                "dual-expert per-chunk additive proprio expects state with shape [B, frames, state_dim], "
                f"got {tuple(value.shape)}."
            )
        if isinstance(mask, torch.Tensor):
            if tuple(mask.shape) != tuple(value.shape):
                raise ValueError(
                    "dual-expert per-chunk additive proprio mask must match state shape, "
                    f"got mask={tuple(mask.shape)}, state={tuple(value.shape)}."
                )
            value = value * mask.to(device=value.device, dtype=value.dtype)
        return value

    def resolve_infer_hidden_proprio_context(
        self,
        state: torch.Tensor | None,
        *,
        fallback_state: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self.uses_per_chunk_proprio_context():
            return None
        selected = self.select_anchor_state(state)
        if selected is None:
            selected = self.select_anchor_state(fallback_state)
        if selected is None:
            raise ValueError("proprio_context_mode=per_chunk_additive requires dual-expert inference state.")
        return selected

    def resolve_text_context(
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
        if not self.uses_text_proprio_context():
            return text_context
        append = getattr(visual_tower.core, "append_proprio_context_tokens", None)
        if not callable(append):
            raise ValueError(
                "Deprecated text-space proprio token mode requires the visual tower core "
                "to support proprio appending."
            )
        return append(text_context, proprio_state)

    @staticmethod
    def encode_hidden_proprio_context(
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
                        "dual-expert chunk-level hidden proprio context is too short for requested frames, "
                        f"got state={tuple(proprio_state.shape)}, chunk_size_frames={chunk_size_frames}, "
                        f"num_frames={num_frames}."
                    )
                frame_state = expanded[:, : int(num_frames), :]
            else:
                raise ValueError(
                    "dual-expert hidden proprio context frame count must match requested frames, be singleton, "
                    "or be chunk-level with `chunk_size_frames`, "
                    f"got state={tuple(proprio_state.shape)}, num_frames={num_frames}."
                )
        else:
            raise ValueError(
                "dual-expert hidden proprio context expects shape [B, state_dim] or [B, frames, state_dim], "
                f"got {tuple(proprio_state.shape)}."
            )
        return encode(frame_state, device=device, dtype=dtype)

    def video_hidden_context_for_tokens(
        self,
        visual_tower: VisualTower,
        proprio_state: torch.Tensor | None,
        *,
        video_latents: torch.Tensor,
        copies: int = 1,
        chunk_size_frames: int | None = None,
    ) -> torch.Tensor | None:
        frame_context = self.encode_hidden_proprio_context(
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

    def action_hidden_context_for_tokens(
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
                "dual-expert action hidden proprio context requires action length divisible by action_tokens_per_frame, "
                f"got action_shape={tuple(action_tokens.shape)}, action_tokens_per_frame={action_tokens_per_frame}."
            )
        num_frames = int(action_tokens.shape[1]) // int(action_tokens_per_frame)
        frame_context = self.encode_hidden_proprio_context(
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
    def proprio_context_token_count(proprio_state: torch.Tensor | None) -> int:
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

    def build_proprio_cross_attention_mask(
        self,
        *,
        resolved_text_context: torch.Tensor,
        proprio_state: torch.Tensor | None,
        query_frames_per_copy: int,
        tokens_per_frame: int,
        chunk_size_frames: int,
        chunk_origin_frame: int = 0,
        singleton_chunk_frame: int | None = None,
        repeat_copies: int = 1,
        global_suffix_token_count: int = 0,
    ) -> torch.Tensor | None:
        proprio_token_count = self.proprio_context_token_count(proprio_state)
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
        query_chunk_ids = frame_chunk_ids_for_origin(
            frame_ids,
            chunk_origin_frame=int(chunk_origin_frame),
            chunk_size=chunk_size,
            singleton_chunk_frame=singleton_chunk_frame,
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

    def append_generalist_mode_text_token(
        self,
        visual_tower: VisualTower,
        text_context: torch.Tensor,
        mode: GeneralistDenoisingMode,
    ) -> tuple[torch.Tensor, int]:
        if not bool(getattr(self.config, "generalist_mode_text_token", False)):
            return text_context, 0
        append = getattr(visual_tower.core, "append_generalist_mode_context_token", None)
        if not callable(append):
            raise ValueError("DualExpert `generalist_mode_text_token=true` requires a visual core mode-token hook.")
        before_tokens = int(text_context.shape[1])
        resolved = append(text_context, mode.value)
        token_count = int(resolved.shape[1]) - before_tokens
        if token_count != 1:
            raise ValueError(
                "DualExpert generalist mode token appending must add exactly one token, "
                f"got token_count={token_count}."
            )
        return resolved, token_count

    def resolve_train_condition_latents(
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
                    "dual-expert training was configured with `require_condition_latents=true`, "
                    "but the latent batch did not provide `condition_latents`."
                )
            return None
        if not isinstance(condition_latents, torch.Tensor):
            raise ValueError(
                "dual-expert `condition_latents` must be a tensor when provided, "
                f"got {type(condition_latents).__name__}."
            )
        if condition_latents.ndim == 5 and tuple(condition_latents.shape) != tuple(video_latents.shape):
            time_first_shape = (
                video_latents.shape[0],
                video_latents.shape[2],
                video_latents.shape[1],
                video_latents.shape[3],
                video_latents.shape[4],
            )
            if tuple(condition_latents.shape) == tuple(time_first_shape):
                condition_latents = condition_latents.permute(0, 2, 1, 3, 4).contiguous()
        if condition_latents.ndim != 5 or tuple(condition_latents.shape) != tuple(video_latents.shape):
            raise ValueError(
                "dual-expert `condition_latents` must match video_latents exactly for train-time video conditioning, "
                f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
            )
        return condition_latents.to(device=video_latents.device, dtype=video_latents.dtype)

    @staticmethod
    def video_condition_source(condition_latents: torch.Tensor | None) -> str:
        return "condition_latents" if condition_latents is not None else "video_latents"

    def context_condition_latent_source(self) -> ContextConditionLatentSource:
        return ContextConditionLatentSource(self.config.context_condition_latent_source)

    def train_clean_video_condition_latents(
        self,
        *,
        video_latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        history_frames: int,
    ) -> tuple[torch.Tensor | None, str]:
        if (
            self.context_condition_latent_source()
            != ContextConditionLatentSource.SINGLE_FRAME_CONDITION_LATENT
        ):
            return condition_latents, self.video_condition_source(condition_latents)
        if condition_latents is None:
            raise ValueError(
                "dual-expert `context_condition_latent_source=single_frame_condition_latent` requires `condition_latents`."
            )
        if history_frames <= 0:
            raise ValueError(
                "dual-expert single-frame condition latents require at least one history/context frame, "
                f"got history_frames={history_frames}."
            )
        clean_condition = video_latents.clone()
        clean_condition[:, :, : int(history_frames)] = condition_latents[:, :, : int(history_frames)].to(
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        return clean_condition, "context_condition_latents"

    def prepend_legacy_prefix_video_latents(
        self,
        *,
        video_latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        hidden_proprio_state: torch.Tensor | None,
        batch: PolicyTrainBatch,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, str]:
        if not self.uses_legacy_prefix_contract():
            return video_latents, hidden_proprio_state, 0, self.video_condition_source(condition_latents)
        if condition_latents is None:
            raise ValueError(
                "`sequence_contract=legacy_prefix_single_frame_perchunk_proprio` requires "
                "precomputed single-frame condition_latents for dual-expert. "
                "Run scripts/augment_lerobot_latents_with_single_frame_condition.py with --source-frame-offset -1."
            )
        if condition_latents.ndim != 5 or int(condition_latents.shape[2]) < 1:
            raise ValueError(
                "dual-expert legacy-prefix condition_latents must have shape [B, C, T>=1, H, W], "
                f"got {tuple(condition_latents.shape)}."
            )
        prefix_latents = condition_latents[:, :, :1].to(
            device=video_latents.device,
            dtype=video_latents.dtype,
        )
        model_video_latents = torch.cat([prefix_latents, video_latents], dim=2)
        if hidden_proprio_state is not None:
            if hidden_proprio_state.ndim != 3:
                raise ValueError(
                    "dual-expert legacy-prefix per-chunk proprio expects target frame states with shape "
                    "[B, target_frames, state_dim], "
                    f"got {tuple(hidden_proprio_state.shape)}."
                )
            prefix_state = self.select_anchor_state(batch.state)
            if prefix_state is None:
                raise ValueError(
                    "`sequence_contract=legacy_prefix_single_frame_perchunk_proprio` requires "
                    "batch.state for the prefix/current proprio frame."
                )
            target_frames = int(video_latents.shape[2])
            if int(hidden_proprio_state.shape[1]) < target_frames:
                raise ValueError(
                    "dual-expert legacy-prefix per-chunk proprio expects at least one state per target frame, "
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
    def legacy_prefix_action_hidden_proprio_state(
        hidden_proprio_state: torch.Tensor | None,
        *,
        prefix_condition_frames: int,
        target_num_frames: int,
        chunk_size_frames: int,
        chunk_origin_frame: int = 0,
    ) -> torch.Tensor | None:
        if hidden_proprio_state is None or int(prefix_condition_frames) <= 0:
            return hidden_proprio_state
        if hidden_proprio_state.ndim != 3:
            raise ValueError(
                "dual-expert legacy-prefix per-chunk proprio expects frame state shape "
                "[B, prefix_plus_target_frames, state_dim], "
                f"got {tuple(hidden_proprio_state.shape)}."
            )
        required_frames = int(prefix_condition_frames) + int(target_num_frames)
        if int(hidden_proprio_state.shape[1]) < required_frames:
            raise ValueError(
                "dual-expert legacy-prefix per-chunk proprio expects prefix plus target frame states, "
                f"got {tuple(hidden_proprio_state.shape)} for required_frames={required_frames}."
            )
        chunk_size = max(1, int(chunk_size_frames))
        target_frame_ids = torch.arange(
            int(target_num_frames),
            device=hidden_proprio_state.device,
            dtype=torch.long,
        )
        chunk_origin = int(chunk_origin_frame)
        target_boundary_ids = (
            torch.div(target_frame_ids - chunk_origin, chunk_size, rounding_mode="floor") * chunk_size
            + chunk_origin
        )
        target_boundary_ids = target_boundary_ids.clamp(0, required_frames - 1)
        return hidden_proprio_state.index_select(dim=1, index=target_boundary_ids)
