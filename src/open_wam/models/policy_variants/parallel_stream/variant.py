from __future__ import annotations

import torch

from open_wam.configs import (
    ActionSpace,
    CurrentBlockCoupling,
    InferenceConfig,
    JointDenoiseTrainingMode,
    ParallelExactCacheWriteMode,
    ParallelRuntimeMode,
    ParallelSequenceContract,
    ParallelStreamPolicyConfig,
    ParallelStreamVariantProfile,
    ProprioContextMode,
    TemporalPositionMode,
    TrainingConfig,
)
from open_wam.data.sample_metadata import SampleConstructionMetadata
from open_wam.models.video_backbone.contracts import CacheState
from open_wam.models.video_backbone.config import SharedVideoTransformerConfig
from open_wam.models.policy_variants.common.layouts import expand_previous_action
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..base import PolicyVariant
from ..contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    RolloutCursor,
)
from .reference_runtime import (
    prepare_parallel_current_frame_action_chunk_train_artifacts,
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    prepare_parallel_fastwam_first_frame_train_artifacts,
    prepare_parallel_prefix_condition_exact_train_artifacts,
    resolve_parallel_current_block_coupling,
    run_parallel_action_conditioned_inference_rollout,
    run_parallel_action_conditioned_train,
    run_parallel_current_frame_action_chunk_inference_rollout,
    run_parallel_exact_cache_warmup,
    run_parallel_exact_inference_rollout,
    run_parallel_exact_train,
    run_parallel_fastwam_first_frame_inference_rollout,
    run_parallel_fastwam_first_frame_train,
)
from .action_adapter import LingbotActionAdapter, build_action_adapter_spec

_PER_CHUNK_PROPRIO_GRANULARITY_CHUNK = "chunk"
_PER_CHUNK_PROPRIO_GRANULARITY_FRAME = "frame"


class ParallelStreamPolicyVariant(PolicyVariant):
    """LingBot-style parallel-stream policy variant.

    The canonical method-1 path is exact-runtime-only. Training and inference
    semantics live in `reference_runtime.py` and execute on the shared runtime
    backbone; this variant intentionally avoids maintaining a second local
    packed-sequence implementation.
    """

    def __init__(
        self,
        config: ParallelStreamPolicyConfig,
        backbone_config: SharedVideoTransformerConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_dim: int,
        action_horizon: int,
        num_frames: int,
    ) -> None:
        super().__init__()
        if config.runtime_mode not in {
            ParallelRuntimeMode.LINGBOT_EXACT,
            ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
            ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
            ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        }:
            raise ValueError(
                "Parallel-stream method 1 now only supports LingBot-exact semantics. "
                f"Got runtime_mode={config.runtime_mode!r}."
            )
        self.config = config
        self.backbone_config = backbone_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_frames = num_frames
        self.exact_action_adapter = LingbotActionAdapter(
            build_action_adapter_spec(config, model_action_dim=action_dim)
        )
        self.reference_profile = self.exact_action_adapter.spec.reference_profile if self.exact_action_adapter.spec is not None else None
        self._validate_reference_profile()

    def _uses_proprio_context(self) -> bool:
        return ProprioContextMode(self.config.proprio_context_mode) != ProprioContextMode.NONE

    def _uses_text_proprio_context(self) -> bool:
        # Deprecated compatibility path; new proprio runs use per-chunk additive context.
        return ProprioContextMode(self.config.proprio_context_mode) == ProprioContextMode.TEXT_CONTEXT_TOKEN

    def _uses_per_chunk_proprio_context(self) -> bool:
        return ProprioContextMode(self.config.proprio_context_mode) == ProprioContextMode.PER_CHUNK_ADDITIVE

    def _uses_generalist_mode_text_token(self) -> bool:
        return bool(self.config.generalist_mode_text_token)

    def _require_proprio_state(self, state: torch.Tensor | None, *, label: str) -> torch.Tensor | None:
        if not self._uses_text_proprio_context():
            return None
        selected = self._select_proprio_state(state)
        if selected is None:
            raise ValueError(f"Proprio context mode is enabled but no state was provided for {label}.")
        return selected

    def _require_train_proprio_context(self, batch: PolicyTrainBatch) -> torch.Tensor | None:
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
        return self._require_proprio_state(batch.state, label="parallel-stream training")

    def _require_per_chunk_proprio_state(
        self,
        batch: PolicyTrainBatch,
        *,
        label: str,
    ) -> tuple[torch.Tensor, str] | None:
        if not self._uses_per_chunk_proprio_context():
            return None
        prefer_chunk_state = self.config.runtime_mode in {
            ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
            ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        }
        if prefer_chunk_state:
            value = batch.extra.get("proprio_context_state")
            mask = batch.extra.get("proprio_context_state_mask")
            granularity = _PER_CHUNK_PROPRIO_GRANULARITY_CHUNK
            if not isinstance(value, torch.Tensor):
                value = batch.extra.get("proprio_context_frames")
                mask = batch.extra.get("proprio_context_frames_mask")
                granularity = _PER_CHUNK_PROPRIO_GRANULARITY_FRAME
        else:
            value = batch.extra.get("proprio_context_frames")
            mask = batch.extra.get("proprio_context_frames_mask")
            granularity = _PER_CHUNK_PROPRIO_GRANULARITY_FRAME
            if not isinstance(value, torch.Tensor):
                value = batch.extra.get("proprio_context_state")
                mask = batch.extra.get("proprio_context_state_mask")
                granularity = _PER_CHUNK_PROPRIO_GRANULARITY_CHUNK
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"proprio_context_mode=per_chunk_additive requires proprio additive context for {label}.")
        if value.ndim != 3:
            raise ValueError(
                "Per-chunk proprio context expects state with shape [B, frames, state_dim], "
                f"got {tuple(value.shape)}."
            )
        if isinstance(mask, torch.Tensor):
            if tuple(mask.shape) != tuple(value.shape):
                raise ValueError(
                    "Per-chunk proprio context mask must match state shape, "
                    f"got mask={tuple(mask.shape)}, state={tuple(value.shape)}."
                )
            value = value * mask.to(device=value.device, dtype=value.dtype)
        return value, granularity

    def _resolve_proprio_state(
        self,
        state: torch.Tensor | None,
        *,
        label: str,
        infer_cache: dict | None = None,
    ) -> torch.Tensor | None:
        if not self._uses_text_proprio_context():
            return None
        selected = self._select_anchor_state(state)
        if selected is None and isinstance(infer_cache, dict):
            cached_state = infer_cache.get("last_proprio_state")
            if isinstance(cached_state, torch.Tensor):
                selected = self._select_anchor_state(cached_state)
        if selected is None:
            raise ValueError(f"Proprio context mode is enabled but no state was provided for {label}.")
        return selected

    def _resolve_per_chunk_proprio_state(
        self,
        state: torch.Tensor | None,
        *,
        label: str,
        infer_cache: dict | None = None,
    ) -> torch.Tensor | None:
        if not self._uses_per_chunk_proprio_context():
            return None
        selected = self._select_anchor_state(state)
        if selected is None and isinstance(infer_cache, dict):
            cached_state = infer_cache.get("last_proprio_state")
            if isinstance(cached_state, torch.Tensor):
                selected = self._select_anchor_state(cached_state)
        if selected is None:
            raise ValueError(f"Per-chunk proprio mode is enabled but no state was provided for {label}.")
        return selected

    def _cache_proprio_state(self, cache: dict, state: torch.Tensor | None) -> None:
        if self._uses_proprio_context() and state is not None:
            cache["last_proprio_state"] = state.detach().clone()

    def attach_visual_tower(self, visual_tower: VisualTower) -> None:
        if self._uses_generalist_mode_text_token():
            configure_mode = getattr(visual_tower.core, "configure_generalist_mode_context_encoder", None)
            if not callable(configure_mode):
                raise ValueError("Generalist mode text-token ablation requires a shared transformer core.")
            configure_mode(enabled=True)
        if self._uses_proprio_context():
            configure = (
                getattr(visual_tower.core, "configure_proprio_context_encoder", None)
                if self._uses_text_proprio_context()
                else getattr(visual_tower.core, "configure_proprio_hidden_context_encoder", None)
            )
            if not callable(configure):
                raise ValueError("Proprio context mode requires a shared transformer core.")
            state_dim = int(visual_tower.state_dim or 0)
            if state_dim <= 0:
                raise ValueError("Proprio context mode requires positive data.action_schema.state_dim.")
            configure(enabled=True, state_dim=state_dim)

    def attach_site(self) -> str:
        return self.config.attach_site

    def _runtime_mode_label(self) -> str:
        return str(self.config.runtime_mode)

    def exact_cache_write_mode(self) -> ParallelExactCacheWriteMode:
        """Cache write contract selected by the exact runtime program."""

        if resolve_parallel_current_block_coupling(self.config) in {
            CurrentBlockCoupling.JOINT,
            CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
            CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
        }:
            return ParallelExactCacheWriteMode.JOINT_PACKED
        return ParallelExactCacheWriteMode.SINGLE_STREAM_STAGED

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend",)

    def _validate_action_layout(self, action_horizon: int, *, num_frames: int) -> None:
        expected_horizon = num_frames * self.config.action_per_frame
        if action_horizon != expected_horizon:
            raise ValueError(
                "Parallel-stream variant requires `action_horizon == num_frames * action_per_frame`, "
                f"got action_horizon={action_horizon}, num_frames={num_frames}, "
                f"action_per_frame={self.config.action_per_frame}"
            )

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        observed_num_frames = int(visual_outputs.frontend.video_latents.shape[2])
        self._validate_action_layout(batch.actions.shape[1], num_frames=observed_num_frames)
        model_actions, model_action_mask = self._prepare_exact_train_actions(
            batch,
            device=visual_outputs.frontend.video_latents.device,
            dtype=visual_outputs.frontend.video_latents.dtype,
        )
        sampled_geometry = self._resolve_train_sampling_metadata(batch, observed_num_frames=observed_num_frames)
        generalist_metadata = self._resolve_generalist_training_metadata(batch)
        proprio_state = self._require_train_proprio_context(batch)
        per_chunk_proprio_payload = self._require_per_chunk_proprio_state(
            batch,
            label="parallel-stream training",
        )
        condition_latents = self._resolve_train_condition_latents(
            batch,
            video_latents=visual_outputs.frontend.video_latents,
        )
        legacy_prefix_contract = (
            self.config.parallel_sequence_contract
            == ParallelSequenceContract.LEGACY_PREFIX_SINGLE_FRAME_PERCHUNK_PROPRIO
        )
        if legacy_prefix_contract and self.config.runtime_mode not in {
            ParallelRuntimeMode.LINGBOT_EXACT,
            ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED,
        }:
            raise ValueError(
                "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` only supports "
                "LingBot exact dual-stream M1 runtime modes."
            )
        if self.config.runtime_mode == ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK:
            train_artifacts = prepare_parallel_current_frame_action_chunk_train_artifacts(
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                video_latents=visual_outputs.frontend.video_latents,
                actions=model_actions,
                action_mask=model_action_mask,
                text_emb=visual_outputs.frontend.conditioning.text_context,
                condition_latents=condition_latents,
                frame_shift=0,
            )
        elif self.config.runtime_mode == ParallelRuntimeMode.FASTWAM_FIRST_FRAME:
            train_artifacts = prepare_parallel_fastwam_first_frame_train_artifacts(
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                video_latents=visual_outputs.frontend.video_latents,
                actions=model_actions,
                action_mask=model_action_mask,
                text_emb=visual_outputs.frontend.conditioning.text_context,
                condition_latents=condition_latents,
                frame_shift=0,
            )
        elif legacy_prefix_contract:
            if not isinstance(condition_latents, torch.Tensor):
                raise ValueError(
                    "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` requires "
                    "precomputed single-frame condition_latents. "
                    "Run scripts/augment_lerobot_latents_with_single_frame_condition.py with --source-frame-offset -1."
                )
            train_artifacts = prepare_parallel_prefix_condition_exact_train_artifacts(
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                video_latents=visual_outputs.frontend.video_latents,
                actions=model_actions,
                action_mask=model_action_mask,
                text_emb=visual_outputs.frontend.conditioning.text_context,
                condition_latents=condition_latents,
                chunk_size_override=sampled_geometry["chunk_size"],
                window_size_override=sampled_geometry["window_size"],
                frame_shift=sampled_geometry["frame_shift"],
                generalist_training_mode_override=generalist_metadata["mode_override"],
                generalist_drop_text_conditioning=generalist_metadata["drop_text"],
                generalist_training_source=generalist_metadata["source"],
            )
        elif self.config.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
            train_artifacts = prepare_parallel_action_conditioned_train_artifacts(
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                video_latents=visual_outputs.frontend.video_latents,
                actions=model_actions,
                action_mask=model_action_mask,
                text_emb=visual_outputs.frontend.conditioning.text_context,
                condition_latents=condition_latents,
                chunk_size_override=sampled_geometry["chunk_size"],
                window_size_override=sampled_geometry["window_size"],
                loss_frame_start=sampled_geometry["loss_frame_start"],
                loss_frame_end=sampled_geometry["loss_frame_end"],
                latent_loss_frame_start=sampled_geometry["latent_loss_frame_start"],
                latent_loss_frame_end=sampled_geometry["latent_loss_frame_end"],
                action_loss_frame_start=sampled_geometry["action_loss_frame_start"],
                action_loss_frame_end=sampled_geometry["action_loss_frame_end"],
                frame_shift=sampled_geometry["frame_shift"],
                chunk_origin_frame=sampled_geometry["chunk_origin_frame"],
                generalist_training_mode_override=generalist_metadata["mode_override"],
                generalist_drop_text_conditioning=generalist_metadata["drop_text"],
                generalist_training_source=generalist_metadata["source"],
            )
        else:
            train_artifacts = prepare_parallel_exact_train_artifacts(
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                video_latents=visual_outputs.frontend.video_latents,
                actions=model_actions,
                action_mask=model_action_mask,
                text_emb=visual_outputs.frontend.conditioning.text_context,
                condition_latents=condition_latents,
                chunk_size_override=sampled_geometry["chunk_size"],
                window_size_override=sampled_geometry["window_size"],
                loss_frame_start=sampled_geometry["loss_frame_start"],
                loss_frame_end=sampled_geometry["loss_frame_end"],
                latent_loss_frame_start=sampled_geometry["latent_loss_frame_start"],
                latent_loss_frame_end=sampled_geometry["latent_loss_frame_end"],
                action_loss_frame_start=sampled_geometry["action_loss_frame_start"],
                action_loss_frame_end=sampled_geometry["action_loss_frame_end"],
                frame_shift=sampled_geometry["frame_shift"],
                chunk_origin_frame=sampled_geometry["chunk_origin_frame"],
            )
        if proprio_state is not None:
            train_artifacts.input_dict["proprio_state"] = proprio_state
        if per_chunk_proprio_payload is not None:
            per_chunk_proprio_state, per_chunk_proprio_granularity = per_chunk_proprio_payload
            if train_artifacts.input_dict.get("prefix_condition_frames"):
                prefix_state = self._select_anchor_state(batch.state)
                if prefix_state is None:
                    raise ValueError(
                        "`parallel_sequence_contract=legacy_prefix_single_frame_perchunk_proprio` prefix "
                        "conditioning requires batch.state for the condition frame."
                    )
                if per_chunk_proprio_granularity == _PER_CHUNK_PROPRIO_GRANULARITY_CHUNK:
                    per_chunk_proprio_state = torch.cat(
                        [
                            prefix_state[:, None, :].to(
                                device=per_chunk_proprio_state.device,
                                dtype=per_chunk_proprio_state.dtype,
                            ),
                            per_chunk_proprio_state,
                        ],
                        dim=1,
                    )
                else:
                    frame_count = int(visual_outputs.frontend.video_latents.shape[2])
                    if int(per_chunk_proprio_state.shape[1]) < frame_count:
                        raise ValueError(
                            "Prefix per-chunk proprio frame context expects at least one state per target frame, "
                            f"got {tuple(per_chunk_proprio_state.shape)} for target_frames={frame_count}."
                        )
                    per_chunk_proprio_state = torch.cat(
                        [
                            prefix_state[:, None, :].to(
                                device=per_chunk_proprio_state.device,
                                dtype=per_chunk_proprio_state.dtype,
                            ),
                            per_chunk_proprio_state[:, :frame_count, :],
                        ],
                        dim=1,
                    )
                    per_chunk_proprio_granularity = _PER_CHUNK_PROPRIO_GRANULARITY_FRAME
            train_artifacts.input_dict["per_chunk_proprio_state"] = per_chunk_proprio_state.to(
                device=visual_outputs.frontend.video_latents.device,
                dtype=visual_outputs.frontend.video_latents.dtype,
            )
            train_artifacts.input_dict["per_chunk_proprio_state_granularity"] = per_chunk_proprio_granularity
        return PolicyPreparedInputs(batch=batch, variant_inputs={"lingbot_train_artifacts": train_artifacts})

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
                    "Parallel-stream training was configured with `require_condition_latents=true`, "
                    "but the latent batch did not provide `condition_latents`."
                )
            return None
        if not isinstance(condition_latents, torch.Tensor):
            raise ValueError(
                "Parallel-stream `condition_latents` must be a tensor when provided, "
                f"got {type(condition_latents).__name__}."
            )
        if condition_latents.ndim != 5:
            raise ValueError(
                "Parallel-stream `condition_latents` must have shape `[B, C, T, H, W]`, "
                f"got {tuple(condition_latents.shape)}."
            )
        if tuple(condition_latents.shape[:2]) != tuple(video_latents.shape[:2]) or tuple(
            condition_latents.shape[-2:]
        ) != tuple(video_latents.shape[-2:]):
            raise ValueError(
                "Parallel-stream `condition_latents` batch/channel/spatial dimensions must match video_latents, "
                f"got condition={tuple(condition_latents.shape)}, video={tuple(video_latents.shape)}."
            )
        return condition_latents.to(device=video_latents.device, dtype=video_latents.dtype)

    @staticmethod
    def _select_anchor_state(state: torch.Tensor | None) -> torch.Tensor | None:
        if state is None:
            return None
        if state.ndim == 2:
            return state
        if state.ndim == 3:
            return state[:, -1, :]
        raise ValueError(
            "Proprio context expects batch state with shape [B, state_dim] or [B, H, state_dim], "
            f"got {tuple(state.shape)}."
        )

    def _select_proprio_state(self, state: torch.Tensor | None) -> torch.Tensor | None:
        if (
            self.config.runtime_mode == ParallelRuntimeMode.FASTWAM_FIRST_FRAME
            and state is not None
            and state.ndim == 3
        ):
            return state[:, 0, :]
        return self._select_anchor_state(state)

    def _resolve_generalist_training_metadata(
        self,
        batch: PolicyTrainBatch,
    ) -> dict[str, object | None]:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        if sample_metadata is None:
            return {"mode_override": None, "drop_text": None, "source": None}
        return {
            "mode_override": sample_metadata.generalist.mode_override,
            "drop_text": sample_metadata.generalist.drop_text_conditioning,
            "source": sample_metadata.generalist.source,
        }

    def _resolve_train_sampling_metadata(
        self,
        batch: PolicyTrainBatch,
        *,
        observed_num_frames: int,
    ) -> dict[str, int | None]:
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(batch.extra.get("metadata"))
        if sample_metadata is None:
            sample_metadata = SampleConstructionMetadata(raw={})
        loss_frame_start, loss_frame_end = sample_metadata.frame_range_or_default(
            observed_num_frames=observed_num_frames,
            error_label="parallel-stream train loss-frame metadata",
        )
        latent_loss_frame_start, latent_loss_frame_end = sample_metadata.frame_range_or_default(
            observed_num_frames=observed_num_frames,
            start_key="latent_loss_frame_start",
            end_key="latent_loss_frame_end",
            default_start=loss_frame_start,
            default_end=loss_frame_end,
            error_label="parallel-stream train latent-loss metadata",
        )
        action_loss_frame_start, action_loss_frame_end = sample_metadata.frame_range_or_default(
            observed_num_frames=observed_num_frames,
            start_key="action_loss_frame_start",
            end_key="action_loss_frame_end",
            default_start=loss_frame_start,
            default_end=loss_frame_end,
            error_label="parallel-stream train action-loss metadata",
        )
        frame_shift = (
            int(sample_metadata.frame_shift)
            if self.config.temporal_position_mode == TemporalPositionMode.GLOBAL_SHIFTED
            and sample_metadata.frame_shift is not None
            else 0
        )
        chunk_origin_frame = 0
        if str(sample_metadata.raw.get("target_alignment", "")) == "next_after_context":
            chunk_origin_frame = int(loss_frame_start)
        return {
            "chunk_size": sample_metadata.sampled_chunk_size_for(observed_num_frames),
            "window_size": sample_metadata.sampled_window_size,
            "loss_frame_start": loss_frame_start,
            "loss_frame_end": loss_frame_end,
            "latent_loss_frame_start": latent_loss_frame_start,
            "latent_loss_frame_end": latent_loss_frame_end,
            "action_loss_frame_start": action_loss_frame_start,
            "action_loss_frame_end": action_loss_frame_end,
            "frame_shift": frame_shift,
            "chunk_origin_frame": chunk_origin_frame,
        }

    def _prepare_exact_train_actions(
        self,
        batch: PolicyTrainBatch,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.exact_action_adapter.supports_raw_actions:
            if batch.actions.shape[-1] != self.action_dim:
                raise ValueError(
                    "Exact LingBot training expects model-space supervision when no action adapter is configured, "
                    f"got action dim {batch.actions.shape[-1]} and model action dim {self.action_dim}."
                )
            action_mask = batch.action_mask.to(device=device, dtype=dtype) if batch.action_mask is not None else None
            return batch.actions.to(device=device, dtype=dtype), action_mask

        resolved_action_space = self.exact_action_adapter.infer_action_space(batch.actions)
        model_actions = self.exact_action_adapter.to_model_action_sequence(
            batch.actions,
            action_space=resolved_action_space,
            device=device,
            dtype=dtype,
        )
        action_mask = batch.action_mask
        if action_mask is None and resolved_action_space == ActionSpace.RAW:
            action_mask = torch.ones_like(batch.actions)
        model_action_mask = (
            self.exact_action_adapter.to_model_action_mask_sequence(
                action_mask,
                action_space=resolved_action_space,
                device=device,
                dtype=dtype,
            )
            if action_mask is not None
            else None
        )
        return model_actions, model_action_mask

    def _append_generalist_mode_text_token(self, reference_transformer: torch.nn.Module, train_artifacts) -> int:
        if not self._uses_generalist_mode_text_token():
            return 0
        raw_mode = train_artifacts.input_dict.get("joint_denoise_training_mode")
        if raw_mode is None:
            raise ValueError(
                "`generalist_mode_text_token = true` requires `joint_denoise_training_mode` "
                "in parallel-stream train artifacts."
            )
        mode = JointDenoiseTrainingMode(raw_mode).value
        latent_dict = train_artifacts.input_dict["latent_dict"]
        action_dict = train_artifacts.input_dict["action_dict"]
        text_emb = latent_dict["text_emb"]
        if action_dict["text_emb"].shape != text_emb.shape:
            raise ValueError(
                "Generalist mode text-token appending expects latent/action text embeddings "
                f"to share shape, got latent={tuple(text_emb.shape)} "
                f"and action={tuple(action_dict['text_emb'].shape)}."
            )
        append = getattr(reference_transformer, "append_generalist_mode_context_token", None)
        if not callable(append):
            raise ValueError(
                "Generalist mode text-token ablation requires the runtime transformer "
                "to support mode-token appending."
            )
        appended_text = append(text_emb, mode)
        token_count = int(appended_text.shape[1] - text_emb.shape[1])
        if token_count != 1:
            raise ValueError(
                "Generalist mode text-token ablation expects exactly one appended token, "
                f"got {token_count}."
            )
        latent_dict["text_emb"] = appended_text
        action_dict["text_emb"] = appended_text
        train_artifacts.input_dict["generalist_mode_text_token"] = mode
        train_artifacts.input_dict["generalist_mode_text_token_count"] = token_count
        return token_count

    def _reference_action_channel_mask(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.exact_action_adapter.spec is None:
            return None
        mask = torch.zeros(self.action_dim, device=device, dtype=dtype)
        used_ids = torch.tensor(self.exact_action_adapter.spec.used_action_channel_ids, device=device, dtype=torch.long)
        mask.index_fill_(0, used_ids, 1.0)
        return mask.view(1, self.action_dim, 1, 1, 1)

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        del visual_outputs
        # Method 1 is intentionally exact-runtime-only. The shared backbone
        # still owns the transformer weights, but train-time packing, attention
        # profile selection, and projection semantics live in the exact runtime
        # helper to preserve LingBot behavior.
        reference_transformer = visual_tower.ensure_runtime_backbone_device(
            action_dim=self.action_dim,
            device=prepared_inputs.batch.actions.device,
        )
        train_artifacts = prepared_inputs.variant_inputs["lingbot_train_artifacts"]
        self._append_generalist_mode_text_token(reference_transformer, train_artifacts)
        proprio_state = train_artifacts.input_dict.get("proprio_state")
        if proprio_state is not None:
            latent_dict = train_artifacts.input_dict["latent_dict"]
            action_dict = train_artifacts.input_dict["action_dict"]
            text_emb = latent_dict["text_emb"]
            append = getattr(reference_transformer, "append_proprio_context_tokens", None)
            if not callable(append):
                raise ValueError(
                    "Deprecated text-space proprio token mode requires the runtime transformer "
                    "to support proprio appending."
                )
            base_text_token_count = int(text_emb.shape[1])
            appended_text = append(text_emb, proprio_state)
            latent_dict["text_emb"] = appended_text
            action_dict["text_emb"] = appended_text
            train_artifacts.input_dict["base_text_token_count"] = base_text_token_count
            train_artifacts.input_dict["proprio_context_token_count"] = int(
                appended_text.shape[1] - base_text_token_count
            )
        runtime_input_dict = dict(train_artifacts.input_dict)
        runtime_input_dict.pop("proprio_state", None)
        if self.config.runtime_mode == ParallelRuntimeMode.FASTWAM_FIRST_FRAME:
            latent_pred, action_pred = run_parallel_fastwam_first_frame_train(
                reference_transformer,
                runtime_input_dict,
            )
        elif self.config.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
            latent_pred, action_pred = run_parallel_action_conditioned_train(
                reference_transformer,
                runtime_input_dict,
            )
        else:
            latent_pred, action_pred = run_parallel_exact_train(
                reference_transformer,
                runtime_input_dict,
            )
        return PolicyTrainOutput(
            policy_features=action_pred,
            metrics={"packed_sequence_length": torch.tensor(float(action_pred.shape[1]), device=action_pred.device)},
            aux={
                "variant": self.config.name,
                "runtime_mode": self.config.runtime_mode,
                "latent_pred": latent_pred,
                "lingbot_train_artifacts": train_artifacts,
                "loss_weights": {
                    "latent": self.training_config.objective_weight("latent"),
                    "action": self.training_config.objective_weight("action"),
                },
                "patch_size": (
                    self.backbone_config.patch_size_t,
                    self.backbone_config.patch_size_h,
                    self.backbone_config.patch_size_w,
                ),
                "debug": {
                    "sampled_chunk_size": train_artifacts.input_dict["chunk_size"],
                    "sampled_window_size": train_artifacts.input_dict["window_size"],
                    "generalist_mode_text_token_count": train_artifacts.input_dict.get(
                        "generalist_mode_text_token_count",
                        0,
                    ),
                },
            },
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        if previous_state is not None:
            return previous_state
        del visual_outputs, context
        cursor = RolloutCursor(current_start_frame=0, block_index=0, chunk_size=self.inference_config.frame_chunk_size)
        return PolicyInferState(
            step_index=0,
            cursor=cursor,
            cache={
                "runtime_mode": self._runtime_mode_label(),
                "cache_name": "open_wam_exact",
                "cache_initialized": False,
                "frame_start": 0,
                "step_index": 0,
                "backbone_cache": visual_tower.resolve_runtime_cache_state(
                    None,
                    cursor=cursor,
                    stage="parallel_stream_lingbot_exact",
                ),
            },
        )

    def reset_reference_runtime(
        self,
        *,
        visual_tower: VisualTower,
        cache_name: str = "open_wam_exact",
    ) -> PolicyInferState:
        visual_tower.reset_runtime_backbone_cache(action_dim=self.action_dim, cache_name=cache_name)
        cursor = RolloutCursor(current_start_frame=0, block_index=0, chunk_size=self.inference_config.frame_chunk_size)
        return PolicyInferState(
            step_index=0,
            cursor=cursor,
            cache={
                "runtime_mode": self._runtime_mode_label(),
                "cache_name": cache_name,
                "cache_initialized": False,
                "frame_start": 0,
                "step_index": 0,
                "backbone_cache": visual_tower.resolve_runtime_cache_state(
                    None,
                    cursor=cursor,
                    stage="parallel_stream_lingbot_exact",
                ),
            },
        )

    def warm_reference_cache(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        *,
        action_history: torch.Tensor,
        infer_state: PolicyInferState,
        action_space: ActionSpace | str = ActionSpace.AUTO,
        frame_start_override: int | None = None,
        action_conditioning_mode: object = "vanilla_joint_rollout",
        proprio_state: torch.Tensor | None = None,
    ) -> PolicyInferState:
        if self.config.runtime_mode in {
            ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
            ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
        }:
            return infer_state
        # Warmup mirrors the original LingBot server lifecycle: observed video
        # and aligned action history are committed to the exact cache before any
        # new chunk is denoised.
        reference_transformer = visual_tower.ensure_runtime_backbone_device(
            action_dim=self.action_dim,
            device=visual_outputs.frontend.video_latents.device,
        )
        observed_video_latents = visual_outputs.frontend.video_latents
        observed_action_latents = self.exact_action_adapter.to_model_action_latents(
            action_history,
            action_per_frame=self.config.action_per_frame,
            action_space=action_space,
            device=observed_video_latents.device,
            dtype=observed_video_latents.dtype,
        )
        resolved_proprio_state = self._resolve_proprio_state(
            proprio_state,
            label="parallel-stream cache warmup",
            infer_cache=infer_state.cache,
        )
        resolved_hidden_proprio_state = self._resolve_per_chunk_proprio_state(
            proprio_state,
            label="parallel-stream cache warmup",
            infer_cache=infer_state.cache,
        )
        next_cache = run_parallel_exact_cache_warmup(
            transformer=reference_transformer,
            backbone_config=self.backbone_config,
            policy_config=self.config,
            inference_config=self.inference_config,
            observed_video_latents=observed_video_latents,
            observed_action_latents=observed_action_latents,
            text_emb=visual_outputs.frontend.conditioning.text_context,
            negative_text_emb=visual_outputs.frontend.conditioning.negative_text_context,
            action_channel_mask=self._reference_action_channel_mask(
                device=observed_video_latents.device,
                dtype=observed_video_latents.dtype,
            ),
            infer_cache=infer_state.cache,
            cache_write_mode=self.exact_cache_write_mode(),
            frame_start_override=frame_start_override,
            action_conditioning_mode=str(getattr(action_conditioning_mode, "value", action_conditioning_mode)),
            proprio_state=resolved_proprio_state,
            hidden_proprio_state=resolved_hidden_proprio_state,
        )
        self._cache_proprio_state(
            next_cache,
            resolved_proprio_state if resolved_proprio_state is not None else resolved_hidden_proprio_state,
        )
        next_cache["backbone_cache"] = visual_tower.resolve_runtime_cache_state(
            next_cache.get("backbone_cache") if isinstance(next_cache.get("backbone_cache"), CacheState) else None,
            cursor=infer_state.cursor,
            stage="parallel_stream_lingbot_exact",
            payload={"cache_name": str(next_cache.get("cache_name", infer_state.cache.get("cache_name", "open_wam_exact")))},
        )
        frame_start = int(next_cache.get("frame_start", infer_state.cursor.current_start_frame))
        return PolicyInferState(
            step_index=int(next_cache["step_index"]),
            cursor=RolloutCursor(
                current_start_frame=frame_start,
                block_index=int(next_cache.get("step_index", infer_state.step_index)),
                chunk_size=self.inference_config.frame_chunk_size,
            ),
            cache=next_cache,
        )

    def generate_reference_chunk(
        self,
        *,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs | None,
        infer_state: PolicyInferState,
        text_context: torch.Tensor | None = None,
        negative_text_context: torch.Tensor | None = None,
        proprio_state: torch.Tensor | None = None,
        advance_frame_start: bool = False,
        skip_video_prediction: bool = False,
        action_conditioning_mode: object = "vanilla_joint_rollout",
    ) -> PolicyInferOutput:
        # Chunk generation stays exact-runtime-native as well. This keeps the
        # canonical method-1 policy variant small: the variant owns rollout
        # control and adapter conversion, while the exact runtime helper owns
        # the LingBot denoising schedule itself.
        if visual_outputs is not None:
            reference_transformer = visual_tower.ensure_runtime_backbone_device(
                action_dim=self.action_dim,
                device=visual_outputs.frontend.video_latents.device,
            )
            condition_latents = visual_outputs.frontend.video_latents
            text_emb = visual_outputs.frontend.conditioning.text_context
            negative_text_emb = visual_outputs.frontend.conditioning.negative_text_context
            output_dtype = condition_latents.dtype
        else:
            reference_transformer = visual_tower.get_runtime_backbone(action_dim=self.action_dim)
            parameter = next(reference_transformer.parameters())
            condition_latents = None
            text_emb = text_context
            negative_text_emb = negative_text_context
            output_dtype = torch.float32 if parameter.device.type == "cpu" else parameter.dtype
        resolved_proprio_state = self._resolve_proprio_state(
            proprio_state,
            label="parallel-stream inference",
            infer_cache=infer_state.cache,
        )
        resolved_hidden_proprio_state = self._resolve_per_chunk_proprio_state(
            proprio_state,
            label="parallel-stream inference",
            infer_cache=infer_state.cache,
        )
        if self.config.runtime_mode == ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK:
            if visual_outputs is None:
                raise ValueError("Current-frame action-chunk inference requires visual outputs for every chunk.")
            infer_artifacts = run_parallel_current_frame_action_chunk_inference_rollout(
                transformer=reference_transformer,
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                inference_config=self.inference_config,
                action_dim=self.action_dim,
                condition_latents=condition_latents,
                text_emb=text_emb,
                negative_text_emb=negative_text_emb,
                action_channel_mask=self._reference_action_channel_mask(
                    device=condition_latents.device,
                    dtype=output_dtype,
                ),
                infer_cache=infer_state.cache,
                advance_frame_start=True,
                proprio_state=resolved_proprio_state,
                hidden_proprio_state=resolved_hidden_proprio_state,
            )
        elif self.config.runtime_mode == ParallelRuntimeMode.FASTWAM_FIRST_FRAME:
            if visual_outputs is None:
                raise ValueError("FastWAM first-frame inference requires visual outputs for every chunk.")
            infer_artifacts = run_parallel_fastwam_first_frame_inference_rollout(
                transformer=reference_transformer,
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                inference_config=self.inference_config,
                action_dim=self.action_dim,
                condition_latents=condition_latents,
                text_emb=text_emb,
                negative_text_emb=negative_text_emb,
                action_channel_mask=self._reference_action_channel_mask(
                    device=condition_latents.device,
                    dtype=output_dtype,
                ),
                infer_cache=infer_state.cache,
                advance_frame_start=True,
                proprio_state=resolved_proprio_state,
                hidden_proprio_state=resolved_hidden_proprio_state,
            )
        elif resolve_parallel_current_block_coupling(self.config) in {
            CurrentBlockCoupling.JOINT,
            CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
            CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
        }:
            if skip_video_prediction:
                raise ValueError("`skip_video_prediction` is only supported by staged exact M1 rollout modes.")
            infer_artifacts = run_parallel_action_conditioned_inference_rollout(
                transformer=reference_transformer,
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                inference_config=self.inference_config,
                action_dim=self.action_dim,
                condition_latents=condition_latents,
                text_emb=text_emb,
                negative_text_emb=negative_text_emb,
                action_channel_mask=self._reference_action_channel_mask(
                    device=parameter.device if visual_outputs is None else condition_latents.device,
                    dtype=output_dtype,
                ),
                infer_cache=infer_state.cache,
                advance_frame_start=advance_frame_start,
                action_conditioning_mode=action_conditioning_mode,
                proprio_state=resolved_proprio_state,
                hidden_proprio_state=resolved_hidden_proprio_state,
            )
        else:
            infer_artifacts = run_parallel_exact_inference_rollout(
                transformer=reference_transformer,
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                inference_config=self.inference_config,
                action_dim=self.action_dim,
                condition_latents=condition_latents,
                text_emb=text_emb,
                negative_text_emb=negative_text_emb,
                action_channel_mask=self._reference_action_channel_mask(
                    device=parameter.device if visual_outputs is None else condition_latents.device,
                    dtype=output_dtype,
                ),
                infer_cache=infer_state.cache,
                advance_frame_start=advance_frame_start,
                skip_video_prediction=skip_video_prediction,
                proprio_state=resolved_proprio_state,
                hidden_proprio_state=resolved_hidden_proprio_state,
            )
        self._cache_proprio_state(
            infer_artifacts.next_cache,
            resolved_proprio_state if resolved_proprio_state is not None else resolved_hidden_proprio_state,
        )
        next_cursor = RolloutCursor(
            current_start_frame=int(
                infer_artifacts.next_cache.get("frame_start", infer_state.cursor.current_start_frame)
            ),
            block_index=int(infer_artifacts.next_cache.get("step_index", infer_state.step_index)),
            chunk_size=self.inference_config.frame_chunk_size,
        )
        infer_artifacts.next_cache["backbone_cache"] = visual_tower.advance_runtime_cache_state(
            visual_tower.resolve_runtime_cache_state(
                infer_state.cache.get("backbone_cache"),
                cursor=infer_state.cursor,
                stage="parallel_stream_lingbot_exact",
                payload={"cache_name": str(infer_state.cache.get("cache_name", "open_wam_exact"))},
            ),
            next_cursor=next_cursor,
            payload_updates={"cache_name": str(infer_artifacts.next_cache.get("cache_name", infer_state.cache.get("cache_name", "open_wam_exact")))},
        )
        raw_chunk_action = self.exact_action_adapter.to_raw_action_sequence(infer_artifacts.action_pred)
        return PolicyInferOutput(
            policy_features=infer_artifacts.action_pred.to(dtype=output_dtype),
            next_state=PolicyInferState(
                step_index=int(infer_artifacts.next_cache["step_index"]),
                cursor=next_cursor,
                cache=infer_artifacts.next_cache,
            ),
            aux={
                "variant": self.config.name,
                "runtime_mode": self.config.runtime_mode,
                "predicted_latents": infer_artifacts.predicted_latents,
                "chunk_action_pred": infer_artifacts.action_pred,
                "raw_chunk_action_pred": raw_chunk_action,
                "debug": infer_artifacts.debug,
            },
        )

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        warmed_state = infer_state
        condition_outputs: VisualStageOutputs | None = visual_outputs
        action_conditioning_mode = context.extra.get("action_conditioning_mode", "vanilla_joint_rollout")
        if (
            context.previous_action is not None
            and self.config.runtime_mode
            not in {
                ParallelRuntimeMode.CURRENT_FRAME_ACTION_CHUNK,
                ParallelRuntimeMode.FASTWAM_FIRST_FRAME,
            }
        ):
            batch_size = visual_outputs.frontend.video_latents.shape[0]
            device = visual_outputs.frontend.video_latents.device
            previous_actions = expand_previous_action(
                previous_action=context.previous_action,
                batch_size=batch_size,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
                device=device,
                dtype=visual_outputs.frontend.video_latents.dtype,
            )
            warmed_state = self.warm_reference_cache(
                visual_tower,
                visual_outputs,
                action_history=previous_actions,
                infer_state=infer_state,
                action_space=ActionSpace.MODEL,
                action_conditioning_mode=str(getattr(action_conditioning_mode, "value", action_conditioning_mode)),
                proprio_state=self._select_proprio_state(context.state),
            )
            condition_outputs = None
        return self.generate_reference_chunk(
            visual_tower=visual_tower,
            visual_outputs=condition_outputs,
            infer_state=warmed_state,
            proprio_state=self._select_proprio_state(context.state),
            action_conditioning_mode=str(getattr(action_conditioning_mode, "value", action_conditioning_mode)),
        )

    def _validate_reference_profile(self) -> None:
        if self.reference_profile is None:
            return
        if self.reference_profile.max_text_tokens != self.backbone_config.max_text_tokens:
            raise ValueError(
                "Exact LingBot reference profile max_text_tokens does not match the backbone config, "
                f"profile={self.reference_profile.max_text_tokens}, config={self.backbone_config.max_text_tokens}."
            )
        if self.reference_profile.action_dim != self.action_dim:
            raise ValueError(
                "Exact LingBot reference profile action_dim does not match the current experiment action dim, "
                f"profile={self.reference_profile.action_dim}, config={self.action_dim}."
            )
        if self.reference_profile.action_per_frame != self.config.action_per_frame:
            raise ValueError(
                "Exact LingBot reference profile action_per_frame does not match the policy config, "
                f"profile={self.reference_profile.action_per_frame}, config={self.config.action_per_frame}."
            )
        if self.reference_profile.frame_chunk_size != self.config.frame_chunk_size:
            raise ValueError(
                "Exact LingBot reference profile frame_chunk_size does not match the policy config, "
                f"profile={self.reference_profile.frame_chunk_size}, config={self.config.frame_chunk_size}."
            )
        if self.reference_profile.frame_chunk_size != self.inference_config.frame_chunk_size:
            raise ValueError(
                "Exact LingBot reference profile frame_chunk_size does not match the inference config, "
                f"profile={self.reference_profile.frame_chunk_size}, config={self.inference_config.frame_chunk_size}."
            )
        if self.reference_profile.attn_window != self.config.attn_window:
            raise ValueError(
                "Exact LingBot reference profile attn_window does not match the policy config, "
                f"profile={self.reference_profile.attn_window}, config={self.config.attn_window}."
            )
        requires_guidance_profile_match = (
            self.config.variant_profile == ParallelStreamVariantProfile.GENERALIST_JOINT_DENOISING
        )
        if (
            self.reference_profile.guidance_scale != self.inference_config.guidance_scale
            and requires_guidance_profile_match
        ):
            raise ValueError(
                "Exact LingBot reference profile guidance_scale does not match the inference config, "
                f"profile={self.reference_profile.guidance_scale}, config={self.inference_config.guidance_scale}."
            )
        if self.reference_profile.action_guidance_scale != self.inference_config.action_guidance_scale:
            raise ValueError(
                "Exact LingBot reference profile action_guidance_scale does not match the inference config, "
                f"profile={self.reference_profile.action_guidance_scale}, config={self.inference_config.action_guidance_scale}."
            )
        if self.reference_profile.video_num_inference_steps != self.inference_config.video_num_inference_steps:
            raise ValueError(
                "Exact LingBot reference profile video_num_inference_steps does not match the inference config, "
                f"profile={self.reference_profile.video_num_inference_steps}, "
                f"config={self.inference_config.video_num_inference_steps}."
            )
        if self.reference_profile.action_num_inference_steps != self.inference_config.action_num_inference_steps:
            raise ValueError(
                "Exact LingBot reference profile action_num_inference_steps does not match the inference config, "
                f"profile={self.reference_profile.action_num_inference_steps}, "
                f"config={self.inference_config.action_num_inference_steps}."
            )
        if self.reference_profile.video_exec_step != self.inference_config.video_exec_step:
            raise ValueError(
                "Exact LingBot reference profile video_exec_step does not match the inference config, "
                f"profile={self.reference_profile.video_exec_step}, config={self.inference_config.video_exec_step}."
            )
        if self.reference_profile.video_sigma_shift != self.training_config.video_sigma_shift:
            raise ValueError(
                "Exact LingBot reference profile video_sigma_shift does not match the training config, "
                f"profile={self.reference_profile.video_sigma_shift}, config={self.training_config.video_sigma_shift}."
            )
        if self.reference_profile.action_sigma_shift != self.training_config.action_sigma_shift:
            raise ValueError(
                "Exact LingBot reference profile action_sigma_shift does not match the training config, "
                f"profile={self.reference_profile.action_sigma_shift}, config={self.training_config.action_sigma_shift}."
            )
