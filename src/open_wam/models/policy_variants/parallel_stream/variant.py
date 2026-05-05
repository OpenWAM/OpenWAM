from __future__ import annotations

from collections.abc import Mapping
import torch

from open_wam.configs import (
    ActionSpace,
    CurrentBlockCoupling,
    InferenceConfig,
    ParallelExactCacheWriteMode,
    ParallelRuntimeMode,
    ParallelStreamPolicyConfig,
    TemporalPositionMode,
    TrainingConfig,
)
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
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_exact_train_artifacts,
    resolve_parallel_current_block_coupling,
    run_parallel_action_conditioned_inference_rollout,
    run_parallel_action_conditioned_train,
    run_parallel_exact_cache_warmup,
    run_parallel_exact_inference_rollout,
    run_parallel_exact_train,
)
from .action_adapter import LingbotActionAdapter, build_action_adapter_spec


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
        if self.config.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
            train_artifacts = prepare_parallel_action_conditioned_train_artifacts(
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                video_latents=visual_outputs.frontend.video_latents,
                actions=model_actions,
                action_mask=model_action_mask,
                text_emb=visual_outputs.frontend.conditioning.text_context,
                chunk_size_override=sampled_geometry["chunk_size"],
                window_size_override=sampled_geometry["window_size"],
                loss_frame_start=sampled_geometry["loss_frame_start"],
                loss_frame_end=sampled_geometry["loss_frame_end"],
                frame_shift=sampled_geometry["frame_shift"],
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
                chunk_size_override=sampled_geometry["chunk_size"],
                window_size_override=sampled_geometry["window_size"],
                loss_frame_start=sampled_geometry["loss_frame_start"],
                loss_frame_end=sampled_geometry["loss_frame_end"],
                frame_shift=sampled_geometry["frame_shift"],
            )
        return PolicyPreparedInputs(batch=batch, variant_inputs={"lingbot_train_artifacts": train_artifacts})

    def _resolve_train_sampling_metadata(
        self,
        batch: PolicyTrainBatch,
        *,
        observed_num_frames: int,
    ) -> dict[str, int | None]:
        metadata_seq = batch.extra.get("metadata")
        sample_metadata: Mapping[str, object] | None = None
        if isinstance(metadata_seq, tuple) and len(metadata_seq) == 1 and isinstance(metadata_seq[0], Mapping):
            sample_metadata = metadata_seq[0]
        elif isinstance(metadata_seq, list) and len(metadata_seq) == 1 and isinstance(metadata_seq[0], Mapping):
            sample_metadata = metadata_seq[0]

        chunk_size: int | None = None
        window_size: int | None = None
        loss_frame_start: int | None = None
        loss_frame_end: int | None = None
        frame_shift = 0
        if sample_metadata is not None:
            sampled_chunk_size = sample_metadata.get("sampled_chunk_size")
            sampled_window_size = sample_metadata.get("sampled_window_size")
            metadata_loss_frame_start = sample_metadata.get("loss_frame_start")
            metadata_loss_frame_end = sample_metadata.get("loss_frame_end")
            metadata_frame_shift = sample_metadata.get("frame_shift")
            if sampled_chunk_size is not None:
                chunk_size = int(sampled_chunk_size)
            if sampled_window_size is not None:
                window_size = int(sampled_window_size)
            if metadata_loss_frame_start is not None:
                loss_frame_start = int(metadata_loss_frame_start)
            if metadata_loss_frame_end is not None:
                loss_frame_end = int(metadata_loss_frame_end)
            if (
                self.config.temporal_position_mode == TemporalPositionMode.GLOBAL_SHIFTED
                and metadata_frame_shift is not None
            ):
                frame_shift = int(metadata_frame_shift)

        if loss_frame_start is None:
            loss_frame_start = 0
        if loss_frame_end is None:
            loss_frame_end = observed_num_frames
        if loss_frame_start < 0 or loss_frame_end < loss_frame_start or loss_frame_end > observed_num_frames:
            raise ValueError(
                "Invalid train loss-frame metadata for parallel-stream variant, "
                f"got start={loss_frame_start}, end={loss_frame_end}, observed_num_frames={observed_num_frames}."
            )
        return {
            "chunk_size": chunk_size,
            "window_size": window_size,
            "loss_frame_start": loss_frame_start,
            "loss_frame_end": loss_frame_end,
            "frame_shift": frame_shift,
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
        if self.config.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED:
            latent_pred, action_pred = run_parallel_action_conditioned_train(
                reference_transformer,
                train_artifacts.input_dict,
            )
        else:
            latent_pred, action_pred = run_parallel_exact_train(
                reference_transformer,
                train_artifacts.input_dict,
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
    ) -> PolicyInferState:
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
        advance_frame_start: bool = False,
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
        if resolve_parallel_current_block_coupling(self.config) in {
            CurrentBlockCoupling.JOINT,
            CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
            CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
        }:
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
        if context.previous_action is not None:
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
            )
            condition_outputs = None
        return self.generate_reference_chunk(
            visual_tower=visual_tower,
            visual_outputs=condition_outputs,
            infer_state=warmed_state,
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
        if self.reference_profile.guidance_scale != self.inference_config.guidance_scale:
            raise ValueError(
                "Exact LingBot reference profile guidance_scale does not match the inference config, "
                f"profile={self.reference_profile.guidance_scale}, config={self.inference_config.guidance_scale}."
            )
        if self.reference_profile.action_guidance_scale != self.inference_config.action_guidance_scale:
            raise ValueError(
                "Exact LingBot reference profile action_guidance_scale does not match the inference config, "
                f"profile={self.reference_profile.action_guidance_scale}, config={self.inference_config.action_guidance_scale}."
            )
        if self.config.runtime_mode == ParallelRuntimeMode.LINGBOT_EXACT:
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
