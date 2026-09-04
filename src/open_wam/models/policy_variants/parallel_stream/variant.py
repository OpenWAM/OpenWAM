from __future__ import annotations

from dataclasses import replace

import torch

from open_wam.configs import (
    ActionSpace,
    CurrentBlockCoupling,
    InferenceConfig,
    ParallelExactCacheWriteMode,
    ParallelRuntimeMode,
    TemporalPositionMode,
    TrainingConfig,
)
from open_wam.configs.backbone import SharedVideoTransformerConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.configs.policy_video_action import supports_dynamics_routing
from open_wam.contracts import SampleConstructionMetadata
from open_wam.models.common.dynamics_objectives import (
    DynamicsSamplePlan,
    resolve_dynamics_rollout_plan,
    resolve_dynamics_sample_plan,
)
from open_wam.models.policy_variants.common.layouts import expand_previous_action
from open_wam.models.video_backbone.contracts import CacheState
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from ..base import VideoActionPolicyVariant
from ..contracts import (
    DecoderArtifactEnvelope,
    DynamicsRolloutRequest,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTemporalGeometry,
    PolicyTrainBatch,
    PolicyTrainOutput,
    PolicyVisualStage,
    RolloutCursor,
)
from .action_adapter import LingbotActionAdapter, build_action_adapter_spec
from .cache_lifecycle import run_parallel_exact_cache_warmup
from .conditioning import ParallelStreamConditioning
from .decoder_artifacts import (
    PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT,
    ParallelDecoderInferArtifacts,
    ParallelDecoderTrainArtifacts,
)
from .forward_execution import (
    run_parallel_action_conditioned_train,
    run_parallel_exact_train,
)
from .packed_rollout import run_parallel_packed_inference_rollout
from .reference_profile import (
    LingbotReferenceRuntimeContract,
    validate_reference_profile,
)
from .runtime_semantics import resolve_parallel_current_block_coupling
from .staged_rollout import run_parallel_staged_inference_rollout
from .training_exact_artifacts import (
    prepare_parallel_action_conditioned_train_artifacts,
    prepare_parallel_exact_train_artifacts,
)
from .training_prefix_artifacts import (
    prepare_parallel_prefix_condition_exact_train_artifacts,
)


class ParallelStreamPolicyVariant(VideoActionPolicyVariant):
    """LingBot-style parallel-stream policy variant.

    The canonical parallel-stream path is exact-runtime-only. Training and inference
    semantics live in role-owned parallel-stream modules and execute on the
    shared runtime backbone; this variant intentionally avoids maintaining a
    second local packed-sequence implementation.
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
        self.config = config
        self.conditioning = ParallelStreamConditioning(config)
        self.backbone_config = backbone_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_frames = num_frames
        self.exact_action_adapter = LingbotActionAdapter(
            build_action_adapter_spec(config, model_action_dim=action_dim)
        )
        self.reference_profile = (
            self.exact_action_adapter.spec.reference_profile
            if self.exact_action_adapter.spec is not None
            else None
        )
        self._validate_reference_runtime_config(self.inference_config)

    def _validate_reference_runtime_config(
        self,
        inference_config: InferenceConfig,
    ) -> None:
        validate_reference_profile(
            self.reference_profile,
            LingbotReferenceRuntimeContract(
                max_text_tokens=self.backbone_config.max_text_tokens,
                action_dim=self.action_dim,
                action_per_frame=self.config.action_per_frame,
                policy_frame_chunk_size=self.config.frame_chunk_size,
                inference_frame_chunk_size=inference_config.frame_chunk_size,
                attn_window=inference_config.attention_window_size,
                guidance_scale=inference_config.guidance_scale,
                require_guidance_scale_match=supports_dynamics_routing(
                    self.config.program
                ),
                action_guidance_scale=inference_config.action_guidance_scale,
                video_num_inference_steps=inference_config.video_num_inference_steps,
                action_num_inference_steps=inference_config.action_num_inference_steps,
                video_exec_step=inference_config.video_exec_step,
                video_sigma_shift=self.training_config.video_sigma_shift,
                action_sigma_shift=self.training_config.action_sigma_shift,
            ),
        )

    @property
    def decoder_artifact_contract(self) -> str:
        return PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT

    @property
    def source_action_channel_ids(self) -> tuple[int, ...]:
        """Expose the adapter projection through the shared pipeline contract."""

        adapter_spec = self.exact_action_adapter.spec
        if adapter_spec is None:
            return ()
        return adapter_spec.used_action_channel_ids

    @property
    def accepted_source_action_shapes(self) -> tuple[tuple[int, int], ...]:
        """Accept raw or already-adapted actions for exact LingBot profiles."""

        adapter_spec = self.exact_action_adapter.spec
        if adapter_spec is None:
            return super().accepted_source_action_shapes
        return tuple(
            dict.fromkeys(
                (
                    (adapter_spec.raw_action_dim, self.action_horizon),
                    (adapter_spec.model_action_dim, self.action_horizon),
                )
            )
        )

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

    def required_visual_stages(self) -> tuple[PolicyVisualStage, ...]:
        return (PolicyVisualStage.FRONTEND,)

    def validate_pipeline_assembly(
        self,
        *,
        data_action_dim: int,
        num_frames: int,
        backbone_num_layers: int,
    ) -> None:
        super().validate_pipeline_assembly(
            data_action_dim=data_action_dim,
            num_frames=num_frames,
            backbone_num_layers=backbone_num_layers,
        )
        self._validate_action_layout(self.action_horizon, num_frames=num_frames)

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
        self._validate_action_layout(
            batch.actions.shape[1], num_frames=observed_num_frames
        )
        model_actions, model_action_mask = self._prepare_exact_train_actions(
            batch,
            device=visual_outputs.frontend.video_latents.device,
            dtype=visual_outputs.frontend.video_latents.dtype,
        )
        sample_metadata = SampleConstructionMetadata.from_batch_metadata(
            batch.extra.get("metadata")
        )
        dynamics_sample_plan = resolve_dynamics_sample_plan(
            program=self.config.program,
            sample_metadata=sample_metadata,
        )
        sampled_geometry = self._resolve_train_sampling_metadata(
            sample_metadata=sample_metadata,
            dynamics_sample_plan=dynamics_sample_plan,
            observed_num_frames=observed_num_frames,
        )
        proprio_state = self.conditioning.resolve_train_proprio_context(batch)
        per_chunk_proprio_payload = (
            self.conditioning.resolve_train_hidden_proprio_context(
                batch,
                label="parallel-stream training",
            )
        )
        condition_latents = self.conditioning.resolve_train_condition_latents(
            batch,
            video_latents=visual_outputs.frontend.video_latents,
            dynamics_sample_plan=dynamics_sample_plan,
        )
        external_condition_prefix = self.conditioning.uses_external_condition_prefix(
            context_prefix_frames_in_sample=sampled_geometry[
                "context_prefix_frames_in_sample"
            ],
            dynamics_sample_plan=dynamics_sample_plan,
        )
        if external_condition_prefix:
            if not isinstance(condition_latents, torch.Tensor):
                raise ValueError(
                    "`context_condition_latent_source=single_frame_condition_latent` with no "
                    "in-sequence context requires precomputed condition_latents. "
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
                chunk_origin_frame=sampled_geometry["chunk_origin_frame"],
                singleton_chunk_frame=sampled_geometry["singleton_chunk_frame"],
                conditional_history_policy=sampled_geometry[
                    "conditional_history_policy"
                ],
                sample_metadata=sample_metadata,
                dynamics_sample_plan=dynamics_sample_plan,
            )
        elif (
            self.config.runtime_mode
            == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
        ):
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
                singleton_chunk_frame=sampled_geometry["singleton_chunk_frame"],
                conditional_history_policy=sampled_geometry[
                    "conditional_history_policy"
                ],
                sample_metadata=sample_metadata,
                dynamics_sample_plan=dynamics_sample_plan,
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
                singleton_chunk_frame=sampled_geometry["singleton_chunk_frame"],
                conditional_history_policy=sampled_geometry[
                    "conditional_history_policy"
                ],
            )
        if proprio_state is not None:
            train_artifacts.input_dict["proprio_state"] = proprio_state
        self.conditioning.attach_train_hidden_proprio_context(
            train_artifacts,
            batch=batch,
            video_latents=visual_outputs.frontend.video_latents,
            payload=per_chunk_proprio_payload,
        )
        return PolicyPreparedInputs(
            batch=batch,
            variant_inputs={
                "parallel_train_artifacts": train_artifacts,
            },
        )

    def _resolve_train_sampling_metadata(
        self,
        *,
        sample_metadata: SampleConstructionMetadata | None,
        dynamics_sample_plan: DynamicsSamplePlan | None,
        observed_num_frames: int,
    ) -> dict[str, int | str | None]:
        if sample_metadata is None:
            sample_metadata = SampleConstructionMetadata(raw={})
        dynamics_sequence = (
            None if dynamics_sample_plan is None else dynamics_sample_plan.sequence
        )
        if dynamics_sequence is not None:
            loss_frame_start, loss_frame_end = dynamics_sequence.loss_frame_range(
                observed_num_frames=observed_num_frames,
            )
            latent_loss_frame_start, latent_loss_frame_end = (
                loss_frame_start,
                loss_frame_end,
            )
            action_loss_frame_start, action_loss_frame_end = (
                loss_frame_start,
                loss_frame_end,
            )
        else:
            loss_frame_start, loss_frame_end = sample_metadata.frame_range_or_default(
                observed_num_frames=observed_num_frames,
                error_label="parallel-stream train loss-frame metadata",
            )
            latent_loss_frame_start, latent_loss_frame_end = (
                sample_metadata.frame_range_or_default(
                    observed_num_frames=observed_num_frames,
                    start_key="latent_loss_frame_start",
                    end_key="latent_loss_frame_end",
                    default_start=loss_frame_start,
                    default_end=loss_frame_end,
                    error_label="parallel-stream train latent-loss metadata",
                )
            )
            action_loss_frame_start, action_loss_frame_end = (
                sample_metadata.frame_range_or_default(
                    observed_num_frames=observed_num_frames,
                    start_key="action_loss_frame_start",
                    end_key="action_loss_frame_end",
                    default_start=loss_frame_start,
                    default_end=loss_frame_end,
                    error_label="parallel-stream train action-loss metadata",
                )
            )
        frame_shift = (
            int(sample_metadata.frame_shift)
            if self.config.temporal_position_mode == TemporalPositionMode.GLOBAL_SHIFTED
            and sample_metadata.frame_shift is not None
            else 0
        )
        chunk_origin_frame = (
            dynamics_sequence.chunk_origin_frame
            if dynamics_sequence is not None
            else sample_metadata.chunk_origin_frame_for(
                observed_num_frames=observed_num_frames,
            )
        )
        singleton_chunk_frame = (
            dynamics_sequence.singleton_chunk_frame
            if dynamics_sequence is not None
            else sample_metadata.singleton_chunk_frame_for(
                observed_num_frames=observed_num_frames,
            )
        )
        conditional_history_policy = (
            dynamics_sequence.history_policy
            if dynamics_sequence is not None
            else sample_metadata.conditional_history_policy
        )
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
            "context_prefix_frames_in_sample": (
                dynamics_sequence.context_prefix_frames
                if dynamics_sequence is not None
                else sample_metadata.context_prefix_frames_in_sample
            ),
            "chunk_origin_frame": chunk_origin_frame,
            "singleton_chunk_frame": singleton_chunk_frame,
            "conditional_history_policy": conditional_history_policy,
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
            action_mask = (
                batch.action_mask.to(device=device, dtype=dtype)
                if batch.action_mask is not None
                else None
            )
            return batch.actions.to(device=device, dtype=dtype), action_mask

        resolved_action_space = self.exact_action_adapter.infer_action_space(
            batch.actions
        )
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
        used_ids = torch.tensor(
            self.exact_action_adapter.spec.used_action_channel_ids,
            device=device,
            dtype=torch.long,
        )
        mask.index_fill_(0, used_ids, 1.0)
        return mask.view(1, self.action_dim, 1, 1, 1)

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        del visual_outputs
        # parallel-stream is intentionally exact-runtime-only. The shared backbone
        # still owns the transformer weights, but train-time packing, attention
        # profile selection, and projection semantics live in the exact runtime
        # helper to preserve LingBot behavior.
        reference_transformer = visual_tower.ensure_runtime_backbone_device(
            action_dim=self.action_dim,
            device=prepared_inputs.batch.actions.device,
        )
        train_artifacts = prepared_inputs.variant_inputs["parallel_train_artifacts"]
        self.conditioning.append_generalist_mode_text_token(
            reference_transformer, train_artifacts
        )
        self.conditioning.append_train_proprio_text_context(
            reference_transformer, train_artifacts
        )
        runtime_input_dict = dict(train_artifacts.input_dict)
        runtime_input_dict.pop("proprio_state", None)
        if (
            self.config.runtime_mode
            == ParallelRuntimeMode.LINGBOT_EXACT_ACTION_CONDITIONED
        ):
            latent_pred, action_pred = run_parallel_action_conditioned_train(
                reference_transformer,
                runtime_input_dict,
            )
        else:
            latent_pred, action_pred = run_parallel_exact_train(
                reference_transformer,
                runtime_input_dict,
            )
        loss_weights = {
            "latent": self.training_config.objective_weight("latent"),
            "action": self.training_config.objective_weight("action"),
        }
        patch_size = (
            self.backbone_config.patch_size_t,
            self.backbone_config.patch_size_h,
            self.backbone_config.patch_size_w,
        )
        decoder_payload = ParallelDecoderTrainArtifacts(
            latent_pred=latent_pred,
            runtime=train_artifacts,
            loss_weights=loss_weights,
            patch_size=patch_size,
        )
        return PolicyTrainOutput(
            policy_features=action_pred,
            metrics={
                "packed_sequence_length": torch.tensor(
                    float(action_pred.shape[1]), device=action_pred.device
                )
            },
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT,
                payload=decoder_payload,
                dynamics_objective=train_artifacts.dynamics_objective,
            ),
            aux={
                "variant": self.config.name,
                "runtime_mode": self.config.runtime_mode,
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
        del visual_outputs
        temporal_geometry = context.require_temporal_geometry()
        cursor = RolloutCursor(
            current_start_frame=0,
            block_index=0,
            chunk_size=temporal_geometry.frame_chunk_size,
        )
        return PolicyInferState(
            step_index=0,
            cursor=cursor,
            temporal_geometry=temporal_geometry,
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
        temporal_geometry: PolicyTemporalGeometry,
    ) -> PolicyInferState:
        inference_config = self._resolve_inference_config(temporal_geometry)
        visual_tower.reset_runtime_backbone_cache(
            action_dim=self.action_dim, cache_name=cache_name
        )
        cursor = RolloutCursor(
            current_start_frame=0,
            block_index=0,
            chunk_size=inference_config.frame_chunk_size,
        )
        return PolicyInferState(
            step_index=0,
            cursor=cursor,
            temporal_geometry=temporal_geometry,
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
        dynamics: DynamicsRolloutRequest | None = None,
        proprio_state: torch.Tensor | None = None,
        hidden_proprio_history: torch.Tensor | None = None,
        temporal_geometry: PolicyTemporalGeometry,
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
        resolved_proprio_state = self.conditioning.resolve_infer_proprio_context(
            proprio_state,
            label="parallel-stream cache warmup",
            infer_cache=infer_state.cache,
        )
        resolved_hidden_proprio_state = (
            self.conditioning.resolve_infer_hidden_proprio_context(
                proprio_state,
                label="parallel-stream cache warmup",
                infer_cache=infer_state.cache,
            )
        )
        dynamics_plan = resolve_dynamics_rollout_plan(
            program=self.config.program,
            request=dynamics,
        )
        resolved_inference_config = self._resolve_inference_config(
            temporal_geometry
        )
        next_cache = run_parallel_exact_cache_warmup(
            transformer=reference_transformer,
            backbone_config=self.backbone_config,
            policy_config=self.config,
            inference_config=resolved_inference_config,
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
            dynamics_objective=dynamics_plan.objective,
            proprio_state=resolved_proprio_state,
            hidden_proprio_state=resolved_hidden_proprio_state,
            hidden_proprio_history=(
                hidden_proprio_history
                if self.conditioning.uses_per_chunk_proprio_context()
                else None
            ),
        )
        self.conditioning.cache_infer_proprio_state(
            next_cache,
            resolved_proprio_state
            if resolved_proprio_state is not None
            else resolved_hidden_proprio_state,
        )
        next_cache["backbone_cache"] = visual_tower.resolve_runtime_cache_state(
            next_cache.get("backbone_cache")
            if isinstance(next_cache.get("backbone_cache"), CacheState)
            else None,
            cursor=infer_state.cursor,
            stage="parallel_stream_lingbot_exact",
            payload={
                "cache_name": str(
                    next_cache.get(
                        "cache_name",
                        infer_state.cache.get("cache_name", "open_wam_exact"),
                    )
                )
            },
        )
        frame_start = int(
            next_cache.get("frame_start", infer_state.cursor.current_start_frame)
        )
        return PolicyInferState(
            step_index=int(next_cache["step_index"]),
            cursor=RolloutCursor(
                current_start_frame=frame_start,
                block_index=int(next_cache.get("step_index", infer_state.step_index)),
                chunk_size=int(
                    next_cache.get(
                        "frame_chunk_size", resolved_inference_config.frame_chunk_size
                    )
                ),
            ),
            cache=next_cache,
            temporal_geometry=temporal_geometry,
        )

    def _pack_dynamics_action(
        self,
        value: torch.Tensor | None,
        *,
        reference_transformer: torch.nn.Module,
    ) -> torch.Tensor | None:
        """Translate the shared model-action sequence into Parallel packing."""

        if value is None:
            return None
        parameter = next(reference_transformer.parameters())
        return self.exact_action_adapter.to_model_action_latents(
            value,
            action_per_frame=self.config.action_per_frame,
            action_space=ActionSpace.MODEL,
            device=parameter.device,
            dtype=parameter.dtype,
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
        dynamics: DynamicsRolloutRequest | None = None,
        temporal_geometry: PolicyTemporalGeometry,
    ) -> PolicyInferOutput:
        # Chunk generation stays exact-runtime-native as well. This keeps the
        # canonical parallel-stream policy variant small: the variant owns rollout
        # control and adapter conversion, while the exact runtime helper owns
        # the LingBot denoising schedule itself.
        resolved_inference_config = self._resolve_inference_config(
            temporal_geometry
        )
        if visual_outputs is not None:
            reference_transformer = visual_tower.ensure_runtime_backbone_device(
                action_dim=self.action_dim,
                device=visual_outputs.frontend.video_latents.device,
            )
            condition_latents = visual_outputs.frontend.video_latents
            text_emb = visual_outputs.frontend.conditioning.text_context
            negative_text_emb = (
                visual_outputs.frontend.conditioning.negative_text_context
            )
            output_dtype = condition_latents.dtype
        else:
            reference_transformer = visual_tower.get_runtime_backbone(
                action_dim=self.action_dim
            )
            parameter = next(reference_transformer.parameters())
            condition_latents = None
            text_emb = text_context
            negative_text_emb = negative_text_context
            output_dtype = (
                torch.float32 if parameter.device.type == "cpu" else parameter.dtype
            )
        resolved_proprio_state = self.conditioning.resolve_infer_proprio_context(
            proprio_state,
            label="parallel-stream inference",
            infer_cache=infer_state.cache,
        )
        resolved_hidden_proprio_state = (
            self.conditioning.resolve_infer_hidden_proprio_context(
                proprio_state,
                label="parallel-stream inference",
                infer_cache=infer_state.cache,
            )
        )
        dynamics_plan = resolve_dynamics_rollout_plan(
            program=self.config.program,
            request=dynamics,
        )
        packed_coupling = resolve_parallel_current_block_coupling(self.config) in {
            CurrentBlockCoupling.JOINT,
            CurrentBlockCoupling.VIDEO_NOISY_TO_ACTION,
            CurrentBlockCoupling.ACTION_NOISY_TO_VIDEO,
        }
        if packed_coupling:
            if skip_video_prediction:
                raise ValueError(
                    "`skip_video_prediction` is only supported by staged exact parallel-stream rollout modes."
                )
            dynamics_plan.require_generation_inputs()
            clean_action_latents = self._pack_dynamics_action(
                dynamics_plan.clean_action,
                reference_transformer=reference_transformer,
            )
            commit_action_latents = self._pack_dynamics_action(
                dynamics_plan.history_action,
                reference_transformer=reference_transformer,
            )
            infer_artifacts = run_parallel_packed_inference_rollout(
                transformer=reference_transformer,
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                inference_config=resolved_inference_config,
                action_dim=self.action_dim,
                condition_latents=condition_latents,
                text_emb=text_emb,
                negative_text_emb=negative_text_emb,
                action_channel_mask=self._reference_action_channel_mask(
                    device=parameter.device
                    if visual_outputs is None
                    else condition_latents.device,
                    dtype=output_dtype,
                ),
                infer_cache=infer_state.cache,
                advance_frame_start=advance_frame_start,
                forced_action_latents=clean_action_latents,
                commit_action_latents=commit_action_latents,
                forced_video_latents=dynamics_plan.clean_video,
                dynamics_objective=dynamics_plan.objective,
                proprio_state=resolved_proprio_state,
                hidden_proprio_state=resolved_hidden_proprio_state,
            )
        else:
            if any(
                value is not None
                for value in (
                    dynamics_plan.clean_action,
                    dynamics_plan.clean_video,
                    dynamics_plan.history_action,
                )
            ):
                raise ValueError(
                    "Staged Parallel Stream programs do not accept dynamics "
                    "clean-modality overrides."
                )
            infer_artifacts = run_parallel_staged_inference_rollout(
                transformer=reference_transformer,
                backbone_config=self.backbone_config,
                policy_config=self.config,
                training_config=self.training_config,
                inference_config=resolved_inference_config,
                action_dim=self.action_dim,
                condition_latents=condition_latents,
                text_emb=text_emb,
                negative_text_emb=negative_text_emb,
                action_channel_mask=self._reference_action_channel_mask(
                    device=parameter.device
                    if visual_outputs is None
                    else condition_latents.device,
                    dtype=output_dtype,
                ),
                infer_cache=infer_state.cache,
                advance_frame_start=advance_frame_start,
                skip_video_prediction=skip_video_prediction,
                proprio_state=resolved_proprio_state,
                hidden_proprio_state=resolved_hidden_proprio_state,
            )
        self.conditioning.cache_infer_proprio_state(
            infer_artifacts.next_cache,
            resolved_proprio_state
            if resolved_proprio_state is not None
            else resolved_hidden_proprio_state,
        )
        next_cursor = RolloutCursor(
            current_start_frame=int(
                infer_artifacts.next_cache.get(
                    "frame_start", infer_state.cursor.current_start_frame
                )
            ),
            block_index=int(
                infer_artifacts.next_cache.get("step_index", infer_state.step_index)
            ),
            chunk_size=int(
                infer_artifacts.next_cache.get(
                    "frame_chunk_size", resolved_inference_config.frame_chunk_size
                )
            ),
        )
        infer_artifacts.next_cache["backbone_cache"] = (
            visual_tower.advance_runtime_cache_state(
                visual_tower.resolve_runtime_cache_state(
                    infer_state.cache.get("backbone_cache"),
                    cursor=infer_state.cursor,
                    stage="parallel_stream_lingbot_exact",
                    payload={
                        "cache_name": str(
                            infer_state.cache.get("cache_name", "open_wam_exact")
                        )
                    },
                ),
                next_cursor=next_cursor,
                payload_updates={
                    "cache_name": str(
                        infer_artifacts.next_cache.get(
                            "cache_name",
                            infer_state.cache.get("cache_name", "open_wam_exact"),
                        )
                    )
                },
            )
        )
        raw_chunk_action = self.exact_action_adapter.to_raw_action_sequence(
            infer_artifacts.action_pred
        )
        decoder_payload = ParallelDecoderInferArtifacts(
            predicted_latents=infer_artifacts.predicted_latents,
            raw_chunk_action_pred=raw_chunk_action,
        )
        return PolicyInferOutput(
            policy_features=infer_artifacts.action_pred.to(dtype=output_dtype),
            next_state=PolicyInferState(
                step_index=int(infer_artifacts.next_cache["step_index"]),
                cursor=next_cursor,
                cache=infer_artifacts.next_cache,
                temporal_geometry=temporal_geometry,
            ),
            decoder_artifacts=DecoderArtifactEnvelope(
                contract=PARALLEL_STREAM_DECODER_ARTIFACT_CONTRACT,
                payload=decoder_payload,
                dynamics_objective=dynamics_plan.objective,
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
        temporal_geometry = context.require_temporal_geometry()
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
                dynamics=context.dynamics,
                proprio_state=self.conditioning.select_rollout_proprio_state(
                    context.state
                ),
                temporal_geometry=temporal_geometry,
            )
            condition_outputs = None
        return self.generate_reference_chunk(
            visual_tower=visual_tower,
            visual_outputs=condition_outputs,
            infer_state=warmed_state,
            proprio_state=self.conditioning.select_rollout_proprio_state(context.state),
            dynamics=context.dynamics,
            temporal_geometry=temporal_geometry,
        )

    def _resolve_inference_config(
        self,
        temporal_geometry: PolicyTemporalGeometry,
    ) -> InferenceConfig:
        resolved = replace(
            self.inference_config,
            frame_chunk_size=int(temporal_geometry.frame_chunk_size),
            attention_window_size=int(temporal_geometry.attention_window_size),
        )
        self._validate_reference_runtime_config(resolved)
        return resolved
