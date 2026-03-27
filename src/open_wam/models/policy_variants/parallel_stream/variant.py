from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import InferenceConfig, ParallelStreamPolicyConfig, TrainingConfig
from open_wam.models.action_decoders import ActionDecoderInferOutput, ActionDecoderTrainOutput
from open_wam.models.common.flow_matching import (
    build_frame_aligned_action_flow_match_train_artifacts,
    build_video_flow_match_inference_scheduler,
    build_video_flow_match_train_artifacts,
    build_action_flow_match_inference_scheduler,
    denoised_actions_from_flow,
    denoised_video_latents_from_flow,
    reduce_frame_aligned_action_flow_match_loss,
    reduce_video_flow_match_loss,
)
from open_wam.models.common.video_geometry import unpatchify_video_tokens
from open_wam.models.video_backbone.contracts import CacheState
from open_wam.models.video_backbone.config import LingbotCompatibleVideoBackboneConfig
from open_wam.models.policy_variants.common.layouts import expand_previous_action
from open_wam.models.policy_variants.common.timesteps import build_token_timestep_context
from open_wam.models.visual_tower.grid_ids import build_action_grid_ids, build_video_grid_ids
from open_wam.models.visual_tower import VisualCoreInput, VisualStageOutputs, VisualTower

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
from .masks import build_parallel_attention_mask
from .packing import ParallelPackedSequenceLayout, action_tokens_to_frame_major, build_parallel_layout
from .positions import build_parallel_position_context
from .reference_runtime import (
    prepare_parallel_exact_train_artifacts,
    run_parallel_exact_cache_warmup,
    run_parallel_exact_inference_rollout,
    run_parallel_exact_train,
)
from .action_adapter import LingbotActionAdapter, build_action_adapter_spec


class ParallelStreamPolicyVariant(PolicyVariant):
    """LingBot-style parallel-stream policy variant."""

    def __init__(
        self,
        config: ParallelStreamPolicyConfig,
        backbone_config: LingbotCompatibleVideoBackboneConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_dim: int,
        action_horizon: int,
        num_frames: int,
    ) -> None:
        super().__init__()
        if config.runtime_mode != "lingbot_exact":
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
        self.video_patch_dim = (
            self.backbone_config.latent_channels
            * self.backbone_config.patch_size_t
            * self.backbone_config.patch_size_h
            * self.backbone_config.patch_size_w
        )
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.video_flow_head = nn.Linear(config.hidden_size, self.video_patch_dim)
        self.action_flow_head = nn.Linear(config.hidden_size, action_dim)

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend",)

    def _validate_action_layout(self, action_horizon: int) -> None:
        expected_horizon = self.num_frames * self.config.action_per_frame
        if action_horizon != expected_horizon:
            raise ValueError(
                "Parallel-stream variant requires `action_horizon == num_frames * action_per_frame`, "
                f"got action_horizon={action_horizon}, num_frames={self.num_frames}, "
                f"action_per_frame={self.config.action_per_frame}"
            )

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        self._validate_action_layout(batch.actions.shape[1])
        model_actions, model_action_mask = self._prepare_exact_train_actions(
            batch,
            device=visual_outputs.frontend.video_latents.device,
            dtype=visual_outputs.frontend.video_latents.dtype,
        )
        train_artifacts = prepare_parallel_exact_train_artifacts(
            backbone_config=self.backbone_config,
            policy_config=self.config,
            training_config=self.training_config,
            video_latents=visual_outputs.frontend.video_latents,
            actions=model_actions,
            action_mask=model_action_mask,
            text_emb=visual_outputs.frontend.conditioning.text_context,
        )
        return PolicyPreparedInputs(batch=batch, variant_inputs={"lingbot_train_artifacts": train_artifacts})

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
        if action_mask is None and resolved_action_space == "raw":
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

    def _build_action_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        self._validate_action_layout(actions.shape[1])
        hidden = self.action_embedder(actions)
        # Validate that the flat action horizon can be viewed as
        # `[B, num_frames, action_per_frame, hidden]`. The returned frame-major
        # view is intentionally discarded here because the downstream packer
        # still expects the flattened `[B, T_action, hidden]` layout.
        action_tokens_to_frame_major(hidden, self.num_frames, self.config.action_per_frame)
        return hidden

    def _pack_sequence(
        self,
        *,
        token_grid,
        video_noisy_tokens: torch.Tensor,
        video_condition_tokens: torch.Tensor,
        action_noisy_tokens: torch.Tensor,
        action_condition_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, ParallelPackedSequenceLayout]:
        layout = build_parallel_layout(
            token_grid=token_grid,
            action_per_frame=self.config.action_per_frame,
            frame_chunk_size=self.config.frame_chunk_size,
            sequence_order=self.config.sequence_order,
            device=video_noisy_tokens.device,
        )
        # Shapes before concatenation:
        # - `video_noisy` / `video_condition`: `[B, T_video, H]`
        # - `action_noisy` / `action_condition`: `[B, T_action, H]`
        # The concatenated sequence is `[B, S_total, H]`, where `S_total` is
        # the sum of span lengths in `layout.spans`.
        stream_map = {
            "video_noisy": video_noisy_tokens,
            "video_condition": video_condition_tokens,
            "action_noisy": action_noisy_tokens,
            "action_condition": action_condition_tokens,
        }
        packed_tokens = torch.cat([stream_map[name] for name in self.config.sequence_order], dim=1)
        return packed_tokens, layout

    def _build_parallel_grid_ids(
        self,
        visual_outputs: VisualStageOutputs,
        layout: ParallelPackedSequenceLayout,
    ) -> torch.Tensor:
        contexts = []
        for name in self.config.sequence_order:
            if name.startswith("video"):
                # Video spans reuse the frontend patch grid:
                # `[1, T_video, 3]` with `(frame, row, col)`-style coordinates.
                contexts.append(
                    build_video_grid_ids(
                        visual_outputs.frontend.token_grid,
                        device=visual_outputs.frontend.video_tokens.device,
                    )
                )
            else:
                # Action spans live in a compact frame-major 1D grid:
                # `[1, T_action, 3]`, aligned so each frame owns
                # `action_per_frame` adjacent action slots.
                contexts.append(
                    build_action_grid_ids(
                        num_frames=visual_outputs.frontend.token_grid.num_frames,
                        action_per_frame=self.config.action_per_frame,
                        device=visual_outputs.frontend.video_tokens.device,
                    )
                )
        return torch.cat(contexts, dim=1)

    def _build_parallel_timestep_values(
        self,
        layout: ParallelPackedSequenceLayout,
        *,
        token_grid,
        video_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        batch_size = video_timesteps.shape[0]
        device = video_timesteps.device
        zero = torch.zeros(batch_size, device=device, dtype=torch.float32)
        expanded_video_timesteps = video_timesteps.repeat_interleave(token_grid.tokens_per_frame, dim=1)
        for name in self.config.sequence_order:
            start, end = layout.spans[name]
            length = end - start
            # Timesteps are expanded from one scalar per sample to one scalar per
            # packed token. Conditioning streams receive zero so the core can
            # distinguish denoised context from actively denoised streams.
            if name == "video_noisy":
                base = expanded_video_timesteps
            elif name == "action_noisy":
                base = action_timesteps
            else:
                base = zero
            if base.ndim == 1:
                values.append(base[:, None].expand(-1, length))
            else:
                if base.shape[1] != length:
                    raise ValueError(
                        f"Expected timestep context length {length} for packed stream '{name}', got {base.shape[1]}."
                    )
                values.append(base)
        return torch.cat(values, dim=1)

    def _build_parallel_timestep_context(
        self,
        *,
        timestep_values: torch.Tensor,
    ) -> torch.Tensor:
        return build_token_timestep_context(timestep_values, self.config.hidden_size)

    def _build_parallel_stream_ids(
        self,
        layout: ParallelPackedSequenceLayout,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        stream_ids = []
        for name in self.config.sequence_order:
            start, end = layout.spans[name]
            length = end - start
            value = 0 if name.startswith("video") else 1
            stream_ids.append(torch.full((batch_size, length), value, device=device, dtype=torch.long))
        return torch.cat(stream_ids, dim=1)

    def _build_noisy_frontend_outputs(
        self,
        visual_tower: VisualTower,
        *,
        visual_outputs: VisualStageOutputs,
        noisy_video_latents: torch.Tensor,
    ) -> VisualStageOutputs:
        noisy_frontend = visual_tower.run_frontend_from_latents(
            noisy_video_latents,
            task_text=None,
            text_context=visual_outputs.frontend.conditioning.text_context,
            negative_text_context=visual_outputs.frontend.conditioning.negative_text_context,
            canonical_video=visual_outputs.frontend.canonical_video,
        )
        return VisualStageOutputs(frontend=noisy_frontend)

    def _decode_parallel_video_flow(
        self,
        *,
        hidden_tokens: torch.Tensor,
        token_grid,
    ) -> torch.Tensor:
        patch_tokens = self.video_flow_head(hidden_tokens)
        return unpatchify_video_tokens(
            patch_tokens,
            token_grid=token_grid,
            latent_channels=self.backbone_config.latent_channels,
        )

    def _run_parallel_core(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        *,
        noisy_video_tokens: torch.Tensor,
        condition_video_tokens: torch.Tensor,
        noisy_action_inputs: torch.Tensor,
        condition_action_inputs: torch.Tensor,
        video_timesteps: torch.Tensor,
        action_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, ParallelPackedSequenceLayout, dict[str, object]]:
        action_noisy_tokens = self._build_action_tokens(noisy_action_inputs)
        action_condition_tokens = self._build_action_tokens(condition_action_inputs)
        packed_tokens, layout = self._pack_sequence(
            token_grid=visual_outputs.frontend.token_grid,
            video_noisy_tokens=noisy_video_tokens,
            video_condition_tokens=condition_video_tokens,
            action_noisy_tokens=action_noisy_tokens,
            action_condition_tokens=action_condition_tokens,
        )
        batch_size = packed_tokens.shape[0]
        position_context = build_parallel_position_context(
            token_grid=visual_outputs.frontend.token_grid,
            layout=layout,
            hidden_size=self.config.hidden_size,
            action_per_frame=self.config.action_per_frame,
            device=packed_tokens.device,
        )[None, :, :].expand(batch_size, -1, -1)
        timestep_values = self._build_parallel_timestep_values(
            layout=layout,
            token_grid=visual_outputs.frontend.token_grid,
            video_timesteps=video_timesteps,
            action_timesteps=action_timesteps,
        )
        timestep_context = self._build_parallel_timestep_context(timestep_values=timestep_values)
        attention_mask = build_parallel_attention_mask(layout, batch_size=batch_size, device=packed_tokens.device)
        # `VisualCoreInput` is the fully packed multimodal view of the sample:
        # - `tokens`: `[B, S_total, H]`
        # - `position_context`: `[B, S_total, H]`
        # - `grid_ids`: `[1, S_total, 3]`
        # - `timestep_values`: `[B, S_total]`
        # - `stream_ids`: `[B, S_total]` with `0=video`, `1=action/register`
        # - `attention_mask`: `[B, 1, S_total, S_total]`
        core_output = visual_tower.run_core(
            VisualCoreInput(
                tokens=packed_tokens,
                token_layout=layout,
                position_context=position_context,
                timestep_context=timestep_context,
                grid_ids=self._build_parallel_grid_ids(visual_outputs, layout),
                timestep_values=timestep_values,
                stream_ids=self._build_parallel_stream_ids(layout, batch_size=batch_size, device=packed_tokens.device),
                attention_mask=attention_mask,
                conditioning=visual_outputs.frontend.conditioning,
            )
        )
        video_start, video_end = layout.spans["video_noisy"]
        action_start, action_end = layout.spans["action_noisy"]
        return (
            core_output.tokens[:, video_start:video_end, :],
            core_output.tokens[:, action_start:action_end, :],
            layout,
            dict(core_output.aux),
        )

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        del visual_outputs
        reference_transformer = visual_tower.ensure_lingbot_reference_transformer_device(
            action_dim=self.action_dim,
            device=prepared_inputs.batch.actions.device,
        )
        train_artifacts = prepared_inputs.variant_inputs["lingbot_train_artifacts"]
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
                "runtime_mode": "lingbot_exact",
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
        visual_tower.reset_lingbot_reference_runtime(action_dim=self.action_dim, cache_name=cache_name)
        cursor = RolloutCursor(current_start_frame=0, block_index=0, chunk_size=self.inference_config.frame_chunk_size)
        return PolicyInferState(
            step_index=0,
            cursor=cursor,
            cache={
                "runtime_mode": "lingbot_exact",
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
        action_space: str = "auto",
    ) -> PolicyInferState:
        reference_transformer = visual_tower.ensure_lingbot_reference_transformer_device(
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
        if visual_outputs is not None:
            reference_transformer = visual_tower.ensure_lingbot_reference_transformer_device(
                action_dim=self.action_dim,
                device=visual_outputs.frontend.video_latents.device,
            )
            condition_latents = visual_outputs.frontend.video_latents
            text_emb = visual_outputs.frontend.conditioning.text_context
            negative_text_emb = visual_outputs.frontend.conditioning.negative_text_context
            output_dtype = condition_latents.dtype
        else:
            reference_transformer = visual_tower.get_lingbot_reference_transformer(action_dim=self.action_dim)
            parameter = next(reference_transformer.parameters())
            condition_latents = None
            text_emb = text_context
            negative_text_emb = negative_text_context
            output_dtype = torch.float32 if parameter.device.type == "cpu" else parameter.dtype
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

    def _constant_video_timestep_grid(
        self,
        *,
        batch_size: int,
        timestep_value: float,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full((batch_size, self.num_frames), fill_value=float(timestep_value), device=device, dtype=torch.float32)

    def _constant_action_timestep_grid(
        self,
        *,
        batch_size: int,
        timestep_value: float,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full((batch_size, self.action_horizon), fill_value=float(timestep_value), device=device, dtype=torch.float32)

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
                action_space="model",
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
