from __future__ import annotations

import torch
from torch import nn

from open_wam.configs import (
    InferenceConfig,
    PostDecodedPolicyConfig,
    TrainingConfig,
    VideoConditionSource,
    VisualReadoutSourceFamily,
)
from open_wam.models.visual_tower import VisualReadoutRequest, VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .common import (
    SharedVisualReadout,
    advance_default_runtime_infer_state,
    build_generated_video_condition_window,
    build_local_video_condition_window,
    prepare_default_runtime_infer_state,
    resolve_video_condition_frame_start,
    resolve_video_condition_sample_seed,
)
from .common.layouts import align_sequence_length
from .contracts import (
    DecoderSequenceContext,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
    VideoConditionWindowContext,
)


class PostDecodedPolicyVariant(PolicyVariant):
    """Policy variant over decoded visual features."""

    def __init__(
        self,
        config: PostDecodedPolicyConfig,
        training_config: TrainingConfig,
        inference_config: InferenceConfig,
        action_horizon: int,
        state_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.inference_config = inference_config
        self.action_horizon = action_horizon
        self.state_proj = nn.Linear(state_dim, config.hidden_size) if config.use_state_projection else None
        self.visual_readout = SharedVisualReadout(config.visual_readout, hidden_size=config.hidden_size)
        if config.visual_readout is not None and config.visual_readout.source_family not in {
            VisualReadoutSourceFamily.FINAL_CORE_TOKENS,
            VisualReadoutSourceFamily.CORE_LAYER_TOKENS,
            VisualReadoutSourceFamily.CORE_MULTI_LAYER_TOKENS,
        }:
            raise ValueError(
                "Post-decoded currently supports only core-based shared visual readout families, "
                f"got {config.visual_readout.source_family!r}."
            )

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        if self.config.visual_readout is None:
            return ("frontend", "core", "decode")
        return ("frontend", "core")

    def requested_visual_readout(self) -> VisualReadoutRequest | None:
        return self.visual_readout.requested_capture()

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        if self.config.visual_readout is None and visual_outputs.decode is None:
            raise ValueError("Post-decoded variant requires decode outputs.")
        return PolicyPreparedInputs(batch=batch)

    def _resolve_decode_output(self, visual_tower: VisualTower, visual_outputs: VisualStageOutputs):
        if self.config.visual_readout is None:
            if visual_outputs.decode is None:
                raise ValueError("Post-decoded variant requires decode outputs.")
            return visual_outputs.decode, "decode", {}
        if visual_outputs.core is None:
            raise ValueError("Post-decoded variant requires core outputs for configured visual readout.")
        resolved_readout = self.visual_readout.resolve_from_core(visual_outputs.core)
        decode_output = visual_tower.decode_tokens(
            visual_outputs.frontend,
            tokens=resolved_readout.tokens,
            token_layout=resolved_readout.token_layout,
        )
        return decode_output, resolved_readout.source_stage, resolved_readout.metadata

    def _extract_policy_features(self, decode_output):
        decoded_features = decode_output.decoded_features
        if decoded_features.ndim == 4:
            frame_features = decoded_features.mean(dim=2)
        elif decoded_features.ndim == 3:
            frame_features = decoded_features
        else:
            raise ValueError(
                "Expected decoded features with shape [B, T, N, D] or [B, T, D], "
                f"got {tuple(decoded_features.shape)}"
            )
        return align_sequence_length(frame_features, self.action_horizon)

    def _build_decoder_sequence_context(
        self,
        visual_outputs: VisualStageOutputs,
        *,
        state,
        source_stage: str,
        readout_metadata: dict[str, object],
        decode_output,
        video_condition_window: VideoConditionWindowContext | None = None,
    ) -> DecoderSequenceContext:
        decoded_features = decode_output.decoded_features
        if video_condition_window is None:
            video_condition_window = build_local_video_condition_window(
                visual_outputs=visual_outputs,
                input_space=self.config.video_condition_input_space,
                local_window_frames=self.config.local_video_window_frames,
                current_frame_index=self.config.current_video_frame_index,
                action_chunk_anchor_mode=self.config.action_chunk_anchor_mode,
                source_stage="frontend",
            )
        return DecoderSequenceContext(
            sequence_tokens=decoded_features,
            sequence_layout={
                "family": "video_feature_policy",
                "kind": ("frame_token_grid" if decoded_features.ndim == 4 else "frame_feature_sequence"),
                "attach_site": str(self.config.attach_site),
                "decode_feature_mode": str(self.config.decode_feature_mode),
                "pooling_mode": str(self.config.pooling_mode),
                **readout_metadata,
            },
            token_grid=visual_outputs.frontend.token_grid,
            frame_count=int(decoded_features.shape[1]),
            source_stage=source_stage,
            state_sequence=state,
            goal_features=visual_outputs.frontend.conditioning.text_context,
            video_condition_window=video_condition_window,
        )

    def _build_generated_video_condition_window(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        *,
        frame_start: int,
        observed_prefix_anchor: str = "start",
        sample_seed: int | None = None,
    ) -> tuple[VideoConditionWindowContext, dict[str, object]]:
        return build_generated_video_condition_window(
            visual_tower=visual_tower,
            visual_outputs=visual_outputs,
            input_space=self.config.video_condition_input_space,
            local_window_frames=self.config.local_video_window_frames,
            current_frame_index=self.config.current_video_frame_index,
            action_chunk_anchor_mode=self.config.action_chunk_anchor_mode,
            frame_start=int(frame_start),
            num_inference_steps=self.inference_config.video_num_inference_steps,
            num_train_timesteps=self.training_config.video_num_train_timesteps,
            sigma_shift=self.training_config.video_sigma_shift,
            guidance_scale=self.inference_config.guidance_scale,
            cache_name="post_decoded_video_condition_future",
            observed_prefix_anchor=observed_prefix_anchor,
            sample_seed=sample_seed,
        )

    def _fuse_state(self, policy_features, state):
        if state is None or self.state_proj is None:
            return policy_features
        return policy_features + self.state_proj(state.mean(dim=1))[:, None, :]

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        decode_output, source_stage, readout_metadata = self._resolve_decode_output(visual_tower, visual_outputs)
        policy_features = self._extract_policy_features(decode_output)
        policy_features = self._fuse_state(policy_features, prepared_inputs.batch.state)
        video_condition_window = None
        video_condition_aux: dict[str, object] = {}
        if self.config.train_video_condition_source == VideoConditionSource.GENERATED_FUTURE:
            with torch.no_grad():
                video_condition_window, video_condition_aux = self._build_generated_video_condition_window(
                    visual_tower,
                    visual_outputs,
                    frame_start=resolve_video_condition_frame_start(prepared_inputs.batch),
                    observed_prefix_anchor="start",
                    sample_seed=resolve_video_condition_sample_seed(prepared_inputs.batch),
                )
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={"policy_feature_norm": policy_features.norm(dim=-1).mean().detach()},
            decoder_sequence_context=self._build_decoder_sequence_context(
                visual_outputs,
                state=prepared_inputs.batch.state,
                source_stage=source_stage,
                readout_metadata=readout_metadata,
                decode_output=decode_output,
                video_condition_window=video_condition_window,
            ),
            aux={
                "variant": self.config.name,
                **video_condition_aux,
            },
        )

    def prepare_infer_state(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        del visual_outputs, context
        return prepare_default_runtime_infer_state(
            visual_tower,
            previous_state=previous_state,
            stage="post_decoded",
        )

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        decode_output, source_stage, readout_metadata = self._resolve_decode_output(visual_tower, visual_outputs)
        policy_features = self._extract_policy_features(decode_output)
        policy_features = self._fuse_state(policy_features, context.state)
        video_condition_window = None
        video_condition_aux: dict[str, object] = {}
        if context.extra.get("video_condition_source") == "generated_future":
            video_condition_window, video_condition_aux = self._build_generated_video_condition_window(
                visual_tower,
                visual_outputs,
                frame_start=int(infer_state.cursor.current_start_frame),
                observed_prefix_anchor=str(context.extra.get("video_condition_observed_prefix_anchor", "start")),
                sample_seed=(
                    None
                    if context.extra.get("video_condition_sample_seed") is None
                    else int(context.extra["video_condition_sample_seed"])
                ),
            )
        return PolicyInferOutput(
            policy_features=policy_features,
            next_state=advance_default_runtime_infer_state(
                visual_tower,
                infer_state=infer_state,
                stage="post_decoded",
            ),
            decoder_sequence_context=self._build_decoder_sequence_context(
                visual_outputs,
                state=context.state,
                source_stage=source_stage,
                readout_metadata=readout_metadata,
                decode_output=decode_output,
                video_condition_window=video_condition_window,
            ),
            aux={
                "variant": self.config.name,
                **video_condition_aux,
            },
        )
