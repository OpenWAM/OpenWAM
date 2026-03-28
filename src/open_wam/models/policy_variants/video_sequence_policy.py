from __future__ import annotations

from open_wam.configs import InferenceConfig, TrainingConfig, VideoSequencePolicyConfig
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .common import advance_default_runtime_infer_state, prepare_default_runtime_infer_state
from .common.layouts import align_sequence_length, pool_frame_tokens, tokens_to_frame_major
from .contracts import (
    DecoderSequenceContext,
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)


class VideoSequencePolicyVariant(PolicyVariant):
    """Sequence-preserving post-core policy family for method-3 decoders.

    This variant deliberately keeps policy semantics minimal:
    - request final shared-core visual tokens
    - preserve frame/token-grid structure for downstream sequence decoders
    - keep state and goal context attached to the decoder-facing payload

    The temporary `policy_features` output remains a pooled visual fallback so
    existing simple decoders can still execute while the dedicated
    `video_sequence_policy` decoder stack is introduced.
    """

    def __init__(
        self,
        config: VideoSequencePolicyConfig,
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
        self.state_dim = state_dim

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend", "core")

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        if visual_outputs.core is None:
            raise ValueError("Video-sequence policy requires shared core outputs.")
        return PolicyPreparedInputs(batch=batch)

    def _require_frame_tokens(self, visual_outputs: VisualStageOutputs):
        if visual_outputs.core is None:
            raise ValueError("Video-sequence policy requires shared core outputs.")
        return tokens_to_frame_major(visual_outputs.core.tokens, visual_outputs.frontend.token_grid)

    def _build_policy_features(self, visual_outputs: VisualStageOutputs):
        frame_tokens = self._require_frame_tokens(visual_outputs)
        frame_features = pool_frame_tokens(frame_tokens, mode="mean")
        return align_sequence_length(frame_features, self.action_horizon)

    def _build_goal_features(self, visual_outputs: VisualStageOutputs):
        if not self.config.use_goal_context:
            return None
        return visual_outputs.frontend.conditioning.text_context

    def _build_decoder_sequence_context(
        self,
        visual_outputs: VisualStageOutputs,
        *,
        state,
    ) -> DecoderSequenceContext:
        frame_tokens = self._require_frame_tokens(visual_outputs)
        return DecoderSequenceContext(
            sequence_tokens=frame_tokens,
            sequence_layout={
                "family": "video_sequence_policy",
                "kind": "frame_token_grid",
                "attach_site": str(self.config.attach_site),
                "temporal_projection": str(self.config.temporal_projection),
            },
            token_grid=visual_outputs.frontend.token_grid,
            frame_count=int(frame_tokens.shape[1]),
            source_stage="core",
            state_sequence=(state if self.config.use_state_context else None),
            goal_features=self._build_goal_features(visual_outputs),
            aux_features={
                "negative_goal_features": visual_outputs.frontend.conditioning.negative_text_context,
            },
        )

    def forward_train(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        prepared_inputs: PolicyPreparedInputs,
    ) -> PolicyTrainOutput:
        del visual_tower
        policy_features = self._build_policy_features(visual_outputs)
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={"policy_feature_norm": policy_features.norm(dim=-1).mean().detach()},
            decoder_sequence_context=self._build_decoder_sequence_context(
                visual_outputs,
                state=prepared_inputs.batch.state,
            ),
            aux={"variant": self.config.name, "method_family": "video_sequence_policy"},
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
            stage="video_sequence_policy",
        )

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        policy_features = self._build_policy_features(visual_outputs)
        return PolicyInferOutput(
            policy_features=policy_features,
            next_state=advance_default_runtime_infer_state(
                visual_tower,
                infer_state=infer_state,
                stage="video_sequence_policy",
            ),
            decoder_sequence_context=self._build_decoder_sequence_context(
                visual_outputs,
                state=context.state,
            ),
            aux={"variant": self.config.name, "method_family": "video_sequence_policy"},
        )
