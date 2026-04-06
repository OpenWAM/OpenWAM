from __future__ import annotations

from torch import nn

from open_wam.configs import (
    InferenceConfig,
    PostDecodedPolicyConfig,
    TrainingConfig,
    VisualReadoutSourceFamily,
)
from open_wam.models.visual_tower import VisualReadoutRequest, VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .common import (
    SharedVisualReadout,
    advance_default_runtime_infer_state,
    prepare_default_runtime_infer_state,
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
    ) -> DecoderSequenceContext:
        decoded_features = decode_output.decoded_features
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
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={"policy_feature_norm": policy_features.norm(dim=-1).mean().detach()},
            decoder_sequence_context=self._build_decoder_sequence_context(
                visual_outputs,
                state=prepared_inputs.batch.state,
                source_stage=source_stage,
                readout_metadata=readout_metadata,
                decode_output=decode_output,
            ),
            aux={"variant": self.config.name},
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
            ),
            aux={"variant": self.config.name},
        )
