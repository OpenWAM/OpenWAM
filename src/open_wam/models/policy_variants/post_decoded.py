from __future__ import annotations

from torch import nn

from open_wam.configs import InferenceConfig, PostDecodedPolicyConfig, TrainingConfig
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .common.layouts import align_sequence_length
from .contracts import (
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

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        return ("frontend", "core", "decode")

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        if visual_outputs.decode is None:
            raise ValueError("Post-decoded variant requires decode outputs.")
        return PolicyPreparedInputs(batch=batch)

    def _extract_policy_features(self, visual_outputs: VisualStageOutputs):
        if visual_outputs.decode is None:
            raise ValueError("Post-decoded variant requires decode outputs.")
        decoded_features = visual_outputs.decode.decoded_features
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
        policy_features = self._extract_policy_features(visual_outputs)
        policy_features = self._fuse_state(policy_features, prepared_inputs.batch.state)
        return PolicyTrainOutput(
            policy_features=policy_features,
            metrics={"policy_feature_norm": policy_features.norm(dim=-1).mean().detach()},
            aux={"variant": self.config.name},
        )

    def prepare_infer_state(
        self,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        previous_state: PolicyInferState | None = None,
    ) -> PolicyInferState:
        if previous_state is None:
            return PolicyInferState(step_index=0, cache={})
        return previous_state

    def forward_infer_step(
        self,
        visual_tower: VisualTower,
        visual_outputs: VisualStageOutputs,
        context: PolicyInferContext,
        infer_state: PolicyInferState,
    ) -> PolicyInferOutput:
        policy_features = self._extract_policy_features(visual_outputs)
        policy_features = self._fuse_state(policy_features, context.state)
        return PolicyInferOutput(
            policy_features=policy_features,
            next_state=PolicyInferState(step_index=infer_state.step_index + 1, cache=dict(infer_state.cache)),
            aux={"variant": self.config.name},
        )
