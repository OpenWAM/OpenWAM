from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from open_wam.configs import AttachSite, InferenceConfig, PoolingMode, PostLatentPolicyConfig, TrainingConfig
from open_wam.models.visual_tower import VisualStageOutputs, VisualTower

from .base import PolicyVariant
from .common import advance_default_runtime_infer_state, prepare_default_runtime_infer_state
from .common.layouts import align_sequence_length, pool_frame_tokens, tokens_to_frame_major
from .contracts import (
    PolicyInferContext,
    PolicyInferOutput,
    PolicyInferState,
    PolicyPreparedInputs,
    PolicyTrainBatch,
    PolicyTrainOutput,
)


class PostLatentPolicyVariant(PolicyVariant):
    """Post-latent policy baseline with temporal structure preservation."""

    def __init__(
        self,
        config: PostLatentPolicyConfig,
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
        self.state_proj = nn.Linear(state_dim, config.hidden_size) if config.use_state_projection else None
        self.query_tokens = nn.Parameter(torch.randn(config.query_count, config.hidden_size)) if config.query_count > 0 else None
        self.query_norm = nn.LayerNorm(config.hidden_size) if config.query_count > 0 else None

    def attach_site(self) -> str:
        return self.config.attach_site

    def required_visual_stages(self) -> tuple[str, ...]:
        if self.config.attach_site == AttachSite.POST_FRONTEND_LATENTS:
            return ("frontend",)
        return ("frontend", "core")

    def prepare_train_inputs(
        self,
        visual_outputs: VisualStageOutputs,
        batch: PolicyTrainBatch,
    ) -> PolicyPreparedInputs:
        if batch.actions.ndim != 3:
            raise ValueError(
                "Expected actions with shape [B, H_action, D_action], "
                f"got {tuple(batch.actions.shape)}"
            )
        return PolicyPreparedInputs(batch=batch)

    def _select_video_tokens(self, visual_outputs: VisualStageOutputs) -> torch.Tensor:
        if self.config.attach_site == AttachSite.POST_FRONTEND_LATENTS:
            return visual_outputs.frontend.video_tokens
        if visual_outputs.core is None:
            raise ValueError("Post-latent variant requires core outputs for post-visual-core attachment.")
        return visual_outputs.core.tokens

    def _extract_policy_features(self, visual_outputs: VisualStageOutputs) -> torch.Tensor:
        tokens = self._select_video_tokens(visual_outputs)
        if self.config.pooling_mode == PoolingMode.COMPAT_GLOBAL_MEAN:
            return tokens.mean(dim=1, keepdim=True).expand(-1, self.action_horizon, -1)
        frame_tokens = tokens_to_frame_major(tokens, visual_outputs.frontend.token_grid)
        if self.query_tokens is not None and self.query_norm is not None:
            frame_features = pool_frame_tokens(frame_tokens, mode="mean")
            queries = self.query_norm(self.query_tokens)[None, :, :].expand(frame_features.shape[0], -1, -1)
            attn_scores = torch.matmul(queries, frame_features.transpose(1, 2)) / math.sqrt(frame_features.shape[-1])
            attn_weights = F.softmax(attn_scores, dim=-1)
            queried_features = torch.matmul(attn_weights, frame_features)
            return align_sequence_length(queried_features, self.action_horizon)
        frame_features = pool_frame_tokens(frame_tokens, mode="mean")
        return align_sequence_length(frame_features, self.action_horizon)

    def _fuse_state(self, policy_features: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        if state is None or self.state_proj is None:
            return policy_features
        state_summary = state.mean(dim=1)
        return policy_features + self.state_proj(state_summary)[:, None, :]

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
            aux={"variant": self.config.name, "attach_site": self.config.attach_site},
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
            stage="post_latent",
        )

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
            next_state=advance_default_runtime_infer_state(
                visual_tower,
                infer_state=infer_state,
                stage="post_latent",
            ),
            aux={"variant": self.config.name, "attach_site": self.config.attach_site},
        )
