"""Configured action-decoder construction and extension dispatch."""

from __future__ import annotations

from open_wam.configs import (
    ActionDecoderName,
    ActionNormalizationMode,
    ExperimentConfig,
    ExtensionActionDecoderConfig,
)
from open_wam.configs.policy_mot import MoTPolicyConfig
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.action_decoders import (
    ActionDecoder,
    DecodedFeatureActionDecoder,
    LingbotParallelActionDecoder,
    MLPActionDecoder,
    MoTActionDecoder,
    VideoConditionedActionDecoder,
    VideoOnlyActionDecoder,
)
from open_wam.models.policy_variants.parallel_stream.action_adapter import (
    build_action_adapter_spec,
)

from .registries import ACTION_DECODER_BUILDERS, _EXTENSION_ACTION_DECODER_BUILDERS


def _build_mlp_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    mot_compat_decoder = (
        isinstance(config.policy_variant, MoTPolicyConfig)
        and decoder_config.name == ActionDecoderName.MLP
    )
    if mot_compat_decoder:
        return MoTActionDecoder(
            hidden_size=decoder_config.hidden_size,
            action_dim=decoder_config.action_dim,
            action_horizon=decoder_config.action_horizon,
            training_config=config.training,
            inference_config=config.inference,
            dropout=decoder_config.dropout,
        )
    return MLPActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def _build_decoded_feature_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return DecodedFeatureActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def _build_video_conditioned_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return VideoConditionedActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        context_dim=decoder_config.context_dim,
        text_context_dim=decoder_config.text_context_dim,
        state_dim=decoder_config.state_dim,
        freq_dim=decoder_config.freq_dim,
        num_layers=decoder_config.num_layers,
        num_heads=decoder_config.num_heads,
        attention_head_dim=decoder_config.attention_head_dim,
        ffn_dim=decoder_config.ffn_dim,
        cross_attn_norm=decoder_config.cross_attn_norm,
        eps=decoder_config.eps,
        input_space=decoder_config.input_space,
        train_mode=decoder_config.train_mode,
        action_chunk_anchor_mode=decoder_config.action_chunk_anchor_mode,
        action_expert_init_mode=decoder_config.action_expert_init_mode,
        rollout_chunk_steps=decoder_config.rollout_chunk_steps,
        direct_latent_channels=decoder_config.direct_latent_channels,
        direct_rgb_patch_size=decoder_config.direct_rgb_patch_size,
        use_text_conditioning=decoder_config.use_text_conditioning,
        use_state_conditioning=decoder_config.use_state_conditioning,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def _build_lingbot_parallel_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    source_action_channel_ids: tuple[int, ...] = ()
    if isinstance(config.policy_variant, ParallelStreamPolicyConfig):
        adapter_spec = build_action_adapter_spec(
            config.policy_variant,
            model_action_dim=decoder_config.action_dim,
        )
        if adapter_spec is not None:
            source_action_channel_ids = adapter_spec.used_action_channel_ids
    action_normalization = config.data.action_target.normalization
    source_action_mean: tuple[float, ...] = ()
    source_action_std: tuple[float, ...] = ()
    if action_normalization.mode == ActionNormalizationMode.GAUSSIAN:
        source_action_mean = action_normalization.mean
        source_action_std = action_normalization.std
    return LingbotParallelActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        dropout=decoder_config.dropout,
        recovered_osc_loss_weight=decoder_config.recovered_osc_loss_weight,
        recovered_osc_position_scale=decoder_config.recovered_osc_position_scale,
        recovered_osc_rotation_scale=decoder_config.recovered_osc_rotation_scale,
        source_action_channel_ids=source_action_channel_ids,
        source_action_mean=source_action_mean,
        source_action_std=source_action_std,
    )


def _build_mot_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return MoTActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def _build_video_only_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return VideoOnlyActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        training_config=config.training,
        inference_config=config.inference,
        dropout=decoder_config.dropout,
    )


def _build_extension_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    assert isinstance(decoder_config, ExtensionActionDecoderConfig)
    builder = _EXTENSION_ACTION_DECODER_BUILDERS.get(decoder_config.extension_type)
    if builder is None:
        registered = ", ".join(_EXTENSION_ACTION_DECODER_BUILDERS.keys()) or "<none>"
        raise ValueError(
            f"Unsupported action decoder extension {decoder_config.extension_type!r}. "
            f"Registered extension types: {registered}. "
            "Load its module with `--extension module[:hook]` before constructing the experiment."
        )
    return builder(config)


def build_action_decoder(config: ExperimentConfig):
    builder = ACTION_DECODER_BUILDERS.get(config.action_decoder.name)
    if builder is None:
        raise ValueError(f"Unsupported action decoder '{config.action_decoder.name}'.")
    action_decoder = builder(config)
    if not isinstance(action_decoder, ActionDecoder):
        raise TypeError(
            f"Action decoder builder returned {type(action_decoder).__name__}; "
            "expected an `open_wam.models.action_decoders.ActionDecoder`."
        )
    return action_decoder


__all__ = ["build_action_decoder"]
