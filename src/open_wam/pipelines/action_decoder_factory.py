"""Configured action-decoder construction and extension dispatch."""

from __future__ import annotations

from open_wam.configs import (
    ActionNormalizationMode,
    ExperimentConfig,
    ExtensionActionDecoderConfig,
)
from open_wam.models.action_decoders import (
    ActionDecoder,
    DualExpertActionDecoder,
    ParallelStreamActionDecoder,
    VideoOnlyActionDecoder,
)

from .registries import _EXTENSION_ACTION_DECODER_BUILDERS, ACTION_DECODER_BUILDERS


def _build_parallel_stream_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    action_normalization = config.data.action_target.normalization
    source_action_mean: tuple[float, ...] = ()
    source_action_std: tuple[float, ...] = ()
    if action_normalization.mode == ActionNormalizationMode.GAUSSIAN:
        source_action_mean = action_normalization.mean
        source_action_std = action_normalization.std
    return ParallelStreamActionDecoder(
        hidden_size=decoder_config.hidden_size,
        action_dim=decoder_config.action_dim,
        action_horizon=decoder_config.action_horizon,
        dropout=decoder_config.dropout,
        recovered_osc_loss_weight=decoder_config.recovered_osc_loss_weight,
        recovered_osc_position_scale=decoder_config.recovered_osc_position_scale,
        recovered_osc_rotation_scale=decoder_config.recovered_osc_rotation_scale,
        source_action_mean=source_action_mean,
        source_action_std=source_action_std,
    )


def _build_dual_expert_action_decoder(config: ExperimentConfig):
    decoder_config = config.action_decoder
    return DualExpertActionDecoder(
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
