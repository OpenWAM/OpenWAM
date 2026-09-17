"""Resolve configured pretrained channel conventions, without owning conversion."""

from open_wam.configs import ActionNormMethod
from open_wam.configs.enums import coerce_enum_value
from open_wam.configs.policy_parallel_stream import ParallelStreamPolicyConfig
from open_wam.models.common.channel_action_adapter import ChannelActionSpec
from .reference_profile import load_reference_profile


def build_action_adapter_spec(
    config: ParallelStreamPolicyConfig,
    *,
    model_action_dim: int,
) -> ChannelActionSpec | None:
    profile = load_reference_profile(config.reference_profile)
    used_action_channel_ids = config.used_action_channel_ids or (
        tuple(profile.used_action_channel_ids) if profile is not None else tuple()
    )
    inverse_used_action_channel_ids = config.inverse_used_action_channel_ids or (
        tuple(profile.inverse_used_action_channel_ids) if profile is not None else tuple()
    )
    if not used_action_channel_ids:
        return None
    action_norm_method = coerce_enum_value(ActionNormMethod, config.action_norm_method)
    if action_norm_method == ActionNormMethod.PROFILE:
        if profile is None:
            raise ValueError("Exact parallel-stream action_norm_method='profile' requires a reference_profile.")
        action_norm_method = profile.action_norm_method
    action_norm_method = coerce_enum_value(ActionNormMethod, action_norm_method)
    norm_q01 = config.norm_q01 or (tuple(profile.norm_q01) if profile is not None else tuple())
    norm_q99 = config.norm_q99 or (tuple(profile.norm_q99) if profile is not None else tuple())

    if len(inverse_used_action_channel_ids) != model_action_dim:
        raise ValueError(
            "Exact parallel-stream inverse channel ids must have length equal to the model action dim, "
            f"got {len(inverse_used_action_channel_ids)} and model_action_dim={model_action_dim}."
        )
    if action_norm_method not in {ActionNormMethod.NONE, ActionNormMethod.QUANTILES}:
        raise ValueError(f"Unsupported exact parallel-stream action_norm_method '{action_norm_method}'.")
    if action_norm_method == ActionNormMethod.QUANTILES and (
        len(norm_q01) != model_action_dim or len(norm_q99) != model_action_dim
    ):
        raise ValueError(
            "Quantile-normalized exact parallel-stream actions require q01/q99 values for every model action channel, "
            f"got len(q01)={len(norm_q01)}, len(q99)={len(norm_q99)}, model_action_dim={model_action_dim}."
        )
    return ChannelActionSpec(
        model_action_dim=model_action_dim,
        raw_action_dim=len(used_action_channel_ids),
        action_norm_method=action_norm_method,
        used_action_channel_ids=tuple(used_action_channel_ids),
        inverse_used_action_channel_ids=tuple(inverse_used_action_channel_ids),
        norm_q01=tuple(norm_q01),
        norm_q99=tuple(norm_q99),
    )
