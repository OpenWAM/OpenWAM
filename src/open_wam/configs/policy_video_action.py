"""Architecture-independent video/action policy configuration semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    ContextConditionLatentSource,
    CurrentBlockCoupling,
    GeneralistDenoisingMode,
    GeneralistTrainingParadigm,
    HistoryStreamVisibility,
    JointTimestepCoupling,
    ProprioContextMode,
    VideoActionProgram,
    VideoActionSequenceContract,
    coerce_fields,
)
from .policy_compatibility import resolve_legacy_policy_field
from .policy_contracts import PolicyVariantConfig

_FIXED_CONDITIONING_MODE_BY_PROGRAM = {
    VideoActionProgram.FORWARD_DYNAMICS: GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
    VideoActionProgram.INVERSE_DYNAMICS: GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
}
_CONDITIONAL_DENOISING_MODES = (
    GeneralistDenoisingMode.ACTION_CONDITIONED_VIDEO,
    GeneralistDenoisingMode.VIDEO_CONDITIONED_ACTION,
)


def fixed_conditioning_mode_for_program(
    program: VideoActionProgram | str | None,
) -> GeneralistDenoisingMode | None:
    """Return the fixed conditional mode selected by a standalone program."""

    if program is None:
        return None
    return _FIXED_CONDITIONING_MODE_BY_PROGRAM.get(VideoActionProgram(program))


def fixed_conditioning_mode_from_probabilities(
    probabilities: Mapping[GeneralistDenoisingMode, float] | None,
) -> GeneralistDenoisingMode | None:
    """Resolve a one-hot conditional distribution, if one is configured."""

    if probabilities is None:
        return None
    active = tuple(
        mode
        for mode in _CONDITIONAL_DENOISING_MODES
        if float(probabilities.get(mode, 0.0)) > 0.0
    )
    if len(active) != 1:
        return None
    selected = active[0]
    if abs(float(probabilities.get(selected, 0.0)) - 1.0) > 1e-9:
        return None
    if abs(float(probabilities.get(GeneralistDenoisingMode.JOINT, 0.0))) > 1e-9:
        return None
    return selected


def conditional_denoising_modes_enabled(
    probabilities: Mapping[GeneralistDenoisingMode | str, object] | None,
) -> bool:
    """Return whether a distribution assigns positive FDM or IDM mass."""

    if probabilities is None:
        return False
    for mode in _CONDITIONAL_DENOISING_MODES:
        value = probabilities.get(mode, probabilities.get(mode.value, 0.0))
        if isinstance(value, bool):
            continue
        try:
            if float(value) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def is_dynamics_routed_paradigm(
    paradigm: object,
) -> bool:
    """Return whether samples use the dynamics source-routing adapter."""

    try:
        return (
            GeneralistTrainingParadigm(paradigm)
            == GeneralistTrainingParadigm.DYNAMICS_ROUTED
        )
    except (TypeError, ValueError):
        return False


def validate_conditional_denoising_data_paradigm(
    *,
    probabilities: Mapping[GeneralistDenoisingMode | str, object] | None,
    paradigm: GeneralistTrainingParadigm | str,
) -> None:
    """Require the target-only data projection whenever FDM or IDM is active."""

    if (
        conditional_denoising_modes_enabled(probabilities)
        and not is_dynamics_routed_paradigm(paradigm)
    ):
        raise ValueError(
            "Positive FDM/IDM `generalist_denoising_mode_probs` require "
            "`generalist_training_paradigm = dynamics_routed` so real and optional "
            "counterfactual samples use the target-only t0 contract. A zero "
            "counterfactual source weight is supported; pure joint may use `demo_only`."
        )


def resolve_fixed_conditioning_mode(
    policy_config: object,
) -> GeneralistDenoisingMode | None:
    """Resolve a standalone or one-hot GJD conditional training mode."""

    program_mode = fixed_conditioning_mode_for_program(
        getattr(policy_config, "program", None)
    )
    if program_mode is not None:
        return program_mode
    return fixed_conditioning_mode_from_probabilities(
        getattr(policy_config, "generalist_denoising_mode_probs", None)
    )


def one_hot_conditioning_mode_probabilities(
    mode: GeneralistDenoisingMode,
) -> dict[GeneralistDenoisingMode, float]:
    """Build the canonical one-hot distribution for one conditioning mode."""

    return {
        candidate: float(candidate == mode)
        for candidate in GeneralistDenoisingMode
    }


def _program_requires_joint_coupling(program: VideoActionProgram) -> bool:
    return program in {
        VideoActionProgram.GENERALIST_JOINT_DENOISING,
        VideoActionProgram.FORWARD_DYNAMICS,
        VideoActionProgram.INVERSE_DYNAMICS,
    }


def current_block_coupling_for_program(
    program: VideoActionProgram | str,
) -> CurrentBlockCoupling:
    """Return the low-level same-chunk coupling owned by one public program."""

    resolved_program = VideoActionProgram(program)
    if _program_requires_joint_coupling(resolved_program):
        return CurrentBlockCoupling.JOINT
    return CurrentBlockCoupling(resolved_program.value)


def resolve_video_action_program_semantics(
    *,
    program: VideoActionProgram | str | None,
    current_block_coupling: CurrentBlockCoupling | str | None,
) -> tuple[VideoActionProgram | None, CurrentBlockCoupling | None]:
    """Resolve a public program into its low-level same-chunk coupling."""

    resolved_program = None if program is None else VideoActionProgram(program)
    resolved_coupling = (
        None
        if current_block_coupling is None
        else CurrentBlockCoupling(current_block_coupling)
    )
    if resolved_program is None and resolved_coupling is not None:
        resolved_program = VideoActionProgram(resolved_coupling.value)
    if resolved_program is not None:
        expected_coupling = current_block_coupling_for_program(resolved_program)
        if resolved_coupling is not None and resolved_coupling != expected_coupling:
            raise ValueError(
                "`policy_variant.program` conflicts with "
                "`policy_variant.current_block_coupling`: "
                f"program={resolved_program.value!r} requires "
                f"current_block_coupling={expected_coupling.value!r}, got "
                f"{resolved_coupling.value!r}."
            )
        resolved_coupling = expected_coupling
    return resolved_program, resolved_coupling


@dataclass(frozen=True)
class VideoActionPolicyConfig(PolicyVariantConfig):
    """Shared semantic envelope for video/action policy architectures.

    Subclasses own parameter topology and backend-specific controls. This base
    owns choices whose meaning must remain identical across architectures.
    """

    noisy_video_condition_prob: float = 0.5
    program: VideoActionProgram | None = None
    joint_timestep_coupling: JointTimestepCoupling = JointTimestepCoupling.MATCH_SIGMA
    # Deprecated boolean alias retained for old checkpoint-era configs.
    couple_action_to_video_timesteps: bool | None = field(default=None, repr=False, compare=False)
    generalist_training_paradigm: GeneralistTrainingParadigm = GeneralistTrainingParadigm.DEMO_ONLY
    generalist_denoising_mode_probs: dict[GeneralistDenoisingMode, float] | None = None
    generalist_mode_text_token: bool = False
    proprio_context_mode: ProprioContextMode = ProprioContextMode.NONE
    history_stream_visibility: HistoryStreamVisibility = HistoryStreamVisibility.FULL
    context_condition_latent_source: ContextConditionLatentSource = ContextConditionLatentSource.VIDEO_LATENTS
    use_condition_latents: bool = True
    require_condition_latents: bool = False
    sequence_contract: VideoActionSequenceContract = VideoActionSequenceContract.DEFAULT
    # Constructor/load alias. Maintained configs and implementation code use
    # `sequence_contract`; serialization omits this compatibility field.
    parallel_sequence_contract: VideoActionSequenceContract | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def normalize_config_override_values(
        self,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Clear program-owned conditional defaults during config overrides."""

        normalized = dict(values)
        if (
            "program" in normalized
            and "generalist_denoising_mode_probs" not in normalized
            and (
                fixed_conditioning_mode_for_program(normalized["program"]) is not None
                or fixed_conditioning_mode_for_program(self.program) is not None
            )
        ):
            # A fixed program owns its one-hot distribution. Clear any derived
            # value when entering, leaving, or switching fixed programs.
            normalized["generalist_denoising_mode_probs"] = None
        return normalized

    def __post_init__(self) -> None:
        super().__post_init__()
        resolved_sequence_contract = resolve_legacy_policy_field(
            canonical_value=self.sequence_contract,
            legacy_value=self.parallel_sequence_contract,
            canonical_default=VideoActionSequenceContract.DEFAULT,
            canonical_name="sequence_contract",
            legacy_name="parallel_sequence_contract",
        )
        object.__setattr__(self, "sequence_contract", resolved_sequence_contract)
        coerce_fields(
            self,
            enum_fields={
                "generalist_training_paradigm": GeneralistTrainingParadigm,
                "proprio_context_mode": ProprioContextMode,
                "history_stream_visibility": HistoryStreamVisibility,
                "context_condition_latent_source": ContextConditionLatentSource,
                "sequence_contract": VideoActionSequenceContract,
                "joint_timestep_coupling": JointTimestepCoupling,
            },
            optional_enum_fields={"program": VideoActionProgram},
        )
        # Constructor aliases are input-only. Clearing them prevents
        # `dataclasses.replace()` from replaying stale aliases over a canonical
        # CLI override.
        object.__setattr__(self, "parallel_sequence_contract", None)
        if self.couple_action_to_video_timesteps is not None:
            object.__setattr__(
                self,
                "joint_timestep_coupling",
                JointTimestepCoupling.MATCH_SIGMA
                if bool(self.couple_action_to_video_timesteps)
                else JointTimestepCoupling.INDEPENDENT,
            )
        object.__setattr__(self, "couple_action_to_video_timesteps", None)
        if not 0.0 <= float(self.noisy_video_condition_prob) <= 1.0:
            raise ValueError(
                "Video/action policies require `0 <= noisy_video_condition_prob <= 1`, "
                f"got noisy_video_condition_prob={self.noisy_video_condition_prob!r}."
            )
        if bool(self.require_condition_latents) and not bool(self.use_condition_latents):
            raise ValueError(
                "`require_condition_latents` cannot be true when `use_condition_latents` is false."
            )


__all__ = [
    "VideoActionPolicyConfig",
    "current_block_coupling_for_program",
    "resolve_video_action_program_semantics",
]
